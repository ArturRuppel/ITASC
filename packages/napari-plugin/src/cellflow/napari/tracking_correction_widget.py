"""Combined Tracking + Correction widget for the nuclear segmentation correction loop.

Both the LapTrack re-tracking and the manual correction tools operate on the same
loaded layer.  A shared Load section at the top sets that layer; both sub-components
update whenever the layer changes.
"""

import numpy as np
import napari
import napari.layers
import tifffile
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QRadioButton,
)
from napari.utils.notifications import show_info, show_error

from .registry import get_state
from .tracking_widget import TrackingTab
from .correction_widget import CorrectionWidget
from .widgets import PipelineFilesWidget


class TrackingCorrectionWidget(QWidget):
    """Parent widget combining LapTrack re-tracking and manual label correction.

    Shared Load controls at the top set the working layer.  Both sub-components
    (Tracking, Correction) receive the same layer reference via set_data_layer().
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._state = get_state(viewer)
        self._data_layer: napari.layers.Labels | None = None

        # ── sub-widgets ───────────────────────────────────────────────────
        self._tracking = TrackingTab(viewer)
        self._correction = CorrectionWidget(viewer)

        # ── layout ────────────────────────────────────────────────────────
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        # ── Mode selection ────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        self._nuclei_radio = QRadioButton("Correct Nuclei")
        self._cells_radio  = QRadioButton("Correct Cells")
        self._nuclei_radio.setChecked(True)
        self._mode_group.addButton(self._nuclei_radio)
        self._mode_group.addButton(self._cells_radio)
        mode_row.addWidget(self._nuclei_radio)
        mode_row.addWidget(self._cells_radio)
        lay.addLayout(mode_row)
        self._nuclei_radio.toggled.connect(self._on_mode_changed)

        # Load row
        load_row = QHBoxLayout()
        self._load_btn = QPushButton("Load nuclear segmentation")
        self._load_btn.setToolTip(
            "Load the nuclear labels layer currently registered in state.\n"
            "Run Cluster Cellpose / Nucleus Ultrack first to populate it."
        )
        self._load_btn.clicked.connect(self._on_load)
        load_row.addWidget(self._load_btn)

        self._load_from_layer_btn = QPushButton("Load from active layer")
        self._load_from_layer_btn.setToolTip("Use the currently active layer in the viewer.")
        self._load_from_layer_btn.clicked.connect(self._on_load_from_layer)
        load_row.addWidget(self._load_from_layer_btn)
        lay.addLayout(load_row)

        self._data_status = QLabel("No layer loaded")
        self._data_status.setStyleSheet("font-style: italic; color: palette(mid);")
        lay.addWidget(self._data_status)

        # File status rows — refreshed based on mode
        self._files_widget = PipelineFilesWidget([
            ("Input", [
                ("2_nucleus_ultrack/nuclear_labels_2d.tif",   "Nuclear labels 2D"),
            ]),
            ("Output", [
                ("3_correction/nuclear_labels_corrected.tif", "Corrected labels"),
            ]),
        ])
        lay.addWidget(self._files_widget)

        # Save row
        self._save_btn = QPushButton("Save corrected labels")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        lay.addWidget(self._save_btn)

        # Sub-sections: correction first, then re-tracking (hidden in Cells mode)
        lay.addWidget(self._correction)
        lay.addWidget(self._tracking)

        # Auto-sync when state's nuclear labels change
        self._state.nuclear_labels_changed.connect(self._on_nuclear_labels_changed)
        self._state.position_changed.connect(self._on_position_changed)
        self._state.pipeline_schema_changed.connect(self._refresh_files)
        self._refresh_files()

    # ── mode helpers ──────────────────────────────────────────────────────

    @property
    def _correcting_cells(self) -> bool:
        return self._cells_radio.isChecked()

    def _on_mode_changed(self) -> None:
        """Update UI when switching between Nuclei / Cells mode."""
        if self._correcting_cells:
            self._load_btn.setText("Load cell segmentation")
            self._load_btn.setToolTip(
                "Load cell labels from 4_cell_ultrack/cell_labels_2d.tif.\n"
                "Also loads nuclear labels as a read-only reference layer."
            )
            self._save_btn.setText("Save corrected cell labels")
            self._tracking.setVisible(False)
        else:
            self._load_btn.setText("Load nuclear segmentation")
            self._load_btn.setToolTip(
                "Load the nuclear labels layer currently registered in state.\n"
                "Run Cluster Cellpose / Nucleus Ultrack first to populate it."
            )
            self._save_btn.setText("Save corrected labels")
            self._tracking.setVisible(True)
        # Clear the current layer when switching mode
        if self._data_layer is not None:
            self._data_layer = None
            self._save_btn.setEnabled(False)
            self._data_status.setText("No layer loaded")
            if self._correction._activate_btn.isChecked():
                self._correction._deactivate()
        self._refresh_files()

    # ── load handlers ─────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return self._tracking.get_params()

    def set_params(self, data: dict) -> None:
        self._tracking.set_params(data)

    def _on_load(self) -> None:
        """Load the appropriate labels layer based on current mode."""
        if self._correcting_cells:
            self._load_cell_labels()
        else:
            self._load_nuclear_labels()

    def _load_nuclear_labels(self) -> None:
        """Load nuclear labels from viewer state or disk."""
        name = self._state.tissue.nuclear_labels_layer
        arr = self._state.tissue.nuclear_labels

        # If the layer is already in the viewer (current position), reuse it.
        if name and name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if not isinstance(layer, napari.layers.Labels):
                self._data_status.setText(f"'{name}' is not a Labels layer")
                return
            if self._state.project_dir is not None:
                self._load_nuclear_zavg(self._state.project_dir, self._state.current_position)
            self._set_data_layer(layer)
            return

        # When a project is open, always load from disk so the current position
        # is used. The state array may belong to a different position.
        if self._state.project_dir is not None:
            disk_arr = self._try_load_from_disk()
            if disk_arr is not None:
                return

        # No project open — fall back to whatever is in state.
        if arr is not None:
            layer_name = name or "Nuclear Labels"
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = arr
                layer = self.viewer.layers[layer_name]
            else:
                layer = self.viewer.add_labels(arr, name=layer_name)
            self._set_data_layer(layer)
            return

        self._data_status.setText(
            "No nuclear labels in state — run Cellpose or Ultrack first"
        )

    def _load_cell_labels(self) -> None:
        """Load cell labels from disk and nuclear labels as reference."""
        project_dir = self._state.project_dir
        if project_dir is None:
            self._data_status.setText("No project open — cannot load cell labels.")
            return

        from cellflow.core.paths import stage_dir
        pos = self._state.current_position

        cell_path = stage_dir(project_dir, pos, "cell_ultrack") / "cell_labels_2d.tif"
        if not cell_path.exists():
            self._data_status.setText(
                "No cell labels found — run Cell Ultrack first."
            )
            return

        try:
            arr = tifffile.imread(str(cell_path)).astype(np.int32)
        except Exception as exc:
            self._data_status.setText(f"Could not read cell_labels_2d.tif: {exc}")
            return

        # Load background images
        self._load_nuclear_zavg(project_dir, pos)

        # Also load nuclear labels as a read-only reference
        nuc_path = stage_dir(project_dir, pos, "correction") / "nuclear_labels_corrected.tif"
        if nuc_path.exists():
            try:
                nuc_arr = tifffile.imread(str(nuc_path)).astype(np.int32)
                nuc_layer_name = "Nuclear Labels (ref)"
                if nuc_layer_name in self.viewer.layers:
                    self.viewer.layers[nuc_layer_name].data = nuc_arr
                else:
                    nuc_layer = self.viewer.add_labels(nuc_arr, name=nuc_layer_name)
                    nuc_layer.opacity = 0.4
            except Exception:
                pass

        # Load cell labels as the editable layer
        layer_name = "Cell Labels"
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = arr.astype(np.uint32)
            layer = self.viewer.layers[layer_name]
        else:
            layer = self.viewer.add_labels(arr.astype(np.uint32), name=layer_name)

        self._set_data_layer(layer)

    def _try_load_from_disk(self) -> "np.ndarray | None":
        """Load nuclear labels from disk for the current project position.

        Input is 2_nucleus_ultrack/nuclear_labels_2d.tif (the flattened Ultrack output).
        If a prior correction session already saved nuclear_labels_corrected.tif,
        that is loaded instead so work can be resumed.
        """
        project_dir = self._state.project_dir
        if project_dir is None:
            return None

        from cellflow.core.paths import stage_dir
        pos = self._state.current_position

        # resume > fresh: prefer corrected file if it already exists
        candidates = [
            (stage_dir(project_dir, pos, "correction") / "nuclear_labels_corrected.tif",
             "Nuclear Labels (corrected)"),
            (stage_dir(project_dir, pos, "nucleus_ultrack") / "nuclear_labels_2d.tif",
             "Nuclear Labels"),
        ]
        for path, layer_name in candidates:
            if not path.exists():
                continue
            try:
                arr = tifffile.imread(str(path)).astype(np.int32)
            except Exception as exc:
                self._data_status.setText(f"Could not read {path.name}: {exc}")
                continue
            self._load_nuclear_zavg(project_dir, pos)
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = arr
                layer = self.viewer.layers[layer_name]
            else:
                layer = self.viewer.add_labels(arr, name=layer_name)
            self._state.set_tissue_nuclear_labels(arr, layer_name)
            self._set_data_layer(layer)
            return arr

        return None

    def _load_nuclear_zavg(self, project_dir, pos: int) -> None:
        """Load cell_zavg and nucleus_zavg as background reference layers.

        Layer order (bottom to top): cell_zavg → nucleus_zavg → labels.
        Nucleus is shown with additive blending and bop_orange LUT.
        """
        from cellflow.core.paths import stage_dir
        raw_dir = stage_dir(project_dir, pos, "raw_import")

        cell_path = raw_dir / "cell_zavg.tif"
        nuc_path = raw_dir / "nucleus_zavg.tif"

        # --- cell_zavg (bottom) ---
        cell_layer_name = "Cell avg"
        if cell_path.exists():
            try:
                cell_img = tifffile.imread(str(cell_path))
                if cell_layer_name in self.viewer.layers:
                    self.viewer.layers[cell_layer_name].data = cell_img
                else:
                    self.viewer.add_image(cell_img, name=cell_layer_name, colormap="gray")
            except Exception:
                pass

        # --- nucleus_zavg (middle) ---
        nuc_layer_name = "Nucleus avg"
        if nuc_path.exists():
            try:
                nuc_img = tifffile.imread(str(nuc_path))
                if nuc_layer_name in self.viewer.layers:
                    layer = self.viewer.layers[nuc_layer_name]
                    layer.data = nuc_img
                    layer.colormap = "bop orange"
                    layer.blending = "additive"
                else:
                    layer = self.viewer.add_image(
                        nuc_img,
                        name=nuc_layer_name,
                        colormap="bop orange",
                        blending="additive",
                    )
            except Exception:
                pass

        # Reorder so cell_zavg is at index 0 (bottom), nucleus above it,
        # then the labels layer stays on top.
        def _layer_index(name):
            try:
                return self.viewer.layers.index(self.viewer.layers[name])
            except KeyError:
                return None

        desired_bottom = [cell_layer_name, nuc_layer_name]
        for target_idx, name in enumerate(desired_bottom):
            idx = _layer_index(name)
            if idx is not None and idx != target_idx:
                self.viewer.layers.move(idx, target_idx)

    def _on_save(self) -> None:
        """Save the current layer to the appropriate output path."""
        if self._data_layer is None:
            show_error("No layer loaded — load a layer first.")
            return

        project_dir = self._state.project_dir
        if project_dir is None:
            show_error("No project open — cannot determine save path.")
            return

        from cellflow.core.paths import stage_dir
        pos = self._state.current_position

        if self._correcting_cells:
            out_path = stage_dir(project_dir, pos, "cell_ultrack") / "cell_labels_2d.tif"
        else:
            out_path = stage_dir(project_dir, pos, "correction") / "nuclear_labels_corrected.tif"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            arr = np.asarray(self._data_layer.data)
            tifffile.imwrite(str(out_path), arr.astype(np.int32))
        except Exception as exc:
            show_error(f"Save failed: {exc}")
            return

        show_info(f"Saved corrected labels → {out_path.name}")
        self._data_status.setText(f"Saved: {out_path.name}  {arr.shape}")
        self._refresh_files()

    def _on_load_from_layer(self) -> None:
        """Use the currently active layer."""
        active = self.viewer.layers.selection.active
        if active is None:
            self._data_status.setText("No active layer")
            return
        if not isinstance(active, napari.layers.Labels):
            self._data_status.setText("Active layer is not a Labels layer")
            return
        self._set_data_layer(active)

    def _set_data_layer(self, layer: napari.layers.Labels) -> None:
        self._data_layer = layer
        shape = layer.data.shape
        self._data_status.setText(f"Layer: '{layer.name}'  {shape}")
        self._save_btn.setEnabled(True)
        self._tracking.set_data_layer(layer)
        self._correction.set_data_layer(layer)

    def _on_nuclear_labels_changed(self) -> None:
        """Auto-update the shared layer when state's nuclear labels are replaced."""
        name = self._state.tissue.nuclear_labels_layer
        if name and name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if isinstance(layer, napari.layers.Labels):
                self._set_data_layer(layer)

    def _on_position_changed(self) -> None:
        """Clear loaded layer when the active position changes to prevent cross-position corruption."""
        if self._data_layer is not None and self._data_layer.name in self.viewer.layers:
            self.viewer.layers.remove(self._data_layer.name)
        self._data_layer = None
        self._save_btn.setEnabled(False)
        self._data_status.setText("No layer loaded — position changed, click Load")
        self._refresh_files()

    def _refresh_files(self) -> None:
        """Refresh file-status rows for the current project/position."""
        from pathlib import Path
        if self._correcting_cells:
            spec = [
                ("Input",  [("3_correction/nuclear_labels_corrected.tif", "Corrected nuclear labels")]),
                ("Output", [("4_cell_ultrack/cell_labels_2d.tif",        "Cell labels 2D")]),
            ]
        else:
            spec = [
                ("Input",  [("2_nucleus_ultrack/nuclear_labels_2d.tif",   "Nuclear labels 2D")]),
                ("Output", [("3_correction/nuclear_labels_corrected.tif", "Corrected labels")]),
            ]
        self._files_widget.update_spec(spec)

        project_dir = self._state.project_dir
        if project_dir is None:
            self._files_widget.refresh(None)
            return
        pos = self._state.current_position
        self._files_widget.refresh(Path(project_dir) / f"pos{pos:02d}")
