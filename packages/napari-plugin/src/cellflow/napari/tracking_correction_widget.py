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
)
from napari.utils.notifications import show_info, show_error

from .registry import get_state
from .widgets import CollapsibleSection
from .tracking_widget import TrackingTab
from .correction_widget import CorrectionWidget


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

        # Load row
        load_row = QHBoxLayout()
        self._load_btn = QPushButton("Load nuclear segmentation")
        self._load_btn.setToolTip(
            "Load the nuclear labels layer currently registered in state.\n"
            "Run Cellpose / Ultrack first to populate it."
        )
        self._load_btn.clicked.connect(self._on_load)
        load_row.addWidget(self._load_btn)

        self._load_from_layer_btn = QPushButton("Load from active layer")
        self._load_from_layer_btn.setToolTip("Use the currently active layer in the viewer.")
        self._load_from_layer_btn.clicked.connect(self._on_load_from_layer)
        load_row.addWidget(self._load_from_layer_btn)
        lay.addLayout(load_row)

        self._data_status = QLabel("No layer loaded")
        self._data_status.setStyleSheet("font-style: italic; color: palette(text);")
        lay.addWidget(self._data_status)

        # Save row
        self._save_btn = QPushButton("Save corrected labels")
        self._save_btn.setToolTip(
            "Save the current layer to 3_correction/nuclear_labels_corrected.tif"
        )
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        lay.addWidget(self._save_btn)

        # Sub-sections
        lay.addWidget(CollapsibleSection("Tracking", self._tracking, expanded=False))
        lay.addWidget(CollapsibleSection("Correction", self._correction, expanded=False))

        # Auto-sync when state's nuclear labels change
        self._state.nuclear_labels_changed.connect(self._on_nuclear_labels_changed)

    # ── load handlers ─────────────────────────────────────────────────────

    def _on_load(self) -> None:
        """Load the nuclear labels layer from viewer state."""
        name = self._state.tissue.nuclear_labels_layer
        arr = self._state.tissue.nuclear_labels

        if name and name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if not isinstance(layer, napari.layers.Labels):
                self._data_status.setText(f"'{name}' is not a Labels layer")
                return
            if self._state.project_dir is not None:
                self._load_nuclear_zavg(self._state.project_dir, self._state.current_position)
            self._set_data_layer(layer)
            return

        if arr is not None:
            layer_name = name or "Nuclear Labels"
            if self._state.project_dir is not None:
                self._load_nuclear_zavg(self._state.project_dir, self._state.current_position)
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = arr
                layer = self.viewer.layers[layer_name]
            else:
                layer = self.viewer.add_labels(arr, name=layer_name)
            self._set_data_layer(layer)
            return

        # Try loading from disk if a project is open
        disk_arr = self._try_load_from_disk()
        if disk_arr is not None:
            return

        self._data_status.setText(
            "No nuclear labels in state — run Cellpose or Ultrack first"
        )

    def _try_load_from_disk(self) -> "np.ndarray | None":
        """Load nuclear labels from disk for the current project position.

        Input is 2_ultrack/nuclear_labels_2d.tif (the flattened Ultrack output).
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
            (stage_dir(project_dir, pos, "tracking") / "nuclear_labels_2d.tif",
             "Nuclear Labels"),
        ]
        for path, layer_name in candidates:
            if not path.exists():
                continue
            try:
                arr = tifffile.imread(str(path)).astype(np.int32)
            except Exception as exc:
                self._data_status.setText(f"Could not read {path.name}: {exc}")
                return None
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
        """Load nucleus_zavg.tif as an Image layer (background reference for correction)."""
        from cellflow.core.paths import stage_dir
        img_path = stage_dir(project_dir, pos, "raw_import") / "nucleus" / "nucleus_zavg.tif"
        if not img_path.exists():
            return
        layer_name = "Nucleus avg"
        try:
            img = tifffile.imread(str(img_path))
        except Exception:
            return
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = img
        else:
            self.viewer.add_image(img, name=layer_name, colormap="gray")

    def _on_save(self) -> None:
        """Save the current layer to 3_correction/nuclear_labels_corrected.tif."""
        if self._data_layer is None:
            show_error("No layer loaded — load a nuclear labels layer first.")
            return

        project_dir = self._state.project_dir
        if project_dir is None:
            show_error("No project open — cannot determine save path.")
            return

        from cellflow.core.paths import stage_dir
        pos = self._state.current_position
        out_path = stage_dir(project_dir, pos, "correction") / "nuclear_labels_corrected.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            arr = np.asarray(self._data_layer.data)
            tifffile.imwrite(str(out_path), arr.astype(np.int32))
        except Exception as exc:
            show_error(f"Save failed: {exc}")
            return

        show_info(f"Saved corrected labels → {out_path.name}")
        self._data_status.setText(
            f"Saved: {out_path.name}  {arr.shape}"
        )

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
