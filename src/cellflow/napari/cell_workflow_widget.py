"""Cell segmentation widget for CellFlow — Flow-Following Segmentation."""
from __future__ import annotations

import logging
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QSpinBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import best_overlapping_label, expand_label_to_foreground
from cellflow.database.tracked import read_full_tracked_stack
from cellflow.napari.cell_boundary_workflow_widget import CellBoundaryWorkflowWidget
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import (
    add_block_button_row,
    add_parameter_grid_row,
    block_grid,
    status_label,
)

logger = logging.getLogger(__name__)

_FILTERED_FLOW_LAYER = "Filtered Flow Magnitude"
_FOREGROUND_MASK_LAYER = "Foreground Mask"
_TRACKED_CELL_LAYER = "Tracked: Cell"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER = "Nucleus z-avg"
_FF_SPIN_WIDTH = 80
_FF_SPIN_MIN_WIDTH = int(_FF_SPIN_WIDTH * 0.9)


class CellWorkflowWidget(QWidget):
    """Cell segmentation — Flow-Following Segmentation."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._ff_worker = None
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        def _stage_files(group_label: str, entries: list[tuple[str, str]]) -> PipelineFilesWidget:
            return PipelineFilesWidget([(group_label, entries)], viewer=self.viewer)

        def _stage_status() -> QLabel:
            label = QLabel("")
            label.setWordWrap(True)
            label.setVisible(False)
            status_label(label)
            return label

        def _stage_progress() -> QProgressBar:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setVisible(False)
            return bar

        def _param_grid():
            grid = block_grid(horizontal_spacing=12, vertical_spacing=4)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            return grid

        def _dspin(lo, hi, val, step, decimals=1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return s

        def _ispin(lo, hi, val, step=1):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return s

        self.filtered_flow_params_widget = QWidget()
        filter_lay = QVBoxLayout(self.filtered_flow_params_widget)
        filter_lay.setContentsMargins(0, 0, 0, 0)
        filter_lay.setSpacing(4)
        filter_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.filtered_flow_input_files = _stage_files("Inputs", [
            ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
            ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
        ])
        filter_lay.addWidget(self.filtered_flow_input_files)
        filter_grid = _param_grid()
        self.ff_median_time_spin   = _ispin(1, 15, 3)
        self.ff_median_space_spin  = _ispin(1, 15, 5)
        self.ff_gauss_time_spin    = _dspin(0.0, 10.0, 0.0, 0.1)
        self.ff_gauss_space_spin   = _dspin(0.0, 10.0, 0.0, 0.1)
        add_parameter_grid_row(filter_grid, 0, 0, "Median t kernel:", self.ff_median_time_spin)
        add_parameter_grid_row(filter_grid, 0, 1, "Median xy kernel:", self.ff_median_space_spin)
        add_parameter_grid_row(filter_grid, 1, 0, "Gaussian t sigma:", self.ff_gauss_time_spin)
        add_parameter_grid_row(filter_grid, 1, 1, "Gaussian xy sigma:", self.ff_gauss_space_spin)
        filter_lay.addLayout(filter_grid)

        self.ff_flow_mag_btn = QPushButton("Create filtered_dp")
        self.ff_flow_mag_btn.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
        self.ff_flow_mag_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        filter_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(filter_btn_row, 0, self.ff_flow_mag_btn)
        filter_lay.addLayout(filter_btn_row)
        self.filtered_flow_status_lbl = _stage_status()
        filter_lay.addWidget(self.filtered_flow_status_lbl)
        self.filtered_flow_progress_bar = _stage_progress()
        filter_lay.addWidget(self.filtered_flow_progress_bar)
        self.filtered_flow_output_files = _stage_files("Outputs", [
            ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
            ("3_cell/filtered_flow_mag.tif", "Filtered flow magnitude"),
        ])
        filter_lay.addWidget(self.filtered_flow_output_files)

        self.filtered_flow_section = CollapsibleSection(
            "Flow Filtering", self.filtered_flow_params_widget, expanded=False
        )
        layout.addWidget(self.filtered_flow_section)

        # ---- Contour Maps + Segmentation (embedded from CellBoundaryWorkflowWidget) ----
        self._seg_widget = CellBoundaryWorkflowWidget(self.viewer, parent=None)
        self._seg_widget.correction_section.setVisible(False)
        self._seg_widget.boundary_selection_section.set_title("Segmentation")
        self._seg_widget.foreground_section.set_title("Foreground Masks")
        self._seg_widget.contour_section.set_title("Contour Maps")
        # Remove the internal stretch so it doesn't create extra dead space
        seg_layout = self._seg_widget.layout()
        last_item = seg_layout.itemAt(seg_layout.count() - 1)
        if last_item and last_item.spacerItem():
            seg_layout.removeItem(last_item)
        layout.addWidget(self._seg_widget)

        self.correction_params_widget = QWidget()
        correction_lay = QVBoxLayout(self.correction_params_widget)
        correction_lay.setContentsMargins(0, 0, 0, 0)
        correction_lay.setSpacing(4)
        correction_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.correction_input_files = _stage_files("Inputs", [
            ("3_cell/tracked_labels.tif", "Cell labels"),
            ("0_input/cell_zavg.tif", "Cell z-avg"),
            ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
        ])
        correction_lay.addWidget(self.correction_input_files)

        self.load_cell_correction_btn = QPushButton("Load Cell Labels")
        self.save_cell_correction_btn = QPushButton("Save Cell Labels")
        self.reassign_cell_ids_btn = QPushButton("Reassign IDs")
        self.expand_selected_cell_btn = QPushButton("Expand Selected Cell")
        for button in (
            self.load_cell_correction_btn,
            self.save_cell_correction_btn,
            self.reassign_cell_ids_btn,
            self.expand_selected_cell_btn,
        ):
            button.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        correction_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(
            correction_btn_row,
            0,
            self.load_cell_correction_btn,
            self.save_cell_correction_btn,
        )
        add_block_button_row(correction_btn_row, 1, self.reassign_cell_ids_btn)
        correction_lay.addLayout(correction_btn_row)

        expand_grid = _param_grid()
        self.expand_cell_max_px_spin = _ispin(0, 999, 25)
        add_parameter_grid_row(
            expand_grid,
            0,
            0,
            "Max expansion px:",
            self.expand_cell_max_px_spin,
        )
        correction_lay.addLayout(expand_grid)
        expand_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(expand_btn_row, 0, self.expand_selected_cell_btn)
        correction_lay.addLayout(expand_btn_row)

        self.correction_status_lbl = _stage_status()
        correction_lay.addWidget(self.correction_status_lbl)
        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
        )
        correction_lay.addWidget(self.correction_widget)
        self.correction_shortcuts_section = CollapsibleSection(
            "Correction Shortcuts",
            self.correction_widget.build_shortcuts_widget(),
            expanded=False,
        )
        correction_lay.addWidget(self.correction_shortcuts_section)
        self.correction_section = CollapsibleSection(
            "Correction", self.correction_params_widget, expanded=False
        )
        layout.addWidget(self.correction_section)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.ff_flow_mag_btn.clicked.connect(self._on_create_flow_mag)
        self.load_cell_correction_btn.clicked.connect(self._on_load_cell_correction)
        self.save_cell_correction_btn.clicked.connect(self._on_save_cell_correction)
        self.reassign_cell_ids_btn.clicked.connect(self._on_reassign_cell_ids)
        self.expand_selected_cell_btn.clicked.connect(self._on_expand_selected_cell)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif" if self._pos_dir else None

    def _dp_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_dp_3dt.tif" if self._pos_dir else None

    def _foreground_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "foreground_masks.tif" if self._pos_dir else None

    def _nucleus_labels_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    def _flow_mag_out_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "filtered_flow_mag.tif" if self._pos_dir else None

    def _filtered_dp_out_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "filtered_dp.tif" if self._pos_dir else None

    def _cell_labels_out_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "tracked_labels.tif" if self._pos_dir else None

    def _cell_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "cell_zavg.tif" if self._pos_dir else None

    def _nucleus_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "nucleus_zavg.tif" if self._pos_dir else None

    # ------------------------------------------------------------------
    # State + status
    # ------------------------------------------------------------------
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._refresh_stage_files(pos_dir)
        self._seg_widget.refresh(pos_dir)

    def _refresh_stage_files(self, pos_dir: Path | None = None) -> None:
        if pos_dir is None:
            pos_dir = self._pos_dir
        for files_widget in (
            self.filtered_flow_input_files,
            self.filtered_flow_output_files,
            self.correction_input_files,
        ):
            files_widget.refresh(pos_dir)

    def _set_correction_status(self, msg: str) -> None:
        self.correction_status_lbl.setText(msg)
        self.correction_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def get_state(self) -> dict:
        return {
            "flow_following": {
                "median_time":  self.ff_median_time_spin.value(),
                "median_space": self.ff_median_space_spin.value(),
                "gauss_time":   self.ff_gauss_time_spin.value(),
                "gauss_space":  self.ff_gauss_space_spin.value(),
            },
            "segmentation": self._seg_widget.get_state(),
        }

    def set_state(self, state: dict) -> None:
        if "flow_following" in state:
            ff = state["flow_following"]
            if "median_time"  in ff: self.ff_median_time_spin.setValue(ff["median_time"])
            if "median_space" in ff: self.ff_median_space_spin.setValue(ff["median_space"])
            if "gauss_time"   in ff: self.ff_gauss_time_spin.setValue(ff["gauss_time"])
            if "gauss_space"  in ff: self.ff_gauss_space_spin.setValue(ff["gauss_space"])
            if "segmentation" in state: 
                self._seg_widget.set_state(state["segmentation"])


    def _set_stage_status(self, stage: str, msg: str) -> None:
        label = self._stage_status_label(stage)
        label.setText(msg)
        label.setVisible(bool(msg))
        logger.info(msg)

    def _stage_status_label(self, stage: str) -> QLabel:
        return {
            "filtered_flow": self.filtered_flow_status_lbl,
        }[stage]

    def _stage_progress_bar(self, stage: str) -> QProgressBar:
        return {
            "filtered_flow": self.filtered_flow_progress_bar,
        }[stage]

    def _set_ff_buttons_running(self, running: bool) -> None:
        self.ff_flow_mag_btn.setEnabled(not running)
        if not running:
            self.filtered_flow_progress_bar.setValue(0)
            self.filtered_flow_progress_bar.setVisible(False)

    def _show_layer(self, layer_name: str, data: np.ndarray, kwargs: dict, adder) -> None:
        if layer_name in self.viewer.layers:
            try:
                self.viewer.layers[layer_name].data = data
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[layer_name])
                adder(data, name=layer_name, **kwargs)
        else:
            adder(data, name=layer_name, **kwargs)

    def _on_stage_progress(self, stage: str, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            bar = self._stage_progress_bar(stage)
            if total > 0:
                bar.setVisible(True)
                bar.setRange(0, total)
                bar.setValue(done)
            self._set_stage_status(stage, msg)
        else:
            self._set_stage_status(stage, str(data))

    def _on_stage_worker_error(self, stage: str, exc: Exception) -> None:
        if self._ff_worker is None:
            return
        self._ff_worker = None
        self._set_ff_buttons_running(False)
        self._set_stage_status(stage, f"Error: {exc}")
        logger.exception("Cell workflow worker error", exc_info=exc)

    # ------------------------------------------------------------------
    # Manual correction
    # ------------------------------------------------------------------
    @staticmethod
    def _broadcast_reference_image(image: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
        if image is None:
            return None
        if image.ndim == 2 and len(shape) >= 3:
            return np.broadcast_to(image[np.newaxis], (shape[0],) + image.shape).copy()
        return image

    def _on_load_cell_correction(self) -> None:
        labels_path = self._cell_labels_out_path()
        cell_zavg_path = self._cell_zavg_path()
        nuc_zavg_path = self._nucleus_zavg_path()
        if labels_path is None or not labels_path.exists():
            self._set_correction_status("No cell labels file found.")
            return
        self._set_correction_status("Loading cell labels...")

        @thread_worker(connect={
            "returned": self._on_load_cell_correction_done,
            "errored": lambda exc: self._set_correction_status(f"Error: {exc}"),
        })
        def _worker():
            labels = read_full_tracked_stack(labels_path)
            cell_zavg = (
                np.asarray(tifffile.imread(str(cell_zavg_path)), dtype=np.float32)
                if cell_zavg_path and cell_zavg_path.exists() else None
            )
            nuc_zavg = (
                np.asarray(tifffile.imread(str(nuc_zavg_path)), dtype=np.float32)
                if nuc_zavg_path and nuc_zavg_path.exists() else None
            )
            return labels, cell_zavg, nuc_zavg

        _worker()

    def _on_load_cell_correction_done(self, result: tuple) -> None:
        labels, cell_zavg, nuc_zavg = result
        if _TRACKED_CELL_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_CELL_LAYER].data = labels
        else:
            self.viewer.add_labels(labels, name=_TRACKED_CELL_LAYER)

        for image, layer_name, cmap in (
            (self._broadcast_reference_image(cell_zavg, labels.shape), _CELL_ZAVG_LAYER, "gray"),
            (self._broadcast_reference_image(nuc_zavg, labels.shape), _NUC_ZAVG_LAYER, "bop orange"),
        ):
            if image is None:
                continue
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = image
            else:
                self.viewer.add_image(image, name=layer_name, colormap=cmap, blending="additive")

        self._set_correction_status(f"Loaded cell label stack {labels.shape} into napari.")
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        self.correction_widget.activate_layer(layer)
        self.correction_section.expand()

    def set_selection_callback(self, fn) -> None:
        """Register a callback for cell correction label selection changes."""
        self.correction_widget.set_selection_callback(fn)

    def select_matching_cell_label(
        self,
        t: int,
        source_label: int,
        *,
        source_labels: np.ndarray | None = None,
    ) -> None:
        """Highlight the cell label that best overlaps a selected nucleus label."""
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            return
        if source_labels is None:
            if "Tracked: Nucleus" not in self.viewer.layers:
                return
            source_labels = np.asarray(self.viewer.layers["Tracked: Nucleus"].data)
        target_labels = np.asarray(self.viewer.layers[_TRACKED_CELL_LAYER].data)
        matched_label = best_overlapping_label(target_labels, source_labels, t, source_label)
        self.correction_widget.select_label(t, matched_label, notify=False)

    def _on_save_cell_correction(self) -> None:
        labels_path = self._cell_labels_out_path()
        if labels_path is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._set_correction_status("No cell labels layer to save.")
            return
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        data = np.asarray(layer.data)
        if data.ndim != 3:
            self._set_correction_status("Cell labels layer is not a 3D stack.")
            return
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            str(labels_path),
            data.astype(np.uint32, copy=False),
            compression="zlib",
        )
        self._refresh_stage_files(self._pos_dir)
        self._set_correction_status(f"Saved {data.shape[0]} frame(s) to {labels_path.name}.")

    def _on_reassign_cell_ids(self) -> None:
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._set_correction_status("No cell labels layer loaded.")
            return
        stack = np.asarray(self.viewer.layers[_TRACKED_CELL_LAYER].data)
        unique_ids = np.unique(stack)
        unique_ids = unique_ids[unique_ids != 0]
        if unique_ids.size == 0:
            self._set_correction_status("No cell IDs to reassign.")
            return
        lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.uint32)
        for new_id, old_id in enumerate(unique_ids, start=1):
            lut[int(old_id)] = new_id
        self.viewer.layers[_TRACKED_CELL_LAYER].data = lut[stack]
        self._set_correction_status(
            f"Reassigned {len(unique_ids)} cell IDs to contiguous range 1-{len(unique_ids)}. Unsaved."
        )

    def _foreground_stack_for_expansion(self) -> np.ndarray | None:
        if _FOREGROUND_MASK_LAYER in self.viewer.layers:
            return np.asarray(self.viewer.layers[_FOREGROUND_MASK_LAYER].data)
        fg_path = self._foreground_path()
        if fg_path is None or not fg_path.exists():
            return None
        foreground = np.asarray(tifffile.imread(str(fg_path)))
        self._show_layer(_FOREGROUND_MASK_LAYER, foreground, {}, self.viewer.add_labels)
        return foreground

    def _on_expand_selected_cell(self) -> None:
        if self._pos_dir is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked cell labels layer loaded.")
            return
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        if self.correction_widget._layer is not layer:
            self._set_correction_status("No active tracked cell labels layer.")
            return
        label_id = int(self.correction_widget._selected_label)
        if label_id == 0:
            self._set_correction_status("No cell selected.")
            return

        labels = np.asarray(layer.data)
        if labels.ndim < 3:
            self._set_correction_status("Tracked cell labels layer is not a 3D stack.")
            return
        t = self._current_time_index(labels.shape[0])
        seg2d = self.correction_widget._frame_view(layer, t)
        if not np.any(seg2d == label_id):
            self._set_correction_status(f"Cell {label_id} not present at t={t}.")
            return

        foreground = self._foreground_stack_for_expansion()
        if foreground is None:
            self._set_correction_status("Foreground mask not found.")
            return
        if foreground.shape != labels.shape:
            self._set_correction_status(
                f"Foreground mask shape {foreground.shape} does not match labels shape {labels.shape}."
            )
            return
        foreground2d = foreground[t]
        while foreground2d.ndim > 2:
            if foreground2d.shape[0] != 1:
                self._set_correction_status(
                    f"Foreground mask frame has unsupported shape {foreground2d.shape}."
                )
                return
            foreground2d = foreground2d[0]

        before = seg2d.copy()
        try:
            added = expand_label_to_foreground(
                seg2d,
                foreground2d,
                label_id,
                max_distance=int(self.expand_cell_max_px_spin.value()),
            )
        except ValueError as exc:
            self._set_correction_status(str(exc))
            return
        if added == 0:
            seed_touches_foreground = bool(np.any((foreground2d > 0) & (before == label_id)))
            if not seed_touches_foreground:
                self._set_correction_status(
                    f"Cell {label_id} does not touch foreground at t={t}."
                )
            else:
                self._set_correction_status(f"Expansion added no pixels for cell {label_id} at t={t}.")
            return

        self.correction_widget._record_history(layer, t, before)
        layer.refresh()
        self.correction_widget._update_highlight(t, label_id)
        self._set_correction_status(
            f"Expanded cell {label_id} at t={t} by {added} px. Unsaved."
        )

    # ------------------------------------------------------------------
    # Run / Cancel
    # ------------------------------------------------------------------
    def _read_dp_tcyx(self, prob_path: Path, dp_path: Path) -> np.ndarray:
        from cellflow.segmentation._array_utils import normalize_seeded_watershed_dp_stack

        prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
        if prob.ndim == 3:
            prob = prob[np.newaxis]
        dp_raw = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
        dp_full = normalize_seeded_watershed_dp_stack(dp_raw, prob.shape)
        return dp_full[:, :, :2].mean(axis=1).astype(np.float32)

    def _on_create_flow_mag(self) -> None:
        if self._pos_dir is None:
            self._set_stage_status("filtered_flow", "No project open.")
            return

        prob_path = self._prob_path()
        dp_path = self._dp_path()
        filtered_dp_path = self._filtered_dp_out_path()
        flow_mag_path = self._flow_mag_out_path()

        for path, name in [
            (prob_path, "cell_prob_3dt.tif"),
            (dp_path,   "cell_dp_3dt.tif"),
        ]:
            if path is None or not path.exists():
                self._set_stage_status("filtered_flow", f"Missing: {name}")
                return
        if filtered_dp_path is None or flow_mag_path is None:
            self._set_stage_status("filtered_flow", "No project open.")
            return

        params_snapshot = self._params_from_ui()
        pos_dir = self._pos_dir

        def _on_done(result):
            self._ff_worker = None
            self._set_ff_buttons_running(False)
            filtered_mag = result
            self._show_layer(
                _FILTERED_FLOW_LAYER,
                filtered_mag,
                {"colormap": "inferno", "blending": "additive"},
                self.viewer.add_image,
            )
            self._refresh_stage_files(pos_dir)
            self._set_stage_status("filtered_flow", "Flow magnitude complete.")

        @thread_worker(connect={
            "yielded":  lambda data: self._on_stage_progress("filtered_flow", data),
            "returned": _on_done,
            "errored":  lambda exc: self._on_stage_worker_error("filtered_flow", exc),
        })
        def _worker():
            from cellflow.segmentation import compute_filtered_flow_vectors

            yield (0, 4, "Loading flow inputs...")
            dp_tcyx = self._read_dp_tcyx(prob_path, dp_path)

            yield (1, 4, "Filtering flow vectors...")
            filtered_dp = compute_filtered_flow_vectors(dp_tcyx, params_snapshot)

            yield (2, 4, "Creating flow magnitude...")
            filtered_mag = np.sqrt(
                filtered_dp[:, 0] ** 2 + filtered_dp[:, 1] ** 2
            ).astype(np.float32)

            yield (3, 4, "Saving flow magnitude...")
            filtered_dp_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(filtered_dp_path), filtered_dp, compression="zlib")
            tifffile.imwrite(str(flow_mag_path), filtered_mag, compression="zlib")
            return filtered_mag

        self._set_stage_status("filtered_flow", "Creating flow magnitude...")
        self._set_ff_buttons_running(True)
        self._ff_worker = _worker()

    def _current_time_index(self, max_t: int) -> int:
        step = getattr(getattr(self.viewer, "dims", None), "current_step", ())
        if not step:
            return 0
        return min(max(int(step[0]), 0), max(max_t - 1, 0))


    def _params_from_ui(self):
        from cellflow.segmentation import FlowFollowingParams
        return FlowFollowingParams(
            median_kernel_time=int(self.ff_median_time_spin.value()),
            median_kernel_space=int(self.ff_median_space_spin.value()),
            gaussian_sigma_time=float(self.ff_gauss_time_spin.value()),
            gaussian_sigma_space=float(self.ff_gauss_space_spin.value()),
        )
