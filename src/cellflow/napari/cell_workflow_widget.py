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

from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import add_block_button_row, block_grid, sweep_parameter_grid

logger = logging.getLogger(__name__)

_FILTERED_FLOW_LAYER = "Filtered Flow Magnitude"
_FOREGROUND_MASK_LAYER = "Foreground Mask"
_CELL_LABELS_LAYER = "Cell Labels"
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

        self.input_files = PipelineFilesWidget([
            ("Inputs", [
                ("1_cellpose/cell_prob_3dt.tif",   "Cell prob 3D+t"),
                ("1_cellpose/cell_dp_3dt.tif",     "Cell dp 3D+t"),
                ("3_cell/foreground_masks.tif",    "Foreground masks"),
                ("2_nucleus/tracked_labels.tif",   "Nucleus tracked labels"),
            ]),
        ], viewer=self.viewer)
        layout.addWidget(self.input_files)

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
        filter_grid = sweep_parameter_grid(spin_width=_FF_SPIN_WIDTH)
        self.ff_median_time_spin   = _ispin(1, 15, 3)
        self.ff_median_space_spin  = _ispin(1, 15, 5)
        self.ff_gauss_time_spin    = _dspin(0.0, 10.0, 0.0, 0.1)
        self.ff_gauss_space_spin   = _dspin(0.0, 10.0, 0.0, 0.1)
        filter_grid.addWidget(QLabel("Median t kernel:"),  1, 0)
        filter_grid.addWidget(self.ff_median_time_spin,    1, 1)
        filter_grid.addWidget(QLabel("Median xy kernel:"), 2, 0)
        filter_grid.addWidget(self.ff_median_space_spin,   2, 1)
        filter_grid.addWidget(QLabel("Gaussian t σ:"),     3, 0)
        filter_grid.addWidget(self.ff_gauss_time_spin,     3, 1)
        filter_grid.addWidget(QLabel("Gaussian xy σ:"),    4, 0)
        filter_grid.addWidget(self.ff_gauss_space_spin,    4, 1)
        filter_grid.setColumnStretch(1, 1)
        filter_lay.addLayout(filter_grid)

        self.ff_flow_mag_btn = QPushButton("Create filtered_dp")
        self.ff_flow_mag_btn.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
        self.ff_flow_mag_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        filter_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(filter_btn_row, 0, self.ff_flow_mag_btn)
        filter_lay.addLayout(filter_btn_row)

        self.filtered_flow_section = CollapsibleSection(
            "Filtered Flow", self.filtered_flow_params_widget, expanded=True
        )
        layout.addWidget(self.filtered_flow_section)

        self.foreground_mask_params_widget = QWidget()
        fg_lay = QVBoxLayout(self.foreground_mask_params_widget)
        fg_lay.setContentsMargins(0, 0, 0, 0)
        fg_lay.setSpacing(4)
        fg_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        fg_grid = sweep_parameter_grid(spin_width=_FF_SPIN_WIDTH)
        self.fg_cellprob_threshold_spin = _dspin(-10.0, 10.0, 0.0, 0.1)
        self.fg_flow_threshold_spin = _dspin(0.0, 10.0, 0.0, 0.1)
        self.fg_min_size_spin = _ispin(0, 100000, 15)
        self.fg_niter_spin = _ispin(1, 2000, 200, step=10)
        fg_grid.addWidget(QLabel("Cellprob threshold:"), 1, 0)
        fg_grid.addWidget(self.fg_cellprob_threshold_spin, 1, 1)
        fg_grid.addWidget(QLabel("Flow threshold:"), 2, 0)
        fg_grid.addWidget(self.fg_flow_threshold_spin, 2, 1)
        fg_grid.addWidget(QLabel("Min size:"), 3, 0)
        fg_grid.addWidget(self.fg_min_size_spin, 3, 1)
        fg_grid.addWidget(QLabel("Niter:"), 4, 0)
        fg_grid.addWidget(self.fg_niter_spin, 4, 1)
        fg_grid.setColumnStretch(1, 1)
        fg_lay.addLayout(fg_grid)

        self.fg_masks_btn = QPushButton("Create foreground_masks")
        self.fg_masks_btn.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
        self.fg_masks_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        fg_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(fg_btn_row, 0, self.fg_masks_btn)
        fg_lay.addLayout(fg_btn_row)

        self.foreground_mask_section = CollapsibleSection(
            "Foreground Mask", self.foreground_mask_params_widget, expanded=True
        )
        layout.addWidget(self.foreground_mask_section)

        self.tracked_labels_params_widget = QWidget()
        labels_lay = QVBoxLayout(self.tracked_labels_params_widget)
        labels_lay.setContentsMargins(0, 0, 0, 0)
        labels_lay.setSpacing(4)
        labels_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        labels_grid = sweep_parameter_grid(spin_width=_FF_SPIN_WIDTH)
        self.ff_flow_weight_spin     = _dspin(0.0, 1.0, 0.5, 0.05, decimals=2)
        self.ff_step_scale_spin      = _dspin(0.05, 1.0, 0.2, 0.05, decimals=2)
        self.ff_max_iter_spin        = _ispin(10, 500, 100, step=10)
        self.ff_capture_radius_spin  = _dspin(0.5, 10.0, 3.0, 0.5)
        labels_grid.addWidget(QLabel("Flow weight:"),       1, 0)
        labels_grid.addWidget(self.ff_flow_weight_spin,     1, 1)
        labels_grid.addWidget(QLabel("Step scale:"),        2, 0)
        labels_grid.addWidget(self.ff_step_scale_spin,      2, 1)
        labels_grid.addWidget(QLabel("Max iterations:"),    3, 0)
        labels_grid.addWidget(self.ff_max_iter_spin,        3, 1)
        labels_grid.addWidget(QLabel("Capture radius:"),    4, 0)
        labels_grid.addWidget(self.ff_capture_radius_spin,  4, 1)
        labels_grid.setColumnStretch(1, 1)
        labels_lay.addLayout(labels_grid)

        self.ff_labels_btn = QPushButton("Create tracked_labels")
        self.ff_labels_btn.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
        self.ff_labels_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        labels_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(labels_btn_row, 0, self.ff_labels_btn)
        labels_lay.addLayout(labels_btn_row)

        self.tracked_labels_section = CollapsibleSection(
            "Tracked Cell Labels", self.tracked_labels_params_widget, expanded=True
        )
        layout.addWidget(self.tracked_labels_section)

        self.ff_cancel_btn = QPushButton("Cancel")
        self.ff_cancel_btn.setEnabled(False)
        self.ff_cancel_btn.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
        self.ff_cancel_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        cancel_row = block_grid(horizontal_spacing=12)
        add_block_button_row(cancel_row, 0, self.ff_cancel_btn)
        layout.addLayout(cancel_row)

        self.ff_input_lbl = QLabel("")
        self.ff_input_lbl.setWordWrap(True)
        layout.addWidget(self.ff_input_lbl)

        self.ff_status_lbl = QLabel("")
        self.ff_status_lbl.setWordWrap(True)
        self.ff_status_lbl.setVisible(False)
        layout.addWidget(self.ff_status_lbl)

        self.ff_progress_bar = QProgressBar()
        self.ff_progress_bar.setRange(0, 100)
        self.ff_progress_bar.setValue(0)
        self.ff_progress_bar.setVisible(False)
        layout.addWidget(self.ff_progress_bar)

        self._update_ff_status_labels()

        self.ff_files = PipelineFilesWidget([
            ("Outputs", [
                ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
                ("3_cell/filtered_flow_mag.tif", "Filtered flow magnitude"),
                ("3_cell/tracked_labels.tif",    "Cell labels"),
            ]),
        ], viewer=self.viewer)
        layout.addWidget(self.ff_files)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.ff_flow_mag_btn.clicked.connect(self._on_create_flow_mag)
        self.fg_masks_btn.clicked.connect(self._on_create_foreground_masks)
        self.ff_labels_btn.clicked.connect(self._on_create_tracked_labels)
        self.ff_cancel_btn.clicked.connect(self._on_cancel_flow_following)

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

    # ------------------------------------------------------------------
    # State + status
    # ------------------------------------------------------------------
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self.input_files.refresh(pos_dir)
        self.ff_files.refresh(pos_dir)
        self._update_ff_status_labels()

    def get_state(self) -> dict:
        return {
            "flow_following": {
                "median_time":     self.ff_median_time_spin.value(),
                "median_space":    self.ff_median_space_spin.value(),
                "gauss_time":      self.ff_gauss_time_spin.value(),
                "gauss_space":     self.ff_gauss_space_spin.value(),
                "flow_weight":     self.ff_flow_weight_spin.value(),
                "step_scale":      self.ff_step_scale_spin.value(),
                "max_iter":        self.ff_max_iter_spin.value(),
                "capture_radius":  self.ff_capture_radius_spin.value(),
            },
            "foreground_mask": {
                "cellprob_threshold": self.fg_cellprob_threshold_spin.value(),
                "flow_threshold":     self.fg_flow_threshold_spin.value(),
                "min_size":           self.fg_min_size_spin.value(),
                "niter":              self.fg_niter_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if "flow_following" in state:
            ff = state["flow_following"]
            if "median_time"    in ff: self.ff_median_time_spin.setValue(ff["median_time"])
            if "median_space"   in ff: self.ff_median_space_spin.setValue(ff["median_space"])
            if "gauss_time"     in ff: self.ff_gauss_time_spin.setValue(ff["gauss_time"])
            if "gauss_space"    in ff: self.ff_gauss_space_spin.setValue(ff["gauss_space"])
            if "flow_weight"    in ff: self.ff_flow_weight_spin.setValue(ff["flow_weight"])
            if "step_scale"     in ff: self.ff_step_scale_spin.setValue(ff["step_scale"])
            if "max_iter"       in ff: self.ff_max_iter_spin.setValue(ff["max_iter"])
            if "capture_radius" in ff: self.ff_capture_radius_spin.setValue(ff["capture_radius"])
        if "foreground_mask" in state:
            fg = state["foreground_mask"]
            if "cellprob_threshold" in fg:
                self.fg_cellprob_threshold_spin.setValue(fg["cellprob_threshold"])
            if "flow_threshold" in fg:
                self.fg_flow_threshold_spin.setValue(fg["flow_threshold"])
            if "min_size" in fg:
                self.fg_min_size_spin.setValue(fg["min_size"])
            if "niter" in fg:
                self.fg_niter_spin.setValue(fg["niter"])

    def _update_ff_status_labels(self) -> None:
        if self._pos_dir is None:
            self.ff_input_lbl.setText("Inputs: no project open.")
            return
        check = "✓"
        cross = "✗"
        prob_ok    = (p := self._prob_path()) is not None and p.exists()
        dp_ok      = (p := self._dp_path()) is not None and p.exists()
        filtered_ok = (p := self._filtered_dp_out_path()) is not None and p.exists()
        fg_ok      = (p := self._foreground_path()) is not None and p.exists()
        nuc_ok     = (p := self._nucleus_labels_path()) is not None and p.exists()
        self.ff_input_lbl.setText(
            f"Inputs: {check if prob_ok else cross} prob  "
            f"{check if dp_ok else cross} dp  "
            f"{check if filtered_ok else cross} filtered dp  "
            f"{check if fg_ok else cross} foreground  "
            f"{check if nuc_ok else cross} nucleus labels"
        )

    def _set_ff_status(self, msg: str) -> None:
        self.ff_status_lbl.setText(msg)
        self.ff_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_ff_buttons_running(self, running: bool) -> None:
        self.ff_flow_mag_btn.setEnabled(not running)
        self.fg_masks_btn.setEnabled(not running)
        self.ff_labels_btn.setEnabled(not running)
        self.ff_cancel_btn.setEnabled(running)
        self.ff_progress_bar.setVisible(running)
        if not running:
            self.ff_progress_bar.setValue(0)

    def _show_layer(self, layer_name: str, data: np.ndarray, kwargs: dict, adder) -> None:
        if layer_name in self.viewer.layers:
            try:
                self.viewer.layers[layer_name].data = data
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[layer_name])
                adder(data, name=layer_name, **kwargs)
        else:
            adder(data, name=layer_name, **kwargs)

    def _on_ff_progress(self, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            if total > 0:
                self.ff_progress_bar.setRange(0, total)
                self.ff_progress_bar.setValue(done)
            self._set_ff_status(msg)
        else:
            self._set_ff_status(str(data))

    def _on_ff_worker_error(self, exc: Exception) -> None:
        if self._ff_worker is None:
            return
        self._ff_worker = None
        self.ff_progress_bar.setVisible(False)
        self._set_ff_buttons_running(False)
        self._set_ff_status(f"Error: {exc}")
        logger.exception("Flow-following worker error", exc_info=exc)

    # ------------------------------------------------------------------
    # Run / Cancel
    # ------------------------------------------------------------------
    def _read_dp_tcyx(self, prob_path: Path, dp_path: Path) -> np.ndarray:
        from cellflow.database.hypotheses import normalize_seeded_watershed_dp_stack

        prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
        if prob.ndim == 3:
            prob = prob[np.newaxis]
        dp_raw = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
        dp_full = normalize_seeded_watershed_dp_stack(dp_raw, prob.shape)
        return dp_full[:, :, :2].mean(axis=1).astype(np.float32)

    def _on_create_flow_mag(self) -> None:
        if self._pos_dir is None:
            self._set_ff_status("No project open.")
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
                self._set_ff_status(f"Missing: {name}")
                return
        if filtered_dp_path is None or flow_mag_path is None:
            self._set_ff_status("No project open.")
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
            self.ff_files.refresh(pos_dir)
            self._update_ff_status_labels()
            self._set_ff_status("Flow magnitude complete.")

        @thread_worker(connect={
            "yielded":  self._on_ff_progress,
            "returned": _on_done,
            "errored":  self._on_ff_worker_error,
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

        self._set_ff_status("Creating flow magnitude...")
        self._set_ff_buttons_running(True)
        self._ff_worker = _worker()

    def _on_create_foreground_masks(self) -> None:
        if self._pos_dir is None:
            self._set_ff_status("No project open.")
            return

        prob_path = self._prob_path()
        filtered_dp_path = self._filtered_dp_out_path()
        fg_path = self._foreground_path()

        for path, name in [
            (prob_path, "cell_prob_3dt.tif"),
            (filtered_dp_path, "filtered_dp.tif (run Filtered Flow first)"),
        ]:
            if path is None or not path.exists():
                self._set_ff_status(f"Missing: {name}")
                return
        if fg_path is None:
            self._set_ff_status("No project open.")
            return

        params_snapshot = self._foreground_params_from_ui()
        pos_dir = self._pos_dir

        def _on_done(result):
            self._ff_worker = None
            self._set_ff_buttons_running(False)
            foreground = result
            self._show_layer(
                _FOREGROUND_MASK_LAYER,
                foreground,
                {},
                self.viewer.add_labels,
            )
            self.input_files.refresh(pos_dir)
            self.ff_files.refresh(pos_dir)
            self._update_ff_status_labels()
            self._set_ff_status("Foreground masks complete.")

        @thread_worker(connect={
            "yielded":  self._on_ff_progress,
            "returned": _on_done,
            "errored":  self._on_ff_worker_error,
        })
        def _worker():
            from cellflow.segmentation import compute_cellpose_foreground_masks

            yield (0, 4, "Loading foreground inputs...")
            prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            filtered_dp = np.asarray(tifffile.imread(str(filtered_dp_path)), dtype=np.float32)

            yield (1, 4, "Creating foreground masks...")
            foreground = compute_cellpose_foreground_masks(
                prob,
                filtered_dp,
                **params_snapshot,
                progress_cb=None,
            )

            yield (3, 4, "Saving foreground masks...")
            fg_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(
                str(fg_path),
                foreground.astype(np.uint8, copy=False),
                compression="zlib",
            )
            return foreground.astype(np.uint8, copy=False)

        self._set_ff_status("Creating foreground masks...")
        self._set_ff_buttons_running(True)
        self._ff_worker = _worker()

    def _on_create_tracked_labels(self) -> None:
        if self._pos_dir is None:
            self._set_ff_status("No project open.")
            return

        filtered_dp_path = self._filtered_dp_out_path()
        fg_path = self._foreground_path()
        nuc_path = self._nucleus_labels_path()
        labels_path = self._cell_labels_out_path()

        for path, name in [
            (filtered_dp_path, "filtered_dp.tif"),
            (fg_path,   "foreground_masks.tif"),
            (nuc_path,  "tracked_labels.tif (2_nucleus)"),
        ]:
            if path is None or not path.exists():
                self._set_ff_status(f"Missing: {name}")
                return
        if labels_path is None:
            self._set_ff_status("No project open.")
            return

        params_snapshot = self._params_from_ui()
        pos_dir = self._pos_dir

        def _on_done(result):
            self._ff_worker = None
            self._set_ff_buttons_running(False)
            labels = result
            self._show_layer(_CELL_LABELS_LAYER, labels, {}, self.viewer.add_labels)
            self.ff_files.refresh(pos_dir)
            self._update_ff_status_labels()
            self._set_ff_status("Tracked labels complete.")

        @thread_worker(connect={
            "yielded":  self._on_ff_progress,
            "returned": _on_done,
            "errored":  self._on_ff_worker_error,
        })
        def _worker():
            from cellflow.segmentation import compute_flow_following_movie

            yield (0, 5, "Loading filtered flow vectors...")
            dp_tcyx = np.asarray(tifffile.imread(str(filtered_dp_path)), dtype=np.float32)

            yield (1, 5, "Loading foreground...")
            foreground = np.asarray(tifffile.imread(str(fg_path)), dtype=bool)

            yield (2, 5, "Loading nucleus labels...")
            nucleus = np.asarray(tifffile.imread(str(nuc_path)), dtype=np.int32)

            yield (3, 5, "Creating tracked labels...")
            _, cell_labels = compute_flow_following_movie(
                foreground, dp_tcyx, nucleus, params_snapshot, progress_cb=None,
                filter_vectors=False,
            )

            yield (4, 5, "Saving tracked labels...")
            tifffile.imwrite(
                str(labels_path),
                cell_labels.astype(np.uint32),
                compression="zlib",
            )
            return cell_labels.astype(np.uint32)

        self._set_ff_status("Creating tracked labels...")
        self._set_ff_buttons_running(True)
        self._ff_worker = _worker()

    def _on_run_flow_following(self) -> None:
        self._on_create_tracked_labels()

    def _on_cancel_flow_following(self) -> None:
        if self._ff_worker is not None:
            worker = self._ff_worker
            self._ff_worker = None
            worker.quit()
        self._set_ff_buttons_running(False)
        self._set_ff_status("Flow-following cancelled.")

    def _params_from_ui(self):
        from cellflow.segmentation import FlowFollowingParams
        return FlowFollowingParams(
            median_kernel_time=int(self.ff_median_time_spin.value()),
            median_kernel_space=int(self.ff_median_space_spin.value()),
            gaussian_sigma_time=float(self.ff_gauss_time_spin.value()),
            gaussian_sigma_space=float(self.ff_gauss_space_spin.value()),
            flow_weight=float(self.ff_flow_weight_spin.value()),
            flow_step_scale=float(self.ff_step_scale_spin.value()),
            max_iterations=int(self.ff_max_iter_spin.value()),
            capture_radius=float(self.ff_capture_radius_spin.value()),
        )

    def _foreground_params_from_ui(self) -> dict[str, object]:
        return {
            "cellprob_threshold": float(self.fg_cellprob_threshold_spin.value()),
            "flow_threshold": float(self.fg_flow_threshold_spin.value()),
            "min_size": int(self.fg_min_size_spin.value()),
            "niter": int(self.fg_niter_spin.value()),
        }
