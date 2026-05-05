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
    QFrame,
    QSpinBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import add_block_button_row, block_grid, sweep_parameter_grid

logger = logging.getLogger(__name__)

_FILTERED_FLOW_LAYER = "Filtered Flow Magnitude"
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
        ])
        layout.addWidget(self.input_files)

        _ff_inner = QWidget()
        ff_lay = QVBoxLayout(_ff_inner)
        ff_lay.setContentsMargins(4, 4, 4, 4)
        ff_lay.setSpacing(4)
        ff_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        ff_scroll = QScrollArea()
        ff_scroll.setWidgetResizable(True)
        ff_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        ff_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ff_scroll.setFrameShape(QFrame.NoFrame)
        ff_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        ff_params_widget = QWidget()
        ff_params_widget.setMinimumWidth(520)
        ff_params_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        ff_params_lay = QVBoxLayout(ff_params_widget)
        ff_params_lay.setContentsMargins(0, 0, 0, 0)
        ff_params_lay.setSpacing(4)
        ff_params_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

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

        ff_grid = sweep_parameter_grid(spin_width=_FF_SPIN_WIDTH)

        self.ff_median_time_spin   = _ispin(1, 15, 3)
        self.ff_median_space_spin  = _ispin(1, 15, 5)
        self.ff_gauss_time_spin    = _dspin(0.0, 10.0, 0.0, 0.1)
        self.ff_gauss_space_spin   = _dspin(0.0, 10.0, 0.0, 0.1)
        ff_grid.addWidget(QLabel("Median t kernel:"),  1, 0)
        ff_grid.addWidget(self.ff_median_time_spin,    1, 1)
        ff_grid.addWidget(QLabel("Median xy kernel:"), 2, 0)
        ff_grid.addWidget(self.ff_median_space_spin,   2, 1)
        ff_grid.addWidget(QLabel("Gaussian t σ:"),     3, 0)
        ff_grid.addWidget(self.ff_gauss_time_spin,     3, 1)
        ff_grid.addWidget(QLabel("Gaussian xy σ:"),    4, 0)
        ff_grid.addWidget(self.ff_gauss_space_spin,    4, 1)

        self.ff_flow_weight_spin     = _dspin(0.0, 1.0, 0.5, 0.05, decimals=2)
        self.ff_step_scale_spin      = _dspin(0.05, 1.0, 0.2, 0.05, decimals=2)
        self.ff_max_iter_spin        = _ispin(10, 500, 100, step=10)
        self.ff_capture_radius_spin  = _dspin(0.5, 10.0, 3.0, 0.5)
        ff_grid.addWidget(QLabel("Flow weight:"),       5, 0)
        ff_grid.addWidget(self.ff_flow_weight_spin,     5, 1)
        ff_grid.addWidget(QLabel("Step scale:"),        6, 0)
        ff_grid.addWidget(self.ff_step_scale_spin,      6, 1)
        ff_grid.addWidget(QLabel("Max iterations:"),    7, 0)
        ff_grid.addWidget(self.ff_max_iter_spin,        7, 1)
        ff_grid.addWidget(QLabel("Capture radius:"),    8, 0)
        ff_grid.addWidget(self.ff_capture_radius_spin,  8, 1)
        ff_grid.setColumnStretch(1, 1)

        ff_params_lay.addLayout(ff_grid)

        self.ff_run_btn    = QPushButton("Run")
        self.ff_cancel_btn = QPushButton("Cancel")
        self.ff_cancel_btn.setEnabled(False)
        for btn in (self.ff_run_btn, self.ff_cancel_btn):
            btn.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(btn_row, 0, self.ff_run_btn, self.ff_cancel_btn)
        ff_params_lay.addLayout(btn_row)

        self.ff_input_lbl = QLabel("")
        self.ff_input_lbl.setWordWrap(True)
        ff_params_lay.addWidget(self.ff_input_lbl)

        self.ff_status_lbl = QLabel("")
        self.ff_status_lbl.setWordWrap(True)
        self.ff_status_lbl.setVisible(False)
        ff_params_lay.addWidget(self.ff_status_lbl)

        self.ff_progress_bar = QProgressBar()
        self.ff_progress_bar.setRange(0, 100)
        self.ff_progress_bar.setValue(0)
        self.ff_progress_bar.setVisible(False)
        ff_params_lay.addWidget(self.ff_progress_bar)

        self.ff_files = PipelineFilesWidget([
            ("", [
                ("3_cell/filtered_flow_mag.tif", "Filtered flow magnitude"),
                ("3_cell/tracked_labels.tif",    "Cell labels"),
            ]),
        ])
        ff_params_lay.addWidget(self.ff_files)
        self._update_ff_status_labels()

        ff_scroll.setWidget(ff_params_widget)
        ff_lay.addWidget(ff_scroll)
        self.flow_section = CollapsibleSection(
            "Flow-Following Segmentation", _ff_inner, expanded=True
        )
        layout.addWidget(self.flow_section)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.ff_run_btn.clicked.connect(self._on_run_flow_following)
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

    def _update_ff_status_labels(self) -> None:
        if self._pos_dir is None:
            self.ff_input_lbl.setText("Inputs: no project open.")
            return
        check = "✓"
        cross = "✗"
        prob_ok    = (p := self._prob_path()) is not None and p.exists()
        dp_ok      = (p := self._dp_path()) is not None and p.exists()
        fg_ok      = (p := self._foreground_path()) is not None and p.exists()
        nuc_ok     = (p := self._nucleus_labels_path()) is not None and p.exists()
        self.ff_input_lbl.setText(
            f"Inputs: {check if prob_ok else cross} prob  "
            f"{check if dp_ok else cross} dp  "
            f"{check if fg_ok else cross} foreground  "
            f"{check if nuc_ok else cross} nucleus labels"
        )

    def _set_ff_status(self, msg: str) -> None:
        self.ff_status_lbl.setText(msg)
        self.ff_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_ff_buttons_running(self, running: bool) -> None:
        self.ff_run_btn.setEnabled(not running)
        self.ff_cancel_btn.setEnabled(running)
        self.ff_progress_bar.setVisible(running)
        if not running:
            self.ff_progress_bar.setValue(0)

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
    def _on_run_flow_following(self) -> None:
        if self._pos_dir is None:
            self._set_ff_status("No project open.")
            return

        prob_path = self._prob_path()
        dp_path = self._dp_path()
        fg_path = self._foreground_path()
        nuc_path = self._nucleus_labels_path()
        flow_mag_path = self._flow_mag_out_path()
        labels_path = self._cell_labels_out_path()

        for path, name in [
            (prob_path, "cell_prob_3dt.tif"),
            (dp_path,   "cell_dp_3dt.tif"),
            (fg_path,   "foreground_masks.tif"),
            (nuc_path,  "tracked_labels.tif (2_nucleus)"),
        ]:
            if path is None or not path.exists():
                self._set_ff_status(f"Missing: {name}")
                return

        params_snapshot = self._params_from_ui()
        pos_dir = self._pos_dir

        def _on_done(result):
            self._ff_worker = None
            self._set_ff_buttons_running(False)
            filtered_mag, labels = result
            for layer_name, data, kwargs, adder in [
                (_FILTERED_FLOW_LAYER, filtered_mag,
                    {"colormap": "inferno", "blending": "additive"},
                    self.viewer.add_image),
                (_CELL_LABELS_LAYER, labels, {}, self.viewer.add_labels),
            ]:
                if layer_name in self.viewer.layers:
                    try:
                        self.viewer.layers[layer_name].data = data
                    except Exception:
                        self.viewer.layers.remove(self.viewer.layers[layer_name])
                        adder(data, name=layer_name, **kwargs)
                else:
                    adder(data, name=layer_name, **kwargs)
            self.ff_files.refresh(pos_dir)
            self._update_ff_status_labels()
            self._set_ff_status("Flow-following segmentation complete.")

        @thread_worker(connect={
            "yielded":  self._on_ff_progress,
            "returned": _on_done,
            "errored":  self._on_ff_worker_error,
        })
        def _worker():
            from cellflow.database.hypotheses import normalize_seeded_watershed_dp_stack
            from cellflow.segmentation import compute_flow_following_movie

            yield (0, 5, "Loading inputs...")
            prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            if prob.ndim == 3:
                prob = prob[np.newaxis]
            dp_raw = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
            dp_full = normalize_seeded_watershed_dp_stack(dp_raw, prob.shape)  # (T, Z, C, Y, X)
            dp_tcyx = dp_full[:, :, :2].mean(axis=1).astype(np.float32)        # (T, 2, Y, X)

            yield (1, 5, "Loading foreground...")
            foreground = np.asarray(tifffile.imread(str(fg_path)), dtype=bool)

            yield (2, 5, "Loading nucleus labels...")
            nucleus = np.asarray(tifffile.imread(str(nuc_path)), dtype=np.int32)

            yield (3, 5, "Running flow-following segmentation...")
            n_t = dp_tcyx.shape[0]

            def _progress(done: int, total: int) -> None:
                # Worker thread cannot yield from inside a callback; we rely on
                # the kernel finishing per-frame fast enough that the bar updates
                # at the next yield. Persist as text only.
                pass

            filtered_dp, cell_labels = compute_flow_following_movie(
                foreground, dp_tcyx, nucleus, params_snapshot, progress_cb=_progress,
            )

            yield (4, 5, "Saving outputs...")
            filtered_mag = np.sqrt(
                filtered_dp[:, 0] ** 2 + filtered_dp[:, 1] ** 2
            ).astype(np.float32)
            flow_mag_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(flow_mag_path), filtered_mag, compression="zlib")
            tifffile.imwrite(
                str(labels_path),
                cell_labels.astype(np.uint32),
                compression="zlib",
            )
            return filtered_mag, cell_labels.astype(np.uint32)

        self._set_ff_status("Running flow-following segmentation...")
        self._set_ff_buttons_running(True)
        self._ff_worker = _worker()

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
