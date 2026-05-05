"""Cell segmentation widget for CellFlow — Contour Maps subwidget."""
from __future__ import annotations

import logging
import shlex
import sys
import tempfile
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
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
from cellflow.napari.ui_style import (
    add_block_button_row,
    block_grid,
    compact_spinbox,
    sweep_parameter_grid,
)

logger = logging.getLogger(__name__)

_CONTOUR_LAYER = "Contour Map: Cell"
_CELLPROB_LAYER = "Cellprob Map: Cell"
_DP_MAG_LAYER = "DP Mag Map: Cell"
_SMOOTHED_CONTOUR_LAYER = "Cell Smoothed Contours"
_CELL_LABELS_LAYER = "Cell Labels"
_CONTOUR_SWEEP_WIDTH = 60
_CONTOUR_SWEEP_MIN_WIDTH = int(_CONTOUR_SWEEP_WIDTH * 0.9)


class CellWorkflowWidget(QWidget):
    """Cell segmentation — Contour Maps."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._build_worker = None
        self._watershed_worker = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        self.input_files = PipelineFilesWidget([
            ("Inputs", [
                ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
                ("1_cellpose/cell_dp_3dt.tif",  "Cell dp 3D+t"),
            ]),
        ])
        layout.addWidget(self.input_files)

        _contour_inner = QWidget()
        contour_lay = QVBoxLayout(_contour_inner)
        contour_lay.setContentsMargins(4, 4, 4, 4)
        contour_lay.setSpacing(4)
        contour_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        cp_params_scroll = QScrollArea()
        cp_params_scroll.setWidgetResizable(True)
        cp_params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        cp_params_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cp_params_scroll.setFrameShape(QFrame.NoFrame)
        cp_params_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        cp_params_widget = QWidget()
        cp_params_widget.setMinimumWidth(520)
        cp_params_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        cp_params_lay = QVBoxLayout(cp_params_widget)
        cp_params_lay.setContentsMargins(0, 0, 0, 0)
        cp_params_lay.setSpacing(4)
        cp_params_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        contour_sweep_grid = sweep_parameter_grid(spin_width=_CONTOUR_SWEEP_WIDTH)

        def _sweep_spin(lo, hi, val, step, decimals=1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return s

        self.cp_min_spin  = _sweep_spin(-20.0, 20.0, -8.0, 1.0)
        self.cp_max_spin  = _sweep_spin(-20.0, 20.0,  0.0, 1.0)
        self.cp_step_spin = _sweep_spin(0.1, 10.0,    1.0, 0.5)
        contour_sweep_grid.addWidget(QLabel("Cellprob:"), 1, 0)
        contour_sweep_grid.addWidget(self.cp_min_spin,  1, 1)
        contour_sweep_grid.addWidget(self.cp_max_spin,  1, 2)
        contour_sweep_grid.addWidget(self.cp_step_spin, 1, 3)
        contour_sweep_grid.setColumnStretch(1, 1)
        contour_sweep_grid.setColumnStretch(2, 1)
        contour_sweep_grid.setColumnStretch(3, 1)

        _gamma_tip = (
            "Gamma correction on Cellpose probability logits before boundary building. "
            "<1 boosts dim signals; >1 suppresses them. 1.0 = no correction. "
            "Contour maps are averaged over all gamma values in [min, max]."
        )
        self.cp_gamma_min_spin  = _sweep_spin(0.05, 5.0, 1.0, 0.05, decimals=2)
        self.cp_gamma_max_spin  = _sweep_spin(0.05, 5.0, 1.0, 0.05, decimals=2)
        self.cp_gamma_step_spin = _sweep_spin(0.05, 2.0, 0.25, 0.05, decimals=2)
        for w in (self.cp_gamma_min_spin, self.cp_gamma_max_spin, self.cp_gamma_step_spin):
            w.setToolTip(_gamma_tip)
        contour_sweep_grid.addWidget(QLabel("Gamma:"), 2, 0)
        contour_sweep_grid.addWidget(self.cp_gamma_min_spin,  2, 1)
        contour_sweep_grid.addWidget(self.cp_gamma_max_spin,  2, 2)
        contour_sweep_grid.addWidget(self.cp_gamma_step_spin, 2, 3)

        self._niter_spin = QSpinBox()
        self._niter_spin.setRange(100, 5000)
        self._niter_spin.setValue(200)
        self._niter_spin.setSingleStep(100)
        self._niter_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self._niter_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        contour_sweep_grid.addWidget(QLabel("Flow iterations:"), 3, 0)
        contour_sweep_grid.addWidget(self._niter_spin, 3, 1)

        cp_params_lay.addLayout(contour_sweep_grid)

        self.preview_contour_btn   = QPushButton("Preview")
        self.build_btn             = QPushButton("Build")
        self.contour_terminal_btn  = QPushButton("Run in Terminal")
        self.cancel_build_btn      = QPushButton("Cancel")
        self.cancel_build_btn.setEnabled(False)

        for btn in (
            self.preview_contour_btn,
            self.build_btn,
            self.contour_terminal_btn,
            self.cancel_build_btn,
        ):
            btn.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        contour_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(
            contour_btn_row,
            0,
            self.preview_contour_btn,
            self.build_btn,
            self.contour_terminal_btn,
            self.cancel_build_btn,
        )
        cp_params_lay.addLayout(contour_btn_row)

        self.contour_input_lbl = QLabel("")
        self.contour_input_lbl.setWordWrap(True)
        cp_params_lay.addWidget(self.contour_input_lbl)

        self.contour_output_lbl = QLabel("")
        self.contour_output_lbl.setWordWrap(True)
        cp_params_lay.addWidget(self.contour_output_lbl)

        self.contour_status_lbl = QLabel("")
        self.contour_status_lbl.setWordWrap(True)
        self.contour_status_lbl.setVisible(False)
        cp_params_lay.addWidget(self.contour_status_lbl)

        self.build_progress_bar = QProgressBar()
        self.build_progress_bar.setRange(0, 100)
        self.build_progress_bar.setValue(0)
        self.build_progress_bar.setVisible(False)

        self.contour_files = PipelineFilesWidget([
            ("", [
                ("3_cell/contour_maps.tif", "Contour maps"),
            ]),
        ])
        cp_params_lay.addWidget(self.build_progress_bar)
        cp_params_lay.addWidget(self.contour_files)
        self._update_contour_status_labels()

        cp_params_scroll.setWidget(cp_params_widget)
        contour_lay.addWidget(cp_params_scroll)
        self.contour_section = CollapsibleSection(
            "1. Contour Maps", _contour_inner, expanded=False
        )
        layout.addWidget(self.contour_section)

        _ws_inner = QWidget()
        ws_lay = QVBoxLayout(_ws_inner)
        ws_lay.setContentsMargins(4, 4, 4, 4)
        ws_lay.setSpacing(4)
        ws_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        ws_params_scroll = QScrollArea()
        ws_params_scroll.setWidgetResizable(True)
        ws_params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        ws_params_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ws_params_scroll.setFrameShape(QFrame.NoFrame)
        ws_params_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        ws_params_widget = QWidget()
        ws_params_widget.setMinimumWidth(520)
        ws_params_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        ws_params_lay = QVBoxLayout(ws_params_widget)
        ws_params_lay.setContentsMargins(0, 0, 0, 0)
        ws_params_lay.setSpacing(4)
        ws_params_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        ws_grid = sweep_parameter_grid(spin_width=_CONTOUR_SWEEP_WIDTH)

        def _ws_dspin(lo, hi, val, step=0.1, decimals=1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return s

        def _ws_ispin(lo, hi, val, step=2):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return s

        self.ws_gauss_space_spin  = _ws_dspin(0.0, 10.0, 1.0)
        self.ws_gauss_time_spin   = _ws_dspin(0.0, 10.0, 1.0)
        self.ws_median_space_spin = _ws_ispin(1, 15, 1)
        self.ws_median_time_spin  = _ws_ispin(1, 15, 1)
        ws_grid.addWidget(QLabel("Gaussian σ space:"), 1, 0)
        ws_grid.addWidget(self.ws_gauss_space_spin,   1, 1)
        ws_grid.addWidget(QLabel("Gaussian σ time:"),  2, 0)
        ws_grid.addWidget(self.ws_gauss_time_spin,    2, 1)
        ws_grid.addWidget(QLabel("Median space:"),     3, 0)
        ws_grid.addWidget(self.ws_median_space_spin,  3, 1)
        ws_grid.addWidget(QLabel("Median time:"),      4, 0)
        ws_grid.addWidget(self.ws_median_time_spin,   4, 1)
        ws_grid.setColumnStretch(1, 1)

        self.ws_compact_space_spin = _ws_dspin(0.0, 100.0, 0.0, step=1.0)
        self.ws_compact_space_spin.setToolTip(
            "Spatial compactness (×0.01 internally). Higher = rounder basins in XY. "
            "0 = pure topographic flood."
        )
        self.ws_compact_time_spin  = _ws_dspin(0.0, 100.0, 10.0, step=1.0)
        self.ws_compact_time_spin.setToolTip(
            "Temporal compactness (×0.01 internally). Higher = less expansion into adjacent frames. "
            "Raise this to restrict the watershed to XY-first expansion."
        )
        ws_grid.addWidget(QLabel("Compactness space:"), 5, 0)
        ws_grid.addWidget(self.ws_compact_space_spin,   5, 1)
        ws_grid.addWidget(QLabel("Compactness time:"),  6, 0)
        ws_grid.addWidget(self.ws_compact_time_spin,    6, 1)

        ws_params_lay.addLayout(ws_grid)

        self.ws_run_btn    = QPushButton("Run")
        self.ws_cancel_btn = QPushButton("Cancel")
        self.ws_cancel_btn.setEnabled(False)
        for btn in (self.ws_run_btn, self.ws_cancel_btn):
            btn.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        ws_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(ws_btn_row, 0, self.ws_run_btn, self.ws_cancel_btn)
        ws_params_lay.addLayout(ws_btn_row)

        self.ws_input_lbl = QLabel("")
        self.ws_input_lbl.setWordWrap(True)
        ws_params_lay.addWidget(self.ws_input_lbl)

        self.ws_status_lbl = QLabel("")
        self.ws_status_lbl.setWordWrap(True)
        self.ws_status_lbl.setVisible(False)
        ws_params_lay.addWidget(self.ws_status_lbl)

        self.ws_progress_bar = QProgressBar()
        self.ws_progress_bar.setRange(0, 100)
        self.ws_progress_bar.setValue(0)
        self.ws_progress_bar.setVisible(False)
        ws_params_lay.addWidget(self.ws_progress_bar)

        self.ws_files = PipelineFilesWidget([
            ("", [
                ("3_cell/smoothed_contours.tif", "Smoothed contours"),
                ("3_cell/tracked_labels.tif",    "Cell labels"),
            ]),
        ])
        ws_params_lay.addWidget(self.ws_files)
        self._update_watershed_status_labels()

        ws_params_scroll.setWidget(ws_params_widget)
        ws_lay.addWidget(ws_params_scroll)
        self.watershed_section = CollapsibleSection(
            "2. 3D Temporal Watershed", _ws_inner, expanded=False
        )
        layout.addWidget(self.watershed_section)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.build_btn.clicked.connect(self._on_build_contour_maps)
        self.contour_terminal_btn.clicked.connect(self._on_run_contour_terminal)
        self.cancel_build_btn.clicked.connect(self._on_cancel_build)
        self.ws_run_btn.clicked.connect(self._on_run_watershed)
        self.ws_cancel_btn.clicked.connect(self._on_cancel_watershed)

    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self.input_files.refresh(pos_dir)
        self.contour_files.refresh(pos_dir)
        self.ws_files.refresh(pos_dir)
        self._update_contour_status_labels()
        self._update_watershed_status_labels()

    def get_state(self) -> dict:
        return {
            "cellprob": {
                "min":        self.cp_min_spin.value(),
                "max":        self.cp_max_spin.value(),
                "step":       self.cp_step_spin.value(),
                "gamma_min":  self.cp_gamma_min_spin.value(),
                "gamma_max":  self.cp_gamma_max_spin.value(),
                "gamma_step": self.cp_gamma_step_spin.value(),
                "niter":      self._niter_spin.value(),
            },
            "watershed_3d": {
                "gauss_space":   self.ws_gauss_space_spin.value(),
                "gauss_time":    self.ws_gauss_time_spin.value(),
                "median_space":  self.ws_median_space_spin.value(),
                "median_time":   self.ws_median_time_spin.value(),
                "compact_space": self.ws_compact_space_spin.value(),
                "compact_time":  self.ws_compact_time_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if "cellprob" in state:
            cp = state["cellprob"]
            if "min"        in cp: self.cp_min_spin.setValue(cp["min"])
            if "max"        in cp: self.cp_max_spin.setValue(cp["max"])
            if "step"       in cp: self.cp_step_spin.setValue(cp["step"])
            if "gamma_min"  in cp: self.cp_gamma_min_spin.setValue(cp["gamma_min"])
            if "gamma_max"  in cp: self.cp_gamma_max_spin.setValue(cp["gamma_max"])
            if "gamma_step" in cp: self.cp_gamma_step_spin.setValue(cp["gamma_step"])
            if "niter"      in cp: self._niter_spin.setValue(cp["niter"])
        if "watershed_3d" in state:
            ws = state["watershed_3d"]
            if "gauss_space"   in ws: self.ws_gauss_space_spin.setValue(ws["gauss_space"])
            if "gauss_time"    in ws: self.ws_gauss_time_spin.setValue(ws["gauss_time"])
            if "median_space"  in ws: self.ws_median_space_spin.setValue(ws["median_space"])
            if "median_time"   in ws: self.ws_median_time_spin.setValue(ws["median_time"])
            if "compact_space" in ws: self.ws_compact_space_spin.setValue(ws["compact_space"])
            if "compact_time"  in ws: self.ws_compact_time_spin.setValue(ws["compact_time"])

    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif" if self._pos_dir else None

    def _dp_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_dp_3dt.tif" if self._pos_dir else None

    def _contour_maps_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "contour_maps.tif" if self._pos_dir else None

    def _foreground_masks_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "foreground_masks.tif" if self._pos_dir else None

    def _nucleus_tracked_labels_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    def _smoothed_contours_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "smoothed_contours.tif" if self._pos_dir else None

    def _cell_labels_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "tracked_labels.tif" if self._pos_dir else None

    def _thresholds(self) -> list[float]:
        lo   = self.cp_min_spin.value()
        hi   = self.cp_max_spin.value()
        step = self.cp_step_spin.value()
        n = max(1, round((hi - lo) / step) + 1)
        return list(np.linspace(lo, hi, n))

    def _cp_gammas(self) -> list[float]:
        gmin  = self.cp_gamma_min_spin.value()
        gmax  = self.cp_gamma_max_spin.value()
        gstep = self.cp_gamma_step_spin.value()
        n = max(1, round((gmax - gmin) / gstep) + 1)
        return list(np.linspace(gmin, gmax, n))

    def _current_t(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) >= 1 else 0

    def _set_contour_status(self, msg: str) -> None:
        self.contour_status_lbl.setText(msg)
        self.contour_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _update_contour_status_labels(self) -> None:
        if self._pos_dir is None:
            self.contour_input_lbl.setText("Inputs: no project open.")
            self.contour_output_lbl.setText("Outputs: no project open.")
            return
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        prob_ok   = prob_path is not None and prob_path.exists()
        dp_ok     = dp_path   is not None and dp_path.exists()
        checkmark = '\u2713'
        x_mark = '\u2717'
        self.contour_input_lbl.setText(
            f"Inputs: {checkmark if prob_ok else x_mark} prob  {checkmark if dp_ok else x_mark} dp"
        )
        contour_path = self._contour_maps_path()
        contour_ok   = contour_path is not None and contour_path.exists()
        self.contour_output_lbl.setText(
            f"Outputs: {checkmark if contour_ok else x_mark} contour_maps.tif"
        )

    def _set_build_buttons_running(self, running: bool) -> None:
        self.build_btn.setEnabled(not running)
        self.preview_contour_btn.setEnabled(not running)
        self.contour_terminal_btn.setEnabled(not running)
        self.cancel_build_btn.setEnabled(running)
        self.build_progress_bar.setVisible(running)
        if not running:
            self.build_progress_bar.setValue(0)

    def _on_build_progress(self, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            if total > 0:
                self.build_progress_bar.setRange(0, total)
                self.build_progress_bar.setValue(done)
            self._set_contour_status(msg)
        else:
            self._set_contour_status(str(data))

    def _on_contour_worker_error(self, exc: Exception) -> None:
        if self._build_worker is None:
            return  # worker was already cleared by cancel — don't overwrite "cancelled" status
        self._build_worker = None
        self.build_progress_bar.setVisible(False)
        self._set_build_buttons_running(False)
        self._set_contour_status(f"Error: {exc}")
        logger.exception("Contour worker error", exc_info=exc)

    def _load_prob_dp(self, prob_path: Path, dp_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load and normalise prob (T,Z,Y,X) and dp (T,Z,2,Y,X) stacks."""
        from cellflow.database.hypotheses import normalize_seeded_watershed_dp_stack

        prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
        if prob.ndim == 3:
            prob = prob[np.newaxis]
        dp_raw = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
        dp = normalize_seeded_watershed_dp_stack(dp_raw, prob.shape)
        dp = dp[:, :, :2]
        return prob, dp

    def _on_preview_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return

        t_frame    = self._current_t()
        thresholds = self._thresholds()
        gammas     = self._cp_gammas()

        def _on_done(result):
            self._build_worker = None
            self._set_build_buttons_running(False)
            boundary, cellprob_zavg, dp_mag_zavg, t_idx = result
            full = np.zeros((cellprob_zavg.shape[0],) + boundary.shape, dtype=np.float32)
            full[t_idx] = boundary
            if _CELLPROB_LAYER in self.viewer.layers:
                try:
                    self.viewer.layers[_CELLPROB_LAYER].data = cellprob_zavg
                except Exception:
                    self.viewer.layers.remove(self.viewer.layers[_CELLPROB_LAYER])
                    self.viewer.add_image(
                        cellprob_zavg, name=_CELLPROB_LAYER,
                        colormap="inferno", blending="additive", visible=True,
                    )
            else:
                self.viewer.add_image(
                    cellprob_zavg, name=_CELLPROB_LAYER,
                    colormap="inferno", blending="additive", visible=True,
                )
            if _DP_MAG_LAYER in self.viewer.layers:
                try:
                    self.viewer.layers[_DP_MAG_LAYER].data = dp_mag_zavg
                except Exception:
                    self.viewer.layers.remove(self.viewer.layers[_DP_MAG_LAYER])
                    self.viewer.add_image(
                        dp_mag_zavg, name=_DP_MAG_LAYER,
                        colormap="cyan", blending="additive", visible=True,
                    )
            else:
                self.viewer.add_image(
                    dp_mag_zavg, name=_DP_MAG_LAYER,
                    colormap="cyan", blending="additive", visible=True,
                )
            if _CONTOUR_LAYER in self.viewer.layers:
                try:
                    self.viewer.layers[_CONTOUR_LAYER].data = full
                except Exception:
                    self.viewer.layers.remove(self.viewer.layers[_CONTOUR_LAYER])
                    self.viewer.add_image(
                        full, name=_CONTOUR_LAYER,
                        colormap="magma", blending="additive", visible=True,
                    )
            else:
                self.viewer.add_image(
                    full, name=_CONTOUR_LAYER,
                    colormap="magma", blending="additive", visible=True,
                )
            step = list(self.viewer.dims.current_step)
            if step:
                step[0] = t_idx
                self.viewer.dims.current_step = tuple(step)
            self._set_contour_status(
                f"Preview contour map t={t_idx} — "
                f"{len(thresholds)} cellprob thresholds, {len(gammas)} gamma value(s)"
            )

        @thread_worker(connect={
            "returned": _on_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation import build_mean_z_consensus_boundary

            prob, dp = self._load_prob_dp(prob_path, dp_path)
            n_t  = min(prob.shape[0], dp.shape[0])
            t_idx = min(max(t_frame, 0), n_t - 1)
            boundary, _ = build_mean_z_consensus_boundary(
                prob[t_idx], dp[t_idx], thresholds, gammas, niter=self._niter_spin.value()
            )
            cellprob_zavg = (1.0 / (1.0 + np.exp(-prob))).mean(axis=1).astype(np.float32)
            dp_mag = np.sqrt(dp[:, :, 0]**2 + dp[:, :, 1]**2)
            dp_mag_zavg = dp_mag.mean(axis=1).astype(np.float32)
            return boundary, cellprob_zavg, dp_mag_zavg, t_idx

        self._set_contour_status(f"Previewing contour map for frame t={t_frame}...")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_build_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path    = self._prob_path()
        dp_path      = self._dp_path()
        contour_path = self._contour_maps_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return

        thresholds = self._thresholds()
        gammas     = self._cp_gammas()

        def _on_done(pos_dir: Path) -> None:
            self._build_worker = None
            self._set_build_buttons_running(False)
            self.contour_files.refresh(pos_dir)
            self._update_contour_status_labels()
            self._set_contour_status("Cell contour maps built.")

        @thread_worker(connect={
            "yielded":  self._on_build_progress,
            "returned": _on_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation import build_mean_z_consensus_boundary

            prob, dp = self._load_prob_dp(prob_path, dp_path)
            n_t = min(prob.shape[0], dp.shape[0])
            contour_frames: list[np.ndarray] = []

            for t in range(n_t):
                yield (t + 1, n_t, f"Building cell contour maps: frame {t + 1}/{n_t}...")
                boundary, _ = build_mean_z_consensus_boundary(
                    prob[t], dp[t], thresholds, gammas, niter=self._niter_spin.value()
                )
                contour_frames.append(boundary.astype(np.float32))

            contour_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(contour_path), np.stack(contour_frames), compression="zlib")
            return self._pos_dir

        gamma_desc = (
            f"gamma={gammas[0]:.2f}"
            if len(gammas) == 1
            else f"gamma={gammas[0]:.2f}-{gammas[-1]:.2f} ({len(gammas)} steps)"
        )
        self._set_contour_status(
            f"Building cell contour maps ({len(thresholds)} cellprob thresholds, {gamma_desc})..."
        )
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_cancel_build(self) -> None:
        if self._build_worker is not None:
            worker = self._build_worker
            self._build_worker = None  # clear before quit so errored callback is a no-op
            worker.quit()
        self._set_build_buttons_running(False)
        self._set_contour_status("Build cancelled.")

    def _on_run_contour_terminal(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path    = self._prob_path()
        dp_path      = self._dp_path()
        contour_path = self._contour_maps_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return

        thresholds = self._thresholds()
        gammas     = self._cp_gammas()

        python_code = (
            "import pathlib\n"
            "import numpy as np\n"
            "import tifffile\n"
            "from cellflow.segmentation import build_mean_z_consensus_boundary\n"
            "from cellflow.database.hypotheses import normalize_seeded_watershed_dp_stack\n"
            f"prob_path = pathlib.Path({str(prob_path)!r})\n"
            f"dp_path = pathlib.Path({str(dp_path)!r})\n"
            f"contour_path = pathlib.Path({str(contour_path)!r})\n"
            f"thresholds = {thresholds!r}\n"
            f"gammas = {gammas!r}\n"
            "prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)\n"
            "if prob.ndim == 3:\n"
            "    prob = prob[np.newaxis]\n"
            "dp_raw = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)\n"
            "dp = normalize_seeded_watershed_dp_stack(dp_raw, prob.shape)\n"
            "dp = dp[:, :, :2]\n"
            "n_t = min(prob.shape[0], dp.shape[0])\n"
            "contour_frames = []\n"
            "for t in range(n_t):\n"
            "    print(f'Building cell contour maps: frame {t + 1}/{n_t}...', flush=True)\n"
            "    boundary, _ = build_mean_z_consensus_boundary(\n"
            "        prob[t], dp[t], thresholds, gammas, niter=" + str(self._niter_spin.value()) + "\n"
            "    )\n"
            "    contour_frames.append(boundary.astype(np.float32))\n"
            "contour_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "print('Writing cell contour maps...', flush=True)\n"
            "tifffile.imwrite(str(contour_path), np.stack(contour_frames), compression='zlib')\n"
            "print('Done.')\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="cellflow_cell_contour_", delete=False
        ) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_contour_status("Cell contour build launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_contour_status(
                "Copied cell contour build command to clipboard (terminal launch unavailable)."
            )

    # ------------------------------------------------------------------
    # 3D Temporal Watershed section
    # ------------------------------------------------------------------

    def _update_watershed_status_labels(self) -> None:
        if self._pos_dir is None:
            self.ws_input_lbl.setText("Inputs: no project open.")
            return
        checkmark = "✓"
        x_mark = "✗"
        contour_ok = (p := self._contour_maps_path()) is not None and p.exists()
        fg_ok      = (p := self._foreground_masks_path()) is not None and p.exists()
        nucleus_ok = (p := self._nucleus_tracked_labels_path()) is not None and p.exists()
        self.ws_input_lbl.setText(
            f"Inputs: {checkmark if contour_ok else x_mark} contour_maps  "
            f"{checkmark if fg_ok else x_mark} foreground_masks  "
            f"{checkmark if nucleus_ok else x_mark} nucleus tracked_labels"
        )

    def _set_watershed_status(self, msg: str) -> None:
        self.ws_status_lbl.setText(msg)
        self.ws_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_watershed_buttons_running(self, running: bool) -> None:
        self.ws_run_btn.setEnabled(not running)
        self.ws_cancel_btn.setEnabled(running)
        self.ws_progress_bar.setVisible(running)
        if not running:
            self.ws_progress_bar.setValue(0)

    def _on_watershed_progress(self, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            if total > 0:
                self.ws_progress_bar.setRange(0, total)
                self.ws_progress_bar.setValue(done)
            self._set_watershed_status(msg)
        else:
            self._set_watershed_status(str(data))

    def _on_watershed_worker_error(self, exc: Exception) -> None:
        if self._watershed_worker is None:
            return
        self._watershed_worker = None
        self.ws_progress_bar.setVisible(False)
        self._set_watershed_buttons_running(False)
        self._set_watershed_status(f"Error: {exc}")
        logger.exception("Watershed worker error", exc_info=exc)

    def _on_run_watershed(self) -> None:
        if self._pos_dir is None:
            self._set_watershed_status("No project open.")
            return

        contour_path  = self._contour_maps_path()
        fg_path       = self._foreground_masks_path()
        nucleus_path  = self._nucleus_tracked_labels_path()
        smoothed_path = self._smoothed_contours_path()
        labels_path   = self._cell_labels_path()

        for path, name in [
            (contour_path,  "contour_maps.tif"),
            (fg_path,       "foreground_masks.tif"),
            (nucleus_path,  "tracked_labels.tif (2_nucleus)"),
        ]:
            if path is None or not path.exists():
                self._set_watershed_status(f"Missing: {name}")
                return

        gauss_space   = self.ws_gauss_space_spin.value()
        gauss_time    = self.ws_gauss_time_spin.value()
        median_space  = self.ws_median_space_spin.value()
        median_time   = self.ws_median_time_spin.value()
        compact_space = self.ws_compact_space_spin.value() * 0.01
        compact_time  = self.ws_compact_time_spin.value() * 0.01
        pos_dir       = self._pos_dir

        def _on_done(result):
            self._watershed_worker = None
            self._set_watershed_buttons_running(False)
            smoothed, labels = result
            for layer_name, data, kwargs in [
                (_SMOOTHED_CONTOUR_LAYER, smoothed, {"colormap": "inferno", "blending": "additive"}),
            ]:
                if layer_name in self.viewer.layers:
                    try:
                        self.viewer.layers[layer_name].data = data
                    except Exception:
                        self.viewer.layers.remove(self.viewer.layers[layer_name])
                        self.viewer.add_image(data, name=layer_name, **kwargs)
                else:
                    self.viewer.add_image(data, name=layer_name, **kwargs)
            if _CELL_LABELS_LAYER in self.viewer.layers:
                try:
                    self.viewer.layers[_CELL_LABELS_LAYER].data = labels
                except Exception:
                    self.viewer.layers.remove(self.viewer.layers[_CELL_LABELS_LAYER])
                    self.viewer.add_labels(labels, name=_CELL_LABELS_LAYER)
            else:
                self.viewer.add_labels(labels, name=_CELL_LABELS_LAYER)
            self.ws_files.refresh(pos_dir)
            self._update_watershed_status_labels()
            self._set_watershed_status("3D temporal watershed complete.")

        @thread_worker(connect={
            "yielded":  self._on_watershed_progress,
            "returned": _on_done,
            "errored":  self._on_watershed_worker_error,
        })
        def _worker():
            from cellflow.segmentation import (
                centroid_markers_from_labels,
                compute_3d_temporal_watershed,
            )

            yield (0, 4, "Loading inputs...")
            contours       = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
            foreground     = np.asarray(tifffile.imread(str(fg_path)), dtype=bool)
            nucleus_labels = np.asarray(tifffile.imread(str(nucleus_path)), dtype=np.int32)

            yield (1, 4, "Extracting nucleus seeds...")
            seeds = centroid_markers_from_labels(nucleus_labels)

            yield (2, 4, "Running 3D temporal watershed...")
            smoothed, labels = compute_3d_temporal_watershed(
                contours, foreground, seeds,
                gaussian_sigma_space=gauss_space,
                gaussian_sigma_time=gauss_time,
                median_kernel_space=median_space,
                median_kernel_time=median_time,
                compactness_space=compact_space,
                compactness_time=compact_time,
            )

            yield (3, 4, "Saving outputs...")
            smoothed_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(smoothed_path), smoothed, compression="zlib")
            tifffile.imwrite(str(labels_path), labels, compression="zlib")
            return smoothed, labels

        self._set_watershed_status("Running 3D temporal watershed...")
        self._set_watershed_buttons_running(True)
        self._watershed_worker = _worker()

    def _on_cancel_watershed(self) -> None:
        if self._watershed_worker is not None:
            worker = self._watershed_worker
            self._watershed_worker = None
            worker.quit()
        self._set_watershed_buttons_running(False)
        self._set_watershed_status("Watershed cancelled.")
