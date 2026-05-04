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
    QHBoxLayout,
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
_CONTOUR_SWEEP_WIDTH = 60
_CONTOUR_SWEEP_MIN_WIDTH = int(_CONTOUR_SWEEP_WIDTH * 0.9)


class CellWorkflowWidget(QWidget):
    """Cell segmentation — Contour Maps."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._build_worker = None
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
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.build_btn.clicked.connect(self._on_build_contour_maps)
        self.contour_terminal_btn.clicked.connect(self._on_run_contour_terminal)
        self.cancel_build_btn.clicked.connect(self._on_cancel_build)

    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self.input_files.refresh(pos_dir)
        self.contour_files.refresh(pos_dir)
        self._update_contour_status_labels()

    def get_state(self) -> dict:
        return {
            "cellprob": {
                "min":        self.cp_min_spin.value(),
                "max":        self.cp_max_spin.value(),
                "step":       self.cp_step_spin.value(),
                "gamma_min":  self.cp_gamma_min_spin.value(),
                "gamma_max":  self.cp_gamma_max_spin.value(),
                "gamma_step": self.cp_gamma_step_spin.value(),
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

    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif" if self._pos_dir else None

    def _dp_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_dp_3dt.tif" if self._pos_dir else None

    def _contour_maps_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "contour_maps.tif" if self._pos_dir else None

    def _thresholds(self) -> list[float]:
        lo   = self.cp_min_spin.value()
        hi   = self.cp_max_spin.value()
        step = self.cp_step_spin.value()
        return list(np.arange(lo, hi + step / 2, step))

    def _cp_gammas(self) -> list[float]:
        gmin  = self.cp_gamma_min_spin.value()
        gmax  = self.cp_gamma_max_spin.value()
        gstep = self.cp_gamma_step_spin.value()
        return list(np.arange(gmin, gmax + gstep / 2, gstep))

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
        self.contour_input_lbl.setText(
            f"Inputs: {'\u2713' if prob_ok else '\u2717'} prob  {'\u2713' if dp_ok else '\u2717'} dp"
        )
        contour_path = self._contour_maps_path()
        contour_ok   = contour_path is not None and contour_path.exists()
        self.contour_output_lbl.setText(
            f"Outputs: {'\u2713' if contour_ok else '\u2717'} contour_maps.tif"
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
            boundary, cellprob_zavg, t_idx = result
            full = np.zeros((cellprob_zavg.shape[0],) + boundary.shape, dtype=np.float32)
            full[t_idx] = boundary
            if _CELLPROB_LAYER in self.viewer.layers:
                self.viewer.layers[_CELLPROB_LAYER].data = cellprob_zavg
            else:
                self.viewer.add_image(
                    cellprob_zavg, name=_CELLPROB_LAYER,
                    colormap="inferno", blending="additive", visible=True,
                )
            if _CONTOUR_LAYER in self.viewer.layers:
                self.viewer.layers[_CONTOUR_LAYER].data = full
            else:
                self.viewer.add_image(full, name=_CONTOUR_LAYER, colormap="magma", visible=True)
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
                prob[t_idx], dp[t_idx], thresholds, gammas
            )
            cellprob_zavg = (1.0 / (1.0 + np.exp(-prob.mean(axis=1)))).astype(np.float32)
            return boundary, cellprob_zavg, t_idx

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
                    prob[t], dp[t], thresholds, gammas
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
            self._build_worker.quit()
            self._build_worker = None
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
            "        prob[t], dp[t], thresholds, gammas\n"
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
