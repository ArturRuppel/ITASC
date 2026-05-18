"""Local Cellpose-SAM widget — per-channel rows with preview, run, cancel."""
from __future__ import annotations

import logging
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._widget_helpers import tool_btn as _tool_btn
from cellflow.napari.divergence_maps_widget import DivergenceMapsWidget
from cellflow.napari.ui_style import stage_header_label, status_label
from cellflow.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    make_pipeline_files_header,
)
from cellflow.segmentation import cellpose_runner

logger = logging.getLogger(__name__)


_PIPELINE_FILES = [
    ("Inputs", [
        ("0_input/nucleus_3dt.tif", "Nucleus 3D+t"),
        ("0_input/cell_3dt.tif", "Cell 3D+t"),
    ]),
    ("Outputs", [
        ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
        ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
        ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
    ]),
]


_REFERENCE_LAYER_NAMES = {
    "nucleus": "Reference: Nucleus 3D+t",
    "cell": "Reference: Cell 3D+t",
}


def _make_status() -> QLabel:
    lbl = QLabel("")
    lbl.setWordWrap(True)
    lbl.setVisible(False)
    status_label(lbl)
    return lbl


def _make_progress() -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setVisible(False)
    return bar


class CellposeWidget(QWidget):
    """Local Cellpose-SAM runner — two rows (Nucleus, Cell)."""

    _progress_signal = Signal(int, int, str)

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._running_stage: str | None = None
        self._worker = None
        self._cancel_requested = False

        self._setup_ui()
        self._connect_signals()
        self._progress_signal.connect(self._progress)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # ── Pipeline files ─────────────────────────────────────────────
        self._files_widget = PipelineFilesWidget(_PIPELINE_FILES, viewer=self.viewer)
        self.output_files_tracker = self._files_widget
        self.input_files_tracker = self._files_widget
        self._pipeline_files_section = CollapsibleSection(
            "Pipeline Files", self._files_widget, expanded=False,
        )
        (
            self.pipeline_files_header,
            self.pipeline_files_header_lbl,
            self.pipeline_files_toggle_btn,
        ) = make_pipeline_files_header(
            self._pipeline_files_section, stage_key="cellpose", parent=self,
        )
        root.addWidget(self.pipeline_files_header)
        root.addWidget(self._pipeline_files_section)

        # ── Nucleus row + params ───────────────────────────────────────
        self.nucleus_params_btn = _tool_btn(
            "⚙", "Show parameters for nucleus Cellpose.", checkable=True,
        )
        self.nucleus_preview_btn = _tool_btn("▷", "Preview on current frame.")
        self.nucleus_run_btn = _tool_btn("▶", "Run nucleus Cellpose on all frames.")
        self.nucleus_section = self._build_nucleus_params_section()
        self.nucleus_section.set_header_visible(False)
        self.nucleus_section.collapse()
        self.nucleus_params_btn.toggled.connect(
            lambda checked: self.nucleus_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Nucleus Cellpose"),
            self.nucleus_params_btn,
            self.nucleus_preview_btn,
            self.nucleus_run_btn,
        ))
        root.addWidget(self.nucleus_section)

        # ── Cell row + params ──────────────────────────────────────────
        self.cell_params_btn = _tool_btn(
            "⚙", "Show parameters for cell Cellpose.", checkable=True,
        )
        self.cell_preview_btn = _tool_btn("▷", "Preview on current frame/z-slice.")
        self.cell_run_btn = _tool_btn("▶", "Run cell Cellpose on all frames.")
        self.cell_section = self._build_cell_params_section()
        self.cell_section.set_header_visible(False)
        self.cell_section.collapse()
        self.cell_params_btn.toggled.connect(
            lambda checked: self.cell_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Cell Cellpose"),
            self.cell_params_btn,
            self.cell_preview_btn,
            self.cell_run_btn,
        ))
        root.addWidget(self.cell_section)

        # ── Status + progress (shared) ─────────────────────────────────
        self.status_lbl = _make_status()
        root.addWidget(self.status_lbl)
        self.progress_bar = _make_progress()
        root.addWidget(self.progress_bar)

        # ── Divergence maps from Cellpose prob/dp outputs ─────────────
        self.divergence_maps_widget = DivergenceMapsWidget(self.viewer)
        root.addWidget(self.divergence_maps_widget)

    def _build_nucleus_params_section(self) -> CollapsibleSection:
        body = QWidget(self)
        form = QFormLayout(body)
        form.setContentsMargins(8, 4, 4, 4)
        form.setSpacing(4)
        self.nuc_3d_chk = QCheckBox("3D mode")
        self.nuc_3d_chk.setChecked(True)
        self.nuc_anisotropy_spin = QDoubleSpinBox()
        self.nuc_anisotropy_spin.setRange(0.1, 20.0)
        self.nuc_anisotropy_spin.setSingleStep(0.1)
        self.nuc_anisotropy_spin.setDecimals(2)
        self.nuc_anisotropy_spin.setValue(1.5)
        self.nuc_diameter_spin = QDoubleSpinBox()
        self.nuc_diameter_spin.setRange(0.0, 500.0)
        self.nuc_diameter_spin.setDecimals(1)
        self.nuc_diameter_spin.setValue(25.0)
        self.nuc_min_size_spin = QSpinBox()
        self.nuc_min_size_spin.setRange(0, 100000)
        self.nuc_min_size_spin.setValue(15)
        self.nuc_gamma_spin = QDoubleSpinBox()
        self.nuc_gamma_spin.setRange(0.1, 5.0)
        self.nuc_gamma_spin.setSingleStep(0.1)
        self.nuc_gamma_spin.setDecimals(2)
        self.nuc_gamma_spin.setValue(1.0)
        form.addRow(self.nuc_3d_chk)
        form.addRow("Anisotropy", self.nuc_anisotropy_spin)
        form.addRow("Diameter", self.nuc_diameter_spin)
        form.addRow("Min size", self.nuc_min_size_spin)
        form.addRow("Gamma", self.nuc_gamma_spin)
        return CollapsibleSection("Nucleus parameters", body, expanded=False)

    def _build_cell_params_section(self) -> CollapsibleSection:
        body = QWidget(self)
        form = QFormLayout(body)
        form.setContentsMargins(8, 4, 4, 4)
        form.setSpacing(4)
        self.cell_diameter_spin = QDoubleSpinBox()
        self.cell_diameter_spin.setRange(0.0, 500.0)
        self.cell_diameter_spin.setDecimals(1)
        self.cell_diameter_spin.setValue(0.0)
        self.cell_min_size_spin = QSpinBox()
        self.cell_min_size_spin.setRange(0, 100000)
        self.cell_min_size_spin.setValue(0)
        self.cell_gamma_spin = QDoubleSpinBox()
        self.cell_gamma_spin.setRange(0.1, 5.0)
        self.cell_gamma_spin.setSingleStep(0.1)
        self.cell_gamma_spin.setDecimals(2)
        self.cell_gamma_spin.setValue(1.0)
        form.addRow("Diameter", self.cell_diameter_spin)
        form.addRow("Min size", self.cell_min_size_spin)
        form.addRow("Gamma", self.cell_gamma_spin)
        return CollapsibleSection("Cell parameters", body, expanded=False)

    @staticmethod
    def _stage_label(text: str) -> QLabel:
        return stage_header_label(QLabel(text), "cellpose")

    @staticmethod
    def _stage_row(label: QLabel, *trailing: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(label)
        row.addStretch(1)
        for w in trailing:
            row.addWidget(w)
        return row

    # ------------------------------------------------------------------
    # Signals (run/cancel handlers are filled in in later tasks)
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.nucleus_run_btn.clicked.connect(self._on_nucleus_run_clicked)
        self.cell_run_btn.clicked.connect(self._on_cell_run_clicked)
        self.nucleus_preview_btn.clicked.connect(self._on_nucleus_preview)
        self.cell_preview_btn.clicked.connect(self._on_cell_preview)

    def _on_nucleus_run_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
            return
        self._run_channel("nucleus")

    def _on_cell_run_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
            return
        self._run_channel("cell")

    def _on_nucleus_preview(self) -> None:
        self._preview_channel("nucleus")

    def _on_cell_preview(self) -> None:
        self._preview_channel("cell")

    def _on_cancel(self) -> None:
        self._cancel_requested = True
        worker = self._worker
        if worker is not None and hasattr(worker, "quit"):
            worker.quit()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _input_path(self, channel: str) -> Path | None:
        if self._pos_dir is None:
            return None
        name = "nucleus_3dt.tif" if channel == "nucleus" else "cell_3dt.tif"
        return self._pos_dir / "0_input" / name

    def _output_dir(self) -> Path | None:
        return None if self._pos_dir is None else self._pos_dir / "1_cellpose"

    # ------------------------------------------------------------------
    # Run flow
    # ------------------------------------------------------------------
    def _build_nucleus_params(self) -> "cellpose_runner.NucleusParams":
        return cellpose_runner.NucleusParams(
            do_3d=self.nuc_3d_chk.isChecked(),
            anisotropy=float(self.nuc_anisotropy_spin.value()),
            diameter=float(self.nuc_diameter_spin.value()),
            min_size=int(self.nuc_min_size_spin.value()),
            gamma=float(self.nuc_gamma_spin.value()),
        )

    def _build_cell_params(self) -> "cellpose_runner.CellParams":
        return cellpose_runner.CellParams(
            diameter=float(self.cell_diameter_spin.value()),
            min_size=int(self.cell_min_size_spin.value()),
            gamma=float(self.cell_gamma_spin.value()),
        )

    def _run_channel(self, channel: str) -> None:
        if self._pos_dir is None:
            self._status("No project open.")
            return
        in_path = self._input_path(channel)
        if in_path is None or not in_path.exists():
            self._status(f"Missing: {in_path.name if in_path else '(no path)'}")
            return
        out_dir = self._output_dir()
        params = (
            self._build_nucleus_params() if channel == "nucleus"
            else self._build_cell_params()
        )
        pos_dir = self._pos_dir
        self._cancel_requested = False

        def _done(result):
            self._worker = None
            self._set_running_stage(None)
            self._clear_progress()
            self._files_widget.refresh(pos_dir)
            self.divergence_maps_widget.refresh(pos_dir)
            label = "Nucleus" if channel == "nucleus" else "Cell"
            self._status(f"{label} Cellpose complete — wrote {channel}_*_3dt.tif")

        def _error(exc):
            self._worker = None
            self._set_running_stage(None)
            self._clear_progress()
            if isinstance(exc, cellpose_runner.CancelledError):
                self._status("Cancelled.")
            else:
                self._status(f"Error: {exc}")
                logger.exception("Cellpose run error", exc_info=exc)

        progress_signal = self._progress_signal

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": _error,
        })
        def _worker():
            yield (0, 1, "Loading input...")
            stack = np.asarray(tifffile.imread(str(in_path)))
            if stack.ndim != 4:
                raise ValueError(
                    f"expected 4D (T,Z,Y,X) input, got shape {stack.shape}"
                )

            def _cb_progress(done, total, msg):
                progress_signal.emit(int(done), int(total), str(msg))

            def _cb_cancel():
                return self._cancel_requested

            if channel == "nucleus":
                prob, dp = cellpose_runner.run_nucleus_stack(
                    stack, params,
                    progress_cb=_cb_progress, cancel_cb=_cb_cancel,
                )
            else:
                prob, dp = cellpose_runner.run_cell_stack(
                    stack, params,
                    progress_cb=_cb_progress, cancel_cb=_cb_cancel,
                )
            yield (1, 1, "Writing outputs...")
            cellpose_runner.write_outputs(prob, dp, out_dir, channel)
            return None

        self._set_running_stage(channel)
        self._status(
            f"Loading Cellpose-SAM model on {cellpose_runner.device_label()} "
            f"(~10s on first run)..." if not cellpose_runner.is_model_loaded()
            else f"Running {channel} Cellpose..."
        )
        self._worker = _worker()

    # ------------------------------------------------------------------
    # Preview flow
    # ------------------------------------------------------------------
    def _current_tz(self) -> tuple[int, int]:
        step = getattr(getattr(self.viewer, "dims", None), "current_step", (0, 0))
        t = int(step[0]) if len(step) >= 1 else 0
        z = int(step[1]) if len(step) >= 2 else 0
        return t, z

    @staticmethod
    def _flow_magnitude(dp: np.ndarray) -> np.ndarray:
        # dp has shape (C, ...) — sum-of-squares over the channel axis.
        return np.sqrt(np.sum(np.asarray(dp, dtype=np.float32) ** 2, axis=0))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

    @staticmethod
    def _flow_contrast_limits(flow: np.ndarray) -> tuple[float, float]:
        # Derive limits from the populated frame only — flow_full is mostly
        # zeros, so napari's auto-contrast undersamples and clips the peaks.
        hi = float(np.asarray(flow, dtype=np.float32).max())
        return 0.0, max(hi, 1e-6)

    def _preview_channel(self, channel: str) -> None:
        if self._running_stage is not None:
            self._status("Cellpose task already running.")
            return
        if self._pos_dir is None:
            self._status("No project open.")
            return
        in_path = self._input_path(channel)
        if in_path is None or not in_path.exists():
            self._status(f"Missing: {in_path.name if in_path else '(no path)'}")
            return

        params = (
            self._build_nucleus_params() if channel == "nucleus"
            else self._build_cell_params()
        )
        self._cancel_requested = False
        self._set_running_stage(channel)
        self._progress(0, 0, f"Loading {channel} reference stack for preview...")
        try:
            stack = np.asarray(tifffile.imread(str(in_path)))
            if stack.ndim != 4:
                raise ValueError(
                    f"expected 4D input (T,Z,Y,X), got {stack.shape}"
                )
            self._show_reference_stack(channel, stack)
            t, z = self._current_tz()
        except Exception as exc:
            self._set_running_stage(None)
            self._clear_progress()
            self._status(f"Error: {exc}")
            logger.exception("Cellpose preview load error", exc_info=exc)
            return

        def _done(result):
            self._worker = None
            self._set_running_stage(None)
            self._clear_progress()
            status_msg, layers = result
            for name, data, kwargs in layers:
                self._show_layer(name, data, kwargs, self.viewer.add_image)
            self._status(status_msg)

        def _error(exc):
            self._worker = None
            self._set_running_stage(None)
            self._clear_progress()
            self._status(f"Error: {exc}")
            logger.exception("Cellpose preview error", exc_info=exc)

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": _error,
        })
        def _worker():
            T, Z = stack.shape[:2]
            t_clamped = min(max(t, 0), T - 1)
            z_clamped = min(max(z, 0), Z - 1)

            if channel == "nucleus":
                if params.do_3d:
                    yield (
                        0, 0,
                        f"Previewing nucleus 3D t={t_clamped} "
                        f"on {cellpose_runner.device_label()} "
                        f"(Z={Z}, anisotropy={params.anisotropy})...",
                    )
                    prob_logits, dp = cellpose_runner.run_nucleus_frame(
                        stack[t_clamped], z=None, params=params,
                    )
                    prob = self._sigmoid(prob_logits)
                    flow = self._flow_magnitude(dp)  # (Z, Y, X)
                    prob_full = np.zeros((T, Z, *prob.shape[-2:]), dtype=np.float32)
                    flow_full = np.zeros_like(prob_full)
                    prob_full[t_clamped] = prob
                    flow_full[t_clamped] = flow
                    flow_clim = self._flow_contrast_limits(flow)
                    status_msg = (
                        f"Preview: nucleus 3D t={t_clamped} "
                        f"(Z={Z}, anisotropy={params.anisotropy})"
                    )
                else:
                    yield (
                        0, 0,
                        f"Previewing nucleus 2D t={t_clamped} z={z_clamped} "
                        f"on {cellpose_runner.device_label()}...",
                    )
                    prob_logits, dp = cellpose_runner.run_nucleus_frame(
                        stack[t_clamped], z=z_clamped, params=params,
                    )
                    prob = self._sigmoid(prob_logits)
                    flow = self._flow_magnitude(dp)  # (Y, X)
                    prob_full = np.zeros((T, Z, *prob.shape), dtype=np.float32)
                    flow_full = np.zeros_like(prob_full)
                    prob_full[t_clamped, z_clamped] = prob
                    flow_full[t_clamped, z_clamped] = flow
                    flow_clim = self._flow_contrast_limits(flow)
                    status_msg = (
                        f"Preview: nucleus 2D t={t_clamped} z={z_clamped} "
                        f"(diameter={params.diameter})"
                    )
                return status_msg, [
                    (
                        "Preview: Nucleus prob",
                        prob_full,
                        {
                            "colormap": "viridis",
                            "blending": "additive",
                            "contrast_limits": (0.0, 1.0),
                        },
                    ),
                    (
                        "Preview: Nucleus flow",
                        flow_full,
                        {
                            "colormap": "inferno",
                            "blending": "additive",
                            "contrast_limits": flow_clim,
                        },
                    ),
                ]

            yield (
                0, 0,
                f"Previewing cell 2D t={t_clamped} z={z_clamped} "
                f"on {cellpose_runner.device_label()}...",
            )
            prob_logits, dp = cellpose_runner.run_cell_frame(
                stack[t_clamped], z=z_clamped, params=params,
            )
            prob = self._sigmoid(prob_logits)
            flow = self._flow_magnitude(dp)
            prob_full = np.zeros((T, Z, *prob.shape), dtype=np.float32)
            flow_full = np.zeros_like(prob_full)
            prob_full[t_clamped, z_clamped] = prob
            flow_full[t_clamped, z_clamped] = flow
            flow_clim = self._flow_contrast_limits(flow)
            return (
                f"Preview: cell t={t_clamped} z={z_clamped} "
                f"(diameter={params.diameter})",
                [
                    (
                        "Preview: Cell prob",
                        prob_full,
                        {
                            "colormap": "viridis",
                            "blending": "additive",
                            "contrast_limits": (0.0, 1.0),
                        },
                    ),
                    (
                        "Preview: Cell flow",
                        flow_full,
                        {
                            "colormap": "inferno",
                            "blending": "additive",
                            "contrast_limits": flow_clim,
                        },
                    ),
                ],
            )

        self._worker = _worker()

    def _show_reference_stack(self, channel: str, stack: np.ndarray) -> None:
        name = _REFERENCE_LAYER_NAMES[channel]
        self._show_layer(
            name, stack,
            {"colormap": "gray", "blending": "additive"},
            self.viewer.add_image,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._files_widget.refresh(pos_dir)
        self.divergence_maps_widget.refresh(pos_dir)

    def get_state(self) -> dict:
        return {
            "nucleus": {
                "do_3d": self.nuc_3d_chk.isChecked(),
                "anisotropy": self.nuc_anisotropy_spin.value(),
                "diameter": self.nuc_diameter_spin.value(),
                "min_size": self.nuc_min_size_spin.value(),
                "gamma": self.nuc_gamma_spin.value(),
            },
            "cell": {
                "diameter": self.cell_diameter_spin.value(),
                "min_size": self.cell_min_size_spin.value(),
                "gamma": self.cell_gamma_spin.value(),
            },
            "divergence_maps": self.divergence_maps_widget.get_state(),
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        nuc = state.get("nucleus", {})
        if isinstance(nuc, dict):
            if "do_3d" in nuc:
                self.nuc_3d_chk.setChecked(bool(nuc["do_3d"]))
            if "anisotropy" in nuc:
                self.nuc_anisotropy_spin.setValue(float(nuc["anisotropy"]))
            if "diameter" in nuc:
                self.nuc_diameter_spin.setValue(float(nuc["diameter"]))
            if "min_size" in nuc:
                self.nuc_min_size_spin.setValue(int(nuc["min_size"]))
            if "gamma" in nuc:
                self.nuc_gamma_spin.setValue(float(nuc["gamma"]))
        cel = state.get("cell", {})
        if isinstance(cel, dict):
            if "diameter" in cel:
                self.cell_diameter_spin.setValue(float(cel["diameter"]))
            if "min_size" in cel:
                self.cell_min_size_spin.setValue(int(cel["min_size"]))
            if "gamma" in cel:
                self.cell_gamma_spin.setValue(float(cel["gamma"]))
        if "divergence_maps" in state:
            self.divergence_maps_widget.set_state(state["divergence_maps"])

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _status(self, msg: str) -> None:
        self.status_lbl.setText(msg)
        self.status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _progress(self, done: int, total: int, msg: str) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self._status(msg)

    def _on_progress(self, data) -> None:
        if isinstance(data, tuple):
            self._progress(*data)
        else:
            self._status(str(data))

    def _clear_progress(self) -> None:
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def _set_running_stage(self, stage_key: str | None) -> None:
        """``None`` means idle; ``'nucleus'`` or ``'cell'`` claims the row."""
        self._running_stage = stage_key
        rows = {
            "nucleus": (
                self.nucleus_params_btn,
                self.nucleus_preview_btn,
                self.nucleus_run_btn,
                "Run nucleus Cellpose on all frames.",
            ),
            "cell": (
                self.cell_params_btn,
                self.cell_preview_btn,
                self.cell_run_btn,
                "Run cell Cellpose on all frames.",
            ),
        }
        if stage_key is None:
            for params_btn, preview_btn, run_btn, tooltip in rows.values():
                params_btn.setEnabled(True)
                preview_btn.setEnabled(True)
                run_btn.setEnabled(True)
                run_btn.setText("▶")
                run_btn.setToolTip(tooltip)
            self._cancel_requested = False
            return
        for key, (params_btn, preview_btn, run_btn, _tooltip) in rows.items():
            if key == stage_key:
                params_btn.setEnabled(True)
                preview_btn.setEnabled(False)
                run_btn.setEnabled(True)
                run_btn.setText("✕")
                run_btn.setToolTip("Cancel.")
            else:
                params_btn.setEnabled(False)
                preview_btn.setEnabled(False)
                run_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Layer helper (mirrors CellWorkflowWidget._show_layer)
    # ------------------------------------------------------------------
    def _show_layer(self, name, data, kwargs, adder):
        if name in self.viewer.layers:
            try:
                layer = self.viewer.layers[name]
                layer.data = data
                clim = kwargs.get("contrast_limits")
                if clim is not None:
                    layer.contrast_limits = clim
                return
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[name])
                adder(data, name=name, **kwargs)
        else:
            adder(data, name=name, **kwargs)
