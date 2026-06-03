"""Per-channel widget that builds nucleus/cell foreground & contour maps
directly from Cellpose ``prob_3dt`` and ``dp_3dt`` outputs.

Mirrors :class:`CellposeWidget` layout (one row per channel with
⚙ params / ▶ run-cancel and a shared status + progress bar).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Literal

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import QTimer, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._paths import NucleusArtifactPaths
from cellflow.napari._widget_helpers import (
    dslider as _dslider,
    islider as _islider,
    tool_btn as _tool_btn,
)
from cellflow.napari.ui_gate import ControlClass, UiGate
from cellflow.napari.ui_style import (
    add_section_pair_row,
    section_grid,
    stage_header_action_button,
    stage_header_label,
    status_label,
)
from cellflow.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    make_pipeline_files_header,
)
from cellflow.segmentation import CancelledError
from cellflow.segmentation.divergence_maps import (
    build_divergence_maps,
    contour_from_dp,
    foreground_from_prob,
)

logger = logging.getLogger(__name__)

# Per-channel live-preview layer names: (foreground, contours).
_PREVIEW_LAYER_NAMES = {
    "nucleus": (
        "Divergence preview: Nucleus foreground",
        "Divergence preview: Nucleus contours",
    ),
    "cell": (
        "Divergence preview: Cell foreground",
        "Divergence preview: Cell contours",
    ),
}


_PIPELINE_FILES = [
    ("Inputs", [
        ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
        ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
        ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
    ]),
    ("Outputs", [
        ("1_cellpose/nucleus_contours.tif", "Nucleus contours"),
        ("1_cellpose/nucleus_foreground.tif", "Nucleus foreground"),
        ("1_cellpose/cell_contours.tif", "Cell contours"),
        ("1_cellpose/cell_foreground.tif", "Cell foreground"),
    ]),
]


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


class DivergenceMapsWidget(QWidget):
    """Build per-channel foreground & contour maps from Cellpose prob/dp."""

    _progress_signal = Signal(int, int, str)

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: QWidget | None = None,
        *,
        show_pipeline_files: bool = True,
        gate: UiGate | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        #: App-wide UI gate; a private one is created for standalone use.
        self.gate = gate if gate is not None else UiGate(self)
        self._pos_dir: Path | None = None
        self._running_stage: str | None = None
        self._worker = None
        self._cancel_event: threading.Event | None = None
        self._show_pipeline_files = bool(show_pipeline_files)

        # Live-preview state. Previews are mutually-exclusive viewer owners, so
        # at most one channel is active at a time.
        self._active_preview_channel: str | None = None
        self._preview_worker = None
        self._preview_pending = False
        # Preview layers whose contrast has not yet been seeded from real data.
        self._preview_needs_autocontrast: set[str] = set()
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._refresh_preview)

        self._setup_ui()
        self._connect_signals()
        self._register_gate_controls()
        self._progress_signal.connect(self._progress)

    # ── UI ───────────────────────────────────────────────────────────
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        if self._show_pipeline_files:
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
        else:
            self._files_widget = None

        # Nucleus row
        self.nucleus_params_btn = _tool_btn(
            "⚙", "Show parameters for nucleus divergence maps.", checkable=True,
        )
        self.nucleus_preview_btn = _tool_btn(
            "◉", "Live preview nucleus divergence maps on the current frame.",
            checkable=True,
        )
        self.nucleus_run_btn = _tool_btn("▶", "Build nucleus divergence maps.")
        for button in (
            self.nucleus_params_btn, self.nucleus_preview_btn, self.nucleus_run_btn,
        ):
            stage_header_action_button(button, "cellpose")
        self.nucleus_section = self._build_channel_params_section("nucleus")
        self.nucleus_section.set_header_visible(False)
        self.nucleus_section.collapse()
        self.nucleus_params_btn.toggled.connect(
            lambda checked: self.nucleus_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Nucleus divergence maps"),
            self.nucleus_params_btn,
            self.nucleus_preview_btn,
            self.nucleus_run_btn,
        ))
        root.addWidget(self.nucleus_section)

        # Cell row
        self.cell_params_btn = _tool_btn(
            "⚙", "Show parameters for cell divergence maps.", checkable=True,
        )
        self.cell_preview_btn = _tool_btn(
            "◉", "Live preview cell divergence maps on the current frame.",
            checkable=True,
        )
        self.cell_run_btn = _tool_btn("▶", "Build cell divergence maps.")
        for button in (
            self.cell_params_btn, self.cell_preview_btn, self.cell_run_btn,
        ):
            stage_header_action_button(button, "cellpose")
        self.cell_section = self._build_channel_params_section("cell")
        self.cell_section.set_header_visible(False)
        self.cell_section.collapse()
        self.cell_params_btn.toggled.connect(
            lambda checked: self.cell_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Cell divergence maps"),
            self.cell_params_btn,
            self.cell_preview_btn,
            self.cell_run_btn,
        ))
        root.addWidget(self.cell_section)

        self.status_lbl = _make_status()
        root.addWidget(self.status_lbl)
        self.progress_bar = _make_progress()
        root.addWidget(self.progress_bar)

    def _build_channel_params_section(
        self, channel: Literal["nucleus", "cell"],
    ) -> CollapsibleSection:
        body = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        body.setLayout(grid)

        fg_reduction = QComboBox()
        fg_reduction.addItems(["mean", "max"])
        fg_reduction.setCurrentText("mean")
        contour_reduction = QComboBox()
        contour_reduction.addItems(["mean", "max"])
        contour_reduction.setCurrentText("mean")
        fg_smoothing_spin = _dslider(0.0, 20.0, 0.0, 0.1, 2)
        fg_median_spin = _islider(0, 20, 0)
        smoothing_spin = _dslider(0.0, 20.0, 1.0, 0.1, 2)
        median_spin = _islider(0, 20, 0)
        row = 0
        add_section_pair_row(
            grid, row,
            "Foreground z-reduction:", fg_reduction,
            "Contour z-reduction:", contour_reduction,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "Foreground smoothing sigma:", fg_smoothing_spin,
            "Foreground median radius:", fg_median_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "Contour smoothing sigma:", smoothing_spin,
            "Contour median radius:", median_spin,
        )

        prefix = "nuc" if channel == "nucleus" else "cell"
        setattr(self, f"{prefix}_fg_reduction", fg_reduction)
        setattr(self, f"{prefix}_contour_reduction", contour_reduction)
        setattr(self, f"{prefix}_fg_smoothing_spin", fg_smoothing_spin)
        setattr(self, f"{prefix}_fg_median_spin", fg_median_spin)
        setattr(self, f"{prefix}_smoothing_spin", smoothing_spin)
        setattr(self, f"{prefix}_median_spin", median_spin)
        return CollapsibleSection(
            f"{channel.title()} parameters", body, expanded=False,
        )

    @staticmethod
    def _stage_label(text: str) -> QLabel:
        return stage_header_label(QLabel(text), "cellpose")

    @staticmethod
    def _stage_row(label: QLabel, *trailing: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(label)
        for w in trailing:
            row.addWidget(w)
        row.addStretch(1)
        return row

    # ── Signals ──────────────────────────────────────────────────────
    def _connect_signals(self) -> None:
        self.nucleus_run_btn.clicked.connect(lambda: self._on_run("nucleus"))
        self.cell_run_btn.clicked.connect(lambda: self._on_run("cell"))
        self.nucleus_preview_btn.toggled.connect(
            lambda checked: self._on_preview_toggled("nucleus", checked)
        )
        self.cell_preview_btn.toggled.connect(
            lambda checked: self._on_preview_toggled("cell", checked)
        )
        for prefix in ("nuc", "cell"):
            getattr(self, f"{prefix}_fg_reduction").currentTextChanged.connect(
                self._on_param_changed
            )
            getattr(self, f"{prefix}_contour_reduction").currentTextChanged.connect(
                self._on_param_changed
            )
            for suffix in (
                "fg_smoothing_spin", "fg_median_spin",
                "smoothing_spin", "median_spin",
            ):
                getattr(self, f"{prefix}_{suffix}").valueChanged.connect(
                    self._on_param_changed
                )
        if hasattr(self.viewer, "dims") and hasattr(self.viewer.dims, "events"):
            try:
                self.viewer.dims.events.current_step.connect(self._on_time_changed)
            except Exception:
                pass

    def _register_gate_controls(self) -> None:
        """Register the two channel rows with the app-wide UI gate.

        Each channel's ◉ live preview is a mutually-exclusive viewer owner.
        Build/run writes data downstream owners view, so it is blocked while any
        owner is active. ⚙ params just toggle a panel and stay available.
        """
        g = self.gate
        has_pos = lambda: self._pos_dir is not None  # noqa: E731
        idle = lambda: self._running_stage is None  # noqa: E731
        for channel, params_btn, preview_btn, run_btn in (
            ("nucleus", self.nucleus_params_btn, self.nucleus_preview_btn, self.nucleus_run_btn),
            ("cell", self.cell_params_btn, self.cell_preview_btn, self.cell_run_btn),
        ):
            token = f"div_preview:{channel}"
            g.register_owner(
                token,
                f"{channel} divergence preview",
                exit_fn=lambda b=preview_btn: b.setChecked(False),
            )
            g.register(params_btn, ControlClass.HARMLESS)
            g.register(
                preview_btn,
                ControlClass.VIEWER_OWNER,
                owner_token=token,
                when=lambda: has_pos() and idle(),
            )
            g.register(
                run_btn,
                ControlClass.RUN_VIEWER,
                when=lambda c=channel: has_pos() and self._running_stage in (None, c),
            )
        g.recompute()

    def _on_run(self, channel: Literal["nucleus", "cell"]) -> None:  # noqa: D401
        if self._running_stage is not None:
            self._on_cancel()
            return
        self._start_worker(channel)

    def _start_worker(self, channel: Literal["nucleus", "cell"]) -> None:
        prob_path, dp_path, contours_out, fg_out = self._channel_paths(channel)
        if prob_path is None:
            self._set_status("No project open.")
            return
        for p in (prob_path, dp_path):
            if p is None or not p.exists():
                self._set_status(f"Missing: {p}")
                return
        params = self._channel_state("nuc" if channel == "nucleus" else "cell")
        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        self._running_stage = channel
        self._set_button_running(channel)

        def _done(report) -> None:
            self._worker = None
            self._cancel_event = None
            self._running_stage = None
            self._set_button_idle()
            self._progress_bar_hide()
            self._set_status(
                f"{channel.title()} divergence maps built ({report.frames} frames)."
            )
            if self._files_widget is not None:
                self._files_widget.refresh(self._pos_dir)

        @thread_worker(
            connect={
                "yielded": self._on_yield,
                "returned": _done,
                "errored": self._on_errored,
            },
        )
        def _worker():
            progress_signal = self._progress_signal

            def _progress_cb(done: int, total: int, msg: str) -> None:
                progress_signal.emit(
                    int(done),
                    int(total),
                    self._channel_progress_message(channel, str(msg)),
                )

            yield (0, 1, f"Starting {channel} divergence maps...")
            return build_divergence_maps(
                prob_path,
                dp_path,
                contours_out,
                fg_out,
                foreground_z_reduction=params["foreground_z_reduction"],
                contour_z_reduction=params["contour_z_reduction"],
                smoothing_sigma=params["smoothing_sigma"],
                median_radius=params["median_radius"],
                foreground_smoothing_sigma=params["foreground_smoothing_sigma"],
                foreground_median_radius=params["foreground_median_radius"],
                progress_cb=_progress_cb,
                cancel=cancel_event.is_set,
            )

        self._worker = _worker()

    def _run_blocking(self, channel: Literal["nucleus", "cell"]) -> None:
        """Synchronous test helper: runs build_divergence_maps in this thread."""
        prob_path, dp_path, contours_out, fg_out = self._channel_paths(channel)
        if prob_path is None or dp_path is None:
            raise RuntimeError("No project open.")
        params = self._channel_state("nuc" if channel == "nucleus" else "cell")
        contours_out.parent.mkdir(parents=True, exist_ok=True)
        build_divergence_maps(
            prob_path,
            dp_path,
            contours_out,
            fg_out,
            foreground_z_reduction=params["foreground_z_reduction"],
            contour_z_reduction=params["contour_z_reduction"],
            smoothing_sigma=params["smoothing_sigma"],
            median_radius=params["median_radius"],
            foreground_smoothing_sigma=params["foreground_smoothing_sigma"],
            foreground_median_radius=params["foreground_median_radius"],
        )

    def _channel_paths(self, channel: Literal["nucleus", "cell"]):
        paths = self._paths()
        if paths is None:
            return None, None, None, None
        if channel == "nucleus":
            return paths.prob, paths.dp, paths.nucleus_contours, paths.nucleus_foreground
        return paths.cell_prob, paths.cell_dp, paths.cell_contours, paths.cell_foreground

    def _set_status(self, msg: str) -> None:
        self.status_lbl.setText(msg)
        self.status_lbl.setVisible(bool(msg))

    def _progress_bar_hide(self) -> None:
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def _set_button_running(self, channel: Literal["nucleus", "cell"]) -> None:
        run_btn = self.nucleus_run_btn if channel == "nucleus" else self.cell_run_btn
        run_btn.setText("✕")
        run_btn.setToolTip("Cancel.")
        # Enablement (other row, previews) is derived by the gate from
        # ``_running_stage``.
        self.gate.recompute()

    def _set_button_idle(self) -> None:
        for btn, tip in (
            (self.nucleus_run_btn, "Build nucleus divergence maps."),
            (self.cell_run_btn, "Build cell divergence maps."),
        ):
            btn.setText("▶")
            btn.setToolTip(tip)
        self.gate.recompute()

    def _on_yield(self, payload) -> None:
        if not isinstance(payload, tuple):
            self._set_status(str(payload))
            return
        done, total, msg = payload
        self._progress(done, total, msg)

    def _on_errored(self, exc: Exception) -> None:
        self._worker = None
        self._cancel_event = None
        self._running_stage = None
        self._set_button_idle()
        self._progress_bar_hide()
        if isinstance(exc, CancelledError):
            self._set_status("Cancelled.")
            return
        logger.exception("Divergence-maps worker error", exc_info=exc)
        self._set_status(f"Error: {exc}")

    def _on_cancel(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._worker is not None and hasattr(self._worker, "quit"):
            self._worker.quit()

    def _progress(self, done: int, total: int, msg: str) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(done)
        self.status_lbl.setText(msg)
        self.status_lbl.setVisible(bool(msg))

    @staticmethod
    def _channel_progress_message(channel: Literal["nucleus", "cell"], msg: str) -> str:
        prefix = "Divergence maps: "
        if msg.startswith(prefix):
            return f"{channel.title()} divergence maps: {msg[len(prefix):]}"
        return msg

    # ── Live preview (single frame) ──────────────────────────────────
    def _on_preview_toggled(self, channel: Literal["nucleus", "cell"], checked: bool) -> None:
        token = f"div_preview:{channel}"
        if checked:
            self._active_preview_channel = channel
            self.gate.claim_viewer(token)
            self._refresh_preview()
        else:
            if self._active_preview_channel == channel:
                self._active_preview_channel = None
            self._preview_pending = False
            self.gate.release_viewer(token)
            self._teardown_preview_layers(channel)
            self._set_status("")

    def _on_param_changed(self, *_args) -> None:
        if self._active_preview_channel is not None:
            self._preview_timer.start()

    def _on_time_changed(self, *_args) -> None:
        if self._active_preview_channel is not None:
            self._preview_timer.start()

    def _refresh_preview(self):
        """Recompute the active channel's current-frame preview off-thread.

        While a pass is in flight, further edits arm ``_preview_pending`` so one
        fresh pass (latest params/frame) fires when the current one returns.
        """
        channel = self._active_preview_channel
        if channel is None:
            return None
        prob_path, dp_path, _, _ = self._channel_paths(channel)
        if (
            prob_path is None or not prob_path.exists()
            or dp_path is None or not dp_path.exists()
        ):
            self._set_status("Cellpose prob/dp not found — run Cellpose first.")
            return None

        shape = self._channel_map_shape(prob_path)
        if shape is None:
            self._set_status("Could not read prob map.")
            return None
        # Create full-length (T, Y, X) layers up front so napari shows a time
        # slider even before any frame is computed; each visited frame fills its
        # own slice on demand.
        self._ensure_preview_layers(channel, shape)
        if self._preview_worker is not None:
            self._preview_pending = True
            return self._preview_worker

        t = max(0, min(self._current_t(), shape[0] - 1))
        params = self._channel_state("nuc" if channel == "nucleus" else "cell")
        self._set_status(f"Computing {channel} divergence preview for frame {t}…")

        @thread_worker(connect={
            "returned": self._preview_done,
            "errored": self._preview_error,
        })
        def _worker():
            fg, contour = self._compute_channel_frame(prob_path, dp_path, t, params)
            return channel, t, fg, contour

        self._preview_worker = _worker()
        return self._preview_worker

    def _preview_done(self, payload) -> None:
        self._preview_worker = None
        channel, t, fg, contour = payload
        if self._active_preview_channel == channel:
            fg_name, ct_name = _PREVIEW_LAYER_NAMES[channel]
            self._fill_image_layer(fg_name, t, fg)
            self._fill_image_layer(ct_name, t, contour)
            self._set_status(f"{channel.title()} divergence preview — frame {t}.")
        if self._preview_pending and self._active_preview_channel is not None:
            self._preview_pending = False
            self._refresh_preview()

    def _preview_error(self, exc: Exception) -> None:
        self._preview_worker = None
        logger.exception("Divergence preview worker error", exc_info=exc)
        self._set_status(f"Preview failed: {exc}")

    def _teardown_preview_layers(self, channel: Literal["nucleus", "cell"]) -> None:
        for name in _PREVIEW_LAYER_NAMES[channel]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
            self._preview_needs_autocontrast.discard(name)

    def _compute_channel_frame(self, prob_path, dp_path, t: int, params: dict):
        """Single-frame foreground + contour maps (worker-thread body)."""
        prob_t = self._read_t_frame(prob_path, t)  # (Z, Y, X)
        dp_t = self._read_t_frame(dp_path, t)       # (Z, 2, Y, X)
        fg = foreground_from_prob(
            prob_t[np.newaxis],
            reduction=params["foreground_z_reduction"],
            smoothing_sigma=params["foreground_smoothing_sigma"],
            median_radius=params["foreground_median_radius"],
        )[0]
        contour = contour_from_dp(
            dp_t[np.newaxis],
            smoothing_sigma=params["smoothing_sigma"],
            median_radius=params["median_radius"],
            reduction=params["contour_z_reduction"],
        )[0]
        return fg, contour

    @staticmethod
    def _read_t_frame(path, t: int) -> np.ndarray:
        """Read frame ``t`` of a (T,Z,Y,X) / (T,Z,2,Y,X) stack as float32.

        Uses a lazy memmap when the TIFF is uncompressed, falling back to a full
        read otherwise. A 3D/4D single-frame stack is returned whole.
        """
        try:
            mm = tifffile.memmap(str(path), mode="r")
        except (ValueError, OSError, MemoryError):
            mm = None
        if mm is not None:
            try:
                arr = mm[t] if mm.ndim >= 4 else mm
                return np.asarray(arr, dtype=np.float32)
            finally:
                del mm
        full = tifffile.imread(str(path))
        return np.asarray(full[t] if full.ndim >= 4 else full, dtype=np.float32)

    def _channel_map_shape(self, prob_path):
        """``(T, Y, X)`` from the prob-map TIFF header (no pixel load)."""
        try:
            with tifffile.TiffFile(str(prob_path)) as tf:
                shape = tf.series[0].shape
        except Exception:
            return None
        if len(shape) >= 4:  # (T, Z, Y, X)
            return int(shape[0]), int(shape[-2]), int(shape[-1])
        return 1, int(shape[-2]), int(shape[-1])

    def _current_t(self) -> int:
        step = getattr(getattr(self.viewer, "dims", None), "current_step", (0,))
        return int(step[0]) if len(step) >= 1 else 0

    def _ensure_preview_layers(self, channel: Literal["nucleus", "cell"], shape) -> None:
        """Create zero-filled ``(T, Y, X)`` preview layers if absent/mis-shaped.

        A full-length stack is what gives napari a time slider; per-frame slices
        are filled on demand as the user scrubs.
        """
        fg_name, ct_name = _PREVIEW_LAYER_NAMES[channel]
        self._ensure_image_layer(fg_name, shape, "gray")
        self._ensure_image_layer(ct_name, shape, "magma")

    def _ensure_image_layer(self, name: str, shape, colormap: str) -> None:
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if tuple(layer.data.shape) == tuple(shape):
                return
            self.viewer.layers.remove(name)
        self.viewer.add_image(
            np.zeros(shape, dtype=np.float32), name=name, colormap=colormap,
        )
        # Seed contrast from the first real frame this layer receives.
        self._preview_needs_autocontrast.add(name)

    def _fill_image_layer(self, name: str, t: int, frame: np.ndarray) -> None:
        if name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        data = layer.data
        if getattr(data, "ndim", 0) != 3 or not 0 <= t < data.shape[0]:
            return
        data[t] = np.asarray(frame, dtype=data.dtype)
        if name in self._preview_needs_autocontrast:
            finite = frame[np.isfinite(frame)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                if hi > lo:
                    layer.contrast_limits = (lo, hi)
                    self._preview_needs_autocontrast.discard(name)
        if hasattr(layer, "refresh"):
            layer.refresh()

    # ── Public API ───────────────────────────────────────────────────
    def refresh(self, pos_dir: Path | str | None) -> None:
        self._pos_dir = None if pos_dir is None or str(pos_dir) == "[no project]" else Path(pos_dir)
        if self._files_widget is not None:
            self._files_widget.refresh(self._pos_dir)
        # A live preview points at the current project's maps; if the project
        # goes away, deactivate it so we don't preview stale paths.
        if self._pos_dir is None and self._active_preview_channel is not None:
            channel = self._active_preview_channel
            btn = self.nucleus_preview_btn if channel == "nucleus" else self.cell_preview_btn
            btn.setChecked(False)
        self._update_enabled()

    def _update_enabled(self) -> None:
        # Enablement is owned by the gate; its predicates read ``_pos_dir`` and
        # ``_running_stage``.
        self.gate.recompute()

    def get_state(self) -> dict:
        return {
            "nucleus": self._channel_state("nuc"),
            "cell": self._channel_state("cell"),
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if "nucleus" in state:
            self._apply_channel_state("nuc", state["nucleus"])
        if "cell" in state:
            self._apply_channel_state("cell", state["cell"])

    def _channel_state(self, prefix: str) -> dict:
        # ``smoothing_sigma`` / ``median_radius`` are the *contour* knobs (kept
        # under their original keys for saved-config back-compat); the
        # ``foreground_*`` keys are the separate foreground knobs.
        return {
            "foreground_z_reduction": getattr(self, f"{prefix}_fg_reduction").currentText(),
            "contour_z_reduction": getattr(self, f"{prefix}_contour_reduction").currentText(),
            "smoothing_sigma": float(getattr(self, f"{prefix}_smoothing_spin").value()),
            "median_radius": int(getattr(self, f"{prefix}_median_spin").value()),
            "foreground_smoothing_sigma": float(getattr(self, f"{prefix}_fg_smoothing_spin").value()),
            "foreground_median_radius": int(getattr(self, f"{prefix}_fg_median_spin").value()),
        }

    def _apply_channel_state(self, prefix: str, state: dict) -> None:
        if "foreground_z_reduction" in state:
            getattr(self, f"{prefix}_fg_reduction").setCurrentText(state["foreground_z_reduction"])
        if "contour_z_reduction" in state:
            getattr(self, f"{prefix}_contour_reduction").setCurrentText(state["contour_z_reduction"])
        if "smoothing_sigma" in state:
            getattr(self, f"{prefix}_smoothing_spin").setValue(float(state["smoothing_sigma"]))
        if "median_radius" in state:
            getattr(self, f"{prefix}_median_spin").setValue(int(state["median_radius"]))
        if "foreground_smoothing_sigma" in state:
            getattr(self, f"{prefix}_fg_smoothing_spin").setValue(float(state["foreground_smoothing_sigma"]))
        if "foreground_median_radius" in state:
            getattr(self, f"{prefix}_fg_median_spin").setValue(int(state["foreground_median_radius"]))

    # ── Path helpers ────────────────────────────────────────────────
    def _paths(self) -> NucleusArtifactPaths | None:
        return NucleusArtifactPaths(self._pos_dir) if self._pos_dir else None
