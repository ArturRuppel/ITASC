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
from napari.qt.threading import thread_worker
from qtpy.QtCore import Signal
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
from cellflow.segmentation.divergence_maps import build_divergence_maps

logger = logging.getLogger(__name__)


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
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._running_stage: str | None = None
        self._worker = None
        self._cancel_event: threading.Event | None = None
        self._show_pipeline_files = bool(show_pipeline_files)

        self._setup_ui()
        self._connect_signals()
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
        self.nucleus_run_btn = _tool_btn("▶", "Build nucleus divergence maps.")
        for button in (self.nucleus_params_btn, self.nucleus_run_btn):
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
            self.nucleus_run_btn,
        ))
        root.addWidget(self.nucleus_section)

        # Cell row
        self.cell_params_btn = _tool_btn(
            "⚙", "Show parameters for cell divergence maps.", checkable=True,
        )
        self.cell_run_btn = _tool_btn("▶", "Build cell divergence maps.")
        for button in (self.cell_params_btn, self.cell_run_btn):
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
            "Smoothing sigma:", smoothing_spin,
            "Median radius:", median_spin,
        )

        prefix = "nuc" if channel == "nucleus" else "cell"
        setattr(self, f"{prefix}_fg_reduction", fg_reduction)
        setattr(self, f"{prefix}_contour_reduction", contour_reduction)
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
        other_btn = self.cell_run_btn if channel == "nucleus" else self.nucleus_run_btn
        run_btn.setText("✕")
        run_btn.setToolTip("Cancel.")
        other_btn.setEnabled(False)

    def _set_button_idle(self) -> None:
        for btn, tip in (
            (self.nucleus_run_btn, "Build nucleus divergence maps."),
            (self.cell_run_btn, "Build cell divergence maps."),
        ):
            btn.setText("▶")
            btn.setToolTip(tip)
            btn.setEnabled(self._pos_dir is not None)

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

    # ── Public API ───────────────────────────────────────────────────
    def refresh(self, pos_dir: Path | str | None) -> None:
        self._pos_dir = None if pos_dir is None or str(pos_dir) == "[no project]" else Path(pos_dir)
        if self._files_widget is not None:
            self._files_widget.refresh(self._pos_dir)
        self._update_enabled()

    def _update_enabled(self) -> None:
        has_pos = self._pos_dir is not None
        for btn in (self.nucleus_run_btn, self.cell_run_btn,
                    self.nucleus_params_btn, self.cell_params_btn):
            btn.setEnabled(has_pos)

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
        return {
            "foreground_z_reduction": getattr(self, f"{prefix}_fg_reduction").currentText(),
            "contour_z_reduction": getattr(self, f"{prefix}_contour_reduction").currentText(),
            "smoothing_sigma": float(getattr(self, f"{prefix}_smoothing_spin").value()),
            "median_radius": int(getattr(self, f"{prefix}_median_spin").value()),
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

    # ── Path helpers ────────────────────────────────────────────────
    def _paths(self) -> NucleusArtifactPaths | None:
        return NucleusArtifactPaths(self._pos_dir) if self._pos_dir else None
