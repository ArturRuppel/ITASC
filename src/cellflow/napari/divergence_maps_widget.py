"""Per-channel widget that builds nucleus/cell foreground & contour maps
directly from Cellpose ``prob_3dt`` and ``dp_3dt`` outputs.

Mirrors :class:`CellposeWidget` layout (one row per channel with
⚙ params / ▶ run-cancel and a shared status + progress bar).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import napari
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
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

from cellflow.napari._paths import NucleusArtifactPaths
from cellflow.napari._widget_helpers import tool_btn as _tool_btn
from cellflow.napari.ui_style import stage_header_label, status_label
from cellflow.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    make_pipeline_files_header,
)

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

    # ── UI ───────────────────────────────────────────────────────────
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

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

        # Nucleus row
        self.nucleus_params_btn = _tool_btn(
            "⚙", "Show parameters for nucleus divergence maps.", checkable=True,
        )
        self.nucleus_run_btn = _tool_btn("▶", "Build nucleus divergence maps.")
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
        form = QFormLayout(body)
        form.setContentsMargins(8, 4, 4, 4)
        form.setSpacing(4)

        fg_reduction = QComboBox()
        fg_reduction.addItems(["mean", "max"])
        fg_reduction.setCurrentText("mean")
        contour_reduction = QComboBox()
        contour_reduction.addItems(["mean", "max"])
        contour_reduction.setCurrentText("mean")
        smoothing_spin = QDoubleSpinBox()
        smoothing_spin.setRange(0.0, 20.0)
        smoothing_spin.setDecimals(2)
        smoothing_spin.setSingleStep(0.1)
        smoothing_spin.setValue(1.0)
        median_spin = QSpinBox()
        median_spin.setRange(0, 20)
        median_spin.setValue(0)
        form.addRow("Foreground z-reduction", fg_reduction)
        form.addRow("Contour z-reduction", contour_reduction)
        form.addRow("Smoothing sigma", smoothing_spin)
        form.addRow("Median radius", median_spin)

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
        row.addStretch(1)
        for w in trailing:
            row.addWidget(w)
        return row

    # ── Signals ──────────────────────────────────────────────────────
    def _connect_signals(self) -> None:
        self.nucleus_run_btn.clicked.connect(lambda: self._on_run("nucleus"))
        self.cell_run_btn.clicked.connect(lambda: self._on_run("cell"))

    # ── Stubs filled in by Task 7 ───────────────────────────────────
    def _on_run(self, channel: Literal["nucleus", "cell"]) -> None:  # noqa: D401
        """Run/cancel dispatch — implemented in Task 7."""
        raise NotImplementedError("Wired in Task 7")

    def _progress(self, done: int, total: int, msg: str) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(done)
        self.status_lbl.setText(msg)
        self.status_lbl.setVisible(bool(msg))

    # ── Public API ───────────────────────────────────────────────────
    def refresh(self, pos_dir: Path | str | None) -> None:
        self._pos_dir = None if pos_dir is None or str(pos_dir) == "[no project]" else Path(pos_dir)
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
