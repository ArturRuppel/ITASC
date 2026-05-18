"""Local Cellpose-SAM widget — per-channel rows with preview, run, cancel."""
from __future__ import annotations

import logging
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
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
from cellflow.napari.cellpose_zavg_viz_widget import CellposeZavgVizWidget
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
        ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
        ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
        ("1_cellpose/cell_prob_zavg.tif", "Cell prob z-avg"),
        ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
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


class CellposeWidget(QWidget):
    """Local Cellpose-SAM runner — two rows (Nucleus, Cell)."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._running_stage: str | None = None
        self._worker = None
        self._cancel_requested = False

        self._setup_ui()
        self._connect_signals()

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

        # ── Z-avg viz (unchanged) ──────────────────────────────────────
        self.zavg_viz_widget = CellposeZavgVizWidget()
        root.addWidget(self.zavg_viz_widget)

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

    def _on_cell_run_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()

    def _on_nucleus_preview(self) -> None:
        pass

    def _on_cell_preview(self) -> None:
        pass

    def _on_cancel(self) -> None:
        self._cancel_requested = True
        worker = self._worker
        if worker is not None and hasattr(worker, "quit"):
            worker.quit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._files_widget.refresh(pos_dir)
        self.zavg_viz_widget.refresh(pos_dir)

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
                self.viewer.layers[name].data = data
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[name])
                adder(data, name=name, **kwargs)
        else:
            adder(data, name=name, **kwargs)
