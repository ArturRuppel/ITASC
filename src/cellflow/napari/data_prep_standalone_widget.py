"""Standalone entry point for the Data Preparation widget."""
from __future__ import annotations

from pathlib import Path

import napari
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.data_prep_widget import DataPrepWidget
from cellflow.napari.hpc_cellpose_widget import HpcCellposeWidget
from cellflow.napari.ui_style import icon_button, muted_label, tiny_button
from cellflow.napari.widgets import CollapsibleSection


class DataPrepStandaloneWidget(QWidget):
    """Standalone wrapper around DataPrepWidget with its own project controls."""

    refresh_requested = Signal(object)  # emits pos_dir: Path | None

    def __init__(self, napari_viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = napari_viewer

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── Project controls ──────────────────────────────────────────
        ctrl_widget = QWidget()
        ctrl_lay = QVBoxLayout(ctrl_widget)
        ctrl_lay.setContentsMargins(0, 0, 0, 0)
        ctrl_lay.setSpacing(4)

        meta_row = QHBoxLayout()
        meta_row.setSpacing(4)

        meta_row.addWidget(QLabel("px:"))
        self.px_edit = QLineEdit()
        self.px_edit.setFixedWidth(40)
        meta_row.addWidget(self.px_edit)

        meta_row.addWidget(QLabel("dt:"))
        self.dt_edit = QLineEdit()
        self.dt_edit.setFixedWidth(40)
        meta_row.addWidget(self.dt_edit)

        meta_row.addWidget(QLabel("P:"))
        self.pos_spin = QSpinBox()
        self.pos_spin.setRange(0, 99)
        self.pos_spin.setFixedWidth(40)
        meta_row.addWidget(self.pos_spin)

        self.refresh_btn = QPushButton("↺")
        icon_button(self.refresh_btn)
        self.refresh_btn.setToolTip("Refresh status")
        meta_row.addWidget(self.refresh_btn)

        ctrl_lay.addLayout(meta_row)

        project_row = QHBoxLayout()
        project_row.setSpacing(4)
        self.project_btn = QPushButton("Project Directory...")
        tiny_button(self.project_btn)
        project_row.addWidget(self.project_btn)
        ctrl_lay.addLayout(project_row)

        self.path_label = QLabel("[no project]")
        muted_label(self.path_label)
        self.path_label.setWordWrap(True)
        ctrl_lay.addWidget(self.path_label)

        main_layout.addWidget(ctrl_widget)

        # ── Data Prep widget ──────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_widget = QWidget()
        scroll_widget.setMinimumWidth(0)
        scroll_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(2, 2, 2, 2)
        scroll_layout.setAlignment(Qt.AlignTop)

        self._data_prep = DataPrepWidget(napari_viewer, self)
        scroll_layout.addWidget(self._data_prep)

        self.hpc_cellpose_widget = HpcCellposeWidget(napari_viewer, self)
        self.hpc_cellpose_section = CollapsibleSection(
            "HPC Cellpose",
            self.hpc_cellpose_widget,
            expanded=False,
        )
        scroll_layout.addWidget(self.hpc_cellpose_section)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

        # ── Signals ───────────────────────────────────────────────────
        self.project_btn.clicked.connect(self._on_set_project_directory)
        self.refresh_btn.clicked.connect(self._refresh)
        self.pos_spin.valueChanged.connect(self._refresh)

    def _on_set_project_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if path:
            self.path_label.setText(path)
            self.path_label.setToolTip(path)
            self._refresh()

    def _refresh(self) -> None:
        path_text = self.path_label.text()
        if path_text and path_text != "[no project]":
            pos = self.pos_spin.value()
            pos_dir = Path(path_text) / f"pos{pos:02d}"
        else:
            pos_dir = None
        self.refresh_requested.emit(pos_dir)
        self._data_prep.refresh(pos_dir)
        self.hpc_cellpose_widget.refresh(pos_dir)
