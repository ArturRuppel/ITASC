"""Main widget for the CellFlow napari plugin."""
from __future__ import annotations

import napari
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QLabel,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QFrame,
    QSizePolicy,
)
from qtpy.QtCore import Qt, QTimer
from pathlib import Path

from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget

# Define tracked files for v2
_TRACKED_FILE_GROUPS = [
    ("Input Data", [
        ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
        ("0_input/cell_zavg.tif", "Cell z-avg"),
    ]),
    ("Cellpose", [
        ("1_cellpose/nucleus_prob.tif", "Nucleus probability"),
        ("1_cellpose/cell_prob.tif", "Cell probability"),
    ]),
    ("Nucleus Workflow", [
        ("2_nucleus/hypotheses.h5", "Hypotheses HDF5"),
        ("2_nucleus/tracked_labels.h5", "Tracked labels"),
    ]),
    ("Cell Workflow", [
        ("3_cell/hypotheses.h5", "Hypotheses HDF5"),
        ("3_cell/tracked_labels.h5", "Tracked labels"),
    ]),
]


class DataPanel(QWidget):
    """Widget for managing project paths, metadata, and configuration."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # ── Metadata row ──────────────────────
        meta_row = QHBoxLayout()
        meta_row.setSpacing(4)
        meta_row.addWidget(QLabel("px (µm):"))
        self.px_edit = QLineEdit()
        self.px_edit.setFixedWidth(45)
        meta_row.addWidget(self.px_edit)

        meta_row.addWidget(QLabel("dt (min):"))
        self.dt_edit = QLineEdit()
        self.dt_edit.setFixedWidth(45)
        meta_row.addWidget(self.dt_edit)

        meta_row.addWidget(QLabel("Cond:"))
        self.cond_edit = QLineEdit()
        meta_row.addWidget(self.cond_edit)

        meta_row.addWidget(QLabel("Pos:"))
        from qtpy.QtWidgets import QSpinBox
        self.pos_spin = QSpinBox()
        self.pos_spin.setRange(0, 99)
        self.pos_spin.setFixedWidth(45)
        meta_row.addWidget(self.pos_spin)
        layout.addLayout(meta_row)

        # ── Project buttons ───────────────────
        project_row = QHBoxLayout()
        self.new_btn = QPushButton("New Project...")
        self.open_btn = QPushButton("Open Project...")
        project_row.addWidget(self.new_btn)
        project_row.addWidget(self.open_btn)
        layout.addLayout(project_row)

        # ── Config buttons ────────────────────
        layout.addWidget(QLabel("<b>Configuration</b>"))
        config_row = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_as_btn = QPushButton("Save As...")
        self.load_btn = QPushButton("Load")
        self.load_from_btn = QPushButton("Load From...")
        
        # Make them compact
        for btn in (self.save_btn, self.save_as_btn, self.load_btn, self.load_from_btn):
            btn.setStyleSheet("font-size: 8pt; padding: 2px;")
            
        config_row.addWidget(self.save_btn)
        config_row.addWidget(self.save_as_btn)
        config_row.addWidget(self.load_btn)
        config_row.addWidget(self.load_from_btn)
        layout.addLayout(config_row)

        self.path_label = QLabel("[no project]")
        self.path_label.setStyleSheet("font-size: 8pt; color: #aaaaaa;")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        # ── File Tracker (Scrollable) ─────────
        tracker_header = QHBoxLayout()
        tracker_header.addWidget(QLabel("<b>File Status</b>"))
        self.refresh_btn = QPushButton("↺")
        self.refresh_btn.setFixedWidth(24)
        self.refresh_btn.setToolTip("Refresh file status")
        tracker_header.addWidget(self.refresh_btn)
        layout.addLayout(tracker_header)

        self.file_tracker = PipelineFilesWidget(_TRACKED_FILE_GROUPS)
        
        scroll = QScrollArea()
        scroll.setWidget(self.file_tracker)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(150)
        scroll.setFrameShape(QFrame.NoFrame)
        layout.addWidget(scroll)

        # Connect signals
        self.open_btn.clicked.connect(self._on_open_project)
        self.refresh_btn.clicked.connect(self._refresh_files)
        self.pos_spin.valueChanged.connect(self._refresh_files)

    def _on_open_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open Project Directory")
        if path:
            self.path_label.setText(path)
            self.path_label.setToolTip(path)
            self._refresh_files()

    def _refresh_files(self) -> None:
        path = self.path_label.text()
        if path and path != "[no project]":
            pos = self.pos_spin.value()
            pos_dir = Path(path) / f"pos{pos:02d}"
            self.file_tracker.refresh(pos_dir)
        else:
            self.file_tracker.refresh(None)


class CellposeWidget(QWidget):
    """Informational panel for external Cellpose output."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Cellpose artifacts expected in 1_cellpose/"))
        layout.addWidget(QLabel("Status: Not Found"))
        layout.addWidget(QPushButton("Refresh status"))


class NucleusWorkflowWidget(QWidget):
    """Nucleus hypothesis generation and tracking."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # 3A. Hypothesis Generation
        layout.addWidget(QLabel("<b>A. Hypothesis Generation</b>"))
        layout.addWidget(QLabel("Min/Max Area:"))
        layout.addWidget(QLineEdit("100, 1000"))
        layout.addWidget(QLabel("Threshold Sweep:"))
        layout.addWidget(QLineEdit("0.1, 0.5, 0.9"))
        layout.addWidget(QPushButton("Generate Nucleus Hypotheses"))

        layout.addSpacing(10)

        # 3B. Seeding
        layout.addWidget(QLabel("<b>B. Seeding</b>"))
        layout.addWidget(QPushButton("Pick Initial Seed"))

        layout.addSpacing(10)

        # 3C. Correction
        layout.addWidget(QLabel("<b>C. Manual Correction</b>"))
        layout.addWidget(QPushButton("Open Correction Tool"))

        layout.addSpacing(10)

        # 3D. Search
        layout.addWidget(QLabel("<b>D. Automated Search</b>"))
        layout.addWidget(QLabel("IoU Threshold:"))
        layout.addWidget(QLineEdit("0.5"))
        layout.addWidget(QPushButton("Propagate Labels"))


class CellWorkflowWidget(QWidget):
    """Cell hypothesis generation and tracking."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # 4A. Seeded Hypothesis Generation
        layout.addWidget(QLabel("<b>A. Seeded Hypothesis Generation</b>"))
        layout.addWidget(QLabel("Seeded Watershed Sigma:"))
        layout.addWidget(QLineEdit("2.0"))
        layout.addWidget(QPushButton("Generate Cell Hypotheses (Seeded)"))

        layout.addSpacing(10)

        # 4B. Search & Selection
        layout.addWidget(QLabel("<b>B. Search & Selection</b>"))
        layout.addWidget(QLabel("IoU Threshold:"))
        layout.addWidget(QLineEdit("0.5"))
        layout.addWidget(QPushButton("Propagate Cell Labels"))


class AnalysisWidget(QWidget):
    """Final analysis and export."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QPushButton("Calculate Topology"))
        layout.addWidget(QPushButton("Calculate Mechanics"))
        layout.addWidget(QPushButton("Export Statistics"))


class CellFlowMainWidget(QWidget):
    """The unified workflow-based UI for CellFlow."""

    def __init__(self, napari_viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # Main scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(5, 5, 5, 5)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.scroll_widget)

        self.layout().addWidget(self.scroll)

        # Add sections
        self.data_panel = DataPanel()
        self.data_section = CollapsibleSection(
            "1. Data Panel", self.data_panel, expanded=True
        )
        self.cellpose_section = CollapsibleSection(
            "2. Cellpose Output", CellposeWidget(), expanded=False
        )
        self.nucleus_section = CollapsibleSection(
            "3. Nucleus Workflow", NucleusWorkflowWidget(), expanded=False
        )
        self.cell_section = CollapsibleSection(
            "4. Cell Workflow", CellWorkflowWidget(), expanded=False
        )
        self.analysis_section = CollapsibleSection(
            "5. Analysis", AnalysisWidget(), expanded=False
        )

        self.scroll_layout.addWidget(self.data_section)
        self.scroll_layout.addWidget(self.cellpose_section)
        self.scroll_layout.addWidget(self.nucleus_section)
        self.scroll_layout.addWidget(self.cell_section)
        self.scroll_layout.addWidget(self.analysis_section)

        # Add stretch at the end
        self.scroll_layout.addStretch()
