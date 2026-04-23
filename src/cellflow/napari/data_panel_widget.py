"""Data panel widget for project and metadata management in CellFlow v2."""
from __future__ import annotations

from pathlib import Path
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.widgets import PipelineFilesWidget

# Define tracked files for the project-wide status view
_TRACKED_FILE_GROUPS = [
    ("Input Data", [
        ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
        ("0_input/cell_zavg.tif", "Cell z-avg"),
        ("0_input/nucleus_3dt.tif", "Nucleus 3D+t"),
        ("0_input/cell_3dt.tif", "Cell 3D+t"),
    ]),
    ("Cellpose", [
        ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
        ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
        ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
        ("1_cellpose/cell_prob_zavg.tif", "Cell prob z-avg"),
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


class ProjectStatusPanel(QWidget):
    """Widget for viewing project file status."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # ── File Tracker (Scrollable) ─────────
        self.file_tracker = PipelineFilesWidget(_TRACKED_FILE_GROUPS)
        
        scroll = QScrollArea()
        scroll.setWidget(self.file_tracker)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(150)
        scroll.setFrameShape(QFrame.NoFrame)
        layout.addWidget(scroll)

    def refresh(self, pos_dir: Path | None) -> None:
        """Update file status display."""
        self.file_tracker.refresh(pos_dir)
