"""Data panel widget for project and metadata management in CellFlow v2."""
from __future__ import annotations

from pathlib import Path
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.widgets import PipelineFilesWidget

# Canonical file groups for the project-wide status view. This rolls up the
# per-stage pipeline files that each subwidget tracks, so it must stay in sync
# with their PipelineFilesWidget contracts:
#   Cellpose          → cellpose_widget._PIPELINE_FILES
#   Nucleus Workflow  → nucleus_workflow_widget._NUCLEUS_PIPELINE_FILE_GROUPS
#   Cell Workflow     → cell_workflow_widget (CellWorkflowWidget._setup_ui)
#   Contact Analysis  → contact_analysis_widget (AggregateQuantificationWidget)
_TRACKED_FILE_GROUPS = [
    ("Input Data", [
        ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
        ("0_input/cell_zavg.tif", "Cell z-avg"),
        ("0_input/nucleus_3dt.tif", "Nucleus 3D+t"),
        ("0_input/cell_3dt.tif", "Cell 3D+t"),
    ]),
    ("Cellpose", [
        ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
        ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ("1_cellpose/nucleus_contours.tif", "Nucleus contours"),
        ("1_cellpose/nucleus_foreground.tif", "Nucleus foreground"),
        ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
        ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
        ("1_cellpose/cell_contours.tif", "Cell contours"),
        ("1_cellpose/cell_foreground.tif", "Cell foreground"),
    ]),
    ("Nucleus Workflow", [
        ("2_nucleus/atoms.tif", "Atoms"),
        ("2_nucleus/ultrack_workdir/data.db", "Ultrack DB"),
        ("2_nucleus/tracked_labels.tif", "Tracked labels"),
    ]),
    ("Cell Workflow", [
        ("3_cell/tracked_labels.tif", "Tracked labels"),
    ]),
    ("Contact Analysis", [
        ("aggregate_quantification/contact_analysis.h5", "Contact analysis"),
    ]),
]


class ProjectStatusPanel(QWidget):
    """Widget for viewing project file status."""

    def __init__(self, viewer=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        self.file_tracker = PipelineFilesWidget(_TRACKED_FILE_GROUPS, viewer=viewer)
        layout.addWidget(self.file_tracker)

    def refresh(self, pos_dir: Path | None) -> None:
        """Update file status display."""
        self.file_tracker.refresh(pos_dir)
