"""Informational panel for external Cellpose output."""
from __future__ import annotations

from pathlib import Path

import napari
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from cellflow.napari.ui_style import muted_label
from cellflow.napari.widgets import PipelineFilesWidget


class CellposeWidget(QWidget):
    """Informational panel for external Cellpose output."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        description = QLabel(
            "Cellpose runs externally on the cluster. This panel only documents "
            "the expected input/output files and loads them into napari."
        )
        description.setWordWrap(True)
        muted_label(description)
        layout.addWidget(description)

        self.files_tracker = PipelineFilesWidget([
            ("Inputs", [
                ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
                ("0_input/cell_zavg.tif", "Cell z-avg"),
                ("0_input/nucleus_3dt.tif", "Nucleus 3D+t"),
                ("0_input/cell_3dt.tif", "Cell 3D+t"),
            ]),
            ("Outputs", [
                ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
                ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
                ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
                ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
                ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
                ("1_cellpose/cell_prob_zavg.tif", "Cell prob z-avg"),
            ]),
        ], viewer=self.viewer)
        layout.addWidget(self.files_tracker)

        self._pos_dir: Path | None = None

    def refresh(self, pos_dir: Path | None) -> None:
        """Update file status display."""
        self._pos_dir = pos_dir
        self.files_tracker.refresh(pos_dir)
