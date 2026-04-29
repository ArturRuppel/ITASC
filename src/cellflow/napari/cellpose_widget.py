"""Informational panel for external Cellpose output."""
from __future__ import annotations

from pathlib import Path

import napari
import tifffile
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

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
                ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
                ("1_cellpose/cell_prob_zavg.tif", "Cell prob z-avg"),
            ]),
        ])
        layout.addWidget(self.files_tracker)

        # Action buttons
        btn_row = QHBoxLayout()
        self.load_nuc_btn = QPushButton("Load Nucleus (z-avg)")
        self.load_cell_btn = QPushButton("Load Cell (z-avg)")
        btn_row.addWidget(self.load_nuc_btn)
        btn_row.addWidget(self.load_cell_btn)
        layout.addLayout(btn_row)

        self.load_nuc_btn.clicked.connect(self._on_load_nucleus)
        self.load_cell_btn.clicked.connect(self._on_load_cell)

        self._pos_dir: Path | None = None

    def refresh(self, pos_dir: Path | None) -> None:
        """Update file status display."""
        self._pos_dir = pos_dir
        self.files_tracker.refresh(pos_dir)

    def _on_load_nucleus(self) -> None:
        if not self._pos_dir:
            return
        
        # Try to load input and output
        nuc_input = self._pos_dir / "0_input/nucleus_zavg.tif"
        nuc_prob = self._pos_dir / "1_cellpose/nucleus_prob_zavg.tif"

        if nuc_input.exists():
            data = tifffile.imread(str(nuc_input))
            self.viewer.add_image(data, name=f"nuc_input_{self._pos_dir.name}", colormap="gray")
        
        if nuc_prob.exists():
            data = tifffile.imread(str(nuc_prob))
            self.viewer.add_image(data, name=f"nuc_prob_{self._pos_dir.name}", colormap="inferno")

    def _on_load_cell(self) -> None:
        if not self._pos_dir:
            return

        cell_input = self._pos_dir / "0_input/cell_zavg.tif"
        cell_prob = self._pos_dir / "1_cellpose/cell_prob_zavg.tif"

        if cell_input.exists():
            data = tifffile.imread(str(cell_input))
            self.viewer.add_image(data, name=f"cell_input_{self._pos_dir.name}", colormap="gray")
        
        if cell_prob.exists():
            data = tifffile.imread(str(cell_prob))
            self.viewer.add_image(data, name=f"cell_prob_{self._pos_dir.name}", colormap="inferno")
