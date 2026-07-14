"""Segmentation input parameter section for the nucleus workflow widget."""
from __future__ import annotations

from qtpy.QtWidgets import QWidget

from itasc.napari.widgets import CollapsibleSection


class NucleusSegmentationInputsWidget(QWidget):
    """Compatibility shell for the removed standalone Ultrack Inputs stage."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        inner = QWidget(self)

        self.section = CollapsibleSection(
            "Ultrack Input Parameters",
            inner,
            expanded=True,
        )
