"""Analysis widget for final processing and export in CellFlow v2."""
from __future__ import annotations

from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QPushButton,
)


class AnalysisWidget(QWidget):
    """Final analysis and export."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QPushButton("Calculate Topology"))
        layout.addWidget(QPushButton("Calculate Mechanics"))
        layout.addWidget(QPushButton("Export Statistics"))
