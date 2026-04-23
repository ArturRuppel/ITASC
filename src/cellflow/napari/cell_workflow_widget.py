"""Cell workflow widget for hypothesis generation and tracking in CellFlow v2."""
from __future__ import annotations

from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
)


class CellWorkflowWidget(QWidget):
    """Cell hypothesis generation and tracking."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # 4A. Seeded Hypothesis Generation
        layout.addWidget(QLabel("<b>A. Seeded Hypothesis Generation</b>"))
        layout.addWidget(QLabel("Seeded Watershed Sigma:"))
        self.sigma_edit = QLineEdit("2.0")
        layout.addWidget(self.sigma_edit)
        self.gen_btn = QPushButton("Generate Cell Hypotheses (Seeded)")
        layout.addWidget(self.gen_btn)

        layout.addSpacing(10)

        # 4B. Search & Selection
        layout.addWidget(QLabel("<b>B. Search & Selection</b>"))
        layout.addWidget(QLabel("IoU Threshold:"))
        self.iou_edit = QLineEdit("0.5")
        layout.addWidget(self.iou_edit)
        self.prop_btn = QPushButton("Propagate Cell Labels")
        layout.addWidget(self.prop_btn)

    def get_state(self) -> dict:
        """Return the current UI state as a dictionary."""
        return {
            "sigma": self.sigma_edit.text(),
            "iou_threshold": self.iou_edit.text(),
        }

    def set_state(self, state: dict) -> None:
        """Update the UI state from a dictionary."""
        if "sigma" in state:
            self.sigma_edit.setText(str(state["sigma"]))
        if "iou_threshold" in state:
            self.iou_edit.setText(str(state["iou_threshold"]))
