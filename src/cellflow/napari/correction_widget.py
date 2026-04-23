"""Global label correction widget for CellFlow v2."""
from __future__ import annotations

import logging
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class CorrectionWidget(QWidget):
    """Global label correction tool with cell inspection."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        # ── Activation ────────────────────────────────────────────────────
        self.activate_btn = QPushButton("Activate Correction")
        self.activate_btn.setCheckable(True)
        self.activate_btn.setToolTip("Enable interactive mouse callbacks for merging/splitting.")
        self.activate_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.activate_btn)

        self.status_lbl = QLabel("Inactive")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.status_lbl)

        # ── Shortcuts Reference ───────────────────────────────────────────
        ref_group = QGroupBox("Shortcuts")
        ref_lay = QVBoxLayout(ref_group)
        ref_lay.setSpacing(2)
        
        shortcuts = [
            ("Left-click", "Select cell"),
            ("Ctrl+Left-click", "Merge / Split"),
            ("Delete", "Erase selected"),
            ("Shift+Drag (L)", "Draw path"),
            ("Shift+Drag (R)", "Split line"),
        ]
        for key, desc in shortcuts:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"<b>{key}</b>"))
            row.addWidget(QLabel(desc))
            ref_lay.addLayout(row)
        layout.addWidget(ref_group)

        # ── Cell Inspector ────────────────────────────────────────────────
        inspect_group = QGroupBox("Cell Inspector")
        inspect_lay = QVBoxLayout(inspect_group)
        
        row_id = QHBoxLayout()
        row_id.addWidget(QLabel("Cell ID:"))
        self.cell_id_spin = QSpinBox()
        self.cell_id_spin.setRange(0, 999999)
        row_id.addWidget(self.cell_id_spin)
        self.go_btn = QPushButton("Go")
        row_id.addWidget(self.go_btn)
        inspect_lay.addLayout(row_id)
        
        self.lifetime_lbl = QLabel("Lifetime: ---")
        self.lifetime_lbl.setStyleSheet("font-size: 9pt; color: #aaa;")
        inspect_lay.addWidget(self.lifetime_lbl)
        
        layout.addWidget(inspect_group)
        layout.addStretch()
