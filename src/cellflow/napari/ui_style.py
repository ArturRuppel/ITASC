from __future__ import annotations

from qtpy.QtWidgets import QSizePolicy

TINY_MARGIN = 2
SECTION_MARGIN = 4
TIGHT_SPACING = 4
DEFAULT_SPIN_WIDTH = 70


def compact_spinbox(widget, width=DEFAULT_SPIN_WIDTH):
    widget.setMaximumWidth(width)
    widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return widget


def action_button(button, expand=False):
    horizontal_policy = (
        QSizePolicy.Policy.Expanding if expand else QSizePolicy.Policy.Fixed
    )
    button.setSizePolicy(horizontal_policy, QSizePolicy.Policy.Fixed)
    return button


def tiny_button(button):
    button.setStyleSheet("font-size: 8pt; padding: 1px 4px;")
    button.setSizePolicy(
        button.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Fixed
    )
    return button


def icon_button(button, width=24, height=None):
    button.setFixedWidth(width)
    if height is not None:
        button.setFixedHeight(height)
    return button


def muted_label(label, size_pt=8):
    label.setStyleSheet(f"color: palette(mid); font-size: {size_pt}pt;")
    return label


def status_label(label, size_pt=8, italic=False):
    style = f"font-size: {size_pt}pt;"
    if italic:
        style += " font-style: italic;"
    label.setStyleSheet(style)
    return label


def danger_button(button):
    button.setStyleSheet(
        """
        QPushButton {
            background-color: #b00020;
            color: white;
        }
        QPushButton:hover {
            background-color: #c62828;
        }
        """
    )
    return button


def checked_success_button(button):
    button.setStyleSheet(
        """
        QPushButton:checked {
            background-color: #2e7d32;
            color: white;
        }
        """
    )
    return button
