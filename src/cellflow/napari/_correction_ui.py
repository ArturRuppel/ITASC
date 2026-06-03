"""Pure Qt sub-widget builders for the nucleus correction widget.

These are the self-contained construction helpers the correction widget used to
carry inline: each takes the widgets/data it needs and returns a ready ``QWidget``
(or mutates a passed-in section), with no back-reference to the host widget. The
host stays responsible for owning the resulting widgets and wiring their signals.
"""
from __future__ import annotations

from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.ui_style import (
    stage_header_action_button,
    stage_header_label,
)
from cellflow.napari.widgets import CollapsibleSection


def flatten_embedded_section(section: CollapsibleSection) -> None:
    """Strip a section's header/margins so it nests flush inside another panel."""
    section.set_header_visible(False)
    section.layout().setContentsMargins(0, 0, 0, 0)
    section._content_frame.layout().setContentsMargins(0, 0, 0, 0)
    section._content_frame.setStyleSheet(
        "QFrame#collapsible_content { border: none; margin: 0px; }"
    )


def build_correction_header(
    parent: QWidget,
    *,
    shortcuts_btn: QWidget,
    params_btn: QWidget,
    active_btn: QWidget,
) -> tuple[QWidget, QLabel]:
    """Build the stage-style correction header; returns ``(header, title_label)``."""
    header = QWidget(parent)
    row = QHBoxLayout(header)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)

    header_lbl = QLabel("Correction")
    stage_header_label(header_lbl, "nucleus")
    for button in (shortcuts_btn, params_btn, active_btn):
        stage_header_action_button(button, "nucleus")
    row.addWidget(header_lbl)
    row.addWidget(shortcuts_btn)
    row.addWidget(params_btn)
    row.addWidget(active_btn)
    row.addStretch(1)
    return header, header_lbl


def build_correction_toolbar(
    parent: QWidget, button_groups: list[tuple[QWidget, ...]]
) -> QWidget:
    """Lay out groups of action buttons with vertical separators between groups."""
    toolbar = QWidget(parent)
    row = QHBoxLayout(toolbar)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)

    def _sep() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    for i, group in enumerate(button_groups):
        if i > 0:
            row.addWidget(_sep())
        for b in group:
            row.addWidget(b)
    row.addStretch(1)
    return toolbar


def build_shortcuts_widget() -> QWidget:
    """Build the static correction-shortcuts reference panel."""
    group = QGroupBox("Correction shortcuts")
    grid = QGridLayout(group)
    grid.setContentsMargins(8, 6, 8, 6)
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(2)
    row = 0
    row = CorrectionWidget._add_shortcut_group(
        grid,
        "Track Workflow",
        [
            ("V", "Validate selected track"),
            ("B", "Anchor selected cell at current frame"),
            ("A / D", "Extend selected track backward / forward"),
            ("Q / E", "Retrack backward / forward"),
            ("Z / C", "Swap with smaller / larger candidate fragment"),
            ("S", "Save tracked labels"),
        ],
        start_row=row,
        is_first=True,
    )
    row = CorrectionWidget._add_shortcut_group(
        grid,
        "Selection",
        [
            ("Left-click", "Select / highlight cell"),
            ("Shift+Left / Shift+Right", "Previous / next cell"),
        ],
        start_row=row,
    )
    row = CorrectionWidget._add_shortcut_group(
        grid,
        "Manual Labels",
        [
            ("Middle-click or Delete", "Erase cell"),
            ("Ctrl+Left-click", "Merge selected with clicked cell"),
            ("Right-click variants", "Swap labels"),
            ("Shift+Left-drag", "Draw / extend cell path"),
            ("Shift+Right-drag", "Split by drawn line"),
        ],
        start_row=row,
    )
    row = CorrectionWidget._add_shortcut_group(
        grid, "History", [("Ctrl+Z", "Undo")], start_row=row
    )
    grid.setColumnStretch(1, 1)
    return group
