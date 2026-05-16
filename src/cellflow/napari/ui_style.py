from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout, QGridLayout, QLabel, QSizePolicy

TINY_MARGIN = 2
SECTION_MARGIN = 4
TIGHT_SPACING = 4
DEFAULT_SPIN_WIDTH = 70
DEFAULT_FIELD_SPACING = 8
DEFAULT_ROW_SPACING = 4
DEFAULT_SWEEP_SPIN_WIDTH = 62
BLOCK_GRID_COLUMNS = 4

SEMANTIC_COLORS = {
    "stage": ("#ffffff", "#ffffff", "#ffffff"),
    "params": ("#ffffff", "#ffffff", "#ffffff"),
    "actions": ("#2e7a9e", "#2e7a9e", "#2e7a9e"),
    "indicators": ("#ffffff", "#ffffff", "#ffffff"),
}

def semantic_color(role: str, level: int = 0) -> str:
    shades = SEMANTIC_COLORS[role]
    index = min(max(level, 0), len(shades) - 1)
    return shades[index]


# ── Theme palette ────────────────────────────────────────────────────────
# Catppuccin Mocha accent palette. To add another flavor later, define a
# parallel dict (e.g. CATPPUCCIN_LATTE) with the same keys and reassign
# ACTIVE_PALETTE. Call sites should reference accents via stage_accent()
# so theme changes propagate automatically.
CATPPUCCIN_MOCHA = {
    "rosewater": "#f5e0dc", "flamingo": "#f2cdcd", "pink":      "#f5c2e7",
    "mauve":     "#cba6f7", "red":      "#f38ba8", "maroon":    "#eba0ac",
    "peach":     "#fab387", "yellow":   "#f9e2af", "green":     "#a6e3a1",
    "teal":      "#94e2d5", "sky":      "#89dceb", "sapphire":  "#74c7ec",
    "blue":      "#89b4fa", "lavender": "#b4befe",
}

ACTIVE_PALETTE = CATPPUCCIN_MOCHA

STAGE_ACCENTS = {
    "project_status":   "sapphire",
    "cellpose":         "peach",
    "nucleus":          "green",
    "cell":             "lavender",
    "contact_analysis": "mauve",
}


def stage_accent(stage_key: str) -> str:
    """Resolve a stage key to its accent hex via the active palette."""
    return ACTIVE_PALETTE[STAGE_ACCENTS[stage_key]]


# Stage status indicator colors. Keyed by status name so call sites stay
# decoupled from specific hexes.
STAGE_STATUS_COLORS = {
    "not_started": "#6c7086",  # Catppuccin overlay0 — muted gray
    "in_progress": CATPPUCCIN_MOCHA["yellow"],
    "done":        CATPPUCCIN_MOCHA["green"],
}


def stage_status_color(status: str) -> str:
    return STAGE_STATUS_COLORS[status]


def _fixed_widget(widget, width=None):
    if width is not None:
        widget.setMaximumWidth(width)
    widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return widget


def compact_spinbox(widget, width=DEFAULT_SPIN_WIDTH):
    return _fixed_widget(widget, width)


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


def status_label(label, size_pt=8, italic=False, muted=False):
    style = f"font-size: {size_pt}pt;"
    if muted:
        style += " color: palette(mid);"
    else:
        style += f" color: {semantic_color('indicators')};"
    if italic:
        style += " font-style: italic;"
    label.setStyleSheet(style)
    return label


def parameter_heading(label, level=1):
    label.setStyleSheet(
        f"font-weight: 600; color: {semantic_color('params', level)};"
    )
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
            font-weight: bold;
        }
        """
    )
    return button


def compact_form_layout():
    layout = QFormLayout()
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
    layout.setHorizontalSpacing(DEFAULT_FIELD_SPACING)
    layout.setVerticalSpacing(DEFAULT_ROW_SPACING)
    return layout


def block_grid(horizontal_spacing=8, vertical_spacing=4):
    layout = QGridLayout()
    layout.setHorizontalSpacing(horizontal_spacing)
    layout.setVerticalSpacing(vertical_spacing)
    for col in range(BLOCK_GRID_COLUMNS):
        layout.setColumnStretch(col, 0)
    return layout


def two_column_parameter_grid(horizontal_spacing=12, vertical_spacing=4):
    return block_grid(horizontal_spacing, vertical_spacing)


def _block_label(text):
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return label


def _add_block_cell(grid, row, column, widget, span=1, alignment=None):
    if alignment is None:
        grid.addWidget(widget, row, column, 1, span)
    else:
        grid.addWidget(widget, row, column, 1, span, alignment)
    return widget


def add_block_pair_row(
    grid,
    row,
    left_label,
    left_widget,
    right_label=None,
    right_widget=None,
    field_width=70,
):
    left_label_widget = _block_label(left_label)
    _add_block_cell(grid, row, 0, left_label_widget)
    _add_block_cell(grid, row, 1, _fixed_widget(left_widget, field_width))

    right_label_widget = None
    if right_widget is not None:
        right_label_widget = _block_label(right_label or "")
        _add_block_cell(grid, row, 2, right_label_widget)
        _add_block_cell(grid, row, 3, _fixed_widget(right_widget, field_width))

    return left_label_widget, left_widget, right_label_widget, right_widget


def add_block_checkbox_row(grid, row, checkbox):
    _add_block_cell(
        grid,
        row,
        0,
        checkbox,
        span=BLOCK_GRID_COLUMNS,
        alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
    )
    return checkbox


def add_block_button_row(grid, row, *buttons):
    count = len(buttons)
    if count == 0:
        return ()
    if count == 1:
        placements = ((0, 4),)
    elif count == 2:
        placements = ((0, 2), (2, 2))
    elif count == 3:
        placements = ((0, 1), (1, 1), (2, 2))
    elif count == 4:
        placements = ((0, 1), (1, 1), (2, 1), (3, 1))
    else:
        raise ValueError("add_block_button_row supports at most four buttons")

    for button, (column, span) in zip(buttons, placements):
        action_button(button, expand=True)
        _add_block_cell(
            grid,
            row,
            column,
            button,
            span=span,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
    return buttons


def add_parameter_grid_row(grid, row, column, label_text, field):
    base_col = column * 2
    label = _block_label(label_text)
    _add_block_cell(grid, row, base_col, label)
    _add_block_cell(grid, row, base_col + 1, _fixed_widget(field))
    return label, field


def sweep_parameter_grid(
    horizontal_spacing=8,
    vertical_spacing=4,
    spin_width=DEFAULT_SWEEP_SPIN_WIDTH,
):
    layout = block_grid(horizontal_spacing, vertical_spacing)
    layout.setColumnMinimumWidth(1, spin_width)
    layout.setColumnMinimumWidth(2, spin_width)
    layout.setColumnMinimumWidth(3, spin_width)

    layout.addWidget(QLabel(""), 0, 0)
    for col, text in enumerate(("min", "max", "step"), start=1):
        header = QLabel(text)
        header.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(header, 0, col)
    return layout


def add_sweep_parameter_row(
    grid,
    row,
    label_text,
    min_widget,
    max_widget,
    step_widget,
    spin_width=DEFAULT_SWEEP_SPIN_WIDTH,
):
    label = _block_label(label_text)
    _add_block_cell(grid, row, 0, label)
    _add_block_cell(grid, row, 1, compact_spinbox(min_widget, spin_width))
    _add_block_cell(grid, row, 2, compact_spinbox(max_widget, spin_width))
    _add_block_cell(grid, row, 3, compact_spinbox(step_widget, spin_width))
    return label, min_widget, max_widget, step_widget
