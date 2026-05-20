from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QFormLayout, QGridLayout, QLabel, QSizePolicy, QToolButton

TINY_MARGIN = 2
SECTION_MARGIN = 4
TIGHT_SPACING = 4
DEFAULT_SPIN_WIDTH = 70
DEFAULT_FIELD_SPACING = 8
DEFAULT_ROW_SPACING = 4
DEFAULT_SWEEP_SPIN_WIDTH = 62
BLOCK_GRID_COLUMNS = 4

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

# Dracula — https://draculatheme.com/contribute
DRACULA = {
    "rosewater": "#ffb86c", "flamingo":  "#ff79c6", "pink":     "#ff79c6",
    "mauve":     "#bd93f9", "red":       "#ff5555", "maroon":   "#ff5555",
    "peach":     "#ffb86c", "yellow":    "#f1fa8c", "green":    "#50fa7b",
    "teal":      "#8be9fd", "sky":       "#8be9fd", "sapphire": "#8be9fd",
    "blue":      "#6272a4", "lavender":  "#bd93f9",
}

# Tokyo Night — https://github.com/enkia/tokyo-night-vscode-theme
TOKYO_NIGHT = {
    "rosewater": "#f7768e", "flamingo":  "#f7768e", "pink":     "#f7768e",
    "mauve":     "#9d7cd8", "red":       "#f7768e", "maroon":   "#db4b4b",
    "peach":     "#ff9e64", "yellow":    "#e0af68", "green":    "#9ece6a",
    "teal":      "#73daca", "sky":       "#7dcfff", "sapphire": "#7aa2f7",
    "blue":      "#7aa2f7", "lavender":  "#bb9af7",
}

# Nord — https://www.nordtheme.com/docs/colors-and-palettes
NORD = {
    "rosewater": "#bf616a", "flamingo":  "#bf616a", "pink":     "#b48ead",
    "mauve":     "#b48ead", "red":       "#bf616a", "maroon":   "#bf616a",
    "peach":     "#d08770", "yellow":    "#ebcb8b", "green":    "#a3be8c",
    "teal":      "#8fbcbb", "sky":       "#88c0d0", "sapphire": "#81a1c1",
    "blue":      "#5e81ac", "lavender":  "#b48ead",
}

# Solarized Dark — https://ethanschoonover.com/solarized/
SOLARIZED_DARK = {
    "rosewater": "#cb4b16", "flamingo":  "#dc322f", "pink":     "#d33682",
    "mauve":     "#6c71c4", "red":       "#dc322f", "maroon":   "#cb4b16",
    "peach":     "#cb4b16", "yellow":    "#b58900", "green":    "#859900",
    "teal":      "#2aa198", "sky":       "#2aa198", "sapphire": "#268bd2",
    "blue":      "#268bd2", "lavender":  "#6c71c4",
}

# CellFlow Field Notes. Earlier second choice after Nord: a quiet custom
# palette with softer workbench-style accents for the visible stages.
FIELD_NOTES = {
    "rosewater": "#c9b7a2", "flamingo":  "#b98175", "pink":     "#b4838f",
    "mauve":     "#6f9f95", "red":       "#a05f55", "maroon":   "#8a5d4f",
    "peach":     "#b88a63", "yellow":    "#b7a56a", "green":    "#8fa77a",
    "teal":      "#6f9f95", "sky":       "#789ca8", "sapphire": "#7f8c8d",
    "blue":      "#6d8aa0", "lavender":  "#9a879d",
}

# CellFlow Museum Label. Muted, warmer, and more material than Nord while
# keeping enough separation between workflow stage accents.
MUSEUM_LABEL = {
    "rosewater": "#d7c6b2", "flamingo":  "#b98276", "pink":     "#a87582",
    "mauve":     "#8f8a6f", "red":       "#a65f4f", "maroon":   "#7f5648",
    "peach":     "#b8794f", "yellow":    "#b69b5e", "green":    "#78906f",
    "teal":      "#6f9388", "sky":       "#7d95a0", "sapphire": "#737a7c",
    "blue":      "#687f91", "lavender":  "#8b7b8f",
}

# CellFlow Dusk Gradient. Previous first choice: based on a compact
# teal-to-plum reference palette with five ordered workflow accents.
DUSK_GRADIENT = {
    "rosewater": "#b48aa4", "flamingo":  "#a3739c", "pink":     "#8c5ca6",
    "mauve":     "#483a5e", "red":       "#9b5d86", "maroon":   "#6f4790",
    "peach":     "#98d2d4", "yellow":    "#b9c5c8", "green":    "#8c5ca6",
    "teal":      "#608a9a", "sky":       "#7eabba", "sapphire": "#608a9a",
    "blue":      "#6f8eaa", "lavender":  "#6f4790",
}

# CellFlow Sunset. Current second choice: warm reference palette mapped
# left-to-right across the visible workflow accents.
SUNSET = {
    "rosewater": "#fccc73", "flamingo":  "#fd9d5d", "pink":     "#d04e6c",
    "mauve":     "#8a5687", "red":       "#fd6b5d", "maroon":   "#d04e6c",
    "peach":     "#fd9d5d", "yellow":    "#fccc73", "green":    "#fd6b5d",
    "teal":      "#d98f71", "sky":       "#c66d78", "sapphire": "#fccc73",
    "blue":      "#b66b83", "lavender":  "#d04e6c",
}

# CellFlow Retro. Saturated green-to-red reference palette mapped left-to-right
# across the visible workflow accents for comparison.
RETRO = {
    "rosewater": "#f5c783", "flamingo":  "#d98872", "pink":     "#8c0027",
    "mauve":     "#8c0027", "red":       "#dd4111", "maroon":   "#8c0027",
    "peach":     "#a1d4b1", "yellow":    "#f1a512", "green":    "#f1a512",
    "teal":      "#2baf90", "sky":       "#7fc7a8", "sapphire": "#2baf90",
    "blue":      "#4ca99a", "lavender":  "#dd4111",
}

# CellFlow Signal Archive. Previous first choice: muted red, parchment, sage,
# teal, and charcoal reference palette mapped across the workflow accents.
SIGNAL_ARCHIVE = {
    "rosewater": "#d7b2a7", "flamingo":  "#c94b4b", "pink":     "#a55b62",
    "mauve":     "#414643", "red":       "#c94b4b", "maroon":   "#8b4447",
    "peach":     "#eae3c3", "yellow":    "#eae3c3", "green":    "#9bb6a1",
    "teal":      "#3b7b7a", "sky":       "#6f9a95", "sapphire": "#c94b4b",
    "blue":      "#3b7b7a", "lavender":  "#3b7b7a",
}

# CellFlow Parent Four. Four-color supplied palette expanded with a darker
# teal companion for the fifth visible workflow accent.
PARENT_FOUR = {
    "rosewater": "#e3b4a6", "flamingo":  "#d96248", "pink":     "#b95058",
    "mauve":     "#01454f", "red":       "#d96248", "maroon":   "#9b493d",
    "peach":     "#d96248", "yellow":    "#e3cc69", "green":    "#77c8a6",
    "teal":      "#026473", "sky":       "#5ca99d", "sapphire": "#e3cc69",
    "blue":      "#2f7d86", "lavender":  "#d96248",
}

# CellFlow Viridis. Five-stop scientific colormap palette sampled from the
# inverted interior 15%-85% range to avoid the darkest and brightest endpoints.
VIRIDIS = {
    "rosewater": "#9bd93c", "flamingo":  "#38b977", "pink":     "#31668e",
    "mauve":     "#463480", "red":       "#31668e", "maroon":   "#463480",
    "peach":     "#38b977", "yellow":    "#9bd93c", "green":    "#21918c",
    "teal":      "#21918c", "sky":       "#31668e", "sapphire": "#9bd93c",
    "blue":      "#31668e", "lavender":  "#31668e",
}

# CellFlow Cividis. Current default: five-stop color-vision-friendly scientific
# colormap sampled from the inverted interior 15%-85% range.
CIVIDIS = {
    "rosewater": "#d6c35d", "flamingo":  "#a79d73", "pink":     "#555c6d",
    "mauve":     "#243c6e", "red":       "#555c6d", "maroon":   "#243c6e",
    "peach":     "#a79d73", "yellow":    "#d6c35d", "green":    "#7d7c78",
    "teal":      "#7d7c78", "sky":       "#555c6d", "sapphire": "#d6c35d",
    "blue":      "#555c6d", "lavender":  "#555c6d",
}

# ACTIVE_PALETTE = CATPPUCCIN_MOCHA
THEME_PALETTES = {
    "Cividis": CIVIDIS,
    "Viridis": VIRIDIS,
    "Signal Archive": SIGNAL_ARCHIVE,
    "Dusk Gradient": DUSK_GRADIENT,
    "Sunset": SUNSET,
    "Nord": NORD,
    "Field Notes": FIELD_NOTES,
    "Museum Label": MUSEUM_LABEL,
}
ACTIVE_THEME_NAME = "Cividis"
ACTIVE_PALETTE = THEME_PALETTES[ACTIVE_THEME_NAME]


def theme_names() -> tuple[str, ...]:
    return tuple(THEME_PALETTES)


def active_theme_name() -> str:
    return ACTIVE_THEME_NAME


def set_active_theme(name: str) -> None:
    global ACTIVE_PALETTE, ACTIVE_THEME_NAME

    ACTIVE_THEME_NAME = name
    ACTIVE_PALETTE = THEME_PALETTES[name]

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


def muted_accent(hex_str: str) -> str:
    """Return a quieter variant of an accent color for nested or compact labels."""
    c = QColor(hex_str)
    h, s, l, a = c.getHslF()
    new_s = max(0.0, s * 0.35)
    new_l = 0.55 + (l - 0.55) * 0.3
    new_l = max(0.0, min(1.0, new_l))
    c.setHslF(h, new_s, new_l, a)
    return c.name()


def muted_stage_accent(stage_key: str) -> str:
    return muted_accent(stage_accent(stage_key))


def stage_header_pill_background(stage_key: str, alpha: int = 38) -> str:
    color = QColor(muted_stage_accent(stage_key))
    red, green, blue, _ = color.getRgb()
    return f"rgba({red}, {green}, {blue}, {alpha})"


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
    if italic:
        style += " font-style: italic;"
    label.setStyleSheet(style)
    return label


def parameter_heading(label):
    label.setStyleSheet("font-weight: 600;")
    return label


def stage_header_label(label, stage_key: str, size_pt: int = 9):
    label.setProperty("cellflow_stage_key", stage_key)
    label.setProperty("cellflow_stage_header_size_pt", size_pt)
    apply_stage_header_label_style(label)
    return label


def stage_header_action_button(button: QToolButton, stage_key: str, size_px: int = 22):
    button.setProperty("cellflow_stage_key", stage_key)
    button.setProperty("cellflow_stage_header_action", True)
    button.setFixedSize(size_px, size_px)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    color = muted_stage_accent(stage_key)
    background = stage_header_pill_background(stage_key)
    button.setStyleSheet(
        "QToolButton { "
        "font-weight: bold; "
        "font-size: 9pt; "
        f"color: {color}; "
        f"background-color: {background}; "
        f"border: 1px solid {color}; "
        "border-radius: 4px; "
        "padding: 1px 4px; "
        "} "
        "QToolButton:hover { "
        f"background-color: {stage_header_pill_background(stage_key, alpha=58)}; "
        "} "
        "QToolButton:checked { "
        f"background-color: {stage_header_pill_background(stage_key, alpha=82)}; "
        f"border: 1px solid {stage_accent(stage_key)}; "
        "} "
        "QToolButton:disabled { "
        "color: palette(mid); "
        "border-color: palette(mid); "
        "background-color: transparent; "
        "}"
    )
    return button


def apply_stage_header_label_style(label):
    stage_key = label.property("cellflow_stage_key")
    if not stage_key:
        return label
    size_pt = label.property("cellflow_stage_header_size_pt") or 9
    label.setStyleSheet(
        "font-weight: bold; "
        f"font-size: {size_pt}pt; "
        f"color: {muted_stage_accent(stage_key)}; "
        f"background-color: {stage_header_pill_background(stage_key)}; "
        "border-radius: 4px; "
        "padding: 1px 6px;"
    )
    return label


def refresh_stage_header_labels(root) -> None:
    for label in root.findChildren(QLabel):
        if label.property("cellflow_stage_key"):
            apply_stage_header_label_style(label)


def danger_button(button):
    return button


def checked_success_button(button):
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


def section_grid():
    """A 4-column grid (label, field, label, field) where field columns
    stretch — so sliders, combos, and labels fill the available width and
    label columns stay aligned across all sections that share the grid."""
    layout = QGridLayout()
    layout.setHorizontalSpacing(DEFAULT_FIELD_SPACING)
    layout.setVerticalSpacing(DEFAULT_ROW_SPACING)
    layout.setColumnStretch(0, 0)
    layout.setColumnStretch(1, 1)
    layout.setColumnStretch(2, 0)
    layout.setColumnStretch(3, 1)
    return layout


def add_section_header(grid, row, widget):
    """Add a heading widget spanning all 4 columns of a section_grid."""
    grid.addWidget(widget, row, 0, 1, 4)
    return widget


def add_section_full_row(grid, row, widget):
    """Add a widget (separator, checkbox, …) spanning all 4 columns."""
    grid.addWidget(widget, row, 0, 1, 4)
    return widget


def add_section_pair_row(
    grid, row,
    left_label, left_widget,
    right_label=None, right_widget=None,
):
    """Add a row with up to two [label][widget] pairs. Widgets keep their
    natural size policy (no fixed-width wrap) so sliders/combos can stretch."""
    left_label_widget = _block_label(left_label)
    grid.addWidget(left_label_widget, row, 0)
    grid.addWidget(left_widget, row, 1)

    right_label_widget = None
    if right_widget is not None:
        right_label_widget = _block_label(right_label or "")
        grid.addWidget(right_label_widget, row, 2)
        grid.addWidget(right_widget, row, 3)
    return left_label_widget, left_widget, right_label_widget, right_widget


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
