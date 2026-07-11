from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

TINY_MARGIN = 2
SECTION_MARGIN = 4
TIGHT_SPACING = 4
DEFAULT_FIELD_SPACING = 8
DEFAULT_ROW_SPACING = 4
BLOCK_GRID_COLUMNS = 4

# ── Theme palette ────────────────────────────────────────────────────────
# Accent palettes keyed by the same color names. THEME_PALETTES selects the
# active set; ACTIVE_THEME_NAME picks the default. Call sites reference accents
# via stage_accent() so theme changes propagate automatically.

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

# CellFlow Signal Archive. Previous first choice: muted red, parchment, sage,
# teal, and charcoal reference palette mapped across the workflow accents.
SIGNAL_ARCHIVE = {
    "rosewater": "#d7b2a7", "flamingo":  "#c94b4b", "pink":     "#a55b62",
    "mauve":     "#414643", "red":       "#c94b4b", "maroon":   "#8b4447",
    "peach":     "#eae3c3", "yellow":    "#eae3c3", "green":    "#9bb6a1",
    "teal":      "#3b7b7a", "sky":       "#6f9a95", "sapphire": "#c94b4b",
    "blue":      "#3b7b7a", "lavender":  "#3b7b7a",
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
    "aggregate":        "teal",
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


def stage_header_disabled_action_color(stage_key: str) -> str:
    color = QColor(muted_stage_accent(stage_key))
    h, s, l, a = color.getHslF()
    color.setHslF(h, s * 0.55, max(0.0, l * 0.62), a)
    return color.name()


# ── Designed-surface tokens (napariTFM ExperimentsList parity) ────────────
# Theme-agnostic so the experiments panel sits on any host background: selected
# rows are a translucent white "lift", text uses a grey ramp, inputs are pills.
COMPACT_SPACING = 4
TEXT_BRIGHT = "#e6edf3"
TEXT_MID = "#aeb6c0"
TEXT_DIM = "#6b7484"
ROW_LIFT_BG = "rgba(255, 255, 255, 13)"   # a selected/raised row surface
HAIRLINE = "rgba(255, 255, 255, 18)"

# Experiment-row overall-status word -> color (amber running, green done, dim queued).
EXPERIMENT_STATUS_COLORS = {
    "run": "#e3b341",
    "done": "#3fb950",
    "queued": TEXT_DIM,
}


def experiment_status_color(label: str) -> str:
    """Color for an experiment row's overall-status word (run/done/queued)."""
    return EXPERIMENT_STATUS_COLORS.get(label, TEXT_DIM)


def experiment_name_color(selected: bool) -> str:
    """Brighten the active row's name; dim the rest."""
    return TEXT_BRIGHT if selected else TEXT_MID


def experiment_row_style(selected: bool, accent: str) -> str:
    """Row container style: a raised, accent-bordered surface when selected."""
    if not selected:
        return (
            "QWidget#experiment_row { background: transparent; "
            "border: 1px solid transparent; border-radius: 8px; }"
        )
    r, g, b, _ = QColor(accent).getRgb()
    return (
        "QWidget#experiment_row { "
        f"background: {ROW_LIFT_BG}; "
        f"border: 1px solid rgba({r}, {g}, {b}, 130); "
        "border-radius: 8px; }"
    )


def action_button_style() -> str:
    """Give a text QToolButton real button chrome (fill, border, rounded,
    hover/pressed feedback) so list actions don't read as flat status labels."""
    return (
        "QToolButton { "
        "background: rgba(255, 255, 255, 10); "
        f"border: 1px solid {HAIRLINE}; border-radius: 6px; "
        f"padding: 4px 12px; color: {TEXT_BRIGHT}; }} "
        "QToolButton:hover { background: rgba(255, 255, 255, 22); "
        "border-color: rgba(255, 255, 255, 46); } "
        "QToolButton:pressed { background: rgba(255, 255, 255, 32); } "
        "QToolButton:disabled { "
        f"color: {TEXT_DIM}; background: rgba(255, 255, 255, 4); "
        "border-color: rgba(255, 255, 255, 10); }"
    )


def mono_input_style() -> str:
    """Themed pill style for a QLineEdit, so config fields aren't raw Qt."""
    return (
        "QLineEdit { "
        "background: rgba(255, 255, 255, 8); "
        f"border: 1px solid {HAIRLINE}; border-radius: 6px; "
        f"padding: 3px 7px; color: {TEXT_BRIGHT}; }} "
        "QLineEdit:focus { border-color: rgba(255, 255, 255, 38); }"
    )


def _fixed_widget(widget, width=None):
    if width is not None:
        widget.setMaximumWidth(width)
    widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return widget


_DISABLED_PUSH_BUTTON_STYLE = (
    "QPushButton:disabled { "
    "color: palette(mid); "
    "background-color: rgba(127, 124, 120, 42); "
    "border: 1px solid rgba(127, 124, 120, 72); "
    "border-radius: 4px; "
    "}"
)


def _append_button_style(button, style: str):
    current = button.styleSheet().strip()
    if style in current:
        return button
    button.setStyleSheet(f"{current} {style}".strip())
    return button


def _disabled_push_button(button):
    if isinstance(button, QPushButton):
        _append_button_style(button, _DISABLED_PUSH_BUTTON_STYLE)
    return button


def action_button(button, expand=False):
    horizontal_policy = (
        QSizePolicy.Policy.Expanding if expand else QSizePolicy.Policy.Fixed
    )
    button.setSizePolicy(horizontal_policy, QSizePolicy.Policy.Fixed)
    _disabled_push_button(button)
    return button


def icon_button(button, width=24, height=None):
    button.setFixedWidth(width)
    if height is not None:
        button.setFixedHeight(height)
    _disabled_push_button(button)
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
    disabled_color = stage_header_disabled_action_color(stage_key)
    background = stage_header_pill_background(stage_key)
    button.setStyleSheet(
        "QToolButton { "
        "font-weight: bold; "
        "font-size: 9pt; "
        f"color: {color}; "
        f"background-color: {background}; "
        "border: none; "
        "border-radius: 4px; "
        "padding: 0; "
        "margin: 0; "
        "text-align: center; "
        "} "
        "QToolButton:hover { "
        f"background-color: {stage_header_pill_background(stage_key, alpha=58)}; "
        "} "
        "QToolButton:checked { "
        f"background-color: {stage_header_pill_background(stage_key, alpha=82)}; "
        "} "
        "QToolButton:disabled { "
        f"color: {disabled_color}; "
        f"background-color: {stage_header_pill_background(stage_key, alpha=28)}; "
        "} "
        "QToolButton:disabled:checked { "
        f"color: {disabled_color}; "
        f"background-color: {stage_header_pill_background(stage_key, alpha=44)}; "
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
    _disabled_push_button(button)
    return button


def checked_success_button(button):
    _disabled_push_button(button)
    return button


def block_grid(horizontal_spacing=8, vertical_spacing=4):
    layout = QGridLayout()
    layout.setHorizontalSpacing(horizontal_spacing)
    layout.setVerticalSpacing(vertical_spacing)
    for col in range(BLOCK_GRID_COLUMNS):
        layout.setColumnStretch(col, 0)
    return layout


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
    _add_section_pair_cell(grid, row, 0, left_label_widget, left_widget)

    right_label_widget = None
    if right_widget is not None:
        right_label_widget = _block_label(right_label or "")
        _add_section_pair_cell(grid, row, 2, right_label_widget, right_widget)
    return left_label_widget, left_widget, right_label_widget, right_widget


def _add_section_pair_cell(grid, row, column, label_widget, widget):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(1)
    label_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
    layout.addWidget(label_widget)
    layout.addWidget(widget)
    grid.addWidget(container, row, column, 1, 2)
    return container


def _block_label(text):
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return label


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
    _add_block_pair_cell(grid, row, 0, left_label_widget, left_widget, field_width)

    right_label_widget = None
    if right_widget is not None:
        right_label_widget = _block_label(right_label or "")
        _add_block_pair_cell(grid, row, 2, right_label_widget, right_widget, field_width)

    return left_label_widget, left_widget, right_label_widget, right_widget


def _add_block_pair_cell(grid, row, column, label_widget, widget, field_width):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(1)
    label_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
    layout.addWidget(label_widget)
    field = widget if widget.property("cellflow_stack_section_label") else _fixed_widget(widget, field_width)
    layout.addWidget(field)
    grid.addWidget(container, row, column, 1, 2)
    return container
