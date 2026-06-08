"""Pure Qt sub-widget builders for the nucleus correction widget.

These are the self-contained construction helpers the correction widget used to
carry inline: each takes the widgets/data it needs and returns a ready ``QWidget``
(or mutates a passed-in section), with no back-reference to the host widget. The
host stays responsible for owning the resulting widgets and wiring their signals.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.ui_style import (
    stage_header_action_button,
    stage_header_label,
)
from cellflow.napari.widgets import CollapsibleSection


def set_checked_without_signal(button: QWidget, checked: bool) -> None:
    """Toggle a checkable button's state without emitting its ``toggled`` signal."""
    old = button.blockSignals(True)
    try:
        button.setChecked(checked)
    finally:
        button.blockSignals(old)


def confirm_unsaved_before_deactivate(parent: QWidget, *, save_noun: str) -> str:
    """Prompt before leaving correction mode with unsaved changes.

    ``save_noun`` names what would be saved (e.g. ``"tracked labels"``). Returns
    ``"save"``, ``"discard"``, or ``"cancel"``; the caller performs the actual
    save and clears its dirty flag.
    """
    choice = QMessageBox.question(
        parent,
        "Save correction changes?",
        (
            "Correction mode has unsaved changes. "
            f"Save {save_noun} before turning correction mode off?"
        ),
        QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        QMessageBox.Save,
    )
    if choice == QMessageBox.Cancel:
        return "cancel"
    if choice == QMessageBox.Save:
        return "save"
    return "discard"


# Width of the slim show-tab a collapsed pane shrinks to.
_PANE_STRIP_W = 24


class CollapsiblePane(QWidget):
    """Wrap a content panel with a titled header whose ✕ collapses it to a tab.

    Expanded, the pane shows a header row (title + a ✕ hide button) over the
    content. Hiding swaps in a slim full-height ``▸`` show-tab and pins the pane
    narrow; clicking the tab expands it again. ``collapsed_changed`` lets the host
    redistribute the surrounding splitter when the state flips.
    """

    collapsed_changed = Signal(bool)

    def __init__(self, content: QWidget, *, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._collapsed = False

        # Claim the full dock height: without an explicit Expanding vertical
        # policy the pane (and its stack) fall back to their content's size hint,
        # which on some platforms / packaged builds leaves the workspace panels
        # bunched at the top instead of filling the dock.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._stack = QStackedWidget(self)
        self._stack.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        outer.addWidget(self._stack)

        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(0, 0, 0, 0)
        page_lay.setSpacing(2)

        header = QWidget()
        header_lay = QHBoxLayout(header)
        header_lay.setContentsMargins(6, 2, 2, 2)
        header_lay.setSpacing(4)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-weight: bold;")
        hide_btn = QToolButton()
        hide_btn.setText("✕")
        hide_btn.setToolTip(f"Hide {title}")
        hide_btn.setAutoRaise(True)
        hide_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        hide_btn.clicked.connect(lambda: self.set_collapsed(True))
        header_lay.addWidget(title_lbl)
        header_lay.addStretch(1)
        header_lay.addWidget(hide_btn)

        page_lay.addWidget(header)
        page_lay.addWidget(content, stretch=1)
        self._stack.addWidget(page)

        # The show-tab pins the ✕/▸ buttons to the same top edge: a thin column
        # with the ▸ button at the top and a stretch beneath, rather than a
        # full-height button whose glyph floats to the vertical centre.
        strip = QWidget()
        strip_lay = QVBoxLayout(strip)
        strip_lay.setContentsMargins(2, 2, 2, 2)
        strip_lay.setSpacing(0)
        strip_btn = QToolButton()
        strip_btn.setText("▸")
        strip_btn.setToolTip(f"Show {title}")
        strip_btn.setAutoRaise(True)
        strip_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        strip_btn.clicked.connect(lambda: self.set_collapsed(False))
        strip_lay.addWidget(strip_btn, alignment=Qt.AlignTop)
        strip_lay.addStretch(1)
        self._strip = strip
        self._stack.addWidget(strip)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        self._stack.setCurrentIndex(1 if collapsed else 0)
        if collapsed:
            self.setMinimumWidth(_PANE_STRIP_W)
            self.setMaximumWidth(_PANE_STRIP_W)
        else:
            self.setMinimumWidth(0)
            self.setMaximumWidth(16777215)
        self.collapsed_changed.emit(collapsed)


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
    view_toggle_btns: tuple[QWidget, ...] = (),
    status_lbl: QWidget | None = None,
) -> tuple[QWidget, QLabel]:
    """Build the full-width correction top bar; returns ``(header, title_label)``.

    Left→right: the stage title, the activate / shortcuts / params toggles, the
    checkable view-toggle tool-buttons, a stretch, then a single one-line status
    label right-aligned at the end of the bar (the save/action status only — the
    track / validated summary lives in the tracking-overview panel). The bar
    spans the whole workspace dock — over the toolbar, gallery, and accordion.
    """
    header = QWidget(parent)
    row = QHBoxLayout(header)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)

    # The title doubles as the inactive plugin-dock entry (a stage "pill" next
    # to the on/off button) and, once correction is active, a proper workspace
    # title. The host toggles between the two looks; it starts as the pill.
    header_lbl = QLabel("Tracking Correction")
    stage_header_label(header_lbl, "nucleus")
    for button in (active_btn, shortcuts_btn, params_btn, *view_toggle_btns):
        stage_header_action_button(button, "nucleus")
    row.addWidget(header_lbl)
    # Activate first, then the reveal toggles, so correction mode can be exited
    # even while the plugin dock is hidden.
    row.addWidget(active_btn)
    row.addWidget(shortcuts_btn)
    row.addWidget(params_btn)
    if view_toggle_btns:
        row.addSpacing(8)
        for button in view_toggle_btns:
            row.addWidget(button)
    row.addStretch(1)
    if status_lbl is not None:
        row.addWidget(status_lbl)
    return header, header_lbl


# Glyphs are scaled up by this factor (relative to each button's default font)
# so the thin vertical action toolbar reads clearly at its narrow width.
_TOOLBAR_ICON_SCALE = 1.6


def _enlarge_glyph(button: QWidget) -> None:
    """Bump a tool button's glyph font, leaving any stylesheet untouched."""
    font = button.font()
    if font.pointSizeF() > 0:
        font.setPointSizeF(font.pointSizeF() * _TOOLBAR_ICON_SCALE)
    else:
        font.setPixelSize(max(1, round(font.pixelSize() * _TOOLBAR_ICON_SCALE)))
    button.setFont(font)


def build_correction_toolbar(
    parent: QWidget, button_groups: list[tuple[QWidget, ...]]
) -> QWidget:
    """Stack groups of action buttons vertically in a thin column, ruled between.

    The toolbar is the narrow leftmost panel of the workspace body splitter, so
    its buttons run top-to-bottom in a single column with bigger glyphs; group
    separators become horizontal rules. The column hugs its buttons' width so it
    stays thin while the accordion to its right absorbs the spare width.
    """
    toolbar = QWidget(parent)
    col = QVBoxLayout(toolbar)
    col.setContentsMargins(2, 2, 2, 2)
    col.setSpacing(4)

    def _sep() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    for i, group in enumerate(button_groups):
        if i > 0:
            col.addWidget(_sep())
        for b in group:
            _enlarge_glyph(b)
            col.addWidget(b)
    col.addStretch(1)
    return toolbar


# The shortcut groups, split into side-by-side columns so the reveal area below
# the top bar reads wide-and-short instead of as one tall stack.
_SHORTCUT_COLUMNS = (
    (
        (
            "Track Workflow",
            (
                ("V", "Validate selected track"),
                ("B", "Anchor selected cell at current frame"),
                ("A / D", "Extend selected track backward / forward"),
                ("Q / E", "Retrack backward / forward"),
                ("Z / C", "Swap with smaller / larger candidate fragment"),
                ("S", "Save tracked labels"),
                ("Space", "Play / stop the movie"),
            ),
        ),
    ),
    (
        (
            "Manual Labels",
            (
                ("Middle-click empty space", "Spawn new cell"),
                ("Middle-click on cell or Delete", "Erase cell"),
                ("Ctrl+Left-click", "Merge with clicked cell, or attach it to the selected track (other frame)"),
                ("Ctrl+Middle-click", "Grow / link selected track here"),
                ("Right-click variants", "Swap labels (selected cell must be in this frame)"),
                ("Shift+Left-drag", "Draw / extend cell path"),
                ("Shift+Right-drag", "Split by drawn line"),
            ),
        ),
        ("History", (("Ctrl+Z", "Undo"),)),
    ),
    (
        (
            "Selection",
            (
                ("Left-click", "Select / highlight cell"),
                ("← / →", "Previous / next thumbnail"),
                ("↑ / ↓", "Thumbnail row up / down"),
                ("Shift+↑ / ↓", "Previous / next track"),
            ),
        ),
    ),
)


def _shortcut_column(groups, footer: QWidget | None = None) -> QWidget:
    """One vertical column of shortcut groups, with an optional footer below."""
    column = QWidget()
    col_lay = QVBoxLayout(column)
    col_lay.setContentsMargins(0, 0, 0, 0)
    col_lay.setSpacing(4)

    grid_host = QWidget()
    grid = QGridLayout(grid_host)
    grid.setContentsMargins(8, 6, 8, 6)
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(2)
    row = 0
    for i, (title, items) in enumerate(groups):
        row = CorrectionWidget._add_shortcut_group(
            grid, title, list(items), start_row=row, is_first=(i == 0)
        )
    grid.setColumnStretch(1, 1)
    col_lay.addWidget(grid_host)
    if footer is not None:
        col_lay.addWidget(footer)
    col_lay.addStretch(1)
    return column


def build_shortcuts_widget(attrib_lbl: QWidget | None = None) -> QWidget:
    """Build the wide multi-column correction-shortcuts reference panel.

    The shortcut groups are arranged in side-by-side columns (so the panel is
    wide and short under the top bar). The disclaimer / attribution label, when
    supplied, rides at the bottom of the last column (alongside Selection).
    """
    group = QGroupBox("Correction shortcuts")
    outer = QVBoxLayout(group)
    # A titled group box draws its title inside the top inset, so the top margin
    # must leave room for it — zeroing all four (as the other panels do) clipped
    # the title against the first shortcut row. Reserve a title's height up top.
    title_h = group.fontMetrics().height() + 6
    outer.setContentsMargins(0, title_h, 0, 0)
    outer.setSpacing(2)

    columns = QHBoxLayout()
    columns.setContentsMargins(0, 0, 0, 0)
    columns.setSpacing(12)
    last = len(_SHORTCUT_COLUMNS) - 1
    for i, col_groups in enumerate(_SHORTCUT_COLUMNS):
        footer = attrib_lbl if i == last else None
        columns.addWidget(
            _shortcut_column(col_groups, footer), alignment=Qt.AlignTop
        )
    columns.addStretch(1)
    outer.addLayout(columns)
    return group
