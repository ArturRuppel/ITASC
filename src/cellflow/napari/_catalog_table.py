"""The catalog as stacked custom-widget rows (replaces the ``QTableWidget``).

A ``QTableWidget`` can only render text cells; the per-position status rail needs
a live widget per row. So the catalog is rebuilt as :class:`CatalogTable` — a fixed
header of column titles over a bounded, scrolling column of :class:`_CatalogRow`
widgets, each carrying its text cells plus a :class:`~cellflow.napari._status_rail.StatusRail`.

The studio owns the record model and hands the table a flat list of
:class:`CatalogRowSpec`s (record rows interleaved with bold group separators); the
table owns only presentation + selection. Selection is reported as *record*
indices (separators never select), so scope semantics are unchanged from the old
table. This is also the substrate the eventual batch driver reuses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._status_rail import StatusRail

#: Text columns: (record key label, header title, fixed pixel width). Kept fixed
#: so the header titles line up over every row's cells.
_TEXT_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("condition", "Condition", 90),
    ("date", "Date", 78),
    ("id", "ID", 90),
    ("inputs", "Inputs", 46),
    ("notes", "Notes", 130),
)
_STATUS_TITLE = "Status"
_STATUS_WIDTH = 78
_MAX_TABLE_HEIGHT = 300


@dataclass
class CatalogRowSpec:
    """One display row: a record row (``record_index`` set) or a separator.

    * Record row — ``record_index`` is the record's index; ``values`` are the text
      cells aligned with :data:`_TEXT_COLUMNS`; ``status`` feeds the rail.
    * Separator — ``record_index`` is ``None``; ``caption`` is the bold group label.
    """

    record_index: int | None
    values: tuple[str, ...] = ()
    status: dict[str, str] = field(default_factory=dict)
    caption: str = ""


def _cell(text: str, width: int) -> QLabel:
    label = QLabel(text)
    label.setFixedWidth(width)
    label.setTextInteractionFlags(Qt.NoTextInteraction)
    # Elide long text to keep columns aligned; the full value is the tooltip.
    if text:
        label.setToolTip(text)
    return label


class _CatalogRow(QFrame):
    """A single selectable catalog record row: text cells + a status rail."""

    clicked = Signal(int, object)  # (record_index, keyboard modifiers)

    def __init__(self, spec: CatalogRowSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.record_index = spec.record_index
        self._selected = False
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 1, 3, 1)
        layout.setSpacing(4)
        values = list(spec.values) + [""] * (len(_TEXT_COLUMNS) - len(spec.values))
        for (_, _, width), value in zip(_TEXT_COLUMNS, values):
            layout.addWidget(_cell(str(value), width))
        self.rail = StatusRail()
        self.rail.setFixedWidth(_STATUS_WIDTH)
        self.rail.set_status(spec.status)
        layout.addWidget(self.rail)
        layout.addStretch(1)
        self._apply_style()

    def set_selected(self, selected: bool) -> None:
        if selected != self._selected:
            self._selected = selected
            self._apply_style()

    def _apply_style(self) -> None:
        if self._selected:
            self.setStyleSheet(
                "_CatalogRow { background-color: rgba(90, 140, 220, 70); "
                "border-left: 2px solid #5a8cdc; }"
            )
        else:
            self.setStyleSheet("_CatalogRow { border-left: 2px solid transparent; }")

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.clicked.emit(self.record_index, event.modifiers())
        super().mousePressEvent(event)


class _SeparatorRow(QFrame):
    """A bold, non-selectable group-separator row spanning the full width."""

    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 1)
        layout.setSpacing(0)
        label = QLabel(caption)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        layout.addWidget(label)
        layout.addStretch(1)


class CatalogTable(QWidget):
    """Stacked custom-row catalog with a per-position status rail.

    Public surface used by the studio: :meth:`set_rows`, :meth:`row_count`,
    :attr:`row_to_record`, :meth:`selected_record_indices`, :meth:`select_records`,
    :meth:`select_all`, :meth:`clear_selection`, and the :attr:`selectionChanged`
    signal.
    """

    selectionChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: Display-row → record index (``None`` for separators).
        self.row_to_record: list[int | None] = []
        #: Record indices in display order (excludes separators) — range anchor space.
        self._data_order: list[int] = []
        #: Currently selected record indices.
        self._selected: set[int] = set()
        #: Range-selection anchor (a record index) for Shift-click.
        self._anchor: int | None = None
        #: record index → its row widget, for selection restyling.
        self._row_widgets: dict[int, _CatalogRow] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = self._build_header()
        outer.addWidget(self._header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMaximumHeight(_MAX_TABLE_HEIGHT)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(0)
        self._body_layout.setAlignment(Qt.AlignTop)
        self._scroll.setWidget(self._body)
        outer.addWidget(self._scroll)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setFrameShape(QFrame.NoFrame)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(3, 1, 3, 1)
        layout.setSpacing(4)
        for _, title, width in _TEXT_COLUMNS:
            label = QLabel(title)
            label.setFixedWidth(width)
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            layout.addWidget(label)
        status = QLabel(_STATUS_TITLE)
        status.setFixedWidth(_STATUS_WIDTH)
        font = status.font()
        font.setBold(True)
        status.setFont(font)
        layout.addWidget(status)
        layout.addStretch(1)
        return header

    # ------------------------------------------------------------------- rebuild
    def set_rows(self, specs: list[CatalogRowSpec]) -> None:
        """Rebuild every display row from *specs*; prune any now-absent selection."""
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.row_to_record = []
        self._data_order = []
        self._row_widgets = {}

        for spec in specs:
            if spec.record_index is None:
                self._body_layout.addWidget(_SeparatorRow(spec.caption))
                self.row_to_record.append(None)
                continue
            row = _CatalogRow(spec)
            row.clicked.connect(self._on_row_clicked)
            self._body_layout.addWidget(row)
            self.row_to_record.append(spec.record_index)
            self._data_order.append(spec.record_index)
            self._row_widgets[spec.record_index] = row

        valid = set(self._data_order)
        pruned = self._selected & valid
        if pruned != self._selected:
            self._selected = pruned
            if self._anchor not in valid:
                self._anchor = None
        self._restyle_rows()

    # ---------------------------------------------------------------- selection
    def selected_record_indices(self) -> list[int]:
        return sorted(self._selected)

    def select_records(self, indices: list[int]) -> None:
        """Replace the selection with *indices* (separators/absent ones dropped)."""
        valid = set(self._data_order)
        self._set_selection({i for i in indices if i in valid})
        self._anchor = indices[-1] if indices else None

    def select_all(self) -> None:
        self._set_selection(set(self._data_order))

    def clear_selection(self) -> None:
        self._set_selection(set())
        self._anchor = None

    def row_count(self) -> int:
        return len(self.row_to_record)

    def record_row_widgets(self) -> list[_CatalogRow]:
        """Row widgets in display order (record rows only) — for tests / styling."""
        return [self._row_widgets[i] for i in self._data_order]

    def _on_row_clicked(self, record_index: int, modifiers) -> None:
        ctrl = bool(modifiers & Qt.ControlModifier)
        shift = bool(modifiers & Qt.ShiftModifier)
        if shift and self._anchor is not None and self._anchor in self._data_order:
            lo = self._data_order.index(self._anchor)
            hi = self._data_order.index(record_index)
            lo, hi = min(lo, hi), max(lo, hi)
            self._set_selection(set(self._data_order[lo : hi + 1]))
        elif ctrl:
            updated = set(self._selected)
            updated ^= {record_index}
            self._set_selection(updated)
            self._anchor = record_index
        else:
            self._set_selection({record_index})
            self._anchor = record_index

    def _set_selection(self, selected: set[int]) -> None:
        if selected == self._selected:
            return
        self._selected = selected
        self._restyle_rows()
        self.selectionChanged.emit()

    def _restyle_rows(self) -> None:
        for index, row in self._row_widgets.items():
            row.set_selected(index in self._selected)
