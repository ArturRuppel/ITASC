"""Docked table of ranked correction errors — the worklist *view*.

Rows are :class:`~cellflow.segmentation.error_scan.CellError` entries (produced
by :func:`~cellflow.segmentation.error_scan.scan_errors`), sorted worst-first
and colour-graded by score. Activating a row emits :attr:`entry_activated` with
``(frame, cell_id)`` so the host can jump the viewer and select the cell;
resolved rows are struck out and dropped to the bottom so the user can work down
the list without losing their place.
"""
from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cellflow.segmentation.error_scan import CellError

# Score 0 → green-ish, 1 → red: a simple two-stop ramp for the score cell.
_LOW = (90, 160, 90)
_HIGH = (200, 70, 70)


def _score_color(score: float) -> QColor:
    s = max(0.0, min(1.0, float(score)))
    r = int(_LOW[0] + (_HIGH[0] - _LOW[0]) * s)
    g = int(_LOW[1] + (_HIGH[1] - _LOW[1]) * s)
    b = int(_LOW[2] + (_HIGH[2] - _LOW[2]) * s)
    return QColor(r, g, b)


class ErrorWorklistPanel(QWidget):
    """A sortable, click-to-navigate table of flagged correction errors."""

    entry_activated = Signal(int, int)  # (frame, cell_id)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._resolved: set[tuple[int, int]] = set()
        self._entries: list[CellError] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self._title = QLabel("No issues")
        self._title.setContentsMargins(6, 2, 6, 2)
        outer.addWidget(self._title)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Frame", "Cell", "Score", "Reason"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setWordWrap(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.cellActivated.connect(self._on_cell_activated)
        self._table.cellClicked.connect(self._on_cell_activated)
        outer.addWidget(self._table)

    def set_entries(self, entries: list[CellError]) -> None:
        """Replace the table contents (keeps resolved marks for matching keys)."""
        self._entries = list(entries)
        live = [e for e in self._entries if (e.t, e.cell_id) not in self._resolved]
        self._title.setText(
            f"{len(live)} issue{'s' if len(live) != 1 else ''}"
            + (f" ({len(self._resolved)} resolved)" if self._resolved else "")
        )
        self._table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            self._fill_row(row, entry)

    def mark_resolved(self, frame: int, cell_id: int) -> None:
        """Strike a row out and re-render (e.g. after the user edits that cell)."""
        self._resolved.add((int(frame), int(cell_id)))
        self.set_entries(self._entries)

    def clear(self) -> None:
        self._resolved.clear()
        self._entries = []
        self._table.setRowCount(0)
        self._title.setText("No issues")

    def _fill_row(self, row: int, entry: CellError) -> None:
        resolved = (entry.t, entry.cell_id) in self._resolved
        cells = [
            QTableWidgetItem(str(entry.t)),
            QTableWidgetItem(str(entry.cell_id)),
            QTableWidgetItem(f"{entry.score:.2f}"),
            QTableWidgetItem(", ".join(entry.reasons)),
        ]
        cells[2].setForeground(_score_color(entry.score))
        for item in cells:
            if resolved:
                font = item.font()
                font.setStrikeOut(True)
                item.setFont(font)
                item.setForeground(QColor(130, 130, 130))
        for col, item in enumerate(cells):
            self._table.setItem(row, col, item)

    def _on_cell_activated(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._entries):
            entry = self._entries[row]
            self.entry_activated.emit(int(entry.t), int(entry.cell_id))


__all__ = ["ErrorWorklistPanel"]
