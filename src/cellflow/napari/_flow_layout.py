"""A wrapping flow layout — lays items left→right and wraps to new rows.

Qt ships no wrapping layout out of the box; this is the canonical
``heightForWidth`` flow layout (per the Qt examples) trimmed to what the
correction workspace needs. Used to stack candidate thumbnails into rows that
reflow as the strip is resized.
"""
from __future__ import annotations

from qtpy.QtCore import QMargins, QPoint, QRect, QSize, Qt
from qtpy.QtWidgets import QLayout


class FlowLayout(QLayout):
    """Left-to-right layout that wraps its items onto new rows as needed."""

    def __init__(
        self,
        parent=None,
        *,
        margin: int = 0,
        h_spacing: int = 4,
        v_spacing: int = 4,
    ) -> None:
        super().__init__(parent)
        self._items: list = []
        self._h_spacing = int(h_spacing)
        self._v_spacing = int(v_spacing)
        self.setContentsMargins(QMargins(margin, margin, margin, margin))

    # -- QLayout plumbing ---------------------------------------------------
    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )
        return size

    # -- layout maths -------------------------------------------------------
    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x = effective.x()
        y = effective.y()
        line_height = 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h_spacing
            if next_x - self._h_spacing > effective.right() and line_height > 0:
                x = effective.x()
                y = y + line_height + self._v_spacing
                next_x = x + hint.width() + self._h_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


__all__ = ["FlowLayout"]
