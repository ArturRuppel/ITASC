"""Shared reusable Qt widgets for the CellFlow napari plugin."""
from __future__ import annotations

from qtpy.QtCore import Qt, QPoint, QTimer
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (
    QFrame,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class _ResizeHandle(QWidget):
    """Draggable bar at the bottom of an expanded CollapsibleSection."""

    _STYLE_NORMAL = (
        "background: #606060; border-radius: 3px; margin: 0 8px;"
    )
    _STYLE_HOVER = (
        "background: #909090; border-radius: 3px; margin: 0 8px;"
    )

    def __init__(self, scroll_area: QScrollArea, parent=None) -> None:
        super().__init__(parent)
        self._scroll = scroll_area
        self._start_y: int | None = None
        self._start_h: int | None = None
        self.setFixedHeight(8)
        self.setCursor(Qt.SizeVerCursor)
        self.setMouseTracking(True)
        self.setStyleSheet(self._STYLE_NORMAL)
        self.setToolTip("Drag to resize")

    def enterEvent(self, event) -> None:
        self.setStyleSheet(self._STYLE_HOVER)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.setStyleSheet(self._STYLE_NORMAL)
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        dot_color = QColor("#c0c0c0")
        painter.setBrush(dot_color)
        painter.setPen(Qt.NoPen)
        cx = self.width() // 2
        cy = self.height() // 2
        dot_r = 2
        spacing = 6
        for dx in (-spacing, 0, spacing):
            painter.drawEllipse(QPoint(cx + dx, cy), dot_r, dot_r)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._start_y = event.globalPos().y()
            self._start_h = self._scroll.height()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._start_y is not None:
            delta = event.globalPos().y() - self._start_y
            new_h = max(40, self._start_h + delta)
            self._scroll.setMinimumHeight(new_h)
            self._scroll.setMaximumHeight(new_h)  # pin height
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._start_y = None
        self._start_h = None
        event.accept()


class CollapsibleSection(QWidget):
    """A labelled section with a toggle button that shows/hides its inner widget.

    The header is a :class:`QToolButton` with a right/down arrow. Clicking it
    expands or collapses ``inner``. When expanded, the content is surrounded
    by a white border frame and wrapped in a scroll area with a resize handle.

    Parameters
    ----------
    title:
        Text shown in the header button.
    inner:
        The widget to show/hide.
    expanded:
        Whether the section starts expanded (default ``False``).
    """

    def __init__(
        self,
        title: str,
        inner: QWidget,
        expanded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._inner = inner
        self._base_title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(0)

        # Header toggle button
        self._toggle = QToolButton()
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setText(title)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setStyleSheet(
            "QToolButton { font-weight: bold; font-size: 10pt; border: none; "
            "padding: 2px; color: white; }"
        )
        self._toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle)

        # White-bordered frame that wraps inner content when expanded
        self._content_frame = QFrame()
        self._content_frame.setObjectName("collapsible_content")
        self._content_frame.setFrameShape(QFrame.NoFrame)
        self._content_frame.setStyleSheet(
            "QFrame#collapsible_content { border: 1px solid #666666; "
            "border-radius: 4px; margin: 0px 2px 2px 2px; }"
        )
        frame_layout = QVBoxLayout(self._content_frame)
        frame_layout.setContentsMargins(4, 4, 4, 4)
        frame_layout.setSpacing(2)

        # Scroll area wrapping inner widget
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidget(inner)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_area.setFrameShape(QFrame.NoFrame)
        frame_layout.addWidget(self._scroll_area)

        # Resize handle at bottom of frame
        self._resize_handle = _ResizeHandle(self._scroll_area)
        frame_layout.addWidget(self._resize_handle)

        self._content_frame.setVisible(expanded)
        layout.addWidget(self._content_frame)

        # Always Preferred policy — height is driven by scroll area's minimumHeight
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        if expanded:
            QTimer.singleShot(0, self._reset_natural_height)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_title(self, title: str) -> None:
        """Update the header text (e.g. to append a status badge)."""
        self._base_title = title
        self._toggle.setText(title)

    @property
    def title(self) -> str:
        return self._base_title

    @property
    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def expand(self) -> None:
        self._toggle.setChecked(True)

    def collapse(self) -> None:
        self._toggle.setChecked(False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content_frame.setVisible(checked)
        if checked:
            # Reset to natural height (unpin any previous drag)
            self._scroll_area.setMaximumHeight(16777215)
            QTimer.singleShot(0, self._reset_natural_height)
        else:
            self._scroll_area.setMinimumHeight(0)
            self._scroll_area.setMaximumHeight(16777215)
            QTimer.singleShot(0, self._notify_collapse)

    def _notify_collapse(self) -> None:
        """Propagate shrink upward after collapsing."""
        self.updateGeometry()
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, CollapsibleSection) and parent.is_expanded:
                QTimer.singleShot(0, parent._reset_natural_height)
                return
            parent.updateGeometry()
            parent = parent.parent()

    def _reset_natural_height(self) -> None:
        h = self._inner.sizeHint().height()
        if h > 10:
            self._scroll_area.setMinimumHeight(h)
            self._notify_ancestor()
        else:
            QTimer.singleShot(50, self._reset_natural_height)

    def _notify_ancestor(self) -> None:
        """Walk up the widget tree and resize the nearest CollapsibleSection ancestor."""
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, CollapsibleSection) and parent.is_expanded:
                QTimer.singleShot(0, parent._reset_natural_height)
                return
            parent = parent.parent()
