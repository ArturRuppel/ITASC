"""Qt dock panel that shows a track's per-frame crops as a film strip.

This is the *view* half of the track-validation film strip: the pixels are
produced by the pure :func:`build_track_film_strip` helper, and this panel just
lays the tiles out in a wrapping grid with frame-number captions — tiles flow
left-to-right and wrap onto a new line when they don't fit the panel width.
Clicking a tile emits :attr:`TrackFilmStripPanel.frame_clicked` so the host can
jump the viewer to that frame.

Only ``rgb_to_qimage`` is import-safe without a running QApplication; the panel
itself needs Qt up (as any widget does).
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPoint, QRect, QSize, Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPixmap
from qtpy.QtWidgets import (
    QLabel,
    QLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._correction_track_path import TrackFilmStrip

# Default on-screen height each (small) crop is scaled up to, with
# nearest-neighbour so the segmentation stays crisp instead of blurring.
_TILE_PX = 96
_TILE_PX_MIN = 20
_TILE_PX_MAX = 512

# Border drawn around the tile for the frame the viewer is currently on.
_CURRENT_FRAME_BORDER = "#ffffff"
# Top marker strips for validated / anchored frames (match the in-canvas
# validated- and anchor-overlay colours).
_VALIDATED_STRIP_COLOR = "#00ff00"
_ANCHOR_STRIP_COLOR = "#b39400"
_MARKER_STRIP_PX = 4


def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    """Convert an ``(h, w, 3)`` uint8 array to an owned RGB888 ``QImage``.

    The returned image owns its buffer (``.copy()``), so the source array is
    free to be garbage-collected.
    """
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (h, w, 3) uint8, got {arr.shape}")
    height, width, _ = arr.shape
    image = QImage(arr.data, width, height, 3 * width, QImage.Format_RGB888)
    return image.copy()


class _FlowLayout(QLayout):
    """A layout that arranges children left-to-right, wrapping onto new lines.

    Qt ships no wrapping layout, so this is the canonical Qt ``FlowLayout``
    example adapted to qtpy: it lets the film strip break onto extra rows when
    the tiles don't fit the available width, instead of scrolling sideways.
    """

    def __init__(self, parent: QWidget | None = None, *, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list = []
        self.setSpacing(spacing)

    def addItem(self, item) -> None:  # noqa: N802 (Qt override)
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 (Qt override)
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):  # noqa: N802 (Qt override)
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802 (Qt override)
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt override)
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt override)
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 (Qt override)
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 (Qt override)
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )
        return size

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x = effective.x()
        y = effective.y()
        line_height = 0
        spacing = self.spacing()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + spacing
            if next_x - spacing > effective.right() and line_height > 0:
                x = effective.x()
                y = y + line_height + spacing
                next_x = x + hint.width() + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


class _ClickableTile(QLabel):
    """A pixmap label that reports its frame index when clicked."""

    clicked = Signal(int)

    def __init__(self, frame: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame = frame
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._frame)
        super().mousePressEvent(event)


class TrackFilmStripPanel(QWidget):
    """Wrapping grid of a track's per-frame crops (breaks onto new lines)."""

    frame_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None, *, tile_px: int = _TILE_PX) -> None:
        super().__init__(parent)
        self._tile_px = int(tile_px)
        self._strip = TrackFilmStrip(tiles=())
        self._title_text = ""
        self._current_frame: int | None = None
        self._tile_cells: dict[int, list[QWidget]] = {}
        # Thumbs in tile order, parallel to ``self._strip.tiles``; lets a swap
        # repaint the existing tiles in place instead of tearing the whole row
        # down and rebuilding it on every keypress.
        self._thumbs: list[_ClickableTile] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._title = QLabel("No track selected")
        self._title.setContentsMargins(6, 2, 6, 2)
        outer.addWidget(self._title)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer.addWidget(self._scroll)

        self._row_host = QWidget()
        self._row = _FlowLayout(self._row_host, spacing=6)
        self._row.setContentsMargins(4, 4, 4, 4)
        self._scroll.setWidget(self._row_host)

    def set_strip(self, strip: TrackFilmStrip, title: str = "") -> None:
        """Show ``strip``'s tiles (empty strip clears it).

        When the new strip covers the same frames as the one on screen (the
        common case while swapping a single track), the existing tiles are
        repainted in place. Only a change in the frame set rebuilds the row, so
        repeated swaps no longer churn the dock's widgets — which is what made
        transient popups flash on every keypress.
        """
        previous = self._strip
        self._strip = strip
        self._title_text = title
        if title:
            self._title.setText(title)
        if strip.is_empty():
            self.clear()
            self._title.setText(title or "No frames for this track")
            return
        if self._same_frames(previous, strip):
            for thumb, tile in zip(self._thumbs, strip.tiles):
                self._render_thumb(thumb, tile)
            return
        self.clear()
        for tile in strip.tiles:
            self._row.addWidget(self._make_tile(tile))

    @staticmethod
    def _same_frames(a: TrackFilmStrip, b: TrackFilmStrip) -> bool:
        """True when both strips list the same frames in the same order."""
        return tuple(t.frame for t in a.tiles) == tuple(t.frame for t in b.tiles)

    def set_tile_size(self, tile_px: int) -> None:
        """Change the on-screen render size of each tile and re-lay the strip."""
        tile_px = max(_TILE_PX_MIN, min(int(tile_px), _TILE_PX_MAX))
        if tile_px == self._tile_px:
            return
        self._tile_px = tile_px
        self.set_strip(self._strip, self._title_text)

    def set_current_frame(self, frame: int | None) -> None:
        """Highlight the tile for ``frame`` (live, without rebuilding the strip)."""
        frame = None if frame is None else int(frame)
        if frame == self._current_frame:
            return
        previous = self._current_frame
        self._current_frame = frame
        for value in (previous, frame):
            if value is None:
                continue
            for cell in self._tile_cells.get(value, ()):
                self._apply_border(cell, value == frame)

    def _make_tile(self, tile) -> QWidget:
        cell = QWidget()
        cell.setObjectName("filmTileCell")
        col = QVBoxLayout(cell)
        col.setContentsMargins(2, 2, 2, 2)
        col.setSpacing(1)

        thumb = _ClickableTile(tile.frame)
        self._render_thumb(thumb, tile)
        thumb.clicked.connect(self.frame_clicked)
        col.addWidget(thumb, alignment=Qt.AlignHCenter)
        self._thumbs.append(thumb)

        caption = QLabel(str(tile.frame))
        caption.setAlignment(Qt.AlignHCenter)
        col.addWidget(caption)

        self._tile_cells.setdefault(int(tile.frame), []).append(cell)
        self._apply_border(cell, int(tile.frame) == self._current_frame)
        return cell

    def _render_thumb(self, thumb: "_ClickableTile", tile) -> None:
        """Paint ``tile``'s crop (plus validated/anchored markers) onto ``thumb``."""
        pixmap = QPixmap.fromImage(rgb_to_qimage(tile.rgb)).scaledToHeight(
            self._tile_px, Qt.FastTransformation
        )
        self._draw_marker_strips(
            pixmap,
            validated=getattr(tile, "validated", False),
            anchored=getattr(tile, "anchored", False),
        )
        thumb.setPixmap(pixmap)
        thumb.setToolTip(self._tile_tooltip(tile))

    @staticmethod
    def _tile_tooltip(tile) -> str:
        tags = []
        if getattr(tile, "validated", False):
            tags.append("validated")
        if getattr(tile, "anchored", False):
            tags.append("anchored")
        suffix = f" ({', '.join(tags)})" if tags else ""
        return f"Frame {tile.frame}{suffix} — click to jump"

    @staticmethod
    def _apply_border(cell: QWidget, is_current: bool) -> None:
        # Reserve the border width even when off so the layout never shifts.
        color = _CURRENT_FRAME_BORDER if is_current else "transparent"
        cell.setStyleSheet(
            f"QWidget#filmTileCell {{ border: 2px solid {color}; border-radius: 2px; }}"
        )

    @staticmethod
    def _draw_marker_strips(pixmap: QPixmap, *, validated: bool, anchored: bool) -> None:
        if not (validated or anchored):
            return
        painter = QPainter(pixmap)
        width = pixmap.width()
        height = _MARKER_STRIP_PX
        if validated and anchored:
            half = width // 2
            painter.fillRect(0, 0, half, height, QColor(_VALIDATED_STRIP_COLOR))
            painter.fillRect(half, 0, width - half, height, QColor(_ANCHOR_STRIP_COLOR))
        elif validated:
            painter.fillRect(0, 0, width, height, QColor(_VALIDATED_STRIP_COLOR))
        else:
            painter.fillRect(0, 0, width, height, QColor(_ANCHOR_STRIP_COLOR))
        painter.end()

    def clear(self) -> None:
        """Remove every tile from the strip."""
        self._tile_cells = {}
        self._thumbs = []
        while self._row.count():
            item = self._row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
