"""Qt panel that shows a track's per-frame crops as a vertical film strip.

This is the *view* half of the track film strip: the pixels are produced by the
pure :func:`build_track_film_strip` helper, and this panel lays the tiles out
**column-major, top-to-bottom** in a ``QGraphicsView`` — time runs down a column
and, when a column fills the visible height, it continues at the top of the next
column to the right (a "line break"). The arrow of time is shown by a faint frame
index at the head of each column plus a ``time ↓ →`` hint in the title, so the
reading order survives the wraps without a number under every tile.

Ctrl+wheel zooms the tiles smaller/bigger; clicking a tile emits
:attr:`TrackFilmStripPanel.frame_clicked` so the host can jump the viewer there.
Only ``rgb_to_qimage`` is import-safe without a running QApplication.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._correction_track_path import TrackFilmStrip

# Default on-screen height each (small) crop is scaled up to, with
# nearest-neighbour so the segmentation stays crisp instead of blurring.
_TILE_PX = 72
_TILE_PX_MIN = 20
_TILE_PX_MAX = 512
_ZOOM_STEP = 8       # px added/removed from the tile size per Ctrl+wheel notch
_GAP = 2             # gap between stacked crops (px)
_COL_GAP = 8         # gap between wrapped columns (px)
_HEADER_PX = 18      # room above each column for its starting-frame label
_DEFAULT_COL_HEIGHT = 600  # column height used before the view has a real size

# Border drawn around the tile for the frame the viewer is currently on.
_CURRENT_FRAME_BORDER = QColor("#ffffff")
# Marker strip for validated / anchored frames (match the overview vocabulary).
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


class TrackFilmStripPanel(QWidget):
    """A track's per-frame crops, stacked top-to-bottom and wrapped into columns."""

    frame_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None, *, tile_px: int = _TILE_PX) -> None:
        super().__init__(parent)
        self._tile_px = int(tile_px)
        self._strip = TrackFilmStrip(tiles=())
        self._title_text = ""
        self._current_frame: int | None = None
        self._frame_items: dict[int, object] = {}   # frame -> QGraphicsPixmapItem
        self._tile_rects: dict[int, QRectF] = {}
        self._border_item = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self._title = QLabel("No track selected")
        self._title.setContentsMargins(6, 2, 6, 2)
        outer.addWidget(self._title)

        self._scene = QGraphicsScene(self)
        self._view = _StripView(self)
        outer.addWidget(self._view)

    def set_strip(self, strip: TrackFilmStrip, title: str = "") -> None:
        """Show ``strip``'s tiles (an empty strip clears the panel)."""
        self._strip = strip
        self._title_text = title
        if strip.is_empty():
            self.clear()
            self._title.setText(title or "No frames for this track")
            return
        self._relayout()

    def set_tile_size(self, tile_px: int) -> None:
        """Change the on-screen size of each tile and re-lay the strip."""
        tile_px = max(_TILE_PX_MIN, min(int(tile_px), _TILE_PX_MAX))
        if tile_px == self._tile_px:
            return
        self._tile_px = tile_px
        if not self._strip.is_empty():
            self._relayout()

    def set_current_frame(self, frame: int | None) -> None:
        """Highlight the tile for ``frame`` (live, without rebuilding the strip)."""
        self._current_frame = None if frame is None else int(frame)
        self._apply_current_border()

    # -- layout -------------------------------------------------------------
    def _relayout(self) -> None:
        """Place every tile column-major, wrapping when a column fills the view."""
        self._scene.clear()
        self._frame_items = {}
        self._tile_rects = {}
        self._border_item = None
        if self._strip.is_empty():
            return

        viewport_h = self._view.viewport().height()
        col_height = (
            viewport_h if viewport_h >= self._tile_px + _HEADER_PX
            else _DEFAULT_COL_HEIGHT
        )
        x = 0.0
        y = float(_HEADER_PX)
        col_w = 0.0
        col_first = True
        for tile in self._strip.tiles:
            pm = self._tile_pixmap(tile)
            w, h = pm.width(), pm.height()
            if not col_first and y + h > col_height:
                x += col_w + _COL_GAP          # wrap into the next column
                y = float(_HEADER_PX)
                col_w = 0.0
                col_first = True
            if col_first:
                self._add_column_header(x, tile.frame)
                col_first = False
            item = self._scene.addPixmap(pm)
            item.setOffset(x, y)
            item.setData(0, int(tile.frame))
            item.setToolTip(self._tile_tooltip(tile))
            self._frame_items[int(tile.frame)] = item
            self._tile_rects[int(tile.frame)] = QRectF(x, y, w, h)
            y += h + _GAP
            col_w = max(col_w, w)
        self._scene.setSceneRect(self._scene.itemsBoundingRect())
        self._title.setText(self._title_with_hint())
        self._apply_current_border()

    def _add_column_header(self, x: float, frame: int) -> None:
        """A faint starting-frame index at the head of a column."""
        text = self._scene.addText(str(int(frame)))
        text.setDefaultTextColor(QColor(160, 160, 160))
        font = text.font()
        font.setPointSizeF(max(6.0, _HEADER_PX - 4))
        text.setFont(font)
        text.setPos(x - 2, -8)

    def _tile_pixmap(self, tile) -> QPixmap:
        pm = QPixmap.fromImage(rgb_to_qimage(tile.rgb)).scaledToHeight(
            self._tile_px, Qt.FastTransformation
        )
        self._draw_marker_strips(
            pm,
            validated=getattr(tile, "validated", False),
            anchored=getattr(tile, "anchored", False),
        )
        return pm

    def _apply_current_border(self) -> None:
        if self._border_item is not None:
            self._scene.removeItem(self._border_item)
            self._border_item = None
        if self._current_frame is None:
            return
        rect = self._tile_rects.get(self._current_frame)
        if rect is None:
            return
        pen = QPen(_CURRENT_FRAME_BORDER, 2)
        self._border_item = self._scene.addRect(rect, pen)

    def _title_with_hint(self) -> str:
        base = self._title_text or f"{len(self._strip.tiles)} frame(s)"
        return f"{base}  ·  time ↓ →"

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
        self._scene.clear()
        self._frame_items = {}
        self._tile_rects = {}
        self._border_item = None


class _StripView(QGraphicsView):
    """Scrollable strip view: Ctrl+wheel zooms tiles, a click reports its frame."""

    def __init__(self, panel: TrackFilmStripPanel) -> None:
        super().__init__(panel._scene, panel)
        self._panel = panel
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.modifiers() & Qt.ControlModifier:
            step = _ZOOM_STEP if event.angleDelta().y() > 0 else -_ZOOM_STEP
            self._panel.set_tile_size(self._panel._tile_px + step)
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        # Re-wrap so columns track the available height.
        if not self._panel._strip.is_empty():
            self._panel._relayout()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mouseReleaseEvent(event)
        item = self.itemAt(event.pos())
        frame = None if item is None else item.data(0)
        if frame is not None:
            self._panel.frame_clicked.emit(int(frame))
