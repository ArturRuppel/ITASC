"""Qt dock panel that shows a track's per-frame crops as a film strip.

This is the *view* half of the track-validation film strip: the pixels are
produced by the pure :func:`build_track_film_strip` helper, and this panel just
lays the tiles out in a horizontal, scrollable row with frame-number captions.
Clicking a tile emits :attr:`TrackFilmStripPanel.frame_clicked` so the host can
jump the viewer to that frame.

Only ``rgb_to_qimage`` is import-safe without a running QApplication; the panel
itself needs Qt up (as any widget does).
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
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
    """Horizontal, scrollable strip of a track's per-frame crops."""

    frame_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None, *, tile_px: int = _TILE_PX) -> None:
        super().__init__(parent)
        self._tile_px = int(tile_px)
        self._strip = TrackFilmStrip(tiles=())
        self._title_text = ""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._title = QLabel("No track selected")
        self._title.setContentsMargins(6, 2, 6, 2)
        outer.addWidget(self._title)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self._scroll)

        self._row_host = QWidget()
        self._row = QHBoxLayout(self._row_host)
        self._row.setContentsMargins(4, 4, 4, 4)
        self._row.setSpacing(6)
        self._row.addStretch(1)
        self._scroll.setWidget(self._row_host)

    def set_strip(self, strip: TrackFilmStrip, title: str = "") -> None:
        """Rebuild the row of tiles from ``strip`` (empty strip clears it)."""
        self._strip = strip
        self._title_text = title
        self.clear()
        if title:
            self._title.setText(title)
        if strip.is_empty():
            self._title.setText(title or "No frames for this track")
            return
        for tile in strip.tiles:
            self._row.insertWidget(self._row.count() - 1, self._make_tile(tile))

    def set_tile_size(self, tile_px: int) -> None:
        """Change the on-screen render size of each tile and re-lay the strip."""
        tile_px = max(_TILE_PX_MIN, min(int(tile_px), _TILE_PX_MAX))
        if tile_px == self._tile_px:
            return
        self._tile_px = tile_px
        self.set_strip(self._strip, self._title_text)

    def _make_tile(self, tile) -> QWidget:
        cell = QWidget()
        col = QVBoxLayout(cell)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(1)

        pixmap = QPixmap.fromImage(rgb_to_qimage(tile.rgb)).scaledToHeight(
            self._tile_px, Qt.FastTransformation
        )
        thumb = _ClickableTile(tile.frame)
        thumb.setPixmap(pixmap)
        thumb.setToolTip(f"Frame {tile.frame} — click to jump")
        thumb.clicked.connect(self.frame_clicked)
        col.addWidget(thumb, alignment=Qt.AlignHCenter)

        caption = QLabel(str(tile.frame))
        caption.setAlignment(Qt.AlignHCenter)
        col.addWidget(caption)
        return cell

    def clear(self) -> None:
        """Remove every tile (keeps the trailing stretch)."""
        while self._row.count() > 1:
            item = self._row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
