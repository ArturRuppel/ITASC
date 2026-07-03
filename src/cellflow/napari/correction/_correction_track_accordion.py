"""Unified correction accordion — track bars that expand inline to thumbnails.

This merges what used to be two separate panels (the swimlane *overview* and the
per-track *film strip*) into one ``QGraphicsView`` + ``QGraphicsScene``. Each
track gets one thin **bar** row, laid top-down on a shared global time axis; the
**selected** track keeps its bar as a header and grows a **wrapped thumbnail
band** of that track's per-frame crops directly beneath it (one track expanded at
a time, driven by selection).

Geometry is width-derived: ``cell_w = (viewport_width − left_gutter) / n_frames``
is recomputed on every resize, so the whole global time axis always fits the
panel width with no horizontal scroll. A single vertical guide marks the current
frame across every row; a horizontal cursor marks the selected row.

Interaction:

* click a **bar** → :attr:`node_activated` ``(frame, cell_id)`` (snapping into the
  nearest present frame when the click lands in a gap);
* click a **thumbnail** → :attr:`frame_clicked` ``(frame)``;
* **Ctrl+wheel** is region-aware (hit-test the cursor): over the expanded
  thumbnail band it resizes the **tiles**; over a bar / elsewhere it changes the
  **bar height** only (``cell_w`` is width-derived, so bars never widen);
* **plain wheel** scrolls the panel vertically through the track list.

The panel is a pure renderer: the controller hands it ready ``LaneView`` structs
and the selected track's :class:`~cellflow.napari.correction._correction_track_path.TrackFilmStrip`.
Only ``rgb_to_qimage`` is import-safe without a running QApplication.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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

from cellflow.core.lineage import _segments_from_frames
from cellflow.napari.correction._correction_track_path import TrackFilmStrip

_CLICK_SLOP = 6  # max drag (px) still treated as a click, not a scroll

# Bar geometry. ``lane_h`` is Ctrl+wheel adjustable (bar height only); ``cell_w``
# is always derived from the viewport width so the time axis fits exactly.
_LANE_H = 14.0
_LANE_H_MIN = 4.0
_LANE_H_MAX = 80.0
_LANE_H_STEP = 2.0
_LANE_PAD = 1.5          # vertical padding inside a bar
_LEFT_GUTTER = 30.0      # left margin reserved for the per-row track-id label
_ROW_GAP = 2.0           # gap below a (collapsed) bar before the next row
_BAND_GAP = 4.0          # gap between a selected bar and its thumbnail band

_PRESENT = QColor(95, 95, 95)        # a present frame with nothing flagged
_VALIDATED = QColor("#00ff00")
_ANCHOR = QColor("#ff8c00")
_FRAME_GUIDE = QColor(255, 210, 70)  # the current-frame vertical cursor
_COL_SELECT = QColor(255, 210, 70)   # selected-track horizontal cursor line
_LABEL = QColor(160, 160, 160)

# Thumbnail tiles (mirrors the old film-strip vocabulary).
_TILE_PX = 64
_TILE_PX_MIN = 20
_TILE_PX_MAX = 512
_TILE_ZOOM_STEP = 8      # px added/removed from the tile size per Ctrl+wheel notch
_TILE_GAP = 2            # gap between thumbnails within a band row
_TILE_ROW_GAP = 4        # gap between wrapped thumbnail rows
_DEFAULT_WIDTH = 600     # fallback panel width before the view has a real size

_CURRENT_FRAME_BORDER = QColor("#ffffff")
_VALIDATED_STRIP_COLOR = "#00ff00"
_ANCHOR_STRIP_COLOR = "#b39400"
_MARKER_STRIP_PX = 4


def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    """Convert an ``(h, w, 3)`` uint8 array to an owned RGB888 ``QImage``.

    The returned image owns its buffer (``.copy()``), so the source array is
    free to be garbage-collected. Relocated here from the retired film-strip
    module, which this panel replaces.
    """
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (h, w, 3) uint8, got {arr.shape}")
    height, width, _ = arr.shape
    image = QImage(arr.data, width, height, 3 * width, QImage.Format_RGB888)
    return image.copy()


@dataclass(frozen=True)
class LaneView:
    """One track's row: present runs plus the frames flagged in each state."""

    cell_id: int
    column: int
    segments: tuple[tuple[int, int], ...]  # inclusive [start, end] present runs
    validated: frozenset[int] = field(default_factory=frozenset)
    anchored: frozenset[int] = field(default_factory=frozenset)

    def present(self, frame: int) -> bool:
        return any(s <= frame <= e for s, e in self.segments)

    def nearest_present(self, frame: int) -> int:
        """The present frame closest to ``frame`` (for jumping into a gap)."""
        best, best_d = self.segments[0][0], None
        for s, e in self.segments:
            cand = min(max(frame, s), e)
            d = abs(cand - frame)
            if best_d is None or d < best_d:
                best, best_d = cand, d
        return best


class TrackAccordionPanel(QWidget):
    """Track bars on a shared time axis; the selected one expands to thumbnails."""

    node_activated = Signal(int, int)  # (frame, cell_id) — a bar click
    frame_clicked = Signal(int)        # (frame) — a thumbnail click

    def __init__(self, parent: QWidget | None = None, *, tile_px: int = _TILE_PX) -> None:
        super().__init__(parent)
        self._lanes: list[LaneView] = []
        self._lanes_by_cell: dict[int, LaneView] = {}
        self._n_frames = 0
        self._selected = 0
        self._current_frame = 0
        self._lane_h = float(_LANE_H)
        self._tile_px = int(tile_px)
        self._strip = TrackFilmStrip(tiles=())
        self._strip_title = ""
        self._title_text = ""

        # Geometry recorded by the last layout, for hit-testing.
        self._cell_w = 0.0
        self._bar_rows: list[tuple[float, float, int]] = []  # (y_top, y_bottom, cell_id)
        self._band_range: tuple[float, float] | None = None  # y span of the open band
        self._tile_rects: dict[int, QRectF] = {}
        self._guide_item = None
        self._col_item = None
        self._border_item = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self._title = QLabel("No lineage")
        self._title.setContentsMargins(6, 2, 6, 2)
        outer.addWidget(self._title)

        self._scene = QGraphicsScene(self)
        self._view = _AccordionView(self)
        outer.addWidget(self._view)

    # -- public API ---------------------------------------------------------
    def set_overview(
        self, lanes: list[LaneView], *, n_frames: int, title: str = ""
    ) -> None:
        """Replace the set of track bars (the selected track's band is kept)."""
        self._lanes = list(lanes)
        self._lanes_by_cell = {int(ln.cell_id): ln for ln in self._lanes}
        self._n_frames = int(n_frames)
        self._title_text = title or f"{len(self._lanes)} track(s)"
        self._relayout()

    def set_strip(self, strip: TrackFilmStrip, title: str = "") -> None:
        """Set the selected track's per-frame thumbnails (its expanded band)."""
        self._strip = strip if strip is not None else TrackFilmStrip(tiles=())
        self._strip_title = title
        self._relayout()

    def set_selection(self, cell_id: int) -> None:
        """Mark ``cell_id`` as the selected/expanded track and re-lay out."""
        self._selected = int(cell_id or 0)
        self._relayout()

    def set_current_frame(self, frame: int) -> None:
        """Move the shared current-frame guide + the open band's tile border."""
        self._current_frame = int(frame)
        self._apply_frame_guide()
        self._apply_current_border()

    def center_on_track(self, cell_id: int) -> None:
        """Scroll vertically so ``cell_id``'s bar row is centered."""
        for top, bottom, cid in self._bar_rows:
            if cid == int(cell_id or 0):
                y = (top + bottom) / 2.0
                vc = self._view.mapToScene(self._view.viewport().rect().center())
                self._view.centerOn(vc.x(), y)
                return

    def center_on_strip(self) -> bool:
        """Scroll vertically so the open thumbnail band is centered.

        Returns ``True`` when a band was present to center on, ``False`` when the
        selected track has no expanded film strip yet (so the caller can fall
        back to centering on the bar row).
        """
        if self._band_range is None:
            return False
        top, bottom = self._band_range
        y = (top + bottom) / 2.0
        vc = self._view.mapToScene(self._view.viewport().rect().center())
        self._view.centerOn(vc.x(), y)
        return True

    def grid_neighbor_frame(
        self, current_frame: int, *, dx: int = 0, dy: int = 0, wrap: bool = False
    ) -> int | None:
        """Frame of the tile ``dx`` columns / ``dy`` rows from ``current_frame``.

        The thumbnails wrap row-major across the band width, so this reconstructs
        that grid from the laid-out tile rects: ``dx`` walks the band in reading
        order (left/right, wrapping at row ends), while ``dy`` jumps a whole row
        up/down keeping the column position (clamped on a shorter last row).
        With ``wrap`` (the viewer's loop mode), running off either end returns to
        the opposite end instead of stopping; otherwise the edge returns ``None``.
        """
        if not self._tile_rects:
            return None
        ordered = sorted(
            self._tile_rects.items(), key=lambda kv: (round(kv[1].y(), 1), kv[1].x())
        )
        rows: list[list[int]] = []
        last_y: float | None = None
        for frame, rect in ordered:
            ry = round(rect.y(), 1)
            if last_y is None or ry != last_y:
                rows.append([])
                last_y = ry
            rows[-1].append(int(frame))

        cur = int(current_frame)
        loc = next(
            ((r, row.index(cur)) for r, row in enumerate(rows) if cur in row), None
        )
        if loc is None:
            # Off-band (e.g. the viewer is on a frame the track skips): step in.
            return rows[0][0] if rows and rows[0] else None
        r, c = loc
        if dx:
            flat = [f for row in rows for f in row]
            j = flat.index(cur) + dx
            if 0 <= j < len(flat):
                return flat[j]
            return flat[j % len(flat)] if wrap and flat else None
        if dy:
            nr = r + dy
            if 0 <= nr < len(rows):
                row = rows[nr]
                return row[min(c, len(row) - 1)]
            if wrap and rows:
                row = rows[nr % len(rows)]
                return row[min(c, len(row) - 1)]
            return None
        return None

    # -- layout -------------------------------------------------------------
    def _row_width(self) -> float:
        vw = self._view.viewport().width()
        return float(vw if vw > _LEFT_GUTTER + 8 else _DEFAULT_WIDTH)

    def _relayout(self) -> None:
        """Lay every bar top-down, growing the selected one into a band."""
        self._scene.clear()
        self._guide_item = None
        self._col_item = None
        self._border_item = None
        self._bar_rows = []
        self._band_range = None
        self._tile_rects = {}
        self._title.setText(getattr(self, "_title_text", "") or "No lineage")

        if not self._lanes or self._n_frames <= 0:
            self._scene.setSceneRect(0, 0, self._row_width(), 1)
            return

        row_width = self._row_width()
        self._cell_w = max((row_width - _LEFT_GUTTER) / self._n_frames, 0.5)
        bar_h = self._lane_h - 2 * _LANE_PAD

        y = 0.0
        for lane in sorted(self._lanes, key=lambda x: x.column):
            self._draw_label(y, lane.cell_id)
            self._draw_bar(lane, y, bar_h)
            self._bar_rows.append((y, y + self._lane_h, int(lane.cell_id)))
            y += self._lane_h
            if int(lane.cell_id) == self._selected and not self._strip.is_empty():
                y += _BAND_GAP
                band_top = y
                y = self._draw_band(y, row_width)
                self._band_range = (band_top, y)
            y += _ROW_GAP

        self._scene.setSceneRect(0, 0, row_width, max(y, 1.0))
        self._apply_column_highlight()
        self._apply_frame_guide()
        self._apply_current_border()

    def _draw_label(self, y: float, cell_id: int) -> None:
        text = self._scene.addText(str(int(cell_id)))
        text.setDefaultTextColor(_LABEL)
        font = text.font()
        font.setPointSizeF(max(6.0, self._lane_h - 6.0))
        text.setFont(font)
        text.setPos(0, y - 2)

    def _draw_bar(self, lane: LaneView, y: float, bar_h: float) -> None:
        top = y + _LANE_PAD
        for s, e in lane.segments:
            self._fill_run(top, bar_h, s, e, _PRESENT)
        # Coalesce flagged frames into contiguous runs before drawing: with a
        # fractional ``cell_w`` and no antialiasing, painting each frame as its
        # own snapped rect leaves sub-pixel seams that read as a ragged
        # "indentation" against the single-rect present bar. One rect per run
        # rounds once and stays flush with the grey bar beneath it.
        for s, e in _runs(lane.validated):
            self._fill_run(top, bar_h, s, e, _VALIDATED)
        for s, e in _runs(lane.anchored):
            self._fill_run(top, bar_h, s, e, _ANCHOR)

    def _fill_run(
        self, top: float, bar_h: float, start: int, end: int, color: QColor
    ) -> None:
        self._scene.addRect(
            _LEFT_GUTTER + start * self._cell_w, top,
            (end - start + 1) * self._cell_w, bar_h, _no_pen(), color,
        )

    def _draw_band(self, y: float, row_width: float) -> float:
        """Lay the selected track's thumbnails row-major, wrapping to the width.

        Returns the ``y`` immediately below the band.
        """
        x = float(_LEFT_GUTTER)
        row_h = 0.0
        row_first = True
        for tile in self._strip.tiles:
            pm = self._tile_pixmap(tile)
            w, h = pm.width(), pm.height()
            if not row_first and x + w > row_width:
                y += row_h + _TILE_ROW_GAP
                x = float(_LEFT_GUTTER)
                row_h = 0.0
            item = self._scene.addPixmap(pm)
            item.setOffset(x, y)
            item.setData(0, int(tile.frame))
            item.setToolTip(self._tile_tooltip(tile))
            self._tile_rects[int(tile.frame)] = QRectF(x, y, w, h)
            x += w + _TILE_GAP
            row_h = max(row_h, h)
            row_first = False
        return y + row_h

    def _tile_pixmap(self, tile) -> QPixmap:
        pm = QPixmap.fromImage(rgb_to_qimage(tile.rgb)).scaledToHeight(
            self._tile_px, Qt.FastTransformation
        )
        if getattr(tile, "placeholder", False):
            _draw_placeholder(pm)
            return pm
        _draw_marker_strips(
            pm,
            validated=getattr(tile, "validated", False),
            anchored=getattr(tile, "anchored", False),
        )
        return pm

    # -- markers ------------------------------------------------------------
    def _apply_column_highlight(self) -> None:
        if self._col_item is not None:
            self._scene.removeItem(self._col_item)
            self._col_item = None
        for top, bottom, cid in self._bar_rows:
            if cid == self._selected:
                y = (top + bottom) / 2.0
                self._col_item = self._scene.addLine(
                    _LEFT_GUTTER, y, _LEFT_GUTTER + self._n_frames * self._cell_w, y,
                    QPen(_COL_SELECT, 1.5),
                )
                self._col_item.setZValue(2)
                return

    def _apply_frame_guide(self) -> None:
        if self._guide_item is not None:
            self._scene.removeItem(self._guide_item)
            self._guide_item = None
        if not self._bar_rows or self._n_frames <= 0:
            return
        x = _LEFT_GUTTER + self._current_frame * self._cell_w + self._cell_w / 2.0
        bottom = self._scene.sceneRect().height()
        self._guide_item = self._scene.addLine(x, 0, x, bottom, QPen(_FRAME_GUIDE, 1.5))
        self._guide_item.setZValue(3)

    def _apply_current_border(self) -> None:
        if self._border_item is not None:
            self._scene.removeItem(self._border_item)
            self._border_item = None
        rect = self._tile_rects.get(self._current_frame)
        if rect is None:
            return
        self._border_item = self._scene.addRect(rect, QPen(_CURRENT_FRAME_BORDER, 2))
        self._border_item.setZValue(4)

    @staticmethod
    def _tile_tooltip(tile) -> str:
        if getattr(tile, "placeholder", False):
            return f"Frame {tile.frame} — track absent (no nucleus)"
        tags = []
        if getattr(tile, "validated", False):
            tags.append("validated")
        if getattr(tile, "anchored", False):
            tags.append("anchored")
        suffix = f" ({', '.join(tags)})" if tags else ""
        return f"Frame {tile.frame}{suffix} — click to jump"

    # -- Ctrl+wheel region helpers ------------------------------------------
    def _over_band(self, scene_y: float) -> bool:
        """True when ``scene_y`` is inside the open thumbnail band."""
        if self._band_range is None:
            return False
        top, bottom = self._band_range
        return top <= scene_y <= bottom

    def _zoom_tiles(self, delta: int) -> None:
        tile_px = max(_TILE_PX_MIN, min(self._tile_px + delta, _TILE_PX_MAX))
        if tile_px == self._tile_px:
            return
        self._tile_px = tile_px
        self._relayout()

    def _zoom_lane_height(self, delta: float) -> None:
        lane_h = max(_LANE_H_MIN, min(self._lane_h + delta, _LANE_H_MAX))
        if lane_h == self._lane_h:
            return
        self._lane_h = lane_h
        self._relayout()

    def _ctrl_wheel_zoom(self, *, up: bool, scene_y: float) -> None:
        """Region-aware Ctrl+wheel: resize tiles over the band, else bar height."""
        if self._over_band(scene_y):
            self._zoom_tiles(_TILE_ZOOM_STEP if up else -_TILE_ZOOM_STEP)
        else:
            self._zoom_lane_height(_LANE_H_STEP if up else -_LANE_H_STEP)

    # -- hit-testing --------------------------------------------------------
    def _activate_at(self, scene_x: float, scene_y: float) -> None:
        """Translate a click into a thumbnail jump or a bar (frame, cell) jump."""
        if self._over_band(scene_y):
            for frame, rect in self._tile_rects.items():
                if rect.contains(scene_x, scene_y):
                    self.frame_clicked.emit(int(frame))
                    return
            return
        for top, bottom, cell_id in self._bar_rows:
            if top <= scene_y <= bottom:
                lane = self._lanes_by_cell.get(cell_id)
                if lane is None or self._cell_w <= 0:
                    return
                frame = int((scene_x - _LEFT_GUTTER) // self._cell_w)
                frame = min(max(frame, 0), max(self._n_frames - 1, 0))
                if not lane.present(frame):
                    frame = lane.nearest_present(frame)
                self.node_activated.emit(int(frame), int(cell_id))
                return


def _no_pen() -> QPen:
    return QPen(Qt.NoPen)


def _runs(frames: frozenset[int]) -> tuple[tuple[int, int], ...]:
    """Collapse a frame set into inclusive ``(start, end)`` contiguous runs."""
    if not frames:
        return ()
    return tuple(
        (seg.start, seg.end) for seg in _segments_from_frames(sorted(frames))
    )


_PLACEHOLDER_FILL = QColor(48, 50, 56)     # empty-tile body (a touch above bg)
_PLACEHOLDER_BORDER = QColor(90, 96, 107)  # dashed frame marking a missing frame


def _draw_placeholder(pixmap: QPixmap) -> None:
    """Paint a tile for a frame the track skips: a dim body with a dashed frame.

    The blank ``rgb`` would otherwise read as a hole in the band on the dark
    scene; this gives missing frames a clearly-empty, dashed-outline thumbnail.
    """
    painter = QPainter(pixmap)
    w, h = pixmap.width(), pixmap.height()
    painter.fillRect(0, 0, w, h, _PLACEHOLDER_FILL)
    pen = QPen(_PLACEHOLDER_BORDER, 1, Qt.DashLine)
    painter.setPen(pen)
    painter.drawRect(0, 0, w - 1, h - 1)
    painter.end()


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


class _AccordionView(QGraphicsView):
    """Plain-wheel scrolls; Ctrl+wheel zooms region-aware; a click reports it."""

    def __init__(self, panel: TrackAccordionPanel) -> None:
        super().__init__(panel._scene, panel)
        self._panel = panel
        self._press_pos = None
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def wheelEvent(self, event) -> None:
        if not (event.modifiers() & Qt.ControlModifier):
            super().wheelEvent(event)  # plain wheel scrolls vertically
            return
        point = event.position().toPoint() if hasattr(event, "position") else event.pos()
        self._panel._ctrl_wheel_zoom(
            up=event.angleDelta().y() > 0,
            scene_y=self.mapToScene(point).y(),
        )
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # cell_w and band wrapping are width-derived, so re-lay on every resize.
        self._panel._relayout()

    def mousePressEvent(self, event) -> None:
        self._press_pos = event.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if self._press_pos is None:
            return
        moved = (event.pos() - self._press_pos).manhattanLength()
        self._press_pos = None
        if moved > _CLICK_SLOP:
            return  # a scroll-drag, not a click
        pt = self.mapToScene(event.pos())
        self._panel._activate_at(pt.x(), pt.y())


__all__ = ["LaneView", "TrackAccordionPanel", "rgb_to_qimage"]
