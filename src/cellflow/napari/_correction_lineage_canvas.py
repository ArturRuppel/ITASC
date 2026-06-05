"""Correction swimlane overview — one column per track on a shared time axis.

The thumbnails-for-every-track layout did not scale: long tracks turned into
columns of mostly-uninformative tiles. This is the dense graph overview that
replaced it; the selected track's per-frame film strip now lives in its own
docked panel (:class:`~cellflow.napari._correction_film_strip.TrackFilmStripPanel`).

One thin *row* per track, ``x`` is time running left→right so all tracks share a
single global frame axis. A track's present runs draw as bars and a gap (a
vanish/return — a likely ID swap) reads as a break. Per-frame status paints into
the row: green = validated, orange = anchored. Because the time axis is shared,
the current-frame cursor is one vertical guide line across every row, and the
selected track is marked by a horizontal cursor line along its row.

The panel is a pure renderer: the controller hands it ready ``LaneView`` structs;
click a row to jump there and select the cell via :attr:`node_activated`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPen
from qtpy.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QVBoxLayout,
    QWidget,
)

_CLICK_SLOP = 6  # max drag (px) still treated as a click, not a pan
_ZOOM_STEP = 1.15

_LANE_H = 12.0   # scene height of one track row
_CELL_W = 6.0    # scene width of one frame within a row
_LANE_PAD = 1.5  # vertical padding inside a row

_PRESENT = QColor(95, 95, 95)          # a present frame with nothing flagged
_VALIDATED = QColor("#00ff00")
_ANCHOR = QColor("#ff8c00")
_FRAME_GUIDE = QColor(255, 210, 70)    # the current-frame vertical cursor
_COL_SELECT = QColor(255, 210, 70)     # selected-track horizontal cursor line


@dataclass(frozen=True)
class LaneView:
    """One track's column: present runs plus the frames flagged in each state."""

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


class LineageCanvasPanel(QWidget):
    """Swimlane overview — one row per track on a shared left→right time axis."""

    node_activated = Signal(int, int)  # (frame, cell_id)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected = 0
        self._current_frame = 0
        self._n_frames = 0
        self._lanes_by_cell: dict[int, LaneView] = {}
        self._col_to_cell: list[int] = []
        self._guide_item = None
        self._col_item = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self._title = QLabel("No lineage")
        self._title.setContentsMargins(6, 2, 6, 2)
        outer.addWidget(self._title)

        self._scene = QGraphicsScene(self)
        self._view = _OverviewView(self)
        outer.addWidget(self._view)

    # -- overview -----------------------------------------------------------
    def set_overview(
        self, lanes: list[LaneView], *, n_frames: int, title: str = ""
    ) -> None:
        """Render the swimlane overview (replacing whatever was shown)."""
        self._scene.clear()
        self._guide_item = None
        self._col_item = None
        self._n_frames = int(n_frames)
        self._lanes_by_cell = {int(ln.cell_id): ln for ln in lanes}
        self._col_to_cell = [
            int(ln.cell_id) for ln in sorted(lanes, key=lambda x: x.column)
        ]
        self._title.setText(title or f"{len(lanes)} track(s)")
        for lane in lanes:
            self._draw_lane(lane)
        self._scene.setSceneRect(self._scene.itemsBoundingRect())
        self._apply_column_highlight()
        self._apply_frame_guide()

    def _draw_lane(self, lane: LaneView) -> None:
        y = lane.column * _LANE_H + _LANE_PAD
        h = _LANE_H - 2 * _LANE_PAD
        # Present runs as one bar each (cheap: a few rects per track, not per frame).
        for s, e in lane.segments:
            self._scene.addRect(
                s * _CELL_W, y, (e - s + 1) * _CELL_W, h, _no_pen(), _PRESENT,
            )
        # Sparse per-frame status marks, drawn over the bars.
        for frame in lane.validated:
            self._fill_cell(y, h, frame, _VALIDATED)
        for frame in lane.anchored:
            self._fill_cell(y, h, frame, _ANCHOR)

    def _fill_cell(self, y: float, h: float, frame: int, color: QColor) -> None:
        self._scene.addRect(frame * _CELL_W, y, _CELL_W, h, _no_pen(), color)

    def set_selection(self, cell_id: int) -> None:
        """Highlight the selected track's row."""
        self._selected = int(cell_id or 0)
        self._apply_column_highlight()

    def set_current_frame(self, frame: int) -> None:
        """Move the shared current-frame guide line."""
        self._current_frame = int(frame)
        self._apply_frame_guide()

    def center_on_track(self, cell_id: int) -> None:
        """Scroll the overview vertically so the track's row is centered.

        Used when a cell is selected in the image viewer: the matching lane
        scrolls to the vertical middle so the user sees it without hunting. The
        horizontal (time) position is preserved, and ``centerOn`` clamps to the
        scroll range — so a row near the top/bottom stops at the edge rather
        than scrolling past the content.
        """
        lane = self._lanes_by_cell.get(int(cell_id or 0))
        if lane is None:
            return
        y = lane.column * _LANE_H + _LANE_H / 2.0
        viewport_center = self._view.mapToScene(
            self._view.viewport().rect().center()
        )
        self._view.centerOn(viewport_center.x(), y)

    def _apply_column_highlight(self) -> None:
        if self._col_item is not None:
            self._scene.removeItem(self._col_item)
            self._col_item = None
        lane = self._lanes_by_cell.get(self._selected)
        if lane is None or self._n_frames <= 0:
            return
        y = lane.column * _LANE_H + _LANE_H / 2.0
        self._col_item = self._scene.addLine(
            0, y, self._n_frames * _CELL_W, y, QPen(_COL_SELECT, 1.5),
        )
        self._col_item.setZValue(2)  # over the lane bars, like a cursor

    def _apply_frame_guide(self) -> None:
        if self._guide_item is not None:
            self._scene.removeItem(self._guide_item)
            self._guide_item = None
        if not self._col_to_cell:
            return
        x = self._current_frame * _CELL_W + _CELL_W / 2.0
        bottom = len(self._col_to_cell) * _LANE_H
        self._guide_item = self._scene.addLine(x, 0, x, bottom, QPen(_FRAME_GUIDE, 1.5))

    # -- hit-testing --------------------------------------------------------
    def _activate_at(self, scene_x: float, scene_y: float) -> None:
        """Translate a click in the overview into a ``(frame, cell)`` jump."""
        row = int(scene_y // _LANE_H)
        if row < 0 or row >= len(self._col_to_cell):
            return
        cell_id = self._col_to_cell[row]
        lane = self._lanes_by_cell[cell_id]
        frame = int(scene_x // _CELL_W)
        frame = min(max(frame, 0), max(self._n_frames - 1, 0))
        if not lane.present(frame):
            frame = lane.nearest_present(frame)
        self.node_activated.emit(int(frame), int(cell_id))


def _no_pen() -> QPen:
    return QPen(Qt.NoPen)


class _OverviewView(QGraphicsView):
    """Drag-to-pan, wheel-to-zoom view that reports column clicks."""

    def __init__(self, panel: LineageCanvasPanel) -> None:
        super().__init__(panel._scene, panel)
        self._panel = panel
        self._press_pos = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if not (event.modifiers() & Qt.ControlModifier):
            super().wheelEvent(event)  # plain wheel scrolls, like everywhere else
            return
        factor = _ZOOM_STEP if event.angleDelta().y() > 0 else 1.0 / _ZOOM_STEP
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._press_pos = event.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mouseReleaseEvent(event)
        if self._press_pos is None:
            return
        moved = (event.pos() - self._press_pos).manhattanLength()
        self._press_pos = None
        if moved > _CLICK_SLOP:
            return  # a pan, not a click
        pt = self.mapToScene(event.pos())
        self._panel._activate_at(pt.x(), pt.y())


__all__ = ["LaneView", "LineageCanvasPanel"]
