"""Combined correction canvas — a dense swimlane overview plus a focus strip.

The thumbnails-for-every-track layout did not scale: long tracks turned into
columns of mostly-uninformative tiles. This replaces it with an
overview-plus-detail pattern, kept as two non-overlapping regions so the summary
stays a clean graph view (room to grow in-place editing later).

* **Overview** (top): one thin *column* per track, ``y`` is time running
  downward so all tracks share a single global frame axis. A track's present
  runs draw as bars and a gap (a vanish/return — a likely ID swap) reads as a
  break. Per-frame status paints into the column: green = validated, orange =
  anchored. Because the time axis is shared, the
  current-frame cursor is one horizontal guide line across every column, and the
  selected track is marked by a vertical cursor line down its column.
* **Detail** (bottom): the *selected* track's film strip — the only place
  per-frame crops are built, so cost is O(1 track) regardless of track count.

The panel is a pure renderer: the controller hands it ready ``LaneView`` structs
and a prepared :class:`TrackFilmStrip`; click a column (or a tile) to jump there
and select the cell via :attr:`node_activated`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPen
from qtpy.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._correction_film_strip import TrackFilmStripPanel
from cellflow.napari._correction_track_path import TrackFilmStrip

_CLICK_SLOP = 6  # max drag (px) still treated as a click, not a pan
_ZOOM_STEP = 1.15

_COL_W = 12.0    # scene width of one track column
_CELL_H = 6.0    # scene height of one frame within a column
_COL_PAD = 1.5   # horizontal padding inside a column

_PRESENT = QColor(95, 95, 95)          # a present frame with nothing flagged
_VALIDATED = QColor("#00ff00")
_ANCHOR = QColor("#ff8c00")
_FRAME_GUIDE = QColor(255, 210, 70)    # the current-frame horizontal cursor
_COL_SELECT = QColor(255, 255, 255)    # selected-track vertical cursor line


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
    """Swimlane overview stacked over the selected track's film strip."""

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

        split = QSplitter(Qt.Vertical)
        self._scene = QGraphicsScene(self)
        self._view = _OverviewView(self)
        split.addWidget(self._view)
        self._detail = TrackFilmStripPanel(tile_px=72)
        self._detail.frame_clicked.connect(self._on_detail_frame_clicked)
        split.addWidget(self._detail)
        # Graph view takes 25% of the height by default; the film strip gets 75%.
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setSizes([100, 300])
        outer.addWidget(split)

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
        x = lane.column * _COL_W + _COL_PAD
        w = _COL_W - 2 * _COL_PAD
        # Present runs as one bar each (cheap: a few rects per track, not per frame).
        for s, e in lane.segments:
            self._scene.addRect(
                x, s * _CELL_H, w, (e - s + 1) * _CELL_H, _no_pen(), _PRESENT,
            )
        # Sparse per-frame status marks, drawn over the bars.
        for frame in lane.validated:
            self._fill_cell(x, w, frame, _VALIDATED)
        for frame in lane.anchored:
            self._fill_cell(x, w, frame, _ANCHOR)

    def _fill_cell(self, x: float, w: float, frame: int, color: QColor) -> None:
        self._scene.addRect(x, frame * _CELL_H, w, _CELL_H, _no_pen(), color)

    def set_selection(self, cell_id: int) -> None:
        """Highlight the selected track's column."""
        self._selected = int(cell_id or 0)
        self._apply_column_highlight()

    def set_current_frame(self, frame: int) -> None:
        """Move the shared current-frame guide and the detail-strip highlight."""
        self._current_frame = int(frame)
        self._apply_frame_guide()
        self._detail.set_current_frame(int(frame))

    def _apply_column_highlight(self) -> None:
        if self._col_item is not None:
            self._scene.removeItem(self._col_item)
            self._col_item = None
        lane = self._lanes_by_cell.get(self._selected)
        if lane is None or self._n_frames <= 0:
            return
        x = lane.column * _COL_W + _COL_W / 2.0
        self._col_item = self._scene.addLine(
            x, 0, x, self._n_frames * _CELL_H, QPen(_COL_SELECT, 1.5),
        )
        self._col_item.setZValue(2)  # over the lane bars, like a cursor

    def _apply_frame_guide(self) -> None:
        if self._guide_item is not None:
            self._scene.removeItem(self._guide_item)
            self._guide_item = None
        if not self._col_to_cell:
            return
        y = self._current_frame * _CELL_H + _CELL_H / 2.0
        right = len(self._col_to_cell) * _COL_W
        self._guide_item = self._scene.addLine(0, y, right, y, QPen(_FRAME_GUIDE, 1.5))

    # -- detail -------------------------------------------------------------
    def set_detail(self, strip: TrackFilmStrip, *, title: str = "") -> None:
        """Show the selected track's film strip in the detail band."""
        self._detail.set_strip(strip, title=title)
        self._detail.set_current_frame(self._current_frame)

    def _on_detail_frame_clicked(self, frame: int) -> None:
        if self._selected:
            self.node_activated.emit(int(frame), int(self._selected))

    # -- hit-testing --------------------------------------------------------
    def _activate_at(self, scene_x: float, scene_y: float) -> None:
        """Translate a click in the overview into a ``(frame, cell)`` jump."""
        col = int(scene_x // _COL_W)
        if col < 0 or col >= len(self._col_to_cell):
            return
        cell_id = self._col_to_cell[col]
        lane = self._lanes_by_cell[cell_id]
        frame = int(scene_y // _CELL_H)
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
