"""Docked lineage swimlanes — the track-presence *view*.

Each track id is a horizontal lane; time runs left→right; a filled bar marks the
frames the cell is present and breaks in the bar are gaps (a vanished/returned
track — a likely ID swap). A vertical cursor marks the current frame, and the
selected track's lane is highlighted. Clicking a bar emits
:attr:`track_activated` with ``(frame, cell_id)`` — the frame under the cursor —
so the host can jump there and select the cell.

Built on ``QGraphicsScene`` rather than a second vispy canvas: lanes are plain
rects, which lays out hundreds of tracks cheaply and keeps this import-safe
without a GL context.
"""
from __future__ import annotations

from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QPen
from qtpy.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from cellflow.segmentation.lineage import LineageModel

_LANE_H = 12
_LANE_GAP = 3
_LABEL_W = 44
_MIN_FRAME_PX = 3
_BAR_COLOR = QColor(90, 140, 200)
_BAR_GAP_COLOR = QColor(150, 90, 90)
_SELECTED_COLOR = QColor(240, 200, 80)
_CURSOR_COLOR = QColor(255, 255, 255, 180)
_LANE_BG = QColor(60, 60, 60)


class _LineageView(QGraphicsView):
    """A graphics view that reports the lane + frame under a click."""

    clicked = Signal(int, int)  # (frame, cell_id)

    def __init__(self, panel: LineagePanel) -> None:
        super().__init__(panel._scene, panel)
        self._panel = panel
        self.setMouseTracking(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            scene_pt = self.mapToScene(event.pos())
            hit = self._panel.hit_test(scene_pt.x(), scene_pt.y())
            if hit is not None:
                frame, cell_id = hit
                self.clicked.emit(frame, cell_id)
        super().mousePressEvent(event)


class LineagePanel(QWidget):
    """Swimlane view of per-track frame presence with click-to-navigate."""

    track_activated = Signal(int, int)  # (frame, cell_id)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model: LineageModel | None = None
        self._frame_px: float = _MIN_FRAME_PX
        self._lane_y: dict[int, float] = {}  # cell_id -> lane top y
        self._selected: int = 0
        self._current_frame: int = 0
        self._cursor_item = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self._title = QLabel("No tracks")
        self._title.setContentsMargins(6, 2, 6, 2)
        outer.addWidget(self._title)

        self._scene = QGraphicsScene(self)
        self._view = _LineageView(self)
        self._view.clicked.connect(self._on_view_clicked)
        outer.addWidget(self._view)

    def set_model(self, model: LineageModel | None) -> None:
        """Render ``model``'s lanes (``None`` or empty clears the panel)."""
        self._model = model
        self._scene.clear()
        self._lane_y.clear()
        self._cursor_item = None
        if model is None or not model.lanes:
            self._title.setText("No tracks")
            return
        self._title.setText(f"{len(model.lanes)} track(s)")
        width = max(1, self._view.viewport().width() - _LABEL_W - 4)
        self._frame_px = max(_MIN_FRAME_PX, width / max(1, model.n_frames))
        for i, lane in enumerate(model.lanes):
            y = i * (_LANE_H + _LANE_GAP)
            self._lane_y[lane.cell_id] = y
            # Lane background spans the full timeline (so gaps read as breaks).
            self._scene.addRect(
                QRectF(_LABEL_W, y, model.n_frames * self._frame_px, _LANE_H),
                QPen(Qt.NoPen), QBrush(_LANE_BG),
            )
            label = self._scene.addText(str(lane.cell_id))
            label.setDefaultTextColor(QColor(200, 200, 200))
            label.setPos(0, y - 4)
            color = _BAR_GAP_COLOR if lane.has_gap else _BAR_COLOR
            for seg in lane.segments:
                x = _LABEL_W + seg.start * self._frame_px
                w = max(self._frame_px, seg.length * self._frame_px)
                rect = self._scene.addRect(
                    QRectF(x, y, w, _LANE_H), QPen(Qt.NoPen), QBrush(color),
                )
                rect.setData(0, int(lane.cell_id))
        self._scene.setSceneRect(self._scene.itemsBoundingRect())
        self._draw_cursor()
        self._apply_selection_highlight()

    def set_selection(self, cell_id: int) -> None:
        """Highlight the lane for ``cell_id`` (0 clears the highlight)."""
        self._selected = int(cell_id or 0)
        self._apply_selection_highlight()

    def set_current_frame(self, frame: int) -> None:
        """Move the vertical current-frame cursor without rebuilding lanes."""
        self._current_frame = int(frame)
        self._draw_cursor()

    def hit_test(self, x: float, y: float) -> tuple[int, int] | None:
        """Map a scene point to ``(frame, cell_id)`` if it lands on a lane."""
        if self._model is None or x < _LABEL_W:
            return None
        frame = int((x - _LABEL_W) / self._frame_px)
        frame = max(0, min(frame, self._model.n_frames - 1))
        for cell_id, lane_top in self._lane_y.items():
            if lane_top <= y <= lane_top + _LANE_H:
                return frame, cell_id
        return None

    def _draw_cursor(self) -> None:
        if self._model is None:
            return
        if self._cursor_item is not None:
            self._scene.removeItem(self._cursor_item)
            self._cursor_item = None
        x = _LABEL_W + self._current_frame * self._frame_px
        height = len(self._model.lanes) * (_LANE_H + _LANE_GAP)
        self._cursor_item = self._scene.addLine(
            x, 0, x, height, QPen(_CURSOR_COLOR, 1),
        )

    def _apply_selection_highlight(self) -> None:
        for item in self._scene.items():
            cell_id = item.data(0) if hasattr(item, "data") else None
            if cell_id is None:
                continue
            is_sel = int(cell_id) == self._selected
            pen = QPen(_SELECTED_COLOR, 2) if is_sel else QPen(Qt.NoPen)
            try:
                item.setPen(pen)
            except Exception:
                pass

    def _on_view_clicked(self, frame: int, cell_id: int) -> None:
        self.track_activated.emit(int(frame), int(cell_id))


__all__ = ["LineagePanel"]
