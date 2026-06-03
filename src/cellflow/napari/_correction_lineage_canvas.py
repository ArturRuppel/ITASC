"""Pannable/zoomable lineage canvas — the unified correction *view*.

One big ``QGraphicsView`` that folds the film strip and lineage graph into a
single picture: every track is a vertical chain of nodes (time runs downward),
each node is that frame's nucleus crop (the film-strip tile), and consecutive
present frames are joined by a plain connector (which skips rows across a gap).
The node box edges carry two orthogonal cursors: its vertical sides light up for
the selected track, its horizontal sides for the current frame. Drag to pan,
wheel to zoom, click a node to jump there and select the cell.

The view is a pure renderer: it is handed ready-to-blit :class:`NodeView` /
:class:`EdgeView` structs (pixels, positions, colours) by its controller, so the
heavy lifting — cropping tiles, scoring, layout — stays testable outside Qt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qtpy.QtCore import QRectF, Signal
from qtpy.QtGui import QColor, QImage, QPen, QPixmap
from qtpy.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QVBoxLayout,
    QWidget,
)

_CLICK_SLOP = 6  # max drag (px) still treated as a click, not a pan
_ZOOM_STEP = 1.15
# Confidence rides on the *cell outline* baked into each crop (by the
# controller). The node's box edges are free for the two orthogonal highlights:
# the box is a column cell, so its *vertical* sides mark the selected track and
# its *horizontal* sides mark the current frame (one node lit on all four edges
# is the selected cell at the current frame).
_TRACK_HL = QColor(255, 255, 255)   # vertical sides → selected track
_FRAME_HL = QColor(255, 210, 70)    # horizontal sides → current frame
_HL_WIDTH = 4
_EDGE_COLOR = QColor(170, 170, 170)  # plain connector between consecutive frames
_EDGE_WIDTH = 2


def _rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = arr.shape
    image = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)


@dataclass(frozen=True)
class NodeView:
    """A ready-to-blit node: a frame crop centred at ``(x, y)`` in scene units."""

    cell_id: int
    t: int
    x: float
    y: float
    rgb: np.ndarray  # (h, w, 3) uint8


@dataclass(frozen=True)
class EdgeView:
    """A plain connector between two node centres."""

    cell_id: int
    x0: float
    y0: float
    x1: float
    y1: float


class LineageCanvasPanel(QWidget):
    """The docked pannable/zoomable lineage canvas with click-to-navigate."""

    node_activated = Signal(int, int)  # (frame, cell_id)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected = 0
        self._current_frame = 0
        self._row_height = 1.0
        self._node_border = 3
        self._scene_width = 0.0
        # Node box geometry, indexed for the two highlights. Highlight line items
        # are added/removed on demand (only the lit track/frame carry items).
        self._boxes_by_track: dict[int, list[QRectF]] = {}
        self._boxes_by_frame: dict[int, list[QRectF]] = {}
        self._track_hl_items: list = []
        self._frame_hl_items: list = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self._title = QLabel("No lineage")
        self._title.setContentsMargins(6, 2, 6, 2)
        outer.addWidget(self._title)

        self._scene = QGraphicsScene(self)
        self._view = _CanvasView(self)
        outer.addWidget(self._view)

    def set_scene(
        self,
        nodes: list[NodeView],
        edges: list[EdgeView],
        *,
        row_height: float,
        scene_width: float,
        title: str = "",
    ) -> None:
        """Render ``nodes`` + ``edges`` (replacing whatever was shown)."""
        self._scene.clear()
        self._boxes_by_track.clear()
        self._boxes_by_frame.clear()
        self._track_hl_items = []
        self._frame_hl_items = []
        self._row_height = max(1.0, float(row_height))
        self._scene_width = float(scene_width)
        self._title.setText(title or f"{len(nodes)} node(s)")
        if not nodes:
            return
        # Edges first so node thumbnails sit on top of their connectors.
        edge_pen = QPen(_EDGE_COLOR, _EDGE_WIDTH)
        for edge in edges:
            self._scene.addLine(edge.x0, edge.y0, edge.x1, edge.y1, edge_pen)
        b = self._node_border
        for node in nodes:
            pm = _rgb_to_pixmap(node.rgb)
            w, h = pm.width(), pm.height()
            left, top = node.x - w / 2.0, node.y - h / 2.0
            box = QRectF(left - b, top - b, w + 2 * b, h + 2 * b)
            self._boxes_by_track.setdefault(int(node.cell_id), []).append(box)
            self._boxes_by_frame.setdefault(int(node.t), []).append(box)
            item = self._scene.addPixmap(pm)
            item.setOffset(left, top)
            item.setData(0, int(node.cell_id))
            item.setData(1, int(node.t))
            item.setToolTip(f"frame {node.t}, cell {node.cell_id}")
        self._scene.setSceneRect(self._scene.itemsBoundingRect())
        self._apply_track_highlight()
        self._apply_frame_highlight()

    def set_selection(self, cell_id: int) -> None:
        """Light the vertical sides of the selected track's node boxes."""
        self._selected = int(cell_id or 0)
        self._apply_track_highlight()

    def set_current_frame(self, frame: int) -> None:
        """Light the horizontal sides of the current frame's node boxes."""
        self._current_frame = int(frame)
        self._apply_frame_highlight()

    def _apply_track_highlight(self) -> None:
        """Vertical sides (left/right) of the selected track's boxes."""
        for item in self._track_hl_items:
            self._scene.removeItem(item)
        self._track_hl_items = []
        pen = QPen(_TRACK_HL, _HL_WIDTH)
        for box in self._boxes_by_track.get(self._selected, ()):
            self._track_hl_items.append(
                self._scene.addLine(box.left(), box.top(), box.left(), box.bottom(), pen)
            )
            self._track_hl_items.append(
                self._scene.addLine(box.right(), box.top(), box.right(), box.bottom(), pen)
            )

    def _apply_frame_highlight(self) -> None:
        """Horizontal sides (top/bottom) of the current frame's boxes."""
        for item in self._frame_hl_items:
            self._scene.removeItem(item)
        self._frame_hl_items = []
        pen = QPen(_FRAME_HL, _HL_WIDTH)
        for box in self._boxes_by_frame.get(self._current_frame, ()):
            self._frame_hl_items.append(
                self._scene.addLine(box.left(), box.top(), box.right(), box.top(), pen)
            )
            self._frame_hl_items.append(
                self._scene.addLine(box.left(), box.bottom(), box.right(), box.bottom(), pen)
            )


class _CanvasView(QGraphicsView):
    """Drag-to-pan, wheel-to-zoom view that reports node clicks."""

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
        item = self.itemAt(event.pos())
        if item is None:
            return
        cell_id, t = item.data(0), item.data(1)
        if cell_id is None or t is None:
            return
        self._panel.node_activated.emit(int(t), int(cell_id))


__all__ = ["EdgeView", "LineageCanvasPanel", "NodeView"]
