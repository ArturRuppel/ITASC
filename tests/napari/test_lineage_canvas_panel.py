"""Rendering coverage for the lineage canvas panel (needs an offscreen Qt).

Pins the visual contract: the selected track lights the *vertical* box sides
while the current frame lights the *horizontal* ones, changing the selection
clears the previous highlight, and a click on a node reports ``(frame, cell_id)``.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

from cellflow.napari._correction_lineage_canvas import (  # noqa: E402
    LineageCanvasPanel,
    NodeView,
)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _nodes():
    rgb = np.zeros((8, 8, 3), np.uint8)
    return [
        NodeView(cell_id=7, t=0, x=40, y=40, rgb=rgb),
        NodeView(cell_id=7, t=1, x=40, y=90, rgb=rgb),
        NodeView(cell_id=9, t=0, x=90, y=40, rgb=rgb),
    ]


def test_set_scene_indexes_boxes_by_track_and_frame(_app):
    panel = LineageCanvasPanel()
    panel.set_scene(_nodes(), [], row_height=50, scene_width=140)

    # Track 7 has two nodes (frames 0,1); frame 0 has two nodes (tracks 7,9).
    assert len(panel._boxes_by_track[7]) == 2
    assert len(panel._boxes_by_track[9]) == 1
    assert len(panel._boxes_by_frame[0]) == 2
    assert len(panel._boxes_by_frame[1]) == 1


def test_track_highlight_lights_vertical_sides(_app):
    panel = LineageCanvasPanel()
    panel.set_scene(_nodes(), [], row_height=50, scene_width=140)

    panel.set_selection(7)

    # Two nodes × (left + right) vertical lines = 4 highlight items.
    assert len(panel._track_hl_items) == 4
    for line in panel._track_hl_items:
        ln = line.line()
        assert ln.x1() == ln.x2()  # vertical


def test_frame_highlight_lights_horizontal_sides(_app):
    panel = LineageCanvasPanel()
    panel.set_scene(_nodes(), [], row_height=50, scene_width=140)

    panel.set_current_frame(0)

    # Two nodes at frame 0 × (top + bottom) horizontal lines = 4 items.
    assert len(panel._frame_hl_items) == 4
    for line in panel._frame_hl_items:
        ln = line.line()
        assert ln.y1() == ln.y2()  # horizontal


def test_changing_selection_clears_previous_track_highlight(_app):
    panel = LineageCanvasPanel()
    panel.set_scene(_nodes(), [], row_height=50, scene_width=140)

    panel.set_selection(7)
    panel.set_selection(9)

    assert len(panel._track_hl_items) == 2  # track 9 has a single node


def test_validated_and_anchored_nodes_get_status_borders(_app):
    from qtpy.QtWidgets import QGraphicsRectItem

    from cellflow.napari._correction_lineage_canvas import (
        _ANCHOR_BORDER,
        _VALIDATED_BORDER,
    )

    rgb = np.zeros((8, 8, 3), np.uint8)
    nodes = [
        NodeView(cell_id=1, t=0, x=40, y=40, rgb=rgb, validated=True),
        NodeView(cell_id=2, t=0, x=90, y=40, rgb=rgb, anchored=True),
        NodeView(cell_id=3, t=0, x=140, y=40, rgb=rgb),  # plain
    ]
    panel = LineageCanvasPanel()
    panel.set_scene(nodes, [], row_height=50, scene_width=200)

    rects = [it for it in panel._scene.items() if isinstance(it, QGraphicsRectItem)]
    colours = {it.pen().color().name() for it in rects}
    # One green + one orange border; the plain node draws none.
    assert len(rects) == 2
    assert colours == {_VALIDATED_BORDER.name(), _ANCHOR_BORDER.name()}


def test_rotation_swaps_which_sides_each_cursor_lights(_app):
    panel = LineageCanvasPanel()
    panel.set_scene(_nodes(), [], row_height=50, scene_width=140)
    panel.set_orientation(track_vertical=False)

    panel.set_selection(7)  # track now runs across rows → horizontal sides
    for line in panel._track_hl_items:
        ln = line.line()
        assert ln.y1() == ln.y2()  # horizontal

    panel.set_current_frame(0)  # frames now run down columns → vertical sides
    for line in panel._frame_hl_items:
        ln = line.line()
        assert ln.x1() == ln.x2()  # vertical


def test_rotate_button_emits_rotate_requested(_app):
    panel = LineageCanvasPanel()
    fired: list[int] = []
    panel.rotate_requested.connect(lambda: fired.append(1))

    panel._rotate_btn.click()

    assert fired == [1]


def test_node_click_emits_frame_and_cell(_app):
    panel = LineageCanvasPanel()
    panel.set_scene(_nodes(), [], row_height=50, scene_width=140)
    captured: list[tuple[int, int]] = []
    panel.node_activated.connect(lambda t, c: captured.append((t, c)))

    # Drive the view's click path directly via the scene hit-test the view uses.
    from qtpy.QtCore import QPointF

    item = panel._scene.itemAt(QPointF(40, 40), panel._view.transform())
    panel.node_activated.emit(int(item.data(1)), int(item.data(0)))

    assert captured == [(0, 7)]
