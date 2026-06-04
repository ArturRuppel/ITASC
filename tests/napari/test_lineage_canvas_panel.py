"""Rendering coverage for the swimlane overview panel (needs an offscreen Qt).

Pins the visual contract of the overview: each track is a row with time
running left→right, present runs draw as bars, per-frame status cells paint over
them (green = validated, orange = anchored), the current frame is a single
vertical guide across all rows, the selected track is a highlighted row, and a
click reports ``(frame, cell_id)`` (snapping into the nearest present frame in a
gap).
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication, QGraphicsLineItem, QGraphicsRectItem  # noqa: E402

from cellflow.napari._correction_lineage_canvas import (  # noqa: E402
    _CELL_W,
    _LANE_H,
    LaneView,
    LineageCanvasPanel,
)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _lanes():
    return [
        LaneView(
            cell_id=7, column=0, segments=((0, 2), (4, 5)),
            validated=frozenset({0}), anchored=frozenset({4}),
        ),
        LaneView(cell_id=9, column=1, segments=((0, 5),)),
    ]


def _rects(panel):
    return [it for it in panel._scene.items() if isinstance(it, QGraphicsRectItem)]


def test_overview_draws_bars_and_status_cells(_app):
    panel = LineageCanvasPanel()
    panel.set_overview(_lanes(), n_frames=6)

    # lane 7: 2 bars + 1 validated + 1 anchored = 4; lane 9: 1 bar = 1.
    assert len(_rects(panel)) == 5
    assert panel._col_to_cell == [7, 9]


def test_current_frame_is_one_vertical_guide(_app):
    panel = LineageCanvasPanel()
    panel.set_overview(_lanes(), n_frames=6)

    panel.set_current_frame(2)

    lines = [it for it in panel._scene.items() if isinstance(it, QGraphicsLineItem)]
    assert len(lines) == 1
    ln = lines[0].line()
    assert ln.x1() == ln.x2() == pytest.approx(2 * _CELL_W + _CELL_W / 2.0)


def test_selected_row_is_a_horizontal_line_that_replaces_not_accumulates(_app):
    panel = LineageCanvasPanel()
    panel.set_overview(_lanes(), n_frames=6)

    panel.set_selection(7)
    first = panel._col_item
    # The selection marker is a horizontal line centered on the track's row.
    assert isinstance(first, QGraphicsLineItem)
    ln = first.line()
    assert ln.y1() == ln.y2() == pytest.approx(0 * _LANE_H + _LANE_H / 2.0)
    assert (ln.x1(), ln.x2()) == pytest.approx((0.0, 6 * _CELL_W))

    panel.set_selection(9)
    assert first is not panel._col_item
    assert panel._col_item is not None
    panel.set_selection(0)
    assert panel._col_item is None


def test_center_on_track_scrolls_row_to_vertical_middle(_app):
    # Many lanes so the scene is taller than the viewport and can actually scroll.
    lanes = [LaneView(cell_id=c, column=c, segments=((0, 5),)) for c in range(60)]
    panel = LineageCanvasPanel()
    panel.set_overview(lanes, n_frames=6)
    panel._view.resize(200, 120)

    panel.center_on_track(40)

    target_y = 40 * _LANE_H + _LANE_H / 2.0
    center_y = panel._view.mapToScene(
        panel._view.viewport().rect().center()
    ).y()
    assert center_y == pytest.approx(target_y, abs=_LANE_H)


def test_center_on_track_ignores_unknown_track(_app):
    panel = LineageCanvasPanel()
    panel.set_overview(_lanes(), n_frames=6)
    # No matching lane → no-op, no exception.
    panel.center_on_track(999)
    panel.center_on_track(0)


def test_lane_click_emits_frame_and_cell(_app):
    panel = LineageCanvasPanel()
    panel.set_overview(_lanes(), n_frames=6)
    captured: list[tuple[int, int]] = []
    panel.node_activated.connect(lambda t, c: captured.append((t, c)))

    # row 0 (cell 7), frame 1 — present.
    panel._activate_at(1 * _CELL_W + 1, 0 * _LANE_H + 1)
    assert captured == [(1, 7)]


def test_click_in_a_gap_snaps_to_nearest_present_frame(_app):
    panel = LineageCanvasPanel()
    panel.set_overview(_lanes(), n_frames=6)
    captured: list[tuple[int, int]] = []
    panel.node_activated.connect(lambda t, c: captured.append((t, c)))

    # row 0 (cell 7), frame 3 — a gap between runs (0,2) and (4,5); snaps to 2.
    panel._activate_at(3 * _CELL_W + 1, 0 * _LANE_H + 1)
    assert captured == [(2, 7)]
