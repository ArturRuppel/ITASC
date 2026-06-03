"""Rendering coverage for the combined canvas panel (needs an offscreen Qt).

Pins the visual contract of the swimlane overview: each track is a column with
time running down, present runs draw as bars, per-frame status cells paint over
them (green/orange/red), the current frame is a single horizontal guide across
all columns, the selected track is a highlighted column, and a click reports
``(frame, cell_id)`` (snapping into the nearest present frame inside a gap).
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication, QGraphicsLineItem, QGraphicsRectItem  # noqa: E402

from cellflow.napari._correction_film_strip import TrackFilmStrip  # noqa: E402
from cellflow.napari._correction_lineage_canvas import (  # noqa: E402
    _CELL_H,
    _COL_W,
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
            validated=frozenset({0}), anchored=frozenset({4}), errors=frozenset({5}),
        ),
        LaneView(cell_id=9, column=1, segments=((0, 5),), errors=frozenset({3})),
    ]


def _rects(panel):
    return [it for it in panel._scene.items() if isinstance(it, QGraphicsRectItem)]


def test_overview_draws_bars_and_status_cells(_app):
    panel = LineageCanvasPanel()
    panel.set_overview(_lanes(), n_frames=6)

    # lane 7: 2 bars + 1 validated + 1 anchored + 1 error = 5; lane 9: 1 bar + 1 error.
    assert len(_rects(panel)) == 7
    assert panel._col_to_cell == [7, 9]


def test_current_frame_is_one_horizontal_guide(_app):
    panel = LineageCanvasPanel()
    panel.set_overview(_lanes(), n_frames=6)

    panel.set_current_frame(2)

    lines = [it for it in panel._scene.items() if isinstance(it, QGraphicsLineItem)]
    assert len(lines) == 1
    ln = lines[0].line()
    assert ln.y1() == ln.y2() == pytest.approx(2 * _CELL_H + _CELL_H / 2.0)


def test_selected_column_highlight_replaces_not_accumulates(_app):
    panel = LineageCanvasPanel()
    panel.set_overview(_lanes(), n_frames=6)

    panel.set_selection(7)
    first = panel._col_item
    panel.set_selection(9)

    assert first is not panel._col_item
    assert panel._col_item is not None
    panel.set_selection(0)
    assert panel._col_item is None


def test_lane_click_emits_frame_and_cell(_app):
    panel = LineageCanvasPanel()
    panel.set_overview(_lanes(), n_frames=6)
    captured: list[tuple[int, int]] = []
    panel.node_activated.connect(lambda t, c: captured.append((t, c)))

    # column 0 (cell 7), frame 1 — present.
    panel._activate_at(0 * _COL_W + 1, 1 * _CELL_H + 1)
    assert captured == [(1, 7)]


def test_click_in_a_gap_snaps_to_nearest_present_frame(_app):
    panel = LineageCanvasPanel()
    panel.set_overview(_lanes(), n_frames=6)
    captured: list[tuple[int, int]] = []
    panel.node_activated.connect(lambda t, c: captured.append((t, c)))

    # column 0 (cell 7), frame 3 — a gap between runs (0,2) and (4,5); snaps to 2.
    panel._activate_at(0 * _COL_W + 1, 3 * _CELL_H + 1)
    assert captured == [(2, 7)]


def test_detail_tile_click_reports_selected_cell(_app):
    panel = LineageCanvasPanel()
    panel.set_overview(_lanes(), n_frames=6)
    panel.set_selection(9)
    captured: list[tuple[int, int]] = []
    panel.node_activated.connect(lambda t, c: captured.append((t, c)))

    panel._on_detail_frame_clicked(3)
    assert captured == [(3, 9)]


def test_set_detail_forwards_to_film_strip(_app):
    panel = LineageCanvasPanel()
    strip = TrackFilmStrip(tiles=())

    panel.set_detail(strip, title="Track 9")

    assert panel._detail._strip is strip
