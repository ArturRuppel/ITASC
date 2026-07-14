"""Rendering + interaction coverage for the unified track accordion panel.

Pins the visual + behavioural contract of :class:`TrackAccordionPanel`, the
single panel that replaced the swimlane overview and the per-track film strip:

* a collapsed track draws as one thin bar (present runs + per-frame status);
* selecting a track keeps its bar as a header and grows a wrapped thumbnail band
  directly beneath it (one track expanded at a time);
* ``cell_w`` is width-derived, so the whole time axis fits the panel on resize;
* Ctrl+wheel is region-aware (tiles over the band, bar height elsewhere);
* a bar click reports ``(frame, cell_id)`` (snapping into a gap) and a thumbnail
  click reports its frame.

A QApplication is required for QtGui, so the module skips cleanly if Qt cannot
start headless.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtCore import QPoint, QPointF, Qt  # noqa: E402
from qtpy.QtGui import QWheelEvent  # noqa: E402
from qtpy.QtWidgets import QApplication, QGraphicsLineItem  # noqa: E402

from itasc.napari.correction._correction_track_accordion import (  # noqa: E402
    _LEFT_GUTTER,
    LaneView,
    TrackAccordionPanel,
)
from itasc.napari.correction._correction_track_path import FilmStripTile, TrackFilmStrip  # noqa: E402


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


def _strip(*frames):
    return TrackFilmStrip(
        tiles=tuple(
            FilmStripTile(frame=f, rgb=np.zeros((5, 5, 3), np.uint8)) for f in frames
        )
    )


def _rects(panel):
    from qtpy.QtWidgets import QGraphicsRectItem

    return [it for it in panel._scene.items() if isinstance(it, QGraphicsRectItem)]


def test_collapsed_bars_geometry(_app):
    panel = TrackAccordionPanel()
    panel.set_overview(_lanes(), n_frames=6)

    # lane 7: 2 present-run bars + 1 validated + 1 anchored = 4; lane 9: 1 bar = 1.
    assert len(_rects(panel)) == 5
    assert [cid for _, _, cid in panel._bar_rows] == [7, 9]
    # No track selected → no expanded band, no selection cursor.
    assert panel._band_range is None
    assert panel._col_item is None


def test_current_frame_is_one_vertical_guide(_app):
    panel = TrackAccordionPanel()
    panel.set_overview(_lanes(), n_frames=6)

    panel.set_current_frame(2)

    lines = [it for it in panel._scene.items() if isinstance(it, QGraphicsLineItem)]
    # Exactly one vertical guide across all rows (no selection line yet).
    assert len(lines) == 1
    ln = lines[0].line()
    expected_x = _LEFT_GUTTER + 2 * panel._cell_w + panel._cell_w / 2.0
    assert ln.x1() == ln.x2() == pytest.approx(expected_x)


def test_selection_keeps_bar_and_grows_band_beneath(_app):
    panel = TrackAccordionPanel(tile_px=32)
    panel.set_overview(_lanes(), n_frames=6)

    panel.set_selection(7)
    panel.set_strip(_strip(0, 4))

    # The selected track's bar is retained as a header row...
    sel_bar = next((top, bot) for top, bot, cid in panel._bar_rows if cid == 7)
    # ...with a selection cursor on it...
    assert panel._col_item is not None
    # ...and a thumbnail band for exactly that track's frames beneath the bar.
    assert set(panel._tile_rects) == {0, 4}
    assert panel._band_range is not None
    assert panel._band_range[0] >= sel_bar[1]


def test_only_one_track_expands_at_a_time(_app):
    panel = TrackAccordionPanel(tile_px=32)
    panel.set_overview(_lanes(), n_frames=6)
    panel.set_selection(7)
    panel.set_strip(_strip(0, 4))
    assert set(panel._tile_rects) == {0, 4}

    # Re-selecting another track (its strip handed in) drops the old band.
    panel.set_selection(9)
    panel.set_strip(_strip(1, 2, 3))
    assert set(panel._tile_rects) == {1, 2, 3}


def test_cell_w_is_width_derived_and_fits(_app, monkeypatch):
    panel = TrackAccordionPanel()
    panel.set_overview(_lanes(), n_frames=6)

    # The whole 6-frame axis spans the panel width (no horizontal scroll).
    assert _LEFT_GUTTER + 6 * panel._cell_w == pytest.approx(panel._row_width())

    # cell_w is re-derived from the available width on each (re)layout — a wider
    # panel yields proportionally wider cells, a narrower one narrower, and the
    # axis fits exactly either way. (An unshown view's viewport ignores resize(),
    # so the width is injected directly; resizeEvent re-runs the same _relayout.)
    monkeypatch.setattr(panel, "_row_width", lambda: 630.0)
    panel._relayout()
    wide = panel._cell_w
    assert _LEFT_GUTTER + 6 * wide == pytest.approx(630.0)

    monkeypatch.setattr(panel, "_row_width", lambda: 330.0)
    panel._relayout()
    narrow = panel._cell_w
    assert _LEFT_GUTTER + 6 * narrow == pytest.approx(330.0)
    assert narrow < wide


def test_ctrl_wheel_is_region_aware(_app):
    panel = TrackAccordionPanel(tile_px=64)
    panel.set_overview(_lanes(), n_frames=6)
    panel.set_selection(7)
    panel.set_strip(_strip(0, 4))

    band_top, band_bottom = panel._band_range
    bar_top, bar_bottom, _ = panel._bar_rows[0]
    y_in_band = (band_top + band_bottom) / 2.0
    y_over_bar = (bar_top + bar_bottom) / 2.0
    assert panel._over_band(y_in_band)
    assert not panel._over_band(y_over_bar)

    # Over the band → tiles resize; bar height is untouched.
    tile0, lane0 = panel._tile_px, panel._lane_h
    panel._ctrl_wheel_zoom(up=True, scene_y=y_in_band)
    assert panel._tile_px > tile0
    assert panel._lane_h == lane0

    # Over a bar → bar height changes; tile size is untouched.
    tile1, lane1 = panel._tile_px, panel._lane_h
    panel._ctrl_wheel_zoom(up=True, scene_y=y_over_bar)
    assert panel._lane_h > lane1
    assert panel._tile_px == tile1

    # Wheel-down reverses the bar-height change.
    lane2 = panel._lane_h
    panel._ctrl_wheel_zoom(up=False, scene_y=y_over_bar)
    assert panel._lane_h < lane2


def test_plain_wheel_scrolls_and_does_not_zoom(_app):
    panel = TrackAccordionPanel(tile_px=48)
    panel.set_overview(_lanes(), n_frames=6)
    panel.set_selection(7)
    panel.set_strip(_strip(0, 4))
    before_tile, before_lane = panel._tile_px, panel._lane_h

    event = QWheelEvent(
        QPointF(10.0, 10.0),
        QPointF(10.0, 10.0),
        QPoint(0, -120),
        QPoint(0, -120),
        Qt.NoButton,
        Qt.NoModifier,           # plain wheel — no Ctrl
        Qt.ScrollUpdate,
        False,
    )
    panel._view.wheelEvent(event)

    # A plain wheel scrolls (handled by the base view); it must not zoom either
    # the tiles or the bar height.
    assert panel._tile_px == before_tile
    assert panel._lane_h == before_lane


def test_bar_click_emits_frame_and_cell(_app):
    panel = TrackAccordionPanel()
    panel.set_overview(_lanes(), n_frames=6)
    captured: list[tuple[int, int]] = []
    panel.node_activated.connect(lambda t, c: captured.append((t, c)))

    top, bottom, _ = panel._bar_rows[0]  # cell 7's row
    x = _LEFT_GUTTER + 1 * panel._cell_w + 1  # frame 1 — present
    panel._activate_at(x, (top + bottom) / 2.0)
    assert captured == [(1, 7)]


def test_bar_click_in_a_gap_snaps_to_nearest_present_frame(_app):
    panel = TrackAccordionPanel()
    panel.set_overview(_lanes(), n_frames=6)
    captured: list[tuple[int, int]] = []
    panel.node_activated.connect(lambda t, c: captured.append((t, c)))

    top, bottom, _ = panel._bar_rows[0]  # cell 7, runs (0,2) and (4,5)
    x = _LEFT_GUTTER + 3 * panel._cell_w + 1  # frame 3 — a gap; snaps to 2
    panel._activate_at(x, (top + bottom) / 2.0)
    assert captured == [(2, 7)]


def test_thumbnail_click_emits_frame(_app):
    panel = TrackAccordionPanel(tile_px=32)
    panel.set_overview(_lanes(), n_frames=6)
    panel.set_selection(7)
    panel.set_strip(_strip(0, 4))
    captured: list[int] = []
    panel.frame_clicked.connect(lambda f: captured.append(f))

    rect = panel._tile_rects[4]
    panel._activate_at(rect.center().x(), rect.center().y())
    assert captured == [4]


# ── arrow-key film-strip grid navigation ────────────────────────────────────


def _grid_panel(_app):
    """A panel with a hand-laid two-row band: row0=[0,1,2], row1=[3,4]."""
    from qtpy.QtCore import QRectF

    panel = TrackAccordionPanel(tile_px=32)
    panel._tile_rects = {
        0: QRectF(30, 0, 32, 32),
        1: QRectF(64, 0, 32, 32),
        2: QRectF(98, 0, 32, 32),
        3: QRectF(30, 40, 32, 32),
        4: QRectF(64, 40, 32, 32),
    }
    return panel


def test_grid_neighbor_left_right_walk_reading_order(_app):
    panel = _grid_panel(_app)
    # Right steps a column, wrapping from a row end into the next row's start.
    assert panel.grid_neighbor_frame(1, dx=1) == 2
    assert panel.grid_neighbor_frame(2, dx=1) == 3
    # Left steps back; nothing before the first tile.
    assert panel.grid_neighbor_frame(3, dx=-1) == 2
    assert panel.grid_neighbor_frame(0, dx=-1) is None


def test_grid_neighbor_up_down_jump_rows_keeping_column(_app):
    panel = _grid_panel(_app)
    assert panel.grid_neighbor_frame(1, dy=1) == 4   # col 1 → row 1 col 1
    assert panel.grid_neighbor_frame(4, dy=-1) == 1   # back up a row
    # A shorter last row clamps the column to its final tile.
    assert panel.grid_neighbor_frame(2, dy=1) == 4
    # No row beyond the edges.
    assert panel.grid_neighbor_frame(3, dy=1) is None
    assert panel.grid_neighbor_frame(0, dy=-1) is None


def test_grid_neighbor_off_band_steps_into_first_tile(_app):
    panel = _grid_panel(_app)
    assert panel.grid_neighbor_frame(99, dx=1) == 0
    # An empty band has nowhere to go.
    panel._tile_rects = {}
    assert panel.grid_neighbor_frame(0, dx=1) is None


def test_grid_neighbor_wrap_loops_around_the_ends(_app):
    panel = _grid_panel(_app)  # flat order [0,1,2,3,4]; rows [0,1,2] / [3,4]
    # Reading-order wrap: past the last tile → first, before the first → last.
    assert panel.grid_neighbor_frame(4, dx=1, wrap=True) == 0
    assert panel.grid_neighbor_frame(0, dx=-1, wrap=True) == 4
    # Without wrap the same steps stop at the edge.
    assert panel.grid_neighbor_frame(4, dx=1, wrap=False) is None
    # Row wrap: down off the last row → top row (column clamped), up → bottom.
    assert panel.grid_neighbor_frame(3, dy=1, wrap=True) == 0
    assert panel.grid_neighbor_frame(0, dy=-1, wrap=True) == 3


def test_center_on_strip_reports_whether_a_band_existed(_app):
    panel = TrackAccordionPanel(tile_px=32)
    panel.set_overview(_lanes(), n_frames=6)
    # No selection yet → no band to center on.
    assert panel.center_on_strip() is False
    panel.set_selection(7)
    panel.set_strip(_strip(0, 4))
    assert panel.center_on_strip() is True
