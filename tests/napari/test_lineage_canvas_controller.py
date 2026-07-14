"""Behavioural coverage for the unified-accordion controller.

Pins the glue the correction widget relies on: refresh assembles ``LaneView``
rows from the swimlane model + validation records and populates the single
accordion panel synced to the selection + current frame; selecting a track
rebuilds only that track's expanded band; an absent stack clears it; a click
drives the navigate callback; teardown drops the panel reference (the host owns
the dock). The Qt panel and the pure builders are stubbed so only the
controller's assembly is under test.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from itasc.napari import lineage_canvas_controller as lcc
from itasc.napari.lineage_canvas_controller import LineageCanvasController
from itasc.segmentation.lineage import LineageModel, TrackLane, TrackSegment


@pytest.fixture
def stubbed(monkeypatch):
    """Stub the Qt accordion panel and the pure builders the controller composes.

    Returns the ``TrackAccordionPanel`` factory (a MagicMock).
    """
    panel_factory = MagicMock(name="TrackAccordionPanel")
    monkeypatch.setattr(lcc, "TrackAccordionPanel", panel_factory)

    # One track (id 7) present at frames 0 and 1.
    model = LineageModel(
        n_frames=2,
        lanes=(TrackLane(cell_id=7, segments=(TrackSegment(0, 1),)),),
    )
    monkeypatch.setattr(lcc, "build_lineage", MagicMock(return_value=model))
    monkeypatch.setattr(
        lcc, "build_track_film_strip",
        MagicMock(return_value=SimpleNamespace(tiles=(), frames=())),
    )
    monkeypatch.setattr(lcc, "read_validated_tracks", lambda _p: {})
    monkeypatch.setattr(lcc, "read_corrections", lambda _p: [])
    return panel_factory


def _viewer():
    window = SimpleNamespace(
        add_dock_widget=MagicMock(return_value=object()),
        remove_dock_widget=MagicMock(),
    )
    return SimpleNamespace(window=window)


def _controller(viewer, *, tracked=None, intensity=None, on_activate=None, pos_dir=None):
    intensity_layer = None if intensity is None else SimpleNamespace(data=intensity)
    return LineageCanvasController(
        viewer,
        tracked_data_provider=lambda: tracked,
        tracked_layer_provider=lambda: None,
        intensity_layer_provider=lambda: intensity_layer,
        selected_label_provider=lambda: 7,
        current_t_provider=lambda: 1,
        on_activate=on_activate or (lambda t, c: None),
        pos_dir_provider=(lambda: pos_dir),
    )


def test_refresh_assembles_lanes_and_populates_panel(stubbed):
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )

    ctrl.refresh()

    # The panel is a bare widget embedded by the host splitter — the controller
    # no longer owns napari docks.
    viewer.window.add_dock_widget.assert_not_called()
    panel = stubbed.return_value
    panel.set_overview.assert_called_once()
    lanes = panel.set_overview.call_args.args[0]
    assert [ln.cell_id for ln in lanes] == [7]
    assert lanes[0].segments == ((0, 1),)
    assert panel.set_overview.call_args.kwargs["n_frames"] == 2
    # Selection + current frame are pushed after the bars are built.
    panel.set_current_frame.assert_called_with(1)
    panel.set_selection.assert_called_with(7)
    # The selected track's expanded band is rendered onto the same panel.
    panel.set_strip.assert_called_once()


def test_refresh_without_stack_clears_panel(stubbed):
    viewer = _viewer()
    ctrl = _controller(viewer, tracked=None)

    ctrl.refresh()

    viewer.window.add_dock_widget.assert_not_called()
    stubbed.assert_not_called()


def test_validated_and_anchored_frames_flag_lanes(stubbed, monkeypatch):
    monkeypatch.setattr(lcc, "read_validated_tracks", lambda _p: {7: {0}})
    monkeypatch.setattr(
        lcc, "read_corrections",
        lambda _p: [SimpleNamespace(kind="anchor", cell_id=7, t=1)],
    )
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
        pos_dir=object(),
    )

    ctrl.refresh()

    lane = stubbed.return_value.set_overview.call_args.args[0][0]
    assert lane.validated == frozenset({0})
    assert lane.anchored == frozenset({1})


def test_set_selection_builds_band_for_occupied_frames(stubbed):
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )
    ctrl.refresh()
    lcc.build_track_film_strip.reset_mock()

    ctrl.set_selection(7)

    stubbed.return_value.set_selection.assert_called_with(7)
    _, kwargs = lcc.build_track_film_strip.call_args
    assert kwargs["frames"] == [0, 1]


def test_refresh_detail_rebuilds_band_without_rescanning_bars(stubbed):
    # The rapid swap path must not re-run the whole-stack lineage build (that
    # froze the GUI on every Z/C); it only re-crops the expanded band.
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )
    ctrl.refresh()
    lcc.build_lineage.reset_mock()
    lcc.build_track_film_strip.reset_mock()

    ctrl.refresh_detail()

    lcc.build_lineage.assert_not_called()
    lcc.build_track_film_strip.assert_called_once()
    stubbed.return_value.set_strip.assert_called()


def test_refresh_status_recolours_without_rescanning_bars(stubbed, monkeypatch):
    # Validation/anchoring only flips per-frame flags; recolour the cached lanes
    # instead of re-running the whole-stack build (which froze the GUI).
    validated = {7: set()}
    monkeypatch.setattr(lcc, "read_validated_tracks", lambda _p: validated)
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
        pos_dir=object(),
    )
    ctrl.refresh()
    lcc.build_lineage.reset_mock()

    # Now the track gets validated at both frames.
    validated[7] = {0, 1}
    ctrl.refresh_status()

    lcc.build_lineage.assert_not_called()
    lane = stubbed.return_value.set_overview.call_args.args[0][0]
    assert lane.cell_id == 7
    assert lane.segments == ((0, 1),)
    assert lane.validated == frozenset({0, 1})


def test_refresh_status_falls_back_to_full_refresh_when_uncached(stubbed):
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )
    # Force a panel to exist without a cached structure (never fully refreshed).
    ctrl._ensure_panel()

    ctrl.refresh_status()

    lcc.build_lineage.assert_called_once()


def test_refresh_status_is_a_noop_before_the_panel_exists(stubbed):
    viewer = _viewer()
    ctrl = _controller(viewer, tracked=np.zeros((2, 12, 12), np.uint32))

    ctrl.refresh_status()  # never refreshed → no panel yet

    lcc.build_lineage.assert_not_called()


def test_refresh_detail_is_a_noop_before_the_panel_exists(stubbed):
    viewer = _viewer()
    ctrl = _controller(viewer, tracked=np.zeros((2, 12, 12), np.uint32))

    ctrl.refresh_detail()  # never refreshed → no panel yet

    viewer.window.add_dock_widget.assert_not_called()
    lcc.build_track_film_strip.assert_not_called()


def test_node_activation_invokes_navigate_callback(stubbed):
    viewer = _viewer()
    on_activate = MagicMock()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
        on_activate=on_activate,
    )
    ctrl.refresh()

    ctrl._on_node_activated(1, 7)

    on_activate.assert_called_once_with(1, 7)


def test_center_on_track_delegates_to_panel(stubbed):
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )
    ctrl.refresh()

    ctrl.center_on_track(7)

    stubbed.return_value.center_on_track.assert_called_with(7)


def test_center_on_track_is_a_noop_before_the_panel_exists(stubbed):
    viewer = _viewer()
    ctrl = _controller(viewer, tracked=np.zeros((2, 12, 12), np.uint32))

    ctrl.center_on_track(7)  # never refreshed → no panel yet

    stubbed.assert_not_called()


def test_center_on_strip_falls_back_to_the_bar_when_no_band(stubbed):
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )
    ctrl.refresh()
    panel = stubbed.return_value
    panel.center_on_strip.return_value = False

    ctrl.center_on_strip(7)

    panel.center_on_strip.assert_called_once_with()
    panel.center_on_track.assert_called_once_with(7)


def test_center_on_strip_skips_the_bar_fallback_when_band_centered(stubbed):
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )
    ctrl.refresh()
    panel = stubbed.return_value
    panel.center_on_strip.return_value = True

    ctrl.center_on_strip(7)

    panel.center_on_track.assert_not_called()


def test_step_film_frame_delegates_to_panel_and_navigates(stubbed):
    viewer = _viewer()
    on_activate = MagicMock()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
        on_activate=on_activate,
    )
    ctrl.refresh()
    panel = stubbed.return_value
    panel.grid_neighbor_frame.return_value = 0

    ctrl.step_film_frame(dx=1)

    # Queried from the current frame (provider → 1) in the requested direction,
    # then navigated to via the same path a thumbnail click takes.
    panel.grid_neighbor_frame.assert_called_once_with(1, dx=1, dy=0, wrap=False)
    on_activate.assert_called_once_with(0, 7)


def test_step_film_frame_is_a_noop_when_no_neighbour(stubbed):
    viewer = _viewer()
    on_activate = MagicMock()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
        on_activate=on_activate,
    )
    ctrl.refresh()
    panel = stubbed.return_value
    panel.grid_neighbor_frame.return_value = None

    ctrl.step_film_frame(dy=-1)

    on_activate.assert_not_called()


def test_teardown_drops_panel(stubbed):
    # The panel is embedded as a bare widget in the host's workspace splitter and
    # deleted when that dock is torn down; teardown only drops the reference so a
    # later re-activate rebuilds it (no napari dock is owned here).
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )
    ctrl.refresh()

    ctrl.teardown()

    viewer.window.remove_dock_widget.assert_not_called()
    assert ctrl._panel is None
