"""Behavioural coverage for the docked combined-canvas controller.

Pins the glue the correction widget relies on: refresh assembles ``LaneView``
rows from the swimlane model + validation records and docks a
populated overview synced to the selection + current frame; selecting a track
rebuilds only that track's detail strip; an absent stack clears it; a click
drives the navigate callback; teardown removes the dock. The Qt panel and the
pure builders are stubbed so only the controller's assembly is under test.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cellflow.napari import lineage_canvas_controller as lcc
from cellflow.napari.lineage_canvas_controller import LineageCanvasController
from cellflow.segmentation.lineage import LineageModel, TrackLane, TrackSegment


@pytest.fixture
def stubbed(monkeypatch):
    """Stub the Qt panels and the pure builders the controller composes.

    Returns the *overview* panel factory; the film-strip panel factory is
    available as ``lcc.TrackFilmStripPanel`` (also a MagicMock).
    """
    panel_factory = MagicMock(name="LineageCanvasPanel")
    monkeypatch.setattr(lcc, "LineageCanvasPanel", panel_factory)
    monkeypatch.setattr(lcc, "TrackFilmStripPanel", MagicMock(name="TrackFilmStripPanel"))

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


def test_refresh_assembles_lanes_and_docks(stubbed):
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )

    ctrl.refresh()

    viewer.window.add_dock_widget.assert_called_once()
    panel = stubbed.return_value
    panel.set_overview.assert_called_once()
    lanes = panel.set_overview.call_args.args[0]
    assert [ln.cell_id for ln in lanes] == [7]
    assert lanes[0].segments == ((0, 1),)
    assert panel.set_overview.call_args.kwargs["n_frames"] == 2
    # Selection + current frame are pushed after the overview is built.
    panel.set_current_frame.assert_called_with(1)
    panel.set_selection.assert_called_with(7)
    # The detail strip is rendered onto the separately-docked film panel.
    lcc.TrackFilmStripPanel.return_value.set_strip.assert_called_once()


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


def test_set_selection_builds_detail_for_occupied_frames(stubbed):
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


def test_refresh_detail_rebuilds_strip_without_rescanning_overview(stubbed):
    # The rapid swap path must not re-run the whole-stack lineage build (that
    # froze the GUI on every Z/C); it only re-crops the detail strip.
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
    lcc.TrackFilmStripPanel.return_value.set_strip.assert_called()


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


def test_teardown_removes_dock(stubbed):
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )
    ctrl.refresh()
    dock = viewer.window.add_dock_widget.return_value

    ctrl.teardown()

    viewer.window.remove_dock_widget.assert_called_once_with(dock)
