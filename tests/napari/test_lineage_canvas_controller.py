"""Behavioural coverage for the docked lineage-canvas controller.

Pins the glue the correction widget relies on — refresh assembles node/edge
views from the graph + cropped tiles and docks a populated canvas synced to the
selection + current frame, an absent stack/intensity clears it, a node click
drives the navigate callback, and teardown removes the dock. The Qt panel and
the pure builders are stubbed so only the controller's assembly is under test.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cellflow.napari import lineage_canvas_controller as lcc
from cellflow.napari.lineage_canvas_controller import LineageCanvasController
from cellflow.segmentation.lineage_graph import (
    GraphEdge,
    GraphNode,
    LineageGraph,
)


@pytest.fixture
def stubbed(monkeypatch):
    """Stub the Qt panel and the pure builders the controller composes."""
    panel_factory = MagicMock(name="LineageCanvasPanel")
    monkeypatch.setattr(lcc, "LineageCanvasPanel", panel_factory)

    # One track (id 7) present at frames 0 and 1.
    graph = LineageGraph(
        n_frames=2,
        nodes=(GraphNode(7, 0), GraphNode(7, 1)),
        edges=(GraphEdge(7, 0, 1),),
    )
    monkeypatch.setattr(lcc, "build_lineage_graph", MagicMock(return_value=graph))
    monkeypatch.setattr(lcc, "assign_columns", MagicMock(return_value={7: 0}))

    tile0 = SimpleNamespace(frame=0, rgb=np.zeros((8, 8, 3), np.uint8), width=8, height=8)
    tile1 = SimpleNamespace(frame=1, rgb=np.zeros((8, 8, 3), np.uint8), width=8, height=8)
    monkeypatch.setattr(
        lcc, "build_track_film_strip",
        MagicMock(return_value=SimpleNamespace(tiles=(tile0, tile1))),
    )
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
        intensity_layer_provider=lambda: intensity_layer,
        selected_label_provider=lambda: 7,
        current_t_provider=lambda: 1,
        on_activate=on_activate or (lambda t, c: None),
        pos_dir_provider=(lambda: pos_dir),
    )


def test_refresh_assembles_nodes_edges_and_docks(stubbed):
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )

    ctrl.refresh()

    viewer.window.add_dock_widget.assert_called_once()
    panel = stubbed.return_value
    panel.set_scene.assert_called_once()
    nodes = panel.set_scene.call_args.args[0]
    edges = panel.set_scene.call_args.args[1]
    assert {n.t for n in nodes} == {0, 1}
    assert nodes[0].cell_id == 7
    assert len(edges) == 1
    # Selection + current frame are pushed after the scene is built.
    panel.set_selection.assert_called_once_with(7)
    panel.set_current_frame.assert_called_once_with(1)


def test_refresh_without_inputs_clears_panel(stubbed):
    viewer = _viewer()
    ctrl = _controller(viewer, tracked=None, intensity=None)

    ctrl.refresh()

    viewer.window.add_dock_widget.assert_not_called()
    stubbed.assert_not_called()


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


def test_refresh_restricts_crop_scan_to_occupied_frames(stubbed):
    # The graph (track 7 at frames 0,1) should drive the crop scan, so the
    # cropper is handed those frames instead of re-scanning the whole stack.
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )

    ctrl.refresh()

    import cellflow.napari.lineage_canvas_controller as mod
    _, kwargs = mod.build_track_film_strip.call_args
    assert kwargs["frames"] == [0, 1]


def test_validated_and_anchored_frames_flag_nodes(stubbed, monkeypatch):
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

    nodes = stubbed.return_value.set_scene.call_args.args[0]
    by_frame = {n.t: n for n in nodes}
    assert by_frame[0].validated and not by_frame[0].anchored
    assert by_frame[1].anchored and not by_frame[1].validated


def test_rotate_request_flips_orientation_and_rebuilds(stubbed):
    viewer = _viewer()
    ctrl = _controller(
        viewer,
        tracked=np.zeros((2, 12, 12), np.uint32),
        intensity=np.zeros((2, 12, 12), np.float32),
    )
    ctrl.refresh()
    panel = stubbed.return_value

    ctrl._on_rotate_requested()

    assert ctrl._rotated is True
    # The panel is told to flip its highlight axes and the scene is rebuilt.
    panel.set_orientation.assert_called_with(track_vertical=False)
    assert panel.set_scene.call_count == 2


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
