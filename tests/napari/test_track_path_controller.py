"""Behavioural coverage for the single all-tracks overlay controller.

Pins the user-relevant outcomes the correction widget relies on: turning the
overlay on builds one napari ``Tracks`` layer of every track; selecting a cell
recolours it in place (the focused track viridis-by-time, the rest a faint
grey) and drops a current-frame tip cross; turning it off removes the layers;
the spotlight mask covers the selected track's footprint only while it's on.

No QApplication — a lightweight fake viewer stands in for the layer list but
builds *real* napari layer models (per the project's GL-free test pattern), so
the focus/overview data swap and colour mapping run for real.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from napari.layers import Points

from cellflow.napari.track_path_controller import (
    OVERVIEW_OPACITY,
    TRACK_LAYER,
    TRACK_TIP_LAYER,
    AllTracksController,
)


class _FakeLayers:
    def __init__(self):
        self._layers = {}
        self.selection = SimpleNamespace(active=None)

    def __contains__(self, name):
        return name in self._layers

    def __getitem__(self, name):
        return self._layers[name]

    def remove(self, item):
        name = item if isinstance(item, str) else getattr(item, "name", item)
        self._layers.pop(name, None)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayers()

    def add_layer(self, layer):
        # The controller builds the (fade-controllable Tracks subclass) layer
        # itself and hands it over, mirroring napari.Viewer.add_layer.
        self.layers._layers[layer.name] = layer
        return layer

    def add_points(self, data, *, name, **kwargs):
        layer = Points(data, name=name, **kwargs)
        self.layers._layers[name] = layer
        return layer


def _stack():
    """A (3, 4, 4) label stack: cell 5 in frames 0 and 1, cell 9 only in 0."""
    stack = np.zeros((3, 4, 4), dtype=np.uint16)
    stack[0, 0:2, 0:2] = 5
    stack[1, 1:3, 1:3] = 5
    stack[0, 3, 3] = 9
    return stack


def _controller(viewer, *, enabled=True, selected_label=5, current_t=0, stack=None):
    state = SimpleNamespace(
        enabled=enabled, selected_label=selected_label, current_t=current_t
    )
    tracked = SimpleNamespace(
        data=_stack() if stack is None else stack, colormap=None, name="tracked"
    )
    ctrl = AllTracksController(
        viewer,
        tracked_layer_provider=lambda: tracked,
        selected_label_provider=lambda: state.selected_label,
        enabled_provider=lambda: state.enabled,
        current_t_provider=lambda: state.current_t,
        status_callback=MagicMock(),
        owned_layers=set(),
    )
    return ctrl, tracked, state


def test_refresh_adds_one_tracks_layer_for_every_track():
    viewer = _FakeViewer()
    ctrl, tracked, _ = _controller(viewer, selected_label=0)

    ctrl.refresh()

    assert TRACK_LAYER in viewer.layers
    assert TRACK_LAYER in ctrl._owned_layers
    layer = viewer.layers[TRACK_LAYER]
    # Both tracks present: 2 rows for cell 5 + 1 for cell 9.
    assert sorted(set(layer.data[:, 0].astype(int))) == [5, 9]
    assert layer.data.shape[0] == 3
    # Overview: a short trailing tail (capped at the stack length) and no head.
    assert layer.tail_length == 3 and layer.head_length == 0
    # Selection handed back to the tracked layer so the user keeps editing it.
    assert viewer.layers.selection.active is tracked


def test_no_selection_is_overview_colouring():
    viewer = _FakeViewer()
    ctrl, _, _ = _controller(viewer, selected_label=0)

    ctrl.refresh()

    layer = viewer.layers[TRACK_LAYER]
    assert layer.color_by == "track_id"
    assert layer.opacity == OVERVIEW_OPACITY
    # No tip cross without a selection.
    assert TRACK_TIP_LAYER not in viewer.layers


def test_focus_keeps_only_the_selected_track_fully_opaque():
    viewer = _FakeViewer()
    ctrl, _, _ = _controller(viewer, selected_label=5)

    ctrl.refresh()

    layer = viewer.layers[TRACK_LAYER]
    # Only the focused track's vertices remain — the surrounding tracks are
    # dropped from the layer entirely (cell 9 is gone).
    assert sorted(set(layer.data[:, 0].astype(int))) == [5]
    assert layer.data.shape[0] == 2  # cell 5's two frames
    assert layer.color_by == "time"
    # Whole trajectory drawn and fully opaque (fade off pins every alpha to 1.0).
    assert layer.tail_length == 3 and layer.head_length == 3
    assert layer.opacity == 1.0
    assert layer.use_fade is False
    # Coloured by its time gradient (oldest → newest differ).
    colors = layer.track_colors
    assert not np.allclose(colors[0], colors[1])


def test_focus_drops_tip_cross_at_current_frame_centroid():
    viewer = _FakeViewer()
    ctrl, _, _ = _controller(viewer, selected_label=5, current_t=0)

    ctrl.refresh()

    assert TRACK_TIP_LAYER in viewer.layers
    tip = viewer.layers[TRACK_TIP_LAYER]
    assert tip.visible
    # Cell 5 occupies rows/cols 0..1 in frame 0 → centroid (0.5, 0.5).
    np.testing.assert_allclose(tip.data, [[0.5, 0.5]])


def test_set_current_frame_moves_then_hides_the_tip():
    viewer = _FakeViewer()
    ctrl, _, state = _controller(viewer, selected_label=5, current_t=0)
    ctrl.refresh()

    state.current_t = 1
    ctrl.set_current_frame(1)
    np.testing.assert_allclose(viewer.layers[TRACK_TIP_LAYER].data, [[1.5, 1.5]])

    # Cell 5 is absent at frame 2 → the tip hides rather than lingering.
    state.current_t = 2
    ctrl.set_current_frame(2)
    assert viewer.layers[TRACK_TIP_LAYER].visible is False


def test_set_focus_zero_restores_overview_and_hides_tip():
    viewer = _FakeViewer()
    ctrl, _, _ = _controller(viewer, selected_label=5)
    ctrl.refresh()
    layer = viewer.layers[TRACK_LAYER]
    assert layer.color_by == "time"  # focus: only the selected track, by time
    assert layer.data.shape[0] == 2

    ctrl.set_focus(0)

    # Overview restores every track and the short trailing comet.
    assert sorted(set(layer.data[:, 0].astype(int))) == [5, 9]
    assert layer.color_by == "track_id"
    assert layer.head_length == 0
    assert layer.use_fade is not False  # overview fades again (default behaviour)
    assert viewer.layers[TRACK_TIP_LAYER].visible is False


def test_refresh_clears_when_disabled():
    viewer = _FakeViewer()
    ctrl, _, state = _controller(viewer)
    ctrl.refresh()
    assert TRACK_LAYER in viewer.layers

    state.enabled = False
    ctrl.refresh()

    assert TRACK_LAYER not in viewer.layers
    assert TRACK_LAYER not in ctrl._owned_layers


def test_clear_removes_both_layers():
    viewer = _FakeViewer()
    ctrl, _, _ = _controller(viewer, selected_label=5)
    ctrl.refresh()
    assert TRACK_LAYER in viewer.layers and TRACK_TIP_LAYER in viewer.layers

    ctrl.clear()

    assert TRACK_LAYER not in viewer.layers
    assert TRACK_TIP_LAYER not in viewer.layers
    assert not ctrl._owned_layers


def test_spotlight_mask_covers_track_union_when_enabled():
    viewer = _FakeViewer()
    ctrl, _, _ = _controller(viewer, selected_label=5)

    mask = ctrl.spotlight_mask(0, 5, None)

    expected = np.any(_stack() == 5, axis=0)
    assert mask is not None
    np.testing.assert_array_equal(mask, expected)


def test_spotlight_mask_is_none_when_off_or_unselected():
    viewer = _FakeViewer()
    ctrl, _, state = _controller(viewer, enabled=False, selected_label=5)
    assert ctrl.spotlight_mask(0, 5, None) is None

    state.enabled = True
    assert ctrl.spotlight_mask(0, 0, None) is None  # nothing selected
    assert ctrl.spotlight_mask(0, 999, None) is None  # absent label → empty union
