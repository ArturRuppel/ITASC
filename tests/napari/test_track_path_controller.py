"""Behavioural coverage for the comet (whole-track overlay) controller.

Pins the user-relevant outcomes the correction widget relies on — turn the comet
on with a track selected and its overlay layer appears; turn it off (or select
nothing) and it's removed; the spotlight mask covers the track's footprint only
while the comet is on. No QApplication: a lightweight fake viewer stands in for
the layer list, and the pure ``build_track_path_overlay`` helper runs for real.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from cellflow.napari.track_path_controller import (
    TRACK_PATH_LAYER,
    TrackPathController,
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

    def add_image(self, data, *, name, rgb=False, blending=None, opacity=1.0):
        layer = SimpleNamespace(name=name, data=data, opacity=opacity)
        self.layers._layers[name] = layer
        return layer


def _stack():
    """A (3, 4, 4) label stack where cell 5 appears in frames 0 and 1."""
    stack = np.zeros((3, 4, 4), dtype=np.uint16)
    stack[0, 0:2, 0:2] = 5
    stack[1, 1:3, 1:3] = 5
    return stack


def _controller(viewer, *, enabled, selected_label, stack=None):
    tracked = SimpleNamespace(
        data=_stack() if stack is None else stack, colormap=None, name="tracked"
    )
    return (
        TrackPathController(
            viewer,
            tracked_layer_provider=lambda: tracked,
            selected_label_provider=lambda: selected_label,
            enabled_provider=lambda: enabled,
            status_callback=MagicMock(),
            owned_layers=set(),
        ),
        tracked,
    )


def test_refresh_adds_comet_layer_for_selected_track():
    viewer = _FakeViewer()
    ctrl, tracked = _controller(viewer, enabled=True, selected_label=5)

    ctrl.refresh()

    assert TRACK_PATH_LAYER in viewer.layers
    assert TRACK_PATH_LAYER in ctrl._owned_layers
    # Selection is handed back to the tracked layer so the user keeps editing it.
    assert viewer.layers.selection.active is tracked


def test_refresh_clears_when_disabled():
    viewer = _FakeViewer()
    ctrl, _ = _controller(viewer, enabled=True, selected_label=5)
    ctrl.refresh()
    assert TRACK_PATH_LAYER in viewer.layers

    # Comet turned off → next refresh removes the overlay.
    ctrl._enabled_provider = lambda: False
    ctrl.refresh()

    assert TRACK_PATH_LAYER not in viewer.layers
    assert TRACK_PATH_LAYER not in ctrl._owned_layers


def test_clear_removes_comet_layer():
    viewer = _FakeViewer()
    ctrl, _ = _controller(viewer, enabled=True, selected_label=5)
    ctrl.refresh()

    ctrl.clear()

    assert TRACK_PATH_LAYER not in viewer.layers
    assert TRACK_PATH_LAYER not in ctrl._owned_layers


def test_spotlight_mask_covers_track_union_when_enabled():
    viewer = _FakeViewer()
    ctrl, _ = _controller(viewer, enabled=True, selected_label=5)

    mask = ctrl.spotlight_mask(0, 5, None)

    # Union across frames 0 and 1 where cell 5 lives.
    expected = np.any(_stack() == 5, axis=0)
    assert mask is not None
    np.testing.assert_array_equal(mask, expected)


def test_spotlight_mask_is_none_when_off_or_unselected():
    viewer = _FakeViewer()
    ctrl, _ = _controller(viewer, enabled=False, selected_label=5)
    assert ctrl.spotlight_mask(0, 5, None) is None

    ctrl._enabled_provider = lambda: True
    # No label selected → no spotlight.
    assert ctrl.spotlight_mask(0, 0, None) is None
    # Label absent from the stack → empty union → None, not a blank mask.
    assert ctrl.spotlight_mask(0, 999, None) is None
