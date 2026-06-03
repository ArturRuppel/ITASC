"""Behavioural coverage for the docked lineage-swimlane controller.

Pins the glue: refresh builds the model and docks a populated panel synced to
the selection + current frame, an absent stack clears it, lane activation drives
the navigate callback, and teardown removes the dock. The Qt panel and the pure
``build_lineage`` are stubbed so only the controller's glue is under test.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cellflow.napari import lineage_controller as lc
from cellflow.napari.lineage_controller import LineageController


@pytest.fixture
def stubbed_panel(monkeypatch):
    panel_factory = MagicMock(name="LineagePanel")
    monkeypatch.setattr(lc, "LineagePanel", panel_factory)
    monkeypatch.setattr(lc, "build_lineage", MagicMock(return_value="model"))
    return panel_factory


def _viewer():
    window = SimpleNamespace(
        add_dock_widget=MagicMock(return_value=object()),
        remove_dock_widget=MagicMock(),
    )
    return SimpleNamespace(window=window)


def _controller(viewer, *, tracked=None, selected=5, current_t=2, on_activate=None):
    return LineageController(
        viewer,
        tracked_data_provider=lambda: tracked,
        selected_label_provider=lambda: selected,
        current_t_provider=lambda: current_t,
        on_activate=on_activate or (lambda t, c: None),
    )


def test_refresh_builds_docks_and_syncs(stubbed_panel):
    viewer = _viewer()
    ctrl = _controller(viewer, tracked=np.zeros((2, 4, 4), np.uint32))

    ctrl.refresh()

    viewer.window.add_dock_widget.assert_called_once()
    panel = stubbed_panel.return_value
    panel.set_model.assert_called_once_with("model")
    panel.set_selection.assert_called_once_with(5)
    panel.set_current_frame.assert_called_once_with(2)


def test_refresh_without_tracked_data_does_not_dock(stubbed_panel):
    viewer = _viewer()
    ctrl = _controller(viewer, tracked=None)

    ctrl.refresh()

    viewer.window.add_dock_widget.assert_not_called()
    stubbed_panel.assert_not_called()


def test_lane_activation_invokes_navigate_callback(stubbed_panel):
    viewer = _viewer()
    on_activate = MagicMock()
    ctrl = _controller(
        viewer, tracked=np.zeros((1, 4, 4), np.uint32), on_activate=on_activate
    )
    ctrl.refresh()

    ctrl._on_track_activated(4, 9)

    on_activate.assert_called_once_with(4, 9)


def test_selection_and_frame_sync_are_noops_without_panel(stubbed_panel):
    viewer = _viewer()
    ctrl = _controller(viewer, tracked=None)

    # No panel docked yet → these must not crash.
    ctrl.set_selection(3)
    ctrl.set_current_frame(7)

    stubbed_panel.assert_not_called()


def test_teardown_removes_dock(stubbed_panel):
    viewer = _viewer()
    ctrl = _controller(viewer, tracked=np.zeros((1, 4, 4), np.uint32))
    ctrl.refresh()
    dock = viewer.window.add_dock_widget.return_value

    ctrl.teardown()

    viewer.window.remove_dock_widget.assert_called_once_with(dock)
