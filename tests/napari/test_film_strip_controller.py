"""Behavioural coverage for the docked track film-strip controller.

These pin the user-relevant outcomes the correction widget relies on — select a
track and a populated strip gets docked; deselect and it clears; leave the mode
and the dock is removed — without standing up a real QApplication. The Qt panel
and the pure strip builder are stubbed so the controller's own glue (gating,
docking lifecycle, current-frame sync) is what's under test.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cellflow.napari import film_strip_controller as fsc
from cellflow.napari.film_strip_controller import FilmStripController


def _layer():
    """A stand-in labels/intensity layer: array data, no colormap dict."""
    return SimpleNamespace(data=np.zeros((3, 4, 4), dtype=np.uint16), colormap=None)


@pytest.fixture
def stubbed_panel(monkeypatch):
    """Replace the Qt panel + pixel builder; return the panel factory mock."""
    panel_factory = MagicMock(name="TrackFilmStripPanel")
    monkeypatch.setattr(fsc, "TrackFilmStripPanel", panel_factory)
    monkeypatch.setattr(
        fsc,
        "build_track_film_strip",
        MagicMock(return_value=SimpleNamespace(frames=(0, 1, 2))),
    )
    return panel_factory


def _controller(viewer, *, selected_label, tracked=None, intensity=None):
    tracked = _layer() if tracked is None else tracked
    intensity = _layer() if intensity is None else intensity
    return FilmStripController(
        viewer,
        tracked_layer_provider=lambda: tracked,
        intensity_layer_provider=lambda: intensity,
        pos_dir_provider=lambda: None,
        current_t_provider=lambda: 2,
        selected_label_provider=lambda: selected_label,
        tile_px=96,
    )


def _viewer():
    window = SimpleNamespace(
        add_dock_widget=MagicMock(return_value=object()),
        remove_dock_widget=MagicMock(),
    )
    return SimpleNamespace(window=window)


def test_refresh_docks_and_populates_for_selected_track(stubbed_panel):
    viewer = _viewer()
    ctrl = _controller(viewer, selected_label=5)

    ctrl.refresh()

    # A panel is docked exactly once, captioned for the selected track, and the
    # current frame is highlighted.
    viewer.window.add_dock_widget.assert_called_once()
    panel = stubbed_panel.return_value
    title = panel.set_strip.call_args.kwargs.get("title") or panel.set_strip.call_args[0][1]
    assert "Track 5" in title
    panel.set_current_frame.assert_called_with(2)


def test_refresh_without_selection_does_not_dock(stubbed_panel):
    viewer = _viewer()
    ctrl = _controller(viewer, selected_label=0)

    ctrl.refresh()

    # Nothing selected and no panel yet → nothing gets docked.
    viewer.window.add_dock_widget.assert_not_called()
    stubbed_panel.assert_not_called()


def test_teardown_removes_dock_and_forgets_panel(stubbed_panel):
    viewer = _viewer()
    ctrl = _controller(viewer, selected_label=5)
    ctrl.refresh()
    dock = viewer.window.add_dock_widget.return_value

    ctrl.teardown()

    viewer.window.remove_dock_widget.assert_called_once_with(dock)
    # Panel is forgotten: a later current-frame sync is a no-op, not a crash.
    panel = stubbed_panel.return_value
    panel.set_current_frame.reset_mock()
    ctrl.set_current_frame(7)
    panel.set_current_frame.assert_not_called()


def test_set_tile_size_applies_live_to_open_panel(stubbed_panel):
    viewer = _viewer()
    ctrl = _controller(viewer, selected_label=5)
    ctrl.refresh()

    ctrl.set_tile_size(160)

    stubbed_panel.return_value.set_tile_size.assert_called_once_with(160)
