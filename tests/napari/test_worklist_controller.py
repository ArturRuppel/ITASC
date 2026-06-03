"""Behavioural coverage for the docked error-worklist controller.

Pins the glue the correction widget relies on — refresh scans and docks a
populated panel, an empty/absent stack clears it, activating a row drives the
host navigate callback, and teardown removes the dock — without a real
QApplication. The Qt panel and the pure scanner are stubbed.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cellflow.napari import worklist_controller as wc
from cellflow.napari.worklist_controller import WorklistController


@pytest.fixture
def stubbed_panel(monkeypatch):
    panel_factory = MagicMock(name="ErrorWorklistPanel")
    monkeypatch.setattr(wc, "ErrorWorklistPanel", panel_factory)
    monkeypatch.setattr(
        wc, "scan_errors", MagicMock(return_value=["e0", "e1"]),
    )
    return panel_factory


def _viewer():
    window = SimpleNamespace(
        add_dock_widget=MagicMock(return_value=object()),
        remove_dock_widget=MagicMock(),
    )
    return SimpleNamespace(window=window)


def _controller(viewer, *, tracked=None, contours=None, on_activate=None):
    return WorklistController(
        viewer,
        tracked_data_provider=lambda: tracked,
        contours_provider=lambda: contours,
        on_activate=on_activate or (lambda t, c: None),
    )


def test_refresh_scans_docks_and_populates(stubbed_panel):
    viewer = _viewer()
    tracked = np.zeros((2, 4, 4), dtype=np.uint32)
    ctrl = _controller(viewer, tracked=tracked)

    ctrl.refresh()

    viewer.window.add_dock_widget.assert_called_once()
    stubbed_panel.return_value.set_entries.assert_called_once_with(["e0", "e1"])


def test_refresh_without_tracked_data_does_not_dock(stubbed_panel):
    viewer = _viewer()
    ctrl = _controller(viewer, tracked=None)

    ctrl.refresh()

    viewer.window.add_dock_widget.assert_not_called()
    stubbed_panel.assert_not_called()


def test_row_activation_invokes_navigate_callback(stubbed_panel):
    viewer = _viewer()
    on_activate = MagicMock()
    ctrl = _controller(
        viewer, tracked=np.zeros((1, 4, 4), np.uint32), on_activate=on_activate
    )
    ctrl.refresh()

    # The controller connected to the panel's signal; invoke its handler.
    ctrl._on_entry_activated(3, 7)

    on_activate.assert_called_once_with(3, 7)


def test_contours_load_failure_falls_back_to_none(stubbed_panel):
    viewer = _viewer()

    def _boom():
        raise OSError("missing")

    ctrl = WorklistController(
        viewer,
        tracked_data_provider=lambda: np.zeros((1, 4, 4), np.uint32),
        contours_provider=_boom,
        on_activate=lambda t, c: None,
    )

    ctrl.refresh()  # must not raise

    # scan_errors still ran (with contours=None).
    assert wc.scan_errors.call_args[0][1] is None


def test_teardown_removes_dock(stubbed_panel):
    viewer = _viewer()
    ctrl = _controller(viewer, tracked=np.zeros((1, 4, 4), np.uint32))
    ctrl.refresh()
    dock = viewer.window.add_dock_widget.return_value

    ctrl.teardown()

    viewer.window.remove_dock_widget.assert_called_once_with(dock)
