"""Coverage for the native-dock takeover used by correction focus mode.

These pin the reversibility contract the correction widget depends on: hiding
returns the docks' prior visibility, restoring puts each back exactly, and a
viewer whose private ``_qt_viewer`` is missing degrades to a no-op instead of
raising into the activation lifecycle.
"""
from __future__ import annotations

from types import SimpleNamespace

from itasc.napari.correction._correction_takeover import (
    hide_native_docks,
    restore_native_docks,
)


class _Dock:
    def __init__(self, visible: bool) -> None:
        self._visible = visible

    def isVisible(self) -> bool:  # noqa: N802 (Qt API shape)
        return self._visible

    def setVisible(self, value: bool) -> None:  # noqa: N802 (Qt API shape)
        self._visible = bool(value)


def _viewer(list_visible=True, controls_visible=False):
    qt = SimpleNamespace(
        dockLayerList=_Dock(list_visible),
        dockLayerControls=_Dock(controls_visible),
    )
    return SimpleNamespace(window=SimpleNamespace(_qt_viewer=qt))


def test_hide_records_prior_visibility_and_hides():
    viewer = _viewer(list_visible=True, controls_visible=False)

    prior = hide_native_docks(viewer)

    assert prior == {"dockLayerList": True, "dockLayerControls": False}
    qt = viewer.window._qt_viewer
    assert qt.dockLayerList.isVisible() is False
    assert qt.dockLayerControls.isVisible() is False


def test_restore_returns_docks_to_prior_state():
    viewer = _viewer(list_visible=True, controls_visible=False)
    prior = hide_native_docks(viewer)

    restore_native_docks(viewer, prior)

    qt = viewer.window._qt_viewer
    assert qt.dockLayerList.isVisible() is True
    assert qt.dockLayerControls.isVisible() is False  # stays hidden as before


def test_missing_qt_viewer_is_a_noop():
    viewer = SimpleNamespace(window=SimpleNamespace())

    assert hide_native_docks(viewer) == {}
    restore_native_docks(viewer, {"dockLayerList": True})  # must not raise


def test_restore_with_empty_state_is_a_noop():
    viewer = _viewer()
    restore_native_docks(viewer, {})
    restore_native_docks(viewer, None)
