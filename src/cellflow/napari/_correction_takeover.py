"""Hide and restore napari's native dock chrome for correction *focus mode*.

The correction widget is effectively a standalone app: when it activates we hand
its panels the whole right column by hiding napari's built-in layer-list and
layer-controls docks, and restore them (to exactly their prior visibility) when
it deactivates. Everything is defensive — napari's private ``_qt_viewer`` dock
attributes differ across versions, so a missing attribute degrades to "leave it
alone" rather than raising into the correction lifecycle.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# napari ``QtViewer`` dock attributes for the chrome we reclaim. ``dockConsole``
# is intentionally left alone (it starts hidden and users toggle it themselves).
_NATIVE_DOCKS = ("dockLayerList", "dockLayerControls")


def _qt_viewer(viewer):
    return getattr(getattr(viewer, "window", None), "_qt_viewer", None)


def hide_native_docks(viewer) -> dict[str, bool]:
    """Hide napari's native docks, returning their prior visibility for restore.

    The returned mapping is opaque — pass it back to :func:`restore_native_docks`.
    An empty mapping means nothing was hidden (and restore is a no-op).
    """
    qt = _qt_viewer(viewer)
    if qt is None:
        return {}
    prior: dict[str, bool] = {}
    for name in _NATIVE_DOCKS:
        dock = getattr(qt, name, None)
        if dock is None:
            continue
        try:
            prior[name] = bool(dock.isVisible())
            dock.setVisible(False)
        except Exception:
            logger.exception("could not hide native dock %s", name)
    return prior


def restore_native_docks(viewer, prior: dict[str, bool] | None) -> None:
    """Restore docks hidden by :func:`hide_native_docks` to their prior state."""
    if not prior:
        return
    qt = _qt_viewer(viewer)
    if qt is None:
        return
    for name, was_visible in prior.items():
        dock = getattr(qt, name, None)
        if dock is None:
            continue
        try:
            dock.setVisible(bool(was_visible))
        except Exception:
            logger.exception("could not restore native dock %s", name)


__all__ = ["hide_native_docks", "restore_native_docks"]
