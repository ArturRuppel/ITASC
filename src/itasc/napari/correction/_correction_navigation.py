"""Keep the selected nucleus on screen without moving the view unnecessarily.

Selecting a cell used to centre the viewer on the track's whole-stack bounding
box and zoom it to a fixed fraction of the canvas. That re-framed the view on
every selection, which is disorienting: the scale changes under you between
edits. So selection now leaves the camera alone — with one exception.

Shift+Up/Down walks the *global* track list, so the next track can sit anywhere
in the field of view, including well outside the current viewport. Landing on a
selection you cannot see is worse than a small pan. :func:`ensure_cell_visible`
therefore pans — and *only* pans, never zooms — when the selected nucleus has
fallen outside the visible region. When it is already comfortably on screen the
camera is not touched at all.

These are pure functions of ``(viewer, layer)`` with no widget state, so they
live here rather than on the correction widget; ``_navigate_to_cell`` calls
:func:`ensure_cell_visible` after selecting a cell.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# A nucleus counts as "visible" only within this fraction of the viewport,
# measured from the centre. Cells in the outer margin are treated as off-screen
# and panned to, so the selection never ends up hugging the canvas edge.
KEEP_VISIBLE_FRACTION = 0.8


def canvas_size_px(viewer):
    """``(height, width)`` of the viewer canvas in pixels, or ``None``.

    napari's ``_qt_viewer.canvas`` is private and its shape varies across
    versions, so a missing/odd attribute degrades to "leave the camera alone".
    """
    try:
        h, w = viewer.window._qt_viewer.canvas.size
        h, w = int(h), int(w)
        if h > 0 and w > 0:
            return h, w
    except Exception:
        logger.debug("track framing: canvas size unavailable", exc_info=True)
    return None


def ensure_cell_visible(viewer, layer, t: int, cell_id: int) -> None:
    """Pan onto the selected nucleus, but only if it is off screen.

    Uses the cell's mask *on frame ``t``* — not the track's whole-stack union —
    so the test is "can the user see what they just selected right now". If its
    centroid already lies within :data:`KEEP_VISIBLE_FRACTION` of the viewport
    the camera is left exactly as it is. Otherwise the camera centre moves to the
    centroid. The zoom is never written, on either path.
    """
    if layer is None or not cell_id:
        return
    try:
        data = np.asarray(layer.data)
        frame = data[int(t)] if data.ndim >= 3 else data
        coords = np.nonzero(frame == int(cell_id))
        if coords[-1].size == 0:
            return
        cy, cx = float(coords[-2].mean()), float(coords[-1].mean())
        coord = (int(t), cy, cx) if data.ndim >= 3 else (cy, cx)
        world = layer.data_to_world(coord)
        wy, wx = float(world[-2]), float(world[-1])
        if cell_is_visible(viewer, wy, wx):
            return
        center = list(viewer.camera.center)
        center[-2:] = [wy, wx]
        viewer.camera.center = tuple(center)
    except Exception:
        logger.exception("focus-mode navigation: visibility pan failed")


def cell_is_visible(viewer, world_y: float, world_x: float) -> bool:
    """Is the world point inside the viewport, inset by the margin?

    ``camera.zoom`` is canvas pixels per world unit, so the world span visible
    along an axis is ``canvas_px / zoom``. When the canvas size or zoom cannot be
    read the answer is ``True`` — an unknown viewport means "assume visible", so
    the fallback is to leave the camera untouched rather than to pan blindly.
    """
    canvas = canvas_size_px(viewer)
    if canvas is None:
        return True
    canvas_h, canvas_w = canvas
    try:
        zoom = float(viewer.camera.zoom)
        cam = viewer.camera.center
        cam_y, cam_x = float(cam[-2]), float(cam[-1])
    except Exception:
        logger.debug("track framing: camera state unavailable", exc_info=True)
        return True
    if zoom <= 0:
        return True
    half_h = (canvas_h / zoom) / 2.0 * KEEP_VISIBLE_FRACTION
    half_w = (canvas_w / zoom) / 2.0 * KEEP_VISIBLE_FRACTION
    return abs(world_y - cam_y) <= half_h and abs(world_x - cam_x) <= half_w
