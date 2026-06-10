"""Frame the selected track in the napari image viewer.

Camera math for the focus-mode navigation: given the active tracked Labels layer
and a selected cell id, pan the viewer onto the track's whole-stack bounding box
and zoom so it fills a fixed fraction of the canvas. These are pure functions of
``(viewer, layer)`` with no widget state, so they live here rather than on the
correction widget; the widget's navigation handlers (``_navigate_to_cell``,
``_step_track``) call :func:`center_viewer_on_cell` after selecting a cell.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# The selected track's bbox is zoomed to span ~this fraction of the canvas in
# its limiting dimension (≈25% of the canvas area), leaving margin around it.
TRACK_VIEWPORT_FRACTION = 0.5


def canvas_size_px(viewer):
    """``(height, width)`` of the viewer canvas in pixels, or ``None``.

    napari's ``_qt_viewer.canvas`` is private and its shape varies across
    versions, so a missing/odd attribute degrades to "leave the zoom alone".
    """
    try:
        h, w = viewer.window._qt_viewer.canvas.size
        h, w = int(h), int(w)
        if h > 0 and w > 0:
            return h, w
    except Exception:
        logger.debug("track framing: canvas size unavailable", exc_info=True)
    return None


def center_viewer_on_cell(viewer, layer, t: int, cell_id: int) -> None:
    """Frame the whole selected track in the image viewer.

    Centers the napari camera on the track's full spatial bounding box — its
    union across *every* frame it appears in, not just frame ``t`` — and zooms so
    that box spans about :data:`TRACK_VIEWPORT_FRACTION` of the canvas in both
    directions (≈25% of the canvas area), leaving margin around the track. ``t``
    only fixes the camera's non-spatial axes (the current frame); the y/x framing
    comes from the whole track.
    """
    if layer is None or not cell_id:
        return
    try:
        data = np.asarray(layer.data)
        coords = np.nonzero(data == int(cell_id))
        if coords[-1].size == 0:
            return
        ys, xs = coords[-2], coords[-1]
        ymin, ymax = float(ys.min()), float(ys.max())
        xmin, xmax = float(xs.min()), float(xs.max())

        def to_world(y: float, x: float):
            coord = (int(t), y, x) if data.ndim >= 3 else (y, x)
            return layer.data_to_world(coord)

        world_c = to_world((ymin + ymax) / 2.0, (xmin + xmax) / 2.0)
        center = list(viewer.camera.center)
        center[-2:] = [float(world_c[-2]), float(world_c[-1])]
        viewer.camera.center = tuple(center)
        zoom_to_track_bbox(viewer, ymin, ymax, xmin, xmax, to_world)
    except Exception:
        logger.exception("lineage navigation: camera framing failed")


def zoom_to_track_bbox(viewer, ymin, ymax, xmin, xmax, to_world) -> None:
    """Zoom so the track's world bbox fills ~:data:`TRACK_VIEWPORT_FRACTION`.

    napari's ``camera.zoom`` is canvas pixels per world unit, so the world span
    visible along an axis is ``canvas_px / zoom``. Picking the smaller of the
    per-axis zooms keeps the larger bbox side at exactly the target fraction and
    the other side within it (the camera zoom is uniform). Degenerate
    (zero-extent) sides are skipped; if no canvas size is available the zoom is
    left untouched.
    """
    canvas = canvas_size_px(viewer)
    if canvas is None:
        return
    canvas_h, canvas_w = canvas
    w0, w1 = to_world(ymin, xmin), to_world(ymax, xmax)
    bbox_h = abs(float(w1[-2]) - float(w0[-2]))
    bbox_w = abs(float(w1[-1]) - float(w0[-1]))
    candidates = []
    if bbox_h > 0:
        candidates.append(TRACK_VIEWPORT_FRACTION * canvas_h / bbox_h)
    if bbox_w > 0:
        candidates.append(TRACK_VIEWPORT_FRACTION * canvas_w / bbox_w)
    if candidates:
        viewer.camera.zoom = min(candidates)
