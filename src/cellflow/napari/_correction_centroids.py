from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


# Uniform white used for the focused track's current-frame tip cross.
FOCUS_CROSS_COLOR = (1.0, 1.0, 1.0, 1.0)

# Single neutral colour shared by the labels outline and the all-tracks overview
# in the default ("outline") view. A saturated green reads clearly against both
# the white cell z-avg and the purple nucleus z-avg, where per-id colours were
# just noise on the thin outlines. The filled (by-id) view uses the per-id
# colormap instead — see ``correction_label_color_map`` / the tracks controller.
NEUTRAL_OVERLAY_COLOR = (0.18, 0.95, 0.32, 1.0)


def neutral_label_color_map() -> dict[int | None, tuple[float, float, float, float] | str]:
    """Colour every non-zero label with the single :data:`NEUTRAL_OVERLAY_COLOR`.

    The ``None`` key is napari's catch-all default colour, so every label maps to
    the neutral colour without enumerating ids; ``0`` stays transparent.
    """
    return {None: NEUTRAL_OVERLAY_COLOR, 0: "transparent"}


def apply_neutral_label_colormap(
    labels_layer: Any,
) -> dict[int | None, tuple[float, float, float, float] | str]:
    """Set a labels layer to the single-colour neutral colormap; return the dict."""
    color_map = neutral_label_color_map()
    try:
        from napari.utils.colormaps import DirectLabelColormap

        labels_layer.colormap = DirectLabelColormap(color_dict=dict(color_map))
    except Exception:
        pass
    return color_map


def correction_label_color_map(
    labels: np.ndarray,
    *,
    color_scale: float = 0.65,
) -> dict[int | None, tuple[float, float, float, float] | str]:
    """Return deterministic non-black colors for labels present in *labels*."""
    labels_arr = np.asarray(labels)
    label_ids = sorted(int(value) for value in np.unique(labels_arr) if int(value) != 0)
    color_map: dict[int | None, tuple[float, float, float, float] | str] = {
        None: "transparent",
        0: "transparent",
    }
    for label_id in label_ids:
        rgba = _label_color(label_id)
        rgba[:3] *= float(color_scale)
        color_map[label_id] = tuple(float(channel) for channel in rgba)
    return color_map


def refresh_label_colormap(
    labels_layer: Any,
    labels: np.ndarray,
    *,
    color_scale: float = 0.65,
) -> dict[int | None, tuple[float, float, float, float] | str]:
    """Refresh a labels layer colormap and return the color dictionary."""
    color_map = correction_label_color_map(labels, color_scale=color_scale)
    try:
        from napari.utils.colormaps import DirectLabelColormap

        labels_layer.colormap = DirectLabelColormap(color_dict=color_map)
    except Exception:
        pass
    return color_map


def ensure_label_colormap_entries(
    labels_layer: Any,
    label_ids: Iterable[int],
    *,
    color_scale: float = 0.65,
) -> dict[int | None, tuple[float, float, float, float] | str]:
    """Ensure a labels layer has deterministic colors for the supplied IDs."""
    color_map = _existing_color_dict(labels_layer)
    for label_id in label_ids:
        label_id = int(label_id)
        if label_id == 0:
            continue
        rgba = _label_color(label_id)
        rgba[:3] *= float(color_scale)
        color_map[label_id] = tuple(float(channel) for channel in rgba)
    try:
        from napari.utils.colormaps import DirectLabelColormap

        labels_layer.colormap = DirectLabelColormap(color_dict=color_map)
    except Exception:
        pass
    return color_map


def _existing_color_dict(
    labels_layer: Any,
) -> dict[int | None, tuple[float, float, float, float] | str]:
    color_map: dict[int | None, tuple[float, float, float, float] | str] = {
        None: "transparent",
        0: "transparent",
    }
    raw = getattr(getattr(labels_layer, "colormap", None), "color_dict", None)
    if isinstance(raw, Mapping):
        color_map.update(raw)
    return color_map


# Per-id label colours reuse napari's own default labels palette (the cyclic
# colormap a Labels layer is coloured with out of the box) so the nuclei read
# with the same well-spread, high-contrast hues napari uses everywhere else.
# The tracks layer mirrors these exact colours by reading the labels layer's
# colour dict back (see the tracks controller).
_LABEL_CMAP = None


def _label_color(label_id: int) -> np.ndarray:
    """RGBA for *label_id* from napari's default cyclic labels colormap."""
    global _LABEL_CMAP
    if _LABEL_CMAP is None:
        from napari.utils.colormaps import label_colormap

        _LABEL_CMAP = label_colormap()
    rgba = _LABEL_CMAP.map(np.asarray([int(label_id)]))[0]
    return np.asarray(rgba, dtype=float)
