from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from napari.utils.colormaps import Colormap

from cellflow.napari.correction._correction_centroids import (
    refresh_label_colormap,
)


@dataclass(frozen=True)
class TrackedLayerLoadResult:
    labels_layer: Any
    color_map: dict[int | None, tuple[float, float, float, float] | str]


def add_correction_image_layer(
    viewer: Any,
    data: np.ndarray,
    *,
    name: str,
    colormap: str,
    owned_layer_names: set[str],
) -> Any:
    """Add a correction-owned reference image layer."""
    arr = np.asarray(data, dtype=np.float32)
    cmap: str | Colormap = colormap
    if cmap == "bop_blue":
        cmap = Colormap(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.25, 1.0, 1.0]],
            name="bop_blue",
        )
    layer = viewer.add_image(arr, name=name, colormap=cmap, blending="minimum")
    owned_layer_names.add(name)
    return layer


def add_tracked_labels_and_track_layer(
    viewer: Any,
    labels: np.ndarray,
    *,
    labels_layer_name: str,
    owned_layer_names: set[str],
    color_scale: float = 0.65,
) -> TrackedLayerLoadResult:
    """Add the correction labels layer with its deterministic colormap.

    The trajectory overlay is no longer rasterised here — it's a live napari
    ``Tracks`` layer owned by ``AllTracksController`` (built on demand from this
    same label data), so this just lays down the labels and returns the colour
    map both share.
    """
    labels_arr = np.asarray(labels)
    labels_layer = viewer.add_labels(
        labels_arr,
        name=labels_layer_name,
        blending="additive",
    )
    labels_layer.blending = "additive"
    owned_layer_names.add(labels_layer_name)
    color_map = refresh_label_colormap(
        labels_layer,
        labels_arr,
        color_scale=color_scale,
    )
    return TrackedLayerLoadResult(labels_layer=labels_layer, color_map=color_map)


def remove_other_correction_label_layers(
    viewer: Any,
    *,
    owned_layer_names: set[str],
    label_layer_type: type,
    prefix: str = "[Correction]",
) -> None:
    """Remove stale correction-prefixed labels not owned by the active session."""
    for layer in list(viewer.layers):
        if (
            layer.name.startswith(prefix)
            and layer.name not in owned_layer_names
            and isinstance(layer, label_layer_type)
        ):
            viewer.layers.remove(layer)
