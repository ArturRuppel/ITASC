from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from napari.utils.colormaps import Colormap

from cellflow.napari._correction_centroids import (
    correction_label_color_map,
    refresh_centroid_cross_layer,
    refresh_label_colormap,
)
from cellflow.napari.contact_analysis_visualization import (
    _nucleus_centroids_by_track,
    _rasterize_track_image,
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
    track_layer_name: str,
    owned_layer_names: set[str],
    color_scale: float = 0.65,
    centroid_layer_name: str | None = None,
) -> TrackedLayerLoadResult:
    """Add correction labels and the matching fading nucleus-track image."""
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

    add_correction_track_layer(
        viewer,
        labels_arr,
        name=track_layer_name,
        owned_layer_names=owned_layer_names,
        color_scale=color_scale,
    )

    if centroid_layer_name is not None:
        refresh_centroid_cross_layer(
            viewer,
            labels_arr,
            color_map=color_map,
            name=centroid_layer_name,
            owned_layer_names=owned_layer_names,
        )

    return TrackedLayerLoadResult(labels_layer=labels_layer, color_map=color_map)


def add_correction_track_layer(
    viewer: Any,
    labels: np.ndarray,
    *,
    name: str,
    owned_layer_names: set[str],
    color_scale: float = 0.65,
) -> dict[int | None, tuple[float, float, float, float] | str]:
    """Add the correction-owned fading nucleus-track image layer."""
    labels_arr = np.asarray(labels)
    color_map = correction_label_color_map(labels_arr, color_scale=color_scale)

    shape = (
        (1, int(labels_arr.shape[0]), int(labels_arr.shape[1]))
        if labels_arr.ndim == 2
        else tuple(int(value) for value in labels_arr.shape[:3])
    )
    track_image = _rasterize_track_image(
        _nucleus_centroids_by_track(labels_arr),
        color_map,
        shape,
    )
    viewer.add_image(
        track_image,
        name=name,
        rgb=True,
        opacity=0.9,
        blending="additive",
    )
    owned_layer_names.add(name)
    return color_map


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
