from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


CENTROID_CROSS_SIZE = 7


@dataclass(frozen=True)
class CentroidPointPayload:
    data: np.ndarray
    border_color: np.ndarray
    face_color: np.ndarray
    features: dict[str, list[int]]


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


def build_centroid_points(
    labels: np.ndarray,
    color_map: dict[int | None, tuple[float, float, float, float] | str],
) -> CentroidPointPayload:
    """Build point positions and colors for all label centroids."""
    labels_arr = np.asarray(labels)
    if labels_arr.ndim == 2:
        labels_arr = labels_arr[np.newaxis, ...]
    if labels_arr.ndim != 3:
        return _empty_payload()

    points: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    label_ids: list[int] = []
    frames: list[int] = []

    for t in range(labels_arr.shape[0]):
        frame_centroids = _frame_centroids(labels_arr[t])
        for label_id, y, x in frame_centroids:
            color = _resolved_color(color_map, label_id)
            points.append((float(t), y, x))
            colors.append(color)
            label_ids.append(label_id)
            frames.append(int(t))

    if not points:
        return _empty_payload()

    color_array = np.asarray(colors, dtype=float)
    return CentroidPointPayload(
        data=np.asarray(points, dtype=float),
        border_color=color_array,
        face_color=color_array,
        features={"label_id": label_ids, "frame": frames},
    )


FOCUS_CROSS_COLOR = (1.0, 1.0, 1.0, 1.0)  # uniform white for the focused cell


def centroid_focus_colors(
    label_ids: Iterable[int],
    frames: Iterable[int],
    focused_lab: int,
    *,
    color_scale: float = 0.65,
) -> np.ndarray:
    """Per-point cross colors when a single cell is focused.

    The focused cell's crosses are a uniform white so they stay correct as the
    track extends or changes; every other cross keeps its deterministic per-label
    color. With ``focused_lab`` falsy, all crosses get their per-label color
    (used to restore the overview on deselect). ``frames`` is unused but kept in
    the signature for callers that pass the centroid frame feature.
    """
    label_ids = [int(value) for value in label_ids]
    n = len(label_ids)
    colors = np.empty((n, 4), dtype=float)
    for idx, label_id in enumerate(label_ids):
        rgba = _label_color(label_id)
        rgba[:3] *= float(color_scale)
        colors[idx] = rgba
    if focused_lab:
        for idx, label_id in enumerate(label_ids):
            if label_id == int(focused_lab):
                colors[idx] = FOCUS_CROSS_COLOR
    return colors


def refresh_centroid_cross_layer(
    viewer: Any,
    labels: np.ndarray,
    *,
    color_map: dict[int | None, tuple[float, float, float, float] | str],
    name: str,
    owned_layer_names: set[str],
) -> Any:
    """Add or update the correction centroid cross points layer."""
    payload = build_centroid_points(labels, color_map)
    if len(payload.data) == 0:
        if name in viewer.layers:
            try:
                viewer.layers.remove(viewer.layers[name])
                owned_layer_names.discard(name)
            except Exception:
                layer = viewer.layers[name]
                layer.data = payload.data
                layer.features = payload.features
        return None

    if name in viewer.layers:
        layer = viewer.layers[name]
        layer.data = payload.data
        layer.border_color = payload.border_color
        layer.face_color = payload.face_color
        layer.features = payload.features
        layer.symbol = "cross"
        layer.size = CENTROID_CROSS_SIZE
    else:
        layer = viewer.add_points(
            payload.data,
            name=name,
            ndim=3,
            symbol="cross",
            size=CENTROID_CROSS_SIZE,
            border_color=payload.border_color,
            face_color=payload.face_color,
            features=payload.features,
            blending="translucent",
        )
    owned_layer_names.add(name)
    return layer


def update_centroid_cross_layer_for_edit(
    viewer: Any,
    labels: np.ndarray,
    *,
    color_map: dict[int | None, tuple[float, float, float, float] | str],
    name: str,
    owned_layer_names: set[str],
    frame: int,
    changed_ids: Iterable[int],
) -> Any:
    """Update centroid points for changed IDs in one frame."""
    if name not in viewer.layers:
        return refresh_centroid_cross_layer(
            viewer,
            labels,
            color_map=color_map,
            name=name,
            owned_layer_names=owned_layer_names,
        )

    labels_arr = np.asarray(labels)
    if labels_arr.ndim == 2:
        labels_arr = labels_arr[np.newaxis, ...]
    if labels_arr.ndim != 3:
        return refresh_centroid_cross_layer(
            viewer,
            labels,
            color_map=color_map,
            name=name,
            owned_layer_names=owned_layer_names,
        )

    frame = int(frame)
    if frame < 0 or frame >= labels_arr.shape[0]:
        return viewer.layers[name]

    ids = {int(label_id) for label_id in changed_ids if int(label_id) != 0}
    if not ids:
        return viewer.layers[name]

    layer = viewer.layers[name]
    data = _as_point_array(getattr(layer, "data", np.empty((0, 3))))
    features = getattr(layer, "features", None)
    label_ids = [int(value) for value in _feature_values(features, "label_id")]
    frames = [int(value) for value in _feature_values(features, "frame")]
    colors = _as_color_array(getattr(layer, "border_color", None), len(data))

    keep_indices = [
        idx
        for idx, (point_frame, label_id) in enumerate(
            zip(frames, label_ids, strict=False)
        )
        if not (point_frame == frame and label_id in ids)
    ]
    new_data = [data[idx] for idx in keep_indices]
    new_label_ids = [label_ids[idx] for idx in keep_indices]
    new_frames = [frames[idx] for idx in keep_indices]
    new_colors = [colors[idx] for idx in keep_indices]

    frame_centroids = {
        label_id: (y, x)
        for label_id, y, x in _frame_centroids(labels_arr[frame])
        if label_id in ids
    }
    for label_id in sorted(ids):
        centroid = frame_centroids.get(label_id)
        if centroid is None:
            continue
        y, x = centroid
        new_data.append(np.asarray([frame, y, x], dtype=float))
        new_label_ids.append(label_id)
        new_frames.append(frame)
        new_colors.append(np.asarray(_resolved_color(color_map, label_id), dtype=float))

    layer.data = (
        np.asarray(new_data, dtype=float).reshape((-1, 3))
        if new_data
        else np.empty((0, 3), dtype=float)
    )
    color_array = (
        np.asarray(new_colors, dtype=float).reshape((-1, 4))
        if new_colors
        else np.empty((0, 4), dtype=float)
    )
    layer.border_color = color_array
    layer.face_color = color_array
    layer.features = {"label_id": new_label_ids, "frame": new_frames}
    layer.symbol = "cross"
    layer.size = CENTROID_CROSS_SIZE
    owned_layer_names.add(name)
    return layer


def _empty_payload() -> CentroidPointPayload:
    return CentroidPointPayload(
        data=np.empty((0, 3), dtype=float),
        border_color=np.empty((0, 4), dtype=float),
        face_color=np.empty((0, 4), dtype=float),
        features={"label_id": [], "frame": []},
    )


def _resolved_color(
    color_map: dict[int | None, tuple[float, float, float, float] | str],
    label_id: int,
) -> tuple[float, float, float, float]:
    raw = color_map.get(int(label_id))
    if isinstance(raw, str) or raw is None:
        rgba = _label_color(label_id)
        return tuple(float(channel) for channel in rgba)
    return tuple(float(channel) for channel in raw)


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


def _as_point_array(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return np.empty((0, 3), dtype=float)
    return arr.reshape((-1, 3))


def _as_color_array(colors: Any, length: int) -> np.ndarray:
    arr = np.asarray(colors, dtype=float) if colors is not None else np.empty((0, 4))
    if arr.size == length * 4:
        return arr.reshape((length, 4))
    return np.zeros((length, 4), dtype=float)


def _frame_centroids(frame: np.ndarray) -> list[tuple[int, float, float]]:
    frame_arr = np.asarray(frame)
    if frame_arr.ndim != 2:
        return []

    flat = frame_arr.ravel()
    foreground = flat != 0
    if not np.any(foreground):
        return []

    labels, inverse, counts = np.unique(
        flat[foreground],
        return_inverse=True,
        return_counts=True,
    )
    flat_indices = np.arange(flat.size, dtype=float)[foreground]
    y_coords = flat_indices // float(frame_arr.shape[1])
    x_coords = flat_indices % float(frame_arr.shape[1])
    sum_y = np.bincount(inverse, weights=y_coords, minlength=len(labels))
    sum_x = np.bincount(inverse, weights=x_coords, minlength=len(labels))

    return [
        (int(label_id), float(sum_y[idx] / counts[idx]), float(sum_x[idx] / counts[idx]))
        for idx, label_id in enumerate(labels)
    ]


def _feature_values(features: Any, name: str) -> list[Any]:
    if features is None:
        return []
    try:
        return list(features[name])
    except Exception:
        return []


def _label_color(label_id: int) -> np.ndarray:
    hue = ((int(label_id) - 1) * 0.618033988749895) % 1.0
    return np.asarray((*_hsv_to_rgb(hue, 0.65, 0.9), 1.0), dtype=float)


def _hsv_to_rgb(
    hue: float,
    saturation: float,
    value: float,
) -> tuple[float, float, float]:
    h = (float(hue) % 1.0) * 6.0
    i = int(h)
    f = h - i
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * f)
    t = value * (1.0 - saturation * (1.0 - f))
    if i == 0:
        return value, t, p
    if i == 1:
        return q, value, p
    if i == 2:
        return p, value, t
    if i == 3:
        return p, q, value
    if i == 4:
        return t, p, value
    return value, p, q
