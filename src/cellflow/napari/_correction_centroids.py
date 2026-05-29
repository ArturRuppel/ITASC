from __future__ import annotations

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
        frame = labels_arr[t]
        for label_id in sorted(int(value) for value in np.unique(frame) if int(value) != 0):
            yy, xx = np.nonzero(frame == label_id)
            if yy.size == 0:
                continue
            color = _resolved_color(color_map, label_id)
            points.append((float(t), float(np.mean(yy)), float(np.mean(xx))))
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
