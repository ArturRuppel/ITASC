from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

__all__ = [
    "build_cell_centroid_points",
    "build_edge_shapes",
    "build_nucleus_track_shapes",
    "build_t1_edge_shapes",
    "build_t1_points",
    "add_artifact_layers",
]

_BORDER_EDGE_COLOR = np.array([0.6, 0.6, 0.6, 1.0], dtype=float)
_CELL_EDGE_COLOR = np.array([0.12156863, 0.46666667, 0.70588235, 1.0], dtype=float)
_T1_EDGE_COLOR = np.array([0.0, 1.0, 0.9, 1.0], dtype=float)
_UNLABELED_COLOR = np.array([0.7, 0.7, 0.7, 1.0], dtype=float)
_MIN_TRACK_ALPHA = 0.12


def build_cell_centroid_points(artifact: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cells = _section(artifact, "cells")
    frame = _column(cells, "frame").astype(float, copy=False)
    y = _column(cells, "centroid_y").astype(float, copy=False)
    x = _column(cells, "centroid_x").astype(float, copy=False)
    points = _stack_points(frame, y, x)
    features = {
        "frame": _column(cells, "frame"),
        "cell_id": _column(cells, "cell_id"),
        "area": _column(cells, "area").astype(float, copy=False),
        "class_label": _column(cells, "class_label"),
    }
    return points, features


def build_edge_shapes(
    artifact: Any,
    *,
    hide_border_edges: bool = False,
    color_by_id: bool = False,
    color_by_label: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    edges = _section(artifact, "edges")
    coord_y = np.asarray(_value(artifact, "coord_y"), dtype=float)
    coord_x = np.asarray(_value(artifact, "coord_x"), dtype=float)

    frame = _column(edges, "frame")
    edge_id = _column(edges, "edge_id")
    cell_a = _column(edges, "cell_a")
    cell_b = _column(edges, "cell_b")
    kind = _column(edges, "kind")
    edge_label = _column(edges, "edge_label")
    length = _column(edges, "length").astype(float, copy=False)
    is_t1_frame = _column(edges, "is_t1_frame").astype(bool, copy=False)
    coord_offset = _column(edges, "coord_offset")
    coord_count = _column(edges, "coord_count")

    lines: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    keep: list[int] = []
    for idx in range(len(frame)):
        if hide_border_edges and str(kind[idx]) == "border":
            continue
        start = int(coord_offset[idx])
        count = int(coord_count[idx])
        if count < 2:
            continue
        stop = start + count
        ys = coord_y[start:stop]
        xs = coord_x[start:stop]
        lines.append(_stack_points(np.full(len(ys), frame[idx], dtype=float), ys, xs))
        keep.append(idx)

    mask = np.asarray(keep, dtype=np.intp)
    if color_by_label:
        colors = _categorical_colors(edge_label[mask])
    elif color_by_id:
        colors = _categorical_colors(edge_id[mask])
    else:
        colors = [_edge_color_for_kind(item) for item in kind[mask]]
    features = {
        "frame": frame[mask],
        "edge_id": edge_id[mask],
        "cell_a": cell_a[mask],
        "cell_b": cell_b[mask],
        "kind": kind[mask],
        "edge_label": edge_label[mask],
        "length": length[mask],
        "is_t1_frame": is_t1_frame[mask],
        "coord_offset": coord_offset[mask],
        "coord_count": coord_count[mask],
    }
    color_array = np.asarray(colors, dtype=float)
    if color_array.size == 0:
        color_array = np.empty((0, 4), dtype=float)
    return lines, color_array, features


def build_t1_points(artifact: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t1_events = _section(artifact, "t1_events")
    frame = _column(t1_events, "frame").astype(float, copy=False)
    y = _column(t1_events, "location_y").astype(float, copy=False)
    x = _column(t1_events, "location_x").astype(float, copy=False)
    points = _stack_points(frame, y, x)
    features = {
        "t1_event_id": _column(t1_events, "t1_event_id"),
        "frame": _column(t1_events, "frame"),
        "edge_id": _column(t1_events, "edge_id"),
        "losing_cell_a": _column(t1_events, "losing_cell_a"),
        "losing_cell_b": _column(t1_events, "losing_cell_b"),
        "gaining_cell_a": _column(t1_events, "gaining_cell_a"),
        "gaining_cell_b": _column(t1_events, "gaining_cell_b"),
        "location_y": _column(t1_events, "location_y").astype(float, copy=False),
        "location_x": _column(t1_events, "location_x").astype(float, copy=False),
    }
    return points, features


def build_t1_edge_shapes(artifact: Any) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    edges = _section(artifact, "edges")
    t1_events = _section(artifact, "t1_events")
    coord_y = np.asarray(_value(artifact, "coord_y"), dtype=float)
    coord_x = np.asarray(_value(artifact, "coord_x"), dtype=float)

    edge_frame = _column(edges, "frame")
    edge_id = _column(edges, "edge_id")
    coord_offset = _column(edges, "coord_offset")
    coord_count = _column(edges, "coord_count")

    event_ids = _column(t1_events, "t1_event_id")
    event_frame = _column(t1_events, "frame")
    event_edge_id = _column(t1_events, "edge_id")
    losing_cell_a = _column(t1_events, "losing_cell_a")
    losing_cell_b = _column(t1_events, "losing_cell_b")
    gaining_cell_a = _column(t1_events, "gaining_cell_a")
    gaining_cell_b = _column(t1_events, "gaining_cell_b")
    location_y = _column(t1_events, "location_y").astype(float, copy=False)
    location_x = _column(t1_events, "location_x").astype(float, copy=False)

    lines: list[np.ndarray] = []
    feature_rows: list[dict[str, Any]] = []
    for event_idx in range(len(event_ids)):
        transition_frame = int(event_frame[event_idx])
        transition_edge_id = int(event_edge_id[event_idx])
        for side, frame in (("before", transition_frame), ("after", transition_frame + 1)):
            row_idx = _find_edge_row(edge_frame, edge_id, frame, transition_edge_id)
            if row_idx is None:
                continue
            start = int(coord_offset[row_idx])
            count = int(coord_count[row_idx])
            if count < 2:
                continue
            stop = start + count
            ys = coord_y[start:stop]
            xs = coord_x[start:stop]
            lines.append(_stack_points(np.full(len(ys), frame, dtype=float), ys, xs))
            feature_rows.append(
                {
                    "t1_event_id": event_ids[event_idx],
                    "frame": frame,
                    "transition_frame": transition_frame,
                    "transition_side": side,
                    "edge_id": transition_edge_id,
                    "losing_cell_a": losing_cell_a[event_idx],
                    "losing_cell_b": losing_cell_b[event_idx],
                    "gaining_cell_a": gaining_cell_a[event_idx],
                    "gaining_cell_b": gaining_cell_b[event_idx],
                    "location_y": location_y[event_idx],
                    "location_x": location_x[event_idx],
                }
            )

    colors = np.tile(_T1_EDGE_COLOR, (len(lines), 1))
    return lines, colors, _feature_columns(
        feature_rows,
        [
            "t1_event_id",
            "frame",
            "transition_frame",
            "transition_side",
            "edge_id",
            "losing_cell_a",
            "losing_cell_b",
            "gaining_cell_a",
            "gaining_cell_b",
            "location_y",
            "location_x",
        ],
    )


def build_nucleus_track_shapes(
    artifact: Any,
    nucleus_labels: np.ndarray,
    *,
    current_frame: int,
    color_cells_by_label: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Return past-only nucleus centroid track segments for one viewer frame."""
    centroids = _nucleus_centroids_by_track(nucleus_labels)
    color_map = _cell_color_map(artifact, color_by_label=color_cells_by_label)
    return _build_nucleus_track_shapes_from_centroids(
        centroids,
        color_map,
        current_frame=current_frame,
    )


def _build_nucleus_track_shapes_from_centroids(
    centroids: dict[int, list[tuple[int, float, float]]],
    color_map: dict[int | None, tuple[float, float, float, float] | str],
    *,
    current_frame: int,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    current_frame = max(0, int(current_frame))

    lines: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    feature_rows: list[dict[str, Any]] = []
    max_age = max(current_frame, 1)
    for cell_id in sorted(centroids):
        rows = centroids[cell_id]
        for previous, current in zip(rows[:-1], rows[1:], strict=False):
            start_frame, start_y, start_x = previous
            end_frame, end_y, end_x = current
            if int(end_frame) > current_frame or int(end_frame) != int(start_frame) + 1:
                continue
            age = current_frame - int(end_frame)
            alpha = max(_MIN_TRACK_ALPHA, 1.0 - (age / max_age))
            color = np.asarray(color_map.get(int(cell_id), _UNLABELED_COLOR), dtype=float).copy()
            color[3] = alpha
            lines.append(
                _stack_points(
                    np.asarray([current_frame, current_frame], dtype=float),
                    np.asarray([start_y, end_y], dtype=float),
                    np.asarray([start_x, end_x], dtype=float),
                )
            )
            colors.append(color)
            feature_rows.append(
                {
                    "cell_id": int(cell_id),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "age": int(age),
                }
            )

    color_array = np.asarray(colors, dtype=float)
    if color_array.size == 0:
        color_array = np.empty((0, 4), dtype=float)
    return lines, color_array, _feature_columns(
        feature_rows,
        ["cell_id", "start_frame", "end_frame", "age"],
    )


def add_artifact_layers(
    viewer: Any,
    artifact: Any,
    prefix: str = "[Artifact] ",
    *,
    color_cells_by_label: bool = False,
    color_edges_by_id: bool = False,
    color_edges_by_label: bool = False,
    hide_border_edges: bool = False,
    cell_labels: np.ndarray | None = None,
    nucleus_labels: np.ndarray | None = None,
    nucleus_track_centroids: dict | None = None,
) -> list[Any]:
    edge_lines, edge_colors, edge_features = build_edge_shapes(
        artifact,
        hide_border_edges=hide_border_edges,
        color_by_id=color_edges_by_id,
        color_by_label=color_edges_by_label,
    )
    edge_cache = _frame_shape_cache(edge_lines, edge_colors, edge_features)
    current_edge_lines, current_edge_colors, current_edge_features = _cached_frame_shapes(
        edge_cache,
        _current_frame(viewer),
    )
    t1_lines, t1_colors, t1_features = build_t1_edge_shapes(artifact)
    t1_cache = _frame_shape_cache(t1_lines, t1_colors, t1_features)
    current_t1_lines, current_t1_colors, current_t1_features = _cached_frame_shapes(
        t1_cache,
        _current_frame(viewer),
    )
    if cell_labels is None:
        cell_labels = _read_label_image(_artifact_label_path(artifact, "cell_tracked_labels_path"))
    if nucleus_labels is None:
        nucleus_labels = _read_label_image(_artifact_label_path(artifact, "nucleus_tracked_labels_path"))
    cell_kwargs: dict[str, Any] = {}
    nucleus_kwargs: dict[str, Any] = {}
    color_dict = _cell_color_map(artifact, color_by_label=color_cells_by_label)
    try:
        from napari.utils.colormaps import DirectLabelColormap
    except Exception:  # pragma: no cover - napari compatibility
        pass
    else:
        label_colormap = DirectLabelColormap(color_dict=color_dict)
        cell_kwargs["colormap"] = label_colormap
        nucleus_kwargs["colormap"] = label_colormap

    track_centroids = nucleus_track_centroids if nucleus_track_centroids is not None else _nucleus_centroids_by_track(nucleus_labels)
    track_color_map = _cell_color_map(artifact, color_by_label=color_cells_by_label)
    track_cache = _track_shape_cache(track_centroids, track_color_map)
    track_lines, track_colors, track_features = _cached_frame_shapes(track_cache, _current_frame(viewer))
    layers = [
        viewer.add_labels(
            cell_labels,
            name=f"{prefix}Cell labels",
            opacity=0.55,
            blending="translucent",
            **cell_kwargs,
        ),
        viewer.add_labels(
            nucleus_labels,
            name=f"{prefix}Nucleus labels",
            opacity=0.65,
            blending="translucent",
            **nucleus_kwargs,
        ),
        viewer.add_shapes(
            track_lines,
            ndim=3,
            name=f"{prefix}Nucleus tracks",
            shape_type="path",
            features=track_features,
            edge_width=2,
            face_color="transparent",
            blending="translucent",
            **_edge_color_kwargs(track_colors),
        ),
        viewer.add_shapes(
            current_edge_lines,
            ndim=3,
            name=f"{prefix}Edges",
            shape_type="path",
            features=current_edge_features,
            edge_width=1,
            face_color="transparent",
            blending="translucent",
            **_edge_color_kwargs(current_edge_colors),
        ),
        viewer.add_shapes(
            current_t1_lines,
            ndim=3,
            name=f"{prefix}T1 edges",
            shape_type="path",
            features=current_t1_features,
            edge_width=1,
            face_color="transparent",
            blending="translucent",
            **_edge_color_kwargs(current_t1_colors),
        ),
    ]
    _connect_frame_shape_layer_to_dims(viewer, layers[2], frame_cache=track_cache)
    _connect_frame_shape_layer_to_dims(viewer, layers[3], frame_cache=edge_cache)
    _connect_frame_shape_layer_to_dims(viewer, layers[4], frame_cache=t1_cache)
    return layers


def _section(artifact: Any, name: str) -> Any:
    if isinstance(artifact, Mapping):
        return artifact[name]
    return getattr(artifact, name)


def _value(artifact: Any, name: str) -> Any:
    if isinstance(artifact, Mapping):
        return artifact[name]
    return getattr(artifact, name)


def _artifact_label_path(artifact: Any, name: str) -> Path:
    return Path(_value(artifact, name))


def _read_label_image(path: Path) -> np.ndarray:
    return np.asarray(tifffile.imread(path))


def _current_frame(viewer: Any) -> int:
    step = getattr(getattr(viewer, "dims", None), "current_step", ())
    if not step:
        return 0
    try:
        return int(step[0])
    except Exception:
        return 0


def _edge_color_kwargs(colors: np.ndarray) -> dict[str, np.ndarray]:
    if len(colors) == 0:
        return {}
    return {"edge_color": colors}


def _frame_shape_cache(
    lines: list[np.ndarray],
    colors: np.ndarray,
    features: dict[str, np.ndarray],
) -> dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]]:
    frames = np.asarray(features.get("frame", np.asarray([], dtype=int))).astype(int, copy=False)
    cache: dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]] = {}
    for frame in sorted(set(frames.tolist())):
        indexes = np.flatnonzero(frames == int(frame))
        cache[int(frame)] = (
            [lines[int(idx)] for idx in indexes],
            colors[indexes] if len(colors) else np.empty((0, 4), dtype=float),
            {name: values[indexes] for name, values in features.items()},
        )
    return cache


def _empty_frame_shapes(
    frame_cache: dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]],
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    feature_names: list[str] = []
    for _lines, _colors, features in frame_cache.values():
        feature_names = list(features)
        break
    return [], np.empty((0, 4), dtype=float), {name: np.asarray([], dtype=object) for name in feature_names}


def _cached_frame_shapes(
    frame_cache: dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]],
    frame: int,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    return frame_cache.get(int(frame), _empty_frame_shapes(frame_cache))


def _track_shape_cache(
    centroids: dict[int, list[tuple[int, float, float]]],
    color_map: dict[int | None, tuple[float, float, float, float] | str],
) -> dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]]:
    max_frame = max(
        (int(frame) for rows in centroids.values() for frame, _y, _x in rows),
        default=0,
    )
    return {
        frame: _build_nucleus_track_shapes_from_centroids(
            centroids,
            color_map,
            current_frame=frame,
        )
        for frame in range(max_frame + 1)
    }


def _connect_frame_shape_layer_to_dims(
    viewer: Any,
    layer: Any,
    *,
    frame_cache: dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]],
) -> None:
    dims = getattr(viewer, "dims", None)
    events = getattr(dims, "events", None)
    current_step_event = getattr(events, "current_step", None)
    connect = getattr(current_step_event, "connect", None)
    if not callable(connect):
        return
    current_disconnect = getattr(current_step_event, "disconnect", None)

    def _update(_event=None) -> None:
        viewer_layers = getattr(viewer, "layers", None)
        if viewer_layers is not None:
            try:
                if layer not in viewer_layers:
                    return
            except Exception:
                pass
        lines, colors, features = _cached_frame_shapes(frame_cache, _current_frame(viewer))
        layer.data = lines
        layer.features = features
        layer.edge_color = colors if len(colors) else "transparent"
        refresh = getattr(layer, "refresh", None)
        if callable(refresh):
            refresh()

    def _disconnect() -> None:
        if callable(current_disconnect):
            try:
                current_disconnect(_update)
            except Exception:
                pass
        if callable(removed_disconnect):
            try:
                removed_disconnect(_on_removed)
            except Exception:
                pass
        try:
            frame_cache.clear()
        except Exception:
            pass

    def _on_removed(event=None) -> None:
        removed = getattr(event, "value", None)
        if removed is layer or getattr(removed, "name", None) == getattr(layer, "name", None):
            _disconnect()

    layers = getattr(viewer, "layers", None)
    layer_events = getattr(layers, "events", None)
    removed_event = getattr(layer_events, "removed", None)
    removed_connect = getattr(removed_event, "connect", None)
    removed_disconnect = getattr(removed_event, "disconnect", None)

    connect(_update)
    if callable(removed_connect):
        removed_connect(_on_removed)
    try:
        layer._cellflow_frame_shape_update = _update
        layer._cellflow_frame_shape_cleanup = _disconnect
    except Exception:
        pass


def _nucleus_centroids_by_track(nucleus_labels: np.ndarray) -> dict[int, list[tuple[int, float, float]]]:
    labels = np.asarray(nucleus_labels)
    if labels.ndim == 2:
        labels = labels[np.newaxis, ...]
    if labels.ndim > 3:
        labels = np.squeeze(labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected time-first 2D/3D nucleus labels, got shape {nucleus_labels.shape}")

    centroids: dict[int, list[tuple[int, float, float]]] = {}
    for frame_idx, frame in enumerate(labels):
        for cell_id in sorted(np.unique(frame).astype(int)):
            if cell_id == 0:
                continue
            coords = np.argwhere(frame == cell_id)
            if coords.size == 0:
                continue
            y, x = coords.mean(axis=0)
            centroids.setdefault(int(cell_id), []).append((int(frame_idx), float(y), float(x)))
    return centroids


def _cell_color_map(
    artifact: Any,
    *,
    color_by_label: bool,
) -> dict[int | None, tuple[float, float, float, float] | str]:
    if color_by_label:
        return _cell_label_color_map(artifact)

    cells = _section(artifact, "cells")
    cell_ids = np.asarray(sorted(set(_column(cells, "cell_id").astype(int))))
    cell_colors = _categorical_colors(cell_ids)
    color_map: dict[int | None, tuple[float, float, float, float] | str] = {
        None: "transparent",
        0: "transparent",
    }
    for cell_id, color in zip(cell_ids, cell_colors, strict=True):
        color_map[int(cell_id)] = tuple(float(channel) for channel in color)
    return color_map


def _cell_label_color_map(artifact: Any) -> dict[int | None, tuple[float, float, float, float] | str]:
    cells = _section(artifact, "cells")
    cell_ids = _column(cells, "cell_id")
    class_labels = _column(cells, "class_label")
    class_colors = _categorical_colors(class_labels)
    color_map: dict[int | None, tuple[float, float, float, float] | str] = {
        None: "transparent",
        0: "transparent",
    }
    for cell_id, color in zip(cell_ids, class_colors):
        cell_id_int = int(cell_id)
        if cell_id_int not in color_map:
            color_map[cell_id_int] = tuple(float(channel) for channel in color)
    return color_map


def _column(table: Any, name: str) -> np.ndarray:
    if isinstance(table, Mapping):
        value = table[name]
    else:
        value = getattr(table, name)
    return np.asarray(value)


def _find_edge_row(frame: np.ndarray, edge_id: np.ndarray, target_frame: int, target_edge_id: int) -> int | None:
    matches = np.flatnonzero(
        (frame.astype(int, copy=False) == target_frame)
        & (edge_id.astype(int, copy=False) == target_edge_id)
    )
    if len(matches) == 0:
        return None
    return int(matches[0])


def _feature_columns(rows: list[dict[str, Any]], names: list[str]) -> dict[str, np.ndarray]:
    if not rows:
        return {name: np.asarray([], dtype=object) for name in names}
    return {name: np.asarray([row[name] for row in rows]) for name in names}


def _stack_points(frame: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(frame) == 0:
        return np.empty((0, 3), dtype=float)
    return np.column_stack((frame.astype(float, copy=False), y.astype(float, copy=False), x.astype(float, copy=False)))


def _edge_color_for_kind(kind: Any) -> np.ndarray:
    if str(kind) == "border":
        return _BORDER_EDGE_COLOR
    return _CELL_EDGE_COLOR


def _categorical_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if len(values) == 0:
        return np.empty((0, 4), dtype=float)

    keys = [str(value) for value in values]
    palette = {
        key: _palette_color(idx)
        for idx, key in enumerate(sorted({key for key in keys if key != ""}))
    }
    colors = np.empty((len(values), 4), dtype=float)
    for idx, key in enumerate(keys):
        colors[idx] = _UNLABELED_COLOR if key == "" else palette[key]
    return colors


def _palette_color(index: int) -> np.ndarray:
    hue = (index * 0.618033988749895) % 1.0
    return np.asarray((*_hsv_to_rgb(hue, 0.65, 0.9), 1.0), dtype=float)


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[float, float, float]:
    sector = int(hue * 6.0)
    fraction = hue * 6.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    sector %= 6
    if sector == 0:
        return value, t, p
    if sector == 1:
        return q, value, p
    if sector == 2:
        return p, value, t
    if sector == 3:
        return p, q, value
    if sector == 4:
        return t, p, value
    return value, p, q
