from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

__all__ = [
    "build_cell_centroid_points",
    "build_edge_shapes",
    "build_t1_edge_shapes",
    "build_t1_points",
    "add_artifact_layers",
]

_BORDER_EDGE_COLOR = np.array([0.6, 0.6, 0.6, 1.0], dtype=float)
_CELL_EDGE_COLOR = np.array([0.12156863, 0.46666667, 0.70588235, 1.0], dtype=float)
_T1_EDGE_COLOR = np.array([0.0, 1.0, 0.9, 1.0], dtype=float)
_UNLABELED_COLOR = np.array([0.7, 0.7, 0.7, 1.0], dtype=float)


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


def add_artifact_layers(
    viewer: Any,
    artifact: Any,
    prefix: str = "[Artifact] ",
    *,
    color_cells_by_label: bool = False,
    color_edges_by_id: bool = False,
    color_edges_by_label: bool = False,
    hide_border_edges: bool = False,
) -> list[Any]:
    edge_lines, edge_colors, edge_features = build_edge_shapes(
        artifact,
        hide_border_edges=hide_border_edges,
        color_by_id=color_edges_by_id,
        color_by_label=color_edges_by_label,
    )
    t1_lines, t1_colors, t1_features = build_t1_edge_shapes(artifact)
    cell_labels = _read_label_image(_artifact_label_path(artifact, "cell_tracked_labels_path"))
    nucleus_labels = _read_label_image(_artifact_label_path(artifact, "nucleus_tracked_labels_path"))
    cell_kwargs: dict[str, Any] = {}
    nucleus_kwargs: dict[str, Any] = {}
    if color_cells_by_label:
        from napari.utils.colormaps import DirectLabelColormap

        color_dict = _cell_label_color_map(artifact)
        cell_kwargs["colormap"] = DirectLabelColormap(color_dict=color_dict)
        nucleus_kwargs["colormap"] = DirectLabelColormap(color_dict=color_dict)

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
            edge_lines,
            name=f"{prefix}Edges",
            shape_type="path",
            features=edge_features,
            edge_color=edge_colors,
            edge_width=1,
            face_color="transparent",
            blending="translucent",
        ),
        viewer.add_shapes(
            t1_lines,
            name=f"{prefix}T1 edges",
            shape_type="path",
            features=t1_features,
            edge_color=t1_colors,
            edge_width=1,
            face_color="transparent",
            blending="translucent",
        ),
    ]
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
