from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = [
    "build_cell_centroid_points",
    "build_edge_shapes",
    "build_t1_points",
    "add_artifact_layers",
]

_BORDER_EDGE_COLOR = np.array([0.6, 0.6, 0.6, 1.0], dtype=float)
_CELL_EDGE_COLOR = np.array([0.12156863, 0.46666667, 0.70588235, 1.0], dtype=float)


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
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    edges = _section(artifact, "edges")
    coord_y = np.asarray(_value(artifact, "coord_y"), dtype=float)
    coord_x = np.asarray(_value(artifact, "coord_x"), dtype=float)

    frame = _column(edges, "frame")
    edge_id = _column(edges, "edge_id")
    cell_a = _column(edges, "cell_a")
    cell_b = _column(edges, "cell_b")
    kind = _column(edges, "kind")
    length = _column(edges, "length").astype(float, copy=False)
    is_t1_frame = _column(edges, "is_t1_frame").astype(bool, copy=False)
    coord_offset = _column(edges, "coord_offset")
    coord_count = _column(edges, "coord_count")

    lines: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    keep: list[int] = []
    for idx in range(len(frame)):
        start = int(coord_offset[idx])
        count = int(coord_count[idx])
        if count < 2:
            continue
        stop = start + count
        ys = coord_y[start:stop]
        xs = coord_x[start:stop]
        lines.append(_stack_points(np.full(len(ys), frame[idx], dtype=float), ys, xs))
        colors.append(_edge_color_for_kind(kind[idx]))
        keep.append(idx)

    mask = np.asarray(keep, dtype=np.intp)
    features = {
        "frame": frame[mask],
        "edge_id": edge_id[mask],
        "cell_a": cell_a[mask],
        "cell_b": cell_b[mask],
        "kind": kind[mask],
        "length": length[mask],
        "is_t1_frame": is_t1_frame[mask],
        "coord_offset": coord_offset[mask],
        "coord_count": coord_count[mask],
    }
    return lines, np.asarray(colors, dtype=float) if colors else np.empty((0, 4), dtype=float), features


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


def add_artifact_layers(viewer: Any, artifact: Any, prefix: str = "[Artifact] ") -> list[Any]:
    cell_points, cell_features = build_cell_centroid_points(artifact)
    edge_lines, edge_colors, edge_features = build_edge_shapes(artifact)
    t1_points, t1_features = build_t1_points(artifact)

    layers = [
        viewer.add_points(
            cell_points,
            name=f"{prefix}Cell centroids",
            features=cell_features,
            size=5,
            face_color="#2f7ed8",
            border_color="white",
            border_width=1,
            blending="translucent",
        ),
        viewer.add_shapes(
            edge_lines,
            name=f"{prefix}Edges",
            shape_type="path",
            features=edge_features,
            edge_color=edge_colors,
            edge_width=2,
            face_color="transparent",
            blending="translucent",
        ),
        viewer.add_points(
            t1_points,
            name=f"{prefix}T1 events",
            features=t1_features,
            size=12,
            symbol="star",
            face_color="red",
            border_color="red",
            border_width=1,
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


def _column(table: Any, name: str) -> np.ndarray:
    if isinstance(table, Mapping):
        value = table[name]
    else:
        value = getattr(table, name)
    return np.asarray(value)


def _stack_points(frame: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(frame) == 0:
        return np.empty((0, 3), dtype=float)
    return np.column_stack((frame.astype(float, copy=False), y.astype(float, copy=False), x.astype(float, copy=False)))


def _edge_color_for_kind(kind: Any) -> np.ndarray:
    if str(kind) == "border":
        return _BORDER_EDGE_COLOR
    return _CELL_EDGE_COLOR
