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
_DEFAULT_TRACK_TAIL = 50
_TRACK_RGB_FLOOR = 0.05
_TRACK_ALPHA_FLOOR = 0.15


# =====================================================================
# Public shape-building functions (preserved for external callers)
# =====================================================================


def build_cell_centroid_points(
    artifact: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
        lines.append(
            _stack_points(np.full(len(ys), frame[idx], dtype=float), ys, xs)
        )
        keep.append(idx)

    mask = np.asarray(keep, dtype=np.intp)
    if color_by_label:
        colors = _categorical_colors(edge_label[mask])
    elif color_by_id:
        colors = _categorical_colors(edge_id[mask])
    else:
        colors = (
            np.array(
                [_edge_color_for_kind(item) for item in kind[mask]], dtype=float
            )
            if len(mask)
            else np.empty((0, 4), dtype=float)
        )
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


def build_t1_points(
    artifact: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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


def build_t1_edge_shapes(
    artifact: Any,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
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
        for side, frame in (
            ("before", transition_frame),
            ("after", transition_frame + 1),
        ):
            row_idx = _find_edge_row(
                edge_frame, edge_id, frame, transition_edge_id
            )
            if row_idx is None:
                continue
            start = int(coord_offset[row_idx])
            count = int(coord_count[row_idx])
            if count < 2:
                continue
            stop = start + count
            ys = coord_y[start:stop]
            xs = coord_x[start:stop]
            lines.append(
                _stack_points(np.full(len(ys), frame, dtype=float), ys, xs)
            )
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
    segments_by_end = _index_track_segments(centroids)
    return _build_track_shapes_for_frame(
        color_map,
        current_frame=current_frame,
        segments_by_end=segments_by_end,
    )


# =====================================================================
# add_artifact_layers — rasterised, zero callbacks
# =====================================================================


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
    track_tail_length: int = _DEFAULT_TRACK_TAIL,
) -> list[Any]:
    if cell_labels is None:
        cell_labels = _read_label_image(
            _artifact_label_path(artifact, "cell_tracked_labels_path")
        )
    if nucleus_labels is None:
        nucleus_labels = _read_label_image(
            _artifact_label_path(artifact, "nucleus_tracked_labels_path")
        )

    shape = cell_labels.shape
    if len(shape) == 2:
        shape = (1, shape[0], shape[1])
    T, H, W = shape[:3]

    edge_labels, edge_color_dict = _rasterize_edge_labels(
        artifact,
        (T, H, W),
        hide_border_edges=hide_border_edges,
        color_by_id=color_edges_by_id,
        color_by_label=color_edges_by_label,
    )

    t1_labels, t1_color_dict = _rasterize_t1_edge_labels(
        artifact, (T, H, W)
    )

    track_centroids = (
        nucleus_track_centroids
        if nucleus_track_centroids is not None
        else _nucleus_centroids_by_track(nucleus_labels)
    )
    track_color_map = _cell_color_map(
        artifact, color_by_label=color_cells_by_label
    )
    track_image = _rasterize_track_image(
        track_centroids,
        track_color_map,
        (T, H, W),
        tail_length=track_tail_length,
    )

    cell_color_dict = _cell_color_map(
        artifact, color_by_label=color_cells_by_label
    )
    cell_kwargs: dict[str, Any] = {}
    nucleus_kwargs: dict[str, Any] = {}
    edge_kwargs: dict[str, Any] = {}
    t1_kwargs: dict[str, Any] = {}
    try:
        from napari.utils.colormaps import DirectLabelColormap
    except Exception:  # pragma: no cover – napari compatibility
        pass
    else:
        cell_cmap = DirectLabelColormap(color_dict=cell_color_dict)
        cell_kwargs["colormap"] = cell_cmap
        nucleus_kwargs["colormap"] = cell_cmap
        edge_kwargs["colormap"] = DirectLabelColormap(color_dict=edge_color_dict)
        t1_kwargs["colormap"] = DirectLabelColormap(color_dict=t1_color_dict)

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
        viewer.add_image(
            track_image,
            name=f"{prefix}Nucleus tracks",
            rgb=True,
            opacity=0.9,
            blending="additive",
        ),
        viewer.add_labels(
            edge_labels,
            name=f"{prefix}Edges",
            opacity=0.8,
            blending="translucent",
            **edge_kwargs,
        ),
        viewer.add_labels(
            t1_labels,
            name=f"{prefix}T1 edges",
            opacity=0.8,
            blending="translucent",
            **t1_kwargs,
        ),
    ]
    return layers


# =====================================================================
# Rasterisation helpers
# =====================================================================


def _line_pixels(
    y0: int, x0: int, y1: int, x1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer (ys, xs) along a 2px-wide line from (y0,x0) to (y1,x1)."""
    n = max(abs(y1 - y0), abs(x1 - x0), 1) + 1
    center_ys = np.rint(np.linspace(y0, y1, n)).astype(np.intp)
    center_xs = np.rint(np.linspace(x0, x1, n)).astype(np.intp)

    # expand perpendicular to the line direction
    dy = y1 - y0
    dx = x1 - x0
    if abs(dy) >= abs(dx):
        # mostly vertical — expand horizontally
        ys = np.concatenate([center_ys, center_ys])
        xs = np.concatenate([center_xs, center_xs + 1])
    else:
        # mostly horizontal — expand vertically
        ys = np.concatenate([center_ys, center_ys + 1])
        xs = np.concatenate([center_xs, center_xs])

    return ys, xs


def _draw_polyline_label(
    canvas: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    value: int,
) -> None:
    """Rasterise a polyline into a (H, W) integer label array."""
    h, w = canvas.shape[:2]
    for i in range(len(ys) - 1):
        py, px = _line_pixels(
            int(round(ys[i])),
            int(round(xs[i])),
            int(round(ys[i + 1])),
            int(round(xs[i + 1])),
        )
        valid = (py >= 0) & (py < h) & (px >= 0) & (px < w)
        canvas[py[valid], px[valid]] = value


# =====================================================================
# Edge rasterisation
# =====================================================================


def _rasterize_edge_labels(
    artifact: Any,
    shape: tuple[int, int, int],
    *,
    hide_border_edges: bool = False,
    color_by_id: bool = False,
    color_by_label: bool = False,
) -> tuple[np.ndarray, dict]:
    """Rasterise edge polylines into a ``(T, H, W)`` label array.

    Returns ``(labels, color_dict)`` ready for
    :class:`~napari.utils.colormaps.DirectLabelColormap`.
    """
    edges = _section(artifact, "edges")
    coord_y = np.asarray(_value(artifact, "coord_y"), dtype=float)
    coord_x = np.asarray(_value(artifact, "coord_x"), dtype=float)

    frame_col = _column(edges, "frame")
    edge_id_col = _column(edges, "edge_id")
    kind_col = _column(edges, "kind")
    edge_label_col = _column(edges, "edge_label")
    coord_offset_col = _column(edges, "coord_offset")
    coord_count_col = _column(edges, "coord_count")

    T, H, W = shape
    labels = np.zeros((T, H, W), dtype=np.int32)

    edge_kind: dict[int, str] = {}
    edge_elabel: dict[int, str] = {}

    for idx in range(len(frame_col)):
        k = str(kind_col[idx])
        if hide_border_edges and k == "border":
            continue
        f = int(frame_col[idx])
        eid = int(edge_id_col[idx])
        if eid == 0:
            continue
        start = int(coord_offset_col[idx])
        count = int(coord_count_col[idx])
        if count < 2 or f < 0 or f >= T:
            continue
        _draw_polyline_label(
            labels[f], coord_y[start : start + count], coord_x[start : start + count], eid
        )
        if eid not in edge_kind:
            edge_kind[eid] = k
            edge_elabel[eid] = str(edge_label_col[idx])

    color_dict: dict[int | None, Any] = {None: "transparent", 0: "transparent"}
    if color_by_label:
        unique_labels = sorted({v for v in edge_elabel.values() if v})
        palette = {lab: _palette_color(i) for i, lab in enumerate(unique_labels)}
        for eid, lab in edge_elabel.items():
            color_dict[eid] = tuple(
                float(c) for c in palette.get(lab, _UNLABELED_COLOR)
            )
    elif color_by_id:
        sorted_ids = sorted(edge_kind)
        id_colors = _categorical_colors(np.asarray(sorted_ids))
        for eid, color in zip(sorted_ids, id_colors):
            color_dict[eid] = tuple(float(c) for c in color)
    else:
        for eid, k in edge_kind.items():
            color_dict[eid] = tuple(float(c) for c in _edge_color_for_kind(k))

    return labels, color_dict


# =====================================================================
# T1-edge rasterisation
# =====================================================================


def _rasterize_t1_edge_labels(
    artifact: Any,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, dict]:
    """Rasterise T1 transition edges into a ``(T, H, W)`` label array."""
    edges = _section(artifact, "edges")
    t1_events = _section(artifact, "t1_events")
    coord_y = np.asarray(_value(artifact, "coord_y"), dtype=float)
    coord_x = np.asarray(_value(artifact, "coord_x"), dtype=float)

    edge_frame = _column(edges, "frame")
    edge_id_col = _column(edges, "edge_id")
    coord_offset = _column(edges, "coord_offset")
    coord_count = _column(edges, "coord_count")

    event_ids = _column(t1_events, "t1_event_id")
    event_frame = _column(t1_events, "frame")
    event_edge_id = _column(t1_events, "edge_id")

    # fast (frame, edge_id) → row lookup
    edge_lookup: dict[tuple[int, int], int] = {}
    for idx in range(len(edge_frame)):
        edge_lookup.setdefault((int(edge_frame[idx]), int(edge_id_col[idx])), idx)

    T, H, W = shape
    labels = np.zeros((T, H, W), dtype=np.int32)
    used_ids: set[int] = set()

    for event_idx in range(len(event_ids)):
        t1_id = int(event_ids[event_idx])
        if t1_id == 0:
            continue
        transition_frame = int(event_frame[event_idx])
        eid = int(event_edge_id[event_idx])
        for f in (transition_frame, transition_frame + 1):
            if f < 0 or f >= T:
                continue
            row_idx = edge_lookup.get((f, eid))
            if row_idx is None:
                continue
            start = int(coord_offset[row_idx])
            count = int(coord_count[row_idx])
            if count < 2:
                continue
            _draw_polyline_label(
                labels[f], coord_y[start : start + count], coord_x[start : start + count], t1_id
            )
            used_ids.add(t1_id)

    t1_color = tuple(float(c) for c in _T1_EDGE_COLOR)
    color_dict: dict[int | None, Any] = {None: "transparent", 0: "transparent"}
    for t1_id in used_ids:
        color_dict[t1_id] = t1_color
    return labels, color_dict


# =====================================================================
# Track rasterisation (fade-to-black RGBA image)
# =====================================================================


def _rasterize_track_image(
    centroids: dict[int, list[tuple[int, float, float]]],
    color_map: dict[int | None, tuple[float, float, float, float] | str],
    shape: tuple[int, int, int],
    *,
    tail_length: int = _DEFAULT_TRACK_TAIL,
) -> np.ndarray:
    """Rasterise nucleus tracks into a ``(T, H, W, 4)`` uint8 RGBA stack.

    Recent segments appear in the cell's colour; older segments transition
    toward black and then transparent via per-frame exponential decay.
    """
    T, H, W = shape
    segments_by_end = _index_track_segments(centroids)

    tail_length = max(1, int(tail_length))
    rgb_decay = float(_TRACK_RGB_FLOOR ** (1.0 / tail_length))
    alpha_decay = float(_TRACK_ALPHA_FLOOR ** (1.0 / tail_length))

    # resolve cell colours once (skip string/"transparent" entries)
    draw_colors: dict[int, np.ndarray] = {}
    for cell_id in centroids:
        raw = color_map.get(int(cell_id), _UNLABELED_COLOR)
        if isinstance(raw, str):
            continue
        c = np.asarray(raw, dtype=np.float32).copy()
        c[3] = 1.0
        draw_colors[int(cell_id)] = c

    output = np.zeros((T, H, W, 4), dtype=np.uint8)
    current = np.zeros((H, W, 4), dtype=np.float32)

    for t in range(T):
        # decay existing pixels: rgb fades fast, alpha fades slower
        current[:, :, :3] *= rgb_decay
        current[:, :, 3] *= alpha_decay

        # draw new segments that end at this frame
        segments = segments_by_end.get(t)
        if segments is not None:
            for cell_id, _sf, sy, sx, _ef, ey, ex in segments:
                color = draw_colors.get(cell_id)
                if color is None:
                    continue
                py, px = _line_pixels(
                    int(round(sy)), int(round(sx)), int(round(ey)), int(round(ex))
                )
                valid = (py >= 0) & (py < H) & (px >= 0) & (px < W)
                current[py[valid], px[valid]] = color

        # write uint8 frame
        output[t] = (current * 255.0).astype(np.uint8)

    return output


# =====================================================================
# Track-segment indexing & shape building (for public API)
# =====================================================================


def _index_track_segments(
    centroids: dict[int, list[tuple[int, float, float]]],
) -> dict[int, list[tuple[int, int, float, float, int, float, float]]]:
    """Index consecutive track segments by *end* frame.

    Each value is a list of
    ``(cell_id, start_frame, start_y, start_x, end_frame, end_y, end_x)``.
    """
    by_end: dict[int, list[tuple[int, int, float, float, int, float, float]]] = {}
    for cell_id, rows in centroids.items():
        for prev, curr in zip(rows[:-1], rows[1:]):
            sf, sy, sx = prev
            ef, ey, ex = curr
            if int(ef) != int(sf) + 1:
                continue
            by_end.setdefault(int(ef), []).append(
                (int(cell_id), int(sf), float(sy), float(sx), int(ef), float(ey), float(ex))
            )
    return by_end


def _build_track_shapes_for_frame(
    color_map: dict[int | None, tuple[float, float, float, float] | str],
    *,
    current_frame: int,
    segments_by_end: dict[int, list[tuple[int, int, float, float, int, float, float]]],
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Build vector track shapes for a single frame (alpha-fade variant)."""
    current_frame = max(0, int(current_frame))
    max_age = max(current_frame, 1)

    lines: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    feature_rows: list[dict[str, Any]] = []

    for end_frame in range(current_frame + 1):
        segments = segments_by_end.get(end_frame)
        if segments is None:
            continue
        age = current_frame - end_frame
        alpha = max(_MIN_TRACK_ALPHA, 1.0 - (age / max_age))
        for cell_id, sf, sy, sx, ef, ey, ex in segments:
            color = np.asarray(
                color_map.get(cell_id, _UNLABELED_COLOR), dtype=float
            ).copy()
            color[3] = alpha
            lines.append(
                _stack_points(
                    np.asarray([current_frame, current_frame], dtype=float),
                    np.asarray([sy, ey], dtype=float),
                    np.asarray([sx, ex], dtype=float),
                )
            )
            colors.append(color)
            feature_rows.append(
                {"cell_id": cell_id, "start_frame": sf, "end_frame": ef, "age": age}
            )

    color_array = (
        np.asarray(colors, dtype=float) if colors else np.empty((0, 4), dtype=float)
    )
    return lines, color_array, _feature_columns(
        feature_rows, ["cell_id", "start_frame", "end_frame", "age"]
    )


# =====================================================================
# Efficient nucleus-centroid extraction (single pass per frame)
# =====================================================================


def _nucleus_centroids_by_track(
    nucleus_labels: np.ndarray,
) -> dict[int, list[tuple[int, float, float]]]:
    labels = np.asarray(nucleus_labels)
    if labels.ndim == 2:
        labels = labels[np.newaxis, ...]
    if labels.ndim > 3:
        labels = np.squeeze(labels)
    if labels.ndim != 3:
        raise ValueError(
            f"Expected time-first 2D/3D nucleus labels, got shape {nucleus_labels.shape}"
        )

    centroids: dict[int, list[tuple[int, float, float]]] = {}
    for frame_idx in range(labels.shape[0]):
        frame = labels[frame_idx]
        flat = frame.ravel()
        order = np.argsort(flat, kind="stable")
        sorted_ids = flat[order]

        change = np.empty(len(sorted_ids), dtype=bool)
        change[0] = True
        np.not_equal(sorted_ids[1:], sorted_ids[:-1], out=change[1:])
        boundaries = np.flatnonzero(change)

        rows_all, cols_all = np.divmod(order, frame.shape[1])

        ends = np.empty_like(boundaries)
        ends[:-1] = boundaries[1:]
        ends[-1] = len(sorted_ids)

        for bi in range(len(boundaries)):
            cell_id = int(sorted_ids[boundaries[bi]])
            if cell_id == 0:
                continue
            s, e = int(boundaries[bi]), int(ends[bi])
            y = float(rows_all[s:e].mean())
            x = float(cols_all[s:e].mean())
            centroids.setdefault(cell_id, []).append((frame_idx, y, x))

    return centroids


# =====================================================================
# Colour maps
# =====================================================================


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
    cmap: dict[int | None, tuple[float, float, float, float] | str] = {
        None: "transparent",
        0: "transparent",
    }
    for cid, color in zip(cell_ids, cell_colors, strict=True):
        cmap[int(cid)] = tuple(float(c) for c in color)
    return cmap


def _cell_label_color_map(
    artifact: Any,
) -> dict[int | None, tuple[float, float, float, float] | str]:
    cells = _section(artifact, "cells")
    cell_ids = _column(cells, "cell_id")
    class_labels = _column(cells, "class_label")
    class_colors = _categorical_colors(class_labels)
    cmap: dict[int | None, tuple[float, float, float, float] | str] = {
        None: "transparent",
        0: "transparent",
    }
    for cid, color in zip(cell_ids, class_colors):
        cid_int = int(cid)
        if cid_int not in cmap:
            cmap[cid_int] = tuple(float(c) for c in color)
    return cmap


# =====================================================================
# Data-access helpers
# =====================================================================


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


def _column(table: Any, name: str) -> np.ndarray:
    if isinstance(table, Mapping):
        value = table[name]
    else:
        value = getattr(table, name)
    return np.asarray(value)


# =====================================================================
# Low-level helpers
# =====================================================================


def _find_edge_row(
    frame: np.ndarray,
    edge_id: np.ndarray,
    target_frame: int,
    target_edge_id: int,
) -> int | None:
    matches = np.flatnonzero(
        (frame.astype(int, copy=False) == target_frame)
        & (edge_id.astype(int, copy=False) == target_edge_id)
    )
    if len(matches) == 0:
        return None
    return int(matches[0])


def _feature_columns(
    rows: list[dict[str, Any]], names: list[str],
) -> dict[str, np.ndarray]:
    if not rows:
        return {name: np.asarray([], dtype=object) for name in names}
    return {name: np.asarray([row[name] for row in rows]) for name in names}


def _stack_points(
    frame: np.ndarray, y: np.ndarray, x: np.ndarray,
) -> np.ndarray:
    if len(frame) == 0:
        return np.empty((0, 3), dtype=float)
    return np.column_stack(
        (
            frame.astype(float, copy=False),
            y.astype(float, copy=False),
            x.astype(float, copy=False),
        )
    )


def _edge_color_for_kind(kind: Any) -> np.ndarray:
    if str(kind) == "border":
        return _BORDER_EDGE_COLOR
    return _CELL_EDGE_COLOR


def _categorical_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if len(values) == 0:
        return np.empty((0, 4), dtype=float)
    keys = [str(v) for v in values]
    palette = {
        key: _palette_color(idx)
        for idx, key in enumerate(sorted({k for k in keys if k != ""}))
    }
    colors = np.empty((len(values), 4), dtype=float)
    for idx, key in enumerate(keys):
        colors[idx] = _UNLABELED_COLOR if key == "" else palette[key]
    return colors


def _palette_color(index: int) -> np.ndarray:
    hue = (index * 0.618033988749895) % 1.0
    return np.asarray((*_hsv_to_rgb(hue, 0.65, 0.9), 1.0), dtype=float)


def _hsv_to_rgb(
    hue: float, saturation: float, value: float,
) -> tuple[float, float, float]:
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
