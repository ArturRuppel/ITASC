from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from itasc.napari._track_render import (
    _UNLABELED_COLOR,
    _nucleus_centroids_by_track,
    _track_label_color_styling,
)

__all__ = [
    "build_cell_centroid_points",
    "build_edge_shapes",
    "build_nucleus_track_shapes",
    "build_t1_edge_shapes",
    "build_t1_points",
    "add_contact_analysis_layers",
]

_BORDER_EDGE_COLOR = np.array([0.6, 0.6, 0.6, 1.0], dtype=float)
_CELL_EDGE_COLOR = np.array([0.12156863, 0.46666667, 0.70588235, 1.0], dtype=float)
_T1_EDGE_COLOR = np.array([0.0, 1.0, 0.9, 1.0], dtype=float)
_MIN_TRACK_ALPHA = 0.12
_DEFAULT_TRACK_TAIL = 50
_TRACK_RGB_FLOOR = 0.05
_TRACK_ALPHA_FLOOR = 0.15
# Match the nucleus correction widget's track-overview styling so the two views
# read the same (itasc.napari.track_path_controller.TRACK_TAIL_WIDTH /
# TRACK_TAIL_LENGTH): a short, moderately wide comet rather than a long thin tail.
_TRACK_TAIL_WIDTH = 4
_TRACK_TAIL_LENGTH = 15
_TRACK_HEAD_LENGTH = 0
_TRACK_OPACITY = 0.9


# =====================================================================
# Public shape-building functions (preserved for external callers)
# =====================================================================


def build_cell_centroid_points(
    contact_analysis: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cells = _section(contact_analysis, "cells")
    frame = _column(cells, "frame").astype(float, copy=False)
    y = _column(cells, "centroid_y").astype(float, copy=False)
    x = _column(cells, "centroid_x").astype(float, copy=False)
    points = _stack_points(frame, y, x)
    cell_id = _column(cells, "cell_id")
    features = {
        "frame": _column(cells, "frame"),
        "cell_id": cell_id,
        "area": _column(cells, "area").astype(float, copy=False),
    }
    return points, features


def build_edge_shapes(
    contact_analysis: Any,
    *,
    hide_border_edges: bool = False,
    color_by_id: bool = False,
    color_by_label: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    edges = _section(contact_analysis, "edges")
    coord_y = np.asarray(_value(contact_analysis, "coord_y"), dtype=float)
    coord_x = np.asarray(_value(contact_analysis, "coord_x"), dtype=float)

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
    contact_analysis: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t1_events = _section(contact_analysis, "t1_events")
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
    contact_analysis: Any,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    edges = _section(contact_analysis, "edges")
    t1_events = _section(contact_analysis, "t1_events")
    coord_y = np.asarray(_value(contact_analysis, "coord_y"), dtype=float)
    coord_x = np.asarray(_value(contact_analysis, "coord_x"), dtype=float)

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
    contact_analysis: Any,
    nucleus_labels: np.ndarray,
    *,
    current_frame: int,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Return past-only nucleus centroid track segments for one viewer frame."""
    centroids = _nucleus_centroids_by_track(nucleus_labels)
    color_map = _cell_color_map(contact_analysis)
    segments_by_end = _index_track_segments(centroids)
    return _build_track_shapes_for_frame(
        color_map,
        current_frame=current_frame,
        segments_by_end=segments_by_end,
    )


# =====================================================================
# add_contact_analysis_layers — rasterised, zero callbacks
# =====================================================================


def add_contact_analysis_layers(
    viewer: Any,
    contact_analysis: Any,
    prefix: str = "[Contact Analysis] ",
    *,
    color_edges_by_id: bool = False,
    color_edges_by_label: bool = False,
    hide_border_edges: bool = False,
    cell_labels: np.ndarray | None = None,
    nucleus_labels: np.ndarray | None = None,
    nucleus_track_centroids: dict | None = None,
    track_tail_length: int = _TRACK_TAIL_LENGTH,
) -> list[Any]:
    """Add contact-analysis overlays as napari-native layers.

    Cells and nuclei are label images; nucleus tracks use a GPU ``Tracks``
    layer (native temporal tail-fade, no per-frame rebuild); cell-cell edges
    and T1 transition edges are drawn as antialiased ``path`` shapes for the
    *current* frame only, swapped on the time slider so the Shapes layer never
    holds more than one frame's worth of polylines.
    """
    # --- edges + T1 edges as per-frame path shapes (added after labels) ------
    edge_cache = _frame_shape_cache(
        *build_edge_shapes(
            contact_analysis,
            hide_border_edges=hide_border_edges,
            color_by_id=color_edges_by_id,
            color_by_label=color_edges_by_label,
        )
    )
    t1_cache = _frame_shape_cache(*build_t1_edge_shapes(contact_analysis))

    # --- label images --------------------------------------------------------
    if cell_labels is None:
        cell_labels = _read_label_image(
            _contact_analysis_label_path(contact_analysis, "cell_tracked_labels_path")
        )
    if nucleus_labels is None:
        nucleus_labels = _read_label_image(
            _contact_analysis_label_path(contact_analysis, "nucleus_tracked_labels_path")
        )

    cell_color_dict = _cell_color_map(contact_analysis)
    cell_kwargs: dict[str, Any] = {}
    nucleus_kwargs: dict[str, Any] = {}
    try:
        from napari.utils.colormaps import DirectLabelColormap
    except Exception:  # pragma: no cover – napari compatibility
        pass
    else:
        cell_cmap = DirectLabelColormap(color_dict=cell_color_dict)
        cell_kwargs["colormap"] = cell_cmap
        nucleus_kwargs["colormap"] = cell_cmap

    # --- nucleus tracks as a native GPU Tracks layer -------------------------
    track_centroids = (
        nucleus_track_centroids
        if nucleus_track_centroids is not None
        else _nucleus_centroids_by_track(nucleus_labels)
    )
    tracks_data, tracks_props = _nucleus_tracks_data(track_centroids)
    tail = max(1, int(track_tail_length))
    track_styling = _track_label_color_styling(
        tracks_props["track_id"], cell_color_dict
    )
    track_color_kwargs: dict[str, Any] = {"color_by": "track_id"}
    if track_styling is not None:
        track_cmap, label_pos = track_styling
        tracks_props = {**tracks_props, "label_pos": label_pos}
        track_color_kwargs = {
            "color_by": "label_pos",
            "colormaps_dict": {"label_pos": track_cmap},
        }

    layers: list[Any] = [
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
    ]
    # napari's Tracks layer rejects empty data, so only add it when present.
    if len(tracks_data):
        layers.append(
            viewer.add_tracks(
                tracks_data,
                name=f"{prefix}Nucleus tracks",
                properties=tracks_props,
                tail_width=_TRACK_TAIL_WIDTH,
                tail_length=tail,
                head_length=_TRACK_HEAD_LENGTH,
                blending="translucent",
                opacity=_TRACK_OPACITY,
                **track_color_kwargs,
            )
        )

    # A per-frame Shapes layer whose cache is empty across *every* frame would
    # render as a permanent blank ("ghost") entry in the layer list that never
    # shows anything, so skip it. A layer that merely has no shapes in the
    # current frame is still added — it fills in as the time slider moves.
    for name, cache in (
        (f"{prefix}Edges", edge_cache),
        (f"{prefix}T1 edges", t1_cache),
    ):
        layer = _add_frame_shape_layer(viewer, name, cache)
        if layer is not None:
            layers.append(layer)
    return layers


# =====================================================================
# Per-frame path-shape layers (edges & T1 edges)
# =====================================================================


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


def _nucleus_tracks_data(
    centroids: dict[int, list[tuple[int, float, float]]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build napari ``Tracks`` ``data`` (``[track_id, t, y, x]``) + properties.

    Rows are sorted by track then time, as the Tracks layer requires.
    """
    rows: list[tuple[float, float, float, float]] = []
    for cell_id in sorted(centroids):
        for frame, y, x in sorted(centroids[cell_id]):
            rows.append((float(int(cell_id)), float(frame), float(y), float(x)))
    if not rows:
        empty = np.empty((0, 4), dtype=float)
        return empty, {"track_id": np.empty(0, dtype=int), "time": np.empty(0, dtype=float)}
    data = np.asarray(rows, dtype=float)
    return data, {"track_id": data[:, 0].astype(int), "time": data[:, 1]}


def _frame_shape_cache(
    lines: list[np.ndarray],
    colors: np.ndarray,
    features: dict[str, np.ndarray],
) -> dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]]:
    """Index path shapes by frame so the Shapes layer holds one frame at a time."""
    frames = np.asarray(
        features.get("frame", np.asarray([], dtype=int))
    ).astype(int, copy=False)
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
    return (
        [],
        np.empty((0, 4), dtype=float),
        {name: np.asarray([], dtype=object) for name in feature_names},
    )


def _cached_frame_shapes(
    frame_cache: dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]],
    frame: int,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    return frame_cache.get(int(frame), _empty_frame_shapes(frame_cache))


def _set_path_shapes(layer: Any, lines: list[np.ndarray]) -> None:
    """Replace a Shapes layer's contents with ``path`` shapes for *lines*.

    napari drops the ``"path"`` shape type when an *emptied* Shapes layer is
    repopulated through the ``data`` setter — the new shapes fall back to the
    ``"polygon"`` default, which closes each contour and joins its endpoints with
    a straight segment. Per-frame edge/T1 layers are empty on the frames between
    events, so they routinely hit this path; re-adding with an explicit
    ``shape_type`` keeps them open polylines. Falls back to a plain ``data``
    assignment for layer stand-ins (e.g. tests) that have no ``add`` method.
    """
    adder = getattr(layer, "add", None)
    if lines and callable(adder):
        layer.data = []
        adder(lines, shape_type="path")
    else:
        layer.data = lines


def _add_frame_shape_layer(
    viewer: Any,
    name: str,
    frame_cache: dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]],
) -> Any | None:
    """Add a per-frame ``path`` Shapes layer, or skip a globally-empty cache.

    Returns the created layer, or ``None`` when the cache holds no shapes in any
    frame — adding it would leave a permanent blank entry in the layer list. The
    layer is seeded with the current frame's shapes and swapped on the slider.
    """
    if not frame_cache:
        return None
    lines, colors, features = _cached_frame_shapes(frame_cache, _current_frame(viewer))
    layer = viewer.add_shapes(
        lines,
        ndim=3,
        name=name,
        shape_type="path",
        features=features,
        edge_width=1,
        face_color="transparent",
        blending="translucent",
        **_edge_color_kwargs(colors),
    )
    _connect_frame_shape_layer_to_dims(viewer, layer, frame_cache=frame_cache)
    return layer


def _connect_frame_shape_layer_to_dims(
    viewer: Any,
    layer: Any,
    *,
    frame_cache: dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]],
) -> None:
    """Swap ``layer.data`` to the current frame on every time-slider change.

    The connection is torn down when the layer is removed; a ``cleanup`` closure
    is stashed on the layer so external clears can disconnect it eagerly.
    """
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
        lines, colors, features = _cached_frame_shapes(
            frame_cache, _current_frame(viewer)
        )
        _set_path_shapes(layer, lines)
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
        if removed is layer or getattr(removed, "name", None) == getattr(
            layer, "name", None
        ):
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
        layer._itasc_frame_shape_update = _update
        layer._itasc_frame_shape_cleanup = _disconnect
    except Exception:
        pass


# =====================================================================
# Rasterisation helpers
# =====================================================================


def _line_pixels(
    y0: int, x0: int, y1: int, x1: int, *, width: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer (ys, xs) along a line from (y0,x0) to (y1,x1)."""
    n = max(abs(y1 - y0), abs(x1 - x0), 1) + 1
    center_ys = np.rint(np.linspace(y0, y1, n)).astype(np.intp)
    center_xs = np.rint(np.linspace(x0, x1, n)).astype(np.intp)
    if int(width) <= 1:
        return center_ys, center_xs

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


# =====================================================================
# Track rasterisation (fade-to-black RGBA image)
# =====================================================================


def _rasterize_track_image(
    centroids: dict[int, list[tuple[int, float, float]]],
    color_map: dict[int | None, tuple[float, float, float, float] | str],
    shape: tuple[int, int, int],
    *,
    tail_length: int = _DEFAULT_TRACK_TAIL,
    line_width: int = 1,
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
                    int(round(sy)),
                    int(round(sx)),
                    int(round(ey)),
                    int(round(ex)),
                    width=line_width,
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


# =====================================================================
# Colour maps
# =====================================================================


def _cell_color_map(
    contact_analysis: Any,
) -> dict[int | None, tuple[float, float, float, float] | str]:
    cells = _section(contact_analysis, "cells")
    cell_ids = np.asarray(sorted(set(_column(cells, "cell_id").astype(int))))
    cell_colors = _categorical_colors(cell_ids)
    cmap: dict[int | None, tuple[float, float, float, float] | str] = {
        None: "transparent",
        0: "transparent",
    }
    for cid, color in zip(cell_ids, cell_colors, strict=True):
        cmap[int(cid)] = tuple(float(c) for c in color)
    return cmap


# =====================================================================
# Data-access helpers
# =====================================================================


def _section(contact_analysis: Any, name: str) -> Any:
    if isinstance(contact_analysis, Mapping):
        return contact_analysis[name]
    return getattr(contact_analysis, name)


def _value(contact_analysis: Any, name: str) -> Any:
    if isinstance(contact_analysis, Mapping):
        return contact_analysis[name]
    return getattr(contact_analysis, name)


def _contact_analysis_label_path(contact_analysis: Any, name: str) -> Path:
    return Path(_value(contact_analysis, name))


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
