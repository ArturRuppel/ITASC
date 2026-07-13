"""Compute the relational nucleus-vs-cell shape table.

One row per ``(frame, cell_id)`` present in **both** the cell and nucleus label
stacks. Because cells are nucleus-seeded, a nucleus carries the same label id as
its cell, so pairing is a direct ``(frame, id)`` inner join — no geometry. Ids
present in only one source are dropped from the join.

The emitted columns are *relational* quantities — ratios and offsets between the
paired nucleus and cell — listed in :data:`RELATIONAL_COLUMNS`. The table is
computed in memory (:func:`compute_relational_table`) and pooled by the aggregate
stage; nothing is persisted per position. Backend-only (no Qt / napari).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from skimage.measure import regionprops

from .core import read_label_stack

#: The measured relational columns (everything but the tidy keys) — the value
#: axis a plot/export chooses from. Kept explicit so the on-disk order is stable.
RELATIONAL_COLUMNS = (
    "nc_area_ratio",
    "centroid_offset_um",
    "centroid_offset_norm",
    "orientation_delta",
    "nc_perimeter_ratio",
    "nc_major_axis_ratio",
    "cell_area_um2",
    "nucleus_area_um2",
)

_KEY_COLUMNS = ("frame", "cell_id")
_COLUMN_ORDER = (*_KEY_COLUMNS, *RELATIONAL_COLUMNS)


def compute_relational_table(
    cell_labels_path: str | Path,
    nucleus_labels_path: str | Path,
    *,
    pixel_size_um: float,
) -> dict[str, np.ndarray]:
    """The relational per-(frame, id) table, computed in memory (no file written)."""
    pixel_size_um = float(pixel_size_um)
    if not pixel_size_um > 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um!r}")
    cell_stack = read_label_stack(Path(cell_labels_path))
    nucleus_stack = read_label_stack(Path(nucleus_labels_path))
    cell_props = _object_props(cell_stack, pixel_size_um)
    nucleus_props = _object_props(nucleus_stack, pixel_size_um)
    rows, _dropped = _join_rows(cell_props, nucleus_props)
    return _columns_from_rows(rows)


def _object_props(
    label_stack: np.ndarray, pixel_size_um: float
) -> dict[tuple[int, int], dict[str, float]]:
    """Map ``(frame, label)`` -> the per-object primitives the join needs."""
    s = pixel_size_um
    out: dict[tuple[int, int], dict[str, float]] = {}
    for frame_idx, frame in enumerate(label_stack):
        for prop in regionprops(frame):
            centroid_y, centroid_x = (float(c) for c in prop.centroid)
            out[(int(frame_idx), int(prop.label))] = {
                "area_um2": float(prop.area) * s * s,
                "perimeter_um": float(prop.perimeter) * s,
                "major_axis_length_um": float(prop.axis_major_length) * s,
                "orientation": float(prop.orientation),
                "centroid_y_um": centroid_y * s,
                "centroid_x_um": centroid_x * s,
            }
    return out


def _join_rows(
    cell_props: dict[tuple[int, int], dict[str, float]],
    nucleus_props: dict[tuple[int, int], dict[str, float]],
) -> tuple[list[dict[str, float | int]], int]:
    """Inner-join on ``(frame, id)``; return rows + the count of dropped ids."""
    shared = sorted(set(cell_props) & set(nucleus_props))
    dropped = len(set(cell_props) ^ set(nucleus_props))
    rows = [
        _relational_row(frame, cell_id, cell_props[(frame, cell_id)], nucleus_props[(frame, cell_id)])
        for frame, cell_id in shared
    ]
    return rows, dropped


def _relational_row(
    frame: int, cell_id: int, cell: dict[str, float], nucleus: dict[str, float]
) -> dict[str, float | int]:
    cell_area = cell["area_um2"]
    offset = math.hypot(
        nucleus["centroid_y_um"] - cell["centroid_y_um"],
        nucleus["centroid_x_um"] - cell["centroid_x_um"],
    )
    equiv_radius = math.sqrt(cell_area / math.pi) if cell_area > 0 else math.nan
    return {
        "frame": int(frame),
        "cell_id": int(cell_id),
        "nc_area_ratio": _ratio(nucleus["area_um2"], cell_area),
        "centroid_offset_um": offset,
        "centroid_offset_norm": offset / equiv_radius if equiv_radius > 0 else math.nan,
        "orientation_delta": _fold_orientation(nucleus["orientation"] - cell["orientation"]),
        "nc_perimeter_ratio": _ratio(nucleus["perimeter_um"], cell["perimeter_um"]),
        "nc_major_axis_ratio": _ratio(
            nucleus["major_axis_length_um"], cell["major_axis_length_um"]
        ),
        "cell_area_um2": cell_area,
        "nucleus_area_um2": nucleus["area_um2"],
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.nan


def _fold_orientation(delta: float) -> float:
    """Fold an ellipse-orientation difference into ``[0, π/2]`` (axes are
    undirected, so a π flip and the sign of the gap are immaterial)."""
    folded = abs(delta) % math.pi
    return min(folded, math.pi - folded)


def _columns_from_rows(rows: list[dict]) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    for name in _COLUMN_ORDER:
        values = [row[name] for row in rows]
        dtype = np.int64 if name in _KEY_COLUMNS else float
        columns[name] = np.asarray(values, dtype=dtype)
    return columns
