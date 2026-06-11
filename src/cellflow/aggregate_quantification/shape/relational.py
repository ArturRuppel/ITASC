"""Build / read the relational nucleus-vs-cell shape table.

One row per ``(frame, cell_id)`` present in **both** the cell and nucleus label
stacks. Because cells are nucleus-seeded, a nucleus carries the same label id as
its cell, so pairing is a direct ``(frame, id)`` inner join — no geometry. Ids
present in only one source are dropped from the join; the dropped count is
surfaced via *progress_cb* so silent loss is visible.

The emitted columns are *relational* quantities — ratios and offsets between the
paired nucleus and cell — listed in :data:`RELATIONAL_COLUMNS`. Persistence is a
flat tidy CSV plus a ``<name>.provenance.json`` sidecar, the same format the
per-object shape tables use (see :mod:`.core`). Backend-only (no Qt / napari).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from skimage.measure import regionprops

from .core import (
    read_label_stack,
    read_table_csv,
    write_provenance,
    write_table_csv,
)

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


def build_relational(
    cell_labels_path: str | Path,
    nucleus_labels_path: str | Path,
    output_path: str | Path,
    *,
    pixel_size_um: float,
    source_path: str | Path | None = None,
    params: dict | None = None,
    quantity_id: str = "",
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Pair each nucleus with its cell (shared ``(frame, id)``) and measure the
    relational shape quantities of :data:`RELATIONAL_COLUMNS`.

    Reads both tracked-label TIFFs, runs ``regionprops`` per frame on each,
    inner-joins on ``(frame, id)``, and writes a tidy CSV plus provenance sidecar
    to *output_path*. Dimensional inputs are scaled by *pixel_size_um*. Ids found
    in only one stack are dropped; the dropped count is reported via
    *progress_cb*.
    """
    pixel_size_um = float(pixel_size_um)
    if not pixel_size_um > 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um!r}")
    total = 4
    cell_labels_path = Path(cell_labels_path)
    nucleus_labels_path = Path(nucleus_labels_path)
    output_path = Path(output_path)
    params = dict(params or {})

    cell_stack = read_label_stack(cell_labels_path)
    nucleus_stack = read_label_stack(nucleus_labels_path)
    _report_progress(progress_cb, 1, total, "read labels")

    cell_props = _object_props(cell_stack, pixel_size_um)
    nucleus_props = _object_props(nucleus_stack, pixel_size_um)
    _report_progress(progress_cb, 2, total, "extract shape")

    rows, dropped = _join_rows(cell_props, nucleus_props)
    _report_progress(
        progress_cb, 3, total, f"join ({len(rows)} paired, {dropped} unpaired dropped)"
    )

    columns = _columns_from_rows(rows)
    write_table_csv(output_path, columns, _COLUMN_ORDER)
    write_provenance(
        output_path,
        quantity_id=quantity_id,
        source_path=source_path,
        cell_labels_path=cell_labels_path,
        nucleus_labels_path=nucleus_labels_path,
        pixel_size_um=pixel_size_um,
        params={**params, "unpaired_dropped": dropped},
        columns=_COLUMN_ORDER,
    )
    _report_progress(progress_cb, 4, total, "write CSV")
    return output_path


def read_relational_table(path: str | Path) -> dict[str, np.ndarray]:
    """Return the relational CSV as a column-major dict of 1-D arrays."""
    return read_table_csv(path)


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


def _report_progress(
    progress_cb: Callable[[int, int, str], None] | None,
    done: int,
    total: int,
    message: str,
) -> None:
    if progress_cb is not None:
        progress_cb(done, total, message)
