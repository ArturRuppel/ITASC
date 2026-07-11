"""Build / read a per-object, per-frame shape table — label-agnostic.

One row per ``(frame, object label)``: area, perimeter, equivalent diameter, the
fitted-ellipse axis lengths and their ratio, circularity, eccentricity,
solidity, extent, orientation, and centroid — all from
:func:`skimage.measure.regionprops`. Dimensional descriptors are converted to
physical units using the caller-supplied ``pixel_size_um`` (µm per pixel): areas
to µm² (``area_um2``), lengths and centroids to µm (the ``*_um`` columns). The
ratios (``aspect_ratio``, ``circularity``, ``eccentricity``, ``solidity``,
``extent``) are dimensionless and scale-invariant.

The core is **label-agnostic**: it runs over whatever label stack it is handed
(cell or nucleus), so the cell / nucleus quantifiers differ only by which input
field they read, their output filename, and nothing else. The object-key column
name is caller-supplied (always ``cell_id`` — really the shared track id).

Persistence is a flat tidy **CSV** plus a ``<name>.provenance.json`` sidecar; the
shape tables are small and flat, so HDF5's hierarchy / partial-read wins do not
apply and CSV is git-/Excel-/pandas-readable. The module is backend-only (no Qt /
napari import), so scripts and the standalone wheel can use it.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops

from cellflow.contact_analysis._provenance import (
    cellflow_version as _cellflow_version,
    report_progress as _report_progress,
)

#: The measured descriptor columns (everything but the tidy keys) — the value
#: axis a plot/export chooses from. Kept explicit so the on-disk order is stable.
DESCRIPTOR_COLUMNS = (
    "area_um2",
    "perimeter_um",
    "equivalent_diameter_um",
    "major_axis_length_um",
    "minor_axis_length_um",
    "aspect_ratio",
    "circularity",
    "eccentricity",
    "solidity",
    "extent",
    "orientation",
    "centroid_y_um",
    "centroid_x_um",
)


def build_object_shape(
    label_path: str | Path,
    output_path: str | Path,
    *,
    pixel_size_um: float,
    object_key: str = "cell_id",
    source_path: str | Path | None = None,
    params: dict | None = None,
    quantity_id: str = "",
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Measure shape descriptors for every labelled object in each frame.

    Reads a 2-D or 2-D+t tracked label TIFF at *label_path*, runs ``regionprops``
    per frame, and writes a tidy CSV (``frame``, *object_key*, then
    :data:`DESCRIPTOR_COLUMNS`) to *output_path* plus a ``<name>.provenance.json``
    sidecar. Dimensional descriptors are scaled by *pixel_size_um* (µm per pixel)
    into physical units. *source_path* is recorded as provenance only.
    """
    pixel_size_um = float(pixel_size_um)
    if not pixel_size_um > 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um!r}")
    total = 3
    label_path = Path(label_path)
    output_path = Path(output_path)
    params = dict(params or {})

    label_stack = read_label_stack(label_path)
    _report_progress(progress_cb, 1, total, "read labels")

    columns = _extract_shape_columns(label_stack, pixel_size_um, object_key)
    _report_progress(progress_cb, 2, total, "extract shape")

    column_order = ("frame", object_key, *DESCRIPTOR_COLUMNS)
    write_table_csv(output_path, columns, column_order)
    write_provenance(
        output_path,
        quantity_id=quantity_id,
        source_path=source_path,
        label_path=label_path,
        pixel_size_um=pixel_size_um,
        params=params,
        columns=column_order,
    )
    _report_progress(progress_cb, 3, total, "write CSV")
    return output_path


def compute_object_shape(
    label_path: str | Path,
    *,
    pixel_size_um: float,
    object_key: str = "cell_id",
) -> dict[str, np.ndarray]:
    """Per-object shape descriptors for every frame, as a column-major table.

    The pure compute behind :func:`build_object_shape` — no file written. Columns:
    ``frame``, *object_key*, then :data:`DESCRIPTOR_COLUMNS`."""
    pixel_size_um = float(pixel_size_um)
    if not pixel_size_um > 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um!r}")
    label_stack = read_label_stack(Path(label_path))
    return _extract_shape_columns(label_stack, pixel_size_um, object_key)


def read_shape_table(path: str | Path) -> dict[str, np.ndarray]:
    """Return the CSV table as a column-major dict of 1-D arrays."""
    return read_table_csv(path)


def _extract_shape_columns(
    label_stack: np.ndarray, pixel_size_um: float, object_key: str
) -> dict[str, np.ndarray]:
    rows: list[dict[str, float | int]] = []
    for frame_idx, frame in enumerate(label_stack):
        for prop in sorted(regionprops(frame), key=lambda item: item.label):
            rows.append(_shape_row(frame_idx, prop, pixel_size_um, object_key))
    column_order = ("frame", object_key, *DESCRIPTOR_COLUMNS)
    key_columns = ("frame", object_key)
    return _columns_from_rows(rows, column_order, key_columns)


def _shape_row(
    frame_idx: int, prop, pixel_size_um: float, object_key: str
) -> dict[str, float | int]:
    # Pixel-unit primitives: ratios are computed from these (scale-invariant),
    # while dimensional outputs are scaled to µm / µm² below.
    area_px = float(prop.area)
    perimeter_px = float(prop.perimeter)
    major_px = float(prop.axis_major_length)
    minor_px = float(prop.axis_minor_length)
    centroid_y, centroid_x = (float(c) for c in prop.centroid)
    s = pixel_size_um
    return {
        "frame": int(frame_idx),
        object_key: int(prop.label),
        "area_um2": area_px * s * s,
        "perimeter_um": perimeter_px * s,
        "equivalent_diameter_um": float(prop.equivalent_diameter_area) * s,
        "major_axis_length_um": major_px * s,
        "minor_axis_length_um": minor_px * s,
        # Degenerate (e.g. single-pixel) regions have a zero minor axis or
        # perimeter; report NaN rather than dividing by zero.
        "aspect_ratio": major_px / minor_px if minor_px > 0 else math.nan,
        "circularity": circularity(area_px, perimeter_px),
        "eccentricity": float(prop.eccentricity),
        "solidity": float(prop.solidity),
        "extent": float(prop.extent),
        "orientation": float(prop.orientation),
        "centroid_y_um": centroid_y * s,
        "centroid_x_um": centroid_x * s,
    }


def circularity(area: float, perimeter: float) -> float:
    """4π·area / perimeter², clamped to ≤ 1 (a perfect disk is 1.0)."""
    if perimeter <= 0:
        return math.nan
    return min(4.0 * math.pi * area / (perimeter * perimeter), 1.0)


# ----------------------------------------------------------- shared CSV helpers
def _columns_from_rows(
    rows: list[dict], column_order: tuple[str, ...], key_columns: tuple[str, ...]
) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    for name in column_order:
        values = [row[name] for row in rows]
        dtype = np.int64 if name in key_columns else float
        columns[name] = np.asarray(values, dtype=dtype)
    return columns


def write_table_csv(
    output_path: Path, columns: dict[str, np.ndarray], column_order: tuple[str, ...]
) -> None:
    """Write *columns* as a flat tidy CSV in *column_order*."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({name: columns[name] for name in column_order})
    frame.to_csv(output_path, index=False)


def read_table_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Read a flat tidy CSV back into a column-major dict of 1-D arrays.

    ``frame`` and any ``*_id`` column are returned as ``int64``; the rest as
    floats (so NaN survives a round-trip)."""
    path = Path(path)
    frame = pd.read_csv(path)
    out: dict[str, np.ndarray] = {}
    for name in frame.columns:
        if name == "frame" or name.endswith("_id"):
            out[name] = frame[name].to_numpy(dtype=np.int64)
        else:
            out[name] = frame[name].to_numpy(dtype=float)
    return out


def write_provenance(
    output_path: Path,
    *,
    quantity_id: str,
    source_path: str | Path | None,
    label_path: str | Path | None = None,
    cell_labels_path: str | Path | None = None,
    nucleus_labels_path: str | Path | None = None,
    pixel_size_um: float,
    params: dict,
    columns: tuple[str, ...],
) -> Path:
    """Write the ``<name>.provenance.json`` sidecar next to *output_path*."""
    sidecar = provenance_path(output_path)
    record = {
        "quantity_id": quantity_id,
        "source_position_path": str(source_path) if source_path else "",
        "pixel_size_um": pixel_size_um,
        "params": dict(params),
        "columns": list(columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cellflow_version": _cellflow_version(),
    }
    if label_path is not None:
        record["label_path"] = str(label_path)
    if cell_labels_path is not None:
        record["cell_labels_path"] = str(cell_labels_path)
    if nucleus_labels_path is not None:
        record["nucleus_labels_path"] = str(nucleus_labels_path)
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True))
    return sidecar


def provenance_path(output_path: str | Path) -> Path:
    """The sidecar path for a table: ``<name>.provenance.json``."""
    return Path(output_path).with_suffix(".provenance.json")


def read_label_stack(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(
            f"Expected a 2-D or 3-D tracked label TIFF at {path}, got shape {arr.shape}"
        )
    return arr.astype(np.int64, copy=False)


