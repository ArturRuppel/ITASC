"""Build / read the per-position cell-shape table.

One row per (frame, cell label): area, perimeter, equivalent diameter, the
fitted-ellipse axis lengths and their ratio, circularity, eccentricity,
solidity, extent, orientation, and centroid — all from
:func:`skimage.measure.regionprops`. Dimensional descriptors are converted to
physical units using the caller-supplied ``pixel_size_um`` (µm per pixel): areas
to µm² (``area_um2``), lengths and centroids to µm (the ``*_um`` columns). The
ratios (``aspect_ratio``, ``circularity``, ``eccentricity``, ``solidity``,
``extent``) are dimensionless and scale-invariant.

Persistence is a self-owned ``cell_shape.h5`` mirroring the contacts
``cells/table`` layout: a ``shape/table`` group of 1-D datasets plus a
``provenance`` group. The module is backend-only (no Qt / napari).
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import tifffile
from skimage.measure import regionprops

#: Output column order. ``frame`` / ``cell_id`` are the tidy keys; the rest are
#: the descriptors. Kept explicit so the on-disk order is stable.
_KEY_COLUMNS = ("frame", "cell_id")
_FLOAT_COLUMNS = (
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
COLUMNS = _KEY_COLUMNS + _FLOAT_COLUMNS
#: The measured descriptor columns (everything but the tidy keys) — the value
#: axis a plot/export chooses from.
DESCRIPTOR_COLUMNS = _FLOAT_COLUMNS


def build_cell_shape(
    *,
    cell_labels_path: str | Path,
    output_path: str | Path,
    pixel_size_um: float,
    source_path: str | Path | None = None,
    params: dict | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Measure shape descriptors for every labelled cell in each frame.

    Reads a 2-D or 2-D+t tracked cell-label TIFF at *cell_labels_path*, runs
    ``regionprops`` per frame, and writes ``cell_shape.h5`` to *output_path*.
    Dimensional descriptors are scaled by *pixel_size_um* (µm per pixel) into
    physical units. *source_path* is recorded as provenance only.
    """
    pixel_size_um = float(pixel_size_um)
    if not pixel_size_um > 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um!r}")
    total = 3
    cell_labels_path = Path(cell_labels_path)
    output_path = Path(output_path)
    params = dict(params or {})

    label_stack = _read_label_stack(cell_labels_path)
    _report_progress(progress_cb, 1, total, "read labels")

    columns = _extract_shape_columns(label_stack, pixel_size_um)
    _report_progress(progress_cb, 2, total, "extract shape")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        provenance = h5.create_group("provenance")
        provenance.attrs["source_position_path"] = str(source_path) if source_path else ""
        provenance.attrs["cell_tracked_labels_path"] = str(cell_labels_path)
        provenance.attrs["pixel_size_um"] = pixel_size_um
        provenance.attrs["params_json"] = json.dumps(params, sort_keys=True)
        provenance.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        provenance.attrs["cellflow_version"] = _cellflow_version()

        _write_column_group(h5.create_group("shape/table", track_order=True), columns)
    _report_progress(progress_cb, 3, total, "write HDF5")
    return output_path


def read_cell_shape(path: str | Path) -> dict[str, np.ndarray]:
    """Return the ``shape/table`` as a column-major dict of 1-D arrays."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        table = h5["shape/table"]
        return {name: dataset[:] for name, dataset in table.items()}


def _extract_shape_columns(
    label_stack: np.ndarray, pixel_size_um: float
) -> dict[str, np.ndarray]:
    rows: list[dict[str, float | int]] = []
    for frame_idx, frame in enumerate(label_stack):
        for prop in sorted(regionprops(frame), key=lambda item: item.label):
            rows.append(_shape_row(frame_idx, prop, pixel_size_um))
    return _columns_from_rows(rows)


def _shape_row(frame_idx: int, prop, pixel_size_um: float) -> dict[str, float | int]:
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
        "cell_id": int(prop.label),
        "area_um2": area_px * s * s,
        "perimeter_um": perimeter_px * s,
        "equivalent_diameter_um": float(prop.equivalent_diameter_area) * s,
        "major_axis_length_um": major_px * s,
        "minor_axis_length_um": minor_px * s,
        # Degenerate (e.g. single-pixel) regions have a zero minor axis or
        # perimeter; report NaN rather than dividing by zero.
        "aspect_ratio": major_px / minor_px if minor_px > 0 else math.nan,
        "circularity": _circularity(area_px, perimeter_px),
        "eccentricity": float(prop.eccentricity),
        "solidity": float(prop.solidity),
        "extent": float(prop.extent),
        "orientation": float(prop.orientation),
        "centroid_y_um": centroid_y * s,
        "centroid_x_um": centroid_x * s,
    }


def _circularity(area: float, perimeter: float) -> float:
    """4π·area / perimeter², clamped to ≤ 1 (a perfect disk is 1.0)."""
    if perimeter <= 0:
        return math.nan
    return min(4.0 * math.pi * area / (perimeter * perimeter), 1.0)


def _columns_from_rows(rows: list[dict]) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    for name in COLUMNS:
        values = [row[name] for row in rows]
        dtype = np.int64 if name in _KEY_COLUMNS else float
        columns[name] = np.asarray(values, dtype=dtype)
    return columns


def _write_column_group(group: h5py.Group, columns: dict[str, np.ndarray]) -> None:
    for name, values in columns.items():
        group.create_dataset(name, data=values)


def _read_label_stack(path: Path) -> np.ndarray:
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


def _report_progress(
    progress_cb: Callable[[int, int, str], None] | None,
    done: int,
    total: int,
    message: str,
) -> None:
    if progress_cb is not None:
        progress_cb(done, total, message)


def _cellflow_version() -> str:
    try:
        from importlib.metadata import version

        return version("cellflow")
    except Exception:
        return "unknown"
