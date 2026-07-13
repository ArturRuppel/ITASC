"""Compute a per-object, per-frame shape table — label-agnostic.

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
field they read. The object-key column name is caller-supplied (always ``cell_id``
— really the shared track id).

The table is computed in memory (:func:`compute_object_shape`) and pooled by the
aggregate stage; nothing is persisted per position. The module is backend-only
(no Qt / napari import), so scripts and the standalone wheel can use it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops

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


def compute_object_shape(
    label_path: str | Path,
    *,
    pixel_size_um: float,
    object_key: str = "cell_id",
) -> dict[str, np.ndarray]:
    """Per-object shape descriptors for every frame, as a column-major table.

    Columns: ``frame``, *object_key*, then :data:`DESCRIPTOR_COLUMNS`."""
    pixel_size_um = float(pixel_size_um)
    if not pixel_size_um > 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um!r}")
    label_stack = read_label_stack(Path(label_path))
    return _extract_shape_columns(label_stack, pixel_size_um, object_key)


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


