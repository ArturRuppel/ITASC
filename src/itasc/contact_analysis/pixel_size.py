"""Resolve a position's physical pixel size (µm per pixel).

Backend-only (no Qt / napari) so the standalone wheel and headless batch runs can
use it. The cell-shape build needs µm/px to turn pixel-unit ``regionprops``
measurements into physical µm / µm². Two sources are consulted, in order:

1. The position's ``itasc_config.json`` — ``metadata.pixel_size_um`` — the key
   the main pipeline widget already writes (``main_widget.py``).
2. The segmented label TIFF's resolution tags — ImageJ ``unit`` + ``resolution``
   or the baseline ``XResolution`` / ``ResolutionUnit`` tags.

Returns ``None`` when neither yields a positive value; callers treat that as
"pixel size unknown" (the cell-shape build is then blocked until one is set).
"""
from __future__ import annotations

import json
from pathlib import Path

import tifffile

from itasc.core.paths import position_config_path

__all__ = ["resolve_pixel_size_um", "pixel_size_from_config", "pixel_size_from_tiff"]

#: Multipliers turning a length in the tag's unit into µm.
_UNIT_TO_UM = {
    "micron": 1.0,
    "microns": 1.0,
    "um": 1.0,
    "µm": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "mm": 1_000.0,
    "millimeter": 1_000.0,
    "cm": 10_000.0,
    "centimeter": 10_000.0,
    "inch": 25_400.0,
    "in": 25_400.0,
}
#: Baseline-TIFF ``ResolutionUnit`` tag values (2 = inch, 3 = centimeter).
_RESOLUTION_UNIT_TO_UM = {2: 25_400.0, 3: 10_000.0}


def resolve_pixel_size_um(
    position_dir: Path | str | None,
    cell_labels_path: Path | str | None,
) -> float | None:
    """µm/px for a position, from its config first, then the label TIFF tags.

    Returns ``None`` when no positive pixel size can be found.
    """
    size = pixel_size_from_config(position_dir)
    if size is not None:
        return size
    return pixel_size_from_tiff(cell_labels_path)


def pixel_size_from_config(position_dir: Path | str | None) -> float | None:
    """``metadata.pixel_size_um`` from ``itasc_config.json`` in *position_dir*."""
    if position_dir is None:
        return None
    config_path = position_config_path(position_dir)
    if not config_path.is_file():
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        metadata = config.get("metadata", {})
        return _positive_float(metadata.get("pixel_size_um"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def pixel_size_from_tiff(cell_labels_path: Path | str | None) -> float | None:
    """µm/px from a TIFF's resolution tags (ImageJ unit, else ResolutionUnit)."""
    if cell_labels_path is None:
        return None
    path = Path(cell_labels_path)
    if not path.is_file():
        return None
    try:
        with tifffile.TiffFile(str(path)) as tf:
            from_ome = _pixel_size_from_ome(tf)
            if from_ome is not None:
                return from_ome
            from_imagej = _pixel_size_from_imagej(tf)
            if from_imagej is not None:
                return from_imagej
            return _pixel_size_from_baseline(tf)
    except (OSError, ValueError, KeyError, tifffile.TiffFileError):
        # An unreadable/empty/corrupt TIFF means "pixel size unknown", not a
        # crash. tifffile raises TiffFileError (not a KeyError/OSError subclass)
        # for a malformed file, so it must be named explicitly.
        return None


def _pixel_size_from_ome(tf: tifffile.TiffFile) -> float | None:
    """OME-TIFF stores µm/px directly as ``Pixels/@PhysicalSizeX`` (+ unit)."""
    xml = tf.ome_metadata
    if not xml:
        return None
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    for el in root.iter():
        if el.tag.endswith("Pixels"):
            value = _positive_float(el.get("PhysicalSizeX"))
            if value is None:
                return None
            unit = str(el.get("PhysicalSizeXUnit", "µm")).strip().lower()
            to_um = _UNIT_TO_UM.get(unit)
            return value * to_um if to_um is not None else None
    return None


def _pixel_size_from_imagej(tf: tifffile.TiffFile) -> float | None:
    """ImageJ stores spacing as ``XResolution`` (pixels/unit) plus a ``unit``."""
    metadata = tf.imagej_metadata or {}
    unit = str(metadata.get("unit", "")).strip().lower()
    to_um = _UNIT_TO_UM.get(unit)
    if to_um is None:
        return None
    per_unit = _tag_ratio(tf, "XResolution")
    if per_unit is None or per_unit <= 0:
        return None
    return to_um / per_unit


def _pixel_size_from_baseline(tf: tifffile.TiffFile) -> float | None:
    """Baseline TIFF: ``XResolution`` pixels/unit with a ``ResolutionUnit``."""
    page = tf.pages[0]
    unit_tag = page.tags.get("ResolutionUnit")
    to_um = _RESOLUTION_UNIT_TO_UM.get(int(unit_tag.value)) if unit_tag else None
    if to_um is None:
        return None
    per_unit = _tag_ratio(tf, "XResolution")
    if per_unit is None or per_unit <= 0:
        return None
    return to_um / per_unit


def _tag_ratio(tf: tifffile.TiffFile, name: str) -> float | None:
    """Read a rational TIFF tag (``(num, den)`` or scalar) as a float."""
    tag = tf.pages[0].tags.get(name)
    if tag is None:
        return None
    value = tag.value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        num, den = value
        return float(num) / float(den) if den else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None
