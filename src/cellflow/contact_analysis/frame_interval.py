"""Resolve a position's frame interval (seconds per frame).

The time twin of :mod:`.pixel_size`. Track dynamics needs the real elapsed time
between frames to turn pixel displacements into µm/s, MSD into µm²/s, and lags
into seconds. Backend-only (no Qt / napari) so the standalone wheel and headless
batch runs can use it. Two sources are consulted, in order:

1. The position's ``cellflow_config.json`` — ``metadata.time_interval_s`` — the
   key the main pipeline widget already writes (``main_widget.py``).
2. The label TIFF's ImageJ ``finterval`` tag (seconds between frames), the
   convention ImageJ/Fiji writes for time-lapse stacks.

Returns ``None`` when neither yields a positive value; callers treat that as
"interval unknown" (the dynamics build is then blocked until one is set).
"""
from __future__ import annotations

import json
from pathlib import Path

import tifffile

__all__ = [
    "resolve_time_interval_s",
    "time_interval_from_config",
    "time_interval_from_tiff",
]


def resolve_time_interval_s(
    position_dir: Path | str | None,
    label_path: Path | str | None,
) -> float | None:
    """Seconds/frame for a position, from its config first, then the TIFF tag.

    Returns ``None`` when no positive interval can be found.
    """
    interval = time_interval_from_config(position_dir)
    if interval is not None:
        return interval
    return time_interval_from_tiff(label_path)


def time_interval_from_config(position_dir: Path | str | None) -> float | None:
    """``metadata.time_interval_s`` from ``cellflow_config.json`` in *position_dir*."""
    if position_dir is None:
        return None
    config_path = Path(position_dir) / "cellflow_config.json"
    if not config_path.is_file():
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        metadata = config.get("metadata", {})
        return _positive_float(metadata.get("time_interval_s"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def time_interval_from_tiff(label_path: Path | str | None) -> float | None:
    """Seconds/frame from a TIFF's ImageJ ``finterval`` metadata."""
    if label_path is None:
        return None
    path = Path(label_path)
    if not path.is_file():
        return None
    try:
        with tifffile.TiffFile(str(path)) as tf:
            metadata = tf.imagej_metadata or {}
            return _positive_float(metadata.get("finterval"))
    except (OSError, ValueError, KeyError, tifffile.TiffFileError):
        # An unreadable/empty/corrupt TIFF means "frame interval unknown", not a
        # crash. tifffile raises TiffFileError (not a KeyError/OSError subclass)
        # for a malformed file, so it must be named explicitly.
        return None


def _positive_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None
