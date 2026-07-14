"""Unified path resolution for the ITASC pipeline directory layout."""
from __future__ import annotations

from pathlib import Path

#: Filename ITASC writes a position's metadata/config into.
CONFIG_NAME = "itasc_config.json"

#: Pre-rename filename, written by the old CellFlow app. Read-only fallback.
_LEGACY_CONFIG_NAME = "cellflow_config.json"


def pos_dir(root: Path | str, pos: int) -> Path:
    """Return <root>/pos<pos:02d>."""
    return Path(root) / f"pos{pos:02d}"


def position_config_path(position_dir: Path | str) -> Path:
    """Path to a position's ITASC config file.

    Prefers the current ``itasc_config.json``. When it is absent but a legacy
    ``cellflow_config.json`` (written before the rename) is present, returns the
    legacy path so old data keeps loading. When neither exists, returns the
    current name, so new writes always use it.
    """
    base = Path(position_dir)
    current = base / CONFIG_NAME
    if current.exists():
        return current
    legacy = base / _LEGACY_CONFIG_NAME
    if legacy.exists():
        return legacy
    return current

