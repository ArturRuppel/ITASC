"""Unified path resolution for the CellFlow pipeline directory layout."""
from __future__ import annotations

from pathlib import Path


def pos_dir(root: Path | str, pos: int) -> Path:
    """Return <root>/pos<pos:02d>."""
    return Path(root) / f"pos{pos:02d}"

