"""TIFF writing helpers for CellFlow grayscale stacks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import tifffile


def imwrite_grayscale(path: str | Path, data: Any, **kwargs: Any) -> None:
    """Write a grayscale TIFF stack without tifffile RGB-shape ambiguity."""
    kwargs.setdefault("photometric", "minisblack")
    tifffile.imwrite(str(path), data, **kwargs)
