"""TIFF storage for tracked nucleus label volumes.

Schema: single multipage TIFF — shape (T, Y, X), dtype uint32.
Frames that have not yet been tracked are stored as all-zeros.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from itasc.core.tiff import imwrite_grayscale

_LABEL_DTYPE = np.uint32


def _load_stack(path: Path) -> np.ndarray:
    """Load the TIFF as (T, Y, X). Returns empty array if file does not exist.

    Tolerates legacy files written with a singleton Z axis: ``(T, 1, Y, X)``
    is squeezed to ``(T, Y, X)`` so the in-memory shape always matches the
    documented schema.
    """
    if not path.exists():
        return np.empty((0, 0, 0), dtype=_LABEL_DTYPE)
    stack = np.asarray(tifffile.imread(str(path)), dtype=_LABEL_DTYPE)
    if stack.ndim == 2:
        stack = stack[np.newaxis]
    elif stack.ndim == 4 and stack.shape[1] == 1:
        stack = stack[:, 0]
    return stack


def write_full_tracked_stack(path: str | Path, stack: np.ndarray) -> None:
    """Write the entire ``(T, Y, X)`` tracked stack in one encode.

    A multipage zlib TIFF cannot be updated in place, so a per-frame writer
    would decode and re-encode the whole file on every frame (O(T²) over a
    full-stack save). This encodes once (O(T)). A legacy singleton-Z axis
    ``(T, 1, Y, X)`` is squeezed to match the documented schema.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.asarray(stack, dtype=_LABEL_DTYPE)
    if stack.ndim == 4 and stack.shape[1] == 1:
        stack = stack[:, 0]
    imwrite_grayscale(path, stack, compression="zlib")


def read_full_tracked_stack(path: str | Path) -> np.ndarray:
    """Read all tracked frames as a (T, Y, X) uint32 array."""
    return _load_stack(Path(path))
