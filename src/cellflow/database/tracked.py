"""TIFF storage for tracked nucleus label volumes.

Schema: single multipage TIFF — shape (T, Y, X), dtype uint32.
Frames that have not yet been tracked are stored as all-zeros.
A frame is considered "tracked" (exists) if it contains at least one non-zero label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

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


def write_tracked_frame(path: str | Path, t: int, labels: np.ndarray) -> None:
    """Write a single tracked frame into tracked_labels.tif."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=_LABEL_DTYPE)
    if labels.ndim == 3:
        labels = labels.max(axis=0)  # (Z, Y, X) → (Y, X)
    H, W = labels.shape
    stack = _load_stack(path)
    if stack.size == 0:
        stack = np.zeros((t + 1, H, W), dtype=_LABEL_DTYPE)
    elif t >= stack.shape[0]:
        extra = np.zeros((t + 1 - stack.shape[0], H, W), dtype=_LABEL_DTYPE)
        stack = np.concatenate([stack, extra], axis=0)
    stack[t] = labels
    tifffile.imwrite(str(path), stack, compression="zlib")


def read_tracked_frame(path: str | Path, t: int) -> np.ndarray:
    """Read a single tracked frame, returned as (Y, X) uint32 array."""
    stack = _load_stack(Path(path))
    if t >= stack.shape[0]:
        raise KeyError(f"Frame t={t} not found in {path}")
    return stack[t]


def read_full_tracked_stack(path: str | Path) -> np.ndarray:
    """Read all tracked frames as a (T, Y, X) uint32 array."""
    return _load_stack(Path(path))


def tracked_n_frames(path: str | Path) -> int:
    """Return the number of timepoints written to tracked_labels.tif."""
    stack = _load_stack(Path(path))
    return stack.shape[0]


def tracked_frame_exists(path: str | Path, t: int) -> bool:
    """Return True if timepoint t has been written (contains non-zero labels)."""
    path = Path(path)
    if not path.exists():
        return False
    stack = _load_stack(path)
    return t < stack.shape[0] and bool(stack[t].any())
