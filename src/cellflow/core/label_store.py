"""TIFF storage for tracked nucleus label volumes.

Schema: single multipage TIFF — shape (T, Y, X), dtype uint32.
Frames that have not yet been tracked are stored as all-zeros.

Frame existence is recorded explicitly in a JSON sidecar (``<name>.frames.json``)
listing the written timepoint indices, so a legitimately-tracked but
all-background frame still reads as "tracked". Files written before the sidecar
existed fall back to the legacy content heuristic (a non-zero label present).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from cellflow.core.tiff import imwrite_grayscale

_LABEL_DTYPE = np.uint32


def _frames_sidecar(path: Path) -> Path:
    """Path of the JSON sidecar recording which timepoints were written."""
    return path.with_name(path.name + ".frames.json")


def _record_written_frames(path: Path, indices) -> None:
    """Add ``indices`` to the set of written timepoints in the sidecar."""
    sidecar = _frames_sidecar(path)
    written: set[int] = set()
    if sidecar.exists():
        try:
            written = {int(i) for i in json.loads(sidecar.read_text())}
        except (ValueError, OSError):
            written = set()
    written.update(int(i) for i in indices)
    sidecar.write_text(json.dumps(sorted(written)))


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

    A multipage zlib TIFF cannot be updated in place, so each
    :func:`write_tracked_frame` call decodes and re-encodes the whole file.
    Looping that over every frame to save a corrected stack is O(T²); call
    this once instead (O(T)). A legacy singleton-Z axis ``(T, 1, Y, X)`` is
    squeezed to match the documented schema.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.asarray(stack, dtype=_LABEL_DTYPE)
    if stack.ndim == 4 and stack.shape[1] == 1:
        stack = stack[:, 0]
    imwrite_grayscale(path, stack, compression="zlib")
    _frames_sidecar(path).write_text(json.dumps(list(range(int(stack.shape[0])))))


def write_tracked_frame(path: str | Path, t: int, labels: np.ndarray) -> None:
    """Write a single tracked frame into tracked_labels.tif.

    Note: this re-encodes the entire multipage TIFF on every call. To save many
    frames at once use :func:`write_full_tracked_stack`, which is O(T) rather
    than O(T²) over a full-stack save.
    """
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
    imwrite_grayscale(path, stack, compression="zlib")
    _record_written_frames(path, [t])


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
    """Return True if timepoint ``t`` has been written.

    Existence means the frame was written, independent of whether it carries
    any labels — an all-background frame can be a valid tracking result. Resolved
    from the sidecar; legacy files without one fall back to the content heuristic
    (a non-zero label present).
    """
    path = Path(path)
    sidecar = _frames_sidecar(path)
    if sidecar.exists():
        try:
            return int(t) in {int(i) for i in json.loads(sidecar.read_text())}
        except (ValueError, OSError):
            pass
    if not path.exists():
        return False
    stack = _load_stack(path)
    return t < stack.shape[0] and bool(stack[t].any())
