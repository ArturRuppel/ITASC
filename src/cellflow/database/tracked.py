"""HDF5 storage for tracked nucleus label volumes.

Schema: t{t:03d}/labels  — shape (Z, Y, X), dtype uint32.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

_LABEL_DTYPE = np.uint32


def write_tracked_frame(path: str | Path, t: int, labels: np.ndarray) -> None:
    """Write a single tracked frame into tracked_labels.h5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=_LABEL_DTYPE)
    with h5py.File(path, "a") as h5:
        grp = h5.require_group(f"t{t:03d}")
        if "labels" in grp:
            del grp["labels"]
        grp.create_dataset("labels", data=labels, compression="gzip", compression_opts=4, shuffle=True)
        grp.attrs["t"] = int(t)
        grp.attrs["label_shape"] = np.asarray(labels.shape, dtype=np.int64)
        if "version" not in h5.attrs:
            h5.attrs["version"] = 1
            h5.attrs["stage"] = "nucleus_tracked"


def read_tracked_frame(path: str | Path, t: int) -> np.ndarray:
    """Read a single tracked frame, returned as (Z, Y, X) uint32 array."""
    with h5py.File(Path(path), "r") as h5:
        return np.asarray(h5[f"t{t:03d}/labels"], dtype=_LABEL_DTYPE)


def tracked_n_frames(path: str | Path) -> int:
    """Return the number of timepoints written to tracked_labels.h5."""
    with h5py.File(Path(path), "r") as h5:
        return sum(1 for k in h5.keys() if k.startswith("t"))


def tracked_frame_exists(path: str | Path, t: int) -> bool:
    """Return True if timepoint t has been written."""
    with h5py.File(Path(path), "r") as h5:
        return f"t{t:03d}" in h5
