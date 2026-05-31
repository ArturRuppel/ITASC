# src/cellflow/tracking_ultrack/atoms.py
"""Atom extraction: residual-conditioned foreground split by contour ridges.

Stage ① of the atom-based candidate pipeline. Pure, deterministic functions
shared by the interactive preview and the full-stack ``atoms.tif`` writer.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_local


def residual(frame: np.ndarray, window: int) -> np.ndarray:
    """Local-mean-subtracted residual: ``clip(frame - localmean(frame), 0)``.

    Flattens each map's per-nucleus offset so a single global threshold works
    everywhere while staying ~0 in flat background. ``window`` is forced odd.
    """
    window = int(window) | 1
    frame = np.asarray(frame, dtype=np.float32)
    local_mean = threshold_local(frame, block_size=window, method="gaussian")
    return np.clip(frame - local_mean, 0.0, None).astype(np.float32)
