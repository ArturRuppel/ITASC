# src/itasc/core/imageops.py
"""Generic, deterministic image operations shared across ITASC pieces.

Pure functions with no ITASC dependencies, usable by segmentation, tracking,
and any other consumer without pulling in a heavier subpackage.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_local


def residual(frame: np.ndarray, window: int, strength: float = 1.0) -> np.ndarray:
    """Local-mean-subtracted residual: ``clip(frame - strength*localmean(frame), 0)``.

    Flattens each map's per-nucleus offset so a single global threshold works
    everywhere while staying ~0 in flat background. ``window`` is forced odd.

    ``strength`` blends between the raw map and the fully-flattened residual:
    ``1.0`` subtracts the whole local background (default), ``0.0`` subtracts
    nothing so the result is the raw (non-negative) map, and values in between
    partially flatten. Lowering it trades uniform-threshold behaviour for
    keeping more of the original signal where the background is already flat.
    """
    window = int(window) | 1
    frame = np.asarray(frame, dtype=np.float32)
    if strength == 0.0:
        # No background subtraction requested (the default): the result is just
        # the non-negative raw map, so skip the O(Y·X) gaussian entirely. This
        # matters because ``CellDivergenceParams.fg_strength`` defaults to 0.0,
        # so every default-config frame — including live previews — hit this.
        return np.clip(frame, 0.0, None)
    local_mean = threshold_local(frame, block_size=window, method="gaussian")
    return np.clip(frame - strength * local_mean, 0.0, None).astype(np.float32)
