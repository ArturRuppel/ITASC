"""Shared selection-highlight rendering: turn a boolean cell mask into the RGBA
overlay the correction studio paints over the labels.

Two visual cues, independently toggleable:

* ``dim`` — uniformly dim everything *outside* the mask (the "spotlight"), so the
  selected cell stays at full brightness while its surroundings recede.
* ``border`` — draw an opaque yellow ring just outside the mask.

The correction widget uses one cue at a time (its ``spotlight`` / ``border``
styles); the aggregate-quantification click-to-load uses both at once so a picked
cell reads clearly against the full labels movie it sits in.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

#: Alpha applied outside the mask in spotlight (``dim``) mode.
SPOTLIGHT_OPACITY = 0.7


def spotlight_rgba(
    mask: np.ndarray,
    *,
    dim: bool = True,
    border: bool = False,
    opacity: float = SPOTLIGHT_OPACITY,
    ring_iterations: int = 2,
) -> np.ndarray:
    """RGBA (``mask.shape + (4,)``, float32) overlay highlighting *mask*.

    With *dim*, everything outside the mask is washed at *opacity* and the mask
    itself left fully transparent. With *border*, an opaque yellow ring of
    *ring_iterations* px is drawn just outside the mask. An all-False mask yields
    an all-zero (fully transparent) overlay.
    """
    data = np.zeros(mask.shape + (4,), dtype=np.float32)
    if dim:
        alpha = np.full(mask.shape, opacity, dtype=np.float32)
        alpha[mask] = 0.0
        data[..., 3] = alpha
    if border:
        ring = binary_dilation(mask, iterations=ring_iterations) & ~mask
        data[ring] = (1.0, 1.0, 0.0, 1.0)  # opaque yellow outline
    return data
