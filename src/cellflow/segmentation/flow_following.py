"""Flow-following cell segmentation: per-frame Euler integration of the
Cellpose flow field with an EDT-direction gravity blend toward tracked nuclei,
plus Voronoi fill for unconverged foreground pixels."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    median_filter,
)


@dataclass(frozen=True, slots=True)
class FlowFollowingParams:
    """Parameters for `compute_flow_following_movie`."""

    median_kernel_time: int = 3
    median_kernel_space: int = 5
    gaussian_sigma_time: float = 0.0
    gaussian_sigma_space: float = 0.0
    flow_weight: float = 0.5
    flow_step_scale: float = 0.2
    max_iterations: int = 100
    capture_radius: float = 3.0


def _fill_foreground(labels: np.ndarray, prob_mask: np.ndarray) -> np.ndarray:
    """Voronoi-fill any foreground pixels that the integrator did not assign."""
    missing = prob_mask & (labels == 0)
    if not missing.any():
        return labels
    _, (iy, ix) = distance_transform_edt(labels == 0, return_indices=True)
    out = labels.copy()
    out[missing] = labels[iy[missing], ix[missing]]
    return out
