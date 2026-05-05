"""Flow-following cell segmentation: per-frame Euler integration of the
Cellpose flow field with an EDT-direction gravity blend toward tracked nuclei,
plus Voronoi fill for unconverged foreground pixels."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numba
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


@numba.njit(parallel=True, cache=True)
def _flow_integrate(
    nuclear_labels: np.ndarray,    # (H, W) int32
    flow: np.ndarray,              # (H, W, 2) float32 — channel 0 = dY, channel 1 = dX
    grav_y: np.ndarray,            # (H, W) float32 — EDT-direction unit vector y
    grav_x: np.ndarray,            # (H, W) float32 — EDT-direction unit vector x
    dist_to_nucleus: np.ndarray,   # (H, W) float32 — EDT distance to nearest nuclear pixel
    nearest_y: np.ndarray,         # (H, W) int32 — y-index of nearest nuclear pixel
    nearest_x: np.ndarray,         # (H, W) int32
    prob_mask: np.ndarray,         # (H, W) bool — foreground mask
    n_steps: int,
    flow_step_scale: float,
    flow_weight: float,
    capture_radius: float,
) -> np.ndarray:
    H, W = nuclear_labels.shape
    result = nuclear_labels.copy()

    for i in numba.prange(H):
        for j in range(W):
            if result[i, j] > 0:
                continue
            if not prob_mask[i, j]:
                continue

            py = float(i)
            px = float(j)
            label = 0

            for _ in range(n_steps):
                iy0 = int(py)
                ix0 = int(px)
                iy0 = max(0, min(H - 2, iy0))
                ix0 = max(0, min(W - 2, ix0))

                fy = py - float(iy0)
                fx = px - float(ix0)

                flow_y = (flow[iy0,     ix0,     0] * (1.0 - fy) * (1.0 - fx) +
                          flow[iy0 + 1, ix0,     0] * fy          * (1.0 - fx) +
                          flow[iy0,     ix0 + 1, 0] * (1.0 - fy) * fx          +
                          flow[iy0 + 1, ix0 + 1, 0] * fy          * fx)

                flow_x = (flow[iy0,     ix0,     1] * (1.0 - fy) * (1.0 - fx) +
                          flow[iy0 + 1, ix0,     1] * fy          * (1.0 - fx) +
                          flow[iy0,     ix0 + 1, 1] * (1.0 - fy) * fx          +
                          flow[iy0 + 1, ix0 + 1, 1] * fy          * fx)

                w = flow_weight

                iy_nn = max(0, min(H - 1, int(py + 0.5)))
                ix_nn = max(0, min(W - 1, int(px + 0.5)))

                step_y = w * flow_y + (1.0 - w) * grav_y[iy_nn, ix_nn]
                step_x = w * flow_x + (1.0 - w) * grav_x[iy_nn, ix_nn]

                py = max(0.0, min(float(H - 1), py + step_y * flow_step_scale))
                px = max(0.0, min(float(W - 1), px + step_x * flow_step_scale))

                iy = max(0, min(H - 1, int(py + 0.5)))
                ix = max(0, min(W - 1, int(px + 0.5)))

                if dist_to_nucleus[iy, ix] <= capture_radius:
                    L = nuclear_labels[nearest_y[iy, ix], nearest_x[iy, ix]]
                    if L > 0:
                        label = L
                        break

            result[i, j] = label

    return result
