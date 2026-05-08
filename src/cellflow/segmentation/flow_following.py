"""Flow-following cell segmentation: per-frame Euler integration of the
Cellpose flow field with an EDT-direction gravity blend toward tracked nuclei."""
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


def compute_filtered_flow_vectors(
    dp_tcyx: np.ndarray,
    params: FlowFollowingParams,
) -> np.ndarray:
    """Return flow vectors after the configured median and Gaussian filters."""
    filtered = np.asarray(dp_tcyx, dtype=np.float32)
    if params.median_kernel_time > 1 or params.median_kernel_space > 1:
        filtered = median_filter(
            filtered,
            size=(
                1,
                int(params.median_kernel_time),
                int(params.median_kernel_space),
                int(params.median_kernel_space),
            ),
        )
    if params.gaussian_sigma_time > 0.0 or params.gaussian_sigma_space > 0.0:
        filtered = gaussian_filter(
            filtered,
            sigma=(
                0.0,
                float(params.gaussian_sigma_time),
                float(params.gaussian_sigma_space),
                float(params.gaussian_sigma_space),
            ),
        )
    return np.asarray(filtered, dtype=np.float32)


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


def compute_flow_following_movie(
    foreground_tyx: np.ndarray,    # (T, Y, X) bool
    dp_tcyx: np.ndarray,           # (T, 2, Y, X) float32
    labels_tyx: np.ndarray,        # (T, Y, X) int32
    params: FlowFollowingParams,
    progress_cb: Callable[[int, int], None] | None = None,
    *,
    filter_vectors: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame flow-following segmentation with pre-integration filtering.

    Returns
    -------
    filtered_dp_tcyx : (T, 2, Y, X) float32 — flow stack after median+Gaussian.
    cell_labels_tyx  : (T, Y, X) int32      — same labelling as input nuclei.
    """
    foreground = np.asarray(foreground_tyx, dtype=bool)
    dp = np.asarray(dp_tcyx, dtype=np.float32)
    labels = np.asarray(labels_tyx, dtype=np.int32)

    T = dp.shape[0]

    filtered = compute_filtered_flow_vectors(dp, params) if filter_vectors else dp

    out_labels = np.zeros_like(labels, dtype=np.int32)

    for t in range(T):
        prob_mask = foreground[t]
        nuclear_labels = labels[t]

        if not prob_mask.any() or not (nuclear_labels > 0).any():
            if progress_cb is not None:
                progress_cb(t + 1, T)
            continue

        flow_yx2 = np.stack(
            [filtered[t, 0], filtered[t, 1]], axis=-1
        ).astype(np.float32)
        mags = np.hypot(flow_yx2[..., 0], flow_yx2[..., 1])
        mean_mag = float(mags[prob_mask].mean()) if prob_mask.any() else 0.0
        if mean_mag > 1e-6:
            flow_yx2 = (flow_yx2 / mean_mag).astype(np.float32)

        dist, (ny, nx) = distance_transform_edt(
            nuclear_labels == 0, return_indices=True
        )
        H, W = nuclear_labels.shape
        yi, xi = np.indices((H, W))
        dy = (ny - yi).astype(np.float32)
        dx = (nx - xi).astype(np.float32)
        norm = np.hypot(dy, dx)
        safe = np.where(norm > 0, norm, 1.0)
        grav_y = (dy / safe).astype(np.float32)
        grav_x = (dx / safe).astype(np.float32)
        inside = nuclear_labels > 0
        grav_y[inside] = 0.0
        grav_x[inside] = 0.0

        integrated = _flow_integrate(
            nuclear_labels.astype(np.int32),
            np.ascontiguousarray(flow_yx2, dtype=np.float32),
            grav_y, grav_x,
            dist.astype(np.float32),
            ny.astype(np.int32), nx.astype(np.int32),
            prob_mask,
            int(params.max_iterations),
            float(params.flow_step_scale),
            float(params.flow_weight),
            float(params.capture_radius),
        )

        out_labels[t] = integrated

        if progress_cb is not None:
            progress_cb(t + 1, T)

    return filtered, out_labels
