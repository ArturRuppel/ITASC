"""Flow-following cell segmentation: per-frame Euler integration of the
Cellpose flow field with an EDT-direction gravity blend toward tracked nuclei."""
from __future__ import annotations

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


# Progressive shell assignment defaults (not user-facing)
_SHELL_WIDTH: float = 5.0
_MAX_SHELLS: int = 50


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


# ---------------------------------------------------------------------------
# Single-frame & consensus helpers (for contour-map alternative path)
# ---------------------------------------------------------------------------


# ===================================================================
# Two-phase flow integration (capture_radius == 0 path)
# ===================================================================


@numba.njit(parallel=True, cache=True)
def _flow_integrate_to_positions(
    nuclear_labels,   # (H, W) int32
    flow,             # (2, H, W) float32
    grav_y,           # (H, W) float32
    grav_x,           # (H, W) float32
    prob_mask,        # (H, W) bool
    n_steps,          # int
    flow_step_scale,  # float
    flow_weight,      # float
):
    """Phase 1: integrate each foreground pixel along flow + gravity.

    Returns
    -------
    result : (H, W) int32
        Label map.  Nucleus pixels keep their label.  Foreground pixels
        that land directly on a nucleus pixel during integration receive
        that label.  Everything else remains 0.
    final_y, final_x : (H, W) float32
        Displaced position of every pixel after integration.
        Non-foreground and nucleus pixels store their original coords.
    """
    H = nuclear_labels.shape[0]
    W = nuclear_labels.shape[1]
    result = nuclear_labels.copy()
    final_y = np.empty((H, W), dtype=np.float32)
    final_x = np.empty((H, W), dtype=np.float32)

    for i in numba.prange(H):
        for j in range(W):
            # Default: pixel stays at its original position
            final_y[i, j] = np.float32(i)
            final_x[i, j] = np.float32(j)

            # Skip nucleus pixels and non-foreground
            if nuclear_labels[i, j] > 0 or not prob_mask[i, j]:
                continue

            py = np.float32(i)
            px = np.float32(j)

            for _ in range(n_steps):
                iy = min(max(int(py), 0), H - 1)
                ix = min(max(int(px), 0), W - 1)

                fy = (
                    flow_weight * flow[0, iy, ix]
                    + (np.float32(1.0) - flow_weight) * grav_y[iy, ix]
                )
                fx = (
                    flow_weight * flow[1, iy, ix]
                    + (np.float32(1.0) - flow_weight) * grav_x[iy, ix]
                )

                py = py + fy * flow_step_scale
                px = px + fx * flow_step_scale
                py = min(max(py, np.float32(0.0)), np.float32(H - 1))
                px = min(max(px, np.float32(0.0)), np.float32(W - 1))

                # Early stop: landed on a nucleus pixel
                iy2 = int(py)
                ix2 = int(px)
                if nuclear_labels[iy2, ix2] > 0:
                    result[i, j] = nuclear_labels[iy2, ix2]
                    break

            final_y[i, j] = py
            final_x[i, j] = px

    return result, final_y, final_x


def _flow_integrate(
    nuclear_labels: np.ndarray,
    flow: np.ndarray,
    grav_y: np.ndarray,
    grav_x: np.ndarray,
    dist: np.ndarray,
    ny: np.ndarray,
    nx: np.ndarray,
    prob_mask: np.ndarray,
    *,
    n_steps: int,
    flow_step_scale: float,
    flow_weight: float,
    capture_radius: float,
) -> np.ndarray:
    """Compatibility integrator for the movie-level flow-following API.

    This is the original capture-radius behavior: foreground pixels are
    advected through a flow/gravity blend and inherit the nearest nucleus label
    only when the advected position reaches a nucleus or enters the configured
    capture radius.  The newer per-frame path below uses the two-phase shell
    assignment helper instead.
    """
    labels = np.asarray(nuclear_labels, dtype=np.int32)
    result = labels.copy()
    foreground = np.asarray(prob_mask, dtype=bool)
    flow_arr = np.asarray(flow, dtype=np.float32)
    if flow_arr.ndim != 3:
        raise ValueError("flow must have shape (2, Y, X) or (Y, X, 2)")
    if flow_arr.shape[0] == 2:
        flow_y = flow_arr[0]
        flow_x = flow_arr[1]
    elif flow_arr.shape[-1] == 2:
        flow_y = flow_arr[..., 0]
        flow_x = flow_arr[..., 1]
    else:
        raise ValueError("flow must have shape (2, Y, X) or (Y, X, 2)")

    height, width = labels.shape
    radius = float(capture_radius)
    for y in range(height):
        for x in range(width):
            if labels[y, x] > 0 or not foreground[y, x]:
                continue

            pos_y = float(y)
            pos_x = float(x)
            for _ in range(int(n_steps)):
                iy = min(max(int(pos_y), 0), height - 1)
                ix = min(max(int(pos_x), 0), width - 1)

                if labels[iy, ix] > 0:
                    result[y, x] = labels[iy, ix]
                    break
                if dist[iy, ix] <= radius:
                    nearest_label = labels[int(ny[iy, ix]), int(nx[iy, ix])]
                    if nearest_label > 0:
                        result[y, x] = nearest_label
                        break

                step_y = (
                    float(flow_weight) * float(flow_y[iy, ix])
                    + (1.0 - float(flow_weight)) * float(grav_y[iy, ix])
                )
                step_x = (
                    float(flow_weight) * float(flow_x[iy, ix])
                    + (1.0 - float(flow_weight)) * float(grav_x[iy, ix])
                )
                pos_y = min(max(pos_y + step_y * float(flow_step_scale), 0.0), height - 1)
                pos_x = min(max(pos_x + step_x * float(flow_step_scale), 0.0), width - 1)

    return result


def compute_flow_following_movie(
    foreground_tyx: np.ndarray,
    dp_tcyx: np.ndarray,
    labels_tyx: np.ndarray,
    params: FlowFollowingParams,
    *,
    filter_vectors: bool = True,
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run capture-radius flow-following for a full time series."""
    foreground = np.asarray(foreground_tyx, dtype=bool)
    labels = np.asarray(labels_tyx, dtype=np.int32)
    flow = np.asarray(dp_tcyx, dtype=np.float32)
    if foreground.ndim != 3 or labels.shape != foreground.shape:
        raise ValueError("foreground and labels must have matching shape (T, Y, X)")
    if flow.shape != (foreground.shape[0], 2, foreground.shape[1], foreground.shape[2]):
        raise ValueError("dp_tcyx must have shape (T, 2, Y, X)")

    filtered = compute_filtered_flow_vectors(flow, params) if filter_vectors else flow
    filtered = np.asarray(filtered, dtype=np.float32)
    total = foreground.shape[0]
    cell_labels = np.zeros_like(labels, dtype=np.int32)

    for t in range(total):
        fg = foreground[t]
        nuclei = labels[t]
        if fg.any() and nuclei.max() > 0:
            dist, indices = distance_transform_edt(
                nuclei == 0,
                return_indices=True,
            )
            yy, xx = np.indices(nuclei.shape)
            dy = indices[0].astype(np.float32) - yy.astype(np.float32)
            dx = indices[1].astype(np.float32) - xx.astype(np.float32)
            norm = np.hypot(dy, dx).astype(np.float32)
            norm[norm == 0.0] = 1.0
            grav_y = dy / norm
            grav_x = dx / norm
            cell_labels[t] = _flow_integrate(
                nuclei,
                filtered[t],
                grav_y,
                grav_x,
                dist.astype(np.float32),
                indices[0].astype(np.int32),
                indices[1].astype(np.int32),
                fg,
                n_steps=int(params.max_iterations),
                flow_step_scale=float(params.flow_step_scale),
                flow_weight=float(params.flow_weight),
                capture_radius=float(params.capture_radius),
            )
        if progress_cb is not None:
            progress_cb(t + 1, total)

    return filtered, cell_labels


def _progressive_shell_assign(
    labels,       # (H, W) int32  — from phase 1
    final_y,      # (H, W) float32
    final_x,      # (H, W) float32
    foreground,   # (H, W) bool
    shell_width=_SHELL_WIDTH,
    max_shells=_MAX_SHELLS,
):
    """Phase 2: grow labels outward in shells through displaced positions.

    Each iteration:
      1. Compute EDT from all currently-labelled pixels.
      2. For each unassigned foreground pixel, look up the distance at
         its *final* (flow-displaced) position.
      3. If within ``shell_width`` → assign the nearest label.
      4. Newly labelled pixels (at their *original* positions) seed the
         next EDT — this is how labels propagate through the
         displaced-position topology.

    A final fallback assigns any remaining pixels to their nearest label
    (no distance limit) so that every foreground pixel is guaranteed a
    label.

    Parameters
    ----------
    labels : ndarray (H, W) int32
        Partially assigned label map from phase 1.
    final_y, final_x : ndarray (H, W) float32
        Displaced positions from phase 1.
    foreground : ndarray (H, W) bool
        Foreground mask (True = should be labelled).
    shell_width : float
        Maximum capture distance per iteration (pixels).
    max_shells : int
        Safety cap on the number of growth iterations.

    Returns
    -------
    result : ndarray (H, W) int32
        Fully assigned label map.
    assignment_order : ndarray (H, W) int32
        Shell index at which each pixel was assigned.  0 = pre-assigned
        (nucleus or phase-1 direct capture), 1 = first shell, etc.
        Fallback pixels receive ``last_shell + 1``.  Non-foreground
        pixels are 0 (ignored downstream).
    """

    H, W = labels.shape
    result = labels.copy()
    unassigned = foreground & (result == 0)

    # Track assignment order: 0 = pre-assigned, shells = 1, 2, ...
    assignment_order = np.zeros((H, W), dtype=np.int32)

    # Integer final positions, clipped to image bounds
    fy = np.clip(np.round(final_y).astype(np.intp), 0, H - 1)
    fx = np.clip(np.round(final_x).astype(np.intp), 0, W - 1)

    shell_idx = 0
    for _ in range(max_shells):
        if not unassigned.any():
            break

        shell_idx += 1

        # EDT from every unlabelled pixel to the nearest labelled pixel
        unlabelled_mask = result == 0
        if not unlabelled_mask.any():
            break
        dist, indices = distance_transform_edt(
            unlabelled_mask, return_indices=True,
        )
        ind_y = indices[0]
        ind_x = indices[1]

        # Query at each unassigned pixel's FINAL (displaced) position
        d = dist[fy, fx]
        ny = ind_y[fy, fx]
        nx = ind_x[fy, fx]
        nearest_label = result[ny, nx]

        can_assign = unassigned & (d <= shell_width) & (nearest_label > 0)
        if not can_assign.any():
            break  # stalled — no pixel's displaced position is close enough

        result[can_assign] = nearest_label[can_assign]
        assignment_order[can_assign] = shell_idx
        unassigned &= ~can_assign

    # ------------------------------------------------------------------
    # Fallback: assign any remaining pixels with no distance limit
    # ------------------------------------------------------------------
    fallback_order = shell_idx + 1
    if unassigned.any() and (result > 0).any():
        unlabelled_mask = result == 0
        if unlabelled_mask.any():
            _dist, indices = distance_transform_edt(
                unlabelled_mask, return_indices=True,
            )
            ny = indices[0][fy, fx]
            nx = indices[1][fy, fx]
            nearest_label = result[ny, nx]
            still_open = unassigned & (nearest_label > 0)
            result[still_open] = nearest_label[still_open]
            assignment_order[still_open] = fallback_order

    return result, assignment_order


# ---------------------------------------------------------------------------
# Shell-confidence boundary helpers
# ---------------------------------------------------------------------------


def _inter_cell_boundaries(labels: np.ndarray) -> np.ndarray:
    """Find boundary pixels between different *nonzero* labels (inner, 4-connected).

    Unlike ``skimage.segmentation.find_boundaries(mode="inner")``, this
    ignores label-vs-background edges — only true cell–cell contacts count.
    This prevents the shell-cutoff sweep from producing spurious boundaries
    at the expanding edge of the assigned region.
    """
    H, W = labels.shape
    fg = labels > 0
    bd = np.zeros((H, W), dtype=bool)
    # up
    bd[1:]     |= fg[1:]     & (labels[:-1] > 0) & (labels[1:]     != labels[:-1])
    # down
    bd[:-1]    |= fg[:-1]    & (labels[1:]  > 0) & (labels[:-1]    != labels[1:])
    # left
    bd[:, 1:]  |= fg[:, 1:]  & (labels[:, :-1] > 0) & (labels[:, 1:]  != labels[:, :-1])
    # right
    bd[:, :-1] |= fg[:, :-1] & (labels[:, 1:]  > 0) & (labels[:, :-1] != labels[:, 1:])
    return bd


# ---------------------------------------------------------------------------
# Core two-phase frame helper (returns assignment order)
# ---------------------------------------------------------------------------


def _flow_following_frame_core(
    foreground_yx: np.ndarray,
    dp_cyx: np.ndarray,
    labels_yx: np.ndarray,
    params: FlowFollowingParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-phase flow-following for a single frame.

    Returns
    -------
    cell_labels : (Y, X) int32
    assignment_order : (Y, X) int32
        Shell index at which each pixel was assigned (0 = nucleus / phase-1).
    """

    H, W = foreground_yx.shape

    # ---- normalise flow by mean foreground magnitude ----
    flow = np.ascontiguousarray(dp_cyx, dtype=np.float32).copy()
    mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    mean_mag = float(mag[foreground_yx].mean()) if foreground_yx.any() else 1.0
    if mean_mag > 0:
        flow /= np.float32(mean_mag)

    # ---- EDT from nucleus pixels → gravity directions ----
    nucleus_mask = labels_yx > 0
    dist, indices = distance_transform_edt(
        ~nucleus_mask, return_indices=True,
    )
    near_y = indices[0]
    near_x = indices[1]

    yy, xx = np.mgrid[:H, :W]
    dy = (near_y - yy).astype(np.float64)
    dx = (near_x - xx).astype(np.float64)
    norm = np.sqrt(dy ** 2 + dx ** 2)
    norm[norm == 0] = 1.0
    grav_y = (dy / norm).astype(np.float32)
    grav_x = (dx / norm).astype(np.float32)

    # ---- Phase 1: flow integration ----
    lab32 = np.ascontiguousarray(labels_yx, dtype=np.int32)
    fg_bool = np.ascontiguousarray(foreground_yx, dtype=np.bool_)
    grav_y = np.ascontiguousarray(grav_y)
    grav_x = np.ascontiguousarray(grav_x)

    result, final_y, final_x = _flow_integrate_to_positions(
        lab32, np.ascontiguousarray(flow), grav_y, grav_x,
        fg_bool,
        int(params.max_iterations),
        np.float32(params.flow_step_scale),
        np.float32(params.flow_weight),
    )

    # ---- Phase 2: progressive shell assignment ----
    result, assignment_order = _progressive_shell_assign(
        result, final_y, final_x, fg_bool,
    )

    return result, assignment_order


def compute_flow_following_frame(
    foreground_yx: np.ndarray,
    dp_cyx: np.ndarray,
    labels_yx: np.ndarray,
    params: FlowFollowingParams,
) -> np.ndarray:
    """Run flow-following segmentation on a single frame.

    Two-phase algorithm:
      1. Integrate every foreground pixel along the flow field
         (blended with EDT gravity).  If a pixel lands directly on a
         nucleus, assign immediately.
      2. Grow labels outward in progressive shells: each iteration
         assigns unassigned pixels whose *displaced* positions are
         within a shell width of an already-labelled pixel's *original*
         position.  This lets labels chain-propagate through the
         displaced-position topology.

    Parameters
    ----------
    foreground_yx : ndarray (Y, X), bool
    dp_cyx : ndarray (2, Y, X), float32  — pre-filtered flow vectors
    labels_yx : ndarray (Y, X), int32  — nucleus tracked labels
    params : FlowFollowingParams

    Returns
    -------
    cell_labels : ndarray (Y, X), int32
    """
    if not foreground_yx.any():
        return np.zeros(foreground_yx.shape, dtype=np.int32)
    if labels_yx.max() == 0:
        return np.zeros(foreground_yx.shape, dtype=np.int32)

    result, _order = _flow_following_frame_core(
        foreground_yx, dp_cyx, labels_yx, params,
    )
    return result
