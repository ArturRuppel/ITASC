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


def _compute_boundary_weight_vectorized(
    cell_labels: np.ndarray,   # (H, W) int32
    order: np.ndarray,         # (H, W) int32  — shell assignment order
) -> np.ndarray:
    """Shell-confidence weighted boundary in O(H·W) instead of O(max_order·H·W).

    Key insight: a pixel (i, j) first becomes a boundary at the cutoff
    ``k_onset = min over qualifying 4-neighbours of max(order[i,j], order[n])``
    and then *stays* a boundary for every subsequent cutoff (inclusion is
    monotone).  Its weight is therefore
    ``(max_order + 1 − k_onset) / (max_order + 1)``.
    """
    H, W = cell_labels.shape
    max_order = int(order.max())
    n_cutoffs = max_order + 1
    if n_cutoffs == 0:
        return np.zeros((H, W), dtype=np.float32)

    fg = cell_labels > 0
    SENTINEL = np.int32(max_order + 2)
    k_onset = np.full((H, W), SENTINEL, dtype=np.int32)

    # Pad so shifted indexing never goes out of bounds
    lab_pad = np.pad(cell_labels, 1, constant_values=0)
    ord_pad = np.pad(order, 1, constant_values=SENTINEL)

    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        n_lab = lab_pad[1 + dr : H + 1 + dr, 1 + dc : W + 1 + dc]
        n_ord = ord_pad[1 + dr : H + 1 + dr, 1 + dc : W + 1 + dc]

        qualifies = fg & (n_lab > 0) & (cell_labels != n_lab)
        onset = np.maximum(order, n_ord)

        # Keep the earliest onset across all qualifying neighbours
        np.minimum(
            k_onset,
            np.where(qualifies, onset, SENTINEL),
            out=k_onset,
        )

    result = np.zeros((H, W), dtype=np.float32)
    valid = k_onset <= max_order
    result[valid] = (
        (max_order + 1 - k_onset[valid]).astype(np.float32) / n_cutoffs
    )
    return result


def build_consensus_boundary_flow_following(
    prob_yx: np.ndarray,
    dp_cyx: np.ndarray,
    labels_yx: np.ndarray,
    cellprob_thresholds: list[float],
    params: FlowFollowingParams,
    reduction: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Build a consensus contour map using flow-following + EDT gravity.

    Changes vs. the original implementation
    ----------------------------------------
    * The inner shell-cutoff sweep is replaced by a closed-form
      vectorised computation (``_compute_boundary_weight_vectorized``),
      reducing work from O(max_order · H · W) to O(H · W) per threshold.
    * The outer threshold loop is executed in a
      ``concurrent.futures.ThreadPoolExecutor``.  The heavy callees
      (numba ``parallel=True``, scipy EDT) all release the GIL, so
      real parallelism is achieved without process-spawn overhead.
    """
    from concurrent.futures import ThreadPoolExecutor

    H, W = prob_yx.shape

    # -- worker executed once per threshold ---------------------------
    def _process_threshold(thresh: float):
        fg_mask = prob_yx > thresh
        if not fg_mask.any():
            return None
        cell_labels, order = _flow_following_frame_core(
            fg_mask, dp_cyx, labels_yx, params,
        )
        boundary = _compute_boundary_weight_vectorized(cell_labels, order)
        foreground = (cell_labels > 0).astype(np.float32)
        return boundary, foreground

    # -- fan out across threads ---------------------------------------
    with ThreadPoolExecutor() as pool:
        results = list(pool.map(_process_threshold, cellprob_thresholds))

    # -- reduce -------------------------------------------------------
    boundary_accum = np.zeros((H, W), dtype=np.float32)
    foreground_accum = np.zeros((H, W), dtype=np.float32)
    n = 0
    for r in results:
        if r is not None:
            boundary_accum += r[0]
            foreground_accum += r[1]
            n += 1

    if n > 0 and reduction == "mean":
        boundary_accum /= n
        foreground_accum /= n

    return boundary_accum, foreground_accum