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
    from scipy.ndimage import distance_transform_edt

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
    from scipy.ndimage import distance_transform_edt

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
    """Run flow-following segmentation on a **single frame**.

    Dispatches between two strategies based on ``params.capture_radius``:

    * ``capture_radius > 0`` — legacy behaviour: wraps
      :func:`compute_flow_following_movie` with ``T=1``.  A pixel is
      captured when it enters the fixed radius around a nucleus.
    * ``capture_radius == 0`` — new two-phase algorithm:
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

    # ------------------------------------------------------------------
    # Legacy path (capture_radius > 0): delegate to existing movie fn
    # ------------------------------------------------------------------
    if params.capture_radius > 0:
        fg_tyx = np.ascontiguousarray(foreground_yx[np.newaxis], dtype=bool)
        dp_tcyx = np.ascontiguousarray(dp_cyx[np.newaxis], dtype=np.float32)
        lab_tyx = np.ascontiguousarray(labels_yx[np.newaxis], dtype=np.int32)
        _, cell_labels_tyx = compute_flow_following_movie(
            fg_tyx, dp_tcyx, lab_tyx, params,
            progress_cb=None, filter_vectors=False,
        )
        return cell_labels_tyx[0]

    # ------------------------------------------------------------------
    # New path (capture_radius == 0): two-phase progressive assignment
    # ------------------------------------------------------------------
    result, _order = _flow_following_frame_core(
        foreground_yx, dp_cyx, labels_yx, params,
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

    This mirrors :func:`build_consensus_boundary_2d` but replaces Cellpose's
    ``compute_masks`` with :func:`compute_flow_following_frame`.  For every
    *cellprob_threshold* the probability map is binarised into a foreground
    mask; flow-following assigns each foreground pixel to the nearest nucleus;
    boundaries are extracted and accumulated across thresholds.

    When ``params.capture_radius == 0`` (two-phase algorithm), the function
    additionally sweeps **shell-confidence cutoffs**: for each threshold, the
    full label map is progressively truncated to pixels assigned within the
    first *k* shells.  Inter-cell boundaries are extracted at every cutoff
    and averaged.  Boundaries between regions that are both assigned early
    (high confidence) appear at all cutoffs and receive a high contour score.
    Boundaries involving late-assigned (ambiguous) pixels appear only at high
    cutoffs and receive a low score.  This naturally hardens stable boundaries
    and weakens uncertain ones.

    Parameters
    ----------
    prob_yx : ndarray, shape (Y, X), dtype float32
        Z-averaged, gamma-corrected probability logits.
    dp_cyx : ndarray, shape (2, Y, X), dtype float32
        Pre-filtered 2-D flow vectors.
    labels_yx : ndarray, shape (Y, X), dtype int32
        Nucleus tracked labels for this frame.
    cellprob_thresholds : list[float]
        Thresholds swept for the consensus boundary.
    params : FlowFollowingParams
        Integration hyper-parameters.
    reduction : {"mean", "sum"}
        How to reduce across thresholds.

    Returns
    -------
    boundary : ndarray, shape (Y, X), dtype float32
        Accumulated boundary confidence in [0, 1] (if *reduction* = "mean").
    foreground : ndarray, shape (Y, X), dtype float32
        Accumulated foreground score in [0, 1] (if *reduction* = "mean").
    """
    from skimage.segmentation import find_boundaries

    H, W = prob_yx.shape
    boundary_accum = np.zeros((H, W), dtype=np.float32)
    foreground_accum = np.zeros((H, W), dtype=np.float32)
    n = 0

    use_shell_sweep = params.capture_radius == 0

    for thresh in cellprob_thresholds:
        fg_mask = prob_yx > thresh
        if not fg_mask.any():
            continue

        if use_shell_sweep:
            # ----------------------------------------------------------
            # Two-phase path: sweep shell-confidence cutoffs
            # ----------------------------------------------------------
            cell_labels, order = _flow_following_frame_core(
                fg_mask, dp_cyx, labels_yx, params,
            )

            max_order = int(order.max())

            # Accumulate boundaries at each cutoff, then normalise
            # within this threshold so each threshold contributes equally.
            thresh_boundary = np.zeros((H, W), dtype=np.float32)
            n_cutoffs = 0

            for cutoff in range(0, max_order + 1):
                partial = np.where(order <= cutoff, cell_labels, 0)
                bd = _inter_cell_boundaries(partial)
                thresh_boundary += bd.astype(np.float32)
                n_cutoffs += 1

            if n_cutoffs > 0:
                thresh_boundary /= n_cutoffs

            boundary_accum += thresh_boundary
            # Foreground score: from the full label map (not truncated)
            foreground_accum += (cell_labels > 0).astype(np.float32)
            n += 1

        else:
            # ----------------------------------------------------------
            # Legacy path: single boundary extraction per threshold
            # ----------------------------------------------------------
            cell_labels = compute_flow_following_frame(
                fg_mask, dp_cyx, labels_yx, params,
            )

            boundary_accum += find_boundaries(
                cell_labels, mode="inner",
            ).astype(np.float32)
            foreground_accum += (cell_labels > 0).astype(np.float32)
            n += 1

    if n > 0 and reduction == "mean":
        boundary_accum /= n
        foreground_accum /= n

    return boundary_accum, foreground_accum