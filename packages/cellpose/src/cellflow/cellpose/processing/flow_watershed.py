"""Flow-guided watershed cell segmentation using Numba-accelerated per-pixel expansion."""

from __future__ import annotations

import math

import numba
import numpy as np
from scipy import ndimage


def _compute_centroids(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centroid_y, centroid_x arrays indexed by label id (0 = background)."""
    max_label = int(labels.max())
    centroid_y = np.zeros(max_label + 1, dtype=np.float32)
    centroid_x = np.zeros(max_label + 1, dtype=np.float32)
    for lid in range(1, max_label + 1):
        pts = np.argwhere(labels == lid)
        if len(pts):
            centroid_y[lid] = pts[:, 0].mean()
            centroid_x[lid] = pts[:, 1].mean()
    return centroid_y, centroid_x


@numba.njit(parallel=True, cache=True)
def _watershed_step(
    result: np.ndarray,       # (H, W) int32 — read-only snapshot of current labels
    next_result: np.ndarray,  # (H, W) int32 — write output here (pre-zeroed)
    flow: np.ndarray,         # (H, W, 2) float32 — flow field pointing toward cell centers
    centroid_y: np.ndarray,   # (max_label+1,) float32
    centroid_x: np.ndarray,   # (max_label+1,) float32
    prob_mask: np.ndarray,    # (H, W) bool — restrict expansion to foreground
    flow_scale: float,
    uniform_growth_rate: float,
    flow_mag_scale: float,
) -> int:
    """Single expansion step.

    For each unassigned foreground pixel, checks all 8-connected labeled neighbours.
    Assigns the pixel to the neighbour whose centroid-to-pixel direction best aligns
    with the (negated) flow field — i.e. the cell most likely to own this pixel.

    The score blends flow-alignment and centroid-distance signals weighted by local
    flow magnitude: strong flow → flow alignment dominates; near-zero flow →
    distance to centroid dominates (Voronoi fallback), eliminating loop-order
    artifacts in low-flow regions.

    Flow alignment decides *which* label wins; every foreground pixel adjacent to
    any label is always assigned (no expansion threshold). Returns number of newly
    assigned pixels.
    """
    H, W = result.shape
    changed = 0

    for i in numba.prange(H):
        for j in range(W):
            cur = result[i, j]
            if cur != 0:
                next_result[i, j] = cur
                continue
            if not prob_mask[i, j]:
                continue

            # Flow magnitude gate — computed once per pixel, not per neighbour
            flow_mag = math.sqrt(flow[i, j, 0] * flow[i, j, 0] + flow[i, j, 1] * flow[i, j, 1])
            flow_weight = math.tanh(flow_mag * flow_mag_scale)

            best_label = 0
            best_score = -2.0  # below minimum possible score; any adjacent label qualifies

            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    if ni < 0 or ni >= H or nj < 0 or nj >= W:
                        continue
                    L = result[ni, nj]
                    if L <= 0:
                        continue

                    # Radial direction from centroid of L to current pixel (i, j)
                    ry = float(i) - centroid_y[L]
                    rx = float(j) - centroid_x[L]
                    dist = math.sqrt(ry * ry + rx * rx)
                    if dist > 1e-6:
                        ry /= dist
                        rx /= dist

                    # Flow alignment score
                    align = ry * (-flow[i, j, 0]) + rx * (-flow[i, j, 1])
                    boost = min(max(align * flow_scale, 0.0), 1.0)
                    flow_score = uniform_growth_rate + (1.0 - uniform_growth_rate) * boost

                    # Distance score — closer centroid is better
                    dist_score = 1.0 / (dist + 1.0)

                    score = flow_weight * flow_score + (1.0 - flow_weight) * dist_score

                    if score > best_score:
                        best_score = score
                        best_label = L

            if best_label > 0:
                next_result[i, j] = best_label
                changed += 1

    return changed


def flow_guided_watershed(
    nuclear_labels: np.ndarray,
    flow_field: np.ndarray,
    cellpose_prob: np.ndarray | None = None,
    flow_scale: float = 1.0,
    cellpose_prob_threshold: float = 0.0,
    flow_smoothing_sigma: float = 0.0,
    max_iterations: int = 50,
    uniform_growth_rate: float = 0.2,
    flow_mag_scale: float = 3.0,
) -> np.ndarray:
    """Flow-guided watershed segmentation — deterministic Numba implementation.

    Expands nuclear seeds outward one pixel-shell per iteration. Each unassigned
    foreground pixel is assigned to whichever adjacent labeled cell's centroid
    direction best aligns with the (negated) cellpose flow field at that pixel.
    Flow alignment decides *which* cell wins at contested boundaries; every
    foreground pixel adjacent to any label is always claimed (no flow threshold).

    Parameters
    ----------
    nuclear_labels : np.ndarray
        Integer label map of segmented nuclei (H, W).
    flow_field : np.ndarray
        2D vector field from cellpose pointing toward cell centers (H, W, 2).
    cellpose_prob : np.ndarray, optional
        Cellpose probability / logit map (H, W). Used to mask expansion:
        pixels below ``cellpose_prob_threshold`` are never assigned.
    flow_scale : float
        Multiplier on flow alignment score — higher values make cell boundaries
        follow the flow more sharply (default 1.0).
    cellpose_prob_threshold : float
        Pixels with ``cellpose_prob < threshold`` are excluded from expansion.
        Set to a very negative value to disable masking (default 0.0).
    flow_smoothing_sigma : float
        Gaussian smoothing applied to the flow field before expansion (default 0.0).
    max_iterations : int
        Maximum expansion iterations. Convergence is checked each step;
        expansion stops early if no new pixels are assigned (default 50).
    uniform_growth_rate : float
        Baseline score when flow alignment is zero. Acts as tie-breaker weight
        between competing labels; does not gate expansion (default 0.2).
    flow_mag_scale : float
        Controls the transition between flow-guided and distance-based scoring.
        ``flow_weight = tanh(flow_magnitude * flow_mag_scale)``: high flow →
        flow alignment dominates; near-zero flow → nearest centroid wins
        (Voronoi fallback). Higher values make the transition sharper (default 3.0).

    Returns
    -------
    np.ndarray
        Integer label map with expanded cell boundaries, dtype int32.
    """
    if flow_smoothing_sigma > 0:
        flow_field = np.stack([
            ndimage.gaussian_filter(flow_field[..., 0], sigma=flow_smoothing_sigma),
            ndimage.gaussian_filter(flow_field[..., 1], sigma=flow_smoothing_sigma),
        ], axis=-1).astype(np.float32)

    prob_mask: np.ndarray
    if cellpose_prob is not None:
        prob_mask = (cellpose_prob >= cellpose_prob_threshold)
    else:
        prob_mask = np.ones(nuclear_labels.shape[:2], dtype=np.bool_)

    # Seeds are always in the mask regardless of probability
    prob_mask = prob_mask | (nuclear_labels > 0)

    result = nuclear_labels.copy().astype(np.int32)
    next_result = np.zeros_like(result)
    centroid_y, centroid_x = _compute_centroids(result)

    for _ in range(max_iterations):
        next_result[:] = 0
        changed = _watershed_step(
            result, next_result, flow_field.astype(np.float32),
            centroid_y, centroid_x, prob_mask.astype(np.bool_),
            float(flow_scale), float(uniform_growth_rate), float(flow_mag_scale),
        )
        result, next_result = next_result, result
        if changed == 0:
            break

    return result
