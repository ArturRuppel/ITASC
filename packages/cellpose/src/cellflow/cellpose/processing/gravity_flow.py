"""Gravity-flow cell segmentation via Euler integration of blended gravity + cellpose flow fields."""

from __future__ import annotations

import math
import numba
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt


@numba.njit(parallel=True, cache=True)
def _gravity_from_centroids(
    centroids_y: np.ndarray,  # (N,) float32
    centroids_x: np.ndarray,  # (N,) float32
    H: int,
    W: int,
    falloff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """N-body gravitational field: sum contributions from all nuclear centroids.

    Each centroid contributes a vector (c − p) / |c − p|^(falloff+1), so the
    influence decays as 1/r^falloff with distance.  falloff=2 is the classic
    inverse-square law; falloff=1 gives a softer, longer-range field.
    The result is normalised to unit vectors.
    """
    N = len(centroids_y)
    grav_y = np.zeros((H, W), dtype=np.float32)
    grav_x = np.zeros((H, W), dtype=np.float32)

    for i in numba.prange(H):
        for j in range(W):
            gy = 0.0
            gx = 0.0
            for k in range(N):
                dy = centroids_y[k] - float(i)
                dx = centroids_x[k] - float(j)
                dist = math.sqrt(dy * dy + dx * dx)
                if dist > 1e-6:
                    w = 1.0 / math.pow(dist, falloff + 1.0)
                    gy += dy * w
                    gx += dx * w
            mag = math.sqrt(gy * gy + gx * gx)
            if mag > 1e-6:
                grav_y[i, j] = gy / mag
                grav_x[i, j] = gx / mag

    return grav_y, grav_x


def _compute_gravity_field(
    nuclear_labels: np.ndarray,
    falloff: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gravitational field from nuclear centroids + EDT distance for capture check.

    Each nucleus is treated as a point mass at its centroid.  The field at each
    pixel is the normalised sum of inverse-power contributions from all nuclei,
    giving a smooth, physics-motivated pull toward cell centres.

    Returns
    -------
    dist_to_nucleus : (H, W) float32 — EDT distance to nearest nuclear pixel
    grav_y : (H, W) float32 — y-component of normalised gravity vector
    grav_x : (H, W) float32 — x-component of normalised gravity vector
    """
    from scipy.ndimage import center_of_mass

    dist_to_nucleus = distance_transform_edt(nuclear_labels == 0).astype(np.float32)

    label_ids = np.unique(nuclear_labels)
    label_ids = label_ids[label_ids > 0]
    if len(label_ids) == 0:
        H, W = nuclear_labels.shape
        return dist_to_nucleus, np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)

    coords = center_of_mass(nuclear_labels > 0, nuclear_labels, label_ids)
    centroids_y = np.array([c[0] for c in coords], dtype=np.float32)
    centroids_x = np.array([c[1] for c in coords], dtype=np.float32)

    H, W = nuclear_labels.shape
    grav_y, grav_x = _gravity_from_centroids(centroids_y, centroids_x, H, W, float(falloff))

    # Zero out gravity inside nuclear labels — particle has already arrived
    inside = nuclear_labels > 0
    grav_y[inside] = 0.0
    grav_x[inside] = 0.0
    return dist_to_nucleus, grav_y, grav_x


@numba.njit(parallel=True, cache=True)
def _flow_integrate(
    nuclear_labels: np.ndarray,  # (H, W) int32
    flow: np.ndarray,            # (H, W, 2) float32 — cellpose (dY, dX), normalised to mean mag 1
    grav_y: np.ndarray,          # (H, W) float32 — gravity unit-vector y
    grav_x: np.ndarray,          # (H, W) float32 — gravity unit-vector x
    dist_to_nucleus: np.ndarray, # (H, W) float32 — EDT distance to nearest nuclear pixel
    nearest_y: np.ndarray,       # (H, W) int32
    nearest_x: np.ndarray,       # (H, W) int32
    prob_mask: np.ndarray,       # (H, W) bool
    n_steps: int,
    flow_step_scale: float,
    flow_weight: float,
    capture_radius: float,
) -> np.ndarray:
    """Assign each foreground pixel to a nuclear label via blended Euler integration.

    At each step the displacement is a blend of two fields:

      step = flow_weight × cellpose_flow + (1 − flow_weight) × gravity

    Both fields are normalised to mean magnitude 1 before integration, so
    flow_weight ∈ [0, 1] directly controls the balance: 0 = pure gravity,
    1 = pure cellpose flow.
    """
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
                # Bilinear interpolation of cellpose flow at (py, px)
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

                # Euler step
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


def stochastic_watershed(
    nuclear_labels: np.ndarray,
    flow_field: np.ndarray,
    cellpose_prob: np.ndarray | None = None,
    flow_scale: float = 1.0,
    cellpose_prob_threshold: float = 0.0,
    max_iterations: int = 50,
    uniform_growth_rate: float = 0.2,
) -> np.ndarray:
    """Stochastic iterative cell expansion anchored to nuclear seeds.

    Each label is grown outward one boundary ring at a time.  The expansion
    probability at a candidate boundary pixel is::

        p = uniform_growth_rate + (1 − uniform_growth_rate) × clip(alignment × flow_scale, 0, 1)

    where ``alignment`` is the dot product of the negated Cellpose flow with
    the outward radial direction from the nuclear centroid.  An optional
    ``cellpose_prob`` map masks out background pixels below
    ``cellpose_prob_threshold``.  After the iterative phase any foreground
    pixels that were not reached are assigned via nearest-label Voronoi.

    Parameters
    ----------
    nuclear_labels : (H, W) int32
        Integer label map of segmented nuclei.
    flow_field : (H, W, 2) float32
        Cellpose displacement field pointing toward cell centres.
    cellpose_prob : (H, W) float32, optional
        Cellpose probability / logit map.  Values below
        ``cellpose_prob_threshold`` are excluded.
    flow_scale : float
        Scales the flow-alignment contribution to the expansion probability.
    cellpose_prob_threshold : float
        Pixels below this value are excluded from expansion (default 0.0).
    max_iterations : int
        Maximum dilation rounds (default 50).
    uniform_growth_rate : float
        Baseline probability that any boundary candidate is assigned
        regardless of flow alignment (default 0.2).

    Returns
    -------
    (H, W) int32
        Label map with expanded cell bodies.
    """
    H, W = nuclear_labels.shape

    label_ids = np.unique(nuclear_labels)
    label_ids = label_ids[label_ids > 0]
    centroids: dict[int, np.ndarray] = {}
    for lbl in label_ids:
        pts = np.argwhere(nuclear_labels == lbl)
        if len(pts) > 0:
            centroids[int(lbl)] = pts.mean(axis=0)

    prob_mask: np.ndarray | None = None
    if cellpose_prob is not None:
        prob_mask = cellpose_prob >= cellpose_prob_threshold

    result = nuclear_labels.copy().astype(np.int32)

    for _ in range(max_iterations):
        unassigned = result == 0
        if not unassigned.any():
            break

        for lbl in np.unique(result[result > 0]):
            lbl = int(lbl)
            if lbl not in centroids:
                continue

            boundary = ndimage.binary_dilation(result == lbl) & unassigned
            if not boundary.any():
                continue

            pts = np.argwhere(boundary)
            center = centroids[lbl]

            radial = pts.astype(float) - center
            norms = np.linalg.norm(radial, axis=1, keepdims=True)
            radial = np.where(norms > 1e-6, radial / norms, 0.0)

            neg_flow = -flow_field[pts[:, 0], pts[:, 1]]  # (N, 2)
            alignment = np.sum(radial * neg_flow, axis=1)
            boost = np.clip(alignment * flow_scale, 0.0, 1.0)
            prob = uniform_growth_rate + (1.0 - uniform_growth_rate) * boost

            if prob_mask is not None:
                prob = np.where(prob_mask[pts[:, 0], pts[:, 1]], prob, 0.0)

            expanded = pts[np.random.rand(len(pts)) < prob]
            if len(expanded) > 0:
                result[expanded[:, 0], expanded[:, 1]] = lbl

    if prob_mask is not None:
        full_mask = prob_mask | (nuclear_labels > 0)
    else:
        full_mask = np.ones((H, W), dtype=np.bool_)
    return _fill_foreground(result, full_mask)


def _fill_foreground(result: np.ndarray, prob_mask: np.ndarray) -> np.ndarray:
    """Safety-net Voronoi fill for any pixels that still didn't converge."""
    unfilled = (result == 0) & prob_mask
    if not unfilled.any():
        return result
    filled = result.copy()
    _, (yi, xi) = distance_transform_edt(filled == 0, return_indices=True)
    filled[unfilled] = result[yi[unfilled], xi[unfilled]]
    return filled


def gravity_flow_segmentation(
    nuclear_labels: np.ndarray,
    flow_field: np.ndarray,
    cellpose_prob: np.ndarray | None = None,
    flow_step_scale: float = 0.2,
    cellpose_prob_threshold: float = 0.0,
    flow_smoothing_sigma: float = 0.0,
    max_iterations: int = 100,
    capture_radius: float = 3.0,
    flow_weight: float = 0.5,
    gravity_falloff: float = 2.0,
) -> np.ndarray:
    """Flow-guided cell segmentation via blended Euler integration.

    Each foreground pixel is advected along a blend of two fields:

    * **Cellpose flow** — normalised to mean magnitude 1 over foreground pixels.
    * **Gravity field** — unit vectors toward the nearest nuclear centroid.

    The blend is ``step = flow_weight × cellpose_flow + (1 − flow_weight) × gravity``
    where ``flow_weight ∈ [0, 1]``: 0 = pure gravity, 1 = pure cellpose flow.

    Parameters
    ----------
    nuclear_labels : np.ndarray
        Integer label map of segmented nuclei (H, W).
    flow_field : np.ndarray
        2D vector field from cellpose pointing toward cell centres (H, W, 2).
    cellpose_prob : np.ndarray, optional
        Cellpose probability / logit map (H, W). Pixels below
        ``cellpose_prob_threshold`` are excluded.
    flow_step_scale : float
        Scale applied to the blended displacement at each Euler step (default 0.2).
    cellpose_prob_threshold : float
        Pixels below this probability are excluded from expansion (default 0.0).
    flow_smoothing_sigma : float
        Gaussian smoothing applied to the cellpose flow before integration (default 0.0).
    max_iterations : int
        Maximum Euler steps per pixel (default 100).
    capture_radius : float
        Euclidean distance (pixels) within which a particle is captured by the
        nearest nuclear seed (default 3.0).
    flow_weight : float
        Blend weight in [0, 1]: 0 = pure gravity, 1 = pure cellpose flow (default 0.5).
        Both fields are normalised to mean magnitude 1 before blending.
    gravity_falloff : float
        Exponent controlling how quickly each nucleus's gravitational influence
        decays with distance: ``weight ∝ 1 / dist^falloff``.  2.0 = classic
        inverse-square law; 1.0 = softer inverse-distance (default 2.0).

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

    if cellpose_prob is not None:
        prob_mask = (cellpose_prob >= cellpose_prob_threshold)
    else:
        prob_mask = np.ones(nuclear_labels.shape[:2], dtype=np.bool_)

    prob_mask = prob_mask | (nuclear_labels > 0)

    # Normalise cellpose flow to mean magnitude 1 over foreground so that
    # flow_weight ∈ [0,1] directly controls the blend with the gravity field.
    flow_mags = np.sqrt((flow_field ** 2).sum(axis=-1))
    mean_mag = float(flow_mags[prob_mask].mean()) if prob_mask.any() else 1.0
    if mean_mag > 1e-6:
        flow_field = (flow_field / mean_mag).astype(np.float32)

    dist_to_nucleus, grav_y, grav_x = _compute_gravity_field(nuclear_labels, falloff=gravity_falloff)

    _, (nearest_y, nearest_x) = distance_transform_edt(nuclear_labels == 0, return_indices=True)

    result = _flow_integrate(
        nuclear_labels.astype(np.int32),
        flow_field.astype(np.float32),
        grav_y,
        grav_x,
        dist_to_nucleus,
        nearest_y.astype(np.int32),
        nearest_x.astype(np.int32),
        prob_mask.astype(np.bool_),
        int(max_iterations),
        float(flow_step_scale),
        float(flow_weight),
        float(capture_radius),
    )

    result = _fill_foreground(result, prob_mask)

    return result
