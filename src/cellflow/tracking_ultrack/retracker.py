"""Centroid-distance LAP retracker for relabelling corrected frames."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.optimize import linear_sum_assignment


def _centroids(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Return {label_id: centroid_yx} for all non-zero labels."""
    ids = [int(i) for i in np.unique(labels) if i != 0]
    if not ids:
        return {}
    coms = center_of_mass(np.ones_like(labels), labels, ids)
    if len(ids) == 1:
        coms = [coms]
    return {lid: np.array(com).ravel() for lid, com in zip(ids, coms)}


def retrack_frame_constrained(
    ref_labels: np.ndarray,
    target_labels: np.ndarray,
    locked_target_ids: set[int],
    max_dist_px: float = 50.0,
    reserved_ids: set[int] | None = None,
) -> np.ndarray:
    """Remap target IDs by centroid proximity without changing locked targets.

    Target cells whose ID is in locked_target_ids keep their existing IDs.
    Those IDs, plus any reserved_ids, are protected from assignment to unlocked
    target cells even if a matching reference cell exists.
    """
    locked_target_ids = set(locked_target_ids)
    reserved_ids = set(reserved_ids or set())
    blocked_ids = locked_target_ids | reserved_ids

    ref_centroids = _centroids(ref_labels)
    tgt_centroids = _centroids(target_labels)

    result = np.zeros_like(target_labels)

    for lid in locked_target_ids:
        if lid in tgt_centroids:
            result[target_labels == lid] = lid

    unlocked_tgt_ids = [tid for tid in tgt_centroids if tid not in locked_target_ids]

    if not unlocked_tgt_ids:
        return result

    available_ref_ids = [rid for rid in ref_centroids if rid not in blocked_ids]

    if not available_ref_ids:
        max_existing = max(
            int(ref_labels.max()) if ref_labels.max() > 0 else 0,
            int(target_labels.max()) if target_labels.max() > 0 else 0,
        )
        next_id = max_existing + 1
        for tid in unlocked_tgt_ids:
            while next_id in blocked_ids:
                next_id += 1
            result[target_labels == tid] = next_id
            next_id += 1
        return result

    ref_pts = np.array([ref_centroids[i] for i in available_ref_ids])
    tgt_pts = np.array([tgt_centroids[i] for i in unlocked_tgt_ids])

    n_ref, n_tgt = len(available_ref_ids), len(unlocked_tgt_ids)

    cost = np.full((n_tgt, n_ref), fill_value=np.inf)
    for ti, tp in enumerate(tgt_pts):
        for ri, rp in enumerate(ref_pts):
            d = float(np.linalg.norm(tp - rp))
            if d <= max_dist_px:
                cost[ti, ri] = d

    sentinel = max_dist_px * 10 * (n_tgt + n_ref + 1)
    finite_cost = np.where(np.isinf(cost), sentinel, cost)
    row_ind, col_ind = linear_sum_assignment(finite_cost)

    remap: dict[int, int] = {}
    for ti, ri in zip(row_ind, col_ind):
        if cost[ti, ri] <= max_dist_px:
            remap[unlocked_tgt_ids[ti]] = available_ref_ids[ri]

    max_existing = max(
        int(ref_labels.max()) if ref_labels.max() > 0 else 0,
        int(target_labels.max()) if target_labels.max() > 0 else 0,
    )
    next_id = max_existing + 1
    for tid in unlocked_tgt_ids:
        if tid not in remap:
            while next_id in blocked_ids:
                next_id += 1
            remap[tid] = next_id
            next_id += 1

    for tid, new_id in remap.items():
        result[target_labels == tid] = new_id

    return result
