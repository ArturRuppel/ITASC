"""Centroid-distance LAP retracker for relabelling corrected frames.

Given a reference frame (whose IDs are trusted) and a target frame (which may
contain arbitrarily-assigned IDs after manual correction), this module remaps
the target IDs so that cells matching the reference keep the reference ID.

Unmatched target cells (new appearances) receive fresh IDs that do not collide
with any ID already present in either frame.
"""
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


def retrack_frame(
    ref_labels: np.ndarray,
    target_labels: np.ndarray,
    max_dist_px: float = 50.0,
) -> np.ndarray:
    """Remap cell IDs in *target_labels* to match *ref_labels* by centroid proximity.

    Each target cell is matched to the nearest reference cell within max_dist_px
    using the Hungarian algorithm.  Matched target cells receive the reference ID.
    Unmatched target cells (new appearances) receive IDs above the current maximum
    to avoid collisions.

    Parameters
    ----------
    ref_labels:
        (Y, X) uint32 — the trusted reference frame.
    target_labels:
        (Y, X) uint32 — the frame to relabel (e.g. after manual correction).
    max_dist_px:
        Maximum centroid distance for a valid match.  Pairs further apart are
        treated as unmatched.

    Returns
    -------
    (Y, X) uint32 array with IDs remapped to match ref_labels where possible.
    """
    ref_centroids = _centroids(ref_labels)
    tgt_centroids = _centroids(target_labels)

    if not tgt_centroids:
        return target_labels.copy()

    result = np.zeros_like(target_labels)

    if not ref_centroids:
        # No reference cells — assign fresh sequential IDs to all target cells.
        next_id = 1
        for tid in tgt_centroids:
            result[target_labels == tid] = next_id
            next_id += 1
        return result

    ref_ids = list(ref_centroids.keys())
    tgt_ids = list(tgt_centroids.keys())
    ref_pts = np.array([ref_centroids[i] for i in ref_ids])
    tgt_pts = np.array([tgt_centroids[i] for i in tgt_ids])

    n_ref, n_tgt = len(ref_ids), len(tgt_ids)

    # Cost matrix: rows = target cells, cols = reference cells.
    cost = np.full((n_tgt, n_ref), fill_value=np.inf)
    for ti, tp in enumerate(tgt_pts):
        for ri, rp in enumerate(ref_pts):
            d = float(np.linalg.norm(tp - rp))
            if d <= max_dist_px:
                cost[ti, ri] = d

    # Solve assignment (minimise cost).  Pairs where cost is inf are blocked by
    # replacing inf with a large finite sentinel so scipy doesn't reject the matrix.
    sentinel = max_dist_px * 10 * (n_tgt + n_ref + 1)
    finite_cost = np.where(np.isinf(cost), sentinel, cost)
    row_ind, col_ind = linear_sum_assignment(finite_cost)

    # Build remapping: target_id -> new_id
    remap: dict[int, int] = {}
    used_ref_ids: set[int] = set()
    for ti, ri in zip(row_ind, col_ind):
        if cost[ti, ri] <= max_dist_px:
            remap[tgt_ids[ti]] = ref_ids[ri]
            used_ref_ids.add(ref_ids[ri])

    # Assign fresh IDs to unmatched target cells, above the current max.
    max_existing = max(
        int(ref_labels.max()) if ref_labels.max() > 0 else 0,
        int(target_labels.max()) if target_labels.max() > 0 else 0,
    )
    next_id = max_existing + 1
    for tid in tgt_ids:
        if tid not in remap:
            remap[tid] = next_id
            next_id += 1

    # Apply remapping.
    for tid, new_id in remap.items():
        result[target_labels == tid] = new_id

    return result


def retrack_frame_constrained(
    ref_labels: np.ndarray,
    target_labels: np.ndarray,
    locked_target_ids: set[int],
    max_dist_px: float = 50.0,
    reserved_ids: set[int] | None = None,
) -> np.ndarray:
    """Like retrack_frame, but target cells whose ID is in locked_target_ids
    keep their existing IDs unchanged. The IDs they hold are also reserved —
    no other (non-locked) target cell may be remapped onto them.

    Locked target cells are copied straight to the output and excluded from the
    LAP entirely. Reference cells that share an ID with a locked target cell are
    also excluded from the LAP — their ID is already occupied, so letting the
    LAP assign it to an unlocked cell would create a collision.

    Additional reserved_ids are protected from assignment to unlocked targets
    even if they are present in the reference frame.
    """
    locked_target_ids = set(locked_target_ids)  # defensive copy
    reserved_ids = set(reserved_ids or set())
    blocked_ids = locked_target_ids | reserved_ids

    ref_centroids = _centroids(ref_labels)
    tgt_centroids = _centroids(target_labels)

    result = np.zeros_like(target_labels)

    # Copy locked cells directly to output first; they participate in nothing else.
    for lid in locked_target_ids:
        if lid in tgt_centroids:
            result[target_labels == lid] = lid

    unlocked_tgt_ids = [tid for tid in tgt_centroids if tid not in locked_target_ids]

    if not unlocked_tgt_ids:
        return result

    # Protected IDs are excluded so the LAP cannot assign them to an unlocked
    # target cell.
    available_ref_ids = [rid for rid in ref_centroids if rid not in blocked_ids]

    if not available_ref_ids:
        # No usable reference cells — assign fresh IDs to all unlocked targets.
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
