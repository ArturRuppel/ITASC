"""Global LAP-based propagator for nucleus tracking.

For each nucleus in the current tracked frame, builds a score matrix against
all candidate nuclei across all (hypothesis, z-slice) combinations for the
next timepoint, then solves the linear assignment problem globally to find the
maximum-weight bipartite matching. Preserves track IDs in the written frame.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree

from cellflow.database.hypotheses import read_hypothesis_labels, list_hypotheses
from cellflow.database.tracked import read_tracked_frame, write_tracked_frame


def _label_stats(labels: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return (areas, centroids) without building per-label boolean masks."""
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return np.zeros(1, dtype=np.int64), {}
    areas = np.bincount(labels.ravel())
    coms = center_of_mass(np.ones_like(labels), labels, ids.tolist())
    if len(ids) == 1:
        coms = [coms]
    centroids = {int(lid): np.array(com).ravel() for lid, com in zip(ids, coms)}
    return areas, centroids


def _label_rel_pixels(
    labels: np.ndarray, centroids: dict[int, np.ndarray]
) -> dict[int, frozenset[tuple[int, int]]]:
    """Return per-label pixel coordinate sets relative to each label's centroid."""
    ys, xs = np.nonzero(labels)
    vals = labels[ys, xs]
    result = {}
    for lid, centroid in centroids.items():
        mask = vals == lid
        cy, cx = np.round(centroid).astype(int)
        result[lid] = frozenset(zip(
            (ys[mask] - cy).tolist(),
            (xs[mask] - cx).tolist(),
        ))
    return result


def _cluster_candidates(
    centroids: np.ndarray,
    dedup_radius_px: float,
) -> list[list[int]]:
    """Group flat candidate indices by centroid proximity using union-find.

    Candidates within dedup_radius_px of each other are treated as the same
    physical cell (e.g. the same nucleus appearing in multiple hypothesis slices).
    """
    n = len(centroids)
    if n == 0:
        return []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = KDTree(centroids)
    for i in range(n):
        for j in tree.query_ball_point(centroids[i], dedup_radius_px):
            if j <= i:
                continue
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def find_best_hypothesis(
    current_labels: np.ndarray,
    candidates: list[np.ndarray],
    iou_threshold: float = 0.3,
    max_dist_px: float = 50.0,
    predicted_centroids: dict[int, np.ndarray] | None = None,
    velocity_sigma_px: float = 25.0,
    unmatched_score: float = 0.1,
    iou_weight: float = 1.0,
    area_weight: float = 1.0,
    velocity_weight: float = 1.0,
    pos_weight: float = 1.0,
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Return (relabeled_next_frame, winning_p_index) or (None, None).

    Builds a score matrix S[i, k] over all current nuclei i and all candidate
    cell clusters k, then solves the linear assignment problem globally.

    Parameters
    ----------
    current_labels:
        (Y, X) uint32 tracked label image for the current frame.
    candidates:
        List of (Y, X) uint32 label images — one per (p, z) combination.
    iou_threshold:
        Minimum IoU to accept a match (hard gate).
    max_dist_px:
        Candidate nuclei whose centroid is farther than this are skipped.
    predicted_centroids:
        Optional dict mapping nucleus ID → predicted centroid.
    velocity_sigma_px:
        Standard deviation (pixels) for the velocity Gaussian.
    unmatched_score:
        Score assigned to the "null" option for each nucleus.
    iou_weight:
        Exponent applied to the IoU term.
    area_weight:
        Exponent applied to the area ratio term.
    velocity_weight:
        Exponent applied to the velocity Gaussian term.
    pos_weight:
        Exponent applied to the positional Gaussian term.
    """
    if not candidates:
        return None, None

    cur_areas, cur_centroids = _label_stats(current_labels)
    cur_ids = sorted(cur_centroids.keys())
    if not cur_ids:
        return None, None

    cur_rel_pixels = _label_rel_pixels(current_labels, cur_centroids)

    flat_cands: list[tuple[int, int, np.ndarray, int, frozenset]] = []
    for entry_idx, cand in enumerate(candidates):
        c_areas, c_centroids = _label_stats(cand)
        c_rel_pixels = _label_rel_pixels(cand, c_centroids)
        for cand_id, centroid in c_centroids.items():
            flat_cands.append((entry_idx, cand_id, centroid, int(c_areas[cand_id]), c_rel_pixels[cand_id]))

    if not flat_cands:
        return None, None

    C = len(flat_cands)
    two_sigma_sq = 2.0 * velocity_sigma_px ** 2
    N = len(cur_ids)

    # Score matrix: rows = source nuclei, cols = candidate cells + N nulls.
    S = np.full((N, C + N), unmatched_score, dtype=np.float64)
    # The null columns (C to C+N-1) must have score=unmatched_score.
    # The assignment logic handles S[i, C+i] = unmatched_score, others are 0.
    for i in range(N):
        S[i, 0:C] = 0.0 # Candidates init
        S[i, C+i] = unmatched_score # Null init

    for i, current_id in enumerate(cur_ids):
        cur_centroid = cur_centroids[current_id]
        cur_area = int(cur_areas[current_id])
        cur_rel = cur_rel_pixels[current_id]
        pred_centroid = predicted_centroids.get(current_id) if predicted_centroids else None
        
        for k, (entry_idx, cand_id, cand_centroid, cand_area, cand_rel) in enumerate(flat_cands):
            # Distance filter
            if np.sqrt(np.sum((cur_centroid - cand_centroid) ** 2)) > max_dist_px:
                continue

            # Standard IoU
            inter = len(cur_rel & cand_rel)
            union = cur_area + cand_area - inter
            iou = inter / union if union > 0 else 0.0
            if iou < iou_threshold:
                continue

            area_ratio = min(cur_area, cand_area) / max(cur_area, cand_area)

            # Positional score
            pos_d2 = float(np.sum((cur_centroid - cand_centroid) ** 2))
            pos_score = np.exp(-pos_d2 / two_sigma_sq)

            # Velocity prediction
            if pred_centroid is not None:
                vel_d2 = float(np.sum((pred_centroid - cand_centroid) ** 2))
                vel_score = np.exp(-vel_d2 / two_sigma_sq)
            else:
                vel_score = 1.0

            S[i, k] = (iou ** iou_weight) * (area_ratio ** area_weight) * \
                      (vel_score ** velocity_weight) * (pos_score ** pos_weight)

    # Solve LAP: maximize total score
    row_ind, col_ind = linear_sum_assignment(-S)

    next_frame = np.zeros_like(current_labels)
    matched_entry_indices: list[int] = []

    for i, k in zip(row_ind, col_ind):
        if k >= C:
            continue
        if S[i, k] <= unmatched_score:
            continue

        entry_idx, cand_id, *_ = flat_cands[k]
        next_frame[candidates[entry_idx] == cand_id] = cur_ids[i]
        matched_entry_indices.append(entry_idx)

    if not matched_entry_indices:
        return None, None

    winning_entry = Counter(matched_entry_indices).most_common(1)[0][0]
    return next_frame, winning_entry


def propagate_one_frame(
    hypotheses_h5: str | Path,
    tracked_h5: str | Path,
    t_current: int,
    iou_threshold: float = 0.3,
    max_dist_px: float = 50.0,
    velocity_sigma_px: float = 25.0,
    iou_weight: float = 1.0,
    area_weight: float = 1.0,
    velocity_weight: float = 1.0,
    pos_weight: float = 1.0,
    unmatched_score: float = 0.1,
) -> int | None:
    """Propagate tracking from t_current to t_current + 1.

    Searches all (p, z) combinations in the hypothesis database for t_next,
    matches each tracked nucleus to its best candidate via global linear
    assignment, then writes a relabeled next frame that preserves track IDs.

    Returns the winning p index, or None if no matches were found.
    """
    hypotheses_h5 = Path(hypotheses_h5)
    tracked_h5 = Path(tracked_h5)

    current_labels = read_tracked_frame(tracked_h5, t_current)  # (Y, X)

    n_p, _ = list_hypotheses(hypotheses_h5)
    if n_p == 0:
        return None

    t_next = t_current + 1

    # Derive per-nucleus velocity from the previous frame if available.
    predicted_centroids: dict[int, np.ndarray] | None = None
    if t_current >= 1:
        try:
            prev_labels = read_tracked_frame(tracked_h5, t_current - 1)
            _, prev_centroids = _label_stats(prev_labels)
            _, cur_centroids = _label_stats(current_labels)
            predicted_centroids = {
                lid: cur_centroids[lid] + (cur_centroids[lid] - prev_centroids[lid])
                for lid in cur_centroids
                if lid in prev_centroids
            }
        except KeyError:
            pass  # no previous frame — velocity scoring disabled for this step

    # Build flat list of (p, z, slice_2d) for every (hypothesis, z-plane)
    entries: list[tuple[int, int, np.ndarray]] = []
    for p in range(n_p):
        try:
            volume = read_hypothesis_labels(hypotheses_h5, t_next, p)  # (Z, Y, X)
        except KeyError:
            return None  # t_next not in hypothesis database
        for z in range(volume.shape[0]):
            entries.append((p, z, volume[z]))

    if not entries:
        return None

    candidates = [e[2] for e in entries]
    next_frame, winner_idx = find_best_hypothesis(
        current_labels, candidates, iou_threshold, max_dist_px,
        predicted_centroids=predicted_centroids,
        velocity_sigma_px=velocity_sigma_px,
        iou_weight=iou_weight,
        area_weight=area_weight,
        velocity_weight=velocity_weight,
        pos_weight=pos_weight,
        unmatched_score=unmatched_score,
    )
    if next_frame is None or winner_idx is None:
        return None

    p_win, _z_win, _slice = entries[winner_idx]
    write_tracked_frame(tracked_h5, t_next, next_frame)
    return p_win
