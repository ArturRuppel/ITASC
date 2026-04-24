"""Greedy per-label IoU propagator for nucleus tracking.

For each nucleus in the current tracked frame, finds the best matching
candidate nucleus across all (hypothesis, z-slice) combinations for the
next timepoint, then writes a relabeled next frame that preserves track IDs.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial import KDTree

from cellflow.database.hypotheses import read_hypothesis_labels, list_hypotheses
from cellflow.database.tracked import read_tracked_frame, write_tracked_frame


def _label_stats(labels: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return (areas, centroids) without building per-label boolean masks.

    areas[label_id] = pixel count (via bincount, one pass).
    centroids: scipy batches all labels in a single labeled_comprehension pass.
    """
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
    """Return per-label pixel coordinate sets relative to each label's centroid.

    Precomputing centroid-relative coords once lets us compute centroid-
    corrected IoU between any two labels as a plain set intersection, with no
    per-pair array shifting.
    """
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


def find_best_hypothesis(
    current_labels: np.ndarray,
    candidates: list[np.ndarray],
    iou_threshold: float = 0.3,
    max_dist_px: float = 50.0,
    predicted_centroids: dict[int, np.ndarray] | None = None,
    velocity_sigma_px: float = 25.0,
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Return (relabeled_next_frame, winning_p_index) or (None, None).

    For each nucleus in current_labels, greedily assigns it to the best
    matching nucleus across all candidate slices, preserving track IDs.

    Parameters
    ----------
    current_labels:
        (Y, X) uint32 tracked label image for the current frame.
    candidates:
        List of (Y, X) uint32 label images — one per (p, z) combination.
    iou_threshold:
        Minimum centroid-corrected IoU to accept a match (hard gate).
    max_dist_px:
        Candidate nuclei whose centroid is farther than this are skipped.
    predicted_centroids:
        Optional dict mapping nucleus ID → predicted centroid for the next
        frame (derived from velocity history). When provided, a Gaussian
        velocity score exp(-d²/(2σ²)) weights the composite score.
    velocity_sigma_px:
        Standard deviation (pixels) for the velocity Gaussian.
    """
    if not candidates:
        return None, None

    cur_areas, cur_centroids = _label_stats(current_labels)
    cur_ids = sorted(cur_centroids.keys())
    if not cur_ids:
        return None, None

    cur_rel_pixels = _label_rel_pixels(current_labels, cur_centroids)

    # Pre-compute per-candidate stats and centroid-relative pixel sets.
    cand_data: list[tuple[np.ndarray, dict[int, np.ndarray], dict[int, frozenset]]] = []
    for cand in candidates:
        c_areas, c_centroids = _label_stats(cand)
        c_rel_pixels = _label_rel_pixels(cand, c_centroids)
        cand_data.append((c_areas, c_centroids, c_rel_pixels))

    # Build a flat index of all candidate centroids for KDTree radius query.
    all_keys: list[tuple[int, int]] = []  # (entry_idx, cand_id)
    all_cand_centroids: list[np.ndarray] = []
    for entry_idx, (_, c_centroids, _) in enumerate(cand_data):
        for cand_id, cand_centroid in c_centroids.items():
            all_keys.append((entry_idx, cand_id))
            all_cand_centroids.append(cand_centroid)

    if not all_keys:
        return None, None

    cand_tree = KDTree(np.array(all_cand_centroids))
    two_sigma_sq = 2.0 * velocity_sigma_px ** 2

    assigned: set[tuple[int, int]] = set()  # (entry_idx, cand_label_id)
    next_frame = np.zeros_like(current_labels)
    matched_entry_indices: list[int] = []

    for current_id in cur_ids:
        cur_centroid = cur_centroids[current_id]
        cur_area = int(cur_areas[current_id])
        cur_rel = cur_rel_pixels[current_id]
        pred_centroid = predicted_centroids.get(current_id) if predicted_centroids else None

        best_score = 0.0
        best_key: tuple[int, int] | None = None

        for idx in cand_tree.query_ball_point(cur_centroid, max_dist_px):
            entry_idx, cand_id = all_keys[idx]
            key = (entry_idx, cand_id)
            if key in assigned:
                continue

            c_areas, c_centroids, c_rel_pixels = cand_data[entry_idx]
            cand_area = int(c_areas[cand_id])

            inter = len(cur_rel & c_rel_pixels[cand_id])
            union = cur_area + cand_area - inter
            iou_cc = inter / union if union > 0 else 0.0

            if iou_cc < iou_threshold:
                continue

            area_ratio = min(cur_area, cand_area) / max(cur_area, cand_area)

            if pred_centroid is not None:
                d2 = float(np.sum((pred_centroid - c_centroids[cand_id]) ** 2))
                vel_score = np.exp(-d2 / two_sigma_sq)
            else:
                vel_score = 1.0

            composite = iou_cc * area_ratio * vel_score

            if composite > best_score:
                best_score = composite
                best_key = key

        if best_key is not None:
            assigned.add(best_key)
            entry_idx, cand_id = best_key
            next_frame[candidates[entry_idx] == cand_id] = current_id
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
) -> int | None:
    """Propagate tracking from t_current to t_current + 1.

    Searches all (p, z) combinations in the hypothesis database for t_next,
    matches each tracked nucleus to its best candidate by per-label IoU,
    area ratio, and velocity consistency, then writes a relabeled next frame
    that preserves track IDs.

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
    )
    if next_frame is None or winner_idx is None:
        return None

    p_win, _z_win, _slice = entries[winner_idx]
    write_tracked_frame(tracked_h5, t_next, next_frame)
    return p_win
