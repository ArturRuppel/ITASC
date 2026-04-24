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
    centroids = {int(lid): np.array(com) for lid, com in zip(ids, coms)}
    return areas, centroids


def _overlap_matrix(current_labels: np.ndarray, cand_labels: np.ndarray) -> np.ndarray:
    """Return intersection count matrix[cur_id, cand_id] in one bincount pass."""
    max_cur = int(current_labels.max())
    max_cand = int(cand_labels.max())
    if max_cur == 0 or max_cand == 0:
        return np.zeros((max_cur + 1, max_cand + 1), dtype=np.int64)
    stride = max_cand + 1
    combined = (
        current_labels.ravel().astype(np.int64) * stride
        + cand_labels.ravel().astype(np.int64)
    )
    counts = np.bincount(combined, minlength=(max_cur + 1) * stride)
    return counts.reshape(max_cur + 1, stride)


def find_best_hypothesis(
    current_labels: np.ndarray,
    candidates: list[np.ndarray],
    iou_threshold: float = 0.3,
    max_dist_px: float = 50.0,
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
        Minimum per-label IoU to accept a match.
    max_dist_px:
        Candidate nuclei whose centroid is farther than this are skipped.
    """
    if not candidates:
        return None, None

    cur_areas, cur_centroids = _label_stats(current_labels)
    cur_ids = sorted(cur_centroids.keys())
    if not cur_ids:
        return None, None

    # Pre-compute per-candidate stats and overlap matrices (all vectorized)
    cand_data: list[tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]] = []
    for cand in candidates:
        c_areas, c_centroids = _label_stats(cand)
        overlap = _overlap_matrix(current_labels, cand)
        cand_data.append((c_areas, c_centroids, overlap))

    # Build a flat index of all candidate centroids for KDTree radius query.
    # This avoids checking max_dist_px via np.linalg.norm in a Python loop.
    all_keys: list[tuple[int, int]] = []  # (entry_idx, cand_id)
    all_cand_centroids: list[np.ndarray] = []
    for entry_idx, (_, c_centroids, _) in enumerate(cand_data):
        for cand_id, cand_centroid in c_centroids.items():
            all_keys.append((entry_idx, cand_id))
            all_cand_centroids.append(cand_centroid)

    if not all_keys:
        return None, None

    cand_tree = KDTree(np.array(all_cand_centroids))

    assigned: set[tuple[int, int]] = set()  # (entry_idx, cand_label_id)
    next_frame = np.zeros_like(current_labels)
    matched_entry_indices: list[int] = []

    for current_id in cur_ids:
        cur_centroid = cur_centroids[current_id]
        cur_area = int(cur_areas[current_id])

        best_score = 0.0
        best_key: tuple[int, int] | None = None

        for idx in cand_tree.query_ball_point(cur_centroid, max_dist_px):
            entry_idx, cand_id = all_keys[idx]
            key = (entry_idx, cand_id)
            if key in assigned:
                continue

            c_areas, _, overlap = cand_data[entry_idx]
            if current_id >= overlap.shape[0] or cand_id >= overlap.shape[1]:
                continue

            inter = int(overlap[current_id, cand_id])
            union = cur_area + int(c_areas[cand_id]) - inter
            score = inter / union if union > 0 else 0.0

            if score >= iou_threshold and score > best_score:
                best_score = score
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
) -> int | None:
    """Propagate tracking from t_current to t_current + 1.

    Searches all (p, z) combinations in the hypothesis database for t_next,
    matches each tracked nucleus to its best candidate by per-label IoU,
    and writes a relabeled next frame that preserves track IDs.

    Returns the winning p index, or None if no matches were found.
    """
    hypotheses_h5 = Path(hypotheses_h5)
    tracked_h5 = Path(tracked_h5)

    current_labels = read_tracked_frame(tracked_h5, t_current)  # (Y, X)

    n_p, _ = list_hypotheses(hypotheses_h5)
    if n_p == 0:
        return None

    t_next = t_current + 1

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
        current_labels, candidates, iou_threshold, max_dist_px
    )
    if next_frame is None or winner_idx is None:
        return None

    p_win, _z_win, _slice = entries[winner_idx]
    write_tracked_frame(tracked_h5, t_next, next_frame)
    return p_win
