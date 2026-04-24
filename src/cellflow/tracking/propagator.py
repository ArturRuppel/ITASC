"""Greedy per-label IoU propagator for nucleus tracking.

For each nucleus in the current tracked frame, finds the best matching
candidate nucleus across all (hypothesis, z-slice) combinations for the
next timepoint, then writes a relabeled next frame that preserves track IDs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cellflow.database.hypotheses import read_hypothesis_labels, list_hypotheses
from cellflow.database.tracked import read_tracked_frame, write_tracked_frame


def _label_masks(labels: np.ndarray) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        masks[int(label_id)] = labels == label_id
    return masks


def _centroid(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    return coords.mean(axis=0) if len(coords) else np.zeros(2)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union > 0 else 0.0


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

    current_masks = _label_masks(current_labels)
    if not current_masks:
        return None, None

    # Pre-build candidate mask dicts: cand_masks[entry_idx] = {label_id: mask}
    cand_masks_per_entry: list[dict[int, np.ndarray]] = [
        _label_masks(c) for c in candidates
    ]
    # Pre-compute centroids for all candidate labels
    cand_centroids: list[dict[int, np.ndarray]] = [
        {lid: _centroid(mask) for lid, mask in entry.items()}
        for entry in cand_masks_per_entry
    ]

    assigned: set[tuple[int, int]] = set()  # (entry_idx, cand_label_id)
    next_frame = np.zeros_like(current_labels)
    matched_entry_indices: list[int] = []

    for current_id in sorted(current_masks):
        cur_mask = current_masks[current_id]
        cur_centroid = _centroid(cur_mask)

        best_score = 0.0
        best_key: tuple[int, int] | None = None
        best_cand_mask: np.ndarray | None = None

        for entry_idx, entry in enumerate(cand_masks_per_entry):
            for cand_id, cand_mask in entry.items():
                key = (entry_idx, cand_id)
                if key in assigned:
                    continue

                dist = float(np.linalg.norm(cur_centroid - cand_centroids[entry_idx][cand_id]))
                if dist > max_dist_px:
                    continue

                score = _iou(cur_mask, cand_mask)
                if score >= iou_threshold and score > best_score:
                    best_score = score
                    best_key = key
                    best_cand_mask = cand_mask

        if best_key is not None and best_cand_mask is not None:
            assigned.add(best_key)
            next_frame[best_cand_mask] = current_id
            matched_entry_indices.append(best_key[0])

    if not matched_entry_indices:
        return None, None

    # Return the entry index that won the most matches
    from collections import Counter
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
