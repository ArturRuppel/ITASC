"""Greedy IoU propagator for nucleus tracking.

Finds the hypothesis parameter set whose 3D label volume best overlaps
the current tracked frame, then writes it as the next tracked frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cellflow.database.hypotheses import read_hypothesis_labels, list_hypotheses
from cellflow.database.tracked import read_tracked_frame, write_tracked_frame


def _centroids_2d(labels: np.ndarray) -> np.ndarray:
    """Return (N, 2) float32 centroid array from a 2D or max-Z-projected label image."""
    from skimage.measure import regionprops

    if labels.ndim == 3:
        labels = labels.max(axis=0)  # max-Z projection → (Y, X)

    props = regionprops(labels.astype(np.int32))
    if not props:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([p.centroid for p in props], dtype=np.float32)


def _mean_min_dist(src: np.ndarray, dst: np.ndarray) -> float:
    """Mean of each src centroid's minimum distance to any dst centroid."""
    if src.size == 0 or dst.size == 0:
        return float("inf")
    # (N_src, N_dst) pairwise squared distances
    diff = src[:, np.newaxis, :] - dst[np.newaxis, :, :]  # (N_src, N_dst, 2)
    sq = (diff * diff).sum(axis=-1)  # (N_src, N_dst)
    return float(np.sqrt(sq.min(axis=1)).mean())


def _binary_iou_3d(a: np.ndarray, b: np.ndarray) -> float:
    """Binary mask IoU over 3D volumes."""
    a_mask = a > 0
    b_mask = b > 0
    intersection = int((a_mask & b_mask).sum())
    union = int((a_mask | b_mask).sum())
    if union == 0:
        return 0.0
    return intersection / union


def find_best_hypothesis(
    current_labels: np.ndarray,
    candidates: list[np.ndarray],
    iou_threshold: float = 0.3,
    max_dist_px: float = 50.0,
) -> int | None:
    """Return the index of the best candidate, or None if none qualifies.

    Parameters
    ----------
    current_labels:
        (Z, Y, X) uint32 tracked label volume for the current frame.
    candidates:
        List of (Z, Y, X) uint32 hypothesis label volumes for the next frame.
    iou_threshold:
        Minimum binary 3D IoU to accept a candidate.
    max_dist_px:
        Candidates whose mean centroid displacement exceeds this (in pixels)
        are skipped before the IoU computation.
    """
    if not candidates:
        return None

    current_centroids = _centroids_2d(current_labels)

    best_idx: int | None = None
    best_iou: float = -1.0

    for idx, cand in enumerate(candidates):
        cand_centroids = _centroids_2d(cand)
        if _mean_min_dist(current_centroids, cand_centroids) > max_dist_px:
            continue

        iou = _binary_iou_3d(current_labels, cand)
        if iou >= iou_threshold and iou > best_iou:
            best_iou = iou
            best_idx = idx

    return best_idx


def propagate_one_frame(
    hypotheses_h5: str | Path,
    tracked_h5: str | Path,
    t_current: int,
    iou_threshold: float = 0.3,
    max_dist_px: float = 50.0,
) -> int | None:
    """Propagate tracking from t_current to t_current + 1.

    Reads the current tracked frame, loads all hypothesis candidates for the
    next timepoint, finds the best match, writes it as the next tracked frame.

    Returns the winning p index, or None if no suitable hypothesis was found.
    """
    hypotheses_h5 = Path(hypotheses_h5)
    tracked_h5 = Path(tracked_h5)

    current_labels = read_tracked_frame(tracked_h5, t_current)

    n_p, _ = list_hypotheses(hypotheses_h5)
    if n_p == 0:
        return None

    t_next = t_current + 1
    candidates: list[np.ndarray] = []
    for p in range(n_p):
        try:
            candidates.append(read_hypothesis_labels(hypotheses_h5, t_next, p))
        except KeyError:
            return None  # t_next not in hypothesis database

    winner = find_best_hypothesis(current_labels, candidates, iou_threshold, max_dist_px)
    if winner is None:
        return None

    write_tracked_frame(tracked_h5, t_next, candidates[winner])
    return winner
