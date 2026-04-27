"""Best-match propagator for nucleus tracking.

For each nucleus in the current tracked frame, gates all candidate nuclei from
all hypotheses for the next timepoint by distance and real spatial IoU, scores
the survivors, and picks the single best match greedily. No clustering, no LAP.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial import KDTree

from cellflow.database.hypotheses import read_hypothesis_labels, list_hypotheses


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


def _nucleus_pixels(labels: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return {label_id: (ys, xs)} pixel coordinate arrays for all non-zero labels."""
    ys_all, xs_all = np.nonzero(labels)
    vals = labels[ys_all, xs_all]
    result = {}
    for lid in np.unique(vals):
        mask = vals == lid
        result[int(lid)] = (ys_all[mask], xs_all[mask])
    return result


def _iou_direct(
    cur_ys: np.ndarray,
    cur_xs: np.ndarray,
    cur_area: int,
    cand_flat_idx: np.ndarray,
    cand_area: int,
    W: int,
) -> float:
    """Direct spatial IoU between the current nucleus and a candidate nucleus."""
    cur_flat = cur_ys * W + cur_xs
    inter = int(np.isin(cur_flat, cand_flat_idx).sum())
    union = cur_area + cand_area - inter
    return inter / union if union > 0 else 0.0


def find_best_hypothesis(
    current_labels: np.ndarray,
    candidates: list[np.ndarray],
    max_dist_px: float = 50.0,
    predicted_centroids: dict[int, np.ndarray] | None = None,
    velocity_sigma_px: float = 25.0,
    unmatched_score: float = 0.1,
    iou_weight: float = 1.0,
    area_weight: float = 1.0,
    velocity_weight: float = 1.0,
    pos_weight: float = 0.0,
    dedup_radius_px: float = 0.0,  # no longer used; kept for API compatibility
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Return (relabeled_next_frame, winning_entry_index) or (None, None).

    For each current nucleus, candidates are gated by distance from the predicted
    position (trajectory extrapolation), then scored. Falls back to current centroid
    when no prediction is available. No IoU threshold gate — per-nucleus greedy matching.
    """
    if not candidates:
        return None, None

    H, W = current_labels.shape
    cur_areas, cur_centroids = _label_stats(current_labels)
    cur_ids = sorted(cur_centroids.keys())
    if not cur_ids:
        return None, None

    cur_pixels = _nucleus_pixels(current_labels)

    # Build flat list of all (entry_idx, cand_id, centroid, area, flat_pixel_idx).
    flat_cands: list[tuple[int, int, np.ndarray, int, np.ndarray]] = []
    for entry_idx, cand in enumerate(candidates):
        c_areas, c_centroids = _label_stats(cand)
        c_pixels = _nucleus_pixels(cand)
        for cid, centroid in c_centroids.items():
            cys, cxs = c_pixels[cid]
            flat_idx = cys * W + cxs
            flat_cands.append((entry_idx, int(cid), centroid, int(c_areas[cid]), flat_idx))

    if not flat_cands:
        return None, None

    cand_centroids_arr = np.vstack([c[2] for c in flat_cands])
    tree = KDTree(cand_centroids_arr)
    two_sigma_sq = 2.0 * velocity_sigma_px ** 2

    next_frame = np.zeros_like(current_labels)
    matched_entry_indices: list[int] = []

    for cur_id in cur_ids:
        cur_centroid = cur_centroids[cur_id]
        cur_area = int(cur_areas[cur_id])
        cur_ys, cur_xs = cur_pixels[cur_id]
        pred_centroid = (predicted_centroids or {}).get(cur_id)

        gate_centroid = pred_centroid if pred_centroid is not None else cur_centroid
        nearby_ks = tree.query_ball_point(gate_centroid, max_dist_px)
        if not nearby_ks:
            continue

        best_score = -1.0
        best_k = -1

        for k in nearby_ks:
            entry_idx, cand_id, cand_centroid, cand_area, cand_flat_idx = flat_cands[k]

            iou = _iou_direct(cur_ys, cur_xs, cur_area, cand_flat_idx, cand_area, W)

            area_ratio = min(cur_area, cand_area) / max(cur_area, cand_area)

            pos_d2 = float(np.sum((cur_centroid - cand_centroid) ** 2))
            pos_score = float(np.exp(-pos_d2 / two_sigma_sq))

            if pred_centroid is not None:
                vel_d2 = float(np.sum((pred_centroid - cand_centroid) ** 2))
                vel_score = float(np.exp(-vel_d2 / two_sigma_sq))
            else:
                vel_score = 1.0

            score = (iou ** iou_weight) * (area_ratio ** area_weight) * \
                    (vel_score ** velocity_weight) * (pos_score ** pos_weight)

            if score > best_score:
                best_score = score
                best_k = k

        if best_k >= 0:
            entry_idx, cand_id, cand_centroid, cand_area, cand_flat_idx = flat_cands[best_k]
            next_frame[candidates[entry_idx] == cand_id] = cur_id
            matched_entry_indices.append(entry_idx)

    if not matched_entry_indices:
        return None, None

    winning_entry = Counter(matched_entry_indices).most_common(1)[0][0]
    return next_frame, winning_entry


def propagate_one_frame(
    hypotheses_h5: str | Path,
    current_labels: np.ndarray,
    t_next: int,
    prev_labels: np.ndarray | None = None,
    max_dist_px: float = 50.0,
    velocity_sigma_px: float = 25.0,
    iou_weight: float = 1.0,
    area_weight: float = 1.0,
    velocity_weight: float = 1.0,
    pos_weight: float = 0.0,
    unmatched_score: float = 0.1,
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Propagate tracking to t_next using current_labels as the source frame.

    Searches all hypotheses in the hypothesis database for t_next, matches each
    tracked nucleus to its best candidate via greedy per-nucleus scoring.

    Returns (relabeled_next_frame, winning_p) or (None, None) if no matches.
    Does not read from or write to any tracked labels file.
    """
    hypotheses_h5 = Path(hypotheses_h5)

    n_p, params_by_p = list_hypotheses(hypotheses_h5)
    if n_p == 0:
        return None, None

    # Derive per-nucleus velocity from the previous frame if provided.
    predicted_centroids: dict[int, np.ndarray] | None = None
    if prev_labels is not None:
        try:
            _, prev_centroids = _label_stats(prev_labels)
            _, cur_centroids = _label_stats(current_labels)
            predicted_centroids = {
                lid: cur_centroids[lid] + (cur_centroids[lid] - prev_centroids[lid])
                for lid in cur_centroids
                if lid in prev_centroids
            }
        except Exception:
            pass

    entries: list[tuple[int, int, np.ndarray]] = []
    for p in params_by_p.keys():
        try:
            volume = read_hypothesis_labels(hypotheses_h5, t_next, p)  # (Z, Y, X) with Z=1
        except (KeyError, ValueError):
            continue
        for z in range(volume.shape[0]):
            entries.append((p, z, volume[z]))

    if not entries:
        return None, None

    candidates = [e[2] for e in entries]
    next_frame, winner_idx = find_best_hypothesis(
        current_labels, candidates, max_dist_px,
        predicted_centroids=predicted_centroids,
        velocity_sigma_px=velocity_sigma_px,
        iou_weight=iou_weight,
        area_weight=area_weight,
        velocity_weight=velocity_weight,
        pos_weight=pos_weight,
        unmatched_score=unmatched_score,
    )
    if next_frame is None or winner_idx is None:
        return None, None

    p_win, _z_win, _slice = entries[winner_idx]
    return next_frame, p_win
