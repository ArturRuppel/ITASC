"""Best-match propagator for nucleus tracking.

For each nucleus in the current tracked frame, gates all candidate nuclei from
all hypotheses for the next timepoint by distance, scores the survivors using
additive shape-quality metrics, and picks the single best match greedily.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial import KDTree, ConvexHull

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


def _circularity(ys: np.ndarray, xs: np.ndarray, area: int) -> float:
    """4π·area / perimeter² using 4-connected boundary edge count."""
    if area == 0:
        return 0.0
    y_min, x_min = int(ys.min()), int(xs.min())
    h = int(ys.max()) - y_min + 3
    w = int(xs.max()) - x_min + 3
    img = np.zeros((h, w), dtype=bool)
    img[ys - y_min + 1, xs - x_min + 1] = True
    perimeter = (
        int(np.sum(img[:-1, :] != img[1:, :]))
        + int(np.sum(img[:, :-1] != img[:, 1:]))
    )
    if perimeter == 0:
        return 1.0
    return min(1.0, (4.0 * np.pi * area) / (perimeter ** 2))


def _solidity(ys: np.ndarray, xs: np.ndarray, area: int) -> float:
    """area / convex_hull_area; penalises holes and concave indentations."""
    pts = np.column_stack([ys, xs])
    if len(pts) < 3:
        return 1.0
    try:
        hull = ConvexHull(pts)
        return min(1.0, area / hull.volume)  # hull.volume == area in 2-D
    except Exception:
        return 1.0


def _iou_position_corrected(
    cur_ys: np.ndarray, cur_xs: np.ndarray, cur_area: int,
    cand_ys: np.ndarray, cand_xs: np.ndarray, cand_area: int,
) -> float:
    """IoU after aligning candidate centroid to current centroid (pure shape similarity)."""
    dy = int(round(float(cur_ys.mean()) - float(cand_ys.mean())))
    dx = int(round(float(cur_xs.mean()) - float(cand_xs.mean())))
    cur_set = set(zip(cur_ys.tolist(), cur_xs.tolist()))
    cand_set = set(zip((cand_ys + dy).tolist(), (cand_xs + dx).tolist()))
    inter = len(cur_set & cand_set)
    union = cur_area + cand_area - inter
    return inter / union if union > 0 else 0.0


def find_best_hypothesis(
    current_labels: np.ndarray,
    candidates: list[np.ndarray],
    max_dist_px: float = 50.0,
    predicted_centroids: dict[int, np.ndarray] | None = None,
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    circularity_weight: float = 1.0,
    solidity_weight: float = 1.0,
    dedup_radius_px: float = 0.0,  # no longer used; kept for API compatibility
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Return (relabeled_next_frame, winning_entry_index) or (None, None).

    Candidates are gated by distance from the predicted position (or current
    centroid when no prediction is available), then ranked by an additive score:

        score = area_weight * area_ratio
              + iou_weight  * position_corrected_iou
              + circularity_weight * circularity_ratio
              + solidity_weight    * solidity
    """
    if not candidates:
        return None, None

    H, W = current_labels.shape
    cur_areas, cur_centroids = _label_stats(current_labels)
    cur_ids = sorted(cur_centroids.keys())
    if not cur_ids:
        return None, None

    cur_pixels = _nucleus_pixels(current_labels)

    # Build flat list of all (entry_idx, cand_id, centroid, area, flat_pixel_idx, ys, xs).
    flat_cands: list[tuple[int, int, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for entry_idx, cand in enumerate(candidates):
        c_areas, c_centroids = _label_stats(cand)
        c_pixels = _nucleus_pixels(cand)
        for cid, centroid in c_centroids.items():
            cys, cxs = c_pixels[cid]
            flat_idx = cys * W + cxs
            flat_cands.append((entry_idx, int(cid), centroid, int(c_areas[cid]), flat_idx, cys, cxs))

    if not flat_cands:
        return None, None

    cand_centroids_arr = np.vstack([c[2] for c in flat_cands])
    tree = KDTree(cand_centroids_arr)

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

        cur_circ = _circularity(cur_ys, cur_xs, cur_area)

        best_score = -1.0
        best_k = -1

        for k in nearby_ks:
            entry_idx, cand_id, cand_centroid, cand_area, cand_flat_idx, cand_ys, cand_xs = flat_cands[k]

            area_ratio = (
                min(cur_area, cand_area) / max(cur_area, cand_area)
                if max(cur_area, cand_area) > 0 else 1.0
            )
            pos_iou = _iou_position_corrected(cur_ys, cur_xs, cur_area, cand_ys, cand_xs, cand_area)
            cand_circ = _circularity(cand_ys, cand_xs, cand_area)
            circ_ratio = (
                min(cur_circ, cand_circ) / max(cur_circ, cand_circ)
                if max(cur_circ, cand_circ) > 0 else 1.0
            )
            sol = _solidity(cand_ys, cand_xs, cand_area)

            score = (
                area_weight        * area_ratio
                + iou_weight       * pos_iou
                + circularity_weight * circ_ratio
                + solidity_weight  * sol
            )

            if score > best_score:
                best_score = score
                best_k = k

        if best_k >= 0:
            entry_idx, cand_id, cand_centroid, cand_area, cand_flat_idx, cand_ys, cand_xs = flat_cands[best_k]
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
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    circularity_weight: float = 1.0,
    solidity_weight: float = 1.0,
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
        area_weight=area_weight,
        iou_weight=iou_weight,
        circularity_weight=circularity_weight,
        solidity_weight=solidity_weight,
    )
    if next_frame is None or winner_idx is None:
        return None, None

    p_win, _z_win, _slice = entries[winner_idx]
    return next_frame, p_win
