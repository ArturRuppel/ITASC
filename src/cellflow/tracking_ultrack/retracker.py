"""Constrained greedy retracker using the shared linker similarity score.

Each unlocked target cell is matched to a reference cell by the same
``scoring.similarity_score`` (area ratio + centroid-corrected IoU - distance)
used by the linker and the Extend tool, gated only by centroid distance.

Matching is greedy best-first rather than a global minimum-cost assignment: a
target always takes its highest-scoring still-free reference. A global
``linear_sum_assignment`` minimises the *total* cost and will trade a cell's
obvious best match away to lower another cell's cost, which in practice handed
cells a far, worse reference; best-first never does that.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import regionprops

from cellflow.tracking_ultrack import scoring


def _label_props(
    labels: np.ndarray, keep_ids: list[int]
) -> dict[int, tuple[np.ndarray, float, np.ndarray]]:
    """Return {label_id: (centroid_yx, area, coords)} for labels in *keep_ids*.

    Labels outside *keep_ids* (locked / reserved) are masked out before
    ``regionprops`` so they cost no centroid, area, or coordinate extraction.
    """
    if not keep_ids:
        return {}
    keep = np.where(np.isin(labels, keep_ids), labels, 0)
    props: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}
    for region in regionprops(keep):
        props[int(region.label)] = (
            np.asarray(region.centroid, dtype=np.float32),
            float(region.area),
            region.coords.astype(np.float32),
        )
    return props


def retrack_frame_constrained(
    ref_labels: np.ndarray,
    target_labels: np.ndarray,
    locked_target_ids: set[int],
    max_dist_px: float = 50.0,
    reserved_ids: set[int] | None = None,
    *,
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    distance_weight: float = 0.05,
) -> np.ndarray:
    """Remap target IDs by linker similarity without changing locked targets.

    Target cells whose ID is in ``locked_target_ids`` keep their existing IDs.
    Those IDs, plus any ``reserved_ids``, are protected from assignment to
    unlocked target cells even if a matching reference cell exists.

    Unlocked target cells are matched to available reference cells greedily in
    descending ``similarity_score`` order: each target takes its best still-free
    reference. A pair is only eligible when its centroid distance is
    ``<= max_dist_px``; area ratio and centroid-corrected IoU contribute as soft
    score terms (no hard gate), matching the Extend tool's behaviour.
    """
    locked_target_ids = set(locked_target_ids)
    reserved_ids = set(reserved_ids or set())
    blocked_ids = locked_target_ids | reserved_ids

    result = np.zeros_like(target_labels)

    tgt_ids = {int(i) for i in np.unique(target_labels) if i != 0}
    for lid in locked_target_ids:
        if lid in tgt_ids:
            result[target_labels == lid] = lid

    unlocked_tgt_ids = [tid for tid in tgt_ids if tid not in locked_target_ids]
    if not unlocked_tgt_ids:
        return result

    ref_ids = {int(i) for i in np.unique(ref_labels) if i != 0}
    available_ref_ids = [rid for rid in ref_ids if rid not in blocked_ids]

    max_existing = max(
        int(ref_labels.max()) if ref_labels.size and ref_labels.max() > 0 else 0,
        int(target_labels.max()) if target_labels.size and target_labels.max() > 0 else 0,
    )

    def _assign_fresh(remap: dict[int, int]) -> dict[int, int]:
        next_id = max_existing + 1
        for tid in unlocked_tgt_ids:
            if tid not in remap:
                while next_id in blocked_ids:
                    next_id += 1
                remap[tid] = next_id
                next_id += 1
        return remap

    if not available_ref_ids:
        remap = _assign_fresh({})
        for tid, new_id in remap.items():
            result[target_labels == tid] = new_id
        return result

    tgt_props = _label_props(target_labels, unlocked_tgt_ids)
    ref_props = _label_props(ref_labels, available_ref_ids)

    tgt_centroids = np.array([tgt_props[t][0] for t in unlocked_tgt_ids])
    ref_centroids = np.array([ref_props[r][0] for r in available_ref_ids])
    dist = cdist(tgt_centroids, ref_centroids)
    gate = dist <= max_dist_px

    # Score every distance-eligible (target, reference) pair.
    n_tgt, _ = dist.shape
    scored: list[tuple[float, float, int, int]] = []
    for ti in range(n_tgt):
        t_centroid, t_area, t_coords = tgt_props[unlocked_tgt_ids[ti]]
        for ri in np.nonzero(gate[ti])[0]:
            r_centroid, r_area, r_coords = ref_props[available_ref_ids[ri]]
            area_ratio = min(t_area, r_area) / max(t_area, r_area)
            iou = scoring.centroid_corrected_iou_from_coords(
                r_coords, r_centroid, t_coords, t_centroid
            )
            score = scoring.similarity_score(
                area_ratio=area_ratio,
                centroid_corrected_iou=iou,
                distance=float(dist[ti, ri]),
                area_weight=area_weight,
                iou_weight=iou_weight,
                distance_weight=distance_weight,
            )
            scored.append((score, float(dist[ti, ri]), ti, int(ri)))

    # Greedy best-first: highest score wins, nearer centroid breaks ties. Each
    # target and reference is consumed once, so a target always lands on its
    # best still-available reference instead of being traded away by a global
    # min-cost solver.
    scored.sort(key=lambda item: (-item[0], item[1]))

    remap: dict[int, int] = {}
    used_refs: set[int] = set()
    for _score, _d, ti, ri in scored:
        tid = unlocked_tgt_ids[ti]
        if tid in remap or ri in used_refs:
            continue
        remap[tid] = available_ref_ids[ri]
        used_refs.add(ri)

    remap = _assign_fresh(remap)
    for tid, new_id in remap.items():
        result[target_labels == tid] = new_id

    return result
