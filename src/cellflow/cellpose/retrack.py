"""DB-free greedy retracker for the standalone segment + track corrector.

This mirrors the constrained retracker in :mod:`cellflow.tracking_ultrack`
(which ships in a separate ``cellflow-tracking`` distribution the standalone
``cellflow-cellpose`` tool does *not* depend on), reduced to the validation-free
case the standalone needs: starting from the current frame, every later frame's
labels are re-linked to the already-retracked neighbour toward the start frame
by the same additive similarity score (area ratio + centroid-corrected IoU -
distance), matched greedily best-first. Unmatched target cells receive fresh
ids. There is no locked/validated/reserved concept here — the standalone has no
validation store — so the algorithm is purely the geometric re-linking.

Everything here is Qt-free and depends only on numpy/scipy/skimage, so it ships
inside the cellpose distro tree and is unit-testable without a viewer.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import regionprops


# ---------------------------------------------------------------------------
# Similarity scoring (ported from cellflow.tracking_ultrack.scoring so the
# standalone stays self-contained — same formula, so the UX matches the app).
# ---------------------------------------------------------------------------
def _rasterize(coords: np.ndarray, mins: np.ndarray, shape: tuple) -> np.ndarray:
    canvas = np.zeros(shape, dtype=bool)
    idx = np.rint(coords - mins).astype(int)
    valid = np.ones(len(idx), dtype=bool)
    for axis, size in enumerate(shape):
        valid &= (idx[:, axis] >= 0) & (idx[:, axis] < size)
    idx = idx[valid]
    if idx.size:
        canvas[tuple(idx.T)] = True
    return canvas


def centroid_corrected_iou_from_coords(
    src_coords: np.ndarray,
    src_centroid: np.ndarray,
    target_coords: np.ndarray,
    target_centroid: np.ndarray,
) -> float:
    """IoU after shifting the target so its centroid matches the source's."""
    if len(src_coords) == 0 or len(target_coords) == 0:
        return 0.0
    shifted = target_coords + (src_centroid - target_centroid)
    all_coords = np.vstack([src_coords, shifted])
    mins = np.floor(all_coords.min(axis=0)).astype(int) - 1
    maxs = np.ceil(all_coords.max(axis=0)).astype(int) + 1
    shape = tuple((maxs - mins + 1).tolist())
    if any(dim <= 0 for dim in shape):
        return 0.0
    mins_f = mins.astype(np.float32)
    src_canvas = _rasterize(src_coords, mins_f, shape)
    tgt_canvas = _rasterize(shifted, mins_f, shape)
    union = np.logical_or(src_canvas, tgt_canvas).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(src_canvas, tgt_canvas).sum() / union)


def similarity_score(
    *,
    area_ratio: float,
    centroid_corrected_iou: float,
    distance: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
) -> float:
    """Additive similarity score (higher = more preferred).

    Shape terms are positive rewards in ``[0, 1]``; distance is a raw pixel
    penalty, so the score can go negative for far candidates.
    """
    return (
        area_weight * area_ratio
        + iou_weight * centroid_corrected_iou
        - distance_weight * distance
    )


# ---------------------------------------------------------------------------
# Frame / stack retracking
# ---------------------------------------------------------------------------
def _label_props(
    labels: np.ndarray,
) -> dict[int, tuple[np.ndarray, float, np.ndarray]]:
    """Return ``{label_id: (centroid_yx, area, coords)}`` for every non-zero id."""
    props: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}
    for region in regionprops(labels):
        props[int(region.label)] = (
            np.asarray(region.centroid, dtype=np.float32),
            float(region.area),
            region.coords.astype(np.float32),
        )
    return props


def retrack_frame(
    ref_labels: np.ndarray,
    target_labels: np.ndarray,
    *,
    max_dist_px: float = 50.0,
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    distance_weight: float = 0.05,
) -> np.ndarray:
    """Remap every target id to its best-matching reference id by similarity.

    Each target cell takes its highest-scoring still-free reference cell
    (greedy best-first), among reference cells whose centroid is within
    ``max_dist_px``. Targets with no eligible reference get a fresh id above all
    existing ids, so no two cells ever collide. Returns a relabelled copy of
    ``target_labels`` (input arrays are not mutated).
    """
    result = np.zeros_like(target_labels)

    tgt_ids = [int(i) for i in np.unique(target_labels) if i != 0]
    if not tgt_ids:
        return result

    ref_ids = [int(i) for i in np.unique(ref_labels) if i != 0]
    max_existing = max(
        int(ref_labels.max()) if ref_labels.size and ref_labels.max() > 0 else 0,
        int(target_labels.max()) if target_labels.size and target_labels.max() > 0 else 0,
    )

    def _assign_fresh(remap: dict[int, int]) -> dict[int, int]:
        next_id = max_existing + 1
        for tid in tgt_ids:
            if tid not in remap:
                remap[tid] = next_id
                next_id += 1
        return remap

    if not ref_ids:
        for tid, new_id in _assign_fresh({}).items():
            result[target_labels == tid] = new_id
        return result

    tgt_props = _label_props(np.where(np.isin(target_labels, tgt_ids), target_labels, 0))
    ref_props = _label_props(np.where(np.isin(ref_labels, ref_ids), ref_labels, 0))

    tgt_centroids = np.array([tgt_props[t][0] for t in tgt_ids])
    ref_centroids = np.array([ref_props[r][0] for r in ref_ids])
    dist = cdist(tgt_centroids, ref_centroids)
    gate = dist <= max_dist_px

    # Score every distance-eligible (target, reference) pair.
    scored: list[tuple[float, float, int, int]] = []
    for ti in range(dist.shape[0]):
        _t_centroid, t_area, t_coords = tgt_props[tgt_ids[ti]]
        for ri in np.nonzero(gate[ti])[0]:
            r_centroid, r_area, r_coords = ref_props[ref_ids[ri]]
            area_ratio = min(t_area, r_area) / max(t_area, r_area)
            iou = centroid_corrected_iou_from_coords(
                r_coords, r_centroid, t_coords, tgt_props[tgt_ids[ti]][0]
            )
            score = similarity_score(
                area_ratio=area_ratio,
                centroid_corrected_iou=iou,
                distance=float(dist[ti, ri]),
                area_weight=area_weight,
                iou_weight=iou_weight,
                distance_weight=distance_weight,
            )
            scored.append((score, float(dist[ti, ri]), ti, int(ri)))

    # Greedy best-first: highest score wins, nearer centroid breaks ties. Each
    # target and reference is consumed once.
    scored.sort(key=lambda item: (-item[0], item[1]))

    remap: dict[int, int] = {}
    used_refs: set[int] = set()
    for _score, _d, ti, ri in scored:
        tid = tgt_ids[ti]
        if tid in remap or ri in used_refs:
            continue
        remap[tid] = ref_ids[ri]
        used_refs.add(ri)

    for tid, new_id in _assign_fresh(remap).items():
        result[target_labels == tid] = new_id
    return result


def retrack_stack(
    stack: np.ndarray,
    *,
    start_frame: int,
    direction: Literal["forward", "backward"],
    max_dist_px: float = 50.0,
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    distance_weight: float = 0.05,
) -> np.ndarray:
    """Retrack a time-first ``(T, Y, X)`` stack outward from ``start_frame``.

    The start frame is kept as the anchor; each later frame in ``direction`` is
    re-linked (:func:`retrack_frame`) to the already-retracked neighbour toward
    the start frame, so corrected ids propagate. Returns a new stack; the input
    is not mutated.
    """
    stack = np.asarray(stack)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError("retrack_stack needs a 3D time-first stack with >= 2 frames.")

    out = stack.copy()
    if direction == "forward":
        frame_range = range(start_frame + 1, out.shape[0])
        neighbour = lambda t: out[t - 1]
    elif direction == "backward":
        frame_range = range(start_frame - 1, -1, -1)
        neighbour = lambda t: out[t + 1]
    else:
        raise ValueError(f"unknown retrack direction: {direction!r}")

    for t in frame_range:
        out[t] = retrack_frame(
            neighbour(t),
            out[t],
            max_dist_px=max_dist_px,
            area_weight=area_weight,
            iou_weight=iou_weight,
            distance_weight=distance_weight,
        )
    return out
