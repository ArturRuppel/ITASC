"""Shared similarity scoring used by both the linker and the greedy retracker."""
from __future__ import annotations

import numpy as np


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
    """IoU after shifting target so its centroid matches source's."""
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


def centroid_corrected_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Convenience wrapper for full-frame boolean masks."""
    coords_a = np.argwhere(mask_a).astype(np.float32)
    coords_b = np.argwhere(mask_b).astype(np.float32)
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0.0
    centroid_a = coords_a.mean(axis=0)
    centroid_b = coords_b.mean(axis=0)
    return centroid_corrected_iou_from_coords(coords_a, centroid_a, coords_b, centroid_b)


def similarity_score(
    *,
    area_ratio: float,
    centroid_corrected_iou: float,
    distance: float,
    d_max: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
) -> float:
    """Additive similarity score (higher = more preferred)."""
    distance_score = 1.0 if d_max <= 0 else max(0.0, 1.0 - distance / d_max)
    return (
        area_weight * area_ratio
        + iou_weight * centroid_corrected_iou
        + distance_weight * distance_score
    )
