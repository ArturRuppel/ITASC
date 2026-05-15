"""Shared Ultrack node geometry helpers."""
from __future__ import annotations

import pickle

import numpy as np


def node_bbox_and_mask(node_id: int, node) -> tuple[tuple[int, int, int, int], np.ndarray]:
    if isinstance(node, (bytes, memoryview)):
        node = pickle.loads(bytes(node))

    bbox = np.asarray(node.bbox)
    ndim = len(bbox) // 2
    if ndim == 3:
        y0, x0 = int(bbox[1]), int(bbox[2])
        y1, x1 = int(bbox[4]), int(bbox[5])
    elif ndim == 2:
        y0, x0 = int(bbox[0]), int(bbox[1])
        y1, x1 = int(bbox[2]), int(bbox[3])
    else:
        raise ValueError(f"Unexpected bbox for node {node_id}: {bbox}")

    mask = np.asarray(node.mask, dtype=bool)
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] == 1 else mask.any(axis=0)
    elif mask.ndim != 2:
        raise ValueError(f"Unexpected mask for node {node_id}: shape {mask.shape}")

    return (y0, x0, y1, x1), np.ascontiguousarray(mask, dtype=bool)


def intersects(
    lhs_bbox: tuple[int, int, int, int],
    lhs_mask: np.ndarray,
    rhs_bbox: tuple[int, int, int, int],
    rhs_mask: np.ndarray,
) -> bool:
    ly0, lx0, ly1, lx1 = lhs_bbox
    ry0, rx0, ry1, rx1 = rhs_bbox
    oy0, ox0 = max(ly0, ry0), max(lx0, rx0)
    oy1, ox1 = min(ly1, ry1), min(lx1, rx1)
    if oy0 >= oy1 or ox0 >= ox1:
        return False

    lhs_crop = lhs_mask[oy0 - ly0: oy1 - ly0, ox0 - lx0: ox1 - lx0]
    rhs_crop = rhs_mask[oy0 - ry0: oy1 - ry0, ox0 - rx0: ox1 - rx0]
    return bool(np.logical_and(lhs_crop, rhs_crop).any())


def raw_iou(
    lhs_bbox: tuple[int, int, int, int],
    lhs_mask: np.ndarray,
    rhs_bbox: tuple[int, int, int, int],
    rhs_mask: np.ndarray,
) -> float:
    ly0, lx0, ly1, lx1 = lhs_bbox
    ry0, rx0, ry1, rx1 = rhs_bbox
    oy0, ox0 = max(ly0, ry0), max(lx0, rx0)
    oy1, ox1 = min(ly1, ry1), min(lx1, rx1)
    intersection = 0
    if oy0 < oy1 and ox0 < ox1:
        lhs_crop = lhs_mask[oy0 - ly0: oy1 - ly0, ox0 - lx0: ox1 - lx0]
        rhs_crop = rhs_mask[oy0 - ry0: oy1 - ry0, ox0 - rx0: ox1 - rx0]
        intersection = int(np.logical_and(lhs_crop, rhs_crop).sum())

    union = int(lhs_mask.sum()) + int(rhs_mask.sum()) - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def node_pickle_ndim(node) -> int:
    if isinstance(node, (bytes, memoryview)):
        node = pickle.loads(bytes(node))
    bbox = np.asarray(node.bbox)
    return len(bbox) // 2


def make_node_pickle(
    t: int,
    mask_2d: np.ndarray,
    bbox: np.ndarray,
    node_id: int,
    *,
    ndim: int = 2,
) -> bytes:
    from ultrack.core.segmentation.node import Node

    min_y, min_x, max_y, max_x = bbox
    if ndim == 3:
        bbox_arr = np.array(
            [0, int(min_y), int(min_x), 1, int(max_y), int(max_x)],
            dtype=np.int64,
        )
        mask = np.asarray(mask_2d, dtype=bool)[np.newaxis]
    else:
        bbox_arr = np.array(
            [int(min_y), int(min_x), int(max_y), int(max_x)],
            dtype=np.int64,
        )
        mask = np.asarray(mask_2d, dtype=bool)
    node = Node.from_mask(time=t, mask=mask, bbox=bbox_arr, node_id=node_id)
    return pickle.dumps(node)
