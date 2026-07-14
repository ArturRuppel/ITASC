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


def intersection_area(
    lhs_bbox: tuple[int, int, int, int],
    lhs_mask: np.ndarray,
    rhs_bbox: tuple[int, int, int, int],
    rhs_mask: np.ndarray,
) -> int:
    """Number of pixels where the two cropped masks overlap."""
    ly0, lx0, ly1, lx1 = lhs_bbox
    ry0, rx0, ry1, rx1 = rhs_bbox
    oy0, ox0 = max(ly0, ry0), max(lx0, rx0)
    oy1, ox1 = min(ly1, ry1), min(lx1, rx1)
    if oy0 >= oy1 or ox0 >= ox1:
        return 0
    lhs_crop = lhs_mask[oy0 - ly0: oy1 - ly0, ox0 - lx0: ox1 - lx0]
    rhs_crop = rhs_mask[oy0 - ry0: oy1 - ry0, ox0 - rx0: ox1 - rx0]
    return int(np.logical_and(lhs_crop, rhs_crop).sum())


def raw_iou(
    lhs_bbox: tuple[int, int, int, int],
    lhs_mask: np.ndarray,
    rhs_bbox: tuple[int, int, int, int],
    rhs_mask: np.ndarray,
) -> float:
    intersection = intersection_area(lhs_bbox, lhs_mask, rhs_bbox, rhs_mask)
    union = int(lhs_mask.sum()) + int(rhs_mask.sum()) - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight ``(y0, x0, y1, x1)`` bounding box of a boolean mask, or ``None`` if empty."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def centroid_gate(mask: np.ndarray) -> tuple[float, float, float] | None:
    """Centroid-distance prefilter for nodes that may overlap ``mask``.

    Returns ``(cy, cx, radius)``: a candidate node is plausible when its
    centroid lies within ``radius`` of ``(cy, cx)``. This replaces requiring the
    node centroid to fall *inside* the source bbox, which drops nodes that
    overlap an elongated / crescent / merged source but whose centroid sits
    outside that box. ``radius`` is the source bbox diagonal — generous enough
    to reach a comparably-sized node touching the source anywhere along its
    extent. Returns ``None`` for an empty mask.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    cy = float(ys.mean())
    cx = float(xs.mean())
    radius = float(np.hypot(ys.max() - ys.min(), xs.max() - xs.min()))
    return cy, cx, radius


def match_mask_to_node(
    session, frame: int, source_mask: np.ndarray
) -> tuple[int, tuple[int, int, int, int], np.ndarray] | None:
    """Best-IoU Ultrack node at ``frame`` overlapping ``source_mask``.

    Returns ``(node_id, bbox, mask_crop)`` for the node whose mask has the
    highest IoU with ``source_mask``, or ``None`` if the mask is empty or
    nothing overlaps. Prefilters NodeDB by centroid distance (not source-bbox
    containment) so a node overlapping an elongated/crescent/merged source —
    whose centroid sits outside the source bbox — is still considered before the
    exact-IoU stage prunes non-overlappers.
    """
    from ultrack.core.database import NodeDB

    src_bbox = mask_bbox(source_mask)
    gate = centroid_gate(source_mask)
    if src_bbox is None or gate is None:
        return None
    sy0, sx0, sy1, sx1 = src_bbox
    src_crop = np.ascontiguousarray(source_mask[sy0:sy1, sx0:sx1], dtype=bool)

    cy, cx, radius = gate
    dist_sq = (NodeDB.y - cy) * (NodeDB.y - cy) + (NodeDB.x - cx) * (NodeDB.x - cx)
    rows = (
        session.query(NodeDB.id, NodeDB.pickle)
        .filter(
            NodeDB.t == frame,
            dist_sq <= radius * radius,
        )
        .all()
    )
    matched_id: int | None = None
    best_iou = 0.0
    matched_geom: tuple[tuple[int, int, int, int], np.ndarray] | None = None
    for nid, blob in rows:
        try:
            bbox, mask_crop = node_bbox_and_mask(int(nid), blob)
        except Exception:
            continue
        iou = raw_iou(src_bbox, src_crop, bbox, mask_crop)
        if iou > best_iou:
            best_iou = iou
            matched_id = int(nid)
            matched_geom = (bbox, mask_crop)
    if matched_id is None or matched_geom is None:
        return None
    bbox, mask_crop = matched_geom
    return matched_id, bbox, mask_crop


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
