from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cellflow.tracking_ultrack._node_geometry import node_bbox_and_mask, raw_iou


@dataclass(frozen=True)
class SwapCandidate:
    node_id: int
    mask_2d: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int


@dataclass
class _SwapCursor:
    source_id: int
    frame: int
    source_centroid: tuple[float, float]
    source_area: int
    candidates: tuple[SwapCandidate, ...]
    displayed_area: int
    cursor: int | None
    baseline_frame: np.ndarray | None = None


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def list_swap_candidates(
    *,
    db_path: Path,
    frame: int,
    source_mask: np.ndarray,
    frame_shape: tuple[int, int],
    protected_mask: np.ndarray | None = None,
) -> list[SwapCandidate]:
    """Return the segmentation-hierarchy branch around ``source_mask``.

    Matches ``source_mask`` to the best-overlapping Ultrack node at ``frame``,
    then walks the hierarchy lattice — the matched node, its ancestors up to the
    root (larger merged segments) and all descendants (smaller fragments) — and
    returns them sorted by area. ``Z``/``C`` step through that area-ordered
    branch, so siblings at a level are visited in size order before descending.
    """
    if not Path(db_path).exists():
        return []

    src_bbox = _mask_bbox(source_mask)
    if src_bbox is None:
        return []
    sy0, sx0, sy1, sx1 = src_bbox
    src_crop = np.ascontiguousarray(source_mask[sy0:sy1, sx0:sx1], dtype=bool)

    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB
    from ultrack.utils.constants import NO_PARENT

    engine = sqla.create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )

    try:
        with Session(engine) as session:
            # Light query: id/parent for every node at the frame to build the tree.
            parent_by_id: dict[int, int] = {}
            for nid, pid in session.query(NodeDB.id, NodeDB.hier_parent_id).filter(
                NodeDB.t == frame
            ):
                parent = NO_PARENT if pid is None else int(pid)
                parent_by_id[int(nid)] = parent
            if not parent_by_id:
                return []

            children: dict[int, list[int]] = {}
            for child_id, parent_id in parent_by_id.items():
                if parent_id != NO_PARENT and parent_id in parent_by_id:
                    children.setdefault(parent_id, []).append(child_id)

            # Match source mask to a node by max IoU. Prefilter to nodes whose
            # centroid falls inside the source bounding box to avoid deserializing
            # every mask in the frame.
            match_rows = (
                session.query(NodeDB.id, NodeDB.pickle)
                .filter(
                    NodeDB.t == frame,
                    NodeDB.y >= sy0,
                    NodeDB.y <= sy1,
                    NodeDB.x >= sx0,
                    NodeDB.x <= sx1,
                )
                .all()
            )
            matched_id: int | None = None
            best_iou = 0.0
            for nid, blob in match_rows:
                try:
                    bbox, mask_crop = node_bbox_and_mask(int(nid), blob)
                except Exception:
                    continue
                iou = raw_iou(src_bbox, src_crop, bbox, mask_crop)
                if iou > best_iou:
                    best_iou = iou
                    matched_id = int(nid)
            if matched_id is None:
                return []

            # Walk up to the root of the matched node's lattice, then collect the
            # root's entire subtree so siblings at every level are reachable.
            root = matched_id
            while True:
                parent = parent_by_id.get(root, NO_PARENT)
                if parent == NO_PARENT or parent not in parent_by_id:
                    break
                root = parent

            branch: set[int] = {root}
            stack = [root]
            while stack:
                cur = stack.pop()
                for child in children.get(cur, ()):
                    if child not in branch:
                        branch.add(child)
                        stack.append(child)

            node_rows = (
                session.query(NodeDB)
                .filter(NodeDB.t == frame, NodeDB.id.in_(branch))
                .all()
            )

            results: list[SwapCandidate] = []
            for node in node_rows:
                try:
                    (y0, x0, y1, x1), mask_crop = node_bbox_and_mask(
                        int(node.id), node.pickle
                    )
                except Exception:
                    continue
                if mask_crop.shape != (y1 - y0, x1 - x0):
                    continue

                full_mask = np.zeros(frame_shape, dtype=bool)
                full_mask[y0:y1, x0:x1] = mask_crop
                if not full_mask.any():
                    continue

                if protected_mask is not None:
                    paintable = full_mask & ~protected_mask
                    if not paintable.any():
                        continue

                area = int(full_mask.sum())
                results.append(
                    SwapCandidate(
                        node_id=int(node.id),
                        mask_2d=full_mask,
                        bbox=(y0, x0, y1, x1),
                        centroid=(float(node.y), float(node.x)),
                        area=area,
                    )
                )
    finally:
        engine.dispose()

    results.sort(key=lambda c: c.area)
    return results


def step_smaller(
    candidates: tuple[SwapCandidate, ...] | list[SwapCandidate],
    displayed_area: int,
) -> int | None:
    best: int | None = None
    for i, c in enumerate(candidates):
        if c.area < displayed_area:
            best = i
    return best


def step_larger(
    candidates: tuple[SwapCandidate, ...] | list[SwapCandidate],
    displayed_area: int,
) -> int | None:
    for i, c in enumerate(candidates):
        if c.area > displayed_area:
            return i
    return None
