from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from itasc.tracking_ultrack._node_geometry import (
    intersection_area,
    mask_bbox,
    match_mask_to_node,
    node_bbox_and_mask,
)


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
    candidates: tuple[SwapCandidate, ...]
    index: int
    baseline_frame: np.ndarray | None = None


def list_swap_candidates(
    *,
    db_path: Path,
    frame: int,
    source_mask: np.ndarray,
    frame_shape: tuple[int, int],
    protected_mask: np.ndarray | None = None,
) -> list[SwapCandidate]:
    """Return the matched node's nesting lineage around ``source_mask``.

    Matches ``source_mask`` to the best-overlapping Ultrack node at ``frame``,
    then walks the candidate **containment lattice** recorded in ``OverlapDB``:
    every node that overlaps the matched node *and* is nested with it — a
    superset (larger merged segment that contains it) or a subset (smaller
    fragment it contains) — plus the matched node itself, returned sorted by
    area. Partially-overlapping neighbours (siblings/cousins) are excluded: they
    are different cells, so swapping onto them would relocate the cell rather
    than resize it. ``Z``/``C`` cycle through this area-sorted lineage by index.

    ``OverlapDB`` is populated by both database builders (atom-union enumeration
    and canonical Ultrack segmentation), whereas ``hier_parent_id`` is only set
    by the latter — so the overlap lattice is the portable source of structure.
    """
    if not Path(db_path).exists():
        return []

    if mask_bbox(source_mask) is None:
        return []

    import sqlalchemy as sqla
    from sqlalchemy import or_
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, OverlapDB

    engine = sqla.create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )

    try:
        with Session(engine) as session:
            # Match source mask to a node by max IoU (prefiltered by bbox so only
            # plausibly-overlapping masks are deserialized).
            matched = match_mask_to_node(session, frame, source_mask)
            if matched is None:
                return []
            matched_id, matched_bbox, matched_mask = matched
            matched_area = int(matched_mask.sum())

            # Candidate set = the matched node plus everything that overlaps it
            # (shares atoms) per OverlapDB. The pair direction is not meaningful
            # here, so match the id on either column.
            overlap_ids: set[int] = set()
            for node_id, ancestor_id in session.query(
                OverlapDB.node_id, OverlapDB.ancestor_id
            ).filter(
                or_(OverlapDB.node_id == matched_id, OverlapDB.ancestor_id == matched_id)
            ):
                overlap_ids.add(int(node_id))
                overlap_ids.add(int(ancestor_id))
            overlap_ids.add(matched_id)

            node_rows = (
                session.query(NodeDB)
                .filter(NodeDB.t == frame, NodeDB.id.in_(overlap_ids))
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

                # Keep only nodes nested with the matched node: one mask must be
                # a subset of the other. Partially-overlapping neighbours (which
                # share atoms but each hold unique ones) are different cells.
                if int(node.id) != matched_id:
                    inter = intersection_area(
                        matched_bbox, matched_mask, (y0, x0, y1, x1), mask_crop
                    )
                    cand_area = int(mask_crop.sum())
                    nested = inter == matched_area or inter == cand_area
                    if not nested:
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


def nearest_area_index(
    candidates: tuple[SwapCandidate, ...] | list[SwapCandidate],
    area: int,
) -> int:
    """Index of the candidate whose area is closest to ``area``.

    Used to seed the cursor on the lineage member that best matches the
    currently displayed (matched) cell, so the first ``Z``/``C`` step moves
    relative to it.
    """
    return min(range(len(candidates)), key=lambda i: abs(candidates[i].area - area))


def cycle_index(count: int, current: int, *, larger: bool) -> int:
    """Next index when cycling an area-sorted candidate list.

    ``larger`` steps toward bigger areas (higher index), otherwise toward
    smaller areas (lower index); both directions wrap around.
    """
    step = 1 if larger else -1
    return (current + step) % count
