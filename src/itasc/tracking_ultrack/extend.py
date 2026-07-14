from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from itasc.tracking_ultrack._node_geometry import (
    match_mask_to_node,
    node_bbox_and_mask,
)

# How many ranked candidates the gallery API surfaces per direction.
_EXTEND_CANDIDATE_LIMIT = 6


@dataclass(frozen=True)
class ExtendAssignment:
    cell_id: int
    candidate_label: int       # ultrack node id of the linked candidate
    candidate_partition: int
    mask_2d: np.ndarray       # bool, full-frame (Y, X)
    bbox: tuple[int, int, int, int]
    weight: float             # LinkDB.weight (association score); higher = better


@dataclass
class ExtendResult:
    target_frame: int
    candidate_label: int
    candidate_partition: int
    mask_2d: np.ndarray       # bool, full-frame (Y, X)
    bbox: tuple[int, int, int, int]  # (y0, x0, y1, x1)
    weight: float             # LinkDB.weight of the chosen link
    assignments: tuple[ExtendAssignment, ...] = ()


@dataclass(frozen=True)
class ExtendCandidates:
    """Ranked extend shortlist for one direction, best-first, for a gallery.

    ``target_frame`` is the adjacent frame the candidates live on (one step in
    ``direction`` from the source); ``assignments`` carries each candidate's
    full-frame mask, bbox and link weight so the UI can render clickable
    thumbnails.
    """

    target_frame: int
    assignments: tuple[ExtendAssignment, ...] = ()

    def is_empty(self) -> bool:
        return not self.assignments


def _ranked_link_targets(
    session, node_id: int, direction: Literal["forward", "backward"]
) -> list[tuple[int, float]]:
    """Linked candidate node ids in the adjacent frame, best (highest weight) first.

    ``LinkDB`` edges point forward in time (``source_id`` at ``t``, ``target_id``
    at ``t + 1``). Forward extension reads rows where the matched node is the
    source and collects the targets; backward extension reads rows where it is
    the target and collects the sources. A NULL weight is treated as ``1.0``.
    """
    from ultrack.core.database import LinkDB

    if direction == "forward":
        rows = (
            session.query(LinkDB.target_id, LinkDB.weight)
            .filter(LinkDB.source_id == node_id)
            .all()
        )
    else:
        rows = (
            session.query(LinkDB.source_id, LinkDB.weight)
            .filter(LinkDB.target_id == node_id)
            .all()
        )
    ranked = [
        (int(nid), float(weight) if weight is not None else 1.0)
        for nid, weight in rows
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _extend_assignments(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,
    db_path: Path,
    limit: int,
) -> tuple[int, list[ExtendAssignment]]:
    """Shared core: ``(target_frame, ranked assignments)`` for the adjacent frame.

    Matches the source label's mask to an Ultrack node, then walks the
    precomputed ``LinkDB`` edges from that node into the adjacent frame, ranked
    by association weight. Only the top ``limit`` linked candidates' masks are
    deserialized. Returns an empty assignment list (with the computed
    ``target_frame``) when the DB is missing, the target frame is out of range,
    the source cell is absent, the source mask matches no node, or the matched
    node has no link in the requested direction.
    """
    target_frame = source_frame + (1 if direction == "forward" else -1)
    if not Path(db_path).exists():
        return target_frame, []
    if target_frame < 0 or target_frame >= tracked_labels.shape[0]:
        return target_frame, []

    source_mask = tracked_labels[source_frame] == source_id
    if not source_mask.any():
        return target_frame, []

    frame_shape = tracked_labels.shape[1:]

    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    engine = sqla.create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    assignments: list[ExtendAssignment] = []
    try:
        with Session(engine) as session:
            matched = match_mask_to_node(session, source_frame, source_mask)
            if matched is None:
                return target_frame, []
            matched_id, _bbox, _mask = matched

            ranked = _ranked_link_targets(session, matched_id, direction)
            if not ranked:
                return target_frame, []

            # Load only the linked candidates' masks (a handful), restricted to
            # the adjacent frame as a safety net against malformed links.
            node_rows = {
                int(node.id): node
                for node in session.query(NodeDB)
                .filter(
                    NodeDB.t == target_frame,
                    NodeDB.id.in_([nid for nid, _ in ranked]),
                )
                .all()
            }
            for nid, weight in ranked:
                node = node_rows.get(nid)
                if node is None:
                    continue
                try:
                    (y0, x0, y1, x1), mask_crop = node_bbox_and_mask(nid, node.pickle)
                except Exception:
                    continue
                if mask_crop.shape != (y1 - y0, x1 - x0):
                    continue
                full_mask = np.zeros(frame_shape, dtype=bool)
                full_mask[y0:y1, x0:x1] = mask_crop
                if not full_mask.any():
                    continue
                assignments.append(
                    ExtendAssignment(
                        cell_id=source_id,
                        candidate_label=nid,
                        candidate_partition=0,
                        mask_2d=full_mask,
                        bbox=(y0, x0, y1, x1),
                        weight=weight,
                    )
                )
                if limit and len(assignments) >= limit:
                    break
    finally:
        engine.dispose()
    return target_frame, assignments


def extend_track_from_db(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,   # (T, Y, X) uint32
    db_path: Path,
) -> ExtendResult | None:
    """Extend a track to the adjacent frame using ultrack ``LinkDB`` candidates.

    The source label's mask is matched to an Ultrack node; the highest-weight
    ``LinkDB`` edge into the adjacent frame picks the continuation. Returns
    ``None`` when the DB is missing, the target frame is out of range, the
    source cell is absent, or the matched node has no link in ``direction`` (the
    caller should show a local status message and leave the frame untouched).
    """
    target_frame, assignments = _extend_assignments(
        source_id=source_id,
        source_frame=source_frame,
        direction=direction,
        tracked_labels=tracked_labels,
        db_path=db_path,
        limit=1,
    )
    if not assignments:
        return None
    best = assignments[0]
    return ExtendResult(
        target_frame=target_frame,
        candidate_label=best.candidate_label,
        candidate_partition=best.candidate_partition,
        mask_2d=best.mask_2d,
        bbox=best.bbox,
        weight=best.weight,
        assignments=(best,),
    )


def list_extend_candidates(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,   # (T, Y, X) uint32
    db_path: Path,
    limit: int = _EXTEND_CANDIDATE_LIMIT,
) -> ExtendCandidates:
    """Ranked extend candidates at the adjacent frame, best-first, for a gallery.

    Same matching as :func:`extend_track_from_db`, but returns the whole ranked
    shortlist of ``LinkDB`` candidates (each with its full-frame mask) instead of
    only the winner, so the correction UI can render clickable candidate
    thumbnails. Returns an empty :class:`ExtendCandidates` (still carrying
    ``target_frame``) when there is no DB, target frame, matched node or link.
    """
    target_frame, assignments = _extend_assignments(
        source_id=source_id,
        source_frame=source_frame,
        direction=direction,
        tracked_labels=tracked_labels,
        db_path=db_path,
        limit=limit,
    )
    return ExtendCandidates(target_frame=target_frame, assignments=tuple(assignments))
