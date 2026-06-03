from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from skimage.measure import regionprops

from cellflow.tracking_ultrack import scoring as _scoring
from cellflow.tracking_ultrack._node_geometry import node_bbox_and_mask

_D_MAX_DEFAULT = 40.0
_GREEDY_CANDIDATE_LIMIT = 5
# How many ranked candidates the gallery API surfaces per direction.
_EXTEND_CANDIDATE_LIMIT = 6


@dataclass(frozen=True)
class ExtendAssignment:
    cell_id: int
    candidate_label: int
    candidate_partition: int
    mask_2d: np.ndarray       # bool, full-frame (Y, X)
    bbox: tuple[int, int, int, int]
    centroid_distance: float
    area_ratio: float
    centroid_corrected_iou: float
    existing_overlap: float
    score: float


@dataclass
class ExtendResult:
    target_frame: int
    candidate_label: int
    candidate_partition: int
    mask_2d: np.ndarray       # bool, full-frame (Y, X)
    bbox: tuple[int, int, int, int]  # (y0, x0, y1, x1)
    centroid_distance: float
    area_ratio: float         # ∈ [0, 1]; 1.0 = identical area
    centroid_corrected_iou: float
    existing_overlap: float   # candidate ∩ (other cells at target) / candidate_area
    assignments: tuple[ExtendAssignment, ...] = ()


@dataclass(frozen=True)
class ExtendCandidates:
    """Ranked extend shortlist for one direction, best-first, for a gallery.

    ``target_frame`` is the adjacent frame the candidates live on (one step in
    ``direction`` from the source); ``assignments`` carries each candidate's
    full-frame mask, bbox and score so the UI can render clickable thumbnails.
    """

    target_frame: int
    assignments: tuple[ExtendAssignment, ...] = ()

    def is_empty(self) -> bool:
        return not self.assignments


def _extend_score(
    *,
    area_ratio: float,
    centroid_corrected_iou: float,
    centroid_distance: float,
    existing_overlap: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
    overlap_penalty: float,
) -> tuple[float, float]:
    weighted_score = _scoring.similarity_score(
        area_ratio=area_ratio,
        centroid_corrected_iou=centroid_corrected_iou,
        distance=centroid_distance,
        area_weight=area_weight,
        iou_weight=iou_weight,
        distance_weight=distance_weight,
    ) - overlap_penalty * existing_overlap
    return (weighted_score, -centroid_distance)


@dataclass(frozen=True)
class _DbCandidate:
    candidate_label: int
    candidate_partition: int
    mask_2d: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


def _mask_centroid_area(mask: np.ndarray) -> tuple[float, float, float] | None:
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return None
    cy, cx = props[0].centroid
    return float(cy), float(cx), float(props[0].area)


def _assignment_for_candidate(
    *,
    cell_id: int,
    reference_mask: np.ndarray,
    reference_stats: tuple[float, float, float],
    target_frame_labels: np.ndarray,
    candidate: _DbCandidate,
    d_max: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
    overlap_penalty: float,
) -> ExtendAssignment | None:
    src_cy, src_cx, src_area = reference_stats
    cand_cy, cand_cx = candidate.centroid
    dist = float(np.hypot(cand_cy - src_cy, cand_cx - src_cx))
    if dist > d_max:
        return None

    cand_area = float(candidate.mask_2d.sum())
    if cand_area == 0:
        return None
    other_cells = (target_frame_labels != 0) & (target_frame_labels != cell_id)
    existing_overlap = float((candidate.mask_2d & other_cells).sum()) / cand_area
    area_ratio = min(src_area, cand_area) / max(src_area, cand_area)
    shape_iou = _scoring.centroid_corrected_iou(reference_mask, candidate.mask_2d)
    score = _extend_score(
        area_ratio=area_ratio,
        centroid_corrected_iou=shape_iou,
        centroid_distance=dist,
        existing_overlap=existing_overlap,
        area_weight=area_weight,
        iou_weight=iou_weight,
        distance_weight=distance_weight,
        overlap_penalty=overlap_penalty,
    )[0]
    return ExtendAssignment(
        cell_id=cell_id,
        candidate_label=candidate.candidate_label,
        candidate_partition=candidate.candidate_partition,
        mask_2d=candidate.mask_2d,
        bbox=candidate.bbox,
        centroid_distance=dist,
        area_ratio=area_ratio,
        centroid_corrected_iou=shape_iou,
        existing_overlap=existing_overlap,
        score=score,
    )


def _result_from_assignment(
    assignment: ExtendAssignment,
    target_frame: int,
    assignments: tuple[ExtendAssignment, ...] | None = None,
) -> ExtendResult:
    return ExtendResult(
        target_frame=target_frame,
        candidate_label=assignment.candidate_label,
        candidate_partition=assignment.candidate_partition,
        mask_2d=assignment.mask_2d,
        bbox=assignment.bbox,
        centroid_distance=assignment.centroid_distance,
        area_ratio=assignment.area_ratio,
        centroid_corrected_iou=assignment.centroid_corrected_iou,
        existing_overlap=assignment.existing_overlap,
        assignments=assignments or (assignment,),
    )


def _top_assignments_for_cell(
    *,
    cell_id: int,
    reference_mask: np.ndarray,
    reference_stats: tuple[float, float, float],
    target_frame_labels: np.ndarray,
    candidates: list[_DbCandidate],
    d_max: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
    overlap_penalty: float,
    limit: int = _GREEDY_CANDIDATE_LIMIT,
) -> list[ExtendAssignment]:
    assignments = []
    for candidate in candidates:
        assignment = _assignment_for_candidate(
            cell_id=cell_id,
            reference_mask=reference_mask,
            reference_stats=reference_stats,
            target_frame_labels=target_frame_labels,
            candidate=candidate,
            d_max=d_max,
            area_weight=area_weight,
            iou_weight=iou_weight,
            distance_weight=distance_weight,
            overlap_penalty=overlap_penalty,
        )
        if assignment is not None:
            assignments.append(assignment)
    assignments.sort(key=lambda item: (item.score, -item.centroid_distance), reverse=True)
    return assignments[:limit]


def _query_db_candidates(
    *,
    db_path: Path,
    target_frame: int,
    center: tuple[float, float],
    d_max: float,
    frame_shape: tuple[int, int],
) -> list[_DbCandidate]:
    """Read every node at ``target_frame`` within ``d_max`` of ``center``.

    The y/x window is applied in SQL so masks for distant nodes are never
    deserialized; each surviving node becomes a full-frame ``_DbCandidate``.
    """
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    cy, cx = center
    engine = sqla.create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    candidates: list[_DbCandidate] = []
    try:
        with Session(engine) as session:
            rows = (
                session.query(NodeDB)
                .filter(
                    NodeDB.t == target_frame,
                    NodeDB.y >= cy - d_max,
                    NodeDB.y <= cy + d_max,
                    NodeDB.x >= cx - d_max,
                    NodeDB.x <= cx + d_max,
                )
                .all()
            )
            for node in rows:
                try:
                    (y0, x0, y1, x1), mask_2d = node_bbox_and_mask(int(node.id), node.pickle)
                except Exception:
                    continue
                if mask_2d.shape != (y1 - y0, x1 - x0):
                    continue
                full_mask = np.zeros(frame_shape, dtype=bool)
                full_mask[y0:y1, x0:x1] = mask_2d
                if not full_mask.any():
                    continue
                candidates.append(
                    _DbCandidate(
                        candidate_label=int(node.id),
                        candidate_partition=0,
                        mask_2d=full_mask,
                        bbox=(y0, x0, y1, x1),
                        centroid=(float(node.y), float(node.x)),
                    )
                )
    finally:
        engine.dispose()
    return candidates


def _ranked_extend_assignments(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,
    db_path: Path,
    d_max: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
    overlap_penalty: float,
    limit: int,
) -> tuple[int, list[ExtendAssignment]]:
    """Shared core: ``(target_frame, ranked assignments)`` for the adjacent frame.

    Returns an empty assignment list (with the computed ``target_frame``) when
    the DB is missing, the target frame is out of range, the source cell is
    absent, or nothing scores within ``d_max``.
    """
    target_frame = source_frame + (1 if direction == "forward" else -1)
    if not Path(db_path).exists():
        return target_frame, []
    if target_frame < 0 or target_frame >= tracked_labels.shape[0]:
        return target_frame, []

    source_mask = tracked_labels[source_frame] == source_id
    if not source_mask.any():
        return target_frame, []
    source_stats = _mask_centroid_area(source_mask)
    if source_stats is None:
        return target_frame, []
    src_cy, src_cx, _src_area = source_stats

    candidates = _query_db_candidates(
        db_path=Path(db_path),
        target_frame=target_frame,
        center=(src_cy, src_cx),
        d_max=d_max,
        frame_shape=tracked_labels.shape[1:],
    )
    if not candidates:
        return target_frame, []

    assignments = _top_assignments_for_cell(
        cell_id=source_id,
        reference_mask=source_mask,
        reference_stats=source_stats,
        target_frame_labels=tracked_labels[target_frame],
        candidates=candidates,
        d_max=d_max,
        area_weight=area_weight,
        iou_weight=iou_weight,
        distance_weight=distance_weight,
        overlap_penalty=overlap_penalty,
        limit=limit,
    )
    return target_frame, assignments


def extend_track_from_db(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,   # (T, Y, X) uint32
    db_path: Path,
    d_max: float = _D_MAX_DEFAULT,
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    distance_weight: float = 0.25,
    overlap_penalty: float = 1.0,
    greedy_overwrite: bool = False,
    validated_tracks: dict[int, set[int]] | None = None,
) -> ExtendResult | None:
    """Extend a track using candidates from ultrack_workdir/data.db.

    Returns None if the DB is missing, target frame is out of range, or no
    candidate within d_max is found.  Widget caller should show a local status
    message on None.
    """
    target_frame, assignments = _ranked_extend_assignments(
        source_id=source_id,
        source_frame=source_frame,
        direction=direction,
        tracked_labels=tracked_labels,
        db_path=db_path,
        d_max=d_max,
        area_weight=area_weight,
        iou_weight=iou_weight,
        distance_weight=distance_weight,
        overlap_penalty=overlap_penalty,
        limit=1,
    )
    if not assignments:
        return None
    return _result_from_assignment(assignments[0], target_frame)


def list_extend_candidates(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,   # (T, Y, X) uint32
    db_path: Path,
    d_max: float = _D_MAX_DEFAULT,
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    distance_weight: float = 0.25,
    overlap_penalty: float = 1.0,
    limit: int = _EXTEND_CANDIDATE_LIMIT,
) -> ExtendCandidates:
    """Ranked extend candidates at the adjacent frame, best-first, for a gallery.

    Same matching and scoring as :func:`extend_track_from_db`, but returns the
    whole ranked shortlist (each with its full-frame mask) instead of only the
    winner, so the correction UI can render clickable candidate thumbnails.
    Returns an empty :class:`ExtendCandidates` (still carrying ``target_frame``)
    when the DB is missing, the target frame is out of range, the source cell is
    absent, or nothing scores within ``d_max``.
    """
    target_frame, assignments = _ranked_extend_assignments(
        source_id=source_id,
        source_frame=source_frame,
        direction=direction,
        tracked_labels=tracked_labels,
        db_path=db_path,
        d_max=d_max,
        area_weight=area_weight,
        iou_weight=iou_weight,
        distance_weight=distance_weight,
        overlap_penalty=overlap_penalty,
        limit=limit,
    )
    return ExtendCandidates(target_frame=target_frame, assignments=tuple(assignments))
