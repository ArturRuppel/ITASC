from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from skimage.measure import regionprops

from cellflow.database.hypotheses import list_hypotheses, read_hypothesis_labels
from cellflow.tracking_ultrack import scoring as _scoring
from cellflow.tracking_ultrack._node_geometry import node_bbox_and_mask

_D_MAX_DEFAULT = 40.0
_GREEDY_CANDIDATE_LIMIT = 5


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


def extend_track(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,   # (T, Y, X) uint32
    hypotheses_path: Path,
    d_max: float = _D_MAX_DEFAULT,
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    distance_weight: float = 0.25,
    overlap_penalty: float = 1.0,
) -> ExtendResult | None:
    T = tracked_labels.shape[0]
    target_frame = source_frame + (1 if direction == "forward" else -1)

    if target_frame < 0 or target_frame >= T:
        return None

    source_mask = tracked_labels[source_frame] == source_id
    if not source_mask.any():
        return None

    props = regionprops(source_mask.astype(np.uint8))
    src_cy, src_cx = props[0].centroid
    src_area = float(props[0].area)

    target_frame_labels = tracked_labels[target_frame]
    other_cells = (target_frame_labels != 0) & (target_frame_labels != source_id)

    n_p, _ = list_hypotheses(hypotheses_path)
    if n_p == 0:
        return None

    best: ExtendResult | None = None
    best_score: tuple[float, float] | None = None

    for p in range(n_p):
        labels_raw = read_hypothesis_labels(hypotheses_path, target_frame, p)
        labels_2d: np.ndarray
        if labels_raw.ndim == 3 and labels_raw.shape[0] == 1:
            labels_2d = labels_raw[0]
        elif labels_raw.ndim == 2:
            labels_2d = labels_raw
        else:
            continue

        for rp in regionprops(labels_2d.astype(np.int32)):
            cy, cx = rp.centroid
            dist = float(np.hypot(cy - src_cy, cx - src_cx))
            if dist > d_max:
                continue

            cand_area = float(rp.area)
            cand_mask = labels_2d == rp.label
            existing_overlap = float((cand_mask & other_cells).sum()) / cand_area
            area_ratio = min(src_area, cand_area) / max(src_area, cand_area)
            shape_iou = _scoring.centroid_corrected_iou(source_mask, cand_mask)
            score = _extend_score(
                area_ratio=area_ratio,
                centroid_corrected_iou=shape_iou,
                centroid_distance=dist,
                existing_overlap=existing_overlap,
                area_weight=area_weight,
                iou_weight=iou_weight,
                distance_weight=distance_weight,
                overlap_penalty=overlap_penalty,
            )

            if best_score is None or score > best_score:
                y0, x0, y1, x1 = rp.bbox
                assignment = ExtendAssignment(
                    cell_id=source_id,
                    candidate_label=int(rp.label),
                    candidate_partition=p,
                    mask_2d=cand_mask,
                    bbox=(y0, x0, y1, x1),
                    centroid_distance=dist,
                    area_ratio=area_ratio,
                    centroid_corrected_iou=shape_iou,
                    existing_overlap=existing_overlap,
                    score=score[0],
                )
                best = _result_from_assignment(assignment, target_frame)
                best_score = score

    return best


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
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    if not db_path.exists():
        return None

    T = tracked_labels.shape[0]
    target_frame = source_frame + (1 if direction == "forward" else -1)
    if target_frame < 0 or target_frame >= T:
        return None

    source_mask = tracked_labels[source_frame] == source_id
    if not source_mask.any():
        return None
    source_stats = _mask_centroid_area(source_mask)
    if source_stats is None:
        return None
    src_cy, src_cx, _src_area = source_stats

    target_frame_labels = tracked_labels[target_frame]

    engine = sqla.create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    candidates: list[_DbCandidate] = []

    with Session(engine) as session:
        rows = (
            session.query(NodeDB)
            .filter(
                NodeDB.t == target_frame,
                NodeDB.y >= src_cy - d_max,
                NodeDB.y <= src_cy + d_max,
                NodeDB.x >= src_cx - d_max,
                NodeDB.x <= src_cx + d_max,
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
            full_mask = np.zeros(tracked_labels.shape[1:], dtype=bool)
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
    engine.dispose()

    if not candidates:
        return None

    if greedy_overwrite:
        assignments = _top_assignments_for_cell(
            cell_id=source_id,
            reference_mask=source_mask,
            reference_stats=source_stats,
            target_frame_labels=target_frame_labels,
            candidates=candidates,
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

    assignments = _top_assignments_for_cell(
        cell_id=source_id,
        reference_mask=source_mask,
        reference_stats=source_stats,
        target_frame_labels=target_frame_labels,
        candidates=candidates,
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
