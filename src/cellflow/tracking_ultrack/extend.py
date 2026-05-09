from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
from skimage.measure import regionprops

from cellflow.database.hypotheses import list_hypotheses, read_hypothesis_labels

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


def _centroid_corrected_iou(source_mask: np.ndarray, candidate_mask: np.ndarray) -> float:
    """IoU after translating candidate pixels to the source centroid."""
    src_y, src_x = np.nonzero(source_mask)
    cand_y, cand_x = np.nonzero(candidate_mask)
    if len(src_y) == 0 or len(cand_y) == 0:
        return 0.0

    dy = int(round(float(src_y.mean()) - float(cand_y.mean())))
    dx = int(round(float(src_x.mean()) - float(cand_x.mean())))
    src_pixels = set(zip(src_y.tolist(), src_x.tolist()))
    cand_pixels = set(zip((cand_y + dy).tolist(), (cand_x + dx).tolist()))
    intersection = len(src_pixels & cand_pixels)
    union = len(src_pixels) + len(cand_pixels) - intersection
    return intersection / union if union > 0 else 0.0


def _extend_score(
    *,
    area_ratio: float,
    centroid_corrected_iou: float,
    centroid_distance: float,
    d_max: float,
    existing_overlap: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
    overlap_penalty: float,
) -> tuple[float, float]:
    distance_score = 1.0 if d_max <= 0 else max(0.0, 1.0 - centroid_distance / d_max)
    weighted_score = (
        area_weight * area_ratio
        + iou_weight * centroid_corrected_iou
        + distance_weight * distance_score
        - overlap_penalty * existing_overlap
    )
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
    target_frame_labels: np.ndarray,
    candidate: _DbCandidate,
    d_max: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
    overlap_penalty: float,
) -> ExtendAssignment | None:
    stats = _mask_centroid_area(reference_mask)
    if stats is None:
        return None
    src_cy, src_cx, src_area = stats
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
    shape_iou = _centroid_corrected_iou(reference_mask, candidate.mask_2d)
    score = _extend_score(
        area_ratio=area_ratio,
        centroid_corrected_iou=shape_iou,
        centroid_distance=dist,
        d_max=d_max,
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


def _masks_are_disjoint(assignments: tuple[ExtendAssignment, ...]) -> bool:
    occupied: np.ndarray | None = None
    for assignment in assignments:
        if occupied is None:
            occupied = assignment.mask_2d.copy()
            continue
        if np.any(occupied & assignment.mask_2d):
            return False
        occupied |= assignment.mask_2d
    return True


def _plan_overwrites_only_assigned_cells(
    assignments: tuple[ExtendAssignment, ...],
    target_frame_labels: np.ndarray,
    protected_ids: set[int],
) -> bool:
    assigned_ids = {int(assignment.cell_id) for assignment in assignments}
    overwritten = np.zeros_like(target_frame_labels, dtype=bool)
    for assignment in assignments:
        overwritten |= assignment.mask_2d
    overwritten_ids = {
        int(v)
        for v in np.unique(target_frame_labels[overwritten])
        if int(v) != 0
    }
    return (overwritten_ids & protected_ids).issubset(assigned_ids)


def _plan_preserves_locked_target_cells(
    assignments: tuple[ExtendAssignment, ...],
    target_frame_labels: np.ndarray,
    locked_target_ids: set[int],
) -> bool:
    if not locked_target_ids:
        return True
    overwritten = np.zeros_like(target_frame_labels, dtype=bool)
    for assignment in assignments:
        overwritten |= assignment.mask_2d
    overwritten_ids = {
        int(v)
        for v in np.unique(target_frame_labels[overwritten])
        if int(v) != 0
    }
    return not (overwritten_ids & locked_target_ids)


def _top_assignments_for_cell(
    *,
    cell_id: int,
    reference_mask: np.ndarray,
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


def _best_greedy_overwrite_plan(
    *,
    source_id: int,
    source_mask: np.ndarray,
    source_frame_labels: np.ndarray,
    target_frame_labels: np.ndarray,
    candidates: list[_DbCandidate],
    d_max: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
    overlap_penalty: float,
    locked_target_ids: set[int] | None = None,
) -> tuple[ExtendAssignment, ...] | None:
    locked_target_ids = set(locked_target_ids or set())
    if source_id in locked_target_ids:
        return None

    source_assignments = _top_assignments_for_cell(
        cell_id=source_id,
        reference_mask=source_mask,
        target_frame_labels=target_frame_labels,
        candidates=candidates,
        d_max=d_max,
        area_weight=area_weight,
        iou_weight=iou_weight,
        distance_weight=distance_weight,
        overlap_penalty=overlap_penalty,
    )
    if not source_assignments:
        return None

    best_plan: tuple[ExtendAssignment, ...] | None = None
    best_score: tuple[float, float] | None = None
    protected_ids = {int(v) for v in np.unique(source_frame_labels) if int(v) != 0}
    protected_ids.discard(source_id)

    for source_assignment in source_assignments:
        if not _plan_preserves_locked_target_cells(
            (source_assignment,),
            target_frame_labels,
            locked_target_ids,
        ):
            continue
        conflict_labels = target_frame_labels[
            source_assignment.mask_2d
            & (target_frame_labels != 0)
            & (target_frame_labels != source_id)
        ]
        conflicted_ids = sorted(
            int(v)
            for v in np.unique(conflict_labels)
            if int(v) != 0 and int(v) in protected_ids
        )
        if not conflicted_ids:
            plan = (source_assignment,)
            score = (source_assignment.score, -source_assignment.centroid_distance)
            if best_score is None or score > best_score:
                best_plan = plan
                best_score = score
            continue

        choices: list[list[ExtendAssignment]] = []
        for cell_id in conflicted_ids:
            reference_mask = source_frame_labels == cell_id
            cell_choices = _top_assignments_for_cell(
                cell_id=cell_id,
                reference_mask=reference_mask,
                target_frame_labels=target_frame_labels,
                candidates=candidates,
                d_max=d_max,
                area_weight=area_weight,
                iou_weight=iou_weight,
                distance_weight=distance_weight,
                overlap_penalty=overlap_penalty,
            )
            if not cell_choices:
                choices = []
                break
            choices.append(cell_choices)
        if not choices:
            continue

        for combo in product(*choices):
            plan = (source_assignment, *combo)
            if not _masks_are_disjoint(plan):
                continue
            if not _plan_preserves_locked_target_cells(
                plan,
                target_frame_labels,
                locked_target_ids,
            ):
                continue
            if not _plan_overwrites_only_assigned_cells(
                plan,
                target_frame_labels,
                protected_ids,
            ):
                continue
            total_score = sum(item.score for item in plan)
            total_distance = sum(item.centroid_distance for item in plan)
            score = (total_score, -total_distance)
            if best_score is None or score > best_score:
                best_plan = plan
                best_score = score

    return best_plan


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
            shape_iou = _centroid_corrected_iou(source_mask, cand_mask)
            score = _extend_score(
                area_ratio=area_ratio,
                centroid_corrected_iou=shape_iou,
                centroid_distance=dist,
                d_max=d_max,
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
    from cellflow.tracking_ultrack.validation_nodes import _node_bbox_and_mask

    if not db_path.exists():
        return None

    T = tracked_labels.shape[0]
    target_frame = source_frame + (1 if direction == "forward" else -1)
    if target_frame < 0 or target_frame >= T:
        return None

    source_mask = tracked_labels[source_frame] == source_id
    if not source_mask.any():
        return None

    source_frame_labels = tracked_labels[source_frame]
    target_frame_labels = tracked_labels[target_frame]
    locked_target_ids = {
        int(cell_id)
        for cell_id, frames in (validated_tracks or {}).items()
        if target_frame in frames
    }

    engine = sqla.create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    candidates: list[_DbCandidate] = []

    with Session(engine) as session:
        rows = session.query(NodeDB).filter(NodeDB.t == target_frame).all()
        for node in rows:
            try:
                (y0, x0, y1, x1), mask_2d = _node_bbox_and_mask(int(node.id), node.pickle)
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
        plan = _best_greedy_overwrite_plan(
            source_id=source_id,
            source_mask=source_mask,
            source_frame_labels=source_frame_labels,
            target_frame_labels=target_frame_labels,
            candidates=candidates,
            d_max=d_max,
            area_weight=area_weight,
            iou_weight=iou_weight,
            distance_weight=distance_weight,
            overlap_penalty=overlap_penalty,
            locked_target_ids=locked_target_ids,
        )
        if not plan:
            return None
        return _result_from_assignment(plan[0], target_frame, plan)

    assignments = _top_assignments_for_cell(
        cell_id=source_id,
        reference_mask=source_mask,
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
