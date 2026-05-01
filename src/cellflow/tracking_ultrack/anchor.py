"""Anchor-frame constraints for Ultrack solves."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlalchemy as sqla
from skimage.measure import regionprops
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class AnchorReport:
    frame_index: int
    n_gt_labels: int
    n_matched: int
    n_unmatched: int
    matched_node_ids: list[int]
    unmatched_labels: list[int]
    mean_matched_iou: float
    min_matched_iou: float


@dataclass(frozen=True)
class AnchorSuppressionReport:
    frame_index: int
    neighbor_offsets: tuple[int, ...]
    suppressed_node_ids: list[int]
    by_frame: dict[int, int]


@dataclass(frozen=True)
class _MaskRecord:
    label_id: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray


def _frame_2d(labels: np.ndarray, frame_index: int) -> np.ndarray:
    arr = np.asarray(labels)
    frame = arr[frame_index]
    if frame.ndim == 3:
        if frame.shape[0] == 1:
            return frame[0]
        raise NotImplementedError("Anchor matching for true 3D labels is not implemented")
    if frame.ndim == 2:
        return frame
    raise ValueError(f"Expected frame to be 2D or singleton-Z 3D, got {frame.shape}")


def _gt_masks(labels: np.ndarray, frame_index: int) -> list[_MaskRecord]:
    frame = _frame_2d(labels, frame_index)
    masks: list[_MaskRecord] = []
    for prop in regionprops(frame):
        y0, x0, y1, x1 = prop.bbox
        mask = np.ascontiguousarray(frame[y0:y1, x0:x1] == prop.label, dtype=bool)
        masks.append(_MaskRecord(int(prop.label), (int(y0), int(x0), int(y1), int(x1)), mask))
    return masks


def _node_mask_record(node_id: int, node) -> _MaskRecord:
    bbox = np.asarray(node.bbox)
    ndim = len(bbox) // 2
    if ndim == 3:
        y0, x0, y1, x1 = int(bbox[1]), int(bbox[2]), int(bbox[4]), int(bbox[5])
    elif ndim == 2:
        y0, x0, y1, x1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    else:
        raise ValueError(f"Unexpected node bbox shape for node {node_id}: {bbox}")

    mask = np.asarray(node.mask, dtype=bool)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask.any(axis=0)
    if mask.ndim != 2:
        raise ValueError(f"Unexpected node mask shape for node {node_id}: {mask.shape}")
    return _MaskRecord(int(node_id), (y0, x0, y1, x1), np.ascontiguousarray(mask, dtype=bool))


def _mask_iou(lhs: _MaskRecord, rhs: _MaskRecord) -> float:
    intersection = _mask_intersection(lhs, rhs)
    lhs_area = int(lhs.mask.sum())
    rhs_area = int(rhs.mask.sum())
    if lhs_area == 0 and rhs_area == 0:
        return 1.0
    if lhs_area == 0 or rhs_area == 0:
        return 0.0
    union = lhs_area + rhs_area - intersection
    return float(intersection / union) if union else 0.0


def _mask_intersection(lhs: _MaskRecord, rhs: _MaskRecord) -> int:
    ly0, lx0, ly1, lx1 = lhs.bbox
    ry0, rx0, ry1, rx1 = rhs.bbox
    oy0, ox0 = max(ly0, ry0), max(lx0, rx0)
    oy1, ox1 = min(ly1, ry1), min(lx1, rx1)

    intersection = 0
    if oy0 < oy1 and ox0 < ox1:
        lhs_crop = lhs.mask[oy0 - ly0: oy1 - ly0, ox0 - lx0: ox1 - lx0]
        rhs_crop = rhs.mask[oy0 - ry0: oy1 - ry0, ox0 - rx0: ox1 - rx0]
        intersection = int(np.logical_and(lhs_crop, rhs_crop).sum())
    return intersection


def _node_containment_in_anchor(anchor: _MaskRecord, node: _MaskRecord) -> float:
    node_area = int(node.mask.sum())
    if node_area == 0:
        return 0.0
    return float(_mask_intersection(anchor, node) / node_area)


def annotate_anchor_frame(
    working_dir: str | Path,
    anchor_labels: np.ndarray,
    *,
    frame_index: int,
    min_iou: float = 0.95,
) -> AnchorReport:
    """Pin the solver's selected nodes at one frame to match an anchor labelmap.

    Matched nodes at ``frame_index`` are marked ``VarAnnotation.REAL``. Every
    other node in the same frame is marked ``VarAnnotation.FAKE`` so the anchor
    frame cannot contain extra non-GT cells. Other frames are left unannotated.
    """
    from ultrack.core.database import NodeDB, VarAnnotation

    working_dir = Path(working_dir)
    db_url = f"sqlite:///{working_dir / 'data.db'}"
    gt_records = _gt_masks(anchor_labels, frame_index)
    matched_node_ids: list[int] = []
    matched_ious: list[float] = []
    unmatched_labels: list[int] = []

    engine = sqla.create_engine(db_url)
    with Session(engine) as session:
        node_rows = (
            session.query(NodeDB.id, NodeDB.pickle)
            .where(NodeDB.t == frame_index)
            .all()
        )
        node_records = [
            _node_mask_record(int(node_id), node)
            for node_id, node in node_rows
        ]

        available = {rec.label_id for rec in node_records}
        by_id = {rec.label_id: rec for rec in node_records}

        candidates: list[tuple[float, int, int]] = []
        for gt in gt_records:
            for node in node_records:
                iou = _mask_iou(gt, node)
                if iou >= min_iou:
                    candidates.append((iou, gt.label_id, node.label_id))

        matched_gt: set[int] = set()
        for iou, gt_label, node_id in sorted(candidates, reverse=True):
            if gt_label in matched_gt or node_id not in available:
                continue
            matched_gt.add(gt_label)
            available.remove(node_id)
            matched_node_ids.append(node_id)
            matched_ious.append(iou)

        for gt in gt_records:
            if gt.label_id not in matched_gt:
                unmatched_labels.append(gt.label_id)

        session.query(NodeDB).where(NodeDB.t == frame_index).update(
            {NodeDB.node_annot: VarAnnotation.FAKE},
            synchronize_session=False,
        )
        if matched_node_ids:
            session.query(NodeDB).where(NodeDB.id.in_(matched_node_ids)).update(
                {NodeDB.node_annot: VarAnnotation.REAL},
                synchronize_session=False,
            )
        session.commit()

    matched_node_ids.sort()
    unmatched_labels.sort()
    return AnchorReport(
        frame_index=int(frame_index),
        n_gt_labels=len(gt_records),
        n_matched=len(matched_node_ids),
        n_unmatched=len(unmatched_labels),
        matched_node_ids=matched_node_ids,
        unmatched_labels=unmatched_labels,
        mean_matched_iou=float(np.mean(matched_ious)) if matched_ious else 0.0,
        min_matched_iou=float(np.min(matched_ious)) if matched_ious else 0.0,
    )


def suppress_anchor_adjacent_fragments(
    working_dir: str | Path,
    anchor_labels: np.ndarray,
    *,
    frame_index: int,
    neighbor_offsets: tuple[int, ...] = (-1, 1),
    min_best_iou: float = 0.60,
    fragment_max_iou_fraction: float = 0.80,
    min_fragment_containment: float = 0.90,
) -> AnchorSuppressionReport:
    """Suppress obvious fragment alternatives next to an anchored frame.

    For each anchored object, each neighboring frame keeps its best-overlapping
    candidate unconstrained. Other candidates are marked FAKE only when they are
    mostly contained inside the anchored object and have substantially lower IoU
    than the best candidate. Ambiguous candidates are left untouched.
    """
    from ultrack.core.database import NodeDB, VarAnnotation

    working_dir = Path(working_dir)
    db_url = f"sqlite:///{working_dir / 'data.db'}"
    anchor_records = _gt_masks(anchor_labels, frame_index)
    suppressed: set[int] = set()
    by_frame: dict[int, int] = {}

    engine = sqla.create_engine(db_url)
    with Session(engine) as session:
        for offset in neighbor_offsets:
            t = int(frame_index + offset)
            if t < 0:
                continue
            node_rows = (
                session.query(NodeDB.id, NodeDB.pickle, NodeDB.node_annot)
                .where(NodeDB.t == t)
                .all()
            )
            node_records = [
                _node_mask_record(int(node_id), node)
                for node_id, node, _annot in node_rows
            ]
            node_annots = {int(node_id): annot for node_id, _node, annot in node_rows}
            frame_suppressed: set[int] = set()

            for anchor in anchor_records:
                scored: list[tuple[float, float, int, _MaskRecord]] = []
                for node in node_records:
                    iou = _mask_iou(anchor, node)
                    containment = _node_containment_in_anchor(anchor, node)
                    if iou > 0.0 or containment > 0.0:
                        scored.append((iou, containment, node.label_id, node))
                if not scored:
                    continue

                best_iou, _best_containment, best_id, best_node = max(
                    scored,
                    key=lambda item: (item[0], item[1]),
                )
                if best_iou < min_best_iou:
                    continue
                best_area = int(best_node.mask.sum())
                iou_cutoff = best_iou * fragment_max_iou_fraction

                for iou, containment, node_id, node in scored:
                    if node_id == best_id:
                        continue
                    if node_annots.get(node_id) == VarAnnotation.REAL:
                        continue
                    node_area = int(node.mask.sum())
                    if (
                        containment >= min_fragment_containment
                        and iou <= iou_cutoff
                        and node_area < best_area
                    ):
                        frame_suppressed.add(node_id)

            if frame_suppressed:
                session.query(NodeDB).where(NodeDB.id.in_(frame_suppressed)).update(
                    {NodeDB.node_annot: VarAnnotation.FAKE},
                    synchronize_session=False,
                )
                suppressed.update(frame_suppressed)
                by_frame[t] = len(frame_suppressed)

        session.commit()

    return AnchorSuppressionReport(
        frame_index=int(frame_index),
        neighbor_offsets=tuple(int(offset) for offset in neighbor_offsets),
        suppressed_node_ids=sorted(suppressed),
        by_frame=dict(sorted(by_frame.items())),
    )
