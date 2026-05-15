"""Inject validated tracked labels as annotated Ultrack nodes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack._node_geometry import (
    intersects,
    make_node_pickle,
    node_bbox_and_mask,
    node_pickle_ndim,
    raw_iou,
)


@dataclass(frozen=True)
class ValidationInjectionReport:
    inserted: int
    skipped_missing: int
    skipped: list[tuple[int, int]]
    faked: int
    overlaps_added: int


@dataclass(frozen=True)
class _MaskRecord:
    cell_id: int
    t: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area: int
    y: float
    x: float


def _frame_mask_for_cell(tracked_labels: np.ndarray, t: int, cell_id: int) -> np.ndarray:
    frame = np.asarray(tracked_labels)[t]
    if frame.ndim == 2:
        return np.asarray(frame == cell_id)
    if frame.ndim == 3:
        return np.asarray(frame == cell_id).any(axis=0)
    raise ValueError(f"Expected tracked frame to be 2D or 3D, got shape {frame.shape}")


def _validated_mask_records(
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> tuple[list[_MaskRecord], list[tuple[int, int]]]:
    records: list[_MaskRecord] = []
    skipped: list[tuple[int, int]] = []
    labels = np.asarray(tracked_labels)
    n_frames = int(labels.shape[0])

    for cell_id, frames in sorted(validated_tracks.items()):
        for raw_t in sorted(frames):
            t = int(raw_t)
            if t < 0 or t >= n_frames:
                skipped.append((int(cell_id), t))
                continue

            mask_2d = _frame_mask_for_cell(labels, t, int(cell_id))
            if not mask_2d.any():
                skipped.append((int(cell_id), t))
                continue

            rows = np.flatnonzero(mask_2d.any(axis=1))
            cols = np.flatnonzero(mask_2d.any(axis=0))
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
            crop = np.ascontiguousarray(mask_2d[y0:y1, x0:x1], dtype=bool)
            ys, xs = np.nonzero(crop)
            records.append(
                _MaskRecord(
                    cell_id=int(cell_id),
                    t=t,
                    bbox=(y0, x0, y1, x1),
                    mask=crop,
                    area=int(crop.sum()),
                    y=float(y0 + ys.mean()),
                    x=float(x0 + xs.mean()),
                )
            )

    return records, skipped


def _overlap_pair(lhs_id: int, rhs_id: int) -> tuple[int, int]:
    return (max(lhs_id, rhs_id), min(lhs_id, rhs_id))


def _generate_node_id(index: int, time: int, max_segments: int) -> int:
    return index + (time + 1) * max_segments


def _best_iou_assignments(
    records: list[_MaskRecord],
    candidates: dict[int, tuple[tuple[int, int, int, int], np.ndarray, int]],
) -> dict[int, int]:
    pairs: list[tuple[float, int, int, int, int]] = []
    for record_index, record in enumerate(records):
        for candidate_id, (candidate_bbox, candidate_mask, _ndim) in candidates.items():
            iou = raw_iou(record.bbox, record.mask, candidate_bbox, candidate_mask)
            pairs.append((-iou, record_index, int(candidate_id), record.cell_id, record.t))

    pairs.sort()
    assigned_records: set[int] = set()
    assigned_candidates: set[int] = set()
    assignments: dict[int, int] = {}
    for _neg_iou, record_index, candidate_id, _cell_id, _t in pairs:
        if record_index in assigned_records or candidate_id in assigned_candidates:
            continue
        assignments[record_index] = candidate_id
        assigned_records.add(record_index)
        assigned_candidates.add(candidate_id)
        if len(assigned_records) == min(len(records), len(candidates)):
            break
    return assignments


def inject_validated_nodes(
    working_dir: str | Path,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
    cfg: TrackingConfig,
) -> ValidationInjectionReport:
    """Replace best-matching candidates with validated masks as fixed REAL nodes.

    The best same-frame candidate by raw IoU is updated in place so its
    hierarchy placement and temporal links are preserved. If no candidate is
    available for a validated mask, a reserved REAL node is inserted instead.
    Other candidates in the same frame that overlap a validated mask are marked
    FAKE and paired with the REAL node in OverlapDB.
    """
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, OverlapDB, VarAnnotation

    records, skipped = _validated_mask_records(validated_tracks, tracked_labels)
    if not records:
        return ValidationInjectionReport(
            inserted=0,
            skipped_missing=len(skipped),
            skipped=skipped,
            faked=0,
            overlaps_added=0,
        )

    working_dir = Path(working_dir)
    engine = sqla.create_engine(f"sqlite:///{working_dir / 'data.db'}")
    inserted = 0
    faked_ids: set[int] = set()
    overlap_pairs: set[tuple[int, int]] = set()
    real_node_ids: set[int] = set()

    with Session(engine) as session:
        sample_node = session.query(NodeDB.pickle).limit(1).scalar()
        fallback_ndim = node_pickle_ndim(sample_node) if sample_node is not None else 2
        next_t_node_id: dict[int, int] = {}
        for t in {record.t for record in records}:
            max_t_node_id = (
                session.query(sqla.func.max(NodeDB.t_node_id))
                .where(NodeDB.t == t)
                .scalar()
            )
            next_t_node_id[t] = int(max_t_node_id or 0) + 1

        records_by_t: dict[int, list[tuple[int, _MaskRecord]]] = {}
        for index, record in enumerate(records):
            records_by_t.setdefault(record.t, []).append((index, record))

        for t, indexed_records in records_by_t.items():
            candidate_rows = (
                session.query(NodeDB.id, NodeDB.pickle)
                .where(NodeDB.t == t)
                .where(NodeDB.t_hier_id != 0)
                .all()
            )
            candidates: dict[int, tuple[tuple[int, int, int, int], np.ndarray, int]] = {}
            for candidate_id, candidate_node in candidate_rows:
                candidate_bbox, candidate_mask = node_bbox_and_mask(
                    int(candidate_id), candidate_node
                )
                candidates[int(candidate_id)] = (
                    candidate_bbox,
                    candidate_mask,
                    node_pickle_ndim(candidate_node),
                )

            frame_records = [record for _index, record in indexed_records]
            local_assignments = _best_iou_assignments(frame_records, candidates)
            real_node_by_local_index: dict[int, int] = {}
            matched_candidate_ids: set[int] = set()

            for local_index, record in enumerate(frame_records):
                bbox_arr = np.asarray(record.bbox, dtype=np.int32)
                candidate_id = local_assignments.get(local_index)
                if candidate_id is None:
                    node_ndim = next(
                        (ndim for _bbox, _mask, ndim in candidates.values()),
                        fallback_ndim,
                    )
                    t_node_id = next_t_node_id[record.t]
                    next_t_node_id[record.t] += 1
                    node_id = _generate_node_id(
                        t_node_id, record.t, cfg.max_segments_per_time
                    )
                    session.add(
                        NodeDB(
                            id=node_id,
                            t=record.t,
                            t_node_id=t_node_id,
                            t_hier_id=0,
                            z=0,
                            y=record.y,
                            x=record.x,
                            area=record.area,
                            pickle=make_node_pickle(
                                record.t,
                                record.mask,
                                bbox_arr,
                                node_id,
                                ndim=node_ndim,
                            ),
                            node_prob=1.0,
                            node_annot=VarAnnotation.REAL,
                        )
                    )
                    inserted += 1
                    real_node_ids.add(node_id)
                    real_node_by_local_index[local_index] = node_id
                    continue

                node_ndim = candidates[candidate_id][2]
                session.query(NodeDB).where(NodeDB.id == candidate_id).update(
                    {
                        NodeDB.y: record.y,
                        NodeDB.x: record.x,
                        NodeDB.area: record.area,
                        NodeDB.pickle: make_node_pickle(
                            record.t,
                            record.mask,
                            bbox_arr,
                            candidate_id,
                            ndim=node_ndim,
                        ),
                        NodeDB.node_prob: 1.0,
                        NodeDB.node_annot: VarAnnotation.REAL,
                    },
                    synchronize_session=False,
                )
                inserted += 1
                real_node_ids.add(candidate_id)
                matched_candidate_ids.add(candidate_id)
                real_node_by_local_index[local_index] = candidate_id

            for local_index, record in enumerate(frame_records):
                real_node_id = real_node_by_local_index[local_index]
                for candidate_id, (
                    candidate_bbox,
                    candidate_mask,
                    _ndim,
                ) in candidates.items():
                    if candidate_id in matched_candidate_ids:
                        continue
                    if not intersects(
                        record.bbox, record.mask, candidate_bbox, candidate_mask
                    ):
                        continue
                    faked_ids.add(candidate_id)
                    overlap_pairs.add(_overlap_pair(real_node_id, candidate_id))

        if faked_ids:
            session.query(NodeDB).where(NodeDB.id.in_(faked_ids)).update(
                {NodeDB.node_annot: VarAnnotation.FAKE},
                synchronize_session=False,
            )

        existing_pairs: set[tuple[int, int]] = set()
        if real_node_ids:
            session.query(OverlapDB).where(
                sqla.or_(
                    OverlapDB.node_id.in_(real_node_ids),
                    OverlapDB.ancestor_id.in_(real_node_ids),
                )
            ).delete(synchronize_session=False)
        if overlap_pairs:
            node_ids = {pair[0] for pair in overlap_pairs} | {pair[1] for pair in overlap_pairs}
            existing_pairs = {
                (int(row.node_id), int(row.ancestor_id))
                for row in session.query(OverlapDB)
                .where(
                    sqla.or_(
                        OverlapDB.node_id.in_(node_ids),
                        OverlapDB.ancestor_id.in_(node_ids),
                    )
                )
                .all()
            }
            for node_id, ancestor_id in sorted(overlap_pairs - existing_pairs):
                session.add(OverlapDB(node_id=node_id, ancestor_id=ancestor_id))

        session.commit()

    engine.dispose()
    added_pairs = len(overlap_pairs - existing_pairs)
    return ValidationInjectionReport(
        inserted=inserted,
        skipped_missing=len(skipped),
        skipped=skipped,
        faked=len(faked_ids),
        overlaps_added=added_pairs,
    )
