"""Inject validated tracked labels as annotated Ultrack nodes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig


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


def _node_bbox_and_mask(node_id: int, node) -> tuple[tuple[int, int, int, int], np.ndarray]:
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


def _intersects(
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


def _overlap_pair(lhs_id: int, rhs_id: int) -> tuple[int, int]:
    return (max(lhs_id, rhs_id), min(lhs_id, rhs_id))


def _generate_node_id(index: int, time: int, max_segments: int) -> int:
    return index + (time + 1) * max_segments


def _make_node_pickle(t: int, mask_2d: np.ndarray, bbox: np.ndarray, node_id: int) -> bytes:
    from ultrack.core.segmentation.node import Node

    min_y, min_x, max_y, max_x = bbox
    bbox_2d = np.array([int(min_y), int(min_x), int(max_y), int(max_x)], dtype=np.int64)
    node = Node.from_mask(time=t, mask=np.asarray(mask_2d, dtype=bool), bbox=bbox_2d, node_id=node_id)
    return pickle.dumps(node)


def inject_validated_nodes(
    working_dir: str | Path,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
    cfg: TrackingConfig,
) -> ValidationInjectionReport:
    """Insert validated masks into Ultrack's DB as fixed REAL nodes.

    Existing candidates in the same frame that overlap a validated mask are
    marked FAKE and paired with the injected node in OverlapDB.
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

    with Session(engine) as session:
        next_t_node_id: dict[int, int] = {}
        for t in {record.t for record in records}:
            max_t_node_id = (
                session.query(sqla.func.max(NodeDB.t_node_id))
                .where(NodeDB.t == t)
                .scalar()
            )
            next_t_node_id[t] = int(max_t_node_id or 0) + 1

        for record in records:
            t_node_id = next_t_node_id[record.t]
            next_t_node_id[record.t] += 1
            node_id = _generate_node_id(t_node_id, record.t, cfg.max_segments_per_time)
            bbox_arr = np.asarray(record.bbox, dtype=np.int32)

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
                    pickle=_make_node_pickle(record.t, record.mask, bbox_arr, node_id),
                    node_prob=1.0,
                    node_annot=VarAnnotation.REAL,
                )
            )
            inserted += 1

            candidates = (
                session.query(NodeDB.id, NodeDB.pickle)
                .where(NodeDB.t == record.t)
                .where(NodeDB.t_hier_id != 0)
                .all()
            )
            for candidate_id, candidate_node in candidates:
                candidate_bbox, candidate_mask = _node_bbox_and_mask(candidate_id, candidate_node)
                if not _intersects(record.bbox, record.mask, candidate_bbox, candidate_mask):
                    continue
                faked_ids.add(int(candidate_id))
                overlap_pairs.add(_overlap_pair(node_id, int(candidate_id)))

        if faked_ids:
            session.query(NodeDB).where(NodeDB.id.in_(faked_ids)).update(
                {NodeDB.node_annot: VarAnnotation.FAKE},
                synchronize_session=False,
            )

        existing_pairs: set[tuple[int, int]] = set()
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
