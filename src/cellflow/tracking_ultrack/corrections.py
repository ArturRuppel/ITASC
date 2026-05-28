"""Per-frame correction primitives for Ultrack solves and exports."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig

LOG = logging.getLogger(__name__)


CorrectionKind = Literal["validated", "anchor"]


@dataclass(frozen=True)
class Correction:
    cell_id: int
    t: int
    kind: CorrectionKind
    y: float
    x: float

    def __post_init__(self) -> None:
        if self.kind not in {"validated", "anchor"}:
            raise ValueError(f"Unknown correction kind: {self.kind!r}")


@dataclass(frozen=True)
class CorrectionDatabaseReport:
    fake_nodes: int = 0
    anchor_nodes: int = 0
    anchor_links: int = 0
    anchor_overlaps_pruned: int = 0
    unmatched_anchors: tuple[Correction, ...] = ()


@dataclass(frozen=True)
class AnchorIncidentLinkReport:
    inserted: int = 0
    anchors_processed: int = 0


@dataclass(frozen=True)
class AnchorTailLinkReport:
    annotated: int = 0
    skipped_no_anchor: int = 0
    skipped_no_link: int = 0


@dataclass(frozen=True)
class PostSolveCorrectionReport:
    remapped_anchor_tracks: int = 0
    stamped_anchors: int = 0
    pasted_validated: int = 0


def _distance(y0: float, x0: float, y1: float, x1: float) -> float:
    return float(np.hypot(float(y0) - float(y1), float(x0) - float(x1)))


def corrections_from_validated_tracks(
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> list[Correction]:
    """Convert the legacy validated-track store to validated corrections."""
    corrections: list[Correction] = []
    labels = np.asarray(tracked_labels)
    for raw_cell_id, frames in sorted(validated_tracks.items()):
        cell_id = int(raw_cell_id)
        for raw_t in sorted(frames):
            t = int(raw_t)
            if t < 0 or t >= labels.shape[0]:
                continue
            centroid = _frame_centroid(np.asarray(labels[t] == cell_id))
            if centroid is None:
                continue
            y, x = centroid
            corrections.append(Correction(cell_id=cell_id, t=t, kind="validated", y=y, x=x))
    return corrections


def apply_corrections_to_database(
    working_dir: str | Path,
    corrections: list[Correction],
    cfg: TrackingConfig,
    *,
    annotate_anchor_links: bool = True,
    tracked_labels: np.ndarray | None = None,
) -> CorrectionDatabaseReport:
    """Apply solve-time correction annotations to ``data.db``.

    Validated frames mark nearby candidates ``FAKE``. Anchors mark the nearest
    surviving candidate ``REAL`` and mark consecutive anchor links ``REAL``.
    In Ultrack 0.6.x these are hard annotations when solve uses
    ``use_annotations=True``.

    ``tracked_labels`` enables IoU-based candidate matching for anchors. When
    provided, the mask of the anchored cell in ``tracked_labels`` is compared
    against NodeDB candidate masks, and the candidate with the highest IoU is
    chosen. This correctly identifies the intended hypothesis even when two
    hierarchical candidates (e.g. parent vs child) have nearly identical
    centroids. Falls back to centroid-distance matching when no IoU > 0 is
    found.
    """
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, NodeDB, OverlapDB, VarAnnotation

    if not corrections:
        return CorrectionDatabaseReport()

    radius = float(cfg.anchor_radius_px)
    engine = sqla.create_engine(f"sqlite:///{Path(working_dir) / 'data.db'}")
    fake_node_ids: set[int] = set()
    resolved_anchor_nodes: dict[tuple[int, int], int] = {}
    unmatched_anchor_list: list[Correction] = []
    anchor_nodes = 0
    anchor_links = 0
    anchor_overlaps_pruned = 0

    with Session(engine) as session:
        for correction in corrections:
            if correction.kind != "validated":
                continue
            rows = session.query(NodeDB.id, NodeDB.y, NodeDB.x).where(
                NodeDB.t == int(correction.t)
            )
            for node_id, y, x in rows:
                if _distance(y, x, correction.y, correction.x) <= radius:
                    fake_node_ids.add(int(node_id))

        if fake_node_ids:
            session.query(NodeDB).where(NodeDB.id.in_(sorted(fake_node_ids))).update(
                {NodeDB.node_annot: VarAnnotation.FAKE},
                synchronize_session=False,
            )

        labels_arr = np.asarray(tracked_labels) if tracked_labels is not None else None

        for correction in corrections:
            if correction.kind != "anchor":
                continue

            nearest: tuple[float, int] | None = None

            # IoU-based matching: find the candidate whose mask best overlaps
            # the cell the user currently has in tracked_labels. This is the
            # correct approach when two hierarchical hypotheses (e.g. parent
            # mask vs smaller child mask) share nearly the same centroid —
            # centroid distance cannot distinguish them, but IoU can.
            if labels_arr is not None:
                t_idx = int(correction.t)
                if 0 <= t_idx < labels_arr.shape[0]:
                    t_frame = labels_arr[t_idx]
                    raw_mask = (
                        (t_frame == int(correction.cell_id)).any(axis=0)
                        if t_frame.ndim == 3
                        else (t_frame == int(correction.cell_id))
                    )
                    if raw_mask.any():
                        from cellflow.tracking_ultrack._node_geometry import (
                            node_bbox_and_mask as _nbm,
                            raw_iou as _raw_iou,
                        )
                        rows_nz = np.flatnonzero(raw_mask.any(axis=1))
                        cols_nz = np.flatnonzero(raw_mask.any(axis=0))
                        y0, y1 = int(rows_nz[0]), int(rows_nz[-1]) + 1
                        x0, x1 = int(cols_nz[0]), int(cols_nz[-1]) + 1
                        anchor_bbox = (y0, x0, y1, x1)
                        anchor_crop = np.ascontiguousarray(raw_mask[y0:y1, x0:x1], dtype=bool)
                        broad = radius * 4
                        best_iou = 0.0
                        best_iou_node: int | None = None
                        for node_id, y, x, node_pickle in session.query(
                            NodeDB.id, NodeDB.y, NodeDB.x, NodeDB.pickle
                        ).where(
                            NodeDB.t == t_idx,
                            NodeDB.node_annot != VarAnnotation.FAKE,
                        ):
                            if _distance(y, x, correction.y, correction.x) > broad:
                                continue
                            cand_bbox, cand_mask = _nbm(int(node_id), node_pickle)
                            iou = _raw_iou(anchor_bbox, anchor_crop, cand_bbox, cand_mask)
                            if iou > best_iou:
                                best_iou = iou
                                best_iou_node = int(node_id)
                        if best_iou_node is not None:
                            nearest = (0.0, best_iou_node)

            # Centroid-distance fallback (used when tracked_labels unavailable
            # or the cell has no mask at this frame in tracked_labels).
            if nearest is None:
                for node_id, y, x in session.query(NodeDB.id, NodeDB.y, NodeDB.x).where(
                    NodeDB.t == int(correction.t),
                    NodeDB.node_annot != VarAnnotation.FAKE,
                ):
                    dist = _distance(y, x, correction.y, correction.x)
                    if dist <= radius and (nearest is None or dist < nearest[0]):
                        nearest = (dist, int(node_id))

            if nearest is None:
                unmatched_anchor_list.append(correction)
                continue
            _dist, node_id = nearest
            session.query(NodeDB).where(NodeDB.id == node_id).update(
                {NodeDB.node_annot: VarAnnotation.REAL},
                synchronize_session=False,
            )
            resolved_anchor_nodes[(int(correction.cell_id), int(correction.t))] = node_id
            anchor_nodes += 1

        # Prune OverlapDB rows where both endpoints are anchor-forced REAL nodes.
        # Two anchors at the same frame can land on hierarchical siblings; without
        # this the ILP is infeasible (nodes_X + nodes_Y <= 1 vs both x >= 1).
        anchor_real_ids = set(resolved_anchor_nodes.values())
        if len(anchor_real_ids) >= 2:
            anchor_overlaps_pruned = (
                session.query(OverlapDB)
                .where(
                    OverlapDB.node_id.in_(anchor_real_ids),
                    OverlapDB.ancestor_id.in_(anchor_real_ids),
                )
                .delete(synchronize_session=False)
            )

        if annotate_anchor_links:
            for cell_id, t in sorted(resolved_anchor_nodes):
                source_id = resolved_anchor_nodes[(cell_id, t)]
                target_id = resolved_anchor_nodes.get((cell_id, t + 1))
                if target_id is None:
                    continue
                link = session.query(LinkDB).where(
                    LinkDB.source_id == source_id,
                    LinkDB.target_id == target_id,
                ).one_or_none()
                if link is None:
                    session.add(
                        LinkDB(
                            source_id=source_id,
                            target_id=target_id,
                            weight=0.0,
                            annotation=VarAnnotation.REAL,
                        )
                    )
                else:
                    link.annotation = VarAnnotation.REAL
                anchor_links += 1

        session.commit()

    engine.dispose()
    return CorrectionDatabaseReport(
        fake_nodes=len(fake_node_ids),
        anchor_nodes=anchor_nodes,
        anchor_links=anchor_links,
        anchor_overlaps_pruned=int(anchor_overlaps_pruned or 0),
        unmatched_anchors=tuple(unmatched_anchor_list),
    )


def ensure_anchor_incident_links(
    working_dir: str | Path,
    cfg: TrackingConfig,
) -> AnchorIncidentLinkReport:
    """Insert missing LinkDB rows for edges incident to anchor (REAL) nodes.

    The linker keeps only the top ``cfg.max_neighbors`` source-side edges per
    target. A node the user selects as an anchor after solving may have been
    pruned out of LinkDB entirely for some — or all — adjacent-frame
    candidates, leaving the solver no way to extend the track from it. This
    fills those gaps for anchor-incident edges, using the same per-pair weight
    formula as the active linker mode.
    """
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, NodeDB, VarAnnotation

    from cellflow.tracking_ultrack.linking import compute_edge_weight

    engine = sqla.create_engine(f"sqlite:///{Path(working_dir) / 'data.db'}")

    inserted = 0
    anchors_processed = 0

    with Session(engine) as session:
        duplicate_groups: dict[tuple[int, int], list[LinkDB]] = {}
        for link in session.query(LinkDB).order_by(
            LinkDB.source_id,
            LinkDB.target_id,
            LinkDB.id,
        ):
            duplicate_groups.setdefault(
                (int(link.source_id), int(link.target_id)),
                [],
            ).append(link)
        for links in duplicate_groups.values():
            if len(links) <= 1:
                continue
            keep = max(
                links,
                key=lambda link: (
                    2
                    if link.annotation == VarAnnotation.REAL
                    else 1
                    if link.annotation == VarAnnotation.FAKE
                    else 0,
                    float(link.weight) if link.weight is not None else float("-inf"),
                    -int(link.id),
                ),
            )
            for link in links:
                if int(link.id) == int(keep.id):
                    continue
                session.delete(link)
        session.flush()

        anchor_rows = session.query(
            NodeDB.id, NodeDB.t, NodeDB.y, NodeDB.x, NodeDB.pickle,
        ).where(NodeDB.node_annot == VarAnnotation.REAL).all()

        if not anchor_rows:
            session.commit()
            engine.dispose()
            return AnchorIncidentLinkReport()

        max_distance = float(cfg.max_distance)
        existing_pairs = {
            (int(source_id), int(target_id))
            for source_id, target_id in session.query(
                LinkDB.source_id, LinkDB.target_id
            )
        }
        new_rows: list[LinkDB] = []

        for anchor_id, anchor_t, anchor_y, anchor_x, anchor_pickle in anchor_rows:
            anchor_id = int(anchor_id)
            anchor_t = int(anchor_t)
            anchor_node = anchor_pickle
            anchors_processed += 1

            for direction in (-1, +1):
                neighbor_t = anchor_t + direction
                neighbor_rows = session.query(
                    NodeDB.id, NodeDB.y, NodeDB.x, NodeDB.pickle,
                ).where(NodeDB.t == neighbor_t).all()
                if not neighbor_rows:
                    continue

                for neigh_id, neigh_y, neigh_x, neigh_pickle in neighbor_rows:
                    neigh_id = int(neigh_id)
                    dist = _distance(anchor_y, anchor_x, neigh_y, neigh_x)
                    if dist > max_distance:
                        continue

                    if direction == +1:
                        source_id, target_id = anchor_id, neigh_id
                        source_node, target_node = anchor_node, neigh_pickle
                    else:
                        source_id, target_id = neigh_id, anchor_id
                        source_node, target_node = neigh_pickle, anchor_node

                    pair = (int(source_id), int(target_id))
                    if pair in existing_pairs:
                        continue

                    weight = compute_edge_weight(source_node, target_node, dist, cfg)
                    if weight is None:
                        continue

                    new_rows.append(
                        LinkDB(
                            source_id=source_id,
                            target_id=target_id,
                            weight=float(weight),
                        )
                    )
                    existing_pairs.add(pair)

        if new_rows:
            session.add_all(new_rows)
            inserted = len(new_rows)
        session.commit()

    engine.dispose()
    return AnchorIncidentLinkReport(
        inserted=inserted,
        anchors_processed=anchors_processed,
    )


def _tracked_mask_bbox(
    tracked_labels: np.ndarray,
    t: int,
    cell_id: int,
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    if t < 0 or t >= tracked_labels.shape[0]:
        return None
    frame = np.asarray(tracked_labels[t])
    mask = (
        (frame == int(cell_id)).any(axis=0)
        if frame.ndim == 3
        else (frame == int(cell_id))
    )
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return (
        (y0, x0, y1, x1),
        np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=bool),
    )


def _best_real_anchor_node_id(
    session,
    correction: Correction,
    cfg: TrackingConfig,
    tracked_labels: np.ndarray | None,
) -> int | None:
    from ultrack.core.database import NodeDB, VarAnnotation

    labels_arr = np.asarray(tracked_labels) if tracked_labels is not None else None
    mask_record = (
        _tracked_mask_bbox(labels_arr, int(correction.t), int(correction.cell_id))
        if labels_arr is not None
        else None
    )

    best_iou: tuple[float, float, int] | None = None
    if mask_record is not None:
        from cellflow.tracking_ultrack._node_geometry import (
            node_bbox_and_mask as _node_bbox_and_mask,
            raw_iou as _raw_iou,
        )

        anchor_bbox, anchor_crop = mask_record
        broad_radius = float(cfg.anchor_radius_px) * 4.0
        rows = session.query(NodeDB.id, NodeDB.y, NodeDB.x, NodeDB.pickle).where(
            NodeDB.t == int(correction.t),
            NodeDB.node_annot == VarAnnotation.REAL,
        )
        for node_id, y, x, node_pickle in rows:
            dist = _distance(y, x, correction.y, correction.x)
            if dist > broad_radius:
                continue
            cand_bbox, cand_mask = _node_bbox_and_mask(int(node_id), node_pickle)
            iou = _raw_iou(anchor_bbox, anchor_crop, cand_bbox, cand_mask)
            candidate = (float(iou), -float(dist), int(node_id))
            if best_iou is None or candidate > best_iou:
                best_iou = candidate
        if best_iou is not None and best_iou[0] > 0:
            return best_iou[2]

    nearest: tuple[float, int] | None = None
    rows = session.query(NodeDB.id, NodeDB.y, NodeDB.x).where(
        NodeDB.t == int(correction.t),
        NodeDB.node_annot == VarAnnotation.REAL,
    )
    for node_id, y, x in rows:
        dist = _distance(y, x, correction.y, correction.x)
        if dist <= float(cfg.anchor_radius_px) and (
            nearest is None or dist < nearest[0]
        ):
            nearest = (float(dist), int(node_id))
    return None if nearest is None else nearest[1]


def _correction_protected_mask(
    corrections: list[Correction],
    tracked_labels: np.ndarray | None,
    *,
    t: int,
    source_cell_id: int,
) -> np.ndarray | None:
    if tracked_labels is None:
        return None
    labels = np.asarray(tracked_labels)
    if t < 0 or t >= labels.shape[0]:
        return None

    protected_ids = {
        int(c.cell_id)
        for c in corrections
        if int(c.t) == int(t) and int(c.cell_id) != int(source_cell_id)
    }
    if not protected_ids:
        return None

    frame = labels[t]
    mask = np.isin(frame, list(protected_ids))
    if mask.ndim == 3:
        mask = mask.any(axis=0)
    return np.asarray(mask, dtype=bool)


def annotate_anchor_tail_links(
    working_dir: str | Path,
    corrections: list[Correction],
    cfg: TrackingConfig,
    *,
    tracked_labels: np.ndarray | None = None,
) -> AnchorTailLinkReport:
    """Force best predecessor and successor links around anchored tracks.

    Consecutive anchor links constrain only the user-confirmed frames. This
    helper walks the best positive same-linker chain backward from the first
    anchor and forward from the last anchor of each cell, mirroring the greedy
    extender while still letting the ILP handle conflicts with other annotated
    cells.
    """
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, NodeDB, OverlapDB, VarAnnotation

    anchor_by_cell: dict[int, list[Correction]] = {}
    for correction in corrections:
        if correction.kind == "anchor":
            anchor_by_cell.setdefault(int(correction.cell_id), []).append(correction)
    if not anchor_by_cell:
        return AnchorTailLinkReport()

    engine = sqla.create_engine(f"sqlite:///{Path(working_dir) / 'data.db'}")
    annotated = skipped_no_anchor = skipped_no_link = 0
    labels_arr = np.asarray(tracked_labels) if tracked_labels is not None else None

    with Session(engine) as session:
        real_source_ids: set[int] = set()
        real_target_ids: set[int] = set()
        forced_node_ids_by_t: dict[int, set[int]] = {}
        for node_id, t in session.query(NodeDB.id, NodeDB.t).where(
            NodeDB.node_annot == VarAnnotation.REAL
        ):
            forced_node_ids_by_t.setdefault(int(t), set()).add(int(node_id))
        for source_id, target_id in session.query(
            LinkDB.source_id,
            LinkDB.target_id,
        ).where(LinkDB.annotation == VarAnnotation.REAL):
            source_id = int(source_id)
            target_id = int(target_id)
            real_source_ids.add(source_id)
            real_target_ids.add(target_id)
            for node_id, t in session.query(NodeDB.id, NodeDB.t).where(
                NodeDB.id.in_([source_id, target_id])
            ):
                forced_node_ids_by_t.setdefault(int(t), set()).add(int(node_id))

        for cell_id, cell_anchors in sorted(anchor_by_cell.items()):
            head = min(cell_anchors, key=lambda item: int(item.t))
            source_id = _best_real_anchor_node_id(session, head, cfg, labels_arr)
            if source_id is None:
                skipped_no_anchor += 1
            else:
                seen_sources: set[int] = set()
                annotated_for_cell = 0
                while source_id not in seen_sources:
                    seen_sources.add(int(source_id))
                    source = (
                        session.query(NodeDB)
                        .where(NodeDB.id == source_id)
                        .one_or_none()
                    )
                    if source is None:
                        if annotated_for_cell == 0:
                            skipped_no_anchor += 1
                        break
                    if int(source_id) in real_target_ids:
                        break
                    predecessor_t = int(source.t) - 1
                    protected_mask = _correction_protected_mask(
                        corrections,
                        labels_arr,
                        t=predecessor_t,
                        source_cell_id=cell_id,
                    )

                    candidates = []
                    rows = (
                        session.query(LinkDB, NodeDB)
                        .join(NodeDB, NodeDB.id == LinkDB.source_id)
                        .where(
                            LinkDB.target_id == int(source_id),
                            NodeDB.t == predecessor_t,
                            NodeDB.node_annot != VarAnnotation.FAKE,
                        )
                        .all()
                    )
                    for link, predecessor in rows:
                        if link.weight is None or float(link.weight) <= 0:
                            continue
                        if int(predecessor.id) in real_source_ids:
                            continue
                        forced_ids = forced_node_ids_by_t.get(
                            int(predecessor.t),
                            set(),
                        )
                        if forced_ids:
                            overlap = (
                                session.query(OverlapDB.node_id)
                                .where(
                                    sqla.or_(
                                        sqla.and_(
                                            OverlapDB.node_id == int(predecessor.id),
                                            OverlapDB.ancestor_id.in_(forced_ids),
                                        ),
                                        sqla.and_(
                                            OverlapDB.ancestor_id
                                            == int(predecessor.id),
                                            OverlapDB.node_id.in_(forced_ids),
                                        ),
                                    )
                                )
                                .first()
                            )
                            if overlap is not None:
                                continue
                        if protected_mask is not None:
                            from cellflow.tracking_ultrack._node_geometry import (
                                node_bbox_and_mask as _node_bbox_and_mask,
                            )

                            (y0, x0, y1, x1), predecessor_crop = _node_bbox_and_mask(
                                int(predecessor.id),
                                predecessor.pickle,
                            )
                            if np.any(
                                predecessor_crop & protected_mask[y0:y1, x0:x1]
                            ):
                                continue
                        candidates.append(
                            (
                                float(link.weight),
                                -_distance(
                                    source.y,
                                    source.x,
                                    predecessor.y,
                                    predecessor.x,
                                ),
                                int(predecessor.id),
                                link,
                            )
                        )

                    if not candidates:
                        if annotated_for_cell == 0:
                            skipped_no_link += 1
                        break

                    _weight, _neg_dist, next_source_id, best_link = max(
                        candidates, key=lambda item: (item[0], item[1])
                    )
                    best_link.annotation = VarAnnotation.REAL
                    real_source_ids.add(int(next_source_id))
                    real_target_ids.add(int(source_id))
                    forced_node_ids_by_t.setdefault(predecessor_t, set()).add(
                        int(next_source_id)
                    )
                    annotated += 1
                    annotated_for_cell += 1
                    source_id = next_source_id

            tail = max(cell_anchors, key=lambda item: int(item.t))
            source_id = _best_real_anchor_node_id(session, tail, cfg, labels_arr)
            if source_id is None:
                skipped_no_anchor += 1
                continue

            seen_sources: set[int] = set()
            annotated_for_cell = 0
            while source_id not in seen_sources:
                seen_sources.add(int(source_id))
                source = (
                    session.query(NodeDB).where(NodeDB.id == source_id).one_or_none()
                )
                if source is None:
                    if annotated_for_cell == 0:
                        skipped_no_anchor += 1
                    break
                if int(source_id) in real_source_ids:
                    break
                target_t = int(source.t) + 1
                protected_mask = _correction_protected_mask(
                    corrections,
                    labels_arr,
                    t=target_t,
                    source_cell_id=cell_id,
                )

                candidates = []
                rows = (
                    session.query(LinkDB, NodeDB)
                    .join(NodeDB, NodeDB.id == LinkDB.target_id)
                    .where(
                        LinkDB.source_id == int(source_id),
                        NodeDB.t == target_t,
                        NodeDB.node_annot != VarAnnotation.FAKE,
                    )
                    .all()
                )
                for link, target in rows:
                    if link.weight is None or float(link.weight) <= 0:
                        continue
                    if int(target.id) in real_target_ids:
                        continue
                    forced_ids = forced_node_ids_by_t.get(int(target.t), set())
                    if forced_ids:
                        overlap = (
                            session.query(OverlapDB.node_id)
                            .where(
                                sqla.or_(
                                    sqla.and_(
                                        OverlapDB.node_id == int(target.id),
                                        OverlapDB.ancestor_id.in_(forced_ids),
                                    ),
                                    sqla.and_(
                                        OverlapDB.ancestor_id == int(target.id),
                                        OverlapDB.node_id.in_(forced_ids),
                                    ),
                                )
                            )
                            .first()
                        )
                        if overlap is not None:
                            continue
                    if protected_mask is not None:
                        from cellflow.tracking_ultrack._node_geometry import (
                            node_bbox_and_mask as _node_bbox_and_mask,
                        )

                        (y0, x0, y1, x1), target_crop = _node_bbox_and_mask(
                            int(target.id), target.pickle
                        )
                        if np.any(target_crop & protected_mask[y0:y1, x0:x1]):
                            continue
                    candidates.append(
                        (
                            float(link.weight),
                            -_distance(source.y, source.x, target.y, target.x),
                            int(target.id),
                            link,
                        )
                    )

                if not candidates:
                    if annotated_for_cell == 0:
                        skipped_no_link += 1
                    break

                _weight, _neg_dist, next_source_id, best_link = max(
                    candidates, key=lambda item: (item[0], item[1])
                )
                best_link.annotation = VarAnnotation.REAL
                real_source_ids.add(int(source_id))
                real_target_ids.add(int(next_source_id))
                forced_node_ids_by_t.setdefault(target_t, set()).add(
                    int(next_source_id)
                )
                annotated += 1
                annotated_for_cell += 1
                source_id = next_source_id

        session.commit()

    engine.dispose()
    return AnchorTailLinkReport(
        annotated=annotated,
        skipped_no_anchor=skipped_no_anchor,
        skipped_no_link=skipped_no_link,
    )


@dataclass(frozen=True)
class HomemadeAnchorInjectionReport:
    injected: int = 0
    skipped_no_mask: int = 0
    skipped_overflow: int = 0


def inject_unmatched_anchor_nodes(
    working_dir: str | Path,
    unmatched_anchors: tuple[Correction, ...],
    tracked_labels: np.ndarray,
    cfg: TrackingConfig,
) -> HomemadeAnchorInjectionReport:
    """Insert REAL NodeDB rows for anchor corrections that had no existing candidate.

    Called when ``apply_corrections_to_database`` found no NodeDB node within
    ``anchor_radius_px`` for one or more anchor corrections. This happens when
    the user anchors a manually-drawn cell that the segmenter never produced a
    candidate for. We extract the cell's mask from ``tracked_labels`` and
    insert a synthetic NodeDB row marked REAL so the ILP is forced to include
    it, then add OverlapDB rows with any spatially overlapping existing nodes
    so the ILP cannot simultaneously select conflicting candidates.
    """
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, OverlapDB, VarAnnotation

    from cellflow.tracking_ultrack._node_geometry import (
        intersects as _intersects,
        make_node_pickle,
        node_bbox_and_mask,
        node_pickle_ndim,
    )

    if not unmatched_anchors:
        return HomemadeAnchorInjectionReport()

    labels = np.asarray(tracked_labels)
    engine = sqla.create_engine(f"sqlite:///{Path(working_dir) / 'data.db'}")
    injected = 0
    skipped_no_mask = 0
    skipped_overflow = 0

    with Session(engine) as session:
        sample = session.query(NodeDB.pickle).limit(1).scalar()
        ndim = node_pickle_ndim(sample) if sample is not None else 2

        # Track per-frame next t_node_id so multiple injections at the same
        # frame don't collide.
        next_t_node_id: dict[int, int] = {}

        new_nodes: list[tuple[int, int, int, int, int, int, int]] = []
        new_nodes_crop: dict[int, tuple[tuple[int, int, int, int], np.ndarray]] = {}

        for correction in unmatched_anchors:
            t = int(correction.t)
            cell_id = int(correction.cell_id)

            if t < 0 or t >= labels.shape[0]:
                skipped_no_mask += 1
                continue

            frame = np.asarray(labels[t])
            mask_2d = (frame == cell_id).any(axis=0) if frame.ndim == 3 else (frame == cell_id)
            if not mask_2d.any():
                skipped_no_mask += 1
                continue

            rows_nz = np.flatnonzero(mask_2d.any(axis=1))
            cols_nz = np.flatnonzero(mask_2d.any(axis=0))
            y0, y1 = int(rows_nz[0]), int(rows_nz[-1]) + 1
            x0, x1 = int(cols_nz[0]), int(cols_nz[-1]) + 1
            crop = np.ascontiguousarray(mask_2d[y0:y1, x0:x1], dtype=bool)
            ys, xs = np.nonzero(crop)
            area = int(crop.sum())
            y_centroid = float(y0 + ys.mean())
            x_centroid = float(x0 + xs.mean())
            bbox_arr = np.array([y0, x0, y1, x1], dtype=np.int32)

            if t not in next_t_node_id:
                max_id = session.query(sqla.func.max(NodeDB.t_node_id)).where(NodeDB.t == t).scalar()
                next_t_node_id[t] = int(max_id or 0) + 1

            t_node_id = next_t_node_id[t]
            if t_node_id >= cfg.max_segments_per_time:
                skipped_overflow += 1
                continue
            next_t_node_id[t] += 1

            node_id = t_node_id + (t + 1) * cfg.max_segments_per_time
            node_pickle = make_node_pickle(t, crop, bbox_arr, node_id, ndim=ndim)

            session.add(
                NodeDB(
                    id=node_id,
                    t=t,
                    t_node_id=t_node_id,
                    t_hier_id=0,
                    z=0,
                    y=y_centroid,
                    x=x_centroid,
                    area=area,
                    pickle=node_pickle,
                    node_prob=1.0,
                    node_annot=VarAnnotation.REAL,
                )
            )
            new_nodes.append((node_id, t, y0, x0, y1, x1))
            new_nodes_crop[node_id] = ((y0, x0, y1, x1), crop)
            injected += 1

        session.flush()

        # For each injected node add OverlapDB rows with spatially conflicting
        # existing nodes so the ILP cannot select both.
        for node_id, t, y0, x0, y1, x1 in new_nodes:
            bbox_new = (y0, x0, y1, x1)
            _, crop_new = new_nodes_crop.get(node_id, (None, None))
            if crop_new is None:
                continue
            candidate_rows = (
                session.query(NodeDB.id, NodeDB.pickle)
                .where(NodeDB.t == t, NodeDB.id != node_id)
                .all()
            )
            for cand_id, cand_pickle in candidate_rows:
                cand_id = int(cand_id)
                cand_bbox, cand_mask = node_bbox_and_mask(cand_id, cand_pickle)
                if not _intersects(bbox_new, crop_new, cand_bbox, cand_mask):
                    continue
                pair_node = max(node_id, cand_id)
                pair_anc = min(node_id, cand_id)
                exists = session.query(OverlapDB.node_id).where(
                    OverlapDB.node_id == pair_node,
                    OverlapDB.ancestor_id == pair_anc,
                ).first()
                if exists is None:
                    session.add(OverlapDB(node_id=pair_node, ancestor_id=pair_anc))

        session.commit()

    engine.dispose()
    return HomemadeAnchorInjectionReport(
        injected=injected,
        skipped_no_mask=skipped_no_mask,
        skipped_overflow=skipped_overflow,
    )


def _frame_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    if mask.ndim == 3:
        mask = mask.any(axis=0)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def _track_centroid_at(labels: np.ndarray, t: int, track_id: int) -> tuple[float, float] | None:
    if t < 0 or t >= labels.shape[0]:
        return None
    return _frame_centroid(np.asarray(labels[t] == track_id))


def _track_lifetime(labels: np.ndarray, track_id: int) -> int:
    return int(sum(np.asarray(labels[t] == track_id).any() for t in range(labels.shape[0])))


def _matching_tracks_at_anchor(
    labels: np.ndarray,
    correction: Correction,
    radius: float,
) -> list[tuple[float, int, int]]:
    frame = labels[int(correction.t)]
    matches: list[tuple[float, int, int]] = []
    for raw_track_id in np.unique(frame):
        track_id = int(raw_track_id)
        if track_id == 0:
            continue
        centroid = _track_centroid_at(labels, int(correction.t), track_id)
        if centroid is None:
            continue
        y, x = centroid
        dist = _distance(y, x, correction.y, correction.x)
        if dist <= radius:
            matches.append((dist, -_track_lifetime(labels, track_id), track_id))
    matches.sort()
    return matches


def _next_fresh_id(labels: np.ndarray, reserved_ids: set[int], used_ids: set[int]) -> int:
    next_id = int(labels.max()) + 1
    while next_id in reserved_ids or next_id in used_ids or next_id == 0:
        next_id += 1
    used_ids.add(next_id)
    return next_id


def _anchor_lineage_track_remaps(
    working_dir: str | Path | None,
    corrections: list[Correction],
    cfg: TrackingConfig,
) -> dict[int, int]:
    """Return raw exported track IDs that should inherit anchored cell IDs.

    Ultrack may split a selected parent chain into multiple exported tracklets
    at branch points. REAL links express the intended identity path, so any
    selected node reachable from an anchor through REAL links should be remapped
    to that anchor's cell ID even if it received a new exported track_id.
    """
    if working_dir is None:
        return {}

    anchor_corrections = [c for c in corrections if c.kind == "anchor"]
    if not anchor_corrections:
        return {}

    try:
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import LinkDB, VarAnnotation
        from ultrack.core.export import to_tracks_layer

        from cellflow.tracking_ultrack.ingest import _build_ultrack_config
    except Exception:
        return {}

    working_dir = Path(working_dir)
    db_path = working_dir / "data.db"
    if not db_path.exists():
        return {}

    try:
        tracks_df, _graph = to_tracks_layer(_build_ultrack_config(cfg, working_dir))
    except Exception:
        return {}
    if tracks_df.empty or "id" not in tracks_df:
        return {}

    tracks_by_node = tracks_df.set_index("id", drop=False)
    selected_node_ids = {int(node_id) for node_id in tracks_by_node.index}
    if not selected_node_ids:
        return {}

    remaps: dict[int, int] = {}
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            real_links: dict[int, list[int]] = {}
            for source_id, target_id in session.query(
                LinkDB.source_id, LinkDB.target_id
            ).where(LinkDB.annotation == VarAnnotation.REAL):
                source_id = int(source_id)
                target_id = int(target_id)
                if source_id in selected_node_ids and target_id in selected_node_ids:
                    real_links.setdefault(source_id, []).append(target_id)
                    real_links.setdefault(target_id, []).append(source_id)

            for correction in anchor_corrections:
                t = int(correction.t)
                frame_rows = tracks_df[tracks_df["t"] == t]
                nearest: tuple[float, int] | None = None
                for row in frame_rows.itertuples(index=False):
                    dist = _distance(row.y, row.x, correction.y, correction.x)
                    if dist <= float(cfg.anchor_radius_px) and (
                        nearest is None or dist < nearest[0]
                    ):
                        nearest = (float(dist), int(row.id))
                if nearest is None:
                    continue

                queue = [nearest[1]]
                seen: set[int] = set()
                while queue:
                    node_id = queue.pop(0)
                    if node_id in seen:
                        continue
                    seen.add(node_id)
                    if node_id not in tracks_by_node.index:
                        continue
                    solver_track_id = int(tracks_by_node.loc[node_id, "track_id"])
                    remaps.setdefault(solver_track_id, int(correction.cell_id))
                    queue.extend(real_links.get(node_id, ()))
    finally:
        engine.dispose()

    return remaps


def _stamp_disk(labels: np.ndarray, t: int, y: float, x: float, radius: float, cell_id: int) -> None:
    if t < 0 or t >= labels.shape[0]:
        return
    frame = labels[t]
    yy, xx = np.ogrid[: frame.shape[-2], : frame.shape[-1]]
    disk = (yy - float(y)) ** 2 + (xx - float(x)) ** 2 <= float(radius) ** 2
    if frame.ndim == 2:
        frame[disk] = int(cell_id)
    elif frame.ndim == 3:
        frame[:, disk] = int(cell_id)
    else:
        raise ValueError(f"Expected exported frame to be 2D or 3D, got {frame.shape}")


def _paste_validated_mask(labels: np.ndarray, tracked_labels: np.ndarray, correction: Correction) -> bool:
    t = int(correction.t)
    cell_id = int(correction.cell_id)
    if t < 0 or t >= labels.shape[0] or t >= tracked_labels.shape[0]:
        return False
    mask = np.asarray(tracked_labels[t] == cell_id)
    if not mask.any():
        return False
    frame = labels[t]
    if mask.ndim == frame.ndim:
        frame[mask] = cell_id
    elif mask.ndim == 2 and frame.ndim == 3:
        frame[:, mask] = cell_id
    elif mask.ndim == 3 and frame.ndim == 2:
        frame[mask.any(axis=0)] = cell_id
    else:
        raise ValueError(
            "Validated mask shape is incompatible with exported labels: "
            f"mask={mask.shape}, exported={frame.shape}"
        )
    return True


def apply_post_solve_corrections(
    exported_labels: np.ndarray,
    corrections: list[Correction],
    tracked_labels: np.ndarray,
    cfg: TrackingConfig,
    *,
    working_dir: str | Path | None = None,
) -> tuple[np.ndarray, PostSolveCorrectionReport]:
    """Apply anchor remap/stamp and validated paste-back to exported labels."""
    if not corrections:
        return exported_labels, PostSolveCorrectionReport()

    labels = exported_labels
    radius = float(cfg.anchor_radius_px)
    anchor_corrections = [c for c in corrections if c.kind == "anchor"]
    validated_corrections = [c for c in corrections if c.kind == "validated"]
    reserved_ids = {int(c.cell_id) for c in corrections}
    used_ids = {int(v) for v in np.unique(labels)}

    claims: dict[int, tuple[float, int, Correction]] = {}
    unsatisfied_anchors: list[Correction] = []
    for correction in anchor_corrections:
        matches = _matching_tracks_at_anchor(labels, correction, radius)
        if not matches:
            unsatisfied_anchors.append(correction)
            continue
        dist, neg_lifetime, track_id = matches[0]
        current = claims.get(track_id)
        if current is None or (dist, neg_lifetime) < (current[0], current[1]):
            if current is not None:
                unsatisfied_anchors.append(current[2])
            claims[track_id] = (dist, neg_lifetime, correction)
        else:
            unsatisfied_anchors.append(correction)

    # solver_id → cell_id for matched anchors. After this disambiguation pass
    # the only solver pixels carrying a reserved cell_id are those owned by
    # the matched anchor (when solver_id == cell_id); every other collision
    # is an unrelated solver track that happens to share the numeric ID, and
    # gets relabeled to a fresh ID so that subsequent stamps/pastes do not
    # produce two disjoint regions sharing the same cell_id.
    solver_track_remap: dict[int, int] = {
        int(sid): int(corr.cell_id)
        for sid, (_d, _l, corr) in claims.items()
    }
    matched_anchor_target_to_solver: dict[int, int] = {
        target_id: solver_id for solver_id, target_id in solver_track_remap.items()
    }
    lineage_track_remaps = _anchor_lineage_track_remaps(working_dir, corrections, cfg)
    for solver_id, cell_id in sorted(lineage_track_remaps.items()):
        if solver_id in solver_track_remap and solver_track_remap[solver_id] != cell_id:
            LOG.warning(
                "Solver track %d claimed by multiple anchor lineages; "
                "keeping cell %d and skipping cell %d",
                solver_id,
                solver_track_remap[solver_id],
                cell_id,
            )
            continue
        solver_track_remap[solver_id] = cell_id
        if solver_id == cell_id:
            matched_anchor_target_to_solver[cell_id] = solver_id
        else:
            matched_anchor_target_to_solver.setdefault(cell_id, solver_id)

    for cell_id in sorted(reserved_ids):
        owning_solver = matched_anchor_target_to_solver.get(cell_id)
        if owning_solver == cell_id:
            # Solver already labeled the owning track with the correct ID;
            # all pixels labeled cell_id belong to that one track.
            continue
        collision_mask = np.asarray(labels == cell_id)
        if collision_mask.any():
            labels[collision_mask] = _next_fresh_id(labels, reserved_ids, used_ids)

    remapped = 0
    for solver_id, target_id in sorted(solver_track_remap.items()):
        if solver_id != target_id:
            labels[labels == solver_id] = target_id
            remapped += 1

    stamped = 0
    for correction in unsatisfied_anchors:
        _stamp_disk(
            labels,
            int(correction.t),
            float(correction.y),
            float(correction.x),
            float(cfg.anchor_stamp_radius_px),
            int(correction.cell_id),
        )
        stamped += 1

    pasted = 0
    for correction in validated_corrections:
        if _paste_validated_mask(labels, tracked_labels, correction):
            pasted += 1

    return labels, PostSolveCorrectionReport(
        remapped_anchor_tracks=remapped,
        stamped_anchors=stamped,
        pasted_validated=pasted,
    )
