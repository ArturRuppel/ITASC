"""Diagnostics for anchored Ultrack solves."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from cellflow.tracking_ultrack.anchor import _gt_masks, _mask_iou, _node_mask_record


@dataclass(frozen=True)
class SelectedOverlap:
    node_id: int
    iou: float
    area: int


@dataclass(frozen=True)
class AnchorCandidateMatch:
    gt_label: int
    best_node_id: int | None
    best_iou: float
    selected: bool
    node_annot: Any
    incoming_link_count: int = 0
    best_incoming_weight: float | None = None
    outgoing_link_count: int = 0
    best_outgoing_weight: float | None = None
    selected_overlaps: list[SelectedOverlap] | None = None


@dataclass(frozen=True)
class AnchorCandidateDiagnostics:
    frame_index: int
    matches: list[AnchorCandidateMatch]


def diagnose_anchor_frame_candidates(
    working_dir: str | Path,
    labels: np.ndarray,
    *,
    frame_index: int,
) -> AnchorCandidateDiagnostics:
    """Report each GT object's best matching NodeDB candidate at one frame."""
    from ultrack.core.database import LinkDB, NodeDB, OverlapDB

    db_url = f"sqlite:///{Path(working_dir) / 'data.db'}"
    gt_records = _gt_masks(labels, frame_index)

    engine = sqla.create_engine(db_url)
    with Session(engine) as session:
        rows = (
            session.query(NodeDB.id, NodeDB.pickle, NodeDB.selected, NodeDB.node_annot)
            .where(NodeDB.t == frame_index)
            .all()
        )
        node_ids = [int(node_id) for node_id, _node, _selected, _annot in rows]
        link_rows = []
        overlap_rows = []
        if node_ids:
            link_rows = (
                session.query(LinkDB.source_id, LinkDB.target_id, LinkDB.weight)
                .where(
                    sqla.or_(
                        LinkDB.source_id.in_(node_ids),
                        LinkDB.target_id.in_(node_ids),
                    )
                )
                .all()
            )
            overlap_rows = (
                session.query(OverlapDB.node_id, OverlapDB.ancestor_id)
                .where(
                    sqla.or_(
                        OverlapDB.node_id.in_(node_ids),
                        OverlapDB.ancestor_id.in_(node_ids),
                    )
                )
                .all()
            )

    node_records = [
        (
            _node_mask_record(int(node_id), node),
            bool(selected),
            node_annot,
        )
        for node_id, node, selected, node_annot in rows
    ]
    node_by_id = {node.label_id: (node, selected, node_annot) for node, selected, node_annot in node_records}

    incoming_weights: dict[int, list[float]] = {node_id: [] for node_id in node_by_id}
    outgoing_weights: dict[int, list[float]] = {node_id: [] for node_id in node_by_id}
    for source_id, target_id, weight in link_rows:
        source_id = int(source_id)
        target_id = int(target_id)
        if target_id in incoming_weights:
            incoming_weights[target_id].append(float(weight))
        if source_id in outgoing_weights:
            outgoing_weights[source_id].append(float(weight))

    selected_overlaps_by_id: dict[int, list[SelectedOverlap]] = {node_id: [] for node_id in node_by_id}
    for node_id, ancestor_id in overlap_rows:
        lhs_id = int(node_id)
        rhs_id = int(ancestor_id)
        for current_id, other_id in ((lhs_id, rhs_id), (rhs_id, lhs_id)):
            current = node_by_id.get(current_id)
            other = node_by_id.get(other_id)
            if current is None or other is None:
                continue
            other_node, other_selected, _other_annot = other
            if not other_selected:
                continue
            current_node, _current_selected, _current_annot = current
            selected_overlaps_by_id[current_id].append(
                SelectedOverlap(
                    node_id=other_id,
                    iou=float(_mask_iou(current_node, other_node)),
                    area=int(other_node.mask.sum()),
                )
            )

    matches: list[AnchorCandidateMatch] = []
    for gt in gt_records:
        best: tuple[float, int, bool, Any] | None = None
        for node, selected, node_annot in node_records:
            iou = _mask_iou(gt, node)
            candidate = (iou, node.label_id, selected, node_annot)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

        if best is None:
            matches.append(
                AnchorCandidateMatch(
                    gt_label=gt.label_id,
                    best_node_id=None,
                    best_iou=0.0,
                    selected=False,
                    node_annot=None,
                    selected_overlaps=[],
                )
            )
            continue

        best_iou, best_node_id, selected, node_annot = best
        incoming = incoming_weights.get(best_node_id, [])
        outgoing = outgoing_weights.get(best_node_id, [])
        matches.append(
            AnchorCandidateMatch(
                gt_label=gt.label_id,
                best_node_id=best_node_id,
                best_iou=float(best_iou),
                selected=selected,
                node_annot=node_annot,
                incoming_link_count=len(incoming),
                best_incoming_weight=max(incoming) if incoming else None,
                outgoing_link_count=len(outgoing),
                best_outgoing_weight=max(outgoing) if outgoing else None,
                selected_overlaps=sorted(
                    selected_overlaps_by_id.get(best_node_id, []),
                    key=lambda overlap: overlap.node_id,
                ),
            )
        )

    return AnchorCandidateDiagnostics(frame_index=int(frame_index), matches=matches)
