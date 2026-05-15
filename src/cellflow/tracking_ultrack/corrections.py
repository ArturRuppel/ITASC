"""Per-frame correction primitives for Ultrack solves and exports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig


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
) -> CorrectionDatabaseReport:
    """Apply solve-time correction annotations to ``data.db``.

    Validated frames mark nearby candidates ``FAKE``. Anchors mark the nearest
    surviving candidate ``REAL`` and mark consecutive anchor links ``REAL``.
    In Ultrack 0.6.x these are hard annotations when solve uses
    ``use_annotations=True``.
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

        for correction in corrections:
            if correction.kind != "anchor":
                continue
            rows = session.query(NodeDB.id, NodeDB.y, NodeDB.x).where(
                NodeDB.t == int(correction.t),
                NodeDB.node_annot != VarAnnotation.FAKE,
            )
            nearest: tuple[float, int] | None = None
            for node_id, y, x in rows:
                dist = _distance(y, x, correction.y, correction.x)
                if dist <= radius and (nearest is None or dist < nearest[0]):
                    nearest = (dist, int(node_id))
            if nearest is None:
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
    matched_anchor_target_to_solver: dict[int, int] = {
        int(corr.cell_id): int(sid)
        for sid, (_d, _l, corr) in claims.items()
    }
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
    for solver_id, (_dist, _neg_lifetime, correction) in sorted(claims.items()):
        target_id = int(correction.cell_id)
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
