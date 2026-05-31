"""Read-only Ultrack database helpers for napari preview tooling."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

import numpy as np


@dataclass(frozen=True)
class HierarchyCutState:
    node_ids: tuple[int, ...]
    height: float | None


@dataclass(frozen=True)
class UltrackDbPreview:
    labels: np.ndarray
    status: str
    probabilities: dict[int, float]
    label_to_node_id: dict[int, int]
    node_id_to_label: dict[int, int]
    node_annotations: dict[int, str]

    def as_tuple(self):
        return (
            self.labels,
            self.status,
            self.probabilities,
            self.label_to_node_id,
            self.node_id_to_label,
            self.node_annotations,
        )


def _engine(db_path: Path):
    import sqlalchemy as sqla

    return sqla.create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )


def query_middle_frame(db_path: Path) -> int | None:
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            frames = sorted(int(r[0]) for r in session.query(NodeDB.t).distinct().all())
    except Exception:
        return None
    finally:
        engine.dispose()
    return frames[len(frames) // 2] if frames else None


def query_frame_range(db_path: Path) -> tuple[int, ...]:
    """Return sorted distinct frame indices present in the database."""
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            frames = sorted(int(r[0]) for r in session.query(NodeDB.t).distinct().all())
    except Exception:
        return ()
    finally:
        engine.dispose()
    return tuple(frames)


def query_connected_nodes(
    db_path: Path, selected_node_id: int
) -> tuple[dict[int, float], dict[int, float]]:
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB

    engine = _engine(db_path)
    predecessors: dict[int, float] = {}
    successors: dict[int, float] = {}
    try:
        with Session(engine) as session:
            rows = (
                session.query(LinkDB.source_id, LinkDB.target_id, LinkDB.weight)
                .filter(
                    (LinkDB.source_id == int(selected_node_id))
                    | (LinkDB.target_id == int(selected_node_id))
                )
                .all()
            )
            for src, tgt, weight in rows:
                wf = float(weight if weight is not None else 1.0)
                if int(tgt) == int(selected_node_id):
                    src_id = int(src)
                    predecessors[src_id] = predecessors.get(src_id, 1.0) * wf
                if int(src) == int(selected_node_id):
                    tgt_id = int(tgt)
                    successors[tgt_id] = successors.get(tgt_id, 1.0) * wf
    finally:
        engine.dispose()
    return predecessors, successors


def _empty_annotation_counts() -> dict[str, int]:
    return {"REAL": 0, "FAKE": 0, "UNKNOWN": 0}


def _annotation_counts_from_rows(rows) -> dict[str, int]:
    counts = _empty_annotation_counts()
    for annotation, count in rows:
        name = annotation_name(annotation)
        counts[name] = counts.get(name, 0) + int(count or 0)
    return counts


def link_annotation_counts(
    db_path: Path, selected_node_id: int | None = None
) -> dict[str, int]:
    """Return link annotation counts for the whole DB or one node's incident links."""
    engine = _engine(db_path)
    try:
        from sqlalchemy import func
        from sqlalchemy.orm import Session
        from ultrack.core.database import LinkDB

        with Session(engine) as session:
            query = session.query(LinkDB.annotation, func.count(LinkDB.source_id))
            if selected_node_id is not None:
                node_id = int(selected_node_id)
                query = query.filter(
                    (LinkDB.source_id == node_id) | (LinkDB.target_id == node_id)
                )
            rows = query.group_by(LinkDB.annotation).all()
            return _annotation_counts_from_rows(rows)
    except Exception:
        return _empty_annotation_counts()
    finally:
        engine.dispose()


def _finite_values(rows) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            raw = row[0]
        except (TypeError, KeyError, IndexError):
            raw = row
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _stats_text(label: str, values: list[float], prefix: str) -> str | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"{label} {prefix}: "
        f"min {float(np.min(arr)):.3f}, "
        f"median {float(np.median(arr)):.3f}, "
        f"mean {float(np.mean(arr)):.3f}, "
        f"max {float(np.max(arr)):.3f}"
    )


def summary_text(db_path: Path, frame: int) -> str:
    from sqlalchemy import func
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, NodeDB, VarAnnotation

    try:
        from ultrack.core.database import OverlapDB
    except Exception:
        OverlapDB = None

    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            n_nodes = int(session.query(func.count(NodeDB.id)).scalar() or 0)
            n_links = int(session.query(func.count(LinkDB.source_id)).scalar() or 0)
            node_prob_values = _finite_values(
                session.query(NodeDB.node_prob)
                .filter(NodeDB.node_prob.isnot(None))
                .all()
            )
            link_weight_values = _finite_values(
                session.query(LinkDB.weight)
                .filter(LinkDB.weight.isnot(None))
                .all()
            )
            n_real = int(
                session.query(func.count(NodeDB.id))
                .filter(NodeDB.node_annot == VarAnnotation.REAL)
                .scalar()
                or 0
            )
            n_fake = int(
                session.query(func.count(NodeDB.id))
                .filter(NodeDB.node_annot == VarAnnotation.FAKE)
                .scalar()
                or 0
            )
            n_unknown = max(n_nodes - n_real - n_fake, 0)
            link_annotation_rows = (
                session.query(LinkDB.annotation, func.count(LinkDB.source_id))
                .group_by(LinkDB.annotation)
                .all()
            )
            link_annotations = _annotation_counts_from_rows(link_annotation_rows)
            frame_nodes = session.query(NodeDB).filter(NodeDB.t == frame).all()
            selected = sum(1 for n in frame_nodes if getattr(n, "selected", False))
            node_ids = [int(n.id) for n in frame_nodes]
            outgoing = incoming = overlaps = 0
            if node_ids:
                outgoing = int(
                    session.query(func.count(LinkDB.source_id))
                    .filter(LinkDB.source_id.in_(node_ids))
                    .scalar()
                    or 0
                )
                incoming = int(
                    session.query(func.count(LinkDB.target_id))
                    .filter(LinkDB.target_id.in_(node_ids))
                    .scalar()
                    or 0
                )
                if OverlapDB is not None:
                    try:
                        overlaps = int(
                            session.query(func.count(OverlapDB.node_id))
                            .filter(
                                OverlapDB.node_id.in_(node_ids)
                                | OverlapDB.ancestor_id.in_(node_ids)
                            )
                            .scalar()
                            or 0
                        )
                    except Exception:
                        overlaps = 0
        stats_parts = [
            text
            for text in (
                _stats_text(
                    "node prob",
                    node_prob_values,
                    f"{len(node_prob_values)}/{n_nodes} scored",
                ),
                _stats_text(
                    "edge weight",
                    link_weight_values,
                    f"{len(link_weight_values)} links",
                ),
            )
            if text is not None
        ]
        lines = [
            f"Database: {n_nodes} nodes, {n_links} links",
            f"Node annotations: REAL {n_real}, FAKE {n_fake}, UNKNOWN {n_unknown}",
            (
                f"Link annotations: REAL links {link_annotations['REAL']}, "
                f"FAKE links {link_annotations['FAKE']}, "
                f"UNKNOWN links {link_annotations['UNKNOWN']}"
            ),
            (
                f"Frame {frame}: {len(node_ids)} nodes, {selected} selected, "
                f"{incoming} in/{outgoing} out links, {overlaps} overlaps"
            ),
        ]
        if stats_parts:
            lines.extend(stats_parts)
        return "\n".join(lines)
    finally:
        engine.dispose()


def query_distinct_heights(db_path: Path) -> tuple[float, ...]:
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            return tuple(
                float(r[0])
                for r in session.query(NodeDB.height)
                .distinct()
                .order_by(NodeDB.height)
                .all()
                if r[0] is not None
            )
    finally:
        engine.dispose()


def query_union_sizes(db_path: Path, frame: int) -> tuple[int, ...]:
    """Distinct atom-union sizes (``height`` = number of merged atoms) present in
    ``frame``, sorted ascending. Drives the vertical "union size" slider: index 0
    is the finest (1 atom = individual atoms), higher indices merge more atoms.
    """
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            rows = (
                session.query(NodeDB.height)
                .filter(NodeDB.t == frame)
                .distinct()
                .order_by(NodeDB.height)
                .all()
            )
    finally:
        engine.dispose()
    return tuple(int(round(float(r[0]))) for r in rows if r[0] is not None)


def query_union_color_classes(
    db_path: Path, frame: int, union_size: int
) -> tuple[tuple[int, ...], ...]:
    """Group the size-``union_size`` candidates of ``frame`` into color classes of
    mutually non-overlapping candidates (a greedy graph coloring of the shared-atom
    overlap graph from ``OverlapDB``).

    Each returned class is a tuple of node ids that can be painted together in one
    full-frame partition without conflict, so every candidate of this size is shown
    in exactly one class while keeping the number of classes (= horizontal slider
    positions) small. Returns ``()`` if no candidates of this size exist.
    """
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    try:
        from ultrack.core.database import OverlapDB
    except Exception:
        OverlapDB = None

    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            node_ids = [
                int(r[0])
                for r in session.query(NodeDB.id)
                .filter(NodeDB.t == frame)
                .filter(NodeDB.height == float(union_size))
                .order_by(NodeDB.id)
                .all()
            ]
            id_set = set(node_ids)
            edges: list[tuple[int, int]] = []
            if OverlapDB is not None and node_ids:
                # OverlapDB stores node_id < ancestor_id, so every intra-set edge has
                # its node_id endpoint in node_ids; filtering on that side is exact.
                rows = (
                    session.query(OverlapDB.node_id, OverlapDB.ancestor_id)
                    .filter(OverlapDB.node_id.in_(node_ids))
                    .all()
                )
                edges = [(int(a), int(b)) for a, b in rows if int(b) in id_set]
    finally:
        engine.dispose()

    return greedy_color_classes(node_ids, edges)


def greedy_color_classes(
    node_ids: list[int], edges: Iterable[tuple[int, int]]
) -> tuple[tuple[int, ...], ...]:
    """Partition ``node_ids`` into classes with no intra-class edge (greedy graph
    coloring). Welsh–Powell order (highest degree first) keeps the class count low;
    ids preserve their original order within each class. Pure helper, no DB."""
    if not node_ids:
        return ()
    adj: dict[int, set[int]] = {nid: set() for nid in node_ids}
    id_set = set(node_ids)
    for a, b in edges:
        if a != b and a in id_set and b in id_set:
            adj[a].add(b)
            adj[b].add(a)

    order = sorted(node_ids, key=lambda n: (-len(adj[n]), n))
    colors: dict[int, int] = {}
    for nid in order:
        used = {colors[m] for m in adj[nid] if m in colors}
        c = 0
        while c in used:
            c += 1
        colors[nid] = c

    n_colors = max(colors.values()) + 1
    classes: list[list[int]] = [[] for _ in range(n_colors)]
    for nid in node_ids:  # stable id order within each class
        classes[colors[nid]].append(nid)
    return tuple(tuple(c) for c in classes)


def _paint_union_partition(
    union_nodes: list, fill_nodes: list, plane_shape: tuple[int, int]
) -> tuple[np.ndarray, list]:
    """Paint a full-frame partition: the merged ``union_nodes`` (the selected merge
    group) take priority, then ``fill_nodes`` cover the leftover territory.

    ``fill_nodes`` should be ordered most-merged first; each is painted all-or-nothing
    only if its whole mask is still free, so a leftover atom is shown inside the
    largest union that still fits rather than dropping back to an individual atom.

    Returns ``(labels, ordered_nodes)`` where ``ordered_nodes[i]`` is painted with
    label ``i + 1`` — matching ``node_preview_metadata`` so click selection lines up.
    """
    parsed: list[tuple] = []
    max_y = max_x = 0
    for node in union_nodes:
        result = node_mask_and_bbox(node)
        if result is None:
            continue
        (y0, x0, y1, x1), mask = result
        max_y, max_x = max(max_y, y1), max(max_x, x1)
        parsed.append((node, (y0, x0, y1, x1), mask, True))
    for node in fill_nodes:
        result = node_mask_and_bbox(node)
        if result is None:
            continue
        (y0, x0, y1, x1), mask = result
        max_y, max_x = max(max_y, y1), max(max_x, x1)
        parsed.append((node, (y0, x0, y1, x1), mask, False))

    base_y, base_x = plane_shape
    labels = np.zeros((max(base_y, max_y, 1), max(base_x, max_x, 1)), dtype=np.uint32)
    ordered: list = []
    for node, (y0, x0, y1, x1), mask, priority in parsed:
        region = labels[y0:y1, x0:x1]
        if region.shape != mask.shape:
            continue
        if priority:
            # Unions never overlap within a class, so they paint freely.
            sub = mask
        else:
            # All-or-nothing: only paint a leftover union/atom if its whole footprint
            # is still free, so larger unions win and nothing is split mid-region.
            if (mask & (region != 0)).any():
                continue
            sub = mask
        if not sub.any():
            continue
        region[sub] = len(ordered) + 1
        ordered.append(node)
    return labels, ordered


def render_union_partition(
    db_path: Path,
    frame: int,
    color_node_ids: Iterable[int],
    *,
    plane_shape: tuple[int, int],
    union_size: int | None = None,
) -> UltrackDbPreview:
    """Full-frame partition for one horizontal slider position: the candidates in
    ``color_node_ids`` painted as merged regions, with the leftover territory filled
    by the largest available unions (size ≤ ``union_size``) so regions outside this
    merge group still show at their most-merged state instead of as raw atoms.

    ``union_size`` caps how merged the leftover fill may be; when ``None`` it is taken
    from the selected candidates (their ``height``), defaulting to atoms only.
    """
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    color_node_ids = tuple(int(n) for n in color_node_ids)
    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            union_nodes: list = []
            if color_node_ids:
                rows = (
                    session.query(NodeDB)
                    .filter(NodeDB.t == frame)
                    .filter(NodeDB.id.in_(color_node_ids))
                    .all()
                )
                by_id = {int(n.id): n for n in rows}
                union_nodes = [by_id[i] for i in color_node_ids if i in by_id]
            if union_size is None:
                heights = [
                    float(n.height) for n in union_nodes if n.height is not None
                ]
                union_size = int(round(max(heights))) if heights else 1
            # Leftover fill: every non-selected candidate up to the current level,
            # most-merged first, so larger unions claim free territory before atoms.
            fill_nodes = (
                session.query(NodeDB)
                .filter(NodeDB.t == frame)
                .filter(NodeDB.height <= float(union_size))
                .filter(NodeDB.id.notin_(color_node_ids) if color_node_ids else True)
                .order_by(NodeDB.height.desc(), NodeDB.id)
                .all()
            )
    finally:
        engine.dispose()

    labels, ordered = _paint_union_partition(union_nodes, fill_nodes, plane_shape)
    if not ordered:
        return _empty_preview(plane_shape, f"No candidates for frame {frame}.")

    probabilities, label_to_node_id, node_id_to_label = node_preview_metadata(ordered)
    annotations = node_annotation_metadata(ordered)
    n_merged = len(union_nodes)
    return UltrackDbPreview(
        labels=labels,
        status=(
            f"Frame {frame}: {len(ordered)} region(s), "
            f"{n_merged} merged candidate(s)."
        ),
        probabilities=probabilities,
        label_to_node_id=label_to_node_id,
        node_id_to_label=node_id_to_label,
        node_annotations=annotations,
    )


def query_hierarchy_cut_states(
    db_path: Path, frame: int, *, source_index: int | None = None
) -> tuple[HierarchyCutState, ...]:
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB
    from ultrack.utils.constants import NO_PARENT

    source_node_ids: tuple[int, ...] = ()
    if source_index is not None:
        try:
            from cellflow.tracking_ultrack.multi_threshold import query_source_node_ids

            source_node_ids = query_source_node_ids(db_path, int(source_index))
        except Exception:
            source_node_ids = ()

    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            query = (
                session.query(NodeDB.id, NodeDB.hier_parent_id, NodeDB.height)
                .filter(NodeDB.t == frame)
                .order_by(NodeDB.height, NodeDB.id)
            )
            if source_index is not None:
                query = query.filter(NodeDB.id.in_(source_node_ids))
            rows = []
            for nid, pid, height in query.all():
                if height is None:
                    continue
                parent_id = NO_PARENT if pid is None else int(pid)
                rows.append((int(nid), parent_id, float(height)))
    except Exception:
        return tuple(HierarchyCutState((), float(h)) for h in query_distinct_heights(db_path))
    finally:
        engine.dispose()

    if not rows:
        return ()

    node_ids = {nid for nid, _, _ in rows}
    heights_by_id = {nid: height for nid, _, height in rows}
    parent_by_id = {
        nid: pid for nid, pid, _ in rows if pid != NO_PARENT and pid in node_ids
    }
    children: dict[int, set[int]] = {}
    for child_id, parent_id in parent_by_id.items():
        children.setdefault(parent_id, set()).add(child_id)

    active = {nid for nid, _, _ in rows if nid not in children}
    if not active:
        active = set(node_ids)

    states: list[HierarchyCutState] = []
    seen: set[tuple[int, ...]] = set()

    def append_state() -> None:
        ordered = tuple(sorted(active, key=lambda n: (heights_by_id[n], n)))
        if ordered in seen:
            return
        seen.add(ordered)
        height = max((heights_by_id[n] for n in ordered), default=None)
        states.append(HierarchyCutState(ordered, height))

    append_state()
    while True:
        promotable = [
            parent_id
            for parent_id, child_ids in children.items()
            if parent_id not in active and child_ids and child_ids.issubset(active)
        ]
        if not promotable:
            break
        min_height = min(heights_by_id[parent_id] for parent_id in promotable)
        for parent_id in sorted(
            p for p in promotable if heights_by_id[p] == min_height
        ):
            active.difference_update(children[parent_id])
            active.add(parent_id)
        append_state()

    return tuple(states)


def query_available_sources(db_path: Path) -> tuple[int, ...]:
    try:
        from cellflow.tracking_ultrack.multi_threshold import query_source_indices

        return query_source_indices(db_path)
    except Exception:
        return ()


def render_hierarchy_cut(
    db_path: Path,
    frame: int,
    height: float,
    *,
    plane_shape: tuple[int, int],
) -> UltrackDbPreview:
    from sqlalchemy.orm import Session, aliased
    from ultrack.core.database import NodeDB
    from ultrack.utils.constants import NO_PARENT

    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            parent = aliased(NodeDB)
            child = aliased(NodeDB)
            same_child = (
                session.query(child.id)
                .where(child.hier_parent_id == NodeDB.id)
                .where(child.height == NodeDB.height)
                .where(NodeDB.height == height)
                .exists()
            )
            nodes = (
                session.query(NodeDB)
                .outerjoin(parent, NodeDB.hier_parent_id == parent.id)
                .where(NodeDB.t == frame)
                .where(NodeDB.height <= height)
                .where(
                    (NodeDB.hier_parent_id == NO_PARENT)
                    | ((NodeDB.height < height) & (parent.height > height))
                    | ((NodeDB.height == height) & (parent.height >= height))
                )
                .where(~same_child)
                .all()
            )
    finally:
        engine.dispose()
    return finalize_hierarchy_nodes(
        nodes,
        frame,
        plane_shape=plane_shape,
        empty_msg=f"No segments at this threshold for frame {frame}.",
        status_suffix=f"at h={height:.2f}",
    )


def render_hierarchy_cut_state(
    db_path: Path,
    frame: int,
    state: HierarchyCutState,
    *,
    plane_shape: tuple[int, int],
) -> UltrackDbPreview:
    if not state.node_ids:
        return render_hierarchy_cut(
            db_path,
            frame,
            float(state.height or 0.0),
            plane_shape=plane_shape,
        )

    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            rows = (
                session.query(NodeDB)
                .where(NodeDB.t == frame)
                .where(NodeDB.id.in_(state.node_ids))
                .all()
            )
    finally:
        engine.dispose()

    by_id = {int(n.id): n for n in rows}
    nodes = [by_id[nid] for nid in state.node_ids if nid in by_id]
    height = "—" if state.height is None else f"{state.height:.2f}"
    return finalize_hierarchy_nodes(
        nodes,
        frame,
        plane_shape=plane_shape,
        empty_msg=f"No hierarchy state segments for frame {frame}.",
        status_suffix=f"at cut state h={height}",
    )


def finalize_hierarchy_nodes(
    nodes: Iterable,
    frame: int,
    *,
    plane_shape: tuple[int, int],
    empty_msg: str,
    status_suffix: str,
) -> UltrackDbPreview:
    nodes = list(nodes)
    if not nodes:
        return _empty_preview(plane_shape, empty_msg)

    labels = paint_nodes(nodes, plane_shape)
    probabilities, label_to_node_id, node_id_to_label = node_preview_metadata(nodes)
    annotations = node_annotation_metadata(nodes)
    return UltrackDbPreview(
        labels=labels,
        status=f"Frame {frame}: {len(nodes)} segment(s) {status_suffix}.",
        probabilities=probabilities,
        label_to_node_id=label_to_node_id,
        node_id_to_label=node_id_to_label,
        node_annotations=annotations,
    )


def _empty_preview(plane_shape: tuple[int, int], status: str) -> UltrackDbPreview:
    return UltrackDbPreview(
        labels=np.zeros(plane_shape, dtype=np.uint32),
        status=status,
        probabilities={},
        label_to_node_id={},
        node_id_to_label={},
        node_annotations={},
    )


def annotation_name(value) -> str:
    if value is None:
        return "UNKNOWN"
    for raw in (getattr(value, "name", None), value, getattr(value, "value", None)):
        if raw is None:
            continue
        if raw in {0, "0"}:
            return "UNKNOWN"
        if raw in {1, "1"}:
            return "REAL"
        if raw in {2, "2"}:
            return "FAKE"
        name = str(raw).split(".")[-1].upper()
        if name in {"REAL", "FAKE", "UNKNOWN"}:
            return name
    return "UNKNOWN"


def node_preview_metadata(nodes) -> tuple[dict[int, float], dict[int, int], dict[int, int]]:
    probabilities: dict[int, float] = {}
    label_to_node_id: dict[int, int] = {}
    node_id_to_label: dict[int, int] = {}
    for label, node in enumerate(nodes, start=1):
        try:
            probability = float(node.node_prob if node.node_prob is not None else 1.0)
        except (TypeError, ValueError):
            probability = 1.0
        probabilities[label] = probability
        try:
            node_id = int(node.id)
        except (TypeError, ValueError):
            continue
        label_to_node_id[label] = node_id
        node_id_to_label[node_id] = label
    return probabilities, label_to_node_id, node_id_to_label


def node_annotation_metadata(nodes) -> dict[int, str]:
    annotations: dict[int, str] = {}
    for node in nodes:
        try:
            node_id = int(node.id)
        except (TypeError, ValueError):
            continue
        annotations[node_id] = annotation_name(getattr(node, "node_annot", None))
    return annotations


def paint_nodes(nodes, plane_shape: tuple[int, int]) -> np.ndarray:
    masks: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    max_y = max_x = 0
    for label, node in enumerate(nodes, start=1):
        parsed = node_mask_and_bbox(node)
        if parsed is None:
            continue
        bbox, mask = parsed
        y0, x0, y1, x1 = bbox
        max_y = max(max_y, y1)
        max_x = max(max_x, x1)
        masks.append((label, bbox, mask))

    base_y, base_x = plane_shape
    labels = np.zeros((max(base_y, max_y, 1), max(base_x, max_x, 1)), dtype=np.uint32)
    for label, (y0, x0, y1, x1), mask in masks:
        target = labels[y0:y1, x0:x1]
        if target.shape != mask.shape:
            continue
        target[mask.astype(bool)] = label
    return labels


def node_mask_and_bbox(node):
    try:
        node_obj = node.pickle
        if isinstance(node_obj, (bytes, memoryview)):
            node_obj = pickle.loads(bytes(node_obj))
        if node_obj is None:
            return None
    except Exception:
        return None

    if isinstance(node_obj, dict):
        bbox, mask = node_obj.get("bbox"), node_obj.get("mask")
    elif isinstance(node_obj, tuple) and len(node_obj) >= 2:
        bbox, mask = node_obj[0], node_obj[1]
    else:
        bbox = getattr(node_obj, "bbox", None)
        mask = getattr(node_obj, "mask", None)
    if bbox is None or mask is None:
        return None

    bbox_array = np.asarray(bbox, dtype=int).ravel()
    if bbox_array.size >= 6:
        y0, x0, y1, x1 = (
            int(bbox_array[1]),
            int(bbox_array[2]),
            int(bbox_array[4]),
            int(bbox_array[5]),
        )
    elif bbox_array.size >= 4:
        y0, x0, y1, x1 = (int(v) for v in bbox_array[:4])
    else:
        return None

    mask_array = np.asarray(mask)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = mask_array[0]
    elif mask_array.ndim > 2:
        mask_array = np.squeeze(mask_array)
    if mask_array.ndim != 2:
        return None
    if mask_array.shape != (y1 - y0, x1 - x0):
        return None
    return (y0, x0, y1, x1), mask_array.astype(bool, copy=False)
