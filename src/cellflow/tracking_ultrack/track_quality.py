"""Per-track quality scoring derived from the solved Ultrack database.

Pure module (no Qt, no napari). The quality of an exported track is the sum of
its nodes' ``node_prob`` plus the sum of its selected edges' ``LinkDB.weight``;
higher is better. Ordering by this score lets the UI relabel so ID 1 is the
best track, 2 the next, and so on.

The score is keyed by the painted label value, which equals the exported
``track_id`` from :func:`ultrack.core.export.to_tracks_layer` — the same
dataframe consumed in :mod:`cellflow.tracking_ultrack.corrections`.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.db_query import _engine, annotation_name

# NULL node_prob / link weight render as 1.0, matching db_query.py.
_NULL_VALUE = 1.0


def _resolve_prob(value: float | None) -> float:
    if value is None:
        return _NULL_VALUE
    return float(value)


def _edge_weight(info: tuple[float | None, str] | None) -> float:
    """Weight contributed by one selected edge.

    ``None`` means no link row was found for the pair (contributes 0). A FAKE
    link contributes 0. Otherwise a NULL weight renders as 1.0.
    """
    if info is None:
        return 0.0
    weight, annotation = info
    if annotation == "FAKE":
        return 0.0
    return _NULL_VALUE if weight is None else float(weight)


def compute_track_scores(
    tracks_df,
    node_probs: Mapping[int, float | None],
    links: Mapping[tuple[int, int], tuple[float | None, str]],
) -> dict[int, float]:
    """Pure scoring core: ``track_id -> Σ node_prob + Σ edge weight``.

    ``tracks_df`` needs ``id``, ``t`` and ``track_id`` columns (one row per
    node). ``node_probs`` maps node id -> raw ``node_prob`` (``None`` allowed).
    ``links`` maps a (source_id, target_id) pair -> (raw weight, annotation
    name); the pair is ordered earlier-frame -> later-frame, matching LinkDB.
    """
    scores: dict[int, float] = {}
    for track_id, group in tracks_df.groupby("track_id"):
        ordered = group.sort_values("t")
        ids = [int(i) for i in ordered["id"].tolist()]
        score = sum(_resolve_prob(node_probs.get(nid)) for nid in ids)
        for src, tgt in zip(ids, ids[1:]):
            score += _edge_weight(links.get((src, tgt)))
        scores[int(track_id)] = score
    return scores


def quality_order(scores: Mapping[int, float]) -> list[int]:
    """track_ids sorted by score desc, then by id asc as a stable tiebreak."""
    return [tid for tid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _query_node_probs(db_path: Path, node_ids: list[int]) -> dict[int, float | None]:
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            rows = (
                session.query(NodeDB.id, NodeDB.node_prob)
                .filter(NodeDB.id.in_(node_ids))
                .all()
            )
    finally:
        engine.dispose()
    return {int(nid): (None if prob is None else float(prob)) for nid, prob in rows}


def _query_links(
    db_path: Path, node_ids: list[int]
) -> dict[tuple[int, int], tuple[float | None, str]]:
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB

    wanted = set(node_ids)
    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            rows = (
                session.query(
                    LinkDB.source_id,
                    LinkDB.target_id,
                    LinkDB.weight,
                    LinkDB.annotation,
                )
                .filter(LinkDB.source_id.in_(node_ids))
                .all()
            )
    finally:
        engine.dispose()
    links: dict[tuple[int, int], tuple[float | None, str]] = {}
    for src, tgt, weight, annotation in rows:
        if int(tgt) not in wanted:
            continue
        links[(int(src), int(tgt))] = (
            None if weight is None else float(weight),
            annotation_name(annotation),
        )
    return links


def track_quality_scores(db_path: str | Path, cfg: TrackingConfig) -> dict[int, float]:
    """Map exported ``track_id`` -> quality score (Σ node_prob + Σ edge weight).

    Read-only over the solved database. Returns ``{}`` when the export yields no
    tracks (e.g. an unsolved or empty database).
    """
    from cellflow.tracking_ultrack.ingest import _build_ultrack_config

    db_path = Path(db_path)
    ultrack_cfg = _build_ultrack_config(cfg, db_path.parent)

    from ultrack.core.export import to_tracks_layer

    tracks_df, _graph = to_tracks_layer(ultrack_cfg)
    if tracks_df is None or tracks_df.empty or "id" not in tracks_df:
        return {}

    node_ids = [int(i) for i in tracks_df["id"].tolist()]
    node_probs = _query_node_probs(db_path, node_ids)
    links = _query_links(db_path, node_ids)
    return compute_track_scores(tracks_df, node_probs, links)
