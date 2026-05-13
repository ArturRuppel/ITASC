"""Multi-threshold Ultrack database builder.

For each contour threshold, runs Ultrack segmentation into a temporary
database, then merges all candidates into a single ``data.db`` so the
ILP solver can cherry-pick the best segmentation level per cell.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.db_build import (
    _build_ultrack_config,
    _load_ultrack_inputs,
    _notify,
    _run_ultrack_segment,
    UltrackDatabaseBuildReport,
)
from cellflow.tracking_ultrack.validation_nodes import _make_node_pickle
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.seed_prior import (
    boost_validated_edges,
    write_seed_prior_node_probs,
)
from cellflow.tracking_ultrack.validation_nodes import inject_validated_nodes

LOG = logging.getLogger(__name__)

SOURCE_NODE_TABLE = "cellflow_ultrack_source_nodes"


# ---------------------------------------------------------------------------
# Merge report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiThresholdMergeReport:
    source_count: int
    nodes_per_source: list[int]
    total_nodes: int
    within_source_overlaps: list[int]
    cross_source_overlaps: int


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _read_nodes_and_overlaps(
    db_path: Path,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Read NodeDB + OverlapDB rows from a single Ultrack DB."""
    from ultrack.core.database import NodeDB, OverlapDB

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    nodes: list[dict[str, Any]] = []
    overlaps: list[tuple[int, int]] = []
    try:
        with Session(engine) as session:
            for row in session.query(NodeDB).all():
                nodes.append(
                    {
                        "old_id": int(row.id),
                        "t": int(row.t),
                        "z": 0.0 if row.z is None else float(row.z),
                        "y": float(row.y),
                        "x": float(row.x),
                        "area": 0 if row.area is None else int(row.area),
                        "frontier": None
                        if row.frontier is None
                        else float(row.frontier),
                        "height": None if row.height is None else float(row.height),
                        "pickle": row.pickle,
                        "node_prob": (
                            1.0 if row.node_prob is None else float(row.node_prob)
                        ),
                        "t_node_id": (
                            int(row.t_node_id)
                            if row.t_node_id is not None
                            else None
                        ),
                        "t_hier_id": (
                            int(row.t_hier_id)
                            if row.t_hier_id is not None
                            else None
                        ),
                        "hier_parent_id": (
                            int(row.hier_parent_id)
                            if row.hier_parent_id is not None
                            else None
                        ),
                        "node_annot": row.node_annot,
                        "appear_annot": row.appear_annot,
                        "disappear_annot": row.disappear_annot,
                        "division_annot": row.division_annot,
                        "segm_annot": row.segm_annot,
                    }
                )
            overlaps = [
                (int(o.node_id), int(o.ancestor_id))
                for o in session.query(OverlapDB).all()
            ]
    finally:
        engine.dispose()

    return nodes, overlaps


def _infer_image_shape(nodes: list[dict[str, Any]]) -> tuple[int, int]:
    """Infer frame dimensions from the largest bounding-box extent."""
    from cellflow.tracking_ultrack.validation_nodes import _node_bbox_and_mask

    max_y = max_x = 0
    for row in nodes:
        bb, _ = _node_bbox_and_mask(row["old_id"], row["pickle"])
        max_y = max(max_y, bb[2])
        max_x = max(max_x, bb[3])
    return max_y, max_x


def _remap_t_hier_ids(db_nodes: list[list[dict[str, Any]]]) -> None:
    """Remap source-local hierarchy ids to globally unique ids."""
    hierarchy_map: dict[tuple[int, int | None], int] = {}
    next_global_id = 1

    for source_index, nodes_in_source in enumerate(db_nodes):
        for node in nodes_in_source:
            key = (source_index, node["t_hier_id"])
            if key not in hierarchy_map:
                hierarchy_map[key] = next_global_id
                next_global_id += 1
            node["global_t_hier_id"] = hierarchy_map[key]


def _source_table_exists(conn) -> bool:
    return (
        conn.execute(
            sqla.text(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name=:table_name"
            ),
            {"table_name": SOURCE_NODE_TABLE},
        ).first()
        is not None
    )


def _create_source_node_table(engine: sqla.Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            sqla.text(
                f"""
                CREATE TABLE IF NOT EXISTS {SOURCE_NODE_TABLE} (
                    node_id INTEGER PRIMARY KEY,
                    source_index INTEGER NOT NULL
                )
                """
            )
        )


def query_source_indices(db_path: str | Path) -> tuple[int, ...]:
    """Return source indices recorded by ``merge_ultrack_databases``."""
    engine = sqla.create_engine(f"sqlite:///{Path(db_path)}")
    try:
        with engine.connect() as conn:
            if not _source_table_exists(conn):
                return ()
            rows = conn.execute(
                sqla.text(
                    f"SELECT DISTINCT source_index FROM {SOURCE_NODE_TABLE} "
                    "ORDER BY source_index"
                )
            ).all()
            return tuple(int(row[0]) for row in rows)
    finally:
        engine.dispose()


def query_source_node_ids(db_path: str | Path, source_index: int) -> tuple[int, ...]:
    """Return merged node ids that originated from ``source_index``."""
    engine = sqla.create_engine(f"sqlite:///{Path(db_path)}")
    try:
        with engine.connect() as conn:
            if not _source_table_exists(conn):
                return ()
            rows = conn.execute(
                sqla.text(
                    f"SELECT node_id FROM {SOURCE_NODE_TABLE} "
                    "WHERE source_index=:source_index ORDER BY node_id"
                ),
                {"source_index": int(source_index)},
            ).all()
            return tuple(int(row[0]) for row in rows)
    finally:
        engine.dispose()


def _ndim_from_pickle(pickle: bytes) -> int:
    from cellflow.tracking_ultrack.validation_nodes import _node_pickle_ndim

    return _node_pickle_ndim(pickle)


def _compute_cross_source_overlaps(
    rows_by_source: list[list[dict[str, Any]]],
) -> set[tuple[int, int]]:
    """Compute overlaps between nodes from different sources at one timepoint."""
    from cellflow.tracking_ultrack.validation_nodes import (
        _intersects,
        _node_bbox_and_mask,
    )

    decoded_by_source: list[list[tuple[int, tuple[int, int, int, int], np.ndarray]]] = []
    for rows in rows_by_source:
        decoded_by_source.append(
            [
                (
                    row["new_id"],
                    *_node_bbox_and_mask(row["old_id"], row["pickle"]),
                )
                for row in rows
            ]
        )

    pairs: set[tuple[int, int]] = set()
    for source_index, source_nodes in enumerate(decoded_by_source[:-1]):
        for other_nodes in decoded_by_source[source_index + 1:]:
            for node_id, bbox, mask in source_nodes:
                for other_id, other_bbox, other_mask in other_nodes:
                    if _intersects(bbox, mask, other_bbox, other_mask):
                        pairs.add((max(node_id, other_id), min(node_id, other_id)))
    return pairs


# ---------------------------------------------------------------------------
# Merge primitive — several DBs → one DB
# ---------------------------------------------------------------------------


def merge_ultrack_databases(
    source_db_paths: Sequence[str | Path],
    output_db_path: str | Path,
    *,
    frame_shape: tuple[int, int] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> MultiThresholdMergeReport:
    """Merge several Ultrack ``data.db`` files into a single database.

    Node IDs are globally remapped so they are unique across the merged
    result.  Original overlap pairs from each source are forwarded (after
    remapping), and *cross-source* overlapping mask pairs are detected by
    comparing decoded node masks from the ``pickle`` blobs.

    Parameters
    ----------
    source_db_paths
        Paths to ``data.db`` files produced by ``ultrack.segment`` or
        other Ultrack-compatible pipelines.
    output_db_path
        Target SQLite path (typically ``.../data.db``).  Overwritten if
        it already exists.
    frame_shape
        ``(height, width)`` of the image frames.  If ``None``, inferred
        from the maximum bounding-box extent of the stored node masks.
    progress_cb
        Optional callback ``(message) -> None`` for progress reporting.
    """
    from ultrack.core.database import Base, NodeDB, OverlapDB

    src_paths = [Path(p) for p in source_db_paths]
    out_path = Path(output_db_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy metadata.toml from the first source that has one — Ultrack solve/export
    # needs it (shape, properties). Without it we get KeyError: 'shape'.
    for src in src_paths:
        src_meta = src.parent / "metadata.toml"
        if src_meta.exists():
            shutil.copy2(str(src_meta), str(out_path.parent / "metadata.toml"))
            break

    if out_path.exists():
        out_path.unlink()

    _notify(progress_cb, f"Reading {len(src_paths)} source databases …")

    db_nodes: list[list[dict[str, Any]]] = []
    db_overlaps: list[list[tuple[int, int]]] = []
    for sp in src_paths:
        nds, ovs = _read_nodes_and_overlaps(sp)
        db_nodes.append(nds)
        db_overlaps.append(ovs)

    # Global ID remapping — simple monotonic counter.
    next_id = 1
    for nds in db_nodes:
        for row in nds:
            row["new_id"] = next_id
            next_id += 1

    # Remap t_hier_id to globally unique values per source database
    _remap_t_hier_ids(db_nodes)

    # Remap hier_parent_id within each source database. Source-local node ids
    # are reused across thresholds, so parent lookups must not cross sources.
    from ultrack.utils.constants import NO_PARENT

    for nds in db_nodes:
        old_to_new = {row["old_id"]: row["new_id"] for row in nds}
        for row in nds:
            old_pid = row.get("hier_parent_id")
            if old_pid is not None and old_pid in old_to_new:
                row["hier_parent_id"] = old_to_new[old_pid]
            elif old_pid == NO_PARENT:
                row["hier_parent_id"] = NO_PARENT
            else:
                # Parent not in merged set (or NULL) - clear it.
                row["hier_parent_id"] = None

    total_nodes = sum(len(nds) for nds in db_nodes)
    _notify(progress_cb, f"Remapped {total_nodes} node ids and hierarchy parents …")

    # Forward within-source overlaps using the new ID map.
    overlap_pairs: set[tuple[int, int]] = set()
    within_counts: list[int] = []
    for nds, ovs in zip(db_nodes, db_overlaps):
        lut = {row["old_id"]: row["new_id"] for row in nds}
        count = 0
        for a, b in ovs:
            ai = lut.get(a)
            bi = lut.get(b)
            if ai is not None and bi is not None:
                pair = (max(ai, bi), min(ai, bi))
                if pair not in overlap_pairs:
                    overlap_pairs.add(pair)
                    count += 1
        within_counts.append(count)

    # Cross-source overlaps via pairwise decoded mask checks.
    if frame_shape is None:
        h, w = _infer_image_shape(
            [row for nds in db_nodes for row in nds]
        )
    else:
        h, w = frame_shape

    _notify(progress_cb, f"Computing cross-source overlaps ({h}×{w}) …")

    nodes_by_db_time: list[dict[int, list[dict[str, Any]]]] = [
        {} for _ in range(len(db_nodes))
    ]
    timepoints: set[int] = set()
    for db_idx, nds in enumerate(db_nodes):
        for row in nds:
            t = row["t"]
            timepoints.add(t)
            nodes_by_db_time[db_idx].setdefault(t, []).append(row)

    for t in sorted(timepoints):
        rows_by_source = [
            nodes_by_db_time[db_idx].get(t, [])
            for db_idx in range(len(db_nodes))
        ]
        overlap_pairs.update(_compute_cross_source_overlaps(rows_by_source))

    cross_count = len(overlap_pairs) - sum(within_counts)

    _notify(
        progress_cb,
        (
            f"Writing merged DB: {total_nodes} nodes, "
            f"{len(overlap_pairs)} overlaps ({cross_count} cross) …"
        ),
    )

    engine = sqla.create_engine(f"sqlite:///{out_path}")
    try:
        Base.metadata.create_all(engine)
        _create_source_node_table(engine)
        with Session(engine) as session:
            # Insert nodes with updated pickles.
            for nds in db_nodes:
                for row in nds:
                    new_id = row["new_id"]
                    t = row["t"]
                    from cellflow.tracking_ultrack.validation_nodes import (
                        _node_bbox_and_mask,
                    )

                    bb, mask = _node_bbox_and_mask(row["old_id"], row["pickle"])
                    ndim = _ndim_from_pickle(row["pickle"])
                    new_pickle = _make_node_pickle(
                        t,
                        mask,
                        np.asarray(bb, dtype=np.int32),
                        new_id,
                        ndim=ndim,
                    )

                    session.add(
                        NodeDB(
                            t=t,
                            id=new_id,
                            z=row["z"],
                            y=row["y"],
                            x=row["x"],
                            area=row["area"],
                            frontier=row["frontier"],
                            height=row["height"],
                            pickle=new_pickle,
                            node_prob=row["node_prob"],
                            t_node_id=row["t_node_id"],
                            t_hier_id=row["global_t_hier_id"],
                            hier_parent_id=row["hier_parent_id"],
                            node_annot=row["node_annot"],
                            appear_annot=row["appear_annot"],
                            disappear_annot=row["disappear_annot"],
                            division_annot=row["division_annot"],
                            segm_annot=row["segm_annot"],
                        )
                    )

            # Insert overlaps.
            for pair in sorted(overlap_pairs):
                session.add(OverlapDB(node_id=pair[0], ancestor_id=pair[1]))

            session.commit()
        source_rows = [
            {"node_id": int(row["new_id"]), "source_index": int(source_idx)}
            for source_idx, nds in enumerate(db_nodes)
            for row in nds
        ]
        if source_rows:
            with engine.begin() as conn:
                conn.execute(
                    sqla.text(
                        f"INSERT INTO {SOURCE_NODE_TABLE} "
                        "(node_id, source_index) VALUES (:node_id, :source_index)"
                    ),
                    source_rows,
                )
    finally:
        engine.dispose()

    return MultiThresholdMergeReport(
        source_count=len(src_paths),
        nodes_per_source=[len(nds) for nds in db_nodes],
        total_nodes=total_nodes,
        within_source_overlaps=within_counts,
        cross_source_overlaps=cross_count,
    )


# ---------------------------------------------------------------------------
# High-level threshold sweep
# ---------------------------------------------------------------------------


def _normalize_full_stack(stack: np.ndarray) -> np.ndarray:
    """Normalize one full movie stack to [0, 1] using its global min/max."""
    arr = np.asarray(stack, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32, copy=False)


def _threshold_normalized_stack(stack: np.ndarray, threshold: float) -> np.ndarray:
    """Zero normalized values below threshold without binarizing kept values."""
    normalized = np.asarray(stack, dtype=np.float32)
    return np.where(normalized < threshold, 0.0, normalized).astype(np.float32)


def build_multithreshold_database(
    contour_maps_path: str | Path,
    foreground_scores_path: str | Path,
    nucleus_prob_zavg_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
    thresholds: Sequence[float],
    *,
    validated_tracks: dict[int, set[int]] | None = None,
    tracked_labels: np.ndarray | None = None,
    use_validated: bool = False,
    progress_cb: Callable[[str], None] | None = None,
) -> UltrackDatabaseBuildReport:
    """Build ``data.db`` from several globally normalized threshold levels.

    For each threshold ``τ`` in *thresholds*, both the full contour-map movie and
    the full foreground-score movie are independently min/max-normalized over all
    timepoints/pixels, then values below ``τ`` are set to zero.  Kept values stay
    continuous rather than being binarized.

    After all thresholds are segmented, the temporary databases are merged
    (with cross-threshold overlaps computed) into the final
    ``{working_dir}/data.db``.  The remainder of the canonical DB-generation
    pipeline (optional injection, scoring, linking, and optional boosting) is
    then run on the merged result.

    Parameters
    ----------
    contour_maps_path, foreground_scores_path
        Inputs for the threshold-sweep database build. ``foreground_scores_path``
        is also used as the node-quality signal after candidate merging.
    nucleus_prob_zavg_path
        Legacy argument retained for callers; no longer used for node scoring.
    working_dir
        Final working directory.  Merged ``data.db`` is written here.
    cfg
        Tracking configuration.
    thresholds
        Ordered list of normalized threshold values to sweep.
    validated_tracks, tracked_labels, use_validated
        Same semantics as :func:`build_ultrack_database`.
    progress_cb
        Optional callback ``(message) -> None`` for progress reporting.
    """
    if use_validated and (not validated_tracks or tracked_labels is None):
        raise ValueError(
            "Validated-aware DB generation requires validated tracks and tracked labels."
        )

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_cb, "Loading contour maps and foreground scores …")
    contours, foreground = _load_ultrack_inputs(
        contour_maps_path, foreground_scores_path
    )
    if contours.shape != foreground.shape:
        raise ValueError(
            "Contour maps and foreground scores must have the same shape."
        )
    if contours.ndim != 3:
        raise ValueError(
            "Contour maps and foreground scores must be 3D movies after loading."
        )
    _, h, w = contours.shape
    contours_norm = _normalize_full_stack(contours)
    foreground_norm = _normalize_full_stack(foreground)

    temp_dirs: list[Path] = []
    temp_dbs: list[Path] = []

    try:
        for idx, threshold in enumerate(thresholds):
            _notify(
                progress_cb,
                f"Threshold {threshold:.3f} ({idx + 1}/{len(thresholds)}) …",
            )
            tmp_dir = working_dir / f"_mt_tmp_{idx}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            temp_dirs.append(tmp_dir)

            contours_thr = _threshold_normalized_stack(contours_norm, threshold)
            foreground_thr = _threshold_normalized_stack(foreground_norm, threshold)
            ultrack_cfg = _build_ultrack_config(cfg, tmp_dir)
            _run_ultrack_segment(foreground_thr, contours_thr, ultrack_cfg, cfg)

            db_path = tmp_dir / "data.db"
            if not db_path.exists():
                raise RuntimeError(
                    f"Ultrack segment did not create {db_path} for threshold {threshold}"
                )
            temp_dbs.append(db_path)

        _notify(progress_cb, "Merging threshold databases …")
        merge_ultrack_databases(
            temp_dbs,
            working_dir / "data.db",
            frame_shape=(int(h), int(w)),
            progress_cb=progress_cb,
        )

        # -----------------------------------------------------------------
        # Standard downstream pipeline on the merged database
        # -----------------------------------------------------------------
        real_nodes = skipped_validated = fake_nodes = overlaps_added = 0
        if use_validated:
            _notify(progress_cb, "Injecting validated nodes …")
            injection = inject_validated_nodes(
                working_dir=working_dir,
                validated_tracks=validated_tracks or {},
                tracked_labels=np.asarray(tracked_labels, dtype=np.uint32),
                cfg=cfg,
            )
            real_nodes = int(injection.inserted)
            skipped_validated = int(injection.skipped_missing)
            fake_nodes = int(injection.faked)
            overlaps_added = int(injection.overlaps_added)
            _notify(
                progress_cb,
                (
                    f"Inserted {real_nodes} REAL node(s), marked {fake_nodes} FAKE "
                    f"candidate(s), skipped {skipped_validated} validated cell-frame(s)."
                ),
            )
            if real_nodes == 0:
                raise ValueError(
                    "No validated masks could be injected; DB build aborted."
                )

        _notify(progress_cb, "Scoring node probabilities …")
        score_report = write_seed_prior_node_probs(
            working_dir, foreground_scores_path, cfg
        )
        scored_nodes = int(getattr(score_report, "scored", 0))
        seed_nodes = int(getattr(score_report, "seeds", 0))
        _notify(
            progress_cb,
            f"Scored {scored_nodes} node(s) using {seed_nodes} seed node(s).",
        )

        _notify(progress_cb, "Linking candidates …")
        for step, total, label in run_linking(working_dir, cfg):
            _notify(progress_cb, f"[link {step}/{total}] {label}")

        boosted_edges = 0
        if use_validated:
            _notify(progress_cb, "Boosting edges incident to validated nodes …")
            boost_report = boost_validated_edges(working_dir, cfg)
            boosted_edges = int(getattr(boost_report, "boosted", 0))
            _notify(
                progress_cb,
                f"Boosted {boosted_edges} link(s) incident to REAL nodes.",
            )

    finally:
        for td in temp_dirs:
            if td.exists():
                shutil.rmtree(td, ignore_errors=True)

    return UltrackDatabaseBuildReport(
        real_nodes=real_nodes,
        skipped_validated=skipped_validated,
        fake_nodes=fake_nodes,
        overlaps_added=overlaps_added,
        scored_nodes=scored_nodes,
        seed_nodes=seed_nodes,
        boosted_edges=boosted_edges,
    )
