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
from cellflow.tracking_ultrack.ingest import _compute_overlaps_vectorized
from cellflow.tracking_ultrack.validation_nodes import _make_node_pickle
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.seed_prior import (
    boost_validated_edges,
    write_seed_prior_node_probs,
)
from cellflow.tracking_ultrack.solve import run_solve
from cellflow.tracking_ultrack.validation_nodes import inject_validated_nodes

LOG = logging.getLogger(__name__)


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


def _ndim_from_pickle(pickle: bytes) -> int:
    from cellflow.tracking_ultrack.validation_nodes import _node_pickle_ndim

    return _node_pickle_ndim(pickle)


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
    reconstructing spatial labelmaps from the ``pickle`` blobs.

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

    total_nodes = sum(len(nds) for nds in db_nodes)
    _notify(progress_cb, f"Remapped {total_nodes} node ids …")

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

    # Cross-source overlaps via vectorised labelmap reconstruction.
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
        nid_lms: list[np.ndarray] = []
        for db_idx in range(len(db_nodes)):
            rows_t = nodes_by_db_time[db_idx].get(t, [])
            lm = np.zeros((h, w), dtype=np.int64)
            if rows_t:
                from cellflow.tracking_ultrack.validation_nodes import (
                    _node_bbox_and_mask,
                )

                for row in rows_t:
                    bb, mask = _node_bbox_and_mask(row["old_id"], row["pickle"])
                    y0, x0, y1, x1 = bb
                    lm[y0:y1, x0:x1][mask] = row["new_id"]
            nid_lms.append(lm)

        cross = _compute_overlaps_vectorized(nid_lms)
        overlap_pairs.update(cross)

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
                            t_hier_id=row["t_hier_id"],
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


def build_multithreshold_database(
    contour_maps_path: str | Path,
    foreground_masks_path: str | Path,
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
    """Build ``data.db`` from several thresholded contour maps.

    For each threshold ``τ`` in *thresholds*:

    1. ``contours_τ`` is created by flooring values below ``τ`` to 0.
    2. ``ultrack.segment`` runs into a temporary working directory.

    After all thresholds are segmented, the temporary databases are merged
    (with cross-threshold overlaps computed) into the final
    ``{working_dir}/data.db``.  The remainder of the canonical pipeline
    (scoring, linking, optional injection / boosting, and solve) is then
    run on the merged result.

    Parameters
    ----------
    contour_maps_path, foreground_masks_path, nucleus_prob_zavg_path
        Standard inputs matching :func:`build_ultrack_database`.
    working_dir
        Final working directory.  Merged ``data.db`` is written here.
    cfg
        Tracking configuration.
    thresholds
        Ordered list of contour threshold values to sweep.
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

    _notify(progress_cb, "Loading contour maps and foreground masks …")
    contours, foreground = _load_ultrack_inputs(
        contour_maps_path, foreground_masks_path
    )
    _, h, w = contours.shape

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

            contours_thr = np.where(contours < threshold, 0.0, contours).astype(
                np.float32
            )
            ultrack_cfg = _build_ultrack_config(cfg, tmp_dir)
            _run_ultrack_segment(foreground, contours_thr, ultrack_cfg, cfg)

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
            working_dir, nucleus_prob_zavg_path, cfg
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
