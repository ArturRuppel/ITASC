# Multi-Threshold Ultrack Database Builder — Design Document

## Goal

Build a single Ultrack database that contains cell segmentation candidates from **multiple contour-map thresholds**. The ILP solver then picks the best cell per timepoint across the entire ensemble, rather than committing to one upstream threshold.

## Architecture Decision

**Approach:** Temp-DB + Merge

For each threshold `τ`:
1. Threshold the contour map (`contours < τ → 0`)
2. Call `ultrack.segment(foreground, contours_τ, ...)` into a **temporary working directory**
3. This writes `NodeDB` + `OverlapDB` into `{tmp}/data.db`

After all thresholds are segmented:
4. Read every `NodeDB` row from all temp DBs
5. Globally remap node IDs (collisions are certain because each temp DB uses the same `_generate_id` formula)
6. Re-serialize each `pickle` blob with the new ID
7. Forward within-threshold `OverlapDB` pairs (with remapped IDs)
8. Compute **cross-threshold overlaps** by reconstructing per-threshold labelmaps at each timepoint and using `_compute_overlaps_vectorized`
9. Write the merged `NodeDB` + `OverlapDB` into `{final_working_dir}/data.db`

Then run the standard pipeline on the merged DB:
- `inject_validated_nodes` (optional)
- `write_seed_prior_node_probs`
- `run_linking`
- `boost_validated_edges` (optional)
- `run_solve`

## Why not other approaches?

| Approach | Pros | Cons |
|---|---|---|
| **Temp-DB + Merge** (chosen) | Preserves exact Ultrack segmentation; reuses existing pipeline | Need to remap IDs; need cross-threshold overlap computation |
| Manual watershed (`skimage.watershed`) | Simpler DB construction; full ID control | Changes segmentation algorithm; need marker strategy |
| Resurrect hypotheses.h5 ingestion | Reuses old multi-partition code | Hypotheses.h5 format was deleted; resurrecting it is more code |
| Run full `build_ultrack_database` per threshold + compare results | Simplest to implement | Not a merge; solver never sees cross-threshold candidates |

## Key Technical Findings

### DB Schema
- Tables: `nodes`, `overlaps`, `links`, `gt_nodes`, `gt_links`
- **No custom indexes** on `nodes`/`overlaps`/`links`
- `nodes` PK is composite `(t, id)` — but IDs are conventionally globally unique via `_generate_id(index, t, max_segments)`
- `OverlapDB`: `(id, node_id, ancestor_id)` where `node_id > ancestor_id`
- `node_id` inside the pickle blob must match the `NodeDB.id` — so pickles must be **re-generated** when IDs change

### ID Remapping
- Each temp DB uses `id = index + (t + 1) * max_segments`
- Two temp DBs covering the same time range will have colliding IDs
- Solution: assign new IDs with a simple global counter; regenerate pickles

### Cross-Threshold Overlaps
- Within-threshold overlaps are already in each temp DB's `OverlapDB`
- Cross-threshold overlaps must be computed from the actual masks
- Method: for each time `t`, reconstruct one labelmap per threshold from `pickle` blobs, then call `_compute_overlaps_vectorized(labelmaps)`
- Requires frame dimensions `(H, W)` — inferred from max bbox extent if not provided

### Pickle Re-Generation
- Extract mask + bbox via `_node_bbox_and_mask(row.id, row.pickle)`
- Detect 2D vs 3D via `_node_pickle_ndim(pickle)`
- Re-serialize via `_make_node_pickle(t, mask, bbox, new_id, ndim=ndim)`

## Planned Module Structure

**File:** `src/cellflow/tracking_ultrack/multi_threshold.py`

```python
"""Multi-threshold Ultrack database builder."""
from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

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
from cellflow.tracking_ultrack.ingest import (
    _compute_overlaps_vectorized,
    _make_node_pickle,
)
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.seed_prior import (
    boost_validated_edges,
    write_seed_prior_node_probs,
)
from cellflow.tracking_ultrack.solve import run_solve
from cellflow.tracking_ultrack.validation_nodes import inject_validated_nodes

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiThresholdMergeReport:
    source_count: int
    nodes_per_source: list[int]
    total_nodes: int
    within_source_overlaps: list[int]
    cross_source_overlaps: int


def _read_nodes_and_overlaps(db_path: Path) -> tuple[list[dict], list[tuple[int, int]]]:
    """Read NodeDB + OverlapDB rows from a single Ultrack DB."""
    from ultrack.core.database import NodeDB, OverlapDB

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            nodes = [
                {
                    "old_id": int(r.id),
                    "t": int(r.t),
                    "z": 0.0 if r.z is None else float(r.z),
                    "y": float(r.y),
                    "x": float(r.x),
                    "area": 0 if r.area is None else int(r.area),
                    "frontier": None if r.frontier is None else float(r.frontier),
                    "height": None if r.height is None else float(r.height),
                    "pickle": r.pickle,
                    "node_prob": 1.0 if r.node_prob is None else float(r.node_prob),
                    "t_node_id": int(r.t_node_id) if r.t_node_id is not None else None,
                    "t_hier_id": int(r.t_hier_id) if r.t_hier_id is not None else None,
                    "node_annot": r.node_annot,
                    "appear_annot": r.appear_annot,
                    "disappear_annot": r.disappear_annot,
                    "division_annot": r.division_annot,
                    "segm_annot": r.segm_annot,
                }
                for r in session.query(NodeDB).all()
            ]
            overlaps = [
                (int(o.node_id), int(o.ancestor_id))
                for o in session.query(OverlapDB).all()
            ]
    finally:
        engine.dispose()
    return nodes, overlaps


def _infer_image_shape(nodes: list[dict]) -> tuple[int, int]:
    """Infer (H, W) from the maximum bounding-box extent of stored masks."""
    from cellflow.tracking_ultrack.validation_nodes import _node_bbox_and_mask

    max_y = max_x = 0
    for row in nodes:
        bb, _ = _node_bbox_and_mask(row["old_id"], row["pickle"])
        max_y = max(max_y, bb[2])
        max_x = max(max_x, bb[3])
    return max_y, max_x


def _read_ndim_from_pickle(pickle: bytes) -> int:
    from cellflow.tracking_ultrack.validation_nodes import _node_pickle_ndim
    return _node_pickle_ndim(pickle)


def merge_ultrack_databases(
    source_db_paths: Sequence[str | Path],
    output_db_path: str | Path,
    *,
    frame_shape: tuple[int, int] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> MultiThresholdMergeReport:
    """Merge several Ultrack ``data.db`` files into one.

    Node IDs are globally remapped.  Within-source overlaps are forwarded
    from each source's ``OverlapDB``.  Cross-source overlaps are computed
    by reconstructing labelmaps from the ``pickle`` blobs.
    """
    from ultrack.core.database import Base, NodeDB, OverlapDB

    src_paths = [Path(p) for p in source_db_paths]
    out_path = Path(output_db_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    _notify(progress_cb, f"Reading {len(src_paths)} source databases …")

    db_nodes: list[list[dict]] = []
    db_overlaps: list[list[tuple[int, int]]] = []
    for sp in src_paths:
        nds, ovs = _read_nodes_and_overlaps(sp)
        db_nodes.append(nds)
        db_overlaps.append(ovs)

    # Global ID remapping
    next_id = 1
    for nds in db_nodes:
        for row in nds:
            row["new_id"] = next_id
            next_id += 1

    total_nodes = sum(len(nds) for nds in db_nodes)
    _notify(progress_cb, f"Remapped {total_nodes} node ids …")

    # Forward within-source overlaps with remapped IDs
    overlap_pairs: set[tuple[int, int]] = set()
    within_counts: list[int] = []
    for nds, ovs in zip(db_nodes, db_overlaps):
        lut = {row["old_id"]: row["new_id"] for row in nds}
        count = 0
        for a, b in ovs:
            ai, bi = lut.get(a), lut.get(b)
            if ai is not None and bi is not None:
                pair = (max(ai, bi), min(ai, bi))
                if pair not in overlap_pairs:
                    overlap_pairs.add(pair)
                    count += 1
        within_counts.append(count)

    # Cross-source overlaps via reconstructed labelmaps
    if frame_shape is None:
        h, w = _infer_image_shape(
            [row for nds in db_nodes for row in nds]
        )
    else:
        h, w = frame_shape

    _notify(progress_cb, f"Computing cross-source overlaps ({h}×{w}) …")

    nodes_by_db_time: list[dict[int, list[dict]]] = [{} for _ in range(len(db_nodes))]
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
            for nds in db_nodes:
                for row in nds:
                    new_id = row["new_id"]
                    t = row["t"]
                    from cellflow.tracking_ultrack.validation_nodes import (
                        _node_bbox_and_mask,
                    )
                    bb, mask = _node_bbox_and_mask(row["old_id"], row["pickle"])
                    ndim = _read_ndim_from_pickle(row["pickle"])
                    new_pickle = _make_node_pickle(
                        t, mask, np.asarray(bb, dtype=np.int32), new_id, ndim=ndim
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

    1. For each threshold ``τ``, create ``contours_τ`` and run
       ``ultrack.segment`` into a temporary directory.
    2. Merge all temporary databases.
    3. Run scoring, linking, optional injection / boosting, and solve.
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

        # Run standard pipeline on merged DB
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
```

## Testing Strategy

1. **Unit test `merge_ultrack_databases`**
   - Create two mock Ultrack DBs with 2–3 non-overlapping / overlapping nodes each
   - Verify merged DB has correct total node count
   - Verify ID uniqueness
   - Verify within-source overlaps are forwarded
   - Verify cross-source overlaps are detected

2. **Unit test `build_multithreshold_database`**
   - Monkeypatch `_run_ultrack_segment` to write synthetic NodeDB + OverlapDB
   - Verify merge is called
   - Verify pipeline steps run in correct order

3. **Integration test** (optional, with real Ultrack)
   - Run on small synthetic `(T=3, Y=64, X=64)` volumes
   - Use 2–3 thresholds
   - Verify solver produces a non-empty solution

## Open Questions / Risks

| Risk | Mitigation |
|---|---|
| Performance: rebuilding labelmaps from pickles per timepoint | Profile on real data; if slow, add spatial binning or cached extent maps |
| `Base.metadata.create_all` missing something `ultrack.segment` creates | Verified: only 5 tables with no custom indexes |
| Large `data.db` file after merge | SQLite supports >1TB; total size is ~N×single threshold |
| `ultrack.segment` writes additional files beyond `data.db` | Verified `DataConfig`: only database and metadata.toml are produced |
