"""Ingest v2 hypothesis HDF5 directly into Ultrack's NodeDB + OverlapDB.

Bypasses ultrack.segment() — each (t, p, label_id) becomes one NodeDB row;
cross-p mask overlaps at the same t become OverlapDB pairs.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import sqlalchemy as sqla
from skimage.measure import regionprops
from sqlalchemy.orm import Session

from cellflow.database.hypotheses import read_hypothesis_labels
from cellflow.tracking_ultrack.config import TrackingConfig

LOG = logging.getLogger(__name__)


def _canonical_hash(labelmap_2d: np.ndarray) -> bytes:
    """Return an 8-byte hash that identifies the partition structure of a 2D labelmap.

    Two labelmaps are considered duplicates if they describe the same set of cell
    regions regardless of how the labels are numbered.  We canonicalize by
    relabelling in raster-scan order of first pixel occurrence (so the first
    non-zero label encountered becomes 1, the second becomes 2, etc.) then hashing
    the resulting byte sequence with BLAKE2b.

    All-zero maps hash identically to each other and correctly to a single entry.
    No division-by-zero or empty-array edge cases: the max-value check handles both.
    """
    flat = labelmap_2d.ravel()
    max_val = int(flat.max()) if flat.size > 0 else 0
    if max_val == 0:
        # All background — hash the zero array directly (shape is already fixed
        # per frame, so all-zero maps of the same shape produce the same hash).
        return hashlib.blake2b(flat.tobytes(), digest_size=8).digest()

    # np.unique with return_index gives the first occurrence (in raster order)
    # for every label value.
    unique_labels, first_indices = np.unique(flat, return_index=True)
    # Drop background (0) — it keeps its value (0) in the canonical map.
    nonzero_mask = unique_labels > 0
    unique_labels = unique_labels[nonzero_mask]
    first_indices = first_indices[nonzero_mask]

    # Sort non-zero labels by their first-occurrence position → canonical 1, 2, 3 ...
    sort_order = np.argsort(first_indices)
    sorted_labels = unique_labels[sort_order]

    # Build a lookup table: old_label_value → canonical_label_value.
    lookup = np.zeros(max_val + 1, dtype=np.uint32)
    lookup[sorted_labels] = np.arange(1, len(sorted_labels) + 1, dtype=np.uint32)

    canonical = lookup[flat]
    return hashlib.blake2b(canonical.tobytes(), digest_size=8).digest()


def _generate_id(index: int, time: int, max_segments: int) -> int:
    return index + (time + 1) * max_segments


def _build_ultrack_config(cfg: TrackingConfig, working_dir: Path):
    from ultrack.config import MainConfig

    return MainConfig(
        data={"working_dir": str(working_dir)},
        linking={
            "max_distance": cfg.max_distance,
            "max_neighbors": cfg.max_neighbors,
            "distance_weight": cfg.distance_weight,
            "n_workers": cfg.link_n_workers,
        },
        tracking={
            "solver_name": _select_solver(),
            "appear_weight": cfg.appear_weight,
            "disappear_weight": cfg.disappear_weight,
            "division_weight": cfg.division_weight,
            "link_function": cfg.link_function,
            "power": cfg.power,
            "bias": cfg.bias,
            "solution_gap": cfg.solution_gap,
            "time_limit": cfg.time_limit,
            "window_size": cfg.window_size if cfg.window_size > 0 else None,
        },
    )


def _select_solver() -> str:
    try:
        import gurobipy  # noqa: F401
        return "GUROBI"
    except ImportError:
        return "CBC"


@dataclass
class _CellRecord:
    node_id: int
    label_val: int          # original label value in the labelmap
    p: int
    bbox: np.ndarray        # (min_y, min_x, max_y, max_x) exclusive-max
    mask: np.ndarray        # bool crop (h, w)
    area: int
    y: float
    x: float


def _extract_cell_records_2d(
    labelmap_2d: np.ndarray,
    t: int,
    p: int,
    index_start: int,
    max_segments: int,
    min_area: int | None,
    max_area: int | None,
    forbidden_mask: np.ndarray | None = None,
) -> tuple[list[_CellRecord], int]:
    """Extract CellRecords from one 2D labelmap. Returns (records, next_index).

    If ``forbidden_mask`` is given (Y×X bool), any region whose pixels intersect
    a True pixel in the mask is silently skipped — used to keep validated cells
    out of the hypothesis DB at ingest time.
    """
    records: list[_CellRecord] = []
    idx = index_start

    for prop in regionprops(labelmap_2d):
        area = int(prop.area)
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        node_id = _generate_id(idx, t, max_segments)
        min_r, min_c, max_r, max_c = prop.bbox
        bbox = np.array([min_r, min_c, max_r, max_c], dtype=np.int32)
        mask = (labelmap_2d[min_r:max_r, min_c:max_c] == prop.label).astype(bool)

        if forbidden_mask is not None:
            forbidden_crop = forbidden_mask[min_r:max_r, min_c:max_c]
            if np.any(mask & forbidden_crop):
                continue

        cy, cx = prop.centroid

        records.append(_CellRecord(
            node_id=node_id,
            label_val=int(prop.label),
            p=p,
            bbox=bbox,
            mask=mask,
            area=area,
            y=float(cy),
            x=float(cx),
        ))
        idx += 1

    return records, idx


def _build_nid_labelmap(
    labelmap_2d: np.ndarray,
    records: list[_CellRecord],
) -> np.ndarray:
    """Return a (Y, X) int64 array with node_ids where cells were kept, 0 elsewhere.

    Uses a vectorized lookup table instead of per-cell masking.
    """
    max_label = int(labelmap_2d.max())
    if max_label == 0 or not records:
        return np.zeros(labelmap_2d.shape, dtype=np.int64)
    lookup = np.zeros(max_label + 1, dtype=np.int64)
    for rec in records:
        if rec.label_val <= max_label:
            lookup[rec.label_val] = rec.node_id
    return lookup[labelmap_2d]


def _compute_overlaps_vectorized(
    nid_lms: list[np.ndarray],
) -> list[tuple[int, int]]:
    """Find cross-partition overlapping node pairs using vectorized labelmap ANDs.

    Encodes each (hi, lo) node-id pair as a single int64 (hi*MAX_ID + lo) so
    that deduplication uses a fast 1D sort rather than 2D column_stack+unique.
    All encoded pairs are concatenated and deduplicated in a single numpy call
    at the end, avoiding per-pair Python overhead.
    """
    n = len(nid_lms)
    if n == 0:
        return []

    # Pre-flatten to 1D and pre-compute nonzero masks once
    flat = [lm.ravel() for lm in nid_lms]
    nz = [f > 0 for f in flat]

    # MAX_ID must exceed every node id present in this frame so that
    # encoding hi*MAX_ID+lo is injective.  Using a fixed constant like 1e7 is
    # wrong at t>=9 (ids start at 10_000_001).  Derive it from the actual data.
    max_id_in_frame = max((int(lm.max()) for lm in nid_lms if lm.size), default=0)
    MAX_ID = max_id_in_frame + 1
    encoded_chunks: list[np.ndarray] = []

    for i in range(n):
        if not nz[i].any():
            continue
        for j in range(i + 1, n):
            combined = nz[i] & nz[j]
            if not combined.any():
                continue
            ni = flat[i][combined]
            nj = flat[j][combined]
            lo = np.minimum(ni, nj)
            hi = np.maximum(ni, nj)
            # Per-pair unique reduces the chunk size before the final dedup
            encoded_chunks.append(np.unique(hi * MAX_ID + lo))

    if not encoded_chunks:
        return []

    all_encoded = np.unique(np.concatenate(encoded_chunks))
    return [(int(e // MAX_ID), int(e % MAX_ID)) for e in all_encoded]


def _list_timepoints(hypotheses_h5: Path) -> list[int]:
    import h5py
    with h5py.File(hypotheses_h5, "r") as f:
        root = f["hypotheses"]
        return sorted(int(k[1:]) for k in root.keys() if k.startswith("t"))


def _list_partitions(hypotheses_h5: Path, t: int) -> list[int]:
    import h5py
    with h5py.File(hypotheses_h5, "r") as f:
        grp = f[f"hypotheses/t{t:03d}"]
        return sorted(int(k[1:]) for k in grp.keys() if k.startswith("p"))


def _make_node_pickle(t: int, mask_2d: np.ndarray, bbox: np.ndarray, node_id: int) -> bytes:
    from ultrack.core.segmentation.node import Node
    # Lift to 3D (1, h, w) so paint_buffer works against a (Z, Y, X) export buffer
    mask_3d = mask_2d[np.newaxis]
    min_y, min_x, max_y, max_x = bbox
    bbox_3d = np.array([0, int(min_y), int(min_x), 1, int(max_y), int(max_x)], dtype=np.int64)
    node = Node.from_mask(time=t, mask=mask_3d, bbox=bbox_3d, node_id=node_id)
    return pickle.dumps(node)


# ---------------------------------------------------------------------------
# Worker function for parallel ingest (must be module-level for pickling)
# ---------------------------------------------------------------------------

def _ingest_frame_worker(args: tuple) -> None:
    """Process one timepoint and write results to a per-frame temp SQLite DB.

    Called in a subprocess via multiprocessing.Pool.map.  Each worker opens its
    own HDF5 handle (read-only) and writes a standalone SQLite DB that the main
    process later bulk-merges into data.db via ATTACH DATABASE.

    Node IDs are timepoint-scoped: _generate_id(index, t, max_segments) =
    index + (t+1)*max_segments.  Since every frame uses a distinct t, IDs
    cannot collide across temp DBs.
    """
    (
        t,
        hypotheses_h5_path,
        max_segments,
        eff_min_area,
        eff_max_area,
        max_partitions,
        tmp_db_path_str,
        forbidden_mask,
    ) = args

    import sqlalchemy as _sqla
    from sqlalchemy.orm import Session as _Session
    from ultrack.core.database import Base as _Base, NodeDB as _NodeDB, OverlapDB as _OverlapDB
    import pandas as _pd

    hypotheses_h5 = Path(hypotheses_h5_path)
    tmp_db_path = Path(tmp_db_path_str)

    # Create isolated temp DB for this frame
    tmp_engine = _sqla.create_engine(f"sqlite:///{tmp_db_path}")
    _Base.metadata.create_all(tmp_engine)

    all_partitions = _list_partitions(hypotheses_h5, t)
    n_raw = len(all_partitions)

    all_records: list[_CellRecord] = []
    nid_lms: list[np.ndarray] = []
    index = 1

    # Deduplication — same logic as serial path
    seen_hashes: set[bytes] = set()
    n_kept = 0
    n_dropped = 0

    for p in all_partitions:
        labels = read_hypothesis_labels(hypotheses_h5, t, p)
        if labels.ndim == 3 and labels.shape[0] == 1:
            labels_2d = labels[0]
        elif labels.ndim == 2:
            labels_2d = labels
        else:
            raise NotImplementedError(f"3D ingestion not yet supported (shape {labels.shape})")

        h = _canonical_hash(labels_2d)
        if h in seen_hashes:
            n_dropped += 1
            continue
        seen_hashes.add(h)

        if max_partitions is not None and n_kept >= max_partitions:
            n_dropped += 1
            continue

        n_kept += 1
        recs, index = _extract_cell_records_2d(
            labels_2d, t, p, index, max_segments, eff_min_area, eff_max_area,
            forbidden_mask=forbidden_mask,
        )
        nid_lm = _build_nid_labelmap(labels_2d, recs)
        all_records.extend(recs)
        nid_lms.append(nid_lm)

    overlap_pairs = _compute_overlaps_vectorized(nid_lms)

    # Write Nodes
    with _Session(tmp_engine) as session:
        session.bulk_save_objects([
            _NodeDB(
                id=rec.node_id,
                t=t,
                t_node_id=rec.node_id - (t + 1) * max_segments,
                t_hier_id=rec.p + 1,
                z=0,
                y=rec.y,
                x=rec.x,
                area=rec.area,
                pickle=_make_node_pickle(t, rec.mask, rec.bbox, rec.node_id),
            )
            for rec in all_records
        ])
        session.commit()

    # Write Overlaps
    if overlap_pairs:
        df_overlaps = _pd.DataFrame(overlap_pairs, columns=["node_id", "ancestor_id"])
        with tmp_engine.begin() as conn:
            df_overlaps.to_sql(
                _OverlapDB.__tablename__, conn,
                if_exists="append", index=False,
                chunksize=50_000, method="multi",
            )

    tmp_engine.dispose()
    return (t, n_raw, n_kept, n_dropped, len(all_records), len(overlap_pairs))


def ingest_hypotheses_to_db(
    hypotheses_h5: Path,
    working_dir: Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
    min_area: int | None = None,
    max_area: int | None = None,
    max_partitions: int | None = None,
    n_frames: int | None = None,
    n_workers: int | None = None,
    forbidden_masks: dict[int, np.ndarray] | None = None,
) -> None:
    """Write v2 hypothesis HDF5 into Ultrack's NodeDB + OverlapDB.

    Parameters
    ----------
    hypotheses_h5:
        Path to v2 ``hypotheses.h5`` file.
    working_dir:
        Directory for Ultrack's ``data.db`` SQLite file.
    cfg:
        Tracking configuration (area filters, ILP parameters).
    overwrite:
        Clear existing DB before ingestion.
    min_area, max_area:
        Optional pixel-area filters applied before insert.
    max_partitions:
        Cap the number of partitions used per frame. Useful for large sweeps.
        None = use all partitions.
    n_frames:
        Limit ingestion to the first ``n_frames`` timepoints from the HDF5.
        None (default) = all timepoints.
    n_workers:
        Number of worker processes for parallel frame ingest.
        None (default) = min(cpu_count(), n_frames, 8).
        1 = serial (no subprocess overhead).
    forbidden_masks:
        Optional ``{t: bool_array (Y, X)}`` map.  Any hypothesis cell whose
        pixels intersect ``forbidden_masks[t]`` at frame ``t`` is silently
        skipped — used by the validate-and-resolve flow to keep validated
        cells out of the hypothesis DB without a separate prune pass.
    """
    import multiprocessing as _mp
    import time as _time

    from ultrack.core.database import Base, NodeDB, OverlapDB, clear_all_data  # noqa: F401

    hypotheses_h5 = Path(hypotheses_h5)
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    ultrack_cfg = _build_ultrack_config(cfg, working_dir)
    db_path = ultrack_cfg.data_config.database_path

    engine = sqla.create_engine(db_path)
    if overwrite:
        Base.metadata.create_all(engine)  # ensure tables exist so drop_all doesn't fail on fresh DB
        clear_all_data(db_path)
    Base.metadata.create_all(engine)

    eff_min_area = min_area if min_area is not None else cfg.min_area
    eff_max_area = max_area if max_area is not None else cfg.max_area
    max_segments = cfg.max_segments_per_time

    all_timepoints = _list_timepoints(hypotheses_h5)
    if n_frames is not None:
        timepoints = all_timepoints[:n_frames]
    else:
        timepoints = all_timepoints
    n_total = len(timepoints)
    LOG.info(f"Ingesting {n_total} timepoints from {hypotheses_h5}")

    # Write shape metadata required by the solver: (T, Z, Y, X)
    # Done in main process (cheap + synchronous) before workers start.
    first_t = timepoints[0]
    first_p = _list_partitions(hypotheses_h5, first_t)[0]
    sample = read_hypothesis_labels(hypotheses_h5, first_t, first_p)
    frame_shape = sample.shape  # (Z, Y, X)
    full_shape = (n_total,) + frame_shape
    ultrack_cfg.data_config.metadata_add({"shape": list(full_shape), "properties": []})

    # Determine actual worker count
    if n_workers is None:
        n_workers = min(cpu_count(), n_total, 8)
    n_workers = max(1, n_workers)

    # Temp DB directory — clean up any leftover DBs from a previous crashed run
    tmp_dir = working_dir / "_tmp_frame_dbs"
    tmp_dir.mkdir(exist_ok=True)
    for t in timepoints:
        stale = tmp_dir / f"frame_{t:04d}.db"
        stale.unlink(missing_ok=True)

    worker_args = [
        (
            t,
            str(hypotheses_h5),
            max_segments,
            eff_min_area,
            eff_max_area,
            max_partitions,
            str(tmp_dir / f"frame_{t:04d}.db"),
            (forbidden_masks.get(t) if forbidden_masks else None),
        )
        for t in timepoints
    ]

    t_start_all = _time.monotonic()

    if n_workers > 1:
        print(f"  Parallel ingest: {n_workers} workers for {n_total} frames …", flush=True)
        engine.dispose()  # close inherited connections before fork so children don't share file descriptors
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_ingest_frame_worker, worker_args)
    else:
        print(f"  Serial ingest: {n_total} frames …", flush=True)
        results = [_ingest_frame_worker(a) for a in worker_args]

    t_parallel_done = _time.monotonic()

    # Print per-frame summary (results arrive in submission order from Pool.map)
    for res in results:
        t_val, n_raw, n_kept, n_dropped, n_nodes, n_overlaps = res
        print(
            f"  t={t_val}: {n_kept}/{n_raw} unique partitions "
            f"(dropped {n_dropped}), {n_nodes} nodes, {n_overlaps} overlaps",
            flush=True,
        )

    # --- Merge temp DBs into main data.db via ATTACH DATABASE ----------------
    print(f"  Merging {n_total} temp DBs into data.db …", flush=True)
    t_merge_start = _time.monotonic()

    # Use raw SQLite with isolation_level=None (autocommit) so that explicit
    # BEGIN/COMMIT transactions do not hold a cross-database lock that would
    # prevent DETACH after the INSERT.
    import sqlite3 as _sqlite3
    main_db_file = str(db_path).replace("sqlite:///", "")
    conn = _sqlite3.connect(main_db_file, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        for t in timepoints:
            frame_db = str(tmp_dir / f"frame_{t:04d}.db")
            conn.execute(f"ATTACH DATABASE '{frame_db}' AS frame")
            conn.execute("BEGIN")
            conn.execute(f"INSERT INTO {NodeDB.__tablename__} SELECT * FROM frame.{NodeDB.__tablename__}")
            # OverlapDB has an auto-increment rowid 'id' — exclude it so SQLite
            # assigns fresh unique rowids in the main DB.
            conn.execute(
                f"INSERT INTO {OverlapDB.__tablename__} (node_id, ancestor_id) "
                f"SELECT node_id, ancestor_id FROM frame.{OverlapDB.__tablename__}"
            )
            conn.execute("COMMIT")
            conn.execute("DETACH DATABASE frame")
    finally:
        conn.close()

    t_merge_done = _time.monotonic()

    # Clean up temp DBs
    for t in timepoints:
        frame_db = tmp_dir / f"frame_{t:04d}.db"
        frame_db.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass  # not empty — leave it

    total = _time.monotonic() - t_start_all
    parallel_s = t_parallel_done - t_start_all
    merge_s = t_merge_done - t_parallel_done
    print(
        f"  Ingest complete — {n_total} frames in {total/60:.1f}min "
        f"(parallel={parallel_s:.1f}s, merge={merge_s:.1f}s)",
        flush=True,
    )
    LOG.info("Ingestion complete.")
