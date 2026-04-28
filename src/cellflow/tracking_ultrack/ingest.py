"""Ingest v2 hypothesis HDF5 directly into Ultrack's NodeDB + OverlapDB.

Bypasses ultrack.segment() — each (t, p, label_id) becomes one NodeDB row;
cross-p mask overlaps at the same t become OverlapDB pairs.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlalchemy as sqla
from skimage.measure import regionprops
from sqlalchemy.orm import Session

from cellflow.database.hypotheses import read_hypothesis_labels
from cellflow.tracking_ultrack.config import TrackingConfig

LOG = logging.getLogger(__name__)


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
) -> tuple[list[_CellRecord], int]:
    """Extract CellRecords from one 2D labelmap. Returns (records, next_index)."""
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


def ingest_hypotheses_to_db(
    hypotheses_h5: Path,
    working_dir: Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
    min_area: int | None = None,
    max_area: int | None = None,
    max_partitions: int | None = None,
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
    """
    from ultrack.core.database import Base, NodeDB, OverlapDB, clear_all_data

    hypotheses_h5 = Path(hypotheses_h5)
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    ultrack_cfg = _build_ultrack_config(cfg, working_dir)
    db_path = ultrack_cfg.data_config.database_path

    if overwrite:
        clear_all_data(db_path)

    engine = sqla.create_engine(db_path)
    Base.metadata.create_all(engine)

    eff_min_area = min_area if min_area is not None else cfg.min_area
    eff_max_area = max_area if max_area is not None else cfg.max_area
    max_segments = cfg.max_segments_per_time

    timepoints = _list_timepoints(hypotheses_h5)
    LOG.info(f"Ingesting {len(timepoints)} timepoints from {hypotheses_h5}")

    # Write shape metadata required by the solver: (T, Z, Y, X)
    first_t = timepoints[0]
    first_p = _list_partitions(hypotheses_h5, first_t)[0]
    sample = read_hypothesis_labels(hypotheses_h5, first_t, first_p)
    frame_shape = sample.shape  # (Z, Y, X)
    full_shape = (len(timepoints),) + frame_shape
    ultrack_cfg.data_config.metadata_add({"shape": list(full_shape), "properties": []})

    import time as _time

    t_start_all = _time.monotonic()
    n_total = len(timepoints)

    for t_idx, t in enumerate(timepoints):
        t_frame_start = _time.monotonic()

        all_partitions = _list_partitions(hypotheses_h5, t)
        if max_partitions is not None:
            all_partitions = all_partitions[:max_partitions]
        n_p = len(all_partitions)

        print(f"  t={t} ({t_idx+1}/{n_total}) — loading {n_p} partitions …", flush=True)

        all_records: list[_CellRecord] = []
        nid_lms: list[np.ndarray] = []
        index = 1

        for p in all_partitions:
            labels = read_hypothesis_labels(hypotheses_h5, t, p)
            if labels.ndim == 3 and labels.shape[0] == 1:
                labels_2d = labels[0]
            elif labels.ndim == 2:
                labels_2d = labels
            else:
                raise NotImplementedError(f"3D ingestion not yet supported (shape {labels.shape})")

            recs, index = _extract_cell_records_2d(
                labels_2d, t, p, index, max_segments, eff_min_area, eff_max_area
            )
            nid_lm = _build_nid_labelmap(labels_2d, recs)
            all_records.extend(recs)
            nid_lms.append(nid_lm)

        t_load = _time.monotonic() - t_frame_start
        print(f"         {len(all_records)} nodes loaded in {t_load:.1f}s — detecting overlaps …", flush=True)

        t_overlap_start = _time.monotonic()
        overlap_pairs = _compute_overlaps_vectorized(nid_lms)
        t_overlap = _time.monotonic() - t_overlap_start
        print(f"         {len(overlap_pairs)} overlap pairs in {t_overlap:.1f}s — writing to DB …", flush=True)

        t_db_start = _time.monotonic()
        import pandas as pd

        # Nodes — build pickles and insert via ORM (manageable count)
        with Session(engine) as session:
            session.bulk_save_objects([
                NodeDB(
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

        # Overlaps — use pandas to_sql for large-scale inserts (10–50× faster than ORM)
        if overlap_pairs:
            df_overlaps = pd.DataFrame(overlap_pairs, columns=["node_id", "ancestor_id"])
            with engine.begin() as conn:
                df_overlaps.to_sql(
                    OverlapDB.__tablename__, conn,
                    if_exists="append", index=False,
                    chunksize=50_000, method="multi",
                )
        t_db = _time.monotonic() - t_db_start

        t_frame = _time.monotonic() - t_frame_start
        elapsed = _time.monotonic() - t_start_all
        frames_done = t_idx + 1
        eta_s = elapsed / frames_done * (n_total - frames_done)
        eta_str = f"{int(eta_s//60)}m{int(eta_s%60):02d}s" if eta_s >= 60 else f"{eta_s:.0f}s"
        print(
            f"         DB write {t_db:.1f}s — frame total {t_frame:.1f}s "
            f"| {frames_done}/{n_total} done, ETA {eta_str}",
            flush=True,
        )

    total = _time.monotonic() - t_start_all
    print(f"  Ingestion complete — {n_total} frames in {total/60:.1f}min", flush=True)
    LOG.info("Ingestion complete.")
