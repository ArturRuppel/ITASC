"""Ingest v2 hypothesis HDF5 directly into Ultrack's NodeDB + OverlapDB.

Bypasses ultrack.segment() — each (t, p, label_id) becomes one NodeDB row;
cross-p mask overlaps at the same t become OverlapDB pairs.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import threading
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


def _cell_mask_hash(bbox: np.ndarray, mask: np.ndarray) -> bytes:
    """Return a hash for one cell mask in frame coordinates."""
    bbox_arr = np.asarray(bbox, dtype=np.int32)
    mask_arr = np.ascontiguousarray(mask, dtype=bool)
    h = hashlib.blake2b(digest_size=16)
    h.update(bbox_arr.tobytes())
    h.update(mask_arr.tobytes())
    return h.digest()


def _generate_id(index: int, time: int, max_segments: int) -> int:
    return index + (time + 1) * max_segments


def _build_ultrack_config(cfg: TrackingConfig, working_dir: Path):
    from ultrack.config import MainConfig
    from ultrack.config.segmentationconfig import NAME_TO_WS_HIER

    ultrack_cfg = MainConfig(
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
    sc = ultrack_cfg.segmentation_config
    sc.min_area = cfg.seg_min_area
    sc.max_area = cfg.seg_max_area
    sc.threshold = cfg.seg_foreground_threshold
    sc.min_frontier = cfg.seg_min_frontier
    sc.ws_hierarchy = NAME_TO_WS_HIER[cfg.seg_ws_hierarchy]
    sc.n_workers = cfg.seg_n_workers
    return ultrack_cfg


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
    seen_cell_hashes: set[bytes] | None = None,
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

        min_r, min_c, max_r, max_c = prop.bbox
        bbox = np.array([min_r, min_c, max_r, max_c], dtype=np.int32)
        mask = (labelmap_2d[min_r:max_r, min_c:max_c] == prop.label).astype(bool)

        if forbidden_mask is not None:
            forbidden_crop = forbidden_mask[min_r:max_r, min_c:max_c]
            if np.any(mask & forbidden_crop):
                continue

        if seen_cell_hashes is not None:
            cell_hash = _cell_mask_hash(bbox, mask)
            if cell_hash in seen_cell_hashes:
                continue
            seen_cell_hashes.add(cell_hash)

        node_id = _generate_id(idx, t, max_segments)
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


def _resolve_ingest_worker_count(n_total: int, n_workers: int | None) -> int:
    """Return a worker count that avoids fork-based pools from background threads."""
    if n_workers is not None:
        return max(1, int(n_workers))
    if threading.current_thread() is not threading.main_thread():
        return 1
    return max(1, min(cpu_count(), n_total, 8))


def _make_node_pickle(t: int, mask_2d: np.ndarray, bbox: np.ndarray, node_id: int) -> bytes:
    from ultrack.core.segmentation.node import Node
    # Lift to 3D (1, h, w) so paint_buffer works against a (Z, Y, X) export buffer
    mask_3d = mask_2d[np.newaxis]
    min_y, min_x, max_y, max_x = bbox
    bbox_3d = np.array([0, int(min_y), int(min_x), 1, int(max_y), int(max_x)], dtype=np.int64)
    node = Node.from_mask(time=t, mask=mask_3d, bbox=bbox_3d, node_id=node_id)
    return pickle.dumps(node)
