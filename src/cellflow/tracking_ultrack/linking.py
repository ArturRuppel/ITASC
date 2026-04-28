"""Linking step: wire NodeDB into LinkDB.

Default mode uses Ultrack's built-in linker.
IoU mode uses the custom IoU-weighted linker lifted from v1.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Generator

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.ingest import _build_ultrack_config


def run_linking(
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Run the linking step, yielding (step, total, label) progress tuples."""
    if cfg.linking_mode == "iou":
        yield from _run_iou_linking(working_dir, cfg, overwrite=overwrite)
        return
    if cfg.linking_mode != "default":
        raise ValueError(f"Unknown linking_mode={cfg.linking_mode!r}")

    total = 3
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    from ultrack.core.linking.processing import link
    from ultrack.core.linking.utils import clear_linking_data

    if overwrite:
        yield (0, total, "Clearing existing links…")
        clear_linking_data(ultrack_cfg.data_config.database_path)
    else:
        yield (0, total, "Skipping link clear (overwrite=False)…")

    yield (1, total, "Running Ultrack linker…")
    link(ultrack_cfg, overwrite=False)

    yield (total, total, "Linking done.")


# ---------------------------------------------------------------------------
# IoU-aware linking (lifted from archive/v1/…/ultrack/linking.py)
# ---------------------------------------------------------------------------

def _node_mask(node) -> np.ndarray | None:
    mask = getattr(node, "mask", None)
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    return mask if mask.ndim > 0 else None


def _node_origin(node, ndim: int) -> np.ndarray | None:
    for attr in ("origin", "offset", "bbox_start", "bbox_min", "start"):
        value = getattr(node, attr, None)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= ndim:
            return arr[-ndim:]
    bbox = getattr(node, "bbox", None)
    if bbox is None:
        return None
    if isinstance(bbox, tuple) and bbox and all(hasattr(item, "start") for item in bbox):
        starts = [0.0 if item.start is None else float(item.start) for item in bbox]
        arr = np.asarray(starts, dtype=np.float32).reshape(-1)
        if arr.size >= ndim:
            return arr[-ndim:]
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size >= ndim:
        return arr[-ndim:]
    return None


def _centroid_tail(centroid: np.ndarray, ndim: int) -> np.ndarray:
    centroid = np.asarray(centroid, dtype=np.float32).reshape(-1)
    return centroid[-ndim:]


def _aligned_mask_iou(source, target) -> float:
    source_mask = _node_mask(source)
    target_mask = _node_mask(target)
    if source_mask is None or target_mask is None:
        return float(source.IoU(target))
    if source_mask.ndim != target_mask.ndim or source_mask.ndim == 0:
        return float(source.IoU(target))
    if not source_mask.any() or not target_mask.any():
        return 0.0

    ndim = int(source_mask.ndim)
    source_origin = _node_origin(source, ndim)
    target_origin = _node_origin(target, ndim)
    if source_origin is None or target_origin is None:
        return float(source.IoU(target))

    source_coords = np.argwhere(source_mask).astype(np.float32) + source_origin
    target_coords = np.argwhere(target_mask).astype(np.float32) + target_origin
    centroid_shift = _centroid_tail(source.centroid, ndim) - _centroid_tail(target.centroid, ndim)
    target_coords = target_coords + centroid_shift

    all_coords = np.vstack([source_coords, target_coords])
    mins = np.floor(all_coords.min(axis=0)).astype(int) - 1
    maxs = np.ceil(all_coords.max(axis=0)).astype(int) + 1
    shape = tuple((maxs - mins + 1).tolist())
    if any(dim <= 0 for dim in shape):
        return 0.0

    def _rasterize(coords: np.ndarray) -> np.ndarray:
        canvas = np.zeros(shape, dtype=bool)
        idx = np.rint(coords - mins).astype(int)
        valid = np.ones(len(idx), dtype=bool)
        for axis, size in enumerate(shape):
            valid &= (idx[:, axis] >= 0) & (idx[:, axis] < size)
        idx = idx[valid]
        if idx.size:
            canvas[tuple(idx.T)] = True
        return canvas

    source_canvas = _rasterize(source_coords)
    target_canvas = _rasterize(target_coords)
    union = np.logical_or(source_canvas, target_canvas).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(source_canvas, target_canvas).sum() / union)


def _blend_score(iou: float, distance: float, max_distance: float, iou_weight: float) -> float:
    iou_weight = float(np.clip(iou_weight, 0.0, 1.0))
    distance_score = max(0.0, 1.0 - float(distance) / float(max_distance))
    return (1.0 - iou_weight) * distance_score + float(np.clip(iou, 0.0, 1.0)) * iou_weight


def _run_iou_linking(
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    total = 4
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    from ultrack.core.database import NodeDB, maximum_time_from_database
    from ultrack.core.linking.processing import add_links
    from ultrack.core.linking.utils import clear_linking_data
    from scipy.spatial import KDTree
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session

    if overwrite:
        yield (0, total, "Clearing existing links…")
        clear_linking_data(ultrack_cfg.data_config.database_path)
    else:
        yield (0, total, "Skipping link clear (overwrite=False)…")

    engine = sqla.create_engine(ultrack_cfg.data_config.database_path)
    max_t = int(maximum_time_from_database(ultrack_cfg.data_config))
    if max_t <= 0:
        yield (total, total, "No frames; skipping IoU linking.")
        return

    yield (1, total, "Computing IoU-weighted links…")
    total_links = 0

    with Session(engine) as session:
        for time in range(max_t):
            source_nodes = [n for (n,) in session.query(NodeDB.pickle).where(NodeDB.t == time)]
            target_rows = list(
                session.query(NodeDB.pickle, NodeDB.z_shift, NodeDB.y_shift, NodeDB.x_shift)
                .where(NodeDB.t == time + 1)
            )
            if not source_nodes or not target_rows:
                continue
            target_nodes = [r[0] for r in target_rows]

            source_pos = np.array([n.centroid for n in source_nodes], dtype=np.float32)
            target_pos = np.array([n.centroid for n in target_nodes], dtype=np.float32)
            # apply shift if non-zero
            for i, row in enumerate(target_rows):
                shift = np.asarray(row[1:], dtype=np.float32)
                target_pos[i] += shift[-target_pos.shape[1]:]

            tree = KDTree(source_pos)
            k = min(len(source_nodes), max(1, 2 * cfg.max_neighbors))
            dists, neigh_idx = tree.query(target_pos, k=k, distance_upper_bound=cfg.max_distance)
            if dists.ndim == 1:
                dists, neigh_idx = dists[:, None], neigh_idx[:, None]

            src_ids, tgt_ids, weights = [], [], []
            for ti, (dist_row, ni_row) in enumerate(zip(dists, neigh_idx)):
                target = target_nodes[ti]
                candidates = []
                for dist, si in zip(dist_row, ni_row):
                    if si >= len(source_nodes) or not np.isfinite(dist):
                        continue
                    source = source_nodes[si]
                    iou = _aligned_mask_iou(source, target)
                    if iou < cfg.min_link_iou:
                        continue
                    w = _blend_score(iou, dist, cfg.max_distance, cfg.iou_weight)
                    if w > 0:
                        candidates.append((w, int(source.id), int(target.id)))
                candidates.sort(reverse=True)
                for w, sid, tid in candidates[:cfg.max_neighbors]:
                    src_ids.append(sid)
                    tgt_ids.append(tid)
                    weights.append(w)

            if src_ids:
                add_links(ultrack_cfg, src_ids, tgt_ids, weights)
                total_links += len(src_ids)

            yield (
                1 + int(math.floor((time + 1) / max(max_t, 1) * 2)),
                total,
                f"Linked t={time + 1}/{max_t} ({len(src_ids)} edges)",
            )

    yield (total, total, f"IoU linking done ({total_links} total edges).")
