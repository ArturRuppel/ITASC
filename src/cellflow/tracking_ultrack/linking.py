"""Linking step: wire NodeDB into LinkDB.

Default mode uses Ultrack's built-in linker.
Shape mode uses the custom shape-scoring linker.
"""
from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Generator

import numpy as np

from cellflow.tracking_ultrack import scoring
from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.ingest import _build_ultrack_config


def run_linking(
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Run the linking step, yielding (step, total, label) progress tuples."""
    if cfg.linking_mode == "shape":
        yield from _run_shape_linking(working_dir, cfg, overwrite=overwrite)
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
# Shape-scoring linking
# ---------------------------------------------------------------------------

def _node_coords_centroid(node, ndim: int | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (absolute-pixel coords, centroid) for node, or None on failure.

    ndim defaults to the mask's own dimensionality.
    """
    mask = getattr(node, "mask", None)
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 0 or not mask.any():
        return None
    if ndim is None:
        ndim = int(mask.ndim)

    # Resolve origin
    origin = None
    for attr in ("origin", "offset", "bbox_start", "bbox_min", "start"):
        value = getattr(node, attr, None)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= ndim:
            origin = arr[-ndim:]
            break
    if origin is None:
        bbox = getattr(node, "bbox", None)
        if bbox is None:
            return None
        if isinstance(bbox, tuple) and bbox and all(hasattr(item, "start") for item in bbox):
            starts = [0.0 if item.start is None else float(item.start) for item in bbox]
            arr = np.asarray(starts, dtype=np.float32).reshape(-1)
            if arr.size >= ndim:
                origin = arr[-ndim:]
        if origin is None:
            arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
            # bbox is [min_0, ..., min_k, max_0, ..., max_k]; take first half for mins
            if arr.size >= ndim * 2 and arr.size % 2 == 0:
                half = arr.size // 2
                origin = arr[:half][-ndim:]
            elif arr.size >= ndim:
                origin = arr[-ndim:]
        if origin is None:
            return None

    coords = np.argwhere(mask).astype(np.float32) + origin
    centroid = np.asarray(node.centroid, dtype=np.float32).reshape(-1)[-ndim:]
    return coords, centroid


def _shape_pair_score(
    src_coords: np.ndarray,
    src_centroid: np.ndarray,
    src_area: float,
    tgt_coords: np.ndarray,
    tgt_centroid: np.ndarray,
    tgt_area: float,
    distance: float,
    cfg: TrackingConfig,
) -> float | None:
    """Gate and score one source→target pair under shape mode.

    Returns None when the pair fails the area-ratio or IoU threshold,
    otherwise returns the similarity score.
    """
    if src_area <= 0 or tgt_area <= 0:
        return None
    area_ratio = min(src_area, tgt_area) / max(src_area, tgt_area)
    if area_ratio < cfg.min_area_ratio:
        return None
    iou = scoring.centroid_corrected_iou_from_coords(
        src_coords, src_centroid, tgt_coords, tgt_centroid
    )
    if iou < cfg.min_link_iou:
        return None
    return scoring.similarity_score(
        area_ratio=area_ratio,
        centroid_corrected_iou=iou,
        distance=distance,
        area_weight=cfg.area_weight,
        iou_weight=cfg.iou_weight,
        distance_weight=cfg.distance_weight,
    )


def compute_edge_weight(
    source_node,
    target_node,
    distance: float,
    cfg: TrackingConfig,
) -> float | None:
    """Per-pair edge weight matching the active linker mode.

    Returns None when shape mode filters the pair out. Default mode never
    filters and always returns a float.
    """
    if cfg.linking_mode == "shape":
        result = _node_coords_centroid(source_node)
        if result is None:
            return None
        src_coords, src_centroid = result

        result = _node_coords_centroid(target_node)
        if result is None:
            return None
        tgt_coords, tgt_centroid = result

        src_area = float(getattr(source_node, "area", len(src_coords)))
        tgt_area = float(getattr(target_node, "area", len(tgt_coords)))
        return _shape_pair_score(
            src_coords, src_centroid, src_area,
            tgt_coords, tgt_centroid, tgt_area,
            distance, cfg,
        )

    if cfg.linking_mode != "default":
        raise ValueError(f"Unknown linking_mode={cfg.linking_mode!r}")
    iou = float(source_node.IoU(target_node))
    return iou - cfg.distance_weight * float(distance)


def _run_shape_linking(
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
        yield (total, total, "No frames; skipping shape linking.")
        return

    yield (1, total, "Computing shape-weighted links…")
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
            for i, row in enumerate(target_rows):
                shift = np.asarray(row[1:], dtype=np.float32)
                target_pos[i] += shift[-target_pos.shape[1]:]

            tree = KDTree(source_pos)
            k = min(len(source_nodes), max(1, 2 * cfg.max_neighbors))
            dists, neigh_idx = tree.query(target_pos, k=k, distance_upper_bound=cfg.max_distance)
            if dists.ndim == 1:
                dists, neigh_idx = dists[:, None], neigh_idx[:, None]

            # Cache (coords, centroid, area) per source node id
            src_cache: dict[int, tuple[np.ndarray, np.ndarray, float] | None] = {}

            src_ids, tgt_ids, weights = [], [], []
            for ti, (dist_row, ni_row) in enumerate(zip(dists, neigh_idx)):
                target = target_nodes[ti]
                candidates = []
                for dist, si in zip(dist_row, ni_row):
                    if si >= len(source_nodes) or not np.isfinite(dist):
                        continue
                    source = source_nodes[si]
                    sid = int(source.id)
                    if sid not in src_cache:
                        result = _node_coords_centroid(source)
                        if result is None:
                            src_cache[sid] = None
                        else:
                            src_coords, src_centroid = result
                            src_area = float(getattr(source, "area", len(src_coords)))
                            src_cache[sid] = (src_coords, src_centroid, src_area)
                    cached = src_cache[sid]
                    if cached is None:
                        continue
                    src_coords, src_centroid, src_area = cached

                    tgt_result = _node_coords_centroid(target)
                    if tgt_result is None:
                        continue
                    tgt_coords, tgt_centroid = tgt_result
                    tgt_area = float(getattr(target, "area", len(tgt_coords)))

                    w = _shape_pair_score(
                        src_coords, src_centroid, src_area,
                        tgt_coords, tgt_centroid, tgt_area,
                        float(dist), cfg,
                    )
                    if w is None:
                        continue
                    candidates.append((w, sid, int(target.id)))

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

    yield (total, total, f"Shape linking done ({total_links} total edges).")
