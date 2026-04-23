"""Custom Ultrack linking helpers with IoU-weighted edge scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Protocol, Sequence

import numpy as np

try:  # Optional, but available in the Ultrack runtime.
    from scipy.spatial import KDTree  # type: ignore
except Exception:  # pragma: no cover - exercised when SciPy is unavailable.
    KDTree = None  # type: ignore[assignment]


class _LinkNode(Protocol):
    """Minimal node surface used by the linker."""

    id: int
    centroid: np.ndarray

    def IoU(self, other: "_LinkNode") -> float: ...


@dataclass(frozen=True)
class WeightedLink:
    """A single candidate link ready for bulk insertion."""

    source_id: int
    target_id: int
    weight: float
    iou: float
    distance: float


def _node_mask(node) -> np.ndarray | None:
    """Return the node mask as a boolean array when available."""
    mask = getattr(node, "mask", None)
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    return mask if mask.ndim > 0 else None


def _centroid_tail(centroid: np.ndarray, ndim: int) -> np.ndarray:
    """Return the spatial tail of a centroid vector."""
    centroid = np.asarray(centroid, dtype=np.float32).reshape(-1)
    if centroid.size < ndim:
        raise ValueError("centroid has fewer dimensions than the mask")
    return centroid[-ndim:]


def _node_origin(node, ndim: int) -> np.ndarray | None:
    """Return the crop origin for a node when it is exposed by the object."""
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


def _aligned_mask_iou(source: _LinkNode, target: _LinkNode) -> float:
    """Compute IoU after translating the target centroid onto the source centroid.

    This keeps the overlap term focused on shape agreement instead of folding in
    pure translation, which is already captured by the centroid-distance term.
    """
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

    source_coords = np.argwhere(source_mask).astype(np.float32)
    target_coords = np.argwhere(target_mask).astype(np.float32)
    source_coords = source_coords + source_origin
    target_coords = target_coords + target_origin

    # Translate the target mask so its centroid matches the source centroid.
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
    inter = np.logical_and(source_canvas, target_canvas).sum()
    return float(inter / union)


def _blend_link_score(iou: float, distance: float, max_distance: float, iou_weight: float) -> float:
    """Blend spatial proximity and IoU into a single link score.

    The score is in [0, 1] when ``iou_weight`` is in [0, 1]. Higher is better.
    """
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")

    iou_weight = float(np.clip(iou_weight, 0.0, 1.0))
    iou = float(np.clip(iou, 0.0, 1.0))
    distance_score = max(0.0, 1.0 - float(distance) / float(max_distance))
    return (1.0 - iou_weight) * distance_score + iou_weight * iou


def _as_centroid_array(nodes: Sequence[_LinkNode]) -> np.ndarray:
    if not nodes:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray([np.asarray(node.centroid, dtype=np.float32) for node in nodes], dtype=np.float32)


def _candidate_neighbors(
    source_pos: np.ndarray,
    target_pos: np.ndarray,
    *,
    max_distance: float,
    max_neighbors: int,
) -> list[list[tuple[int, float]]]:
    """Return candidate source indices and distances for each target node."""
    n_source = int(source_pos.shape[0])
    n_target = int(target_pos.shape[0])
    if n_source == 0 or n_target == 0:
        return [[] for _ in range(n_target)]

    k = min(n_source, max(1, 2 * int(max_neighbors)))

    if KDTree is not None:
        tree = KDTree(source_pos)
        distances, neighbors = tree.query(
            target_pos,
            k=k,
            distance_upper_bound=float(max_distance),
        )
        distances = np.asarray(distances)
        neighbors = np.asarray(neighbors)
        if distances.ndim == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]
        candidate_lists: list[list[tuple[int, float]]] = []
        for dist_row, neigh_row in zip(distances, neighbors):
            row: list[tuple[int, float]] = []
            for dist, neigh_idx in zip(dist_row, neigh_row):
                neigh_idx = int(neigh_idx)
                if neigh_idx >= n_source or not np.isfinite(dist):
                    continue
                row.append((neigh_idx, float(dist)))
            candidate_lists.append(row)
        return candidate_lists

    # Fallback: dense pairwise distance matrix. This is still vectorized and is
    # only used when SciPy is unavailable.
    diff = target_pos[:, None, :] - source_pos[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    candidate_lists = []
    for dist_row in distances:
        valid = np.where(dist_row <= float(max_distance))[0]
        if valid.size == 0:
            candidate_lists.append([])
            continue
        order = np.argsort(dist_row[valid])[:k]
        candidate_lists.append([(int(valid[i]), float(dist_row[valid[i]])) for i in order])
    return candidate_lists


def compute_weighted_links(
    source_nodes: Sequence[_LinkNode],
    target_nodes: Sequence[_LinkNode],
    *,
    max_distance: float,
    max_neighbors: int,
    iou_weight: float,
    min_link_iou: float = 0.1,
) -> list[WeightedLink]:
    """Compute IoU-weighted candidate edges between two consecutive frames."""
    if max_neighbors <= 0:
        return []

    source_pos = _as_centroid_array(source_nodes)
    target_pos = _as_centroid_array(target_nodes)
    candidate_lists = _candidate_neighbors(
        source_pos,
        target_pos,
        max_distance=max_distance,
        max_neighbors=max_neighbors,
    )

    links: list[WeightedLink] = []
    for target_idx, candidates in enumerate(candidate_lists):
        target = target_nodes[target_idx]
        neighborhood: list[tuple[float, float, float, int, int]] = []
        for source_idx, distance in candidates:
            source = source_nodes[source_idx]
            iou = _aligned_mask_iou(source, target)
            if iou < float(min_link_iou):
                continue
            weight = _blend_link_score(iou, distance, max_distance, iou_weight)
            if weight <= 0:
                continue
            neighborhood.append((weight, -distance, iou, int(source.id), int(target.id)))

        neighborhood.sort(reverse=True)
        for weight, neg_distance, iou, source_id, target_id in neighborhood[: int(max_neighbors)]:
            links.append(
                WeightedLink(
                    source_id=source_id,
                    target_id=target_id,
                    weight=float(weight),
                    iou=float(iou),
                    distance=float(-neg_distance),
                )
            )

    return links


def _timepoint_nodes(
    session,
    *,
    time: int,
    node_db,
) -> tuple[list[_LinkNode], list[_LinkNode], np.ndarray]:
    """Load source and target nodes for a single consecutive frame pair."""
    current_nodes = [node for node, in session.query(node_db.pickle).where(node_db.t == time)]
    query = session.query(
        node_db.pickle,
        node_db.z_shift,
        node_db.y_shift,
        node_db.x_shift,
    ).where(node_db.t == time + 1)
    next_rows = list(query)
    next_nodes = [row[0] for row in next_rows]
    next_shift = np.asarray([row[1:] for row in next_rows], dtype=np.float32) if next_rows else np.empty((0, 3), dtype=np.float32)
    return current_nodes, next_nodes, next_shift


def run_iou_linking(
    working_dir: str | Path,
    cfg,
    *,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Insert IoU-weighted links into the Ultrack database."""
    total = 4
    wd = Path(working_dir)
    from cellflow.ultrack.stages.tracking import _build_ultrack_config

    ultrack_cfg = _build_ultrack_config(cfg, wd)

    from ultrack.core.database import NodeDB, maximum_time_from_database
    from ultrack.core.linking.processing import add_links
    from ultrack.core.linking.utils import clear_linking_data

    if overwrite:
        yield (0, total, "Clearing existing links from DB...")
        clear_linking_data(ultrack_cfg.data_config.database_path)
    else:
        yield (0, total, "Skipping DB clear (overwrite=False)...")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    engine = create_engine(ultrack_cfg.data_config.database_path, hide_parameters=True)
    max_t = int(maximum_time_from_database(ultrack_cfg.data_config))
    if max_t <= 0:
        yield (total, total, "No frames found; skipping IoU linking.")
        return

    yield (1, total, "Computing IoU-weighted links...")
    total_links = 0
    with Session(engine) as session:
        for time in range(max_t):
            source_nodes, target_nodes, target_shift = _timepoint_nodes(session, time=time, node_db=NodeDB)
            if not source_nodes or not target_nodes:
                continue

            target_pos = _as_centroid_array(target_nodes)
            n_dim = target_pos.shape[1]
            if target_shift.size > 0:
                target_shift = target_shift[:, -n_dim:]
                target_pos = target_pos + target_shift

            source_pos = _as_centroid_array(source_nodes)
            candidate_lists = _candidate_neighbors(
                source_pos,
                target_pos,
                max_distance=float(cfg.max_distance),
                max_neighbors=int(cfg.max_neighbors),
            )

            links: list[WeightedLink] = []
            for target_idx, candidates in enumerate(candidate_lists):
                target = target_nodes[target_idx]
                neighborhood: list[tuple[float, float, float, int, int]] = []
                for source_idx, distance in candidates:
                    source = source_nodes[source_idx]
                    iou = _aligned_mask_iou(source, target)
                    if iou < float(cfg.min_link_iou):
                        continue
                    weight = _blend_link_score(iou, distance, float(cfg.max_distance), float(cfg.iou_weight))
                    if weight <= 0:
                        continue
                    neighborhood.append((weight, -distance, iou, int(source.id), int(target.id)))

                neighborhood.sort(reverse=True)
                for weight, neg_distance, iou, source_id, target_id in neighborhood[: int(cfg.max_neighbors)]:
                    links.append(
                        WeightedLink(
                            source_id=source_id,
                            target_id=target_id,
                            weight=float(weight),
                            iou=float(iou),
                            distance=float(-neg_distance),
                        )
                    )

            if links:
                add_links(
                    ultrack_cfg,
                    [link.source_id for link in links],
                    [link.target_id for link in links],
                    [link.weight for link in links],
                )
                total_links += len(links)

            yield (
                1 + int(math.floor((time + 1) / max(max_t, 1) * 2)),
                total,
                f"Linked timepoint {time + 1}/{max_t} ({len(links)} edges).",
            )

    yield (total, total, f"IoU linking done ({total_links} edges).")
