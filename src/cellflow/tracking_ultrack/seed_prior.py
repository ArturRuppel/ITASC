"""Resolve-time node probability scoring from image quality and validated seeds."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.validation_nodes import _node_bbox_and_mask as _node_mask_record


@dataclass(frozen=True)
class SeedPriorReport:
    scored: int
    seeds: int


@dataclass(frozen=True)
class _NodeScoreRecord:
    node_id: int
    t: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area: int
    y: float
    x: float


def _load_signal_stack(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Quality signal image not found: {path}")

    arr = np.asarray(tifffile.imread(path), dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError(
        f"Expected quality signal image to be 2D, 3D, or singleton-Z 4D, got {arr.shape}"
    )


def _binary_dilation_2d(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, mode="constant")
    h, w = mask.shape
    dilated = np.zeros((h, w), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            dilated |= padded[dy:dy + h, dx:dx + w]
    return dilated


def compute_drop_frac(frame: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray) -> float:
    y0, x0, y1, x1 = bbox
    frame = np.asarray(frame, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    inside = frame[y0:y1, x0:x1][mask]
    if inside.size == 0:
        return 0.0
    inside_median = float(np.median(inside))

    pad_y0 = max(0, y0 - 1)
    pad_x0 = max(0, x0 - 1)
    pad_y1 = min(frame.shape[0], y1 + 1)
    pad_x1 = min(frame.shape[1], x1 + 1)

    expanded = np.zeros((pad_y1 - pad_y0, pad_x1 - pad_x0), dtype=bool)
    inner_y0 = y0 - pad_y0
    inner_x0 = x0 - pad_x0
    expanded[
        inner_y0:inner_y0 + mask.shape[0],
        inner_x0:inner_x0 + mask.shape[1],
    ] = mask
    ring = _binary_dilation_2d(expanded) & ~expanded
    if not ring.any():
        return 0.0

    ring_values = frame[pad_y0:pad_y1, pad_x0:pad_x1][ring]
    return float(np.mean(ring_values < inside_median))


def compute_mask_circularity(mask: np.ndarray) -> float:
    from skimage.measure import perimeter

    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area == 0:
        return 0.0

    perimeter_px = float(perimeter(mask, neighborhood=4))
    if perimeter_px <= 0.0:
        return 0.0

    circularity = 4.0 * math.pi * float(area) / (perimeter_px * perimeter_px)
    return float(np.clip(circularity, 0.0, 1.0))


def _affinity(node: _NodeScoreRecord, seed: _NodeScoreRecord, cfg: TrackingConfig) -> float:
    if node.area <= 0 or seed.area <= 0:
        return 0.0
    dt = abs(int(node.t) - int(seed.t))
    if dt > cfg.seed_max_dt:
        return 0.0

    area_ratio = float(node.area) / float(seed.area)
    size_similarity = np.exp(-abs(np.log(area_ratio)) / cfg.seed_sigma_area)
    dist = float(np.hypot(node.y - seed.y, node.x - seed.x))
    spatial_decay = np.exp(-((dist / cfg.seed_sigma_space) ** 2))
    temporal_decay = np.exp(-(dt / cfg.seed_tau_time))
    return float(size_similarity * spatial_decay * temporal_decay)


def write_seed_prior_node_probs(
    working_dir: str | Path,
    intensity_image_path: str | Path,
    cfg: TrackingConfig,
) -> SeedPriorReport:
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, VarAnnotation

    signal = _load_signal_stack(intensity_image_path)
    engine = sqla.create_engine(f"sqlite:///{Path(working_dir) / 'data.db'}")

    with Session(engine) as session:
        rows = session.query(
            NodeDB.id,
            NodeDB.t,
            NodeDB.pickle,
            NodeDB.area,
            NodeDB.y,
            NodeDB.x,
            NodeDB.node_annot,
        ).all()

        records: list[_NodeScoreRecord] = []
        seed_records: list[_NodeScoreRecord] = []
        for node_id, t, node, area, y, x, annot in rows:
            bbox, mask = _node_mask_record(int(node_id), node)
            record = _NodeScoreRecord(
                node_id=int(node_id),
                t=int(t),
                bbox=bbox,
                mask=mask,
                area=int(area),
                y=float(y),
                x=float(x),
            )
            if annot == VarAnnotation.REAL:
                seed_records.append(record)
            else:
                records.append(record)

        scored = 0
        for record in records:
            if record.t >= signal.shape[0]:
                raise ValueError(
                    f"Quality signal image has {signal.shape[0]} frame(s), "
                    f"cannot score node at t={record.t}"
                )
            drop_frac = compute_drop_frac(signal[record.t], record.bbox, record.mask)
            best_affinity = max(
                (_affinity(record, seed, cfg) for seed in seed_records),
                default=0.0,
            )
            circularity = compute_mask_circularity(record.mask)
            node_prob = float(
                cfg.quality_weight * (drop_frac ** cfg.quality_exponent)
                + cfg.circularity_weight * circularity
                + cfg.seed_weight * best_affinity
            )
            session.query(NodeDB).where(NodeDB.id == record.node_id).update(
                {NodeDB.node_prob: node_prob},
                synchronize_session=False,
            )
            scored += 1

        for seed in seed_records:
            session.query(NodeDB).where(NodeDB.id == seed.node_id).update(
                {NodeDB.node_prob: 1.0},
                synchronize_session=False,
            )
        session.commit()

    engine.dispose()
    return SeedPriorReport(scored=scored, seeds=len(seed_records))
