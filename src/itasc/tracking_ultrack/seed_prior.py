"""Resolve-time node probability scoring from image quality and validated seeds."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import tifffile

from itasc.tracking_ultrack._node_geometry import node_bbox_and_mask
from itasc.tracking_ultrack.config import TrackingConfig


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


def _seed_node_prob(base: float, max_base: float, exponent: float) -> float:
    """Normalise ``base`` to [0, 1] and raise it to ``exponent``.

    Guards the zero-quality case: ``0 ** 0 == 1`` and ``0 ** negative == inf``
    would both corrupt the seed prior, so a non-positive normalised base scores
    a flat ``0.0`` regardless of exponent.
    """
    normalized = base / max_base if max_base > 0 else base
    if normalized <= 0.0:
        return 0.0
    return float(normalized ** exponent)


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
            NodeDB.node_annot,
        ).all()

        anchor_ids: list[int] = []
        score_updates: list[dict[str, object]] = []
        max_base = cfg.quality_weight + cfg.circularity_weight
        for node_id, t, node, annot in rows:
            if annot == VarAnnotation.REAL:
                anchor_ids.append(int(node_id))
                continue
            bbox, mask = node_bbox_and_mask(int(node_id), node)
            record = _NodeScoreRecord(
                node_id=int(node_id),
                t=int(t),
                bbox=bbox,
                mask=mask,
            )
            if record.t >= signal.shape[0]:
                raise ValueError(
                    f"Quality signal image has {signal.shape[0]} frame(s), "
                    f"cannot score node at t={record.t}"
                )
            drop_frac = compute_drop_frac(signal[record.t], record.bbox, record.mask)
            circularity = compute_mask_circularity(record.mask)
            base = cfg.quality_weight * drop_frac + cfg.circularity_weight * circularity
            node_prob = _seed_node_prob(base, max_base, cfg.quality_exponent)
            # NodeDB has a composite primary key (id, t); bulk_update_mappings
            # needs every PK column present in each row mapping.
            score_updates.append(
                {"id": record.node_id, "t": record.t, "node_prob": node_prob}
            )

        # One bulk UPDATE instead of a statement per node (was O(n) round-trips).
        if score_updates:
            session.bulk_update_mappings(NodeDB, score_updates)
        if anchor_ids:
            session.query(NodeDB).where(NodeDB.id.in_(anchor_ids)).update(
                {NodeDB.node_prob: 1.0},
                synchronize_session=False,
            )
        session.commit()
        scored = len(score_updates)

    engine.dispose()
    return SeedPriorReport(scored=scored, seeds=len(anchor_ids))
