from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cellflow.tracking_ultrack._node_geometry import node_bbox_and_mask


@dataclass(frozen=True)
class SwapCandidate:
    node_id: int
    mask_2d: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int


@dataclass
class _SwapCursor:
    source_id: int
    frame: int
    source_centroid: tuple[float, float]
    source_area: int
    candidates: tuple[SwapCandidate, ...]
    displayed_area: int
    cursor: int | None


def list_swap_candidates(
    *,
    db_path: Path,
    frame: int,
    source_centroid: tuple[float, float],
    radius_px: float,
    frame_shape: tuple[int, int],
    protected_mask: np.ndarray | None = None,
) -> list[SwapCandidate]:
    if not Path(db_path).exists():
        return []

    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    engine = sqla.create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    results: list[SwapCandidate] = []
    src_y, src_x = source_centroid

    try:
        with Session(engine) as session:
            rows = session.query(NodeDB).filter(NodeDB.t == frame).all()
            for node in rows:
                try:
                    (y0, x0, y1, x1), mask_crop = node_bbox_and_mask(
                        int(node.id), node.pickle
                    )
                except Exception:
                    continue
                if mask_crop.shape != (y1 - y0, x1 - x0):
                    continue

                cy, cx = float(node.y), float(node.x)
                dist = float(np.hypot(cy - src_y, cx - src_x))
                if dist > radius_px:
                    continue

                full_mask = np.zeros(frame_shape, dtype=bool)
                full_mask[y0:y1, x0:x1] = mask_crop
                if not full_mask.any():
                    continue

                if protected_mask is not None:
                    paintable = full_mask & ~protected_mask
                    if not paintable.any():
                        continue

                area = int(full_mask.sum())
                results.append(
                    SwapCandidate(
                        node_id=int(node.id),
                        mask_2d=full_mask,
                        bbox=(y0, x0, y1, x1),
                        centroid=(cy, cx),
                        area=area,
                    )
                )
    finally:
        engine.dispose()

    results.sort(key=lambda c: c.area)
    return results


def step_smaller(
    candidates: tuple[SwapCandidate, ...] | list[SwapCandidate],
    displayed_area: int,
) -> int | None:
    best: int | None = None
    for i, c in enumerate(candidates):
        if c.area < displayed_area:
            best = i
    return best


def step_larger(
    candidates: tuple[SwapCandidate, ...] | list[SwapCandidate],
    displayed_area: int,
) -> int | None:
    for i, c in enumerate(candidates):
        if c.area > displayed_area:
            return i
    return None
