from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from skimage.measure import regionprops

from cellflow.database.hypotheses import list_hypotheses, read_hypothesis_labels

_D_MAX_DEFAULT = 40.0


@dataclass
class ExtendResult:
    target_frame: int
    candidate_label: int
    candidate_partition: int
    mask_2d: np.ndarray       # bool, full-frame (Y, X)
    bbox: tuple[int, int, int, int]  # (y0, x0, y1, x1)
    centroid_distance: float
    area_ratio: float         # ∈ [0, 1]; 1.0 = identical area
    existing_overlap: float   # candidate ∩ (other cells at target) / candidate_area


def extend_track(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,   # (T, Y, X) uint32
    hypotheses_path: Path,
    d_max: float = _D_MAX_DEFAULT,
) -> ExtendResult | None:
    T = tracked_labels.shape[0]
    target_frame = source_frame + (1 if direction == "forward" else -1)

    if target_frame < 0 or target_frame >= T:
        return None

    source_mask = tracked_labels[source_frame] == source_id
    if not source_mask.any():
        return None

    props = regionprops(source_mask.astype(np.uint8))
    src_cy, src_cx = props[0].centroid
    src_area = float(props[0].area)

    target_frame_labels = tracked_labels[target_frame]
    other_cells = (target_frame_labels != 0) & (target_frame_labels != source_id)

    n_p, _ = list_hypotheses(hypotheses_path)
    if n_p == 0:
        return None

    best: ExtendResult | None = None
    best_score: tuple[float, float] | None = None  # (area_ratio, -dist) for ranking

    for p in range(n_p):
        labels_raw = read_hypothesis_labels(hypotheses_path, target_frame, p)
        labels_2d: np.ndarray
        if labels_raw.ndim == 3 and labels_raw.shape[0] == 1:
            labels_2d = labels_raw[0]
        elif labels_raw.ndim == 2:
            labels_2d = labels_raw
        else:
            continue

        for rp in regionprops(labels_2d.astype(np.int32)):
            cy, cx = rp.centroid
            dist = float(np.hypot(cy - src_cy, cx - src_cx))
            if dist > d_max:
                continue

            cand_area = float(rp.area)
            cand_mask = labels_2d == rp.label
            existing_overlap = float((cand_mask & other_cells).sum()) / cand_area
            area_ratio = min(src_area, cand_area) / max(src_area, cand_area)
            combined = area_ratio * (1.0 - existing_overlap)
            score = (combined, -dist)

            if best_score is None or score > best_score:
                y0, x0, y1, x1 = rp.bbox
                best = ExtendResult(
                    target_frame=target_frame,
                    candidate_label=int(rp.label),
                    candidate_partition=p,
                    mask_2d=cand_mask,
                    bbox=(y0, x0, y1, x1),
                    centroid_distance=dist,
                    area_ratio=area_ratio,
                    existing_overlap=existing_overlap,
                )
                best_score = score

    return best


def extend_track_from_db(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,   # (T, Y, X) uint32
    db_path: Path,
    d_max: float = _D_MAX_DEFAULT,
) -> ExtendResult | None:
    """Extend a track using candidates from ultrack_workdir/data.db.

    Returns None if the DB is missing, target frame is out of range, or no
    candidate within d_max is found.  Widget caller should show a local status
    message on None.
    """
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB
    from cellflow.tracking_ultrack.validation_nodes import _node_bbox_and_mask

    if not db_path.exists():
        return None

    T = tracked_labels.shape[0]
    target_frame = source_frame + (1 if direction == "forward" else -1)
    if target_frame < 0 or target_frame >= T:
        return None

    source_mask = tracked_labels[source_frame] == source_id
    if not source_mask.any():
        return None

    props = regionprops(source_mask.astype(np.uint8))
    src_cy, src_cx = props[0].centroid
    src_area = float(props[0].area)

    target_frame_labels = tracked_labels[target_frame]
    other_cells = (target_frame_labels != 0) & (target_frame_labels != source_id)

    engine = sqla.create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    best: ExtendResult | None = None
    best_score: tuple[float, float] | None = None

    with Session(engine) as session:
        candidates = session.query(NodeDB).filter(NodeDB.t == target_frame).all()
        for node in candidates:
            # Use centroid from NodeDB for quick distance filter before unpickling
            cy = float(node.y)
            cx = float(node.x)
            dist = float(np.hypot(cy - src_cy, cx - src_cx))
            if dist > d_max:
                continue
            try:
                (y0, x0, y1, x1), mask_2d = _node_bbox_and_mask(int(node.id), node.pickle)
            except Exception:
                continue
            if mask_2d.shape != (y1 - y0, x1 - x0):
                continue
            full_mask = np.zeros(tracked_labels.shape[1:], dtype=bool)
            full_mask[y0:y1, x0:x1] = mask_2d
            cand_area = float(full_mask.sum())
            if cand_area == 0:
                continue
            existing_overlap = float((full_mask & other_cells).sum()) / cand_area
            area_ratio = min(src_area, cand_area) / max(src_area, cand_area)
            combined = area_ratio * (1.0 - existing_overlap)
            score = (combined, -dist)
            if best_score is None or score > best_score:
                best_score = score
                best = ExtendResult(
                    target_frame=target_frame,
                    candidate_label=int(node.id),
                    candidate_partition=0,
                    mask_2d=full_mask,
                    bbox=(y0, x0, y1, x1),
                    centroid_distance=dist,
                    area_ratio=area_ratio,
                    existing_overlap=existing_overlap,
                )
    engine.dispose()
    return best
