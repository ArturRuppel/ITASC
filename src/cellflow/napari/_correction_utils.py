from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class RetrackDirectionResult:
    stack: np.ndarray
    n_retracked: int
    n_skipped: int
    first_target_frame: int


def frame_view_2d(arr: np.ndarray, t: int) -> np.ndarray | None:
    """Return frame *t* as a 2D view when the stack shape is unambiguous."""
    if arr.ndim < 3 or t < 0 or t >= arr.shape[0]:
        return None
    view = arr[t]
    while view.ndim > 2:
        if view.shape[0] != 1:
            return None
        view = view[0]
    return view


def reassign_ids_ordered(
    stack: np.ndarray, order: list[int]
) -> tuple[np.ndarray, int, dict[int, int]]:
    """Relabel non-zero IDs to contiguous IDs from 1, following *order*.

    Old IDs listed in *order* are assigned new IDs first (best → 1, next → 2,
    …); any present IDs not in *order* follow in ascending order. Passing an
    empty *order* reproduces plain numeric compaction.
    """
    unique_ids = np.unique(stack)
    unique_ids = unique_ids[unique_ids != 0]
    if unique_ids.size == 0:
        return stack, 0, {}
    present = {int(x) for x in unique_ids}
    seen: set[int] = set()
    ordered_ids: list[int] = []
    for old_id in order:
        old_id = int(old_id)
        if old_id in present and old_id not in seen:
            ordered_ids.append(old_id)
            seen.add(old_id)
    ordered_ids.extend(sorted(present - seen))
    lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.uint32)
    old_to_new: dict[int, int] = {}
    for new_id, old_id in enumerate(ordered_ids, start=1):
        lut[old_id] = new_id
        old_to_new[old_id] = new_id
    return lut[stack], len(ordered_ids), old_to_new


def reassign_ids_stack(stack: np.ndarray) -> tuple[np.ndarray, int, dict[int, int]]:
    """Compact non-zero label IDs in a stack to contiguous IDs from 1."""
    return reassign_ids_ordered(stack, [])


def reorder_stack_by_quality(
    stack: np.ndarray,
    scores: Mapping[int, float],
    pos_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Relabel a tracked stack so track IDs follow quality order, best → 1.

    *scores* maps track_id -> quality score (see
    :func:`cellflow.tracking_ultrack.track_quality.track_quality_scores`). When
    *pos_dir* is given, existing validations/anchors are remapped onto the new
    IDs so prior work stays attached. Returns ``(relabeled_stack, old_to_new)``;
    a no-op (empty mapping, original array) when there is nothing to reorder.
    """
    from cellflow.tracking_ultrack.track_quality import quality_order

    stack = np.asarray(stack)
    order = quality_order(scores) if scores else []
    if not order:
        return stack, {}
    relabeled, _n, old_to_new = reassign_ids_ordered(
        stack.astype(np.uint32, copy=False), order
    )
    if old_to_new and pos_dir is not None:
        from cellflow.database.validation import remap_validated_tracks

        remap_validated_tracks(Path(pos_dir), old_to_new)
    return relabeled, old_to_new


def remove_unvalidated_labels(
    data: np.ndarray,
    validated_tracks: dict[int, set[int]],
) -> tuple[int, int]:
    """Remove labels not validated for their frame from a 2D or time-first stack."""
    frame_count = int(data.shape[0]) if data.ndim >= 3 else 1
    changed_pixels = changed_frames = 0
    for t in range(frame_count):
        frame = data[t] if data.ndim >= 3 else data
        if frame.ndim != 2:
            raise ValueError("Tracked layer must be a time-first stack.")
        validated_ids = {
            cid for cid, frames in validated_tracks.items() if t in frames
        }
        remove_mask = frame != 0
        if validated_ids:
            remove_mask &= ~np.isin(frame, list(validated_ids))
        n_remove = int(np.count_nonzero(remove_mask))
        if not n_remove:
            continue
        frame[remove_mask] = 0
        changed_pixels += n_remove
        changed_frames += 1
    return changed_frames, changed_pixels


def retrack_stack_direction(
    stack: np.ndarray,
    *,
    start_frame: int,
    direction: Literal["forward", "backward"],
    fully_validated_frames: set[int],
    validated_cells_at_frame: Callable[[int], set[int]],
    retrack_frame: Callable[..., np.ndarray],
    max_dist_px: float,
    reserved_ids: set[int],
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    distance_weight: float = 0.05,
) -> RetrackDirectionResult:
    """Retrack a time-first stack in one direction, skipping validated frames."""
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError("Tracked layer must be a 3D time-first stack.")

    out = stack.copy()
    n_retracked = n_skipped = 0
    if direction == "forward":
        frame_range = range(start_frame + 1, out.shape[0])
        previous_frame = lambda t: out[t - 1]
    elif direction == "backward":
        frame_range = range(start_frame - 1, -1, -1)
        previous_frame = lambda t: out[t + 1]
    else:
        raise ValueError(f"Unknown retrack direction: {direction!r}")

    first_target_frame = next(iter(frame_range), start_frame)
    for t in frame_range:
        if t in fully_validated_frames:
            n_skipped += 1
            continue
        out[t] = retrack_frame(
            previous_frame(t),
            out[t],
            validated_cells_at_frame(t),
            max_dist_px=max_dist_px,
            reserved_ids=reserved_ids,
            area_weight=area_weight,
            iou_weight=iou_weight,
            distance_weight=distance_weight,
        )
        n_retracked += 1

    return RetrackDirectionResult(
        stack=out,
        n_retracked=n_retracked,
        n_skipped=n_skipped,
        first_target_frame=first_target_frame,
    )
