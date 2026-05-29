from __future__ import annotations

import numpy as np


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


def reassign_ids_stack(stack: np.ndarray) -> tuple[np.ndarray, int, dict[int, int]]:
    """Compact non-zero label IDs in a stack to contiguous IDs from 1."""
    unique_ids = np.unique(stack)
    unique_ids = unique_ids[unique_ids != 0]
    if unique_ids.size == 0:
        return stack, 0, {}
    lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.uint32)
    old_to_new: dict[int, int] = {}
    for new_id, old_id in enumerate(unique_ids, start=1):
        lut[old_id] = new_id
        old_to_new[int(old_id)] = new_id
    return lut[stack], len(unique_ids), old_to_new


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
