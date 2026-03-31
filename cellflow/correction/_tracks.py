"""
Track correction operations on a (T, H, W) Labels stack.

In CellFlow's tracking representation, label value == track ID.
All operations modify the full (T, H, W) array in-place and return True
on success, False when rejected.

Operations
----------
merge_tracks      Replace track B with A from `from_frame` onwards.
split_track       Assign a new label to track A from `from_frame` onwards.
swap_tracks       Swap labels A and B from `from_frame` onwards.
delete_track      Zero-out track A from `from_frame` onwards.
reassign_cell     Change one cell's label in one specific frame and propagate
                  the new ID forward by IoU until the track ends or conflicts.
interpolate_track Fill a gap in track A between two frames using centroid
                  nearest-neighbour matching.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass


# ── helpers ───────────────────────────────────────────────────────────────────

def _free_label(seg: np.ndarray) -> int:
    return int(seg.max()) + 1


def _centroid(frame: np.ndarray, label: int) -> tuple[float, float] | None:
    mask = frame == label
    if not np.any(mask):
        return None
    r, c = center_of_mass(mask)
    return float(r), float(c)


def _best_iou_match(frame_a: np.ndarray, frame_b: np.ndarray, label_a: int) -> int | None:
    """Return the label in frame_b that overlaps most with label_a in frame_a."""
    mask_a = frame_a == label_a
    if not np.any(mask_a):
        return None
    candidates = np.unique(frame_b[mask_a])
    candidates = candidates[candidates != 0]
    if len(candidates) == 0:
        return None
    best_lab, best_iou = None, 0.0
    for lab in candidates:
        mask_b = frame_b == lab
        inter = np.sum(mask_a & mask_b)
        union = np.sum(mask_a | mask_b)
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_lab = int(lab)
    return best_lab


def _nearest_centroid_match(
    frame: np.ndarray, target_rc: tuple[float, float]
) -> int | None:
    """Return the label in frame whose centroid is closest to target_rc."""
    labels = np.unique(frame)
    labels = labels[labels != 0]
    if len(labels) == 0:
        return None
    best_lab, best_dist = None, float("inf")
    for lab in labels:
        rc = _centroid(frame, lab)
        if rc is None:
            continue
        dist = (rc[0] - target_rc[0]) ** 2 + (rc[1] - target_rc[1]) ** 2
        if dist < best_dist:
            best_dist = dist
            best_lab = int(lab)
    return best_lab


def _label_at(frame: np.ndarray, pos: tuple) -> int:
    r = max(0, min(int(round(float(pos[-2]))), frame.shape[0] - 1))
    c = max(0, min(int(round(float(pos[-1]))), frame.shape[1] - 1))
    return int(frame[r, c])


# ── public operations ─────────────────────────────────────────────────────────

def merge_tracks(
    seg: np.ndarray,
    track_a: int,
    track_b: int,
    from_frame: int = 0,
) -> bool:
    """
    Merge track B into track A from *from_frame* onwards.

    All pixels labelled *track_b* at or after *from_frame* are relabelled
    *track_a*.  Frames before *from_frame* are unchanged.
    """
    if track_a == 0 or track_b == 0 or track_a == track_b:
        return False
    for t in range(from_frame, seg.shape[0]):
        seg[t][seg[t] == track_b] = track_a
    return True


def split_track(
    seg: np.ndarray,
    track_id: int,
    from_frame: int,
) -> int | None:
    """
    Split track *track_id* at *from_frame*.

    Pixels labelled *track_id* at or after *from_frame* are relabelled with a
    new unique value.  Returns the new label, or None if track_id is not
    present from *from_frame* onwards.
    """
    if track_id == 0:
        return None
    present = any(np.any(seg[t] == track_id) for t in range(from_frame, seg.shape[0]))
    if not present:
        return None
    new_lab = _free_label(seg)
    for t in range(from_frame, seg.shape[0]):
        seg[t][seg[t] == track_id] = new_lab
    return new_lab


def swap_tracks(
    seg: np.ndarray,
    track_a: int,
    track_b: int,
    from_frame: int,
) -> bool:
    """
    Swap labels *track_a* and *track_b* from *from_frame* onwards.

    Uses a temporary label to avoid collisions during the swap.
    """
    if track_a == 0 or track_b == 0 or track_a == track_b:
        return False
    tmp = _free_label(seg)
    for t in range(from_frame, seg.shape[0]):
        frame = seg[t]
        mask_a = frame == track_a
        mask_b = frame == track_b
        frame[mask_a] = tmp
        frame[mask_b] = track_a
        frame[frame == tmp] = track_b
    return True


def delete_track(
    seg: np.ndarray,
    track_id: int,
    from_frame: int,
) -> bool:
    """
    Delete track *track_id* from *from_frame* onwards (set pixels to 0).
    """
    if track_id == 0:
        return False
    for t in range(from_frame, seg.shape[0]):
        seg[t][seg[t] == track_id] = 0
    return True


def reassign_cell(
    seg: np.ndarray,
    frame: int,
    old_label: int,
    new_label: int,
) -> bool:
    """
    Change the label of a cell in one frame and propagate forward by IoU.

    The cell labelled *old_label* at *frame* is relabelled *new_label*.
    In subsequent frames, the cell that best overlaps it (by IoU) inherits
    the new label, until the track terminates or no match is found.

    Any existing pixels labelled *new_label* in frames after *frame* are
    first bumped to a fresh label so they don't collide.
    """
    if old_label == 0 or new_label == 0 or old_label == new_label:
        return False
    if not np.any(seg[frame] == old_label):
        return False

    # bump any forward uses of new_label to avoid collision
    bump = _free_label(seg)
    for t in range(frame + 1, seg.shape[0]):
        if np.any(seg[t] == new_label):
            seg[t][seg[t] == new_label] = bump

    # relabel in the target frame
    seg[frame][seg[frame] == old_label] = new_label

    # propagate forward by IoU
    current_label = new_label
    for t in range(frame + 1, seg.shape[0]):
        next_lab = _best_iou_match(seg[t - 1], seg[t], current_label)
        if next_lab is None:
            break
        seg[t][seg[t] == next_lab] = current_label

    return True


def interpolate_track(
    seg: np.ndarray,
    track_id: int,
    frame_start: int,
    frame_end: int,
) -> bool:
    """
    Fill a gap in *track_id* between *frame_start* and *frame_end*.

    The track must be present at both endpoints.  For each intermediate frame,
    the cell whose centroid is closest to the linearly interpolated position
    is relabelled *track_id*.
    """
    if track_id == 0:
        return False
    if frame_end <= frame_start:
        return False

    c_start = _centroid(seg[frame_start], track_id)
    c_end = _centroid(seg[frame_end], track_id)
    if c_start is None or c_end is None:
        return False

    n = frame_end - frame_start
    for i, t in enumerate(range(frame_start + 1, frame_end), start=1):
        alpha = i / n
        target_r = c_start[0] + alpha * (c_end[0] - c_start[0])
        target_c = c_start[1] + alpha * (c_end[1] - c_start[1])
        best = _nearest_centroid_match(seg[t], (target_r, target_c))
        if best is not None and best != track_id:
            seg[t][seg[t] == best] = track_id

    return True
