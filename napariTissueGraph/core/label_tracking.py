"""Label-based cell tracking via IoU matching and contour extraction."""
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def match_labels(
    frame_t: np.ndarray,
    frame_t1: np.ndarray,
    min_iou: float = 0.3,
    max_area_change: float = float('inf'),
) -> Dict[int, Optional[int]]:
    """Match labels across two consecutive frames by intersection-over-union.

    Args:
        frame_t: Label frame at time t (2D integer array, 0=background).
        frame_t1: Label frame at time t+1.
        min_iou: Minimum IoU for a match.
        max_area_change: Maximum allowed area ratio between matched labels.
            Ratio is max(a1,a2)/min(a1,a2). Matches exceeding this are
            rejected. Default inf (no limit).

    Returns:
        Dict mapping label_t -> label_t1 (or None if no match above threshold).
    """
    labels_t = set(np.unique(frame_t)) - {0}
    labels_t1 = set(np.unique(frame_t1)) - {0}

    if not labels_t or not labels_t1:
        return {label: None for label in labels_t}

    mapping: Dict[int, Optional[int]] = {}
    claimed: set = set()

    # Precompute masks for frame_t1 labels
    masks_t1 = {label: (frame_t1 == label) for label in labels_t1}
    areas_t1 = {label: np.sum(mask) for label, mask in masks_t1.items()}

    # For each label in frame_t, find best IoU match in frame_t1
    for label_t in labels_t:
        mask_t = frame_t == label_t
        area_t = np.sum(mask_t)

        best_iou = 0.0
        best_match = None

        for label_t1 in labels_t1:
            if label_t1 in claimed:
                continue
            intersection = np.sum(mask_t & masks_t1[label_t1])
            if intersection == 0:
                continue
            union = area_t + areas_t1[label_t1] - intersection
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_match = label_t1

        if best_iou >= min_iou and best_match is not None:
            # Check area change constraint
            area_ratio = max(area_t, areas_t1[best_match]) / max(
                min(area_t, areas_t1[best_match]), 1
            )
            if area_ratio <= max_area_change:
                mapping[label_t] = best_match
                claimed.add(best_match)
            else:
                mapping[label_t] = None
        else:
            mapping[label_t] = None

    return mapping


def assign_track_ids(
    label_stack: np.ndarray,
    min_iou: float = 0.3,
    max_area_change: float = float('inf'),
) -> Dict[int, Dict[int, int]]:
    """Assign consistent track IDs across all frames via IoU matching.

    Args:
        label_stack: Shape (T, H, W) integer labels.
        min_iou: Minimum IoU threshold for matching.
        max_area_change: Maximum allowed area ratio between matched labels.

    Returns:
        Dict: frame_idx -> {cell_label -> track_id}
    """
    n_frames = label_stack.shape[0]
    track_assignments: Dict[int, Dict[int, int]] = {}
    next_track_id = 0

    # First frame: each cell gets a new track ID
    labels_0 = set(np.unique(label_stack[0])) - {0}
    frame_0_tracks: Dict[int, int] = {}
    for label in sorted(labels_0):
        frame_0_tracks[label] = next_track_id
        next_track_id += 1
    track_assignments[0] = frame_0_tracks

    # Subsequent frames: match to previous frame
    for f in range(1, n_frames):
        matching = match_labels(
            label_stack[f - 1], label_stack[f],
            min_iou=min_iou, max_area_change=max_area_change,
        )
        prev_tracks = track_assignments[f - 1]
        frame_tracks: Dict[int, int] = {}

        # Carry forward matched labels
        for label_prev, label_next in matching.items():
            if label_next is not None and label_prev in prev_tracks:
                frame_tracks[label_next] = prev_tracks[label_prev]

        # Assign new track IDs to unmatched labels in frame f
        labels_f = set(np.unique(label_stack[f])) - {0}
        for label in sorted(labels_f):
            if label not in frame_tracks:
                frame_tracks[label] = next_track_id
                next_track_id += 1

        track_assignments[f] = frame_tracks

    return track_assignments


def label_to_vertices(
    label_frame: np.ndarray,
    cell_id: int,
) -> Optional[np.ndarray]:
    """Extract ordered boundary vertices of a cell from a label frame.

    Args:
        label_frame: 2D integer array.
        cell_id: The label value of the cell.

    Returns:
        Nx2 array of (y, x) ordered boundary points, or None if not found.
    """
    mask = (label_frame == cell_id).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)
    # cv2 returns (N, 1, 2) in (x, y) order; convert to (N, 2) in (y, x)
    pts = contour.squeeze(axis=1)
    if pts.ndim != 2 or len(pts) < 3:
        return None

    return np.column_stack([pts[:, 1], pts[:, 0]])  # (y, x)
