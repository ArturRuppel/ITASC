"""Comparison metrics for tracked labelmaps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackSummary:
    n_tracks: int
    average_length: float
    track_lengths: dict[int, int]


def tracked_label_summary(labels: np.ndarray) -> TrackSummary:
    """Return track count and frame-presence lengths for a tracked labelmap."""
    arr = np.asarray(labels)
    if arr.ndim < 2:
        raise ValueError(f"Expected labels with time as axis 0, got shape {arr.shape}")

    track_lengths: dict[int, int] = {}
    for t in range(arr.shape[0]):
        ids = np.unique(arr[t])
        for label_id in ids:
            label_int = int(label_id)
            if label_int == 0:
                continue
            track_lengths[label_int] = track_lengths.get(label_int, 0) + 1

    lengths = list(track_lengths.values())
    average_length = float(np.mean(lengths)) if lengths else 0.0
    return TrackSummary(
        n_tracks=len(track_lengths),
        average_length=average_length,
        track_lengths=track_lengths,
    )


def binary_labelmap_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Return foreground IoU after binarizing two labelmaps with ``label > 0``."""
    left = np.asarray(lhs) > 0
    right = np.asarray(rhs) > 0
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} != {right.shape}")

    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(left, right).sum()
    return float(intersection / union)
