"""Per-track temporal presence ("swimlane") model from a tracked label stack.

This is the data half of the correction *lineage graph*: for each track id it
records the frame ranges where the cell is present, collapsed into contiguous
segments so a gap (track vanishes then returns — a likely ID swap or missed
link) reads as a break between segments. It is deliberately array-only — true
parent/daughter division edges live in the Ultrack database and are a separate,
heavier concern layered on top later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TrackSegment:
    """A contiguous run of frames ``[start, end]`` (inclusive) for one track."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True, slots=True)
class TrackLane:
    """One track id and the frame segments where it is present."""

    cell_id: int
    segments: tuple[TrackSegment, ...]

    @property
    def first_frame(self) -> int:
        return self.segments[0].start

    @property
    def last_frame(self) -> int:
        return self.segments[-1].end

    @property
    def n_frames(self) -> int:
        return sum(seg.length for seg in self.segments)

    @property
    def has_gap(self) -> bool:
        return len(self.segments) > 1


@dataclass(frozen=True, slots=True)
class LineageModel:
    """All track lanes plus the total frame count, for the lineage panel."""

    n_frames: int
    lanes: tuple[TrackLane, ...]

    def lane_for(self, cell_id: int) -> TrackLane | None:
        for lane in self.lanes:
            if lane.cell_id == cell_id:
                return lane
        return None


def _segments_from_frames(frames: list[int]) -> tuple[TrackSegment, ...]:
    """Collapse a sorted frame list into contiguous inclusive segments."""
    segments: list[TrackSegment] = []
    start = prev = frames[0]
    for t in frames[1:]:
        if t == prev + 1:
            prev = t
            continue
        segments.append(TrackSegment(start=start, end=prev))
        start = prev = t
    segments.append(TrackSegment(start=start, end=prev))
    return tuple(segments)


def build_lineage(tracked: np.ndarray) -> LineageModel:
    """Build a :class:`LineageModel` from a ``(T, Y, X)`` tracked label stack.

    A singleton Z axis (``(T, 1, Y, X)``) is squeezed; a single 2D frame is
    treated as one timepoint. Lanes are sorted by first appearance, then id.
    """
    arr = np.asarray(tracked)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"tracked must be (T, Y, X); got shape {arr.shape}")

    n_t = arr.shape[0]
    frames_of: dict[int, list[int]] = {}
    for t in range(n_t):
        for cell_id in np.unique(arr[t]).tolist():
            if cell_id == 0:
                continue
            frames_of.setdefault(int(cell_id), []).append(t)

    lanes = [
        TrackLane(cell_id=cell_id, segments=_segments_from_frames(frames))
        for cell_id, frames in frames_of.items()
    ]
    lanes.sort(key=lambda lane: (lane.first_frame, lane.cell_id))
    return LineageModel(n_frames=n_t, lanes=tuple(lanes))


__all__ = ["LineageModel", "TrackLane", "TrackSegment", "build_lineage"]
