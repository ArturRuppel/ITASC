"""Unit tests for cellflow.segmentation.lineage."""
from __future__ import annotations

import numpy as np

from cellflow.segmentation.lineage import build_lineage


def _stack(t: int = 4, size: int = 8) -> np.ndarray:
    return np.zeros((t, size, size), dtype=np.uint32)


def test_contiguous_track_is_a_single_segment() -> None:
    arr = _stack()
    for t in range(4):
        arr[t, 0:2, 0:2] = 1

    model = build_lineage(arr)

    assert model.n_frames == 4
    lane = model.lane_for(1)
    assert lane is not None
    assert len(lane.segments) == 1
    assert lane.segments[0].start == 0
    assert lane.segments[0].end == 3
    assert lane.n_frames == 4
    assert not lane.has_gap


def test_gap_splits_into_two_segments() -> None:
    arr = _stack()
    arr[0, 4:6, 4:6] = 2
    arr[1, 4:6, 4:6] = 2
    # frame 2 missing
    arr[3, 4:6, 4:6] = 2

    lane = build_lineage(arr).lane_for(2)

    assert lane is not None
    assert lane.has_gap
    assert [(s.start, s.end) for s in lane.segments] == [(0, 1), (3, 3)]
    assert lane.first_frame == 0
    assert lane.last_frame == 3
    assert lane.n_frames == 3


def test_lanes_sorted_by_track_id() -> None:
    # Order follows track id, not first appearance, so a cell keeps its row
    # regardless of when it shows up (correction actions relabel by id).
    arr = _stack()
    arr[2, 0, 0] = 5  # appears at frame 2
    arr[0, 1, 1] = 9  # appears at frame 0
    arr[0, 2, 2] = 3  # appears at frame 0, lower id

    order = [lane.cell_id for lane in build_lineage(arr).lanes]

    assert order == [3, 5, 9]


def test_singleton_z_axis_is_squeezed() -> None:
    arr = np.zeros((3, 1, 8, 8), dtype=np.uint32)
    for t in range(3):
        arr[t, 0, 0, 0] = 1

    model = build_lineage(arr)

    assert model.n_frames == 3
    assert model.lane_for(1) is not None


def test_empty_stack_has_no_lanes() -> None:
    model = build_lineage(_stack())
    assert model.lanes == ()
    assert model.n_frames == 4
