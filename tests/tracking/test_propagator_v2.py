"""Tests for the anchor-LAP nucleus propagator v2."""
import numpy as np
import pytest

from cellflow.tracking.propagator_v2 import find_best_hypothesis_v2, PropagationContext


def _make_square(shape, row, col, size, label):
    """Return a label array with one filled square."""
    arr = np.zeros(shape, dtype=np.uint32)
    arr[row:row + size, col:col + size] = label
    return arr


def test_two_spatially_separated_nuclei_both_matched():
    """Two nuclei with spatially separated candidates are both matched via LAP assignment."""
    shape = (100, 200)
    sz = 10

    current = _make_square(shape, 10, 10, sz, 1)
    current[10:10 + sz, 150:150 + sz] = 2

    cand = np.zeros(shape, dtype=np.uint32)
    cand[12:12 + sz, 12:12 + sz] = 1   # matches nucleus 1
    cand[10:10 + sz, 148:148 + sz] = 2  # matches nucleus 2

    ctx = PropagationContext(current_labels=current)
    next_frame, winner = find_best_hypothesis_v2(ctx, [cand])

    assert next_frame is not None
    assert winner == 0
    assert 1 in np.unique(next_frame)
    assert 2 in np.unique(next_frame)


def test_no_candidates_returns_none():
    """Empty candidate list returns (None, None)."""
    shape = (50, 50)
    current = _make_square(shape, 10, 10, 10, 1)
    ctx = PropagationContext(current_labels=current)
    result = find_best_hypothesis_v2(ctx, [])
    assert result == (None, None)


def test_empty_current_returns_none():
    """Empty current labels returns (None, None)."""
    shape = (50, 50)
    current = np.zeros(shape, dtype=np.uint32)
    cand = _make_square(shape, 10, 10, 10, 1)
    ctx = PropagationContext(current_labels=current)
    result = find_best_hypothesis_v2(ctx, [cand])
    assert result == (None, None)


def test_perfect_overlap_returns_track_id():
    """Candidate identical to current nucleus → IoU=1, track ID preserved."""
    shape = (60, 60)
    current = _make_square(shape, 10, 10, 15, 1)
    cand = _make_square(shape, 10, 10, 15, 1)

    ctx = PropagationContext(current_labels=current)
    next_frame, winner = find_best_hypothesis_v2(ctx, [cand])

    assert next_frame is not None
    assert winner == 0
    assert int(next_frame[17, 17]) == 1


def test_multiple_hypotheses_same_location_picks_best():
    """Three candidates at the same location; the one with highest IoU wins."""
    shape = (80, 80)
    current = _make_square(shape, 20, 20, 20, 1)

    cand0 = _make_square(shape, 20, 20, 20, 1)   # perfect match
    cand1 = _make_square(shape, 22, 22, 10, 1)   # smaller, worse IoU
    cand2 = _make_square(shape, 18, 18, 25, 1)   # larger, worse IoU

    ctx = PropagationContext(current_labels=current)
    next_frame, winner = find_best_hypothesis_v2(ctx, [cand0, cand1, cand2])

    assert next_frame is not None
    assert winner == 0


def test_zero_overlap_candidate_dropped_by_threshold():
    """A far-away candidate with zero overlap fails min_match_iou threshold, returns (None, None)."""
    shape = (100, 100)
    current = _make_square(shape, 10, 10, 20, 1)
    cand = _make_square(shape, 80, 80, 20, 1)  # no overlap, far away

    ctx = PropagationContext(current_labels=current)
    next_frame, winner = find_best_hypothesis_v2(ctx, [cand], min_match_iou=0.1)

    assert next_frame is None
    assert winner is None


def test_two_nuclei_no_cross_assignment():
    """Two nuclei each have one clearly matching candidate; LAP prevents cross-assignment."""
    shape = (100, 200)
    sz = 12

    current = np.zeros(shape, dtype=np.uint32)
    current[10:10+sz, 10:10+sz] = 1
    current[10:10+sz, 160:160+sz] = 2

    cand = np.zeros(shape, dtype=np.uint32)
    cand[11:11+sz, 11:11+sz] = 10   # near nucleus 1
    cand[11:11+sz, 161:161+sz] = 20  # near nucleus 2

    ctx = PropagationContext(current_labels=current)
    next_frame, winner = find_best_hypothesis_v2(ctx, [cand])

    assert next_frame is not None
    tracked = {int(v) for v in np.unique(next_frame) if v != 0}
    assert tracked == {1, 2}
