"""Tests for the best-match nucleus propagator."""
import numpy as np
import pytest

from cellflow.tracking.propagator import find_best_hypothesis


def _make_square(shape, row, col, size, label):
    """Return a label array with one filled square."""
    arr = np.zeros(shape, dtype=np.uint32)
    arr[row:row + size, col:col + size] = label
    return arr


def test_two_spatially_separated_nuclei_both_matched():
    """Two nuclei with spatially separated candidates are both matched.

    Nucleus 1 (top-left) and nucleus 2 (top-right) each have one good
    candidate far from the other. Greedy per-nucleus matching assigns each
    to its own best candidate independently.
    """
    shape = (100, 200)
    sz = 10

    current = _make_square(shape, 10, 10, sz, 1)
    current[10:10 + sz, 150:150 + sz] = 2

    cand_a = _make_square(shape, 12, 12, sz, 1)   # matches nucleus 1
    cand_b = _make_square(shape, 10, 148, sz, 1)  # matches nucleus 2

    next_frame, winner = find_best_hypothesis(
        current, [cand_a, cand_b],
        max_dist_px=50.0,
    )

    assert next_frame is not None
    assert winner is not None
    assert 1 in np.unique(next_frame)
    assert 2 in np.unique(next_frame)


def test_distance_gate_drops_far_candidate():
    """A candidate whose centroid is beyond max_dist_px from the predicted position is rejected."""
    shape = (100, 100)
    current = _make_square(shape, 10, 10, 20, 1)
    cand = _make_square(shape, 80, 80, 20, 1)  # centroid ~80px away

    next_frame, winner = find_best_hypothesis(
        current, [cand],
        max_dist_px=30.0,
    )

    assert next_frame is None
    assert winner is None


def test_no_candidates_returns_none():
    shape = (50, 50)
    current = _make_square(shape, 10, 10, 10, 1)
    result = find_best_hypothesis(current, [], max_dist_px=50.0)
    assert result == (None, None)


def test_empty_current_returns_none():
    shape = (50, 50)
    current = np.zeros(shape, dtype=np.uint32)
    cand = _make_square(shape, 10, 10, 10, 1)
    result = find_best_hypothesis(current, [cand], max_dist_px=50.0)
    assert result == (None, None)


def test_perfect_overlap_returns_track_id():
    """Candidate identical to current nucleus → IoU=1, track ID preserved."""
    shape = (60, 60)
    current = _make_square(shape, 10, 10, 15, 1)
    cand = _make_square(shape, 10, 10, 15, 1)

    next_frame, winner = find_best_hypothesis(
        current, [cand],
        max_dist_px=5.0,
    )

    assert next_frame is not None
    assert winner == 0
    assert int(next_frame[17, 17]) == 1


def test_multiple_hypotheses_same_location_picks_best():
    """Three candidates at the same location; the one with highest IoU wins."""
    shape = (80, 80)
    current = _make_square(shape, 20, 20, 20, 1)

    cand0 = _make_square(shape, 20, 20, 20, 1)   # perfect match
    cand1 = _make_square(shape, 22, 22, 10, 1)   # smaller
    cand2 = _make_square(shape, 18, 18, 25, 1)   # larger

    next_frame, winner = find_best_hypothesis(
        current, [cand0, cand1, cand2],
        max_dist_px=20.0,
    )

    assert next_frame is not None
    assert winner == 0


def test_centroid_corrected_iou_breaks_equal_area_tie():
    """Equal-area candidates should prefer the translated matching shape."""
    shape = (50, 50)
    current = np.zeros(shape, dtype=np.uint32)
    current[10:14, 10:14] = 1
    current[14:18, 10:12] = 1

    wrong_shape = np.zeros(shape, dtype=np.uint32)
    wrong_shape[10:13, 10:18] = 1

    translated_match = np.zeros(shape, dtype=np.uint32)
    translated_match[10:14, 20:24] = 1
    translated_match[14:18, 20:22] = 1

    next_frame, winner = find_best_hypothesis(
        current,
        [wrong_shape, translated_match],
        max_dist_px=20.0,
        area_weight=1.0,
        iou_weight=1.0,
        circularity_weight=0.0,
        solidity_weight=0.0,
    )

    assert next_frame is not None
    assert winner == 1
    assert int(next_frame[12, 22]) == 1
    assert int(next_frame[11, 11]) == 0



def test_distant_nucleus_not_matched_to_far_candidate():
    """A nucleus with no nearby candidates stays unmatched.

    Nucleus 1 is far from the only available candidates (which are near
    nucleus 2). The distance gate keeps nucleus 1 unmatched while nucleus 2
    is correctly assigned.
    """
    shape = (100, 200)
    sz = 12

    current = np.zeros(shape, dtype=np.uint32)
    current[10:10 + sz, 10:10 + sz] = 1    # nucleus 1 (will have no good match)
    current[10:10 + sz, 100:100 + sz] = 2  # nucleus 2

    # Both hypothesis images carry a candidate at nucleus 2's next position only.
    cand_a = np.zeros(shape, dtype=np.uint32)
    cand_a[11:11 + sz, 101:101 + sz] = 1

    cand_b = np.zeros(shape, dtype=np.uint32)
    cand_b[12:12 + sz, 102:102 + sz] = 1  # slightly shifted duplicate

    next_frame, _ = find_best_hypothesis(
        current, [cand_a, cand_b],
        max_dist_px=50.0,
        dedup_radius_px=10.0,
    )

    assert next_frame is not None
    tracked = {int(v) for v in np.unique(next_frame) if v != 0}
    # Nucleus 2 should be matched; nucleus 1 must NOT appear at the same spot.
    assert 2 in tracked
    assert 1 not in tracked  # went to null, not colliding with nucleus 2


def test_two_nuclei_no_cross_assignment():
    """Two nuclei each have one clearly matching candidate; neither should
    steal the other's match (validates the assignment constraint)."""
    shape = (100, 200)
    sz = 12

    current = np.zeros(shape, dtype=np.uint32)
    current[10:10+sz, 10:10+sz] = 1
    current[10:10+sz, 160:160+sz] = 2

    cand = np.zeros(shape, dtype=np.uint32)
    cand[11:11+sz, 11:11+sz] = 10   # near nucleus 1
    cand[11:11+sz, 161:161+sz] = 20  # near nucleus 2

    next_frame, winner = find_best_hypothesis(
        current, [cand],
        max_dist_px=30.0,
    )

    assert next_frame is not None
    tracked = {int(v) for v in np.unique(next_frame) if v != 0}
    assert tracked == {1, 2}
