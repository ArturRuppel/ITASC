"""Tests for the LAP-based nucleus propagator."""
import numpy as np
import pytest

from cellflow.tracking.propagator import find_best_hypothesis


def _make_square(shape, row, col, size, label):
    """Return a label array with one filled square."""
    arr = np.zeros(shape, dtype=np.uint32)
    arr[row:row + size, col:col + size] = label
    return arr


def test_hungarian_beats_greedy_order():
    """Global assignment outperforms greedy when two nuclei compete.

    Nucleus 1 (top-left) and nucleus 2 (top-right) each have one good
    candidate. Greedy (ID-sorted) would assign nucleus 1 first regardless;
    Hungarian finds the globally optimal pairing so both are matched.
    """
    shape = (100, 200)
    sz = 10

    current = _make_square(shape, 10, 10, sz, 1)
    current[10:10 + sz, 150:150 + sz] = 2

    cand_a = _make_square(shape, 12, 12, sz, 1)   # matches nucleus 1
    cand_b = _make_square(shape, 10, 148, sz, 1)  # matches nucleus 2

    next_frame, winner = find_best_hypothesis(
        current, [cand_a, cand_b],
        iou_threshold=0.1,
        max_dist_px=50.0,
    )

    assert next_frame is not None
    assert winner is not None
    assert 1 in np.unique(next_frame)
    assert 2 in np.unique(next_frame)


def test_iou_gate_drops_unmatched():
    """A candidate whose centroid-corrected IoU is below threshold is rejected.

    Current: large 20×20 square. Candidate: tiny 3×3 square.
    IoU = 9 / (400 + 9 - 9) ≈ 0.022, well below any reasonable threshold.
    """
    shape = (100, 100)
    current = _make_square(shape, 10, 10, 20, 1)
    cand = _make_square(shape, 50, 50, 3, 1)

    next_frame, winner = find_best_hypothesis(
        current, [cand],
        iou_threshold=0.3,
        max_dist_px=200.0,
    )

    assert next_frame is None
    assert winner is None


def test_no_candidates_returns_none():
    shape = (50, 50)
    current = _make_square(shape, 10, 10, 10, 1)
    result = find_best_hypothesis(current, [], iou_threshold=0.3, max_dist_px=50.0)
    assert result == (None, None)


def test_empty_current_returns_none():
    shape = (50, 50)
    current = np.zeros(shape, dtype=np.uint32)
    cand = _make_square(shape, 10, 10, 10, 1)
    result = find_best_hypothesis(current, [cand], iou_threshold=0.3, max_dist_px=50.0)
    assert result == (None, None)


def test_perfect_overlap_returns_track_id():
    """Candidate identical to current nucleus → IoU=1, track ID preserved."""
    shape = (60, 60)
    current = _make_square(shape, 10, 10, 15, 1)
    cand = _make_square(shape, 10, 10, 15, 1)

    next_frame, winner = find_best_hypothesis(
        current, [cand],
        iou_threshold=0.5,
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
        iou_threshold=0.1,
        max_dist_px=20.0,
    )

    assert next_frame is not None
    assert winner == 0


def test_velocity_predicted_centroid_extends_search():
    """When predicted centroid is far from the current position, the search
    radius is centered on the prediction so fast-moving nuclei are still found.

    Nucleus moved 60px right between t-1 and t. Predicted t+1 centroid is
    another 60px right. max_dist_px=40 would miss the candidate from the
    current position but reaches it from the predicted position.
    """
    shape = (100, 200)
    current = _make_square(shape, 20, 90, 15, 1)   # centroid ~(27, 97)
    cand = _make_square(shape, 20, 150, 15, 1)      # centroid ~(27, 157)

    predicted = {1: np.array([27.0, 157.0])}

    next_frame, winner = find_best_hypothesis(
        current, [cand],
        iou_threshold=0.0,
        max_dist_px=40.0,
        predicted_centroids=predicted,
    )

    assert next_frame is not None
    assert 1 in np.unique(next_frame)


def test_dedup_prevents_collision():
    """Two source nuclei must not both be assigned to the same physical location.

    Nucleus 1 is faint (no good match). Nucleus 2 has a clear match. Two
    hypothesis images both contain a candidate at nucleus 2's next position.
    Without deduplication, nucleus 1 could steal the duplicate column and appear
    at the same location as nucleus 2. With deduplication those two columns are
    merged into one cluster, so only one nucleus can win it.
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
        iou_threshold=0.1,
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
        iou_threshold=0.1,
        max_dist_px=30.0,
    )

    assert next_frame is not None
    tracked = {int(v) for v in np.unique(next_frame) if v != 0}
    assert tracked == {1, 2}
