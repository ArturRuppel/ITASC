"""Tests for the constrained centroid-LAP ID retracker."""
import numpy as np

import itasc.tracking_ultrack.retracker as retracker
from itasc.tracking_ultrack.retracker import retrack_frame_constrained


def _sq(shape, row, col, size, label):
    """Return a label array with one filled square."""
    arr = np.zeros(shape, dtype=np.uint32)
    arr[row:row + size, col:col + size] = label
    return arr


def _add(a, b):
    """Overlay two label arrays (no overlap assumed)."""
    return (a + b).astype(np.uint32)


def test_module_does_not_export_unconstrained_retrack_frame():
    assert not hasattr(retracker, "retrack_frame")

def test_constrained_locked_cell_keeps_id():
    """A locked target cell must keep its original ID even when a nearby ref
    cell would cause the LAP to remap it."""
    shape = (100, 100)
    # ref has cell ID 1 at (10,10); target has cell ID 7 at the same spot.
    # Without locking, ID 7 would be remapped to 1.  With locking it stays 7.
    ref = _sq(shape, 10, 10, 8, 1)
    tgt = _sq(shape, 10, 10, 8, 7)

    result = retrack_frame_constrained(ref, tgt, locked_target_ids={7}, max_dist_px=20.0)
    ids = set(int(i) for i in np.unique(result) if i != 0)
    assert ids == {7}


def test_constrained_unlocked_inherits_validated_ref_id():
    """An unvalidated target cell near a validated reference cell should inherit
    the reference ID — this is the primary use-case for the whole feature."""
    shape = (100, 100)
    # ref has validated cell ID 5 at (10,10).
    # target has an unlocked cell at almost the same spot with ID 99.
    ref = _sq(shape, 10, 10, 8, 5)
    tgt = _sq(shape, 11, 11, 8, 99)

    result = retrack_frame_constrained(ref, tgt, locked_target_ids=set(), max_dist_px=20.0)
    ids = set(int(i) for i in np.unique(result) if i != 0)
    assert ids == {5}


def test_constrained_unlocked_cannot_steal_locked_id():
    """If a reference cell has the same ID as a locked target cell, an unlocked
    target cell near that reference must NOT receive the locked ID."""
    shape = (100, 100)
    # locked target cell ID=3 far away (top-left).
    # ref also has ID=3 at (50,50).
    # unlocked target cell at (50,50) — next to ref ID=3.
    # The LAP should NOT assign ID=3 to the unlocked cell; it gets a fresh ID.
    locked_cell = _sq(shape, 5, 5, 8, 3)
    ref_cell = _sq(shape, 50, 50, 8, 3)
    unlocked_cell = _sq(shape, 50, 50, 8, 9)

    tgt = _add(locked_cell, unlocked_cell)
    ref = ref_cell

    result = retrack_frame_constrained(ref, tgt, locked_target_ids={3}, max_dist_px=20.0)
    ids = set(int(i) for i in np.unique(result) if i != 0)
    assert 3 in ids  # locked cell present
    assert 9 not in ids  # original ID gone
    fresh_ids = ids - {3}
    assert len(fresh_ids) == 1
    fresh = next(iter(fresh_ids))
    # Fresh ID must not be 3 (the locked ID) and must be above max existing.
    assert fresh != 3
    assert fresh > max(int(ref.max()), int(tgt.max()))


def test_constrained_unlocked_cannot_steal_reserved_validated_id():
    """A reserved validated track ID must not be assigned to an unlocked target."""
    shape = (100, 100)
    ref = _sq(shape, 50, 50, 8, 5)
    tgt = _sq(shape, 50, 50, 8, 99)

    result = retrack_frame_constrained(
        ref,
        tgt,
        locked_target_ids=set(),
        max_dist_px=20.0,
        reserved_ids={5},
    )

    ids = set(int(i) for i in np.unique(result) if i != 0)
    assert 5 not in ids
    assert len(ids) == 1
    assert next(iter(ids)) > max(int(ref.max()), int(tgt.max()))


def test_constrained_empty_locked_set_remaps_by_centroid_proximity():
    """With no locked cells, targets inherit nearby reference IDs."""
    shape = (100, 100)
    ref = _add(_sq(shape, 10, 10, 8, 1), _sq(shape, 10, 60, 8, 2))
    tgt = _add(_sq(shape, 12, 12, 8, 42), _sq(shape, 11, 62, 8, 99))

    result = retrack_frame_constrained(ref, tgt, locked_target_ids=set(), max_dist_px=20.0)

    ids = set(int(i) for i in np.unique(result) if i != 0)
    assert ids == {1, 2}


def test_iou_term_disambiguates_size_matched_cells():
    """When two candidates are near-equidistant, area/IoU should break the tie
    toward the same-shape match instead of pure centroid distance."""
    shape = (100, 100)
    # ref: small cell ID 1 at (40,40); large cell ID 2 at (40,52).
    ref = _add(_sq(shape, 40, 40, 4, 1), _sq(shape, 40, 52, 12, 2))
    # tgt: a large cell sitting between them, overlapping ID 2's footprint far
    # more than ID 1's. Pure centroid distance could drift toward ID 1; the
    # area+IoU terms must keep it on ID 2.
    tgt = _sq(shape, 40, 52, 12, 77)

    result = retrack_frame_constrained(ref, tgt, locked_target_ids=set(), max_dist_px=30.0)
    ids = set(int(i) for i in np.unique(result) if i != 0)
    assert ids == {2}


def test_greedy_keeps_obvious_match_over_global_tradeoff():
    """A near-perfect overlap must not be traded away to give another target a
    home. A global min-cost solver minimises the *total* cost: here it pushes
    target A off its perfect ref so target B (whose only reachable ref is that
    same one) is not left unmatched. Greedy best-first instead keeps A on its
    true reference and lets B fall through to a fresh id."""
    shape = (100, 100)
    # ref 1 at (40,40); ref 2 far to the right at (40,70).
    ref = _add(_sq(shape, 40, 40, 8, 1), _sq(shape, 40, 70, 8, 2))
    # tgt A overlaps ref 1 perfectly (obvious match -> id 1) and is also within
    # range of ref 2. tgt B sits left of ref 1: it can reach ref 1 but NOT ref 2.
    # A global solver assigns A->ref2 + B->ref1 (total cost lower than leaving B
    # unmatched); greedy gives ref 1 to A and B gets a fresh id.
    tgt = _add(_sq(shape, 40, 40, 8, 50), _sq(shape, 40, 30, 8, 60))

    result = retrack_frame_constrained(
        ref, tgt, locked_target_ids=set(), max_dist_px=35.0
    )
    ids = set(int(i) for i in np.unique(result) if i != 0)
    # A keeps its obvious match; ref 2 is never pulled in to rehome B.
    assert result[40, 40] == 1
    assert 2 not in ids
    # B (left cell) gets a fresh id, not ref 1's id stolen from A.
    assert result[40, 30] != 1
    assert result[40, 30] > max(int(ref.max()), int(tgt.max()))


def test_zero_iou_and_area_weight_recovers_distance_only_match():
    """With only the distance term active, the nearest reference wins."""
    shape = (100, 100)
    ref = _add(_sq(shape, 10, 10, 8, 1), _sq(shape, 10, 60, 8, 2))
    tgt = _sq(shape, 11, 11, 8, 99)

    result = retrack_frame_constrained(
        ref,
        tgt,
        locked_target_ids=set(),
        max_dist_px=80.0,
        area_weight=0.0,
        iou_weight=0.0,
        distance_weight=1.0,
    )
    ids = set(int(i) for i in np.unique(result) if i != 0)
    assert ids == {1}
