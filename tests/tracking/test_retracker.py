"""Tests for the centroid-LAP ID retracker."""
import numpy as np
import pytest

from cellflow.tracking.retracker import retrack_frame, retrack_frame_constrained


def _sq(shape, row, col, size, label):
    """Return a label array with one filled square."""
    arr = np.zeros(shape, dtype=np.uint32)
    arr[row:row + size, col:col + size] = label
    return arr


def _add(a, b):
    """Overlay two label arrays (no overlap assumed)."""
    return (a + b).astype(np.uint32)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_identical_frames_preserve_ids():
    """When ref and target are identical, all IDs are unchanged."""
    shape = (50, 50)
    ref = _add(_sq(shape, 5, 5, 8, 1), _sq(shape, 5, 35, 8, 2))
    result = retrack_frame(ref, ref.copy(), max_dist_px=20.0)
    np.testing.assert_array_equal(result, ref)


def test_shifted_cells_remapped():
    """Cells that moved a small distance should inherit the reference ID."""
    shape = (100, 100)
    ref = _add(_sq(shape, 10, 10, 8, 1), _sq(shape, 10, 60, 8, 2))
    # Target has the same cells but with different (random) IDs and slightly shifted.
    tgt = _add(_sq(shape, 12, 12, 8, 42), _sq(shape, 11, 62, 8, 99))

    result = retrack_frame(ref, tgt, max_dist_px=20.0)
    ids = set(int(i) for i in np.unique(result) if i != 0)
    assert ids == {1, 2}, f"Expected {{1, 2}}, got {ids}"


def test_new_cell_gets_fresh_id():
    """A cell in the target with no nearby reference cell gets a fresh ID."""
    shape = (100, 100)
    ref = _sq(shape, 10, 10, 8, 1)
    # Target has the original cell plus a brand-new one far away.
    tgt = _add(_sq(shape, 10, 10, 8, 7), _sq(shape, 70, 70, 8, 8))

    result = retrack_frame(ref, tgt, max_dist_px=20.0)
    ids = set(int(i) for i in np.unique(result) if i != 0)
    # Cell 1 from ref must be present; the new cell must have an ID > max(ref, tgt).
    assert 1 in ids
    new_ids = ids - {1}
    assert len(new_ids) == 1
    assert next(iter(new_ids)) > max(int(ref.max()), int(tgt.max()))


def test_lost_cell_leaves_no_gap():
    """A ref cell absent from the target should not appear in the result."""
    shape = (100, 100)
    ref = _add(_sq(shape, 10, 10, 8, 1), _sq(shape, 10, 60, 8, 2))
    # Target only has one of the two cells.
    tgt = _sq(shape, 10, 10, 8, 55)

    result = retrack_frame(ref, tgt, max_dist_px=20.0)
    ids = set(int(i) for i in np.unique(result) if i != 0)
    assert ids == {1}


def test_exceeds_max_dist_treated_as_new():
    """A large displacement beyond max_dist_px means the cell is treated as new."""
    shape = (200, 200)
    ref = _sq(shape, 10, 10, 8, 1)
    # Target cell is 100 px away, well beyond max_dist_px=20.
    tgt = _sq(shape, 110, 110, 8, 1)

    result = retrack_frame(ref, tgt, max_dist_px=20.0)
    ids = set(int(i) for i in np.unique(result) if i != 0)
    # ID 1 from ref must NOT be assigned; a fresh ID above max should appear.
    assert 1 not in ids
    assert len(ids) == 1
    assert next(iter(ids)) > max(int(ref.max()), int(tgt.max()))


def test_empty_target_returns_zeros():
    """An all-zero target should produce an all-zero result."""
    shape = (50, 50)
    ref = _sq(shape, 5, 5, 8, 1)
    tgt = np.zeros(shape, dtype=np.uint32)
    result = retrack_frame(ref, tgt, max_dist_px=20.0)
    assert result.sum() == 0


def test_empty_ref_assigns_sequential_ids():
    """With no reference cells, target cells get fresh IDs starting from 1."""
    shape = (50, 50)
    ref = np.zeros(shape, dtype=np.uint32)
    tgt = _add(_sq(shape, 5, 5, 8, 42), _sq(shape, 5, 35, 8, 99))
    result = retrack_frame(ref, tgt, max_dist_px=20.0)
    ids = set(int(i) for i in np.unique(result) if i != 0)
    assert len(ids) == 2
    assert all(i >= 1 for i in ids)


def test_id_collision_avoided():
    """New cell IDs must not collide with any existing ID in ref or target."""
    shape = (100, 100)
    ref = _sq(shape, 10, 10, 8, 5)
    # Target has cell near ref (will get id=5) plus a far cell.
    tgt = _add(_sq(shape, 10, 10, 8, 5), _sq(shape, 80, 80, 8, 3))

    result = retrack_frame(ref, tgt, max_dist_px=20.0)
    ids = sorted(int(i) for i in np.unique(result) if i != 0)
    assert 5 in ids
    assert len(ids) == 2
    # The second ID must be above max(ref.max(), tgt.max()) = 5.
    second_id = [i for i in ids if i != 5][0]
    assert second_id > max(int(ref.max()), int(tgt.max()))


def test_output_dtype():
    """Result array must be uint32."""
    shape = (30, 30)
    ref = _sq(shape, 5, 5, 5, 1)
    tgt = _sq(shape, 5, 5, 5, 99)
    result = retrack_frame(ref, tgt, max_dist_px=10.0)
    assert result.dtype == np.uint32


# ---------------------------------------------------------------------------
# retrack_frame_constrained
# ---------------------------------------------------------------------------

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


def test_constrained_empty_locked_set_matches_retrack_frame():
    """With no locked cells, retrack_frame_constrained must produce the same
    result as retrack_frame."""
    shape = (100, 100)
    ref = _add(_sq(shape, 10, 10, 8, 1), _sq(shape, 10, 60, 8, 2))
    tgt = _add(_sq(shape, 12, 12, 8, 42), _sq(shape, 11, 62, 8, 99))

    r1 = retrack_frame(ref, tgt, max_dist_px=20.0)
    r2 = retrack_frame_constrained(ref, tgt, locked_target_ids=set(), max_dist_px=20.0)
    np.testing.assert_array_equal(r1, r2)
