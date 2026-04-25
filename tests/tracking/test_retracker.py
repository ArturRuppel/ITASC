"""Tests for the centroid-LAP ID retracker."""
import numpy as np
import pytest

from cellflow.tracking.retracker import retrack_frame


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
