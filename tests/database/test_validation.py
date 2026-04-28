"""Tests for the frame validation metadata module."""
import json

import pytest

from cellflow.database.validation import (
    invalidate_frame,
    invalidate_cells,
    is_frame_fully_validated,
    is_validated,
    read_all_validated_cells,
    read_validated_cells,
    read_validated_frames,
    validate_all_cells_in_frame,
    validate_cells,
    validate_frame,
    write_all_validated_cells,
    write_validated_frames,
)


@pytest.fixture()
def pos_dir(tmp_path):
    d = tmp_path / "pos000"
    d.mkdir()
    return d


def test_empty_by_default(pos_dir):
    assert read_validated_frames(pos_dir) == set()
    assert not is_validated(pos_dir, 0)


def test_validate_single_frame(pos_dir):
    validate_frame(pos_dir, 3)
    assert is_validated(pos_dir, 3)
    assert not is_validated(pos_dir, 0)


def test_validate_multiple_frames(pos_dir):
    for t in (0, 2, 5):
        validate_frame(pos_dir, t)
    assert read_validated_frames(pos_dir) == {0, 2, 5}


def test_invalidate_removes_frame(pos_dir):
    validate_frame(pos_dir, 1)
    validate_frame(pos_dir, 2)
    invalidate_frame(pos_dir, 1)
    assert not is_validated(pos_dir, 1)
    assert is_validated(pos_dir, 2)


def test_invalidate_nonexistent_is_noop(pos_dir):
    """Removing a frame that was never validated should not raise."""
    invalidate_frame(pos_dir, 99)
    assert read_validated_frames(pos_dir) == set()


def test_write_read_roundtrip(pos_dir):
    frames = {0, 4, 7, 11}
    write_validated_frames(pos_dir, frames)
    assert read_validated_frames(pos_dir) == frames


def test_file_persists_across_calls(pos_dir):
    validate_frame(pos_dir, 0)
    validate_frame(pos_dir, 5)
    # Re-read without in-memory cache.
    result = read_validated_frames(pos_dir)
    assert result == {0, 5}


def test_json_file_location(pos_dir):
    validate_frame(pos_dir, 0)
    json_file = pos_dir / "2_nucleus" / "validated_frames.json"
    assert json_file.exists()


def test_corrupt_json_returns_empty(pos_dir):
    json_file = pos_dir / "2_nucleus" / "validated_frames.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text("not valid json {{")
    assert read_validated_frames(pos_dir) == set()


# ---------------------------------------------------------------------------
# Cell-level validation tests
# ---------------------------------------------------------------------------


def test_read_validated_cells_empty_by_default(pos_dir):
    assert read_validated_cells(pos_dir, 0) == set()


def test_read_all_validated_cells_empty_by_default(pos_dir):
    assert read_all_validated_cells(pos_dir) == {}


def test_validate_cells_adds_ids(pos_dir):
    validate_cells(pos_dir, 3, [1, 5, 9])
    assert read_validated_cells(pos_dir, 3) == {1, 5, 9}


def test_validate_cells_id0_silently_dropped(pos_dir):
    validate_cells(pos_dir, 0, [0, 2, 4])
    result = read_validated_cells(pos_dir, 0)
    assert 0 not in result
    assert result == {2, 4}


def test_validate_cells_only_zero_is_noop(pos_dir):
    validate_cells(pos_dir, 0, [0])
    assert read_all_validated_cells(pos_dir) == {}


def test_validate_cells_idempotent(pos_dir):
    validate_cells(pos_dir, 2, [7, 8])
    validate_cells(pos_dir, 2, [7, 8])
    assert read_validated_cells(pos_dir, 2) == {7, 8}


def test_validate_cells_accumulates(pos_dir):
    validate_cells(pos_dir, 1, [10])
    validate_cells(pos_dir, 1, [20])
    assert read_validated_cells(pos_dir, 1) == {10, 20}


def test_validate_cells_accepts_set_tuple_generator(pos_dir):
    validate_cells(pos_dir, 5, {3, 4})
    validate_cells(pos_dir, 5, (5, 6))
    validate_cells(pos_dir, 5, (x for x in [7]))
    assert read_validated_cells(pos_dir, 5) == {3, 4, 5, 6, 7}


def test_invalidate_cells_removes_specified(pos_dir):
    validate_cells(pos_dir, 4, [1, 2, 3])
    invalidate_cells(pos_dir, 4, [2])
    assert read_validated_cells(pos_dir, 4) == {1, 3}


def test_invalidate_cells_nonexistent_id_is_noop(pos_dir):
    validate_cells(pos_dir, 4, [1, 2])
    invalidate_cells(pos_dir, 4, [99])
    assert read_validated_cells(pos_dir, 4) == {1, 2}


def test_invalidate_cells_last_id_removes_frame_entry(pos_dir):
    validate_cells(pos_dir, 6, [5])
    invalidate_cells(pos_dir, 6, [5])
    data = read_all_validated_cells(pos_dir)
    assert 6 not in data


def test_invalidate_cells_removes_from_frames_cache(pos_dir):
    # Mark frame as fully validated in the cache.
    validate_frame(pos_dir, 7)
    assert is_validated(pos_dir, 7)
    # Invalidating any cell must evict it from the frames cache.
    validate_cells(pos_dir, 7, [1, 2, 3])
    invalidate_cells(pos_dir, 7, [1])
    assert not is_validated(pos_dir, 7)


def test_invalidate_cells_on_frame_not_in_cache_is_safe(pos_dir):
    validate_cells(pos_dir, 8, [1, 2])
    # Frame 8 is NOT in validated_frames.json — should not raise.
    invalidate_cells(pos_dir, 8, [1])
    assert read_validated_cells(pos_dir, 8) == {2}


def test_validate_all_cells_in_frame_writes_cells_and_cache(pos_dir):
    validate_all_cells_in_frame(pos_dir, 10, {1, 2, 3})
    assert read_validated_cells(pos_dir, 10) == {1, 2, 3}
    assert is_validated(pos_dir, 10)


def test_validate_all_cells_in_frame_excludes_zero(pos_dir):
    validate_all_cells_in_frame(pos_dir, 11, {0, 5, 6})
    result = read_validated_cells(pos_dir, 11)
    assert 0 not in result
    assert result == {5, 6}


def test_is_frame_fully_validated_true(pos_dir):
    validate_cells(pos_dir, 3, [1, 2, 3])
    assert is_frame_fully_validated(pos_dir, 3, {1, 2, 3})


def test_is_frame_fully_validated_partial(pos_dir):
    validate_cells(pos_dir, 3, [1, 2])
    assert not is_frame_fully_validated(pos_dir, 3, {1, 2, 3})


def test_is_frame_fully_validated_empty_current_ids(pos_dir):
    # No cells to validate → False.
    assert not is_frame_fully_validated(pos_dir, 0, set())


def test_is_frame_fully_validated_current_ids_only_zero(pos_dir):
    # Only background → treated as empty → False.
    assert not is_frame_fully_validated(pos_dir, 0, {0})


def test_is_frame_fully_validated_ignores_zero_in_current(pos_dir):
    validate_cells(pos_dir, 5, [2, 3])
    # current_ids has background; only non-zero IDs matter.
    assert is_frame_fully_validated(pos_dir, 5, {0, 2, 3})


def test_validated_cells_json_location(pos_dir):
    validate_cells(pos_dir, 0, [1])
    json_file = pos_dir / "2_nucleus" / "validated_cells.json"
    assert json_file.exists()


def test_validated_cells_corrupt_json_returns_empty(pos_dir):
    json_file = pos_dir / "2_nucleus" / "validated_cells.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text("not valid json {{{{")
    assert read_all_validated_cells(pos_dir) == {}
    assert read_validated_cells(pos_dir, 0) == set()


def test_write_read_all_validated_cells_roundtrip(pos_dir):
    data = {0: {1, 2, 3}, 5: {10, 20}}
    write_all_validated_cells(pos_dir, data)
    result = read_all_validated_cells(pos_dir)
    assert result == data


def test_write_all_validated_cells_drops_empty_frames(pos_dir):
    data = {0: {1, 2}, 3: set()}
    write_all_validated_cells(pos_dir, data)
    result = read_all_validated_cells(pos_dir)
    assert 3 not in result
    assert result == {0: {1, 2}}


def test_write_all_validated_cells_drops_frame_with_only_zero(pos_dir):
    data = {0: {0}, 2: {1}}
    write_all_validated_cells(pos_dir, data)
    result = read_all_validated_cells(pos_dir)
    assert 0 not in result
    assert result == {2: {1}}


def test_json_uses_string_keys(pos_dir):
    write_all_validated_cells(pos_dir, {7: {3, 4}})
    raw = json.loads((pos_dir / "2_nucleus" / "validated_cells.json").read_text())
    assert "7" in raw


def test_multiple_frames_independent(pos_dir):
    validate_cells(pos_dir, 0, [1, 2])
    validate_cells(pos_dir, 1, [3, 4])
    assert read_validated_cells(pos_dir, 0) == {1, 2}
    assert read_validated_cells(pos_dir, 1) == {3, 4}
    assert read_validated_cells(pos_dir, 2) == set()
