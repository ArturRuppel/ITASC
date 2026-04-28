"""Tests for the frame validation metadata module."""
import json

import pytest

from cellflow.database.validation import (
    invalidate_frame,
    invalidate_track,
    is_track_validated,
    is_validated,
    read_validated_cells_at_frame,
    read_validated_frames,
    read_validated_tracks,
    validate_frame,
    validate_track,
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
# Track-level validation tests (nucleus workflow)
# ---------------------------------------------------------------------------


def test_read_validated_tracks_empty_by_default(pos_dir):
    assert read_validated_tracks(pos_dir) == {}


def test_read_validated_cells_at_frame_empty_by_default(pos_dir):
    assert read_validated_cells_at_frame(pos_dir, 0) == set()


def test_validate_track_adds_frames(pos_dir):
    validate_track(pos_dir, 47, [10, 11, 12])
    result = read_validated_tracks(pos_dir)
    assert result == {47: {10, 11, 12}}


def test_validate_track_idempotent(pos_dir):
    validate_track(pos_dir, 5, [3, 4])
    validate_track(pos_dir, 5, [3, 4])
    assert read_validated_tracks(pos_dir) == {5: {3, 4}}


def test_validate_track_accumulates(pos_dir):
    validate_track(pos_dir, 5, [3])
    validate_track(pos_dir, 5, [4, 5])
    assert read_validated_tracks(pos_dir) == {5: {3, 4, 5}}


def test_validate_track_multiple_cells(pos_dir):
    validate_track(pos_dir, 47, [10, 11])
    validate_track(pos_dir, 82, [3, 4, 5])
    result = read_validated_tracks(pos_dir)
    assert result == {47: {10, 11}, 82: {3, 4, 5}}


def test_validate_track_accepts_list_tuple_generator(pos_dir):
    validate_track(pos_dir, 10, [1, 2])
    validate_track(pos_dir, 10, (3,))
    validate_track(pos_dir, 10, (x for x in [4]))
    assert read_validated_tracks(pos_dir)[10] == {1, 2, 3, 4}


def test_validate_track_empty_frames_is_noop(pos_dir):
    validate_track(pos_dir, 10, [])
    assert read_validated_tracks(pos_dir) == {}


def test_invalidate_track_removes_entry(pos_dir):
    validate_track(pos_dir, 47, [10, 11, 12])
    invalidate_track(pos_dir, 47)
    assert read_validated_tracks(pos_dir) == {}


def test_invalidate_track_nonexistent_is_noop(pos_dir):
    invalidate_track(pos_dir, 999)
    assert read_validated_tracks(pos_dir) == {}


def test_invalidate_track_only_removes_target(pos_dir):
    validate_track(pos_dir, 47, [10, 11])
    validate_track(pos_dir, 82, [3, 4])
    invalidate_track(pos_dir, 47)
    assert read_validated_tracks(pos_dir) == {82: {3, 4}}


def test_is_track_validated_true(pos_dir):
    validate_track(pos_dir, 47, [10, 11])
    assert is_track_validated(pos_dir, 47)


def test_is_track_validated_false_missing(pos_dir):
    assert not is_track_validated(pos_dir, 47)


def test_is_track_validated_false_after_invalidate(pos_dir):
    validate_track(pos_dir, 47, [10])
    invalidate_track(pos_dir, 47)
    assert not is_track_validated(pos_dir, 47)


def test_read_validated_cells_at_frame(pos_dir):
    validate_track(pos_dir, 47, [10, 11, 12])
    validate_track(pos_dir, 82, [3, 4, 5])
    assert read_validated_cells_at_frame(pos_dir, 10) == {47}
    assert read_validated_cells_at_frame(pos_dir, 3) == {82}
    assert read_validated_cells_at_frame(pos_dir, 11) == {47}
    assert read_validated_cells_at_frame(pos_dir, 99) == set()


def test_read_validated_cells_at_frame_multiple_cells(pos_dir):
    validate_track(pos_dir, 1, [5])
    validate_track(pos_dir, 2, [5])
    assert read_validated_cells_at_frame(pos_dir, 5) == {1, 2}


def test_validated_cells_json_location(pos_dir):
    validate_track(pos_dir, 10, [0])
    json_file = pos_dir / "2_nucleus" / "validated_cells.json"
    assert json_file.exists()


def test_validated_cells_json_schema(pos_dir):
    """Keys are cell ID strings; values are lists of frame ints."""
    validate_track(pos_dir, 47, [10, 11, 12])
    validate_track(pos_dir, 82, [3, 4, 5])
    raw = json.loads((pos_dir / "2_nucleus" / "validated_cells.json").read_text())
    assert "47" in raw
    assert "82" in raw
    assert raw["47"] == [10, 11, 12]
    assert raw["82"] == [3, 4, 5]


def test_validated_cells_corrupt_json_returns_empty(pos_dir):
    json_file = pos_dir / "2_nucleus" / "validated_cells.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text("not valid json {{{{")
    assert read_validated_tracks(pos_dir) == {}
    assert read_validated_cells_at_frame(pos_dir, 0) == set()


def test_read_write_roundtrip(pos_dir):
    validate_track(pos_dir, 47, [10, 11, 12, 13, 14])
    validate_track(pos_dir, 82, [3, 4, 5])
    result = read_validated_tracks(pos_dir)
    assert result == {47: {10, 11, 12, 13, 14}, 82: {3, 4, 5}}
