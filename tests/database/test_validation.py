"""Tests for the frame validation metadata module."""
import pytest

from cellflow.database.validation import (
    invalidate_frame,
    is_validated,
    read_validated_frames,
    validate_frame,
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
