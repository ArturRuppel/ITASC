"""Tests for the frame validation metadata module."""
import json

import pytest

from cellflow.database.validation import (
    add_correction,
    add_corrections,
    invalidate_frame,
    invalidate_track,
    is_track_validated,
    is_validated,
    read_corrections,
    read_validated_cells_at_frame,
    read_validated_frames,
    read_validated_tracks,
    remap_validated_tracks,
    write_corrections,
    validate_frame,
    validate_track,
    write_validated_frames,
)
from cellflow.tracking_ultrack.corrections import Correction


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


def test_read_corrections_empty_by_default(pos_dir):
    assert read_corrections(pos_dir) == []


def test_write_corrections_roundtrip_flat_schema(pos_dir):
    corrections = [
        Correction(cell_id=7, t=0, kind="validated", y=12.5, x=20.25),
        Correction(cell_id=9, t=1, kind="anchor", y=30.0, x=40.0),
    ]

    write_corrections(pos_dir, corrections)

    assert read_corrections(pos_dir) == corrections
    raw = json.loads((pos_dir / "2_nucleus" / "corrections.json").read_text())
    assert raw == [
        {"cell_id": 7, "t": 0, "kind": "validated", "y": 12.5, "x": 20.25},
        {"cell_id": 9, "t": 1, "kind": "anchor", "y": 30.0, "x": 40.0},
    ]


def test_add_correction_replaces_same_cell_frame_kind(pos_dir):
    add_correction(pos_dir, Correction(cell_id=7, t=0, kind="anchor", y=1.0, x=2.0))
    add_correction(pos_dir, Correction(cell_id=7, t=0, kind="anchor", y=3.0, x=4.0))

    assert read_corrections(pos_dir) == [
        Correction(cell_id=7, t=0, kind="anchor", y=3.0, x=4.0)
    ]


def test_add_corrections_empty_is_noop(pos_dir):
    add_corrections(pos_dir, [])
    assert read_corrections(pos_dir) == []


def test_add_corrections_matches_per_call_add(pos_dir):
    batch = [
        Correction(cell_id=5, t=t, kind="validated", y=float(t), x=float(t))
        for t in range(4)
    ]
    add_corrections(pos_dir, batch)
    assert read_corrections(pos_dir) == sorted(
        batch, key=lambda c: (c.t, c.cell_id, c.kind)
    )


def test_add_corrections_replaces_existing_key_and_dedupes_last_wins(pos_dir):
    add_correction(pos_dir, Correction(cell_id=7, t=0, kind="anchor", y=1.0, x=2.0))
    add_corrections(
        pos_dir,
        [
            Correction(cell_id=7, t=0, kind="anchor", y=3.0, x=4.0),
            Correction(cell_id=7, t=0, kind="anchor", y=5.0, x=6.0),
        ],
    )
    assert read_corrections(pos_dir) == [
        Correction(cell_id=7, t=0, kind="anchor", y=5.0, x=6.0)
    ]


def test_add_corrections_validated_drops_existing_anchors_for_cell(pos_dir):
    write_corrections(
        pos_dir,
        [
            Correction(cell_id=7, t=0, kind="anchor", y=1.0, x=2.0),
            Correction(cell_id=8, t=0, kind="anchor", y=3.0, x=4.0),
        ],
    )
    add_corrections(
        pos_dir, [Correction(cell_id=7, t=0, kind="validated", y=9.0, x=9.0)]
    )
    assert [(c.cell_id, c.t, c.kind) for c in read_corrections(pos_dir)] == [
        (7, 0, "validated"),
        (8, 0, "anchor"),
    ]


def test_invalidate_track_removes_validated_corrections_only(pos_dir):
    write_corrections(
        pos_dir,
        [
            Correction(cell_id=7, t=0, kind="validated", y=1.0, x=2.0),
            Correction(cell_id=7, t=1, kind="anchor", y=3.0, x=4.0),
            Correction(cell_id=8, t=0, kind="validated", y=5.0, x=6.0),
        ],
    )

    invalidate_track(pos_dir, 7)

    assert [(c.cell_id, c.t, c.kind) for c in read_corrections(pos_dir)] == [
        (8, 0, "validated"),
        (7, 1, "anchor"),
    ]


def test_remap_validated_tracks_remaps_flat_corrections(pos_dir):
    write_corrections(
        pos_dir,
        [
            Correction(cell_id=7, t=0, kind="validated", y=1.0, x=2.0),
            Correction(cell_id=8, t=1, kind="anchor", y=3.0, x=4.0),
            Correction(cell_id=9, t=2, kind="validated", y=5.0, x=6.0),
        ],
    )

    remap_validated_tracks(pos_dir, {7: 1, 8: 2})

    assert [(c.cell_id, c.t, c.kind) for c in read_corrections(pos_dir)] == [
        (1, 0, "validated"),
        (2, 1, "anchor"),
    ]


def test_remap_validated_tracks_no_phantom_on_partial_compaction(pos_dir):
    """Contiguous compaction with gaps must remap each store exactly once.

    Regression: ``remap_validated_tracks`` used to write the remapped
    corrections and then re-read them through ``read_validated_tracks`` (which
    merges corrections + legacy) and remap a *second* time into the legacy
    store. When the mapping is not the identity — i.e. any real reassign-to-
    contiguous where new IDs overlap old IDs — the second pass injected phantom
    validations. Here old IDs ``{1,3,5}`` compact to ``{1,2,3}``: track 3→2
    (validated only at t=0) must not inherit track 5's t=9.
    """
    write_corrections(
        pos_dir,
        [
            Correction(cell_id=3, t=0, kind="validated", y=1.0, x=1.0),
            Correction(cell_id=5, t=9, kind="validated", y=1.0, x=1.0),
        ],
    )

    remap_validated_tracks(pos_dir, {1: 1, 3: 2, 5: 3})

    assert read_validated_tracks(pos_dir) == {2: {0}, 3: {9}}


def test_read_corrections_corrupt_json_returns_empty(pos_dir):
    json_file = pos_dir / "2_nucleus" / "corrections.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text("not valid json")

    assert read_corrections(pos_dir) == []
