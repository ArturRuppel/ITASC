"""Unit tests for the Qt-free per-position pipeline status source (the rail)."""
from __future__ import annotations

import os
from pathlib import Path

from cellflow.napari._stage_status import (
    DONE,
    MISSING,
    STAGE_CELL,
    STAGE_CELLPOSE,
    STAGE_CONTACTS,
    STAGE_NUCLEUS,
    STALE,
    UNKNOWN,
    WORKING,
    position_stage_status,
)


def _touch(path: Path, *, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_empty_position_all_missing(tmp_path: Path) -> None:
    status = position_stage_status(tmp_path)
    assert status == {
        STAGE_CELLPOSE: MISSING,
        STAGE_NUCLEUS: MISSING,
        STAGE_CELL: MISSING,
        STAGE_CONTACTS: MISSING,
    }


def test_none_position_all_unknown() -> None:
    status = position_stage_status(None)
    assert set(status.values()) == {UNKNOWN}
    assert set(status) == {STAGE_CELLPOSE, STAGE_NUCLEUS, STAGE_CELL, STAGE_CONTACTS}


def test_cellpose_done_needs_both_foreground_and_contours(tmp_path: Path) -> None:
    _touch(tmp_path / "1_cellpose" / "nucleus_foreground.tif")
    # Only foreground yet — not done.
    assert position_stage_status(tmp_path)[STAGE_CELLPOSE] == MISSING
    _touch(tmp_path / "1_cellpose" / "nucleus_contours.tif")
    assert position_stage_status(tmp_path)[STAGE_CELLPOSE] == DONE


def test_nucleus_working_then_committed_then_stale(tmp_path: Path) -> None:
    working = tmp_path / "2_nucleus" / "tracked_labels.tif"
    committed = tmp_path / "nucleus_labels.tif"

    _touch(working, mtime=1000)
    assert position_stage_status(tmp_path)[STAGE_NUCLEUS] == WORKING

    _touch(committed, mtime=2000)
    assert position_stage_status(tmp_path)[STAGE_NUCLEUS] == DONE

    # Re-run the working file after commit → stale.
    _touch(working, mtime=3000)
    assert position_stage_status(tmp_path)[STAGE_NUCLEUS] == STALE


def test_cell_working_then_committed(tmp_path: Path) -> None:
    working = tmp_path / "3_cell" / "tracked_labels.tif"
    committed = tmp_path / "cell_labels.tif"

    _touch(working, mtime=1000)
    assert position_stage_status(tmp_path)[STAGE_CELL] == WORKING

    _touch(committed, mtime=2000)
    assert position_stage_status(tmp_path)[STAGE_CELL] == DONE


def test_contacts_done_when_h5_present(tmp_path: Path) -> None:
    assert position_stage_status(tmp_path)[STAGE_CONTACTS] == MISSING
    _touch(tmp_path / "contact_analysis.h5")
    assert position_stage_status(tmp_path)[STAGE_CONTACTS] == DONE


def test_stages_are_independent(tmp_path: Path) -> None:
    # A committed nucleus does not imply cell or contacts progress.
    _touch(tmp_path / "2_nucleus" / "tracked_labels.tif", mtime=1000)
    _touch(tmp_path / "nucleus_labels.tif", mtime=2000)
    status = position_stage_status(tmp_path)
    assert status[STAGE_NUCLEUS] == DONE
    assert status[STAGE_CELL] == MISSING
    assert status[STAGE_CONTACTS] == MISSING
    assert status[STAGE_CELLPOSE] == MISSING
