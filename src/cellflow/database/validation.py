"""Persistent validation metadata for the nucleus workflow.

Frame-level validation (validated_frames.json):
    A "fully-validated" frame is one where every current (non-zero) cell ID has
    been individually validated.  The file acts as a *cache* so UI counters can
    count fully-validated frames without scanning the whole stack.
    Used by the cell workflow (3_cell). Stays untouched.

    Schema: JSON array of ints, e.g. [0, 3, 7].

Track-level validation (validated_cells.json):
    Tracks which frames have been validated for each cell (track) ID.
    Used by the nucleus workflow (2_nucleus).

    Schema: JSON object keyed by cell ID string, value is a list of frame ints,
    e.g. {"47": [10, 11, 12], "82": [3, 4, 5]}.
    Cell IDs with no validated frames are omitted entirely (sparse).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _path(pos_dir: Path) -> Path:
    return pos_dir / "2_nucleus" / "validated_frames.json"


def read_validated_frames(pos_dir: Path) -> set[int]:
    """Return the set of validated frame indices, or an empty set if none."""
    p = _path(pos_dir)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        return set(int(t) for t in data)
    except Exception:
        return set()


def write_validated_frames(pos_dir: Path, frames: set[int]) -> None:
    """Persist the full set of validated frames."""
    p = _path(pos_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(frames)))


def validate_frame(pos_dir: Path, t: int) -> None:
    """Mark frame t as validated."""
    frames = read_validated_frames(pos_dir)
    frames.add(t)
    write_validated_frames(pos_dir, frames)


def invalidate_frame(pos_dir: Path, t: int) -> None:
    """Remove the validated mark from frame t."""
    frames = read_validated_frames(pos_dir)
    frames.discard(t)
    write_validated_frames(pos_dir, frames)


def is_validated(pos_dir: Path, t: int) -> bool:
    """Return True if frame t is in the validated set."""
    return t in read_validated_frames(pos_dir)


# ---------------------------------------------------------------------------
# Track-level validation (nucleus workflow)
# ---------------------------------------------------------------------------

def _cells_path(pos_dir: Path) -> Path:
    return pos_dir / "2_nucleus" / "validated_cells.json"


def read_validated_tracks(pos_dir: Path) -> dict[int, set[int]]:
    """Return {cell_id: {frames}} for all validated tracks.

    Empty dict if the file is missing or corrupt.
    JSON keys are cell ID strings; values are lists of frame ints.
    """
    p = _cells_path(pos_dir)
    if not p.exists():
        return {}
    try:
        raw: dict = json.loads(p.read_text())
        return {int(k): set(int(f) for f in vs) for k, vs in raw.items() if vs}
    except Exception:
        return {}


def _write_validated_tracks(pos_dir: Path, data: dict[int, set[int]]) -> None:
    """Persist the full {cell_id: {frames}} map. Entries with empty sets are dropped."""
    p = _cells_path(pos_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        str(cell_id): sorted(frames)
        for cell_id, frames in data.items()
        if frames
    }
    p.write_text(json.dumps(serialisable))


def read_validated_cells_at_frame(pos_dir: Path, t: int) -> set[int]:
    """Return all cell IDs that have frame *t* in their validated set.

    Derived from the track-keyed store; suitable for overlay rendering.
    """
    return {cell_id for cell_id, frames in read_validated_tracks(pos_dir).items() if t in frames}


def is_track_validated(pos_dir: Path, cell_id: int) -> bool:
    """Return True if *cell_id* has any entry in the validated-tracks store."""
    return cell_id in read_validated_tracks(pos_dir)


def validate_track(pos_dir: Path, cell_id: int, frames: Iterable[int]) -> None:
    """Add the given frames to *cell_id*'s validated set (idempotent, accumulates).

    Creates an entry for *cell_id* if none exists yet.
    """
    frames_set = set(frames)
    if not frames_set:
        return
    data = read_validated_tracks(pos_dir)
    existing = data.get(cell_id, set())
    data[cell_id] = existing | frames_set
    _write_validated_tracks(pos_dir, data)


def invalidate_track(pos_dir: Path, cell_id: int) -> None:
    """Remove the entire entry for *cell_id* from the validated-tracks store.

    No-op if *cell_id* is not present.
    """
    data = read_validated_tracks(pos_dir)
    if cell_id in data:
        del data[cell_id]
        _write_validated_tracks(pos_dir, data)
