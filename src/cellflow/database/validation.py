"""Persistent validation metadata for the nucleus workflow.

Frame-level validation (validated_frames.json):
    A "fully-validated" frame is one where every current (non-zero) cell ID has
    been individually validated.  The file acts as a *cache* so UI counters can
    count fully-validated frames without scanning the whole stack.

    Schema: JSON array of ints, e.g. [0, 3, 7].

Cell-level validation (validated_cells.json):
    Tracks which specific cell IDs have been validated at each frame.

    Schema: JSON object with string-keyed frame indices mapping to arrays of
    validated cell IDs, e.g. {"0": [3, 7, 11], "5": [1, 2, 3, 4]}.
    Frames with zero validated cells are omitted entirely (sparse).
    Cell ID 0 (background) is always excluded.
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
# Cell-level validation
# ---------------------------------------------------------------------------

def _cells_path(pos_dir: Path) -> Path:
    return pos_dir / "2_nucleus" / "validated_cells.json"


def read_all_validated_cells(pos_dir: Path) -> dict[int, set[int]]:
    """Return the full {t: {cell_ids}} map. Empty dict if file missing/corrupt."""
    p = _cells_path(pos_dir)
    if not p.exists():
        return {}
    try:
        raw: dict = json.loads(p.read_text())
        return {int(k): set(int(v) for v in vs) - {0} for k, vs in raw.items()}
    except Exception:
        return {}


def write_all_validated_cells(pos_dir: Path, data: dict[int, set[int]]) -> None:
    """Persist the full map. Frames with empty sets are dropped from the file."""
    p = _cells_path(pos_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        str(t): sorted(ids - {0})
        for t, ids in data.items()
        if ids - {0}
    }
    p.write_text(json.dumps(serialisable))


def read_validated_cells(pos_dir: Path, t: int) -> set[int]:
    """Return the set of validated cell IDs at frame t (empty set if none).

    Cell ID 0 is background and is always excluded.
    """
    return read_all_validated_cells(pos_dir).get(t, set())


def validate_cells(pos_dir: Path, t: int, ids: Iterable[int]) -> None:
    """Add the given cell IDs to frame t's validated set. ID 0 silently ignored.

    Does NOT update validated_frames.json — the caller would need to supply
    the full set of current cell IDs to decide whether the frame is now fully
    validated.  Use validate_all_cells_in_frame for that.
    """
    ids_set = set(ids) - {0}
    if not ids_set:
        return
    data = read_all_validated_cells(pos_dir)
    existing = data.get(t, set())
    data[t] = existing | ids_set
    write_all_validated_cells(pos_dir, data)


def invalidate_cells(pos_dir: Path, t: int, ids: Iterable[int]) -> None:
    """Remove the given cell IDs from frame t's validated set.

    Non-existent IDs are a no-op.  If the set becomes empty the frame entry is
    dropped entirely.  If frame t was in validated_frames.json it is removed
    there too — invalidating any cell can never leave the frame fully validated.
    """
    ids_set = set(ids) - {0}
    data = read_all_validated_cells(pos_dir)
    if t in data:
        data[t] = data[t] - ids_set
        if not data[t]:
            del data[t]
        write_all_validated_cells(pos_dir, data)
    # Remove from the fully-validated frames cache unconditionally when any
    # matching IDs were requested (even if t wasn't in validated_cells).
    if ids_set:
        invalidate_frame(pos_dir, t)


def validate_all_cells_in_frame(pos_dir: Path, t: int, all_ids: set[int]) -> None:
    """Mark every cell ID in ``all_ids`` as validated at frame t (excluding 0).

    Also adds t to validated_frames.json (the fully-validated cache).
    ``all_ids`` is the caller-provided complete set of current cell IDs at t;
    this function does not read the labelmap.
    """
    ids_set = all_ids - {0}
    data = read_all_validated_cells(pos_dir)
    existing = data.get(t, set())
    data[t] = existing | ids_set
    write_all_validated_cells(pos_dir, data)
    validate_frame(pos_dir, t)


def is_frame_fully_validated(pos_dir: Path, t: int, current_ids: set[int]) -> bool:
    """Return True iff every id in ``current_ids`` (minus 0) is validated at t.

    Returns False when ``current_ids`` contains no non-zero IDs (nothing to
    validate).
    """
    required = current_ids - {0}
    if not required:
        return False
    validated = read_validated_cells(pos_dir, t)
    return required <= validated
