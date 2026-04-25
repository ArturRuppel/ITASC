"""Persistent set of validated frame indices for the nucleus workflow.

A "validated" frame has IDs that are trusted anchors — the retracker will not
modify them and can use them as reference frames when relabelling neighbours.

Schema: JSON file at <pos_dir>/2_nucleus/validated_frames.json containing a
single array of integers, e.g. [0, 3, 7].
"""
from __future__ import annotations

import json
from pathlib import Path


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
