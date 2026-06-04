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
from collections.abc import Iterable

import numpy as np

from cellflow.tracking_ultrack.corrections import Correction


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
    corrections = read_corrections(pos_dir)
    if corrections:
        data: dict[int, set[int]] = {}
        for correction in corrections:
            if correction.kind == "validated":
                data.setdefault(int(correction.cell_id), set()).add(int(correction.t))
        legacy = _read_legacy_validated_tracks(pos_dir)
        for cell_id, frames in legacy.items():
            data.setdefault(cell_id, set()).update(frames)
        return data
    return _read_legacy_validated_tracks(pos_dir)


def _read_legacy_validated_tracks(pos_dir: Path) -> dict[int, set[int]]:
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

    Creates an entry for *cell_id* if none exists yet. Any existing per-frame
    anchor corrections for the same cell are dropped — whole-track validation
    supersedes them (the post-solve paste-back stamps the validated geometry,
    so anchor pins on the same cell are redundant).
    """
    frames_set = set(frames)
    if not frames_set:
        return
    data = read_validated_tracks(pos_dir)
    existing = data.get(cell_id, set())
    data[cell_id] = existing | frames_set
    _write_validated_tracks(pos_dir, data)

    corrections = read_corrections(pos_dir)
    if corrections:
        filtered = [
            c
            for c in corrections
            if not (int(c.cell_id) == int(cell_id) and c.kind == "anchor")
        ]
        if len(filtered) != len(corrections):
            write_corrections(pos_dir, filtered)



def invalidate_track(pos_dir: Path, cell_id: int) -> None:
    """Remove the entire entry for *cell_id* from the validated-tracks store.

    No-op if *cell_id* is not present.
    """
    corrections = read_corrections(pos_dir)
    if corrections:
        write_corrections(
            pos_dir,
            [
                c
                for c in corrections
                if not (int(c.cell_id) == int(cell_id) and c.kind == "validated")
            ],
        )
    data = read_validated_tracks(pos_dir)
    if cell_id in data:
        del data[cell_id]
        _write_validated_tracks(pos_dir, data)


def remap_validated_tracks(pos_dir: Path, old_to_new: dict[int, int]) -> None:
    """Remap cell IDs in the validated-tracks store using *old_to_new* mapping.

    IDs not present in the mapping are dropped.
    """
    corrections = read_corrections(pos_dir)
    # Read the legacy store *directly* (not via ``read_validated_tracks``, which
    # merges in the corrections we are about to rewrite) so each store is remapped
    # exactly once. Reading the merged view here would remap the corrections-derived
    # IDs a second time and inject phantom validations whenever the mapping is not
    # the identity — i.e. for any real contiguous compaction with gaps.
    legacy = _read_legacy_validated_tracks(pos_dir)
    if corrections:
        write_corrections(
            pos_dir,
            [
                Correction(
                    cell_id=int(old_to_new[int(c.cell_id)]),
                    t=int(c.t),
                    kind=c.kind,
                    y=float(c.y),
                    x=float(c.x),
                )
                for c in corrections
                if int(c.cell_id) in old_to_new
            ],
        )
    remapped = {
        old_to_new[cell_id]: frames
        for cell_id, frames in legacy.items()
        if cell_id in old_to_new
    }
    _write_validated_tracks(pos_dir, remapped)


# ---------------------------------------------------------------------------
# Unified correction list (nucleus workflow)
# ---------------------------------------------------------------------------

def _corrections_path(pos_dir: Path) -> Path:
    return pos_dir / "2_nucleus" / "corrections.json"


def read_corrections(pos_dir: Path) -> list[Correction]:
    """Return persisted per-frame corrections, or an empty list if unavailable."""
    p = _corrections_path(pos_dir)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text())
        corrections = [
            Correction(
                cell_id=int(item["cell_id"]),
                t=int(item["t"]),
                kind=item["kind"],
                y=float(item["y"]),
                x=float(item["x"]),
            )
            for item in raw
        ]
    except Exception:
        return []
    return sorted(corrections, key=lambda c: (c.t, c.cell_id, c.kind))


def write_corrections(pos_dir: Path, corrections: Iterable[Correction]) -> None:
    """Persist the full flat correction list to ``corrections.json``."""
    p = _corrections_path(pos_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    serialisable = [
        {
            "cell_id": int(c.cell_id),
            "t": int(c.t),
            "kind": c.kind,
            "y": float(c.y),
            "x": float(c.x),
        }
        for c in sorted(corrections, key=lambda item: (item.t, item.cell_id, item.kind))
    ]
    p.write_text(json.dumps(serialisable))


def add_correction(pos_dir: Path, correction: Correction) -> None:
    """Add or replace a correction for the same cell, frame, and kind.

    Writing a ``validated`` correction for a cell additionally drops every
    ``anchor`` correction for that cell — validation is whole-track and
    supersedes any anchor pinning on the same cell.
    """
    add_corrections(pos_dir, [correction])


def add_corrections(pos_dir: Path, corrections: Iterable[Correction]) -> None:
    """Add or replace several corrections with a single read+write.

    Each correction replaces any existing one for the same cell, frame, and
    kind (later entries in *corrections* win over earlier ones). As with
    :func:`add_correction`, a ``validated`` correction for a cell additionally
    drops every ``anchor`` correction for that cell.

    Unlike calling :func:`add_correction` in a loop, this reads and writes
    ``corrections.json`` exactly once, so validating a long track stays cheap
    instead of re-parsing and re-serialising the whole file per frame.
    """
    new = list(corrections)
    if not new:
        return

    # Keys (cell_id, t, kind) being written; cell_ids whose anchors validation drops.
    replaced_keys = {(int(c.cell_id), int(c.t), c.kind) for c in new}
    validated_cells = {int(c.cell_id) for c in new if c.kind == "validated"}

    def _keep(c: Correction) -> bool:
        if (int(c.cell_id), int(c.t), c.kind) in replaced_keys:
            return False
        if c.kind == "anchor" and int(c.cell_id) in validated_cells:
            return False
        return True

    existing = [c for c in read_corrections(pos_dir) if _keep(c)]

    # Dedupe within the batch itself (last wins for a repeated key).
    deduped: dict[tuple[int, int, str], Correction] = {}
    for c in new:
        deduped[(int(c.cell_id), int(c.t), c.kind)] = c

    write_corrections(pos_dir, existing + list(deduped.values()))


def _centroid_of(mask: np.ndarray) -> tuple[float, float] | None:
    if mask.ndim == 3:
        mask = mask.any(axis=0)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def add_anchor(
    pos_dir: Path,
    cell_id: int,
    t: int,
    y: float,
    x: float,
    tracked_labels: np.ndarray,
) -> int:
    """Add an anchor for *cell_id* at frame *t* and back-fill intermediate frames.

    If another anchor for the same cell already exists at some frame ``A``,
    find the nearest such ``A`` (smallest ``|A - t|``). If ``cell_id`` is
    present in ``tracked_labels`` at *every* frame in ``[min(A, t), max(A, t)]``,
    add an anchor correction for each intermediate frame using the cell's
    centroid in ``tracked_labels`` at that frame.

    Returns the number of intermediate anchors added (0 if no neighbor exists
    or the track is not consecutive).

    The strictness rule guarantees we never anchor onto a frame where the cell
    isn't actually present in the corrected layer.
    """
    cell_id = int(cell_id)
    t = int(t)

    existing = read_corrections(pos_dir)
    other_anchor_frames = sorted(
        int(c.t)
        for c in existing
        if int(c.cell_id) == cell_id and c.kind == "anchor" and int(c.t) != t
    )

    add_correction(
        pos_dir,
        Correction(cell_id=cell_id, t=t, kind="anchor", y=float(y), x=float(x)),
    )

    if not other_anchor_frames:
        return 0

    neighbor = min(other_anchor_frames, key=lambda f: (abs(f - t), f))
    lo, hi = (neighbor, t) if neighbor < t else (t, neighbor)
    interior = list(range(lo + 1, hi))
    if not interior:
        return 0

    n_frames = int(tracked_labels.shape[0])
    centroids: dict[int, tuple[float, float]] = {}
    for f in interior:
        if f < 0 or f >= n_frames:
            return 0
        mask = np.asarray(tracked_labels[f] == cell_id)
        c = _centroid_of(mask)
        if c is None:
            return 0
        centroids[f] = c

    for f, (cy, cx) in centroids.items():
        add_correction(
            pos_dir,
            Correction(cell_id=cell_id, t=f, kind="anchor", y=cy, x=cx),
        )
    return len(centroids)
