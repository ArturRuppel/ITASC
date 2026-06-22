"""The curation artifact — hand QC exclusions joined onto the measurement tables.

A separate, git-versioned tidy table kept apart from the disposable measurement
tables (the dividing line is who made the decisions in it). Schema, one row per
exclusion::

    experiment_id, position_id, frame, excluded, exclusion_reason

``frame`` empty/NA means *the whole position* (every frame). At export the table
is **left-joined** onto a measurement table by the natural keys the table already
carries — a frame-level exclusion matches ``(experiment_id, position_id, frame)``;
a position-level exclusion (``frame`` NA) matches ``(experiment_id, position_id)``
— marking matched rows ``excluded = True`` and copying the reason. Rows with no
entry default to kept; filter, don't delete. The measurement source is never
mutated.

This restores the curation that commit 95df159 removed, re-keyed on the natural
keys (the old version, at rev 39d0df2, joined on a deterministic row ``id``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = [
    "CURATION_COLUMNS",
    "empty_curation",
    "read_curation",
    "write_curation",
    "append_exclusion",
    "remove_exclusion",
    "apply_curation",
    "filter_excluded",
]

#: The columns a curation CSV carries.
CURATION_COLUMNS = (
    "experiment_id",
    "position_id",
    "frame",
    "excluded",
    "exclusion_reason",
)


def read_curation(path: Path | str | None) -> pd.DataFrame | None:
    """Read the curation CSV at *path*, or ``None`` when *path* is unset/absent.

    A missing file is not an error: an uncurated series simply keeps every row.
    """
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    return pd.read_csv(path)


def empty_curation() -> pd.DataFrame:
    """An empty curation table with the canonical columns (object dtype).

    Object columns keep an empty ``frame`` as ``NA`` (whole-position) rather than
    coercing the column to a float that would render NaN on write.
    """
    return pd.DataFrame({col: pd.Series(dtype="object") for col in CURATION_COLUMNS})


def write_curation(path: Path | str, curation: pd.DataFrame) -> None:
    """Write *curation* to *path* as CSV (creating the parent dir), index-free."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    curation.to_csv(path, index=False)


def append_exclusion(
    curation: pd.DataFrame | None,
    *,
    experiment_id: str,
    position_id: str,
    frame: int | None,
    reason: str,
) -> pd.DataFrame:
    """Return *curation* with one exclusion row added.

    ``frame=None`` records a **whole-position** exclusion (stored as NA); a numeric
    *frame* records a single-frame exclusion. Idempotent on the
    ``(experiment_id, position_id, frame)`` key: an existing row with the same key
    is replaced (its reason updated) rather than duplicated. The input is not
    mutated.
    """
    base = empty_curation() if curation is None else curation
    out = remove_exclusion(
        base, experiment_id=experiment_id, position_id=position_id, frame=frame
    )
    new_row = {
        "experiment_id": str(experiment_id),
        "position_id": str(position_id),
        "frame": pd.NA if frame is None else int(frame),
        "excluded": True,
        "exclusion_reason": str(reason),
    }
    return pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)


def remove_exclusion(
    curation: pd.DataFrame | None,
    *,
    experiment_id: str,
    position_id: str,
    frame: int | None,
) -> pd.DataFrame:
    """Return *curation* without the row(s) matching the given key.

    ``frame=None`` removes the **whole-position** row (``frame`` NA); a numeric
    *frame* removes that single-frame row. A non-matching key is a no-op. The
    input is not mutated.
    """
    if curation is None or len(curation) == 0:
        return empty_curation()
    out = curation.copy()
    key = (out["experiment_id"].astype(str) == str(experiment_id)) & (
        out["position_id"].astype(str) == str(position_id)
    )
    if frame is None:
        key &= out["frame"].isna()
    else:
        frame_numeric = pd.to_numeric(out["frame"], errors="coerce")
        key &= frame_numeric.notna() & (frame_numeric == float(frame))
    return out.loc[~key].reset_index(drop=True)


def apply_curation(
    table: pd.DataFrame, curation: pd.DataFrame | None
) -> pd.DataFrame:
    """Return *table* with ``excluded`` / ``exclusion_reason`` columns marked.

    A measurement row is marked excluded iff a curation entry matches it by either
    ``(experiment_id, position_id, frame)`` (frame-level) or
    ``(experiment_id, position_id)`` with the curation ``frame`` NA
    (position-level). Keys are compared as strings (CSV round-trips ids as
    strings); ``frame`` as an integer. Unmatched rows default to kept with an empty
    reason. The input frame is not mutated.
    """
    out = table.copy()
    out["excluded"] = False
    out["exclusion_reason"] = ""
    if curation is None or len(curation) == 0:
        return out

    exp = out["experiment_id"].astype(str)
    pos = out["position_id"].astype(str)

    for _, entry in curation.iterrows():
        if not bool(entry.get("excluded", True)):
            continue  # a future un-exclude override; ignored for now
        key = (exp == str(entry["experiment_id"])) & (pos == str(entry["position_id"]))
        frame = entry.get("frame")
        if pd.notna(frame):
            if "frame" not in out.columns:
                continue  # a frame-level entry cannot match a frameless table
            key &= out["frame"].astype("int64") == int(frame)
        out.loc[key, "excluded"] = True
        out.loc[key, "exclusion_reason"] = str(entry.get("exclusion_reason", ""))

    return out


def filter_excluded(marked: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop the rows :func:`apply_curation` marked excluded.

    Returns ``(kept, n_dropped)`` where *kept* has the two marker columns removed
    (so it matches the source schema) and a reset index. A frame with no
    ``excluded`` column is returned unchanged with ``n_dropped == 0``.
    """
    if "excluded" not in marked.columns:
        return marked, 0
    excluded = marked["excluded"].astype(bool)
    n_dropped = int(excluded.sum())
    drop_cols = [c for c in ("excluded", "exclusion_reason") if c in marked.columns]
    kept = marked.loc[~excluded].drop(columns=drop_cols).reset_index(drop=True)
    return kept, n_dropped
