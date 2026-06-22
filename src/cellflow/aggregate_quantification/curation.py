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

__all__ = ["CURATION_COLUMNS", "read_curation", "apply_curation", "filter_excluded"]

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
