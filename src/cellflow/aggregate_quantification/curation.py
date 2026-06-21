"""The curation artifact — hand QC decisions, joined into the export by ``id``.

The third artifact in the contract (spec §1, §4): a small git-versioned CSV with
columns ``id, excluded, qc_reason`` — *one per experiment series*, authored by a
human in the curation notebook. It is the **curated derived** data and is kept
deliberately separate from the **automatic derived** measurement tables (the
dividing line is who made the decisions in it — see
``~/Projects/electronic_labbook/README.md``).

At export the curation is **left-joined** onto a measurement table by the
deterministic row ``id`` (the same mechanism as the NLS ``class_label`` join),
yielding the *export frame* — never mutating the disposable measurement source.
Rows with no curation entry default to *kept* (``excluded = False``); filter, don't
delete.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["CURATION_COLUMNS", "read_curation", "apply_curation"]

#: The curated columns left-joined onto a table (besides the ``id`` join key).
CURATION_COLUMNS = ("excluded", "qc_reason")


def read_curation(path: Path | str | None) -> pd.DataFrame | None:
    """Read the curation CSV at *path*, or ``None`` when *path* is unset/absent.

    A missing file is not an error: an uncurated series simply has every row kept.
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
    """Return *table* with ``excluded`` / ``qc_reason`` left-joined by ``id``.

    *curation* may be ``None`` (or carry only some of the rows): rows without a
    curation entry default to ``excluded = False`` and an empty ``qc_reason``. Only
    the curated columns are taken from *curation*; stray columns and curation rows
    with no matching measurement ``id`` are dropped. The input frame is not
    mutated.
    """
    out = table.copy()
    if curation is not None and "id" in curation.columns:
        take = ["id", *(c for c in CURATION_COLUMNS if c in curation.columns)]
        out = out.merge(curation[take], on="id", how="left")
    if "excluded" in out.columns:
        out["excluded"] = out["excluded"].fillna(False).astype(bool)
    else:
        out["excluded"] = False
    if "qc_reason" in out.columns:
        out["qc_reason"] = out["qc_reason"].fillna("").astype(str)
    else:
        out["qc_reason"] = ""
    return out
