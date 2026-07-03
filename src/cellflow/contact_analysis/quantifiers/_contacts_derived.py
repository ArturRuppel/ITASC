"""Shared plumbing for the contacts-*derived* quantifiers.

The neighbor count / density / signed-contact-length quantities are all computed
from a position's already-built ``contact_analysis.h5``. They used to be derived at
*plot* time, which made opening a panel re-walk the contact graph for every
in-scope position. They are now first-class Build products: each owns a quantifier
that runs the (unchanged) compute function once and persists a tidy CSV, so
plotting is a plain pooled read.

This module factors out what those quantifiers share: loading the contacts
artifact for a position. Tidy-CSV persistence is quantity-agnostic and lives in
:mod:`._tidy_table`; it is re-exported here so the derived quantifiers can keep
calling ``derived.persist`` / ``derived.read_derived_table``. All of it is
**label-agnostic**.
"""
from __future__ import annotations

from pathlib import Path

from cellflow.contact_analysis.contacts.reader import (
    PositionContactAnalysis,
    read_position_contacts,
)
from cellflow.contact_analysis.quantifier import PositionInputs
from cellflow.contact_analysis.quantifiers._tidy_table import (
    persist,
    read_derived_table,
)

__all__ = [
    "PositionContactAnalysis",
    "load_analysis",
    "persist",
    "read_derived_table",
]


def load_analysis(inputs: PositionInputs) -> PositionContactAnalysis:
    """Read the position's contacts artifact, or raise a clear error.

    The derived quantifiers ``require`` ``contact_analysis_path``, so a missing /
    not-yet-built file is a real precondition failure (surfaced per-position by
    the studio build loop) rather than a silent skip.
    """
    path = inputs.contact_analysis_path
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(
            "contact_analysis.h5 not found — build 'Cell–cell contacts' for this "
            f"position first (looked for {path!r})."
        )
    return read_position_contacts(path)
