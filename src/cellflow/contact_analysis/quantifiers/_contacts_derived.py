"""Shared plumbing for the contacts-*derived* quantifiers.

The neighbor count / density / signed-contact-length quantities are all computed
from a position's already-built ``contact_analysis.h5``. They are **pooled**
quantities: each runs its compute function over the contacts artifact at aggregate
time (in :meth:`Quantifier.compute_object_table`) and the tidy table goes straight
into the pooled output — no per-position artifact is persisted.

This module factors out the one thing those quantifiers share: loading the
contacts artifact for a position. All of it is **label-agnostic**.
"""
from __future__ import annotations

from pathlib import Path

from cellflow.contact_analysis.contacts.reader import (
    PositionContactAnalysis,
    read_position_contacts,
)
from cellflow.contact_analysis.quantifier import PositionInputs

__all__ = [
    "PositionContactAnalysis",
    "load_analysis",
]


def load_analysis(inputs: PositionInputs) -> PositionContactAnalysis:
    """Read the position's contacts artifact, or raise a clear error.

    The derived quantifiers ``require`` ``contact_analysis_path``, so a missing /
    not-yet-built file is a real precondition failure (a position lacking its
    ``contact_analysis.h5`` is scoped out before pooling) rather than a silent
    skip.
    """
    path = inputs.contact_analysis_path
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(
            "contact_analysis.h5 not found — build 'Cell–cell contacts' for this "
            f"position first (looked for {path!r})."
        )
    return read_position_contacts(path)
