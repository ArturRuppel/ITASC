"""Shared plumbing for the contacts-*derived* quantifiers.

The neighbor count / enrichment / contact-type z-score / density / energetics
quantities are all computed from a position's already-built
``contact_analysis.h5`` (plus its optional NLS sidecar CSV). They used to be
derived at *plot* time, which made opening a panel re-run a 1000-shuffle null and
re-walk the contact graph for every in-scope position. They are now first-class
Build products: each owns a quantifier that runs the (unchanged) compute function
once and persists a tidy CSV, so plotting is a plain pooled read.

This module factors out what those quantifiers share: loading the contacts
artifact and NLS labels for a position. Tidy-CSV persistence is quantity-agnostic
and lives in :mod:`._tidy_table`; it is re-exported here so the derived
quantifiers can keep calling ``derived.persist`` / ``derived.read_derived_table``.
"""
from __future__ import annotations

from pathlib import Path

from cellflow.aggregate_quantification.contacts.nls_classification import (
    nls_classification_csv_path,
    read_nls_classification_csv,
)
from cellflow.aggregate_quantification.contacts.reader import (
    PositionContactAnalysis,
    read_position_contact_analysis,
)
from cellflow.aggregate_quantification.quantifier import PositionInputs
from cellflow.aggregate_quantification.quantifiers._tidy_table import (
    persist,
    read_derived_table,
)

__all__ = [
    "PositionContactAnalysis",
    "load_analysis",
    "load_labels",
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
    return read_position_contact_analysis(path)


def load_labels(inputs: PositionInputs) -> dict[int, str] | None:
    """The position's ``cell_id -> NLS label`` map, or ``None`` when unclassified."""
    return nls_labels_for_position(inputs.position_dir)


def nls_labels_for_position(position_dir: Path) -> dict[int, str] | None:
    """The ``cell_id -> NLS label`` map for *position_dir*, or ``None`` when
    unclassified. Reads the NLS classification sidecar CSV; the sidecar is a
    position artifact (not a contacts artifact), so cell density can reuse it for
    its optional per-class breakdown without taking a contacts dependency."""
    csv_path = nls_classification_csv_path(position_dir)
    if not csv_path.is_file():
        return None
    labels = read_nls_classification_csv(csv_path)
    return labels or None
