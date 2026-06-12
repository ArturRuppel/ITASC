"""Shared plumbing for the contacts-*derived* quantifiers.

The neighbor count / enrichment / contact-type z-score / density / energetics
quantities are all computed from a position's already-built
``contact_analysis.h5`` (plus its optional NLS sidecar CSV). They used to be
derived at *plot* time, which made opening a panel re-run a 1000-shuffle null and
re-walk the contact graph for every in-scope position. They are now first-class
Build products: each owns a quantifier that runs the (unchanged) compute function
once and persists a tidy CSV, so plotting is a plain pooled read.

This module factors out what those quantifiers share: loading the contacts
artifact and NLS labels for a position, persisting a column-major table, and
reading it back with **string columns preserved** (the shape family's
``read_table_csv`` coerces every non-key column to float, which would destroy the
``contact_type`` / ``focal_label`` / ``label`` columns these tables carry).
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from cellflow.aggregate_quantification.contacts.nls_classification import (
    nls_classification_csv_path,
    read_nls_classification_csv,
)
from cellflow.aggregate_quantification.contacts.reader import (
    PositionContactAnalysis,
    read_position_contact_analysis,
)
from cellflow.aggregate_quantification.quantifier import PositionInputs
from cellflow.aggregate_quantification.shape.core import write_table_csv


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
    csv_path = nls_classification_csv_path(inputs.position_dir)
    if not csv_path.is_file():
        return None
    labels = read_nls_classification_csv(csv_path)
    return labels or None


def persist(output_path: Path, table: Mapping[str, np.ndarray]) -> Path:
    """Write *table* (column-major) to a tidy CSV, in declaration order."""
    write_table_csv(Path(output_path), dict(table), tuple(table.keys()))
    return Path(output_path)


def read_derived_table(path: str | Path) -> dict[str, np.ndarray]:
    """Read a derived CSV back into a column-major dict, preserving dtypes.

    ``frame`` / ``*_id`` columns are ``int64``; object (string) columns stay
    object; everything else is float (so NaN survives). Mirrors the contract the
    pooling layer expects from :meth:`Quantifier.object_table`.
    """
    frame = pd.read_csv(path)
    out: dict[str, np.ndarray] = {}
    for name in frame.columns:
        col = frame[name]
        if name == "frame" or name.endswith("_id"):
            out[name] = col.to_numpy(dtype=np.int64)
        else:
            # Keep pandas' inferred dtype: object for the string axes
            # (contact_type / focal_label / neighbor_label / label), float/int for
            # the numeric values. Forcing float here would corrupt the labels.
            out[name] = col.to_numpy()
    return out
