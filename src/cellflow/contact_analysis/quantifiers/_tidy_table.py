"""Generic tidy-CSV persistence shared by Build quantifiers.

A quantifier owns its artifact format; several persist a column-major table whose
value columns may be strings (``contact_type`` / ``label`` / ``focal_label`` …).
The shape family's ``read_table_csv`` coerces every non-key column to float, which
would destroy those string axes — so these two helpers round-trip a tidy CSV with
dtypes preserved. They are quantity-agnostic (used by the contacts-derived metrics
*and* cell density); nothing here depends on contacts.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from cellflow.contact_analysis.shape.core import write_table_csv


def persist(output_path: Path, table: Mapping[str, np.ndarray]) -> Path:
    """Write *table* (column-major) to a tidy CSV, in declaration order."""
    write_table_csv(Path(output_path), dict(table), tuple(table.keys()))
    return Path(output_path)


def read_derived_table(path: str | Path) -> dict[str, np.ndarray]:
    """Read a tidy CSV back into a column-major dict, preserving dtypes.

    ``frame`` / ``*_id`` columns are ``int64``; object (string) columns stay
    object; everything else keeps pandas' inferred dtype (so NaN survives).
    Mirrors the contract the pooling layer expects from
    :meth:`Quantifier.object_table`.
    """
    frame = pd.read_csv(path)
    out: dict[str, np.ndarray] = {}
    for name in frame.columns:
        col = frame[name]
        if name == "frame" or name.endswith("_id"):
            out[name] = col.to_numpy(dtype=np.int64)
        else:
            # Keep pandas' inferred dtype: object for the string axes, float/int
            # for the numeric values. Forcing float here would corrupt labels.
            out[name] = col.to_numpy()
    return out
