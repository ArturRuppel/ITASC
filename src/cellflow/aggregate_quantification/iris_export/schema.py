"""Infer an Iris ``data/schema.json`` from a CellFlow tidy table.

Column typing is the load-bearing decision here. ``date`` (and the other spine /
object keys) MUST be typed ``identifier``, not ``categorical``: Iris dodges a
categorical colour into one sub-mark per level — which would split the box into
one box per date — whereas an *identifier* colour with a per-point layer present
colours each point in place and leaves a single box per group. That is the
SuperPlot idiom the analyses rely on (see Iris ``compiler.py`` and
``docs/superpowers/specs/2026-06-17-iris-export-design.md``).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

SCHEMA_VERSION = "1.0"

#: Row keys that index an object / replicate rather than carry a measured value.
#: Typed ``identifier`` so the ``date`` colour drives per-point SuperPlot
#: colouring instead of a categorical dodge.
IDENTIFIER_COLUMNS = frozenset({
    "date", "position_id", "cell_id", "frame", "t1_event_id",
    "label", "focal_label", "neighbor_label", "focal_id", "partner_id",
})
#: Comparison / grouping axes and other genuine categorical factors.
CATEGORICAL_COLUMNS = frozenset({"condition", "class_label", "contact_type", "role"})
#: Iris bookkeeping columns — never emitted to the schema.
META_COLUMNS = frozenset({"id", "excluded"})

#: An unrecognized string column is categorical below this cardinality, else an
#: identifier (free text / high-cardinality keys).
_CATEGORICAL_MAX_CARDINALITY = 50

#: Physical-unit suffixes on a descriptor's leaf name → display unit. Checked in
#: order, so ``_um2`` wins over ``_um``.
_UNIT_SUFFIXES = (("_um2", "µm²"), ("_um", "µm"))


def infer_schema(df: pd.DataFrame) -> dict:
    """Build the ``{schema_version, columns:[...]}`` schema for *df*.

    ``id`` / ``excluded`` are skipped (Iris bookkeeping). Every other column gets
    a ``type`` (``identifier`` | ``categorical`` | ``numeric``), a human ``label``
    (the leaf after the last ``.``), an optional ``unit``, and — for categorical
    columns — the sorted ``levels``.
    """
    columns: list[dict[str, Any]] = []
    for name in df.columns:
        if name in META_COLUMNS:
            continue
        col_type = _column_type(df, name)
        col: dict[str, Any] = {"name": name, "type": col_type, "label": _label(name)}
        unit = _unit(name)
        if unit:
            col["unit"] = unit
        if col_type == "categorical":
            col["levels"] = sorted(str(v) for v in df[name].dropna().unique())
        columns.append(col)
    return {"schema_version": SCHEMA_VERSION, "columns": columns}


def numeric_descriptors(schema: dict) -> list[str]:
    """The numeric value columns of *schema*, in declaration order — the columns
    the SuperPlot template plots on the y axis."""
    return [c["name"] for c in schema["columns"] if c["type"] == "numeric"]


def _column_type(df: pd.DataFrame, name: str) -> str:
    if name in IDENTIFIER_COLUMNS:
        return "identifier"
    if name in CATEGORICAL_COLUMNS:
        return "categorical"
    if pd.api.types.is_numeric_dtype(df[name]):
        return "numeric"
    if int(df[name].nunique(dropna=True)) <= _CATEGORICAL_MAX_CARDINALITY:
        return "categorical"
    return "identifier"


def _label(name: str) -> str:
    return name.split(".")[-1].replace("_", " ")


def _unit(name: str) -> str | None:
    leaf = name.split(".")[-1]
    for suffix, unit in _UNIT_SUFFIXES:
        if leaf.endswith(suffix):
            return unit
    return None
