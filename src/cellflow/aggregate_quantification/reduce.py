"""The reduce layer — a transparent, composable table→table pipeline.

A seam between the aggregated-table loader and the plotting backend
(:mod:`.plotting`): table in, table out, pure and in-memory (nothing persisted).
The layer is **transparent composable primitives — no hidden cleverness; the user
owns correctness.**

A reduce spec is an **ordered pipeline of steps**, each one of two primitives,
freely orderable and repeatable:

* :class:`Filter` — keep rows matching ``column op value``.
* :class:`Collapse` — plain group-by: aggregate the value columns to one row per
  ``by``-combination.

An empty pipeline is the identity ("no reduction").

**Chaining expresses the statistics.** The pseudoreplication-safe nested
reduction is just chained single-rung collapses, composed explicitly by the user:

* ``Collapse(by=[…, "cell_id"])`` then ``Collapse(by=[…, "position_id"])`` → the
  equal-weighted per-position result (each cell counts once within its position,
  each position once within its condition);
* a single ``Collapse(by=[…, "position_id"])`` → the flat pooled result (rows
  weighted by frame/cell count).

The two diverge **only** when a collapse spans an intermediate level with unequal
child counts; for a single rung (frames → cell) flat and nested are identical.
Nothing here decides this for the user — the pipeline order is the knob.

This module is backend-only (no Qt / pandas-display state) so the standalone
``cellflow-aggregate`` wheel and headless / batch runs use it unchanged.
"""
from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "Filter",
    "Collapse",
    "Step",
    "ReduceSpec",
    "run_pipeline",
    "FILTER_OPS",
    "COLLAPSE_STATS",
    "IDENTITY_COLUMNS",
    "unit_collapse_chain",
]

#: Comparison operators a :class:`Filter` understands, mapped to a (left, right)
#: predicate. ``==`` / ``!=`` compare as strings so a categorical axis
#: (``class_label``, ``contact_type``) and a numeric one behave alike; the
#: ordered operators coerce both sides to numbers.
FILTER_OPS: dict[str, str] = {
    "==": "equal",
    "!=": "not equal",
    ">": "greater than",
    ">=": "at least",
    "<": "less than",
    "<=": "at most",
}

#: The reductions a :class:`Collapse` offers. ``mean`` / ``median`` aggregate the
#: numeric value columns; ``count`` reports the group size in an ``n`` column.
COLLAPSE_STATS: tuple[str, ...] = ("mean", "median", "count")

#: Columns that index a row rather than carry a measured value. On a collapse,
#: an identity column **not** in ``by`` is dropped (it is being collapsed away),
#: never averaged — averaging a ``cell_id`` or a ``frame`` is meaningless. The
#: catalogue-metadata axes, the per-object / per-event keys, and the curve abscissa
#: all count as identity; everything else is a value column.
IDENTITY_COLUMNS: frozenset[str] = frozenset(
    {
        # catalogue metadata
        "condition",
        "experiment_id",
        "date",
        "position_id",
        # per-object / per-frame keys
        "frame",
        "cell_id",
        "label",
        # per-edge / per-event keys
        "t1_event_id",
        "role",
        "contact_type",
        "focal_id",
        "partner_id",
        "focal_label",
        "neighbor_label",
        # curve abscissa
        "lag_s",
    }
)


@dataclass(frozen=True)
class Filter:
    """Keep rows where ``column op value``.

    ``value`` is compared as text for ``==`` / ``!=`` (so ``class_label ==
    "epithelial"`` and ``frame == 0`` both work) and as a number for the ordered
    operators (``frame >= 10``). A filter on a column the table lacks is a no-op,
    so a pipeline carried over from another table never raises — it simply does
    not narrow.
    """

    column: str
    op: str
    value: Any

    def __post_init__(self) -> None:
        if self.op not in FILTER_OPS:
            raise ValueError(f"op must be one of {tuple(FILTER_OPS)}, got {self.op!r}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or self.column not in df.columns:
            return df
        column = df[self.column]
        if self.op in ("==", "!="):
            mask = column.astype(str) == str(self.value)
            mask = mask if self.op == "==" else ~mask
            return df[mask]
        # Ordered comparisons are numeric on both sides; non-numeric cells become
        # NaN and so never satisfy a ``<``/``>`` test (dropped).
        left = pd.to_numeric(column, errors="coerce")
        if left.notna().sum() == 0:
            # A fully non-numeric column under an ordered op is a config error;
            # keep every row (no-op) rather than silently emptying the table.
            return df
        try:
            right = float(self.value)
        except (TypeError, ValueError):
            return df.iloc[0:0]
        if self.op == ">":
            mask = left > right
        elif self.op == ">=":
            mask = left >= right
        elif self.op == "<":
            mask = left < right
        else:  # "<="
            mask = left <= right
        return df[mask.fillna(False)]


@dataclass(frozen=True)
class Collapse:
    """Plain group-by: one row per ``by``-combination.

    ``mean`` / ``median`` aggregate every numeric **value** column (an identity
    column outside ``by`` is dropped — it is being collapsed away, never
    averaged). A non-numeric attribute outside ``by`` is **kept if constant**
    within each group and **dropped if it varies** — to preserve a varying
    attribute, add it to ``by``. **Every** collapse attaches the current group
    size as an ``n`` column (whole-table collapse → ``n = len(df)``), so an
    ``n``-threshold filter (drop undersampled units) works after any collapse, not
    only after ``count``. An empty ``by`` collapses the whole table to a single
    row.

    ``n`` is **reserved**: each collapse recomputes it to the *current* group size
    and never treats a pre-existing ``n`` as a value column to average — chaining
    ``collapse by cell`` then ``collapse by position`` yields per-position child
    counts, not the mean of the per-cell counts.
    """

    by: tuple[str, ...]
    stat: str = "mean"

    def __post_init__(self) -> None:
        if self.stat not in COLLAPSE_STATS:
            raise ValueError(
                f"stat must be one of {COLLAPSE_STATS}, got {self.stat!r}"
            )
        # Normalize ``by`` to a tuple so the dataclass stays hashable / comparable
        # whether built from a list or a tuple.
        object.__setattr__(self, "by", tuple(self.by))

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        by = [c for c in self.by if c in df.columns]
        # ``n`` is reserved for the group size — excluded here so it is never
        # carried through as a value/attribute and never averaged; it is always
        # recomputed below to the current group size.
        others = [c for c in df.columns if c not in by and c != "n"]

        if self.stat == "count":
            # Plain tally per group; attributes constant within a group ride along.
            grouped = df.groupby(by, dropna=False, sort=False) if by else _whole(df)
            out = grouped.size().reset_index(name="n") if by else _one_row({"n": len(df)})
            kept = _constant_attributes(df, by, others)
            return out.merge(kept, on=by, how="left") if (by and not kept.empty) else out

        agg = "median" if self.stat == "median" else "mean"
        numeric_values = [
            c for c in others if c not in IDENTITY_COLUMNS and _is_numeric(df[c])
        ]
        attributes = [c for c in others if c not in IDENTITY_COLUMNS and c not in numeric_values]

        if not by:
            row: dict[str, Any] = {}
            for c in numeric_values:
                values = pd.to_numeric(df[c], errors="coerce")
                row[c] = float(getattr(values, agg)()) if values.notna().any() else float("nan")
            row["n"] = len(df)
            base = _one_row(row)
            kept = _constant_attributes(df, by, attributes)
            return pd.concat([base, kept], axis=1) if not kept.empty else base

        grouped = df.groupby(by, dropna=False, sort=False)
        out = grouped[numeric_values].agg(agg).reset_index() if numeric_values else (
            df[by].drop_duplicates().reset_index(drop=True)
        )
        out = out.merge(grouped.size().reset_index(name="n"), on=by, how="left")
        kept = _constant_attributes(df, by, attributes)
        if not kept.empty:
            out = out.merge(kept, on=by, how="left")
        return out


#: A single pipeline step.
Step = Filter | Collapse
#: An ordered, freely-repeatable pipeline of steps. Empty ⇒ identity.
ReduceSpec = Sequence[Step]


def run_pipeline(df: pd.DataFrame, spec: ReduceSpec) -> pd.DataFrame:
    """Apply *spec*'s steps to *df* in order, returning the reduced table.

    Pure and in-memory; an empty *spec* returns *df* unchanged (identity). Order
    is honoured — ``filter`` then ``collapse`` narrows before aggregating, the
    reverse aggregates first.
    """
    out = df
    for step in spec:
        out = step.apply(out)
    return out


def unit_collapse_chain(
    present_nesting: Sequence[str],
    group: Iterable[str],
    level: str,
    stat: str,
) -> tuple[Collapse, ...]:
    """The chained-collapse equivalent of ``plotting.reduce_to_units``.

    ``present_nesting`` is the biological nesting present in the table, coarse→fine
    (a prefix of ``("date", "position_id", "cell_id")``); ``group`` the comparison
    group-by; ``level`` the independent unit (``"cell"`` | ``"position"`` |
    ``"date"``). Returns one :class:`Collapse` per rung: the first collapses the
    frame axis (group by group + every nesting key), each later one drops the
    finest nesting key and climbs one level — equal-weighting each parent's
    children, exactly as the level machinery does. Lets the explicit reduce
    pipeline reproduce the level convenience as composed steps.
    """
    entity = {"cell": "cell_id", "position": "position_id", "date": "date"}[level]
    present = [k for k in present_nesting if k in ("date", "position_id", "cell_id")]
    group = list(group)
    if not present:
        return ()
    unit = present[: present.index(entity) + 1] if entity in present else present
    chain: list[Collapse] = []
    levels = list(present)
    while True:
        chain.append(Collapse(by=tuple(dict.fromkeys([*group, *levels])), stat=stat))
        if len(levels) <= len(unit):
            return tuple(chain)
        levels = levels[:-1]


def _is_numeric(series: pd.Series) -> bool:
    """True only when the column is genuinely numeric.

    Requires *every* non-null value to parse as a number, so a categorical
    column that merely contains a stray parseable value (e.g. a grade ``"5"``
    among labels) is not mistaken for a value column and silently averaged.
    """
    if pd.api.types.is_numeric_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    return bool(pd.to_numeric(non_null, errors="coerce").notna().all())


def _constant_attributes(
    df: pd.DataFrame, by: list[str], attributes: Collection[str]
) -> pd.DataFrame:
    """One row per group carrying each *attribute* that is constant within every
    group (varying ones are dropped). Empty (no columns) when none qualify."""
    attributes = [c for c in attributes if c in df.columns]
    if not attributes:
        return pd.DataFrame()
    if not by:
        keep = {c: df[c].iloc[0] for c in attributes if df[c].nunique(dropna=False) <= 1}
        return _one_row(keep) if keep else pd.DataFrame()
    grouped = df.groupby(by, dropna=False, sort=False)
    constant = [c for c in attributes if (grouped[c].nunique(dropna=False) <= 1).all()]
    if not constant:
        return pd.DataFrame()
    return grouped[constant].first().reset_index()


def _whole(df: pd.DataFrame):
    return df.groupby(np.zeros(len(df), dtype=int), sort=False)


def _one_row(data: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([data]) if data else pd.DataFrame(index=[0])
