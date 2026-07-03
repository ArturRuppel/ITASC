"""Aggregated *shape tables* — index-keyed, materialized views of the products.

Contact Analysis builds one per-position artifact per quantity (a tidy
``object_table``). This module pools those, across the in-scope positions, into a
small set of **aggregated, index-keyed tables** — one file per distinct natural
index — written **flat** at ``<out_dir>/<name>.csv``. They are **materialized
views**: regenerate-whole, never upserted, so a re-aggregate rewrites the file from
scratch and CSV stays viable (no concurrent partial write into a shared file). The
per-position artifacts remain the normalized source of truth; these tables are a
reproducible projection of them.

The pooled tables are **label-agnostic**: a per-cell subpopulation classification
(NLS ``class_label`` etc.) is not joined here. A consumer that wants a class split
joins it downstream from the dataset that defines it (keyed on
``experiment_id, position_id, cell_id``).

Partitioning principle: **one table per quantifier.** Each
:class:`~cellflow.contact_analysis.quantifier.Quantifier` that aggregates
declares its natural index (``table_keys``); its tidy ``object_table`` is pooled
across positions into a table named by the quantifier's ``quantity_id``. Value
columns stay namespaced by ``quantity_id`` (``cell_shape.area_um2``) so a later
joined *view* across tables never has colliding names. The keys and the catalogue
metadata (``condition`` / ``experiment_id`` / ``date`` / ``position_id``) stay
bare. (Previously several quantities sharing an index were outer-joined into one
wide table — that produced god tables and is gone.)

Generalizes the old one-quantity-at-a-time pooling
(``napari…plots._pooling.pool_quantity``) to "all quantities of an index,
persisted." Backend-only (no Qt) so headless / batch runs and the standalone
wheel use it unchanged.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .quantifier import Quantifier, available_quantifiers
from .records import output_for_record

__all__ = [
    "ShapeTableSpec",
    "shape_table_registry",
    "table_for_quantity",
    "build_table",
    "read_table",
    "aggregate",
    "table_path",
    "catalogue_root",
    "METADATA_COLUMNS",
]

#: The catalogue-metadata axes stamped (bare) onto every pooled row, in order.
METADATA_COLUMNS = ("condition", "experiment_id", "date", "position_id")


@dataclass(frozen=True)
class ShapeTableSpec:
    """One aggregated table: a single quantifier's pooled ``object_table`` — the
    table name (its ``quantity_id``), the index keys, and the contributing quantity
    (``quantity_ids`` carries the one id, kept a tuple for the pooling machinery)."""

    name: str
    keys: tuple[str, ...]
    quantity_ids: tuple[str, ...]


def shape_table_registry() -> dict[str, ShapeTableSpec]:
    """Map table name → :class:`ShapeTableSpec`. Each aggregating quantifier — one
    that declares a ``table_keys`` index — is its **own** table, named by its
    ``quantity_id`` and keyed by its own grain. No quantifier shares a table with
    another (no god tables)."""
    return {
        q_cls.quantity_id: ShapeTableSpec(
            name=q_cls.quantity_id,
            keys=tuple(q_cls.table_keys),
            quantity_ids=(q_cls.quantity_id,),
        )
        for q_cls in available_quantifiers()
        if q_cls.table_keys
    }


def table_for_quantity(quantity_id: str) -> str | None:
    """The aggregated table *quantity_id* lands in — its own ``quantity_id`` when it
    aggregates (declares ``table_keys``), else ``None`` (contacts and other
    non-aggregated quantities)."""
    for q_cls in available_quantifiers():
        if q_cls.quantity_id == quantity_id:
            return q_cls.quantity_id if q_cls.table_keys else None
    return None


def _quantifiers_for(spec: ShapeTableSpec) -> list[Quantifier]:
    by_id = {q_cls.quantity_id: q_cls for q_cls in available_quantifiers()}
    return [by_id[qid]() for qid in spec.quantity_ids if qid in by_id]


def build_table(name: str, records: Iterable[dict]) -> pd.DataFrame:
    """Pool the quantifier *name*'s built product across *records* into one frame.

    For each in-scope record the quantifier's tidy ``object_table`` is read (value
    columns namespaced by ``quantity_id``) and the catalogue metadata is stamped;
    the per-position frames are then concatenated. The result is label-agnostic (no
    ``class_label`` join). Empty when nothing is built. In-memory only — see
    :func:`aggregate` to persist.
    """
    spec = shape_table_registry().get(name)
    if spec is None:
        raise KeyError(f"No aggregated table named {name!r}")
    quantifiers = _quantifiers_for(spec)
    frames: list[pd.DataFrame] = []
    for record in records:
        merged = _position_frame(record, quantifiers, spec.keys)
        if merged is None:
            continue
        # Insert in reverse so the columns end up condition · experiment_id · date · position_id.
        for key, value in reversed(list(_position_metadata(record).items())):
            merged.insert(0, key, value)
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True)
    _assign_row_id(pooled, spec.keys)
    return pooled


#: Identity axes that prefix every row ``id``, ahead of the table's own grain keys.
#: ``date`` is a descriptor, not identity, so it is excluded.
_ROW_ID_IDENTITY = ("experiment_id", "condition", "position_id")
#: Separator joining the id components; chosen not to occur in the identifier
#: values it concatenates.
_ROW_ID_SEP = "|"


def _assign_row_id(df: pd.DataFrame, keys: tuple[str, ...]) -> None:
    """Insert a deterministic ``id`` first column: the row's identity axes plus the
    table's grain keys, joined as a string. Stable across regeneration (a function
    of identity, not row order) so a row keeps the same identity across rebuilds —
    what Iris keys on, and what any upstream annotation joined by ``id`` relies on."""
    components = [*_ROW_ID_IDENTITY, *(k for k in keys if k not in _ROW_ID_IDENTITY)]
    present = [c for c in components if c in df.columns]
    if not present:
        return
    # Vectorized string join (``str.cat``). The previous ``agg('|'.join, axis=1)``
    # raised "Expected a one-dimensional object" whenever ``present`` named a
    # duplicate column label; ``str.cat`` over explicit single columns is robust.
    row_id = df[present[0]].astype(str)
    for column in present[1:]:
        row_id = row_id.str.cat(df[column].astype(str), sep=_ROW_ID_SEP)
    df.insert(0, "id", row_id)


def _position_frame(
    record: dict, quantifiers: list[Quantifier], keys: tuple[str, ...]
) -> pd.DataFrame | None:
    """Outer-join (within one position) every built co-targeting quantity on the
    table's keys, namespacing each quantity's value columns. ``None`` when no
    targeting quantity is built for this position."""
    merged: pd.DataFrame | None = None
    for quantifier in quantifiers:
        path = output_for_record(quantifier, record)
        if not quantifier.is_built(path):
            continue
        table = quantifier.object_table(path)
        if table is None:
            continue
        df = pd.DataFrame({k: np.asarray(v) for k, v in dict(table).items()})
        present_keys = [k for k in keys if k in df.columns]
        value_cols = [c for c in df.columns if c not in present_keys]
        df = df[[*present_keys, *value_cols]].rename(
            columns={c: f"{quantifier.quantity_id}.{c}" for c in value_cols}
        )
        if merged is None:
            merged = df
        else:
            on = [k for k in keys if k in merged.columns and k in df.columns]
            merged = merged.merge(df, on=on, how="outer") if on else pd.concat(
                [merged, df], ignore_index=True
            )
    return merged


def _position_metadata(record: dict) -> dict[str, str]:
    """The per-position descriptor columns stamped onto every pooled row.

    The recognized identity/descriptor axes (:data:`METADATA_COLUMNS`) come first,
    followed by any extra free-form columns (folder-derived nesting levels or
    manual constant tags) carried in ``record["columns"]``. Extras ride along as
    constant descriptors; they never alter the row-id identity hash."""
    meta = {
        "condition": str(record.get("condition", "")),
        "experiment_id": str(record.get("experiment_id", "")),
        "date": str(record.get("date", "")),
        "position_id": str(record.get("id", "")),
    }
    for key, value in (record.get("columns") or {}).items():
        if key not in meta:  # recognized axes already stamped above
            meta[key] = str(value)
    return meta


def catalogue_root(records: Sequence[dict]) -> Path:
    """A stable home for the aggregated tables: the common ancestor of the
    in-scope positions' folders. Falls back to the first position's parent, then
    to the current directory when no position folder is known."""
    dirs = [Path(r["position_path"]) for r in records if r.get("position_path")]
    if not dirs:
        return Path.cwd()
    if len(dirs) == 1:
        return dirs[0].parent
    try:
        return Path(os.path.commonpath([str(d) for d in dirs]))
    except ValueError:  # paths on different drives (Windows) — no common root
        return dirs[0].parent


def table_path(out_dir: Path | str, name: str) -> Path:
    """Where table *name*'s CSV lives: **flat** under *out_dir*."""
    return Path(out_dir) / f"{name}.csv"


def aggregate(
    records: Sequence[dict], out_dir: Path | str | None = None
) -> dict[str, Path]:
    """Regenerate every aggregated table for the in-scope *records*; return the
    table name → written CSV path map.

    Materialized-view semantics: each table is rebuilt whole and its CSV
    overwritten (a previously-written table for a now-out-of-scope position is
    simply not regenerated with that position). A table that pools to no rows is
    skipped (no empty file). *out_dir* defaults to :func:`catalogue_root`.
    """
    root = Path(out_dir) if out_dir is not None else catalogue_root(records)
    written: dict[str, Path] = {}
    for name in shape_table_registry():
        df = build_table(name, records)
        if df.empty:
            continue
        path = table_path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        written[name] = path
    return written


def read_table(path: Path | str) -> pd.DataFrame:
    """Read an aggregated-table CSV, restoring integer dtype on the key columns
    (``frame`` / ``*_id``) so downstream group-bys match the in-memory build."""
    df = pd.read_csv(path)
    for column in df.columns:
        # The integer keys (``frame`` / per-object ``*_id``); ``position_id`` is a
        # string metadata column, not an integer key, so it is excluded. Only a
        # numeric, fully-populated column is narrowed (an outer-join NaN keeps the
        # column float).
        if (column == "frame" or column.endswith("_id")) and column != "position_id":
            if pd.api.types.is_numeric_dtype(df[column]) and df[column].notna().all():
                df[column] = df[column].astype(np.int64)
    return df
