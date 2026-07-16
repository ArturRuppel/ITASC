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
:class:`~itasc.contact_analysis.quantifier.Quantifier` that aggregates
declares its natural index (``table_keys``); its tidy ``object_table`` is pooled
across positions into a table named by the quantifier's ``quantity_id``. Value
columns stay namespaced by ``quantity_id`` (``cell_shape.area_um2``) so a later
joined *view* across tables never has colliding names. The keys and the catalogue
metadata (the classification columns — whatever the widget defined) stay
bare. (Previously several quantities sharing an index were outer-joined into one
wide table — that produced god tables and is gone.)

Generalizes the old one-quantity-at-a-time pooling
(``napari…plots._pooling.pool_quantity``) to "all quantities of an index,
persisted." Backend-only (no Qt) so headless / batch runs and the standalone
wheel use it unchanged.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ._provenance import itasc_version as _itasc_version
from .quantifier import Quantifier, available_quantifiers
from .records import available_fields, position_inputs_from_record, record_build_params

__all__ = [
    "ShapeTableSpec",
    "shape_table_registry",
    "table_for_quantity",
    "build_table",
    "read_table",
    "aggregate",
    "table_path",
    "catalogue_root",
    "PROVENANCE_NAME",
]

#: The single run-level provenance sidecar written beside the pooled tables. The
#: cheap quantities are computed in memory and never persisted per-position, so
#: their old per-position ``.provenance.json`` sidecars are gone; this one file
#: records how a whole aggregate run was produced instead.
PROVENANCE_NAME = "provenance.json"


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


def build_table(
    name: str,
    records: Iterable[dict],
    *,
    params: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Pool the quantifier *name*'s per-position table across *records* into one frame.

    For each in-scope record the quantifier's tidy table is computed in memory via
    :meth:`~itasc.contact_analysis.quantifier.Quantifier.compute_object_table`
    (value columns namespaced by ``quantity_id``) and the catalogue metadata is
    stamped; the per-position frames are then concatenated. *params* carries the
    shared build knobs (e.g. ``fov_area_mm2``, ``pixel_size_um``) — the same ones
    :func:`~itasc.contact_analysis.pipeline.build_quantities` threads into
    ``build``, so a param-gated quantifier (cell shape needs ``pixel_size_um``)
    pools consistently with what was actually built. The result is label-agnostic
    (no ``class_label`` join). Empty when nothing computes. In-memory only — see
    :func:`aggregate` to persist.
    """
    spec = shape_table_registry().get(name)
    if spec is None:
        raise KeyError(f"No aggregated table named {name!r}")
    quantifiers = _quantifiers_for(spec)
    frames: list[pd.DataFrame] = []
    identity_cols: list[str] = []
    seen: set[str] = set()
    for record in records:
        merged = _position_frame(record, quantifiers, spec.keys, params)
        if merged is None:
            continue
        meta = _position_metadata(record)
        for key in meta:
            if key not in seen:
                seen.add(key)
                identity_cols.append(key)
        # Insert in reverse so the metadata columns lead the frame in bag order.
        for key, value in reversed(list(meta.items())):
            merged.insert(0, key, value)
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True)
    _assign_row_id(pooled, identity_cols, spec.keys)
    return pooled


#: Separator joining the id components; chosen not to occur in the identifier
#: values it concatenates.
_ROW_ID_SEP = "|"


def _assign_row_id(
    df: pd.DataFrame, identity_cols: Sequence[str], keys: tuple[str, ...]
) -> None:
    """Insert a deterministic ``id`` first column: the row's identity (the
    classification columns) plus the table's grain keys, joined as a string.
    Stable across regeneration (a function of identity, not row order) so a row
    keeps the same identity across rebuilds — what downstream consumers key on,
    and what any upstream annotation joined by ``id`` relies on."""
    components = [*identity_cols, *(k for k in keys if k not in identity_cols)]
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
    record: dict,
    quantifiers: list[Quantifier],
    keys: tuple[str, ...],
    params: Mapping[str, object] | None,
) -> pd.DataFrame | None:
    """Outer-join (within one position) every co-targeting quantity on the table's
    keys, namespacing each quantity's value columns. ``None`` when no targeting
    quantity yields a table for this position."""
    inputs = position_inputs_from_record(record, params)
    merged: pd.DataFrame | None = None
    for quantifier in quantifiers:
        if not set(quantifier.requires) <= available_fields(inputs):
            continue
        if quantifier.missing_build_params(record_build_params(quantifier, record, params)):
            continue
        table = quantifier.compute_object_table(inputs, params=dict(params) if params else None)
        if not table:
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
    """The per-position descriptor columns stamped onto every pooled row — the
    record's classification columns (``record["columns"]``, the widget columns),
    verbatim. Their combination is the position's identity: it forms the row-id
    prefix and is what the aggregator checks for uniqueness."""
    return {key: str(value) for key, value in (record.get("columns") or {}).items()}


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
    records: Sequence[dict],
    out_dir: Path | str | None = None,
    *,
    params: Mapping[str, object] | None = None,
    quantities: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Regenerate every aggregated table for the in-scope *records*; return the
    table name → written CSV path map.

    Materialized-view semantics: each table is rebuilt whole and its CSV
    overwritten (a previously-written table for a now-out-of-scope position is
    simply not regenerated with that position). A table that pools to no rows is
    skipped (no empty file). *out_dir* defaults to :func:`catalogue_root`. *params*
    is forwarded to :func:`build_table` (the shared build knobs the param-gated
    quantifiers need to compute at all).

    *quantities* restricts which tables are written: ``None`` (the default) writes
    every registered table; a sequence writes only those whose ``quantity_id`` it
    names (unknown / non-tabular ids — e.g. the ``contacts`` producer — are simply
    absent from the table registry and ignored). An empty sequence writes nothing.
    """
    _require_unique_identity(records)
    root = Path(out_dir) if out_dir is not None else catalogue_root(records)
    selected = None if quantities is None else set(quantities)
    written: dict[str, Path] = {}
    quant_meta: dict[str, dict[str, object]] = {}
    for name in shape_table_registry():
        if selected is not None and name not in selected:
            continue
        df = build_table(name, records, params=params)
        if df.empty:
            # In scope for this run but pooled to no rows. Still recorded in
            # provenance — uniformly with the tables that did materialize — so
            # every quantifier the seam attempted is auditable, not just the
            # ones that happened to write a file. No empty CSV is written.
            quant_meta[name] = {"rows": 0, "columns": []}
            continue
        path = table_path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        written[name] = path
        quant_meta[name] = {"rows": int(len(df)), "columns": list(df.columns)}
    if written:
        _write_run_provenance(root, records, quant_meta, params)
    return written


def _identity(record: dict) -> tuple[tuple[str, str], ...]:
    """A position's identity: its classification columns as a name-sorted tuple of
    ``(column, value)`` pairs. Two positions with equal identities would pool their
    cells under one row-id and silently merge — the collision the aggregator refuses."""
    columns = record.get("columns") or {}
    return tuple(sorted((str(k), str(v)) for k, v in columns.items()))


def _require_unique_identity(records: Sequence[dict]) -> None:
    """Refuse to aggregate when the classification columns do not uniquely identify
    the in-scope positions.

    The pooled row-id is the combination of a position's columns plus the grain
    keys; if two positions share every column value, their cells collapse onto the
    same identities and merge silently. Rather than corrupt the pooled table, stop
    and explain which positions collide and that a distinguishing column is needed.
    """
    groups: dict[tuple[tuple[str, str], ...], list[dict]] = {}
    for record in records:
        groups.setdefault(_identity(record), []).append(record)
    collisions = {ident: recs for ident, recs in groups.items() if len(recs) > 1}
    if not collisions:
        return
    blocks = []
    for ident, recs in collisions.items():
        shown = ", ".join(f"{name}={value!r}" for name, value in ident) or "(no columns)"
        paths = "\n".join(f"    {r.get('position_path', '?')}" for r in recs)
        blocks.append(f"  {shown}\n{paths}")
    detail = "\n".join(blocks)
    raise ValueError(
        "Cannot aggregate: these positions are not uniquely identified by their "
        "columns. Positions sharing identical column values would pool their cells "
        "under one identity and silently merge. Add or edit a column that "
        f"distinguishes them:\n{detail}"
    )


def _write_run_provenance(
    root: Path,
    records: Sequence[dict],
    quant_meta: Mapping[str, dict[str, object]],
    params: Mapping[str, object] | None,
) -> Path:
    """Write the run-level ``provenance.json`` beside the pooled tables.

    One file per aggregate run (not per position — the cheap quantities are
    computed in memory, so there are no per-position artifacts to sidecar). It
    records the shared build params, the contributing positions (identity +
    their source paths), and — under ``quantifiers`` — **every** in-scope pooled
    quantifier the run attempted, each with its pooled row count and columns
    (``rows: 0`` / empty ``columns`` for one that yielded nothing and so wrote no
    CSV). Provenance is thus a property of the quantifier seam, uniform across
    quantifiers, rather than a record of only the tables that materialized. Plus
    the itasc version and a UTC timestamp — enough to reconstruct how the
    tables were produced.
    """
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "itasc_version": _itasc_version(),
        "params": dict(params or {}),
        "positions": [_provenance_position(r) for r in records],
        "quantifiers": {name: dict(meta) for name, meta in quant_meta.items()},
    }
    path = Path(root) / PROVENANCE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return path


def _provenance_position(record: dict) -> dict[str, str]:
    """Identity + source paths for one contributing position, for provenance.

    The identity is the position's classification columns (verbatim), alongside its
    source paths — enough to trace any pooled row back to the folder it came from."""
    prov = {
        "position_path": str(record.get("position_path", "")),
        "contact_analysis_path": str(record.get("contact_analysis_path", "")),
    }
    prov.update({key: str(value) for key, value in (record.get("columns") or {}).items()})
    return prov


def read_table(path: Path | str) -> pd.DataFrame:
    """Read an aggregated-table CSV, restoring integer dtype on the key columns
    (``frame`` / ``*_id``) so downstream group-bys match the in-memory build."""
    df = pd.read_csv(path)
    for column in df.columns:
        # The integer grain keys (``frame`` / per-object ``*_id``). The row-id
        # column is named exactly ``id`` (never matches ``*_id``) and stays a
        # string; classification columns are strings and only narrow when they
        # happen to be fully numeric — harmless for a descriptor. Only a numeric,
        # fully-populated column is narrowed (an outer-join NaN keeps it float).
        if column == "frame" or column.endswith("_id"):
            if pd.api.types.is_numeric_dtype(df[column]) and df[column].notna().all():
                df[column] = df[column].astype(np.int64)
    return df
