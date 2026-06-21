"""The napari-free orchestration surface for Aggregate Quantification.

Four composable functions thread the existing headless stages — discovery,
per-position build, aggregate, export — into one pipeline the CLI, notebooks, and
(during the napari parallel-run) the Qt studio all drive:

    catalog = build_catalog(root, cell_name=..., nucleus_name=..., out_csv=...)
    build_quantities(catalog)                 # one .build() per (quantifier, position)
    tables = aggregate(catalog)               # pooled, index-keyed CSVs
    export(tables["cell_shape"].parent)       # tidy artifacts + .iris bundles

This module *composes* — it owns no compute. Discovery lives in :mod:`.catalog`,
the per-position units in :mod:`.quantifier`, the record→inputs bridge in
:mod:`.records`, pooling in :mod:`.shape_tables`, and the ``.iris`` writer in
:mod:`.iris_export`. The only orchestration that previously lived nowhere but the
napari studio — the per-position **build loop** — is :func:`build_quantities`.

Backend-only (no Qt / napari): the standalone wheel and headless batch runs use
it unchanged.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import fields as _dataclass_fields
from pathlib import Path

import pandas as pd

from . import shape_tables
from .catalog import discover_catalog_entries, save_catalog
from .iris_export.export import export_dir as _export_iris
from .quantifier import PositionInputs, Quantifier, available_quantifiers
from .records import output_for_record, position_inputs_from_record

__all__ = [
    "build_catalog",
    "build_quantities",
    "aggregate",
    "export",
]

#: Tidy-table artifact formats :func:`export` knows how to write. ``csv`` copies
#: the aggregated table through unchanged; ``parquet`` re-encodes it.
_KNOWN_FORMATS = ("parquet", "csv")


def build_catalog(
    root: Path | str,
    *,
    cell_name: str | None = None,
    nucleus_name: str | None = None,
    out_csv: Path | str | None = None,
) -> list[dict]:
    """Discover the positions under *root* and return them as catalogue records.

    Wraps :func:`~cellflow.aggregate_quantification.catalog.discover_catalog_entries`:
    a *position* is any folder holding at least one named input (cell and/or
    nucleus labels). When *out_csv* is given, the discovered skeleton is written
    there (:func:`~cellflow.aggregate_quantification.catalog.save_catalog`) for an
    analyst to fill in ``condition`` / ``date`` / ``notes`` before building.

    The records carry the discovered paths but no metadata; that is fine for an
    immediate build (which keys only on the inputs) and is the editable skeleton
    when persisted.
    """
    entries = discover_catalog_entries(
        root, cell_name=cell_name, nucleus_name=nucleus_name
    )
    if out_csv is not None:
        save_catalog(out_csv, entries)
    return entries


def build_quantities(
    catalog: Iterable[dict],
    *,
    quantifiers: Sequence[Quantifier] | None = None,
    params: Mapping[str, object] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> None:
    """Build every applicable quantity for every position in *catalog*.

    THE extracted build loop (previously trapped in the napari studio): one job
    per ``(quantifier, position)`` where the quantifier
    :meth:`~cellflow.aggregate_quantification.quantifier.Quantifier.can_build` the
    position's inputs; each job calls ``.build(inputs, output_path, params=...)``,
    overwriting an existing artifact. Positions lacking a quantifier's inputs are
    skipped.

    *quantifiers* defaults to one instance of every registered quantifier
    (:func:`~cellflow.aggregate_quantification.quantifier.available_quantifiers`).
    *params* (the shared build knobs — z-score shuffles, density FOV, pixel size,
    frame interval) is threaded only into quantifiers that opt in via
    ``wants_build_params``; the rest keep their own ``params`` schema clean. A
    quantifier whose ``required_build_params`` *params* does not satisfy is skipped
    whole (mirrors the studio greying the metric out), so "build everything" stays
    usable when an optional knob like the density FOV is unset.
    *progress_cb* is called ``(done, total, position_name)`` before each build.

    Quantifiers run in **dependency order** (:func:`_dependency_order`): a producer
    like contacts builds before the derived quantities whose ``requires`` names its
    ``produces``. Planning tracks, per position, which ``PositionInputs`` fields a
    producer *will* make available, so a derived quantity is scheduled (not silently
    skipped) even though its input is unbuilt at the start of the run; ``total`` is
    therefore known up front. At build time each position's inputs are re-derived so
    a dependent sees the artifact its producer just wrote. Exceptions propagate — a
    failed build aborts the run rather than being silently swallowed.
    """
    quants = (
        list(quantifiers)
        if quantifiers is not None
        else [cls() for cls in available_quantifiers()]
    )
    quants = _dependency_order(quants)
    records = list(catalog)
    # Per position, the PositionInputs fields available so far. A producer planned
    # below grows this set for the dependents planned after it.
    available = [_available_fields(position_inputs_from_record(r)) for r in records]

    jobs: list[tuple[Quantifier, dict, dict | None]] = []
    for quantifier in quants:
        # A metric missing a required shared param (e.g. density's FOV) is greyed
        # out in the studio; here we skip it whole rather than fail its builds.
        if quantifier.missing_build_params(params):
            continue
        # Only quantifiers that opt in get the shared bar's knobs; the rest keep
        # their own ``params`` schema clean (mirrors the studio's build planning).
        q_params = dict(params) if (params and quantifier.wants_build_params) else None
        for record, fields in zip(records, available):
            if not set(quantifier.requires) <= fields:
                continue
            jobs.append((quantifier, record, q_params))
            produced = _produced_field_for(quantifier, record)
            if produced is not None:
                fields.add(produced)

    total = len(jobs)
    for index, (quantifier, record, q_params) in enumerate(jobs, start=1):
        # Re-derive now: a producer built earlier this run has written its
        # artifact, so the dependent's inputs resolve it (``position_inputs_from_record``
        # gates a produced input on the file existing).
        inputs = position_inputs_from_record(record)
        if progress_cb is not None:
            progress_cb(index, total, inputs.position_dir.name)
        quantifier.build(inputs, output_for_record(quantifier, record), params=q_params)


def _dependency_order(quants: Sequence[Quantifier]) -> list[Quantifier]:
    """*quants* sorted so each producer precedes the quantifiers whose ``requires``
    names its ``produces``. Independent quantifiers keep their given order; a
    dependency cycle raises ``ValueError``."""
    produced_by = {q.produces: q for q in quants if q.produces}
    order: list[Quantifier] = []
    state: dict[int, str] = {}  # id(q) -> "visiting" | "done"

    def visit(q: Quantifier) -> None:
        mark = state.get(id(q))
        if mark == "done":
            return
        if mark == "visiting":
            name = q.quantity_id or type(q).__name__
            raise ValueError(f"Quantifier dependency cycle involving {name!r}")
        state[id(q)] = "visiting"
        for field in q.requires:
            producer = produced_by.get(field)
            if producer is not None and producer is not q:
                visit(producer)
        state[id(q)] = "done"
        order.append(q)

    for q in quants:
        visit(q)
    return order


def _available_fields(inputs: PositionInputs) -> set[str]:
    """The populated (non-``None``) ``PositionInputs`` field names — the satisfied
    prerequisites a quantifier's ``requires`` is checked against."""
    return {f.name for f in _dataclass_fields(inputs) if getattr(inputs, f.name) is not None}


def _produced_field_for(quantifier: Quantifier, record: dict) -> str | None:
    """The ``PositionInputs`` field *quantifier*'s build makes available to
    dependents for *record*, or ``None`` when its output will not be surfaced as an
    input. Mirrors :func:`position_inputs_from_record`: a produced field is only
    visible when the record points its same-named input path at the built artifact
    (so planning predicts exactly what re-derivation will resolve at build time)."""
    if not quantifier.produces:
        return None
    read_path = record.get(quantifier.produces)
    if read_path and Path(read_path) == output_for_record(quantifier, record):
        return quantifier.produces
    return None


def aggregate(
    catalog: Sequence[dict], out_dir: Path | str | None = None
) -> dict[str, Path]:
    """Pool every built product across *catalog* into the index-keyed tables.

    Thin pass-through to
    :func:`cellflow.aggregate_quantification.shape_tables.aggregate`: returns the
    table name → written CSV path map. *out_dir* defaults to the catalogue root
    (the common ancestor of the positions). The tables land under
    ``<out_dir>/aggregate_quantification/<name>.csv``.
    """
    return shape_tables.aggregate(catalog, out_dir)


def export(
    tables_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    formats: Sequence[str] = _KNOWN_FORMATS,
) -> list[Path]:
    """Write tidy artifacts + ``.iris`` bundles from the aggregated tables.

    *tables_dir* is the directory holding the aggregated tidy CSVs (what
    :func:`aggregate` wrote into — the ``aggregate_quantification`` folder). For
    each table CSV found there, a copy is emitted in every requested *formats*
    entry (``csv`` passes through, ``parquet`` re-encodes); then
    :func:`cellflow.aggregate_quantification.iris_export.export_dir` writes one
    ``.iris`` document per curated table. *out_dir* defaults to *tables_dir*; the
    ``.iris`` bundles land in ``<out_dir>/iris/``.

    Returns every written path (tidy artifacts then ``.iris`` bundles).
    """
    tables_dir = Path(tables_dir)
    out_dir = Path(out_dir) if out_dir is not None else tables_dir
    unknown = [fmt for fmt in formats if fmt not in _KNOWN_FORMATS]
    if unknown:
        raise ValueError(
            f"unknown export format(s) {unknown}; known: {list(_KNOWN_FORMATS)}"
        )

    written: list[Path] = []
    for csv_path in sorted(tables_dir.glob("*.csv")):
        frame: pd.DataFrame | None = None
        for fmt in formats:
            if fmt == "csv":
                target = out_dir / csv_path.name
                if target.resolve() == csv_path.resolve():
                    continue  # source already in place; no self-copy
                out_dir.mkdir(parents=True, exist_ok=True)
                target.write_bytes(csv_path.read_bytes())
                written.append(target)
            elif fmt == "parquet":
                if frame is None:
                    frame = pd.read_csv(csv_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir / f"{csv_path.stem}.parquet"
                frame.to_parquet(target, index=False)
                written.append(target)

    written.extend(_export_iris(tables_dir, out_dir=out_dir / "iris"))
    return written
