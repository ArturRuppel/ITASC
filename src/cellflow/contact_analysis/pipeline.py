"""The napari-free orchestration surface for Contact Analysis.

Three composable functions thread the existing headless stages — discovery,
per-position build, aggregate — into one pipeline the CLI, notebooks, and (during
the napari parallel-run) the Qt studio all drive:

    catalog = build_catalog(root, cell_name=..., nucleus_name=..., out_csv=...)
    build_quantities(catalog)                 # one .build() per (quantifier, position)
    tables = aggregate(catalog, out_dir)      # pooled, index-keyed CSVs (flat)

Or, end to end from a TOML run-config: ``run("config.toml")``.

Everything produced is **label-agnostic** tidy CSVs: there is no classification
step and no plot/figure export — a subpopulation classification and any plots are a
downstream, dataset-specific concern, computed from these tables.

This module *composes* — it owns no compute. Discovery lives in :mod:`.catalog`,
the per-position units in :mod:`.quantifier`, the record→inputs bridge in
:mod:`.records`, and pooling in :mod:`.shape_tables`. The only orchestration that
previously lived nowhere but the napari studio — the per-position **build loop** —
is :func:`build_quantities`.

Backend-only (no Qt / napari): the standalone wheel and headless batch runs use
it unchanged.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from . import shape_tables
from .catalog import discover_catalog_entries, load_catalog, save_catalog
from .config import RunConfig, load_config, write_config
from .quantifier import Quantifier, available_quantifiers
from .records import (
    available_fields,
    output_for_record,
    position_inputs_from_record,
    record_build_params,
)

__all__ = [
    "author_config",
    "build_catalog",
    "build_quantities",
    "select_quantifiers",
    "aggregate",
    "run",
]


def build_catalog(
    root: Path | str,
    *,
    cell_name: str | None = None,
    nucleus_name: str | None = None,
    out_csv: Path | str | None = None,
) -> list[dict]:
    """Discover the positions under *root* and return them as catalogue records.

    Wraps :func:`~cellflow.contact_analysis.catalog.discover_catalog_entries`:
    a *position* is any folder holding at least one named input (cell and/or
    nucleus labels). When *out_csv* is given, the discovered skeleton is written
    there (:func:`~cellflow.contact_analysis.catalog.save_catalog`) for an
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
    """Build and persist every **producer** quantity for every position in *catalog*.

    THE extracted build loop (previously trapped in the napari studio): one job
    per ``(quantifier, position)`` where the quantifier
    :meth:`~cellflow.contact_analysis.quantifier.Quantifier.can_build` the
    position's inputs; each job calls ``.build(inputs, output_path, params=...)``,
    overwriting an existing artifact. Positions lacking a quantifier's inputs are
    skipped.

    Only **producers** — quantifiers whose ``produces`` another quantifier's
    ``requires`` names (in practice just ``contacts``, writing
    ``contact_analysis.h5``) — persist a per-position artifact here. Every other
    (cheap) quantity is never built to disk per-position: it is instead computed
    in memory, straight from a position's raw inputs, inside
    :func:`~cellflow.contact_analysis.shape_tables.aggregate` /
    :func:`~cellflow.contact_analysis.shape_tables.build_table` via
    :meth:`~cellflow.contact_analysis.quantifier.Quantifier.compute_object_table`.
    A normal run therefore writes only ``contact_analysis.h5`` per position; the
    pooled tables are the only place the cheap quantities live.

    *quantifiers* defaults to one instance of every registered quantifier
    (:func:`~cellflow.contact_analysis.quantifier.available_quantifiers`); it is
    filtered down to producers before planning, so passing non-producers here is
    harmless (they are simply skipped).
    *params* (the shared build knobs — z-score shuffles, density FOV, pixel size,
    frame interval) is threaded only into quantifiers that opt in via
    ``wants_build_params``; the rest keep their own ``params`` schema clean. A
    quantifier whose ``required_build_params`` *params* does not satisfy is skipped
    whole (mirrors the studio greying the metric out), so "build everything" stays
    usable when an optional knob like the density FOV is unset.
    *progress_cb* is called ``(done, total, position_name)`` before each build.
    Exceptions propagate — a failed build aborts the run rather than being
    silently swallowed.
    """
    quants = (
        list(quantifiers)
        if quantifiers is not None
        else [cls() for cls in available_quantifiers()]
    )
    quants = _dependency_order(quants)
    quants = [q for q in quants if q.produces]  # only producers persist (contacts)
    records = list(catalog)
    jobs: list[tuple[Quantifier, dict, dict | None]] = []
    for quantifier in quants:
        q_params = dict(params) if (params and quantifier.wants_build_params) else None
        for record in records:
            inputs = position_inputs_from_record(record)
            if not set(quantifier.requires) <= available_fields(inputs):
                continue
            if quantifier.missing_build_params(record_build_params(quantifier, record, params)):
                continue
            jobs.append((quantifier, record, q_params))

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


def select_quantifiers(quantities: Sequence[str]) -> list[Quantifier]:
    """Instantiate the quantifiers a run should build for the selected *quantities*.

    Empty *quantities* selects **every** registered quantifier. A non-empty list
    selects those ``quantity_id``\\ s plus, transitively, any **producer** whose
    ``produces`` field a selected (or pulled-in) quantifier ``requires`` — so asking
    for a contacts-derived metric silently brings contacts along instead of leaving
    it unbuildable. Order follows registration; the build loop re-sorts by
    dependency.
    """
    classes = list(available_quantifiers())
    if not quantities:
        return [cls() for cls in classes]

    by_id = {cls.quantity_id: cls for cls in classes}
    unknown = [q for q in quantities if q not in by_id]
    if unknown:
        raise ValueError(f"unknown quantit(y/ies) {unknown}; known: {sorted(by_id)}")
    produced_by = {cls.produces: cls for cls in classes if cls.produces}

    selected: dict[str, type[Quantifier]] = {}

    def add(cls: type[Quantifier]) -> None:
        if cls.quantity_id in selected:
            return
        selected[cls.quantity_id] = cls
        for field in cls.requires:
            producer = produced_by.get(field)
            if producer is not None and producer is not cls:
                add(producer)

    for qid in quantities:
        add(by_id[qid])
    # Preserve registration order for a stable, readable build sequence.
    return [cls() for cls in classes if cls.quantity_id in selected]


def aggregate(
    catalog: Sequence[dict],
    out_dir: Path | str | None = None,
    *,
    params: Mapping[str, object] | None = None,
    quantities: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Pool every built product across *catalog* into the index-keyed tables.

    Thin pass-through to
    :func:`cellflow.contact_analysis.shape_tables.aggregate`: returns the
    table name → written CSV path map. The tables are written **flat** under
    *out_dir* (``<out_dir>/<name>.csv``); *out_dir* defaults to the catalogue root
    (the common ancestor of the positions). *params* carries the shared build
    knobs (pixel size, frame interval, FOV area, …) so a param-gated cheap
    quantity (e.g. cell shape needing ``pixel_size_um``) pools with the same
    knobs a normal build would have used. *quantities* restricts which tables are
    written (``None`` = all); see :func:`shape_tables.aggregate`.
    """
    return shape_tables.aggregate(catalog, out_dir, params=params, quantities=quantities)


def author_config(
    out_dir: Path | str,
    records: Sequence[dict],
    *,
    tables_dir: str | None = None,
    quantities: Sequence[str] = (),
    params: Mapping[str, object] | None = None,
    catalog_name: str = "catalog.csv",
    config_name: str = "config.toml",
) -> Path:
    """Write ``catalog.csv`` + ``config.toml`` into *out_dir*; return the config path.

    The composition point behind the studio's "Save config…" / "Run": persist the
    in-memory *records* to a catalog CSV, then author a run-config beside it that
    points at that CSV (a relative ``catalog`` key, so the folder stays
    relocatable). *tables_dir* is written as the config's ``out_dir`` (where the
    flat pooled tables land); ``None`` leaves it unset (defaults to the catalogue
    root at run time). ``run(author_config(...))`` reproduces the UI's run
    headlessly. Creates *out_dir* if missing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_catalog(out_dir / catalog_name, records)
    return write_config(
        out_dir / config_name,
        catalog=catalog_name,
        out_dir=tables_dir,
        quantities=quantities,
        params=params,
    )


def run(
    config_path: Path | str, *, progress_cb=None, build: bool = True
) -> dict[str, Path]:
    """Run the whole pipeline from a TOML run-config: the "author once, then run".

    Loads the :class:`~cellflow.contact_analysis.config.RunConfig`, then
    threads its choices through the stages: load the catalog CSV, build the
    selected *quantities* — of which only producers (``contacts``) actually
    persist a per-position artifact, see :func:`build_quantities` — and aggregate
    into the flat measurement tables under the config's ``out_dir`` (default: the
    catalogue root), where the cheap quantities are computed in memory. The same
    ``quantities`` selection restricts which tables ``aggregate`` writes (empty =
    all), so a pool-only run honors the choice too. Returns
    the table name → written CSV path map. The optional *progress_cb* is
    forwarded to the build stage.

    Set *build* to ``False`` for **pool-only**: skip the per-position build stage
    entirely and go straight to aggregate. The producer's ``.build`` overwrites its
    ``.h5`` unconditionally (it is not guarded like ``ensure_contacts``), so a plain
    ``run`` recomputes every position's contacts even when the artifact already
    exists. ``build=False`` reads the existing ``.h5`` via each quantifier's
    ``compute_object_table`` and computes the cheap quantities in memory — a
    load-and-pool with no recompute. Positions missing their ``.h5`` simply
    contribute nothing (scope them out before calling).
    """
    cfg: RunConfig = load_config(config_path)
    catalog = load_catalog(cfg.catalog)
    # The global build knobs (``pixel_size_um`` / ``time_interval_s``) are read off
    # each record by ``position_inputs_from_record``; in the studio they are stamped
    # per-position, here they come from the config. Stamp config params onto every
    # record that does not already carry them, then also pass them to the build loop
    # (the required-param gate + opt-in ``wants_build_params`` quantifiers).
    for record in catalog:
        for key, value in cfg.params.items():
            record.setdefault(key, value)
    if build:
        build_quantities(
            catalog,
            quantifiers=select_quantifiers(cfg.quantities),
            params=cfg.params or None,
            progress_cb=progress_cb,
        )
    return aggregate(
        catalog, cfg.out_dir, params=cfg.params or None, quantities=cfg.quantities or None
    )
