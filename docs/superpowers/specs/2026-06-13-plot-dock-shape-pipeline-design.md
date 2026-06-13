# Plot-dock Shape pipeline — rationalizing the runtime plotting interface

**Date:** 2026-06-13
**Status:** Design approved, pending implementation plan

## Problem

The Aggregate Quantification studio has a real pipeline boundary at **Aggregate**:
the *Run Aggregate* button pools what is built on disk into the index-keyed shape
tables (written once under `<catalogue>/aggregate_quantification/`). Everything
*after* that boundary is computed live, per plot, inside the **plot dock**
(`PlotPanel`, opened as a tab when the user clicks *Plot*).

The plot dock's interface had grown confusing. After an earlier change it read:

```
Aggregate → Reduce → Plot → Filter → Styling
```

Three things were wrong:

1. **Term collision.** The dock's "Aggregate" section (Value / Stat / Group-by)
   reuses the studio's "Aggregate" term, which means something entirely different
   (the on-disk pooling step). Two unrelated things called the same name.
2. **Split data-shaping.** *Filter* (keep rows matching a predicate) and *Reduce*
   (collapse to an independent unit) are both "shape the rows before plotting,"
   yet they lived in two separate, differently-shaped sections.
3. **No shape feedback.** A collapse silently changes the number of rows (frames →
   cells → positions). The user could not see what each operation did to the data.

## Goal

Rationalize the **plot dock only** — nothing moves to the studio; the controls
are inherently live and per-plot. The dock becomes:

```
Shape → Plot → Styling
```

- **Shape** — one unified, ordered pipeline of `filter` and `collapse` steps with
  a live row-count trail. Replaces both the old Filter section and the Reduce
  (collapse) editor.
- **Plot** — absorbs Value, Group-by, and Stat (the old "Aggregate" controls, no
  longer called that) alongside plot type and plot-specific options.
- **Styling** — unchanged.

## Non-goals

- No change to the studio (`AggregateQuantificationStudioWidget`) or its
  Aggregate area. The on-disk aggregation boundary stays exactly where it is.
- No relocation of plotting controls into the studio. They stay in the dock,
  where they re-render live.
- No carry-over of pipeline steps across a product/value switch (reseed instead).

## Design

### A. Section restructure

The `PlotPanel` sections go from `Aggregate / Reduce / Plot / Filter / Styling`
to **`Shape / Plot / Styling`**:

- The `CollapsePipelineEditor` (reduce editor) and the
  `_build_filters` / `_rebuild_filters` checkbox machinery are **both replaced**
  by a single new `ShapePipelineEditor` widget housed in the **Shape** section.
- Value, Stat, and Group-by move into the **Plot** section (they define the
  plot's mapping). Their widget attribute names are preserved
  (`_value_combo`, `_stat_combo`, `_group_checks`, …) so the rename is structural,
  not a wholesale rewrite.

### B. The Shape pipeline widget (`ShapePipelineEditor`)

One ordered, editable list of steps, each created via `[+ filter]` or
`[+ collapse]`:

- **filter step** → `Filter(column, op, value)`:
  - `[column ▾] [op ▾] [value]`.
  - `op` drawn from `FILTER_OPS` (`==`, `!=`, `>`, `>=`, `<`, `<=`).
  - The value control adapts to the column: a **dropdown of distinct values** for
    a categorical column, a **free numeric entry** for a numeric one.
  - The column dropdown lists **only columns present at that point in the
    pipeline** (so `n` appears as a filterable column only *after* a collapse).
- **collapse step** → `Collapse(by, stat)`:
  - The existing `by`-checkbox grid + `stat` dropdown (`COLLAPSE_STATS`), as in the
    current collapse editor.
- Per-step controls: reorder (`↑`/`↓`) and remove (`×`), as the collapse editor
  already has.
- **Row-count trail**: each step row displays the running row count *after* that
  step (`12,400 → 8,200 → 312 …`); the section header shows the starting count.
- **Default seed** for a table: one `collapse by <finest unit>` step (the current
  default), with no filter steps.

Headless contract (unit-testable without data or a render):

- `set_columns(columns, default)` — reset selectable columns and seed the default
  pipeline (no signal), as today's editor does, but `columns` is now the column
  set used to offer both filter columns and collapse `by` columns.
- `pipeline() -> tuple[Step, …]` — the current ordered pipeline as backend
  `Filter | Collapse` steps. Steps referencing no present column are skipped.
- `set_row_counts(counts: Sequence[int | None])` — display-only; the panel pushes
  per-step counts here after each render (see ownership below).
- `changed` — emitted on every edit.

### C. Backend: `n` on every collapse (reduce.py)

`Collapse.apply` currently emits an `n` group-size column only for `stat="count"`;
`mean`/`median` drop the group size. To make `filter n ≥ 5` (drop undersampled
units — the pseudoreplication payoff) work after any collapse, **every collapse
attaches `n` as the current group size**:

- `mean` / `median` collapse output gains an `n` column = group size (whole-table
  no-`by` collapse → `n = len(df)`).
- **`n` is reserved**: each collapse *recomputes* `n` to the current group size and
  **never treats a pre-existing `n` as a value column to average**. Chaining
  `collapse by cell` then `collapse by position` therefore yields per-position
  group sizes, not the mean of child counts.
- Downstream plotting selects a chosen value column and ignores the extra `n`
  column; `summary_table` already reports `n` separately.

### D. Wiring in `PlotPanel`

- **One pipeline, fed to the backend unchanged.** `run_pipeline` already applies a
  mixed `Filter | Collapse` list in order, and `reduce_to_units` routes
  `PlotSpec.collapse` straight through `run_pipeline` (plotting.py). So
  `current_spec` builds the full Shape pipeline (filters + collapses interleaved)
  and hands it in as the spec's pipeline — **no new backend plumbing**. Filtering
  is no longer applied separately in `_render_df`; the whole Shape pipeline runs
  as one.
- **Group-by survives collapses.** As today, the active Group-by columns are
  unioned into each **collapse** step's `by` (never into filter steps) so a
  comparison axis is never collapsed away. The row-count trail is computed from the
  *same* group-by-unioned spec, so the displayed counts match what is plotted.
- **Row-count trail ownership.** The widget stays headless and data-free:
  `pipeline()` returns raw steps; the **panel** owns the dataframe + group-by, runs
  the cumulative `run_pipeline`, and calls `set_row_counts([...])` with the count
  after each step.
- **Picking / click-to-load.** `_plot_df` becomes the fully-reduced table (what is
  actually drawn), keeping pick-row stamping aligned with the plotted points.
  `_effective_level` (load granularity) keeps its current logic but reads the last
  **collapse** step in the pipeline (filter steps ignored); the empty-pipeline
  fallback (`_idempotent_level`) is unchanged.
- **Product/value swap** (catalog mode): on a switch that changes columns, reseed
  the default collapse and drop steps referencing absent columns (no carry-over).

### E. Testing

- **reduce.py**: `n` attached on `mean`/`median` collapse (group size; no-`by`
  whole-table case; recompute-on-chain gives per-parent counts; a reserved
  pre-existing `n` is overwritten, not averaged).
- **ShapePipelineEditor** (headless): mixed pipeline → correct `ReduceSpec`; filter
  step categorical-dropdown vs numeric-entry; offered columns reflect pipeline
  position (`n` only after a collapse); default seed; reseed on product swap;
  `set_row_counts` rendering; `changed` fires on edits.
- **plot_panel**: `current_spec` builds the mixed pipeline with group-by unioned
  into collapse steps only; the old filter-checkbox tests migrate to pipeline
  filter steps; `_collapse_editor` references become `_shape_editor`; Value / Stat /
  Plot combos keep their attribute names while living in the Plot section.

## Affected files

- `src/cellflow/aggregate_quantification/reduce.py` — `n` on every collapse.
- `src/cellflow/napari/aggregate_quantification/reduce_editor.py` → reworked/renamed
  into `shape_editor.py` (`ShapePipelineEditor`).
- `src/cellflow/napari/aggregate_quantification/plot_panel.py` — section
  restructure, Shape wiring, row-count trail, picking alignment.
- `tests/napari/test_reduce_editor.py` → `tests/napari/test_shape_editor.py`.
- `tests/napari/test_plot_panel.py` — migrate filter/collapse references.
- `tests/aggregate_quantification/test_reduce.py` — `n`-on-collapse coverage.
