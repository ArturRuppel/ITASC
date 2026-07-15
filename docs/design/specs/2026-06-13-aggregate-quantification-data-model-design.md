# Aggregate Quantification — Data-Model Redesign

Date: 2026-06-13

Reworks how Aggregate Quantification *persists* and *consumes* quantities. The
compute seam (the `Quantifier` registry) and the per-position build are kept.
What changes is everything downstream of the build: instead of plotting by
pooling N positions × M per-position artifacts live on every plot, we **persist a
small set of aggregated, index-keyed tables** and read plots from those, with an
explicit **reduce** stage in between.

## Why

The study is called *Aggregate* Quantification but it never persists an
aggregate. Today every quantifier writes a per-position artifact
(`position_dir/aggregate_quantification/…`), and the only aggregation is
in-memory at plot time (`plots/_pooling.py::pool_quantity` reads each in-scope
position's `object_table` and concatenates on every plot). Consequences:

- The on-disk data model is N positions × ~13 quantifiers of scattered files,
  none of them the thing you actually plot or hand to pandas/Prism.
- The pooling + reduction logic is welded inside the napari/plotting layer
  (`plotting.py::reduce_to_units`), so there is no clean seam to add other
  table transforms.

The fix is **not** to delete the per-position artifacts — they stay, visible, and
keep their crash-safe, incremental, per-position build (compute is inherently
per-position: its input is a per-position label image). We **add** a layer of
aggregated tables on top of them.

## The pipeline

```
Catalogue          positions table; selection = scope (empty = whole catalogue)   [unchanged]
   │
Build              metric checkboxes + Run → per-position artifacts               [unchanged]
   │   ↳ auto-aggregate the affected tables after a Run finishes
Aggregate          one button (selection-scoped) + status of the shape tables     [NEW]
   │   writes  <catalogue>/aggregate_quantification/{cells_by_frame,…}.csv
Reduce             ordered filter/collapse pipeline over a picked table           [NEW]
   │
Plot               draws the reduced table (no more live pooling)                 [rewired]
```

The aggregated tables are **materialized views**: regenerate-whole, never
upsert. Because a re-aggregate rewrites the file from scratch, there is never a
concurrent partial write into a shared file, and CSV stays viable. Per-position
artifacts are the normalized source of truth; the aggregated tables are a
reproducible projection of them.

### Ownership decisions (locked)

- **Keep per-position artifacts**, visible, exactly as today. Do **not** hide
  them. Add the aggregated tables alongside.
- **Aggregated tables are regenerated snapshots**, not the source of truth and
  not upserted into. A build is per-position and isolated; aggregation pools.
- Non-tabular viz geometry (contacts edges/T1 in `contact_analysis.h5`) is
  per-position by nature (the overlay reads it locally; derived quantifiers read
  it as input) and is unaffected.

## Section 1 — The aggregated file set

Grouping principle: **by the natural index a measurement is keyed on.** Every
quantity already emits a tidy `object_table` with a definite index; grouping by
that index collapses the quantifier zoo to one file per distinct index. All
tables carry the metadata prefix `condition · date · position_id` and pool every
in-scope position into a single file at `<catalogue>/aggregate_quantification/`.

| File | Index keys (after metadata) | Quantities that land here | Default plotting regime |
|---|---|---|---|
| `cells_by_frame` | `frame, cell_id` | cell_shape, nucleus_shape, shape_relational, contacts cell-morphometry, neighbor_count, cell_density, cell/nucleus dynamics *instantaneous* (pos, vel, speed) | distribution over cells (collapse frames→cell) |
| `cell_neighbors_by_frame` | `frame, cell_id, neighbor_label` | neighbor_enrichment | distribution / heatmap over cell × neighbor-type |
| `edges_by_frame` | `frame, focal_id, partner_id` | signed_contact_length (+ `contact_type` axis) | distribution / potential over edges |
| `contact_types_by_frame` | `frame, contact_type` | contact_type_zscore | bar / line over contact-type |
| `frames` | `frame` | collective dynamics (order param, corr length, NN dist, n_cells) | line over frame, then bar across positions |
| `tracks` | `cell_id` | per-track dynamics summary (MSD slope, persistence, n_frames…) | distribution over tracks |
| `dac_curves` | `lag_s` | direction-autocorrelation curve | line over lag |

**Scope of implementation.** Define all seven indices as the target schema.
Implement the **main four** first — `cells_by_frame`, `edges_by_frame`,
`contact_types_by_frame`, `frames` — which cover almost everything routinely
plotted. Wire the specialized three (`cell_neighbors_by_frame`, `tracks`,
`dac_curves`) as their plots come online.

**No ragged cross-index merges.** Quantities of different index never share a
file (that would mean a table full of structural NaN). Contact data therefore
legitimately spans two files: the raw per-edge table (`edges_by_frame`) and the
per-type z-score aggregate (`contact_types_by_frame`). That is expected.

### How aggregation builds a table

For each shape table, for each quantity that targets it and is built for an
in-scope position:

1. Read the position's `object_table` (the quantity's tidy per-object table).
2. Outer-join, **within that position**, every targeting quantity on the table's
   key columns (so a `cells_by_frame` row carries cell_shape *and* neighbor_count
   *and* density columns side by side; ids never collide across positions).
3. Stamp `condition / date / position_id`.
4. Concatenate across positions; write one CSV.

The NLS `class_label` sidecar joins by `cell_id` into `cells_by_frame` exactly as
`_pooling.py` does today (absent → `unclassified`).

This generalizes today's `plots/_pooling.py::pool_quantity` from "one quantity at
a time, in memory" to "all quantities of an index, persisted."

## Section 2 — The reduce layer

A seam **between the CSV loader and `build_figure`**: table in, table out, pure
and in-memory (nothing new persisted). The reduce layer is **transparent
composable primitives — no hidden cleverness; the user owns correctness.**

A reduce spec is an **ordered pipeline of steps**, each one of two primitives,
freely orderable and repeatable:

- **`filter(predicate)`** — keep rows matching `col op value`
  (`class_label == "epithelial"`, `frame >= 10`, `contact_type != "unlabelled"`).
- **`collapse(by=[keys], stat=mean|median|count)`** — plain group-by: aggregate
  the value columns down to one row per `by`-combination.

An empty pipeline is the identity ("no reduction").

**Chaining expresses the statistics.** The pseudoreplication-safe nested
reduction is just chained single-rung collapses, and the user composes it
explicitly:

- `collapse(by=[…,cell_id])` then `collapse(by=[…,position_id])` → the
  equal-weighted per-position result (each cell counts once within its position,
  each position once within its condition).
- a single `collapse(by=[…,position_id])` → the flat pooled result (rows weighted
  by frame/cell count).

The two diverge **only** when the collapse spans an intermediate level with
unequal child counts; for a single rung (`frames → cell`) flat and nested are
identical. Nothing in the layer decides this for the user — the pipeline order is
the knob.

### Mechanics (locked)

- **Defaults are a pre-filled, fully-editable pipeline.** Each shape table opens
  with a sensible starting pipeline (e.g. `cells_by_frame` →
  `collapse(by=[…,cell_id])` so plots open per-cell). It is a starting point, not
  enforced — the user can edit or clear it. This is how "every file has a default
  plotting regime" stays true without magic.
- **`collapse` on a column that is neither a group key nor a numeric value**
  (e.g. `class_label` when collapsing by `cell_id`): keep it if constant within
  the group, drop it if it varies. To preserve a varying attribute, add it to
  `by`.
- **`derive` (new column from an expression) is deferred** (YAGNI). The pipeline
  stays open so it slots in later without rework.

### Relationship to today's code

`plotting.py::reduce_to_units` and the `PlotSpec.level` ("cell" / "position" /
"date") machinery encode the nested reduction implicitly. In the new model that
logic moves into the explicit reduce pipeline as composed `collapse` steps; the
implementation plan decides whether `PlotSpec.level` is retired outright or kept
as a thin convenience that expands to a default collapse chain. The careful
statistics are preserved — they just become user-visible composition instead of a
hidden plot parameter.

## Section 3 — The UI

Inserts two stages into the existing vertical studio stack and rewires the plot
data source.

**Aggregate area** — a thin new collapsible between Build and Plot:

- a **Run Aggregate** button, scope = the catalogue selection (empty = whole
  catalogue), reading **what is built on disk** for those positions (not the Build
  checkboxes — those mean *what to compute*; reusing them for *what to aggregate*
  would overload them and risk partial tables).
- a small **status list** of the shape tables: built / empty, row count,
  last-written.
- **auto-fires after each Build Run** for the affected tables, so the common path
  never goes stale; the button is for re-aggregating a subset or aggregating
  without rebuilding.

**Reduce editor** — its own panel (distinct from plotting), bound to the
currently-picked table. An ordered, reorderable list of steps:

```
cells_by_frame   (148,201 rows)
  1. filter   class_label == epithelial                         [×]
  2. collapse by=[condition,date,position_id,cell_id] stat=mean [×]
  [+ filter] [+ collapse]                            → 3,012 rows
```

Empty list = no reduction. Each table opens with its default pipeline (editable).
Switching tables switches to that table's pipeline — **one active pipeline per
table** ("this table, reduced this way").

**Plot** — sources the reduced table instead of pooling per-position live.
`build_figure` / `PlotSpec` already take a value column + group-by, so the generic
plots (hist / box / violin / bar / line) point at a column of the reduced table.
The bespoke plots (potential landscape, DAC curve, neighbor-enrichment heatmap)
are unchanged **except** their data source flips from live-pool to "read the
reduced table."

### UI decisions (locked)

1. Reduce is its **own region**, not folded into each plot panel — set the
   reduction once per table, not per plot.
2. **One pipeline per table** (not global, not per-plot) — the simplest honest
   model.
3. **Data-source flip only** — keep every existing plot type, change only where
   it reads from. Fully generic "any column, any plot" is a later option, not
   this unit.

## Backend deltas

- **New `aggregate_quantification/shape_tables.py` (backend, no Qt).** A registry
  of the shape tables (name → key columns) and an `aggregate(records, scope)`
  that pools built `object_table`s into the index-keyed tables and writes the
  CSVs. Generalizes `plots/_pooling.py::pool_quantity`. Each `Quantifier` declares
  its target table + key columns (a small class attribute pair), so a new
  quantifier joins an existing table by declaration.
- **New `aggregate_quantification/reduce.py` (backend, no Qt).** The `filter` /
  `collapse` primitives and the pipeline runner (`table, spec -> table`). Lifts
  the reduction semantics out of `plotting.py`.
- **`plots/_pooling.py` live pooling is removed**; the plot area loads a shape-
  table CSV and applies the reduce pipeline instead.
- **Studio** gains the Aggregate area and the Reduce editor; the plot area's data
  source is rewired; Build auto-triggers aggregation of affected tables.

All new backend modules stay Qt-free so the standalone `cellflow-aggregate` wheel
and headless/batch runs use them unchanged.

## Test plan

- **Aggregation (backend):** building two positions of one index → a single
  pooled CSV with both positions' rows, metadata stamped, ids non-colliding;
  multiple quantities of one index → one table with all value columns
  outer-joined on the keys; a position missing a quantity → NaN in those columns,
  not an error; NLS `class_label` joined by `cell_id`, absent → `unclassified`.
- **Materialized-view semantics:** re-aggregating rewrites the file whole;
  removing a position from scope and re-aggregating drops its rows.
- **Reduce (backend):** empty pipeline = identity; `filter` keeps only matching
  rows; single `collapse` = flat group-by; chained single-rung collapses =
  equal-weighted nested result (golden compare vs today's `reduce_to_units`);
  `collapse` keeps a constant attribute column, drops a varying one; order of
  steps matters and is honored.
- **Plot:** every existing plot renders from a reduced table identical to what it
  rendered from live pooling for the same effective reduction (golden-figure or
  golden-`plotted_table` compare); no plot path still calls live pooling.
- **UI:** Run Aggregate writes the tables for the selected scope; Build auto-
  aggregates affected tables; switching the picked table switches its reduce
  pipeline; a table opens with its default pipeline.
- Run: `QT_QPA_PLATFORM=offscreen uv run --quiet pytest tests/`.

## Out of scope / follow-ons

- `derive` reduce primitive (expression columns).
- Fully generic "any numeric column, any plot type" plot area (this unit only
  flips the data source).
- The specialized three tables' plots (`cell_neighbors_by_frame`, `tracks`,
  `dac_curves`) beyond defining their schema.
- Staleness *indicators* on the Aggregate status list beyond the auto-refresh
  after Build (e.g. "tables older than their per-position inputs").
- Parquet (instead of CSV) for the large `cells_by_frame` / `edges_by_frame`
  tables — revisit if CSV load time bites at cohort scale.
```
