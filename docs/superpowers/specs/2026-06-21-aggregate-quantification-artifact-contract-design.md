# Aggregate Quantification: artifact contract & orchestration

**Status:** designed; pre-implementation. Settles most of the "Deferred" list in
the migration plan.
**Date:** 2026-06-21
**Scope:** the artifact contract (config → tidy table → Iris-only export; curation
removed — see the §1 banner), the
identity model that anchors it, the curated/derived separation, and the
registry-driven orchestration the `cellflow-aggregate` command runs.
**Companions:** builds on
`2026-06-19-aggregate-quantification-migration-plan-design.md` (the migration
plan, Approach A) and `2026-06-19-aggregate-quantification-depart-napari-design.md`
(the direction note). Output formats follow `2026-06-17-iris-export-design.md`.
**External principle:** the curated/derived split follows the four-content-kinds
model in `~/Projects/electronic_labbook/README.md` ("the dividing line for derived
data is *who made the decisions in it*").

## 1. Artifacts and lifecycles

> **Amended 2026-06-21 (curation removed from the pipeline).** This section
> originally described **three** artifacts, the third being a CellFlow *curation*
> file (`excluded`, `qc_reason`) joined into the export. That is gone. The sibling
> **Iris** project removed its dedicated row-exclusion mechanism entirely (see
> `~/Projects/Iris/docs/superpowers/specs/2026-06-21-remove-exclusion-carve-out-design.md`):
> a `.iris` is now a **pure function of (input table + analysis spec)**, and the
> way to drop rows is an ordinary **boolean flag column set upstream + a `filter`
> reduce step** — not a baked-in curated column. CellFlow therefore no longer
> emits or joins a curation artifact, and the export bakes in **no human
> judgement**. If row-level QC is reintroduced later it is the Iris-native way: a
> flag column produced by the pipeline plus a premade filter step in the analysis
> spec, whose own "why/who/when" provenance lives upstream where it belongs.

The pipeline produces and consumes two artifacts. Each maps onto one of the ELN's
content kinds, and the mapping dictates how it is stored and versioned.

| # | Artifact | Holds | ELN kind | Treatment | Authored |
|---|---|---|---|---|---|
| 1 | **Config** (the catalog + TOML run-config) | position paths, `experiment_id`, `condition`, `date`, position id; run-level `quantities` / `params` / `export_dir` | *code* | git-versioned | upfront, by hand |
| 2 | **Tidy table(s)** | identity columns (folded in from config) + measurement columns | *automatic derived* | **disposable**, regenerable, not committed | by the command |

The flow:

```
config (1) ──[command]──► tidy table (2) ──[export, Iris-only]──► .iris bundle
```

One consequence is load-bearing:

- **Config folds into the table at build time and the table stays disposable.**
  Because `experiment_id` / `condition` are deterministic *parameters*, the tidy
  table can carry them and still be a pure function of config + raw data + code —
  re-running reproduces it byte-for-row. This is why the IDs may live "in the same
  file" as the measurements.

The export is **Iris-only**: the canonical tidy CSVs stay under the catalogue root
(`aggregate_quantification/`) for programmatic use; `export` writes one `.iris`
bundle per Iris-selected table into `<export_dir>/iris/` and re-emits nothing in
other formats (no CSV/parquet mirror).

## 2. Identity model

### `experiment_id` is a blocking factor crossed with `condition`

Replicate identity lives in an explicit **`experiment_id`** (rename of the
informal "repetition" concept). It is **orthogonal to `condition`**: two rows with
the same `experiment_id` and different `condition` are *the same experiment* —
paired / matched observations. So the identity axes are a crossed grid, not nested:

```
observation key   = (experiment_id, condition, position_id, cell_id, frame)
position identity  = (experiment_id, condition, position_id)   ← validated unique at load
replicate ("n")    = number of distinct experiment_id values   ← not positions, not cells
date               = free-text descriptor; OFF the identity / sort axes
```

The paired interpretation (paired SuperPlots with connecting lines, paired /
repeated-measures statistics, handling experiments unbalanced across conditions)
is **downstream — Iris owns it.** The pipeline's only obligation is to *carry
`experiment_id` as a correct, first-class column*. We do not implement pairing or
stats here.

### Naming-collision cleanup (required, do first)

`experiment_id` **already exists in the code as an alias for `date`** —
`catalog.py:145, 167, 286`. Adopting the new meaning is a *repurpose*, not a fresh
column. Those three `experiment_id → date` fallbacks must be deleted, or an old
catalog will silently feed `date` into the paired key.

### Deterministic row `id`

Every tidy row carries a stable `id` (Iris already keys on it — `iris_export/
export.py:87`, `_with_meta_columns`). It **must be a deterministic function of the
identity tuple** `(experiment_id, position_id, cell_id, frame)`, not a row counter,
so a row keeps the same identity across regenerations of the table — which Iris
keys on, and which any upstream flag/annotation joined by `id` relies on.

### Schema changes

- **Config / catalog:** add `experiment_id`. Keep `date` as descriptor.
  `REQUIRED_CSV_COLUMNS` is currently `(path, date, condition, id)`
  (`catalog.py:36`).
- **Aggregation:** `METADATA_COLUMNS` is `(condition, date, position_id)`
  (`shape_tables.py`); change to `(condition, experiment_id, position_id)` and
  carry `date` as a non-axis descriptor column.
- **Iris export:** the grouping hierarchy is `date → position_id → <object_key> →
  frame` (`iris_export/export.py:22`); change the replicate axis to
  `experiment_id`, with `condition` as the comparison axis.
- **Validation:** at catalog load, error if `(experiment_id, condition,
  position_id)` is not unique. This replaces the silent-merge bug where `id`
  defaults to the folder name (`catalog.py:215`) and collides; the human resolves a
  collision by editing `position_id` in the config.

## 3. Tidy-table contract: partitioning & columns

### Partition by quantifier, not by grain — no god tables

The current aggregation rule is **table = grain**: quantifiers sharing
`table_keys` are outer-joined into one file (`shape_tables.build_table`). That
collapses six unrelated *measurement domains* — `cell_shape`, `nucleus_shape`,
`cell_dynamics`, `nucleus_dynamics`, `shape_relational`, `neighbor_count`, all at
`(frame, cell_id)` — into `cells_by_frame`, a **~50–60 column god table** (shape
alone is ~12 columns via `shape/core.py`, emitted twice for cell and nucleus).
That is wide-format sprawl, not tidy: tidy means one observational unit per table,
not one unit × every domain measured on it.

**Decision: the table is the quantifier (one measurement domain), at its grain.**
Output files become one-per-quantifier:

```
cell_shape          (frame, cell_id)        nucleus_shape       (frame, cell_id)
cell_dynamics       (frame, cell_id)        nucleus_dynamics    (frame, cell_id)
nucleus_cell_shape  (frame, cell_id)        neighbor_count      (frame, cell_id)
cell_density        (frame, label)          contact_type_zscore (frame, contact_type)
neighbor_enrichment (frame, cell_id, focal_label, neighbor_label)   ← large by ROW count
signed_contact_length (frame, t1_event_id, role, contact_type)
```

This makes **the quantifier the single unit of compute *and* output**, with
several consequences:

- **No god table** — adding a quantifier adds a *file*, never widens a shared one
  (the modularity invariant of §5, extended to outputs).
- **Re-join stays trivial** — the deterministic `id` (and shared `frame`/`cell_id`)
  let a consumer join `cell_shape` + `cell_dynamics` for a combined view; curation
  (§4) joins onto every table uniformly by `id`.
- **`build_table` simplifies** — pool-per-quantifier-across-positions, no
  cross-quantifier outer join; the "quantifiers sharing a `shape_table` must
  declare matching `table_keys`" coupling constraint is dropped.

**Cost & mitigation:** cross-domain analysis needs a join, and there are more
files. An optional **export-time "view"** may join a *selected* set of per-domain
tables into one wide file on request — storage stays per-domain, a combined table
is a delivery option, not the default. For the SuperPlot use case (one metric at a
time) per-domain files are already the more convenient shape.

### Compute selectivity

`build_quantities` currently runs **every** registered quantifier that `can_build`.
That changes to **config-driven selection**:

- **Quantifier-level** (the lever): the config (artifact 1) carries a `quantities`
  list; only listed quantifiers run, so only their files appear — "compute less" and
  "fewer files" become the same control. Default is a **small sensible set, not
  "all"**.
- **Metric-level** (within a quantifier, e.g. only `area_um2` + `eccentricity` of
  shape's ~12) is **deferred** — splitting files largely defuses it: an unwanted
  metric is one column in a narrow domain file, not bloat in a god table.
  `shape/core.py:SHAPE_COLUMNS` makes an allowlist feasible later if a specific
  quantifier proves expensive per-metric (check the dynamics/MSD ones first).
- `neighbor_enrichment` / `cell_neighbors_by_frame` is the *row-count* god table
  (frame × cell × neighbor), orthogonal to width; the control there is simply not
  computing it unless asked.

### Per-table column contract

Each per-quantifier table is one row per object-per-frame, with this column shape:

| Group | Columns | Source | Authority |
|---|---|---|---|
| Identity | `experiment_id`, `condition`, `position_id`, `cell_id` (or the table's object key), `frame` | config + data | config (params) |
| Row key | `id` (deterministic, stable) | derived | derived |
| Subpopulation | `class_label` | NLS sidecar | derived (left-joined in `build_table`) |
| Descriptor | `date` (free text, not an axis) | config | config |
| Measurements | the quantifier's own columns, tidy & typed | the quantifier | derived |

Export is **Iris-only** (`.iris` bundles); the canonical tidy tables stay as CSV
under the catalogue root for programmatic use. No Parquet/CSV mirror is emitted.

## 4. Curated / derived separation

> **Amended 2026-06-21 — curation removed (see §1 banner).** CellFlow no longer
> emits a curation artifact and the export bakes in no human judgement. The
> paragraphs below are retained as the *rationale* that still holds, but the
> concrete `(id, excluded, qc_reason)` left-join is gone: it was built against
> Iris's old exclusion mechanism, which Iris has since removed.

- The principle stands: the dividing line for derived data is *who made the
  decisions in it*. Human QC judgement must not be silently fused into the
  disposable, regenerable measurement table.
- **Iris-native realisation.** With Iris's exclusion mechanism gone, the correct
  way to express "drop these rows" is a **boolean flag column carried on the table
  + a `filter` reduce step** in the analysis spec. Iris just declares "I filtered
  on `<flag> == false`" — honest and fully reproducible — and the flag's own
  provenance lives upstream in whatever sets it.
- **Filter, don't delete** still holds: flagged rows stay in the table; the figure
  reflects the filtered `n`. The raw measurement table stays complete and auditable.
- The **NLS `class_label`** left-join in `shape_tables.build_table` (by `cell_id`,
  absent → `unclassified`) remains — it is *automatic derived* (the classifier is
  code), not human curation, so it is not affected by this change.

## 5. Modular architecture (already in place)

The orchestrator does not need a new modular design — one exists and holds at the
worst case (contacts). Three layers:

| Layer | Files | Size | Role |
|---|---|---|---|
| **Engines** | `contacts/`, `dynamics/`, `shape/` | ~3,640 lines | Heavy domain transforms; own their persistence (`contact_analysis.h5`, etc.); know nothing about catalogs / tables / Iris. |
| **Quantifier adapters** | `quantifiers/*.py` | ~820 lines, 45–87 each | Thin; auto-register via `quantity_id` (`quantifier.py:111` `__init_subclass__`); declare `requires`/`produces` (DAG edges) and `shape_table`/`table_keys` (where their `object_table()` lands). |
| **Registry aggregation** | `shape_tables.py` | — | Pools every built adapter's `object_table()` into index-keyed tidy tables by `shape_table` membership; outer-joins on keys; stamps config metadata; left-joins the NLS `class_label`. |

**The modularity invariant:** the command is registry-driven and
transformation-agnostic (`build_quantities` defaults to every registered
quantifier — `pipeline.py:107`; `shape_tables` pools by membership). **Adding a
transformation never touches the command, aggregator, or exporter** — you write an
engine (if heavy) plus a thin adapter, and it auto-registers. Proof: contacts is a
2,127-line engine behind a 63-line adapter (`quantifiers/contacts.py`),
structurally identical to the 46-line `neighbor_count.py`.

Note **two disposable derived sub-layers**, both regenerable: per-position engine
artifacts (e.g. `contact_analysis.h5`) and the pooled tidy tables. The tidy-table
contract (§3) applies only to the adapter's `object_table()`; engines stay free to
use h5/npz/whatever for their intermediates.

## 6. Orchestration: the command and the DAG scheduler

The "command" is the `config → tidy table` transform — i.e. the CLI's centre of
gravity: `discover` (scaffold the config) → `build` (run the quantifier DAG) →
`aggregate` (registry pool) → `export` (tidy artifacts + `.iris`, joining
curation).

### First-class task: a dependency scheduler for the build DAG

`build_quantities` currently **plans all jobs up front from inputs computed before
any build runs** (`pipeline.py:110, 112–126`). Derived quantifiers gate on a
producer's artifact — `neighbor_count`, `neighbor_enrichment`,
`contact_type_zscore`, `cell_density`, and `signed_contact_length` all consume
`contact_analysis_path`, which `contacts` produces. **On a cold run their
`can_build` is `False` (the h5 does not exist yet) and they are silently skipped**
— the first non-napari run would emit a tidy table missing every contacts-derived
column, with no error. The napari studio hid this by invoking the build loop in
dependency *waves*; the extracted command has no scheduler.

Required: the command must **topologically sort quantifiers by `requires` /
`produces` and re-derive `PositionInputs` between waves** so dependents see
freshly built producer artifacts. This is a correctness blocker for the cut gate,
not a footnote.

## 7. Status of the migration plan's deferred decisions

| Deferred item | Status |
|---|---|
| Tidy-table artifact **formats** | **Superseded** — export is now **Iris-only** (§1 banner, §4). Canonical tidy tables stay CSV under the catalogue root; no Parquet/CSV mirror. |
| Unique-`id` rule for discovery | **Resolved** — compositional identity + load-time uniqueness validation; deterministic row `id` (§2). |
| CLI subcommand **surface (center of gravity)** | **Resolved in shape** — config-in / table-out: `discover` / `build` / `aggregate` / `export` (§6). Exact flags & progress reporting still open. |
| Notebook surface | **Partly resolved** — a headless **report** notebook (papermill/nbconvert → HTML, reads the tidy table). The QC/curation notebook is **dropped** with curation (§1 banner). Plotting library & content still open. |
| **Config format** | **Resolved** — a TOML **run-config** (`config.py:load_config` → `RunConfig`) is the "author once, then run" knob file: `catalog` (path to the CSV catalog), `quantities`, `[params]`, `export_dir`. The per-position **catalog stays CSV** (tabular, many-row, own relative-path resolution). |
| **Quantities selection source** | **Resolved** — config `quantities` list, consumed by `pipeline.select_quantifiers` (empty = all; a subset pulls in dependency producers transitively, so naming a contacts-derived metric brings `contacts` along). |
| **Output dir layout** | **Resolved** — no historical trail (config-once-then-run): a single **flat `export/`** dir from `export_dir`, holding `iris/`. Measurement tables stay under the catalogue root. |
| **Curation** | **Removed** — see the §1 banner. CellFlow emits no curation artifact; row QC, if reintroduced, is an upstream flag column + a filter reduce step (Iris-native). |
| napari teardown sequencing | Still deferred — gated on the cut gate. |
| Config-driven runner (Approach B) | **Resolved (in)** — `pipeline.run(config.toml)` threads load_catalog → build (selected quantities) → aggregate → Iris-only export. |

## 8. Open items

- **`experiment_id` required vs optional:** **resolved** — optional with a
  `date`-fallback default so existing hand-written catalogs still load; the column
  is authoritative when filled.
- **CLI flags, progress reporting, notebook plotting library** — as above.
- **Iris pre-plot selection** — which per-quantifier tables/metrics get premade
  SuperPlots in the `.iris` bundle. Separate, downstream concern (depends on the
  table layout above); decide later. Today `export` ships only `cell_shape`
  (`iris_export/export.py:TABLES_TO_EXPORT`); revisit which per-quantifier tables
  earn premade SuperPlots.
- **Export-time "view"** — whether/how to offer joined wide tables across selected
  per-quantifier files (storage stays per-domain regardless).
- **ELN `stamp()` integration** (later): tidy table → `kind="derived"`, curation /
  config → `kind="curated"` / code. Downstream, not now.
