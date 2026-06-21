# Aggregate Quantification: artifact contract & orchestration

**Status:** designed; pre-implementation. Settles most of the "Deferred" list in
the migration plan.
**Date:** 2026-06-21
**Scope:** the three-artifact contract (config → tidy table → curation), the
identity model that anchors it, the curated/derived separation, and the
registry-driven orchestration the `cellflow-aggregate` command runs.
**Companions:** builds on
`2026-06-19-aggregate-quantification-migration-plan-design.md` (the migration
plan, Approach A) and `2026-06-19-aggregate-quantification-depart-napari-design.md`
(the direction note). Output formats follow `2026-06-17-iris-export-design.md`.
**External principle:** the curated/derived split follows the four-content-kinds
model in `~/Projects/electronic_labbook/README.md` ("the dividing line for derived
data is *who made the decisions in it*").

## 1. Three artifacts, three lifecycles

The pipeline produces and consumes exactly three artifacts. Each maps onto one of
the ELN's content kinds, and the mapping dictates how it is stored and versioned.

| # | Artifact | Holds | ELN kind | Treatment | Authored |
|---|---|---|---|---|---|
| 1 | **Config** (the catalog) | position paths, `experiment_id`, `condition`, `date`, position id | *code* | git-versioned | upfront, by hand |
| 2 | **Tidy table(s)** | identity columns (folded in from config) + measurement columns | *automatic derived* | **disposable**, regenerable, not committed | by the command |
| 3 | **Curation** | `excluded`, `qc_reason` per row | *curated derived* | git-versioned, revisable | after, in the notebook |

The flow:

```
config (1) ──[command]──► tidy table (2) ──[inspect in notebook]──► curation (3)
                              └────────── exported table = (2) ⨝ (3) ──────────┘
```

Two consequences are load-bearing:

- **Config folds into the table at build time and the table stays disposable.**
  Because `experiment_id` / `condition` are deterministic *parameters*, the tidy
  table can carry them and still be a pure function of config + raw data + code —
  re-running reproduces it byte-for-row. This is why the IDs may live "in the same
  file" as the measurements: they are not curation.
- **Only post-hoc human judgement is held out.** `excluded` / `qc_reason` are not a
  function of code (a human looked at the data and decided), so they live in the
  separate, git-versioned curation artifact and are *joined back*, never written
  into the disposable table as a system of record.

The exported tidy table (the file Iris / Prism / R read) is itself a disposable
join result — `excluded` appears there as a derived column, authoritative only in
artifact 3.

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
so that curation references (artifact 3) survive a regeneration of the table.

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

## 3. Tidy-table column contract

One row per object-per-frame (for `cells_by_frame`, one row per cell-per-frame):

| Group | Columns | Source | Authority |
|---|---|---|---|
| Identity | `experiment_id`, `condition`, `position_id`, `cell_id`, `frame` | config + data | config (params) |
| Row key | `id` (deterministic, stable) | derived | derived |
| QC | `excluded` (bool), `qc_reason`, `class_label` | curation + NLS sidecar | **curation (artifact 3)** |
| Descriptor | `date` (free text, not an axis) | config | config |
| Measurements | one column per quantity, tidy & typed | quantifiers | derived |

Default export formats: **Parquet + CSV** (per the migration plan default;
Parquet for typing/size, CSV for the Prism / Excel / Tableau path). Excel only on
demand.

## 4. Curated / derived separation

- **Curation is one file per experiment series**, living at the series /
  aggregate root (matching the ELN's `CODE`-without-`-NN` aggregate-folder
  concept). Columns: `(id, excluded, qc_reason)`. Git-versioned.
- It is **left-joined into the exported table by `id`** — the same mechanism by
  which the NLS `class_label` sidecar already joins in `shape_tables.build_table`
  (`class_label` left-joined by `cell_id`, absent → `unclassified`). The NLS
  sidecar is the *pattern*; the new curation artifact is a single series-level file
  rather than per-position.
- **Filter, don't delete.** Excluded rows stay in the table with the flag set;
  downstream filtering (`reduce.Filter`, which is a no-op on a missing column —
  `reduce.py:101`) and Iris (`excluded`) honour it. The raw measurement table stays
  complete and auditable.
- **The curation notebook writes the curation file, never the measurement table.**
  Notebook/kernel state is never authoritative; the artifact is.

## 5. Modular architecture (already in place)

The orchestrator does not need a new modular design — one exists and holds at the
worst case (contacts). Three layers:

| Layer | Files | Size | Role |
|---|---|---|---|
| **Engines** | `contacts/`, `dynamics/`, `shape/` | ~3,640 lines | Heavy domain transforms; own their persistence (`contact_analysis.h5`, etc.); know nothing about catalogs / tables / Iris. |
| **Quantifier adapters** | `quantifiers/*.py` | ~820 lines, 45–87 each | Thin; auto-register via `quantity_id` (`quantifier.py:111` `__init_subclass__`); declare `requires`/`produces` (DAG edges) and `shape_table`/`table_keys` (where their `object_table()` lands). |
| **Registry aggregation** | `shape_tables.py` | — | Pools every built adapter's `object_table()` into index-keyed tidy tables by `shape_table` membership; outer-joins on keys; stamps config metadata; left-joins curated columns. |

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
| Tidy-table artifact **formats** | **Resolved** — Parquet + CSV default; Excel on demand (§3). |
| Unique-`id` rule for discovery | **Resolved** — compositional identity + load-time uniqueness validation; deterministic row `id` (§2). |
| CLI subcommand **surface (center of gravity)** | **Resolved in shape** — config-in / table-out: `discover` / `build` / `aggregate` / `export` (§6). Exact flags & progress reporting still open. |
| Notebook surface | **Partly resolved** — split into an interactive **QC/curation** notebook (writes artifact 3) and a headless **report** notebook (papermill/nbconvert → HTML, reads the final joined table). Plotting library & content still open. |
| napari teardown sequencing | Still deferred — gated on the cut gate. |
| Config-driven runner (Approach B) | Out of scope. |

## 8. Open items

- **Config format:** keep the flat **CSV** catalog (Excel-editable, matches the
  discover→fill workflow) — recommended — or elevate to TOML/YAML. CSV unless
  structure it cannot express is needed.
- **`experiment_id` required vs optional:** recommend **optional** with a
  `date`-fallback default so existing hand-written catalogs still load; the column
  is authoritative when filled.
- **Curation file** exact location/name within the series root, and `qc_reason`
  vocabulary (free text vs controlled categories).
- **Flag-without-excluding:** whether QC needs a categorical "review / outlier,
  keep in data" status beyond the `excluded` boolean.
- **CLI flags, progress reporting, notebook plotting library** — as above.
- **ELN `stamp()` integration** (later): tidy table → `kind="derived"`, curation /
  config → `kind="curated"` / code. Downstream, not now.
