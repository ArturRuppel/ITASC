# Lean aggregate stage: pool-only cheap quantities, persist only contacts, slim the catalog

Date: 2026-07-11 · Scope: `contact_analysis/` (quantifier seam, pooling, pipeline,
catalog, records) + the full-app catalog adapter (`napari/main_widget.py`) + docs.

## Problem

Today every quantifier writes a per-position artifact into
`<pos>/aggregate_quantification/` (a `.csv` or `.h5`), and the aggregate step reads
those files back and pools them into flat tables. That means each position folder
accumulates a pile of derived files — `cell_shape.csv`, `nucleus_shape.csv`,
`shape_relational.csv`, `cell_density.csv`, `cell_dynamics.h5`,
`nucleus_dynamics.h5`, `neighbor_count.csv`, `signed_contact_length.csv` — every one
of which is **cheap to recompute from the labels (or from the contacts h5)** and is
duplicated into the pooled tables anyway. The only per-position derived artifact that
is expensive to produce is `contact_analysis.h5` (the contact graph).

Three things follow from that observation:

1. The cheap per-position files are clutter. They should not be persisted; they
   should live only in the pooled aggregate tables, where every position is combined.
2. Once only `contact_analysis.h5` persists per position, the folder that holds it is
   no longer a grab-bag of "quantification outputs" — it holds exactly one file, and
   the folder should be renamed to `4_contact_analysis/` to mirror the staged layout
   (`0_input … 3_cell`).
3. The project catalog CSV can shrink to identity + the canonical artifact paths.

## The principle: producers persist, pooled leaves compute in memory

The split we want already exists in the quantifier metadata. Every registered
quantifier is exactly one of two kinds:

| quantifier | reads | `produces` | `table_keys` (pooled?) |
|---|---|---|---|
| **contacts** | cell labels | `contact_analysis_path` | — (not pooled) |
| cell_shape / nucleus_shape / shape_relational | labels | — | ✓ |
| cell_dynamics / nucleus_dynamics | labels | — | ✓ |
| cell_density | cell labels | — | ✓ |
| neighbor_count / signed_contact_length | contacts h5 | — | ✓ |

`contacts` is the **only** producer (its h5 is consumed as the input of
`neighbor_count` / `signed_contact_length`) and the **only** non-pooled quantifier.
Every other quantifier declares `table_keys` (it pools) and produces nothing. So the
rule is:

- **A producer persists** its artifact per position (only `contacts`, →
  `contact_analysis.h5`).
- **A pooled leaf never persists** per position: it computes its tidy table in memory
  at aggregate time and goes straight into the pooled table.

No new per-quantifier flag: the predicate "persists" is `bool(quantifier.produces)`.

## Workstream A: the quantifier seam (compute at pool time)

### A1. New seam method: in-memory compute

Add to `Quantifier`:

```python
def compute_object_table(
    self, inputs: PositionInputs, *, params: dict | None = None
) -> Mapping[str, np.ndarray] | None:
    """The pooled tidy table for one position, computed directly from inputs.

    Pooled quantifiers implement this; it returns the same column-major table
    that ``object_table`` used to return after a disk round-trip, but never
    touches disk. ``None`` when this position yields no rows."""
    raise NotImplementedError
```

The extraction is mechanical because every pooled quantifier's `build()` **already
computes the full table and then writes it**:

- Shape family (`build_object_shape` in `shape/core.py`): the table is `columns`,
  computed at `_extract_shape_columns(...)` before `write_table_csv`. Extract a
  `compute_object_shape(label_path, *, pixel_size_um, object_key) -> dict` that
  returns `columns`; `compute_object_table` calls it. (`shape_relational` mirrors this
  in `shape/relational.py`.)
- Contacts-derived (`neighbor_count`, `signed_contact_length`): the table is already
  in hand — `cell_neighbor_counts(derived.load_analysis(inputs))` — before
  `derived.persist`. `compute_object_table` returns it directly.
- Dynamics (`cell_dynamics`, `nucleus_dynamics`): same compute-then-write shape (h5
  writer). Extract the compute; verify per-module in the plan.
- `cell_density`: same, via `_tidy_table`.

Each pooled quantifier keeps its `table_keys`, `requires`, `required_build_params`,
`wants_build_params`, and metadata unchanged; only the persistence methods change.

### A2. Pooled quantifiers gain `compute_object_table`; keep their writers

Add `compute_object_table` to each pooled quantifier (via the `compute_object_shape`
extract for the shape family; direct return for the contacts-derived and dynamics
families). **Keep** their existing `build` / `read` / `object_table` /
`default_output_name` — they are still called by the live napari studio (see A5), so
this change does not delete them. `contacts` is untouched.

The per-position writers (`build_object_shape`, `write_table_csv`, `write_provenance`,
`_tidy_table.persist` / `read_derived_table`) therefore stay. Removing them is a
follow-up gated on retiring the interactive studio's per-position build UI (the pending
"aggregate front-end refocus" TODO), and is **out of scope** here. This spec's win is
that the *canonical pipeline* (`run()`) stops persisting the cheap quantities, not that
the writer code is deleted yet.

### A3. Pooling computes in memory, threading params

`shape_tables._position_frame` today does `output_for_record(quantifier, record)` →
`is_built(path)` → `object_table(path)`. Rewrite it to:

```python
inputs = position_inputs_from_record(record)
table = quantifier.compute_object_table(inputs, params=params_for(quantifier, record))
if table is None:
    continue
```

Consequences threaded through the pooling API:

- `build_table(name, records, *, params=None)` and `aggregate(records, out_dir=None,
  *, params=None)` gain a `params` argument. This is the real interface change:
  pixel size / frame interval / z-score shuffles now reach the quantifiers at
  **aggregate** time (they used to be applied at build time). Per-position values
  (`pixel_size_um` stamped on a record) still arrive via `PositionInputs`; the shared
  params arrive via the new argument.
- The `required_build_params` gate (skip `cell_shape` when no pixel size, etc.) and
  `wants_build_params` opt-in move from `build_quantities` into the pooling loop, so a
  quantifier missing a required param is skipped for that position rather than raising.
  Reuse the existing `missing_build_params` / `_record_build_params` logic verbatim,
  relocated.

### A4. `build_quantities` builds only producers

`build_quantities` now persists only producers (`bool(q.produces)` — in practice just
`contacts`). Pooled leaves are no longer built here (they are computed in `aggregate`),
so the dependency-ordering / produced-field planning that existed to sequence derived
builds after their producer is no longer exercised by pooled leaves. Simplify: build
the producer(s); keep the ordering machinery only if a producer-on-producer dependency
exists (none today) — otherwise remove it. `run()` passes `cfg.params` to both
`build_quantities` (unchanged) and the newly param-aware `aggregate`.

### A5. napari studio interaction (the scope boundary)

`BuildArea` (`studio_plugins.py`), live in the standalone `cellflow-aggregate` front-end
(`contact_analysis_studio.py`), shows a per-quantifier "built X / applicable N" badge
via `is_built(output_for_record(...))` and can trigger per-position builds. Under this
spec:

- The **canonical `run()` pipeline** persists only `contacts` and pools the rest, so it
  no longer writes the cheap per-position files. This is where Artur's "these live only
  in the pooled tables" holds.
- The **interactive studio is left as-is**: its per-position build/coverage path still
  works (the writers are retained per A2). A user who manually builds a pooled quantity
  in the studio still writes a stray per-position file. Reframing / removing that UI is
  the pending front-end-refocus work, not this spec.

So the guarantee this spec delivers is scoped to the headless pipeline. Fully removing
per-position pooled persistence from the UI is explicitly deferred. (This is the fork
Artur signed off on: keep the change surgical rather than pull the doomed studio UI into
scope.)

## Workstream B: slim the project catalog CSV

The catalog's job is to point the aggregate at each position's persisted inputs. After
Workstream A those inputs are: the position folder, its committed label images, and its
`contact_analysis.h5`. Everything else is recomputed. So the CSV columns become:

- `position_path` — the position folder (kept).
- `contact_analysis_path` — canonical `4_contact_analysis/contact_analysis.h5`
  (Workstream C), stored relative to `position_path`.
- `cell_labels` — canonical `cell_labels.tif`, the **committed** output the finalize
  button writes (`NucleusArtifactPaths.cell_labels`), stored relative.
- `nucleus_labels` — canonical `nucleus_labels.tif` (`NucleusArtifactPaths.nucleus_labels`),
  stored relative.
- `condition`, `experiment_id`, `id` — identity.
- free-form tag columns (folder-nesting levels / manual constant tags) — appended.

**Dropped end-to-end:** `path` (the old free-form contacts path — superseded by the
canonical `contact_analysis_path`), `date`, `notes`.

`date` leaves not just the catalog but the pooled-table metadata: remove it from
`shape_tables.METADATA_COLUMNS` and `_position_metadata` (it is already excluded from
the row-id identity, so this only removes an always-blank descriptor column). If a
downstream dataset wants a date axis it adds one as a free-form tag.

Changes:

- `catalog.py`: `CSV_COLUMNS` drops `path`, `date`, `notes`; `REQUIRED_CSV_COLUMNS`
  becomes `("position_path", "condition", "id")` — `position_path` is now the anchor
  every canonical path is resolved against, so it is required; the canonical
  `contact_analysis_path` / `cell_labels` / `nucleus_labels` are derivable and need not
  gate. Load stays lenient about **extra** columns (ignored) and about the dropped
  columns being **absent**, so any recent catalog (which already carries
  `position_path`) still loads and re-saves slim; a genuinely ancient CSV that predates
  `position_path` is not a supported input (no migration).
- `main_widget._catalog_record_for_position`: **fix** — stamp the committed label paths
  (`cell_labels.tif` / `nucleus_labels.tif`) and the canonical
  `4_contact_analysis/contact_analysis.h5`, replacing the current pre-commit
  `3_cell/tracked_labels.tif` / `2_nucleus/tracked_labels.tif` stamps. Reuse
  `NucleusArtifactPaths` and `CONTACT_ANALYSIS_RELPATH` so the layout knowledge lives
  in one place.

## Workstream C: rename the output folder to `4_contact_analysis`

`OUTPUT_SUBDIR` (`contact_analysis/quantifier.py`): `"aggregate_quantification"` →
`"4_contact_analysis"`. `CONTACT_ANALYSIS_RELPATH` derives from it, so it updates for
free. Repoint the three stray hardcoded literals at the constant instead of the string:

- `napari/main_widget.py:717` (`pos_dir / "aggregate_quantification" / "contact_analysis.h5"`)
- `napari/contact_analysis_widget.py:156`
- `napari/data_panel_widget.py:45`

Fix the docstrings/comments that name the old folder (`_paths.py`, `main_widget.py`
docstring, `catalog.py`, `_experiments_panel.py` ASCII tree, the quantifier module
docstrings). No data migration: pre-existing `aggregate_quantification/` folders on
disk are simply orphaned (accepted).

## Consequences and non-goals

- **Per-position provenance sidecars** for the pooled quantities disappear with their
  artifacts. This supersedes the "all quantifiers write provenance JSON" TODO item at
  the per-position grain; run-level provenance for the pooled tables, if wanted, is a
  separate follow-up and is **out of scope** here.
- **Re-aggregation now recomputes** the cheap quantities every run (no per-position
  cache). This is the intended trade (cheap to compute, per Artur); contacts stays
  cached as its h5.
- **Out of scope:** the pooled-table output format and `out_dir` behaviour (unchanged);
  the `.iris` export; the table-editing UI for the catalog (its own later spec); any
  change to how `contact_analysis.h5` itself is built or read.

## Testing

- **Seam:** `compute_object_table` for each pooled quantifier returns the same
  column-major table its old `build`→`object_table` round-trip produced (golden: build
  once the old way in a fixture, assert equality of columns/values). `contacts`
  unchanged.
- **Pooling:** `aggregate(records, params=...)` produces the same pooled tables as
  before for a fixture catalog, with **no** per-position files written (assert the
  position folder contains only `contact_analysis.h5`, no `*.csv`/`*_dynamics.h5`).
  The `required_build_params` gate still skips a position lacking pixel size.
- **`build_quantities`:** builds `contact_analysis.h5` and nothing else per position.
- **Catalog:** a saved slim CSV has exactly the new columns (no `path`/`date`/`notes`);
  round-trips (save → load → same rows); the adapter stamps `cell_labels.tif` /
  `nucleus_labels.tif` / `4_contact_analysis/contact_analysis.h5`; a fat legacy CSV
  still loads and re-saves slim.
- **Pooled metadata:** pooled tables carry no `date` column; row-id identity unchanged.
- **Rename:** `CONTACT_ANALYSIS_RELPATH` and the built h5 land under
  `4_contact_analysis/`; the three repointed literals resolve there.
- Qt-touching tests bootstrap via napari `get_qapp()` (bare `QApplication([])` aborts
  under pytest).

## Plan decomposition

One spec, but likely **two implementation plans** (decided at plan-writing time):
Plan 1 = Workstream A (the seam — the substantive engineering, self-contained test
surface); Plan 2 = Workstreams B + C (catalog slim + folder rename — the data-contract
and naming change, which depends on A having emptied the folder). Order: A before B/C.
