# Aggregate Quantification: napari → backend migration plan

**Status:** partial design — settled decisions recorded; remaining decisions
deferred (see "Deferred").
**Date:** 2026-06-19
**Scope:** the migration *plan* for moving `cellflow-aggregate` off napari onto a
backend orchestrator driven by CLI + Python API.
**Companion:** builds on the direction note
`2026-06-19-aggregate-quantification-depart-napari-design.md` and the
`2026-06-17-iris-export-design.md`.

## Settled decisions

1. **Full migration, one phased spec.** The departure (backend API, discovery,
   notebooks/artifacts, napari teardown) is treated as a single migration with
   phases, not separate products.
2. **Parallel-run, then cut.** Build the backend orchestrator + CLI/API +
   notebooks/exports to the cut gate *while napari still works*; delete the Qt
   aggregate UI only once the new path is proven.
3. **Interface: CLI + Python API.** A `cellflow-aggregate` CLI for
   reproducible / non-technical runs, backed by the same Python functions
   notebooks import. (CLI subcommand surface itself is deferred.)
4. **Cut gate = artifact parity.** Deletion is allowed once a real dataset runs
   end-to-end via CLI/API and produces correct tidy tables + `.iris` bundles.
   Notebook plotting only needs to be "good enough", not a 1:1 reproduction of
   every napari `plot_panel` chart.

## Chosen architecture (Approach A — approved)

A napari-free orchestrator that the CLI, notebooks, and (during parallel-run) the
napari studio all drive.

### New module: `aggregate_quantification/pipeline.py`

The single orchestration surface — four composable, napari-free functions:

```python
def build_catalog(root, *, cell_name, nucleus_name, out_csv=None) -> list[dict]
    # wraps discover_catalog_entries + save_catalog (skeleton CSV)

def build_quantities(catalog, *, quantifiers=None, params=None, progress_cb=None) -> None
    # THE extracted loop: for each record × each registered quantifier that
    # can_build, call .build(inputs, output_path, ...). This is the only
    # orchestration logic currently trapped in aggregate_quantification_studio.py.

def aggregate(catalog, out_dir=None) -> dict[str, Path]
    # thin pass-through to shape_tables.aggregate (already headless)

def export(tables_dir, out_dir=None, *, formats=("parquet", "csv")) -> list[Path]
    # writes tidy artifacts + delegates to iris_export.export_dir for .iris
```

### Public API (`aggregate_quantification/__init__.py`)

Currently empty. It gains a curated re-export of exactly the four pipeline
functions plus `load_catalog` / `save_catalog` and `available_quantifiers`. That
is the stable surface notebooks import and the CLI calls; nothing else is public.

### What does NOT move

`catalog.py`, `shape_tables.py`, `reduce.py`, `quantifier.py`, and `iris_export/`
stay put — `pipeline.py` *composes* them. Confirmed during exploration that the
stages already exist headless:

- discovery — `discover_catalog_entries` / `load_catalog` / `save_catalog`
  (`catalog.py`)
- per-position build — `Quantifier.build(inputs, output_path, *, params,
  progress_cb)` over `available_quantifiers()` (`quantifier.py`)
- aggregate — `shape_tables.aggregate(records, out_dir)` (registry-driven, pools
  every table across the catalog, writes CSVs)
- reduce — `reduce.run_pipeline(df, ReduceSpec)`
- export — `iris_export.export_dir(...)` (already a CLI + tidy-CSV → `.iris`)

The **only** orchestration genuinely trapped in the napari studio is the
per-position **build loop** (catalog record × quantifier `.build()`). Extraction
is essentially lifting that loop into `build_quantities`.

### Dependency direction

`pipeline` → (`catalog`, `quantifier`, `shape_tables`, `reduce`, `iris_export`).
During parallel-run the napari studio is refactored to depend on `pipeline`
instead of containing the build loop — this proves the extraction is complete
before the Qt layer is deleted.

## Deferred

Most of the items below are now settled in
`2026-06-21-aggregate-quantification-artifact-contract-design.md` (the
three-artifact contract, identity model, curated/derived split, and the
registry-driven orchestration). Updated status:

- **Tidy-table artifact formats** — **resolved:** Parquet + CSV default, Excel on
  demand. (Output directory *layout* still open.)
- **Unique-`id` rule for discovery** — **resolved:** compositional identity
  `(experiment_id, condition, position_id)` validated unique at load, plus a
  deterministic row `id`. Replaces the folder-name default (`catalog.py:215`).
- **CLI subcommand surface** — **resolved in shape:** config-in / table-out
  (`discover` / `build` / `aggregate` / `export`). Exact flags / progress
  reporting still open.
- **Notebook surface** — **partly resolved:** split into an interactive
  QC/curation notebook (writes the curation artifact) and a headless report
  notebook (papermill → HTML). Plotting library / content still open.
- **napari teardown sequencing** — still deferred; gated on the cut gate.
- **Config-driven runner (Approach B)** — out of scope.

New first-class implementation task identified during design: the build loop needs
a **dependency scheduler**. `build_quantities` plans all jobs up front, so on a
cold run the contacts-*derived* quantifiers are silently skipped (their producer
`contact_analysis.h5` does not exist yet). The command must topologically order by
`requires`/`produces` and re-derive inputs between waves. See the artifact-contract
spec, §6.
