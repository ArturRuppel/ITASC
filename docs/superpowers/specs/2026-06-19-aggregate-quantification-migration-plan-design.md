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

All of the following are explicitly **not yet decided**:

- **CLI subcommand surface** — exact commands/flags (`discover` / `build` /
  `export`, config options, progress reporting).
- **Tidy-table artifact formats & layout** — Parquet vs CSV vs Excel specifics,
  output directory structure, naming.
- **Notebook surface** — analyst notebook(s) content, plotting library, headless
  HTML report generation.
- **napari teardown sequencing** — which files get deleted in what order once the
  cut gate is met.
- **Unique-`id` rule for discovery** — folder-name collisions once paths are
  semantically meaningless (carried over from the direction note;
  `catalog.py:215`).
- **Config-driven runner (Approach B)** — possible later thin layer over the
  orchestrator for one-file reproducibility; not in scope now.
