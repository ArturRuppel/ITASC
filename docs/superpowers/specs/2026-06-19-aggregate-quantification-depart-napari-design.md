# Aggregate Quantification: depart napari for a backend + artifacts framework

**Status:** decided direction, pre-implementation (one decision deferred — see "Open / deferred")
**Date:** 2026-06-19
**Scope:** retire the napari UI for `cellflow-aggregate`; keep and surface the
backend through tidy artifacts + notebooks.

## Statement of need

Aggregate Quantification is cross-position, batch statistical work (pooling,
SuperPlots, MSD/density stats). napari's image canvas adds nothing to it. The
interactive image-linked affordances that justified putting it in napari were a
gimmick, and — decisively — **no on-image editing is required** here, only
inspection. Meanwhile the real requirement is that **non-technical users can
export and visualize their data in-house or with other tools** (Excel, GraphPad
Prism, R, Iris). A ~7,000-line Qt UI is the wrong surface for all of that.

This note records the decisions already settled in discussion so they're pinned
before any code moves.

## What we already have (the part that survives)

- **The compute backend is already napari-free.** Everything in
  `src/cellflow/aggregate_quantification/` has zero real napari imports — the only
  "napari" hits are docstrings/comments. The Qt code lives entirely separately in
  `src/cellflow/napari/aggregate_quantification*` (~7k lines, ~22 files;
  `plot_panel.py` alone is 1,759 lines), and is a thin shell over backend
  functions.
- **Discovery already exists and is dumb-by-design.** `discover_catalog_entries`
  (`catalog.py:178`) is pure `pathlib.rglob` by file presence, groups matches per
  position folder, and correctly returns **no** experimental metadata.
- **The catalog is a plain CSV** (`load_catalog` / `save_catalog`,
  position-folder-anchored with relative file paths) — already a good non-technical
  handoff artifact.
- **Positions are self-describing about provenance.** `contacts/build.py:101-110`
  stamps a `provenance` group into `contact_analysis.h5` (`source_position_path`,
  label paths, params, `created_at`, version).

## Decisions

### 1. Retire the napari UI; do not replace it 1:1.

Delete the `cellflow.napari.aggregate_quantification*` layer. The interactive
image↔plot plugins (`_click_to_load`, `visualize_contacts`, `nls_classification`)
are dropped — there is no editing requirement, only inspection, and inspection
ports cleanly off the Qt canvas. (If on-image *inspection* is ever wanted back, it
maps to matplotlib + an `ipywidgets` frame slider for overlays, and `bokeh` linked
selection for "click a cell → highlight in plot". Not built now.)

### 2. The foundation is an artifact contract, not a viewer.

Jupyter is **one consumer, not the foundation** — it is the wrong surface for
non-technical users (needs an env, a kernel, run-cells-in-order). The product for
those users is **clean, tool-agnostic files**:

- **Tidy tables** in open formats — Parquet for size/typing, plus CSV/Excel for
  the Prism/Excel/Tableau crowd. One row per observation, indexed by
  position/frame/cell.
- **`.iris` bundles** (already in flight; see
  `2026-06-17-iris-export-design.md`) — the premade-SuperPlot, no-code visualize
  path. This is the same outward direction and validates it.

If this layer is clean, the non-technical requirement is met regardless of what
else exists.

### 3. Notebooks are the analyst surface (only).

Jupyter is for you/the analyst: exploration, new quantifiers, ad-hoc plots.
Notebooks also run **headless** (papermill / nbconvert) to emit a self-contained
**HTML report** — shareable, no kernel — covering "send a collaborator the
results" without making them touch Python.

### 4. Defer the self-serve dashboard.

A web app (Voilà / Streamlit / Panel) is justified **only if** non-technical users
need to slice the data interactively themselves. Current need is "good files +
premade plots they open elsewhere," so this is **not built** speculatively. If it
ever is, Voilà is preferred (reuses notebook code, lowest marginal cost).

### 5. Discovery and annotation are separate concerns.

The path **must not** encode experimental structure (which folder is which
condition) — coupling semantics to storage layout is fragile and breaks on
reorg. Therefore:

- **Discovery** stays a dumb file-presence `rglob` (what `discover_catalog_entries`
  already is). It locates positions; it does not interpret them.
- **Annotation** (`condition` / `date`) comes from the **editable catalog CSV** —
  the only place that information has ever lived (it was hand-typed in the napari
  table; nothing in the pipeline writes it to disk; downstream it is catalog-level
  metadata broadcast onto rows, `shape_tables.py:60`, `plotting.py:130`). The build
  flow:

  ```python
  entries = discover_catalog_entries(root, cell_name="cell_labels.tif",
                                      nucleus_name="nucleus_labels.tif")
  save_catalog("catalog.csv", entries)   # skeleton: blank condition/date
  # → human fills condition/date in Excel (or a LIMS export) →
  catalog = load_catalog("catalog.csv")  # source of truth for the run
  ```

  Path-template discovery (deriving metadata from path segments) is explicitly
  **rejected** for this reason.

## Open / deferred

- **Unique `id` rule for discovery (deferred decision).** Discovery currently sets
  `id = position_dir.name` (`catalog.py:215`). Once paths stop encoding structure,
  folder names can collide (many `Pos0`, flat UUIDs). `position_id` is a grouping
  key downstream (pooling, SuperPlots), so a collision silently merges two
  positions' cells. The catalog anchors safely on the absolute `position_path`, but
  `id` does not. Likely fix: default `id` to the position path relative to `root`,
  human-overridable in the CSV — but this is **not yet decided**.
- The public backend API surface (`aggregate_quantification/__init__.py` currently
  exposes nothing) needs defining so notebooks call a stable
  `load → quantify → reduce → plot/export` entry set. Not specified here.
