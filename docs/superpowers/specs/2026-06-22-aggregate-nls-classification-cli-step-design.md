# Aggregate Quantification: NLS classification as a CLI engine step

**Status:** decided direction, pre-implementation (open questions listed)
**Date:** 2026-06-22
**Scope:** make headless NLS classification an optional, config-flagged pipeline
step (like every other step), and remove the interactive napari NLS UI.
**Companion:** the napari refocus
(`2026-06-22-aggregate-napari-frontend-refocus-design.md`) that deletes the UI side.

## Statement of need

NLS classification produces the per-cell `class_label` (nuclear marker positive /
negative) that the analysis depends on (it is what split the COV2D dataset). The
classification logic is **already fully headless** — the only thing the interactive
napari plugin uniquely does is *trigger* it and pick a threshold by eye. For the
CLI engine to be self-sufficient (and for the napari layer to become a thin
front-end), classification must run as a config-driven step, not a hand-operated
widget.

## What we already have (survives)

- `contacts/nls_classification.py` is the whole headless engine:
  `measure_track_nls_intensity` → a threshold (`auto_threshold`,
  `split_tracks_otsu`, `split_tracks_two_clusters`) → `classify_by_threshold` →
  `classify_position_nls_to_csv` writes the per-position **sidecar CSV** at
  `nls_classification_csv_path(position_path)`.
- The **join already happens automatically**: at aggregate time,
  `shape_tables.py:211` left-joins each position's `{cell_id: class_label}` by
  `cell_id` onto every cell-keyed table **whenever the sidecar exists**. So nothing
  downstream needs to change — the missing piece is only *who writes the sidecar*.

## Decisions

### 1. A distinct optional pipeline step, not a quantifier.

Classification produces a sidecar that is *joined onto* the measurement tables, not
a measurement table of its own — so it is not a `Quantifier`. It is a new step that
runs **before `aggregate`** (so the join finds fresh sidecars), gated by config.
Add `classify(catalog, *, params)` to `pipeline.py` (iterates positions, calls
`classify_position_nls_to_csv` per position) and call it from `run()` when enabled.

### 2. Config-gated, like every other step.

`RunConfig` gains an optional `[nls]` table:

```toml
[nls]
enabled = true
image   = "0_input/NLS_zavg.tif"   # per-position relative path (absolute also ok)
method  = "auto"                    # auto | otsu | two_cluster | fixed
threshold = 0.0                     # used only when method = "fixed"
```

When `enabled` is false / the table absent, the step is skipped (positions that
already have a sidecar still get their `class_label` joined — unchanged behaviour).

### 3. Per-position thresholding (default), method-selectable.

Default `method = "auto"` thresholds **per position** (current behaviour, robust to
per-position intensity drift). `otsu` / `two_cluster` expose the existing splitters;
`fixed` pins one threshold across the series. (Global auto-threshold across the
series is Open #1.)

### 4. Remove the interactive napari NLS UI.

Delete `napari/aggregate_quantification/plugins/nls_classification.py` and its
wiring. The TODO "NLS Classifier" batch features are subsumed by the config step
(one config classifies the whole catalog). The headless engine module is untouched.

## Open / deferred

1. **Global vs per-position threshold** — a single series-wide auto threshold
   (pooling all tracks) vs the per-position default. Per-position is safer; expose
   global later if needed.
2. **Where the sidecar lives** — keep `nls_classification_csv_path(position_path)`
   (per-position, beside the inputs) unchanged. Revisit only if curation/exclusion
   wants classification centralized.
3. **Multi-class / >2 labels** — current engine is binary (positive/negative); not
   broadened here.
4. **Provenance** — the step should stamp how each sidecar was produced (method,
   threshold, image) for reproducibility; align with the quantifier provenance-JSON
   TODO item.
