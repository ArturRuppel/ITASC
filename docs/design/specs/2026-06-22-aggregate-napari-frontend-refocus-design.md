# Aggregate Quantification: napari front-end refocus

**Status:** decided direction, pre-implementation (open questions listed)
**Date:** 2026-06-22
**Scope:** refocus the napari Aggregate Quantification layer from an interactive
studio into a thin **front-end + curator** for the CLI engine — remove all
in-napari plotting; keep and refocus discover&add + run; host the curation tool.
**Supersedes:** the full teardown in
`2026-06-19-aggregate-quantification-depart-napari-design.md`. That spec's premise
("batch stats need no image canvas") still holds for *plotting*; it is overturned
only for *curation*, which is image-linked (see the curation specs). The decision is
therefore **keep napari, but as a front-end, not a studio** — everything in one
place for the user, with Iris owning visualization.

## Statement of need

The CLI engine (config-driven `run()`, Iris-only export, the analysis subpackage) is
now the canonical compute + visualization path; a real dataset (COV2D) runs
end-to-end through it. The ~7k-line napari studio is mostly an interactive plotting
shell that now duplicates — and lags — the engine. Non-technical users still need a
GUI to *drive* the engine and to *curate*, but not to plot. So napari keeps the two
jobs the canvas earns (drive + curate) and sheds the rest.

## What survives / what goes

**Survives (refocused):**
- **Discover & add** — the existing standalone discovery panel
  (`aggregate_quantification_widget.py`) over `build_catalog` / `load_catalog` /
  `merge_catalog_records`: pick a root + file names → discovered positions → catalog
  CSV, with `condition` / `date` editable in the table (the only place that metadata
  lives).
- **Run** — drives the pipeline (`build_quantities` is already called from the
  studio; refocus to the full `run()` path: classify? → build → aggregate → export →
  optional static-figure render), with progress via the existing `progress_cb`.
- **Curation tool** — added here (own spec).

**Goes (deleted):**
- All in-napari plotting: `aggregate_quantification/plots/` (`shape.py`,
  `dynamics.py`, `contacts.py`, `_pool_plot.py`, `_pooling.py`),
  `plot_panel.py` (~1,759 lines), `dynamics_curves_panel.py`, `shape_editor.py`,
  the plot docks (`plugins/_plot_dock.py`), and `plugins/_click_to_load.py`.
- The interactive **NLS classification** UI (moved to a CLI step — own spec).
- **Contact visualization** is *not* deleted but **repurposed** as the curation
  display (own spec), not a standalone plotting feature.

## Decisions

### 1. napari = front-end (discover&add + run) + curator. No plotting.

Visualization is Iris only: the `.iris` bundles + the static SuperPlot/figure render
(`iris_export.figures`, the `[plots].render` flag) + the analysis-subpackage figures.

### 2. The run front-end is config-driven.

The widget gathers inputs (root, file names, catalog metadata, selected quantities,
export dir, `[nls]`, `[plots].render`) and runs the engine. It should **write a
`config.toml`** alongside the catalog so a GUI run is reproducible on the CLI (and
vice versa) — the GUI and CLI are two faces of the same `run()`.

### 3. Reuse, don't rewrite, the discovery panel.

The discovery UI already exists and is sound; refocus rather than rebuild. Strip its
coupling to the deleted plot docks.

## Components (after refocus)

- `AggregateQuantificationWidget` — slimmed: discovery section + catalog table +
  quantity selection + run/progress + open-output; hosts the curation tool.
- The curation widget (own spec).
- No plot panels, no NLS widget.

## Open / deferred

1. **Build config from UI vs point at an existing `config.toml`** — the widget could
   author the config from form fields, or let the user pick a hand-written one. Lean:
   author it (non-technical users), with the file written for reproducibility.
2. **`cellflow-aggregate` napari manifest** — the standalone wheel still registers a
   napari widget; confirm the refocused widget is what it resolves to, and that
   nothing pulls a deleted plot module.
3. **Teardown sequencing** — delete the plot layer only after the curation tool +
   run front-end are proven (the depart-napari "cut gate" discipline), to avoid a
   window with no working GUI.
4. **Tests** — the deleted plot panels have tests; identify which to drop vs. which
   cover backend plotting that moved to Iris and should be rehomed.
