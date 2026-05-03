# Meta Analyzer Design Sketch

Date: 2026-05-01

## Purpose

The meta analyzer is the downstream exploration layer for CellFlow. It should assemble final analysis outputs across several tissues per experiment, several experiments per condition, and several conditions to compare.

The primary user is exploratory: fast interactive inspection, slicing, visualization, and metric discovery. The secondary use case is scripted production analysis. Sharing with collaborators matters later, so tables and plots should be exportable, but the first design should be tailored to the main exploratory workflow.

## Direction

The preferred direction is backend-first and Napari-first:

- The core model, catalog, query layer, identity resolver, and metric system should have no Napari dependency.
- Napari should be the first frontend because raw images, labels, contact overlays, spatial drilldown, screenshots, and movie exports fit naturally there.
- The design should not recreate the archived web dashboard all at once.
- A future web frontend should remain possible because the backend API and saved workspace artifacts are frontend-neutral.

## Storage Model

The meta analyzer should not duplicate the canonical analysis H5 files. It should store references to source data and load them lazily.

Two storage layers are useful:

1. Source catalog
   - Study hierarchy: condition, experiment, tissue/position.
   - Paths to canonical analysis H5 files, raw images, labels, configs, and provenance.
   - Validation of the rigid CellFlow folder contract.

2. Analyzer workspace
   - Saved filters and cohort definitions.
   - Derived tables.
   - Metric runs and cached expensive results.
   - Plot specifications.
   - Notes and export records.
   - Export support for tables as CSV and plots as PNG/SVG, with movie/GIF export for spatial overlays.

The workspace format can be decided later, likely HDF5 or SQLite plus files depending on the shape of saved artifacts.

## Study Data Model

The hierarchy should be explicit:

```text
condition -> experiment -> tissue -> frame -> object
```

Core IDs:

- `condition_id`
- `experiment_id`
- `tissue_id` or `position_id`
- `frame`
- `cell_id`
- `track_id`
- `contact_id` or sorted `(cell_a, cell_b)`
- `edge_track_id`
- `event_id`

Canonical table views:

- `tissues`: condition, experiment, tissue path, acquisition metadata, analysis status.
- `frames`: tissue, frame, time, density/quality summaries.
- `cells`: one row per cell per frame; geometry, area, shape, position, velocity, track ID.
- `contacts`: one row per junction/contact per frame; endpoint cells, length, coordinate reference, force/stress fields when available.
- `tracks`: one row per tracked cell; duration, start/end frame, displacement, summary stats.
- `edge_tracks`: one row per tracked contact/edge trajectory; duration, signed length dynamics, linked T1s.
- `events`: T1s and future topology events.
- `identity_annotations`: raw manual or imported labels with scope and provenance.
- `cells_resolved`: cells plus effective identity.
- `contacts_resolved`: contacts plus `identity_a`, `identity_b`, and contact class such as `AA`, `AB`, `BB`, or `unknown`.

Ragged data such as edge coordinates should not be forced into flat tables. Tables should carry stable keys or coordinate references, and source loaders should fetch full geometry for overlays and export.

## Identity Handling

The analyzer should consume and manage identity annotations, not classify cells from fluorescence images itself.

Identity should be represented as annotations with a resolver:

- Track-level identity is the initial common case.
- Cell-frame overrides should be possible for corrections or ambiguous cases.
- The schema should leave room for spatial or region-based identity in later experiments.

Downstream analysis should consume resolved views, so metrics do not care whether identity came from a track label, a cell-frame label, or a future spatial identity source.

## Core Modules

Proposed backend package structure:

- `cellflow.meta.catalog`: discover and index conditions, experiments, tissues, source paths, and metadata.
- `cellflow.meta.sources`: lazy loaders for analysis H5s, TIFFs, labels, and folder-contract validation.
- `cellflow.meta.tables`: normalized table views for cells, contacts, tracks, edge tracks, events, and tissues.
- `cellflow.meta.identity`: identity annotation storage, import/export, and resolution.
- `cellflow.meta.metrics`: code-defined metric registry.
- `cellflow.meta.workspace`: saved filters, derived tables, metric runs, plot specs, notes, and exports.
- `cellflow.napari.meta_widget`: thin Napari frontend over the backend APIs.

Every useful operation should be callable from Python without starting Napari.

## Incremental Build Plan

Build this slowly as usable vertical slices.

### Phase 1: Source Browser

Open a study/project hierarchy, discover analysis H5s and raw image/label files through the existing folder contract, and show basic metadata.

Napari UI:

- Select condition, experiment, tissue, and frame.
- Load raw image.
- Load labels.
- Overlay contacts/edges from analysis geometry.
- Clear overlays.
- Optionally preview cells/contacts for the current frame.

No metric registry is required in this first slice.

### Phase 2: Resolved Tables

Expose canonical lazy tables:

- `frames`
- `cells`
- `contacts`
- `tracks`
- `edge_tracks`
- `events`
- `tissue_summary`

Add identity annotation import/export and the resolver that produces `cells_resolved` and `contacts_resolved`.

### Phase 3: First Metric Module

Implement one backend metric cleanly before building a broader UI. The exact first metric is deferred. It should return object-level and tissue-level outputs and be callable from scripts.

### Phase 4: Generic Metric Workbench UI

Add a Napari dock panel for:

- selecting cohorts/slices,
- running registered metric modules,
- viewing summary tables and plots,
- drilling down to spatial evidence,
- exporting tables and figures.

### Phase 5: Metric Expansion

Add metrics one at a time, with tests, table outputs, plot presets, and export support.

Candidate metric families:

- cell morphology: area, perimeter, shape index, eccentricity;
- motion: speed, persistence, displacement summaries;
- topology: neighbor count, contact turnover, T1 participation;
- local context: density, local A/B composition, distance to heterotypic interface;
- contact properties: length, lifetime, AA/AB/BB class, T1 involvement;
- mechanics: tension, stress, pressure, ForceSys-derived values;
- tissue-level mixing/segregation summaries.

## Scientific Workflow

The first broad scientific goal is comparative metric discovery:

- Are A cells meaningfully different from B cells?
- Are differences dependent on tissue-level properties such as density?
- Are AA, AB, and BB contacts different?
- Which metrics reveal robust differences?

The metric workbench should show both:

- pooled object-level signal for discovery, and
- tissue/experiment-level aggregate signal to avoid overinterpreting repeated cells, frames, or contacts.

Metric screening should initially emphasize robust descriptive signals rather than formal p-values:

- effect size,
- uncertainty,
- sample coverage,
- missingness,
- per-tissue consistency,
- group imbalance warnings.

Formal statistical models can come later.

## Frontend Choice

Napari is favored for the first frontend because the exploratory workflow is spatial and image-centric. It should provide linked inspection from tables/plots back to raw image, labels, and contact overlays.

Known Napari risks:

- Rich sortable/filterable tables require careful Qt model/view work.
- Interactive plotting and layout state require discipline.
- The dock widget must behave like an analysis cockpit, not a thin stage-control panel.

The mitigation is to keep computation, queries, metrics, saved filters, and workspace artifacts outside Napari.

## Deferred Decisions

- Final name of the module.
- Workspace storage format: HDF5 vs SQLite plus files.
- Exact first metric module.
- Whether a future web frontend is worth building.
- Full statistics/modeling layer.
