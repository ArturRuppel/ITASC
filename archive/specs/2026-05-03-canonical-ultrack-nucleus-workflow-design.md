# Canonical Ultrack Nucleus Workflow Design

Date: 2026-05-03

## Goal

Move the nucleus workflow back to canonical Ultrack candidate generation for
tracking quality. The main tracking path should build Ultrack's own hierarchical
candidate database from contour and foreground mask inputs, then link, solve,
export, browse, and correct that database-backed result.

The old CellFlow HDF5 hypothesis-generation path remains available in code for
experiments and compatibility, but it is no longer visible as the normal nucleus
tracking workflow.

## Evidence

The comparison experiment at:

`/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/2_nucleus/ultrack_hierarchy_vs_direct_experiment/20260503_111650`

ran both branches with the same node-probability scoring and tracking settings.

Results:

- direct H5 ingest: GT recall at IoU 0.5 = 0.7821
- canonical Ultrack hierarchy: GT recall at IoU 0.5 = 0.8447
- direct H5 ingest: nodes with hierarchy parent = 0
- canonical Ultrack hierarchy: nodes with hierarchy parent = 1892

This indicates that the canonical hierarchy/candidate-generation path provides a
better candidate universe than the direct flat H5 ingest path.

## Top-Level Workflow

The nucleus workflow UI should expose five top-level sections:

1. Contour Maps
2. Ultrack Database Generation
3. Ultrack Database Browser
4. Ultrack Tracking
5. Correction

`Ultrack Tracking` and `Correction` must be separate top-level sections, not
nested subsections inside a shared top-level container.

## Deprecated Workflow Pieces

The following are deprecated for the normal tracking path:

- HDF5 hypothesis generation UI
- HDF5 database browser UI
- `cellflow.database.hypotheses` as the primary tracking candidate source
- direct `hypotheses.h5 -> NodeDB/OverlapDB` ingest as the default tracking path

Deprecated code should not be deleted in the first migration. It should remain
available for compatibility, experiments, and old correction code while callers
are migrated. Public or likely-public entry points should warn or be clearly
marked as deprecated.

## File Contract

Canonical nucleus tracking files:

- `1_cellpose/nucleus_prob_3dt.tif`
- `1_cellpose/nucleus_dp_3dt.tif`
- `1_cellpose/nucleus_prob_zavg.tif`
- `2_nucleus/contour_maps.tif`
- `2_nucleus/foreground_masks.tif`
- `2_nucleus/ultrack_workdir/data.db`
- `2_nucleus/ultrack_workdir/metadata.toml`
- `2_nucleus/tracked_labels.tif`
- `2_nucleus/validated_cells.json`

`foreground_masks.tif` is a canonical external input. It replaces the old
`foreground_maps.tif` contract for tracking. The old `foreground_maps.tif` name
is deprecated and should not be treated as the required input for canonical
Ultrack tracking.

If contour-map generation still computes a foreground-like diagnostic internally,
that output is diagnostic only. It must not be confused with the required
external `foreground_masks.tif`.

## Shared Widget Structure

Every top-level workflow section should show:

- required inputs at the top
- load buttons beside loadable inputs
- main controls
- `Run` button
- `Run in Terminal` button
- local status label
- local progress bar
- outputs at the bottom
- load buttons beside loadable outputs

Each section owns its status/progress surface. User feedback should not depend
only on one widget-wide status label.

## Section 1: Contour Maps

Purpose: generate contour maps from Cellpose nucleus probability and flow data.

Inputs:

- `1_cellpose/nucleus_prob_3dt.tif`
- `1_cellpose/nucleus_dp_3dt.tif`

Outputs:

- `2_nucleus/contour_maps.tif`

Behavior:

- preserve the existing contour-map generation controls where practical
- expose local run and terminal-run controls
- show input/output file status directly in the section
- do not present contour generation as producing the canonical foreground mask

## Section 2: Ultrack Database Generation

Purpose: create the canonical Ultrack candidate database and links.

Inputs:

- `2_nucleus/contour_maps.tif`
- `2_nucleus/foreground_masks.tif`
- `1_cellpose/nucleus_prob_zavg.tif`

Outputs:

- `2_nucleus/ultrack_workdir/data.db`
- `2_nucleus/ultrack_workdir/metadata.toml`

Run pipeline:

1. load contour maps and foreground masks
2. call canonical `ultrack.segment(foreground_masks, contour_maps, config)`
3. run `write_seed_prior_node_probs(...)` against `nucleus_prob_zavg.tif`
4. run linking

This section includes linking because the database browser should inspect both
candidate nodes and link neighborhoods. `Ultrack Tracking` should be able to
start from an already linked database.

Primary controls:

- min area
- max area
- foreground threshold
- min frontier
- watershed hierarchy mode (`area`, `dynamics`, `volume`)
- max distance
- max neighbors
- linking mode (`default`, `iou`)
- IoU weight when applicable
- quality exponent for node-prob scoring
- solver/link transform power if it affects later solve behavior
- worker controls where they matter

## Section 3: Ultrack Database Browser

Purpose: inspect `ultrack_workdir/data.db`, not `hypotheses.h5`.

Inputs:

- `2_nucleus/ultrack_workdir/data.db`

Outputs:

- none required in the first slice

Browser modes:

- database summary
- per-frame node/link/overlap/selected counts
- selected-solution preview when solved nodes exist
- hierarchy-threshold preview before solve
- single-hierarchy preview
- node inspector
- link-neighborhood view
- conflict/overlap-neighborhood view

The browser should never paint all candidate nodes into one image without an
explicit render rule because candidate nodes intentionally overlap. Render modes
must state what subset of nodes is being painted.

Implementation notes:

- query `NodeDB` and `OverlapDB`/`LinkDB` directly
- deserialize `NodeDB.pickle` only for the current frame and active render mode
- paint masks into a buffer via `node.paint_buffer(...)` or bbox/mask pasting
- use a small cache keyed by database path, frame, render mode, and threshold

## Section 4: Ultrack Tracking

Purpose: solve a linked Ultrack candidate database and export tracked labels.

Inputs:

- linked `2_nucleus/ultrack_workdir/data.db`
- `2_nucleus/ultrack_workdir/metadata.toml`

Outputs:

- `2_nucleus/tracked_labels.tif`

Run pipeline:

1. solve the existing linked database
2. export selected tracks to `tracked_labels.tif`
3. load or offer to load the exported labels into napari

Normal tracking should not rebuild candidates. Candidate/database generation is
owned by `Ultrack Database Generation`.

### Resolve From Validated

Resolve from validated remains feasible and should be preserved.

Inputs:

- `2_nucleus/contour_maps.tif`
- `2_nucleus/foreground_masks.tif`
- `1_cellpose/nucleus_prob_zavg.tif`
- current `2_nucleus/tracked_labels.tif`
- `2_nucleus/validated_cells.json`

Output:

- updated `2_nucleus/tracked_labels.tif`

Resolve pipeline:

1. rebuild a fresh canonical Ultrack database from foreground masks and contours
2. inject validated nodes or mark validated-equivalent candidates
3. suppress or prune conflicting candidates as required
4. score nodes with `write_seed_prior_node_probs(...)`
5. link
6. solve with annotations or validated constraints
7. export labels
8. merge or preserve validated track identities according to the existing
   validated-resolve contract

This replaces the old validated-resolve source:

- old: `hypotheses.h5 -> direct ingest -> validate/resolve`
- new: `foreground_masks.tif + contour_maps.tif -> ultrack.segment -> validate/resolve`

## Section 5: Correction

Purpose: edit exported tracked labels and validation state.

Inputs:

- `2_nucleus/tracked_labels.tif`
- `2_nucleus/validated_cells.json`
- `2_nucleus/ultrack_workdir/data.db` for Extend only

Outputs:

- updated `2_nucleus/tracked_labels.tif`
- updated `2_nucleus/validated_cells.json`

Correction contains:

- correction activation and inspector
- validation controls and shortcuts
- load/save tracked labels
- reassign IDs
- extend backward/forward
- retrack backward/forward
- extend parameters
- retrack parameters

### Extend

Extend should migrate from H5 candidates to the Ultrack database.

Old source:

- `tracked_labels + hypotheses.h5`

New source:

- `tracked_labels + ultrack_workdir/data.db`

The DB-backed extend operation should:

1. read the source cell mask from the current tracked labels
2. query `NodeDB` candidates at the target frame
3. deserialize candidate node masks for that frame
4. score candidates by distance, area similarity, and mask agreement
5. paint the best candidate with the source track ID into the target frame

Extend remains a local manual edit. It should not mutate `data.db` and should
not rerun the ILP automatically.

If `data.db` is missing, Extend should show a clear local status error telling
the user to run database generation first.

### Retrack

Retrack should initially keep its current label-based behavior.

Current behavior:

- source corrected frame
- target tracked frame
- centroid-distance constrained assignment
- relabel target frame

This does not depend on `hypotheses.h5`, so it can survive the canonical DB
migration with minimal changes.

A future candidate-aware retrack mode may query `NodeDB` candidates, but that is
not part of the first migration.

## Error Handling

Each section should report missing inputs in its own status label.

Required failures:

- database generation fails if `foreground_masks.tif` is missing
- database generation fails if `contour_maps.tif` is missing
- database generation fails if `nucleus_prob_zavg.tif` is missing for scoring
- database browser fails clearly if `data.db` is missing
- tracking fails clearly if `data.db` is missing or unlinked
- resolve from validated fails clearly if tracked labels or validation data are
  missing
- extend fails clearly if `data.db` is missing

## Terminal Runs

Every run-capable top-level section must provide `Run in Terminal`.

Terminal scripts should be self-contained enough to run outside napari and must
include:

- explicit input paths
- explicit output paths
- captured parameter values
- `if __name__ == "__main__":` guard when multiprocessing may occur

## Testing

Tests should cover:

- old hypothesis-generation UI is no longer visible in the nucleus workflow
- old H5 database browser UI is no longer visible
- top-level section order and titles
- every section exposes input labels, output labels, status label, progress bar,
  run button, and terminal-run button where applicable
- database generation calls canonical `ultrack.segment`
- database generation runs node-prob scoring and linking
- database browser reads `data.db` and can render current-frame selected nodes
- tracking solves and exports from an existing linked DB
- resolve from validated no longer requires `hypotheses.h5`
- extend queries `NodeDB` instead of `hypotheses.h5`
- retrack still works without `hypotheses.h5`
- missing-input status messages are section-local

## Acceptance Criteria

- The visible nucleus workflow has exactly these top-level sections:
  `Contour Maps`, `Ultrack Database Generation`, `Ultrack Database Browser`,
  `Ultrack Tracking`, and `Correction`.
- HDF5 hypothesis generation and the old HDF5 database browser are hidden from
  the normal UI and marked deprecated in code.
- `foreground_masks.tif` is the canonical foreground input.
- `foreground_maps.tif` is treated as deprecated or diagnostic only.
- Database generation creates `ultrack_workdir/data.db` using canonical
  `ultrack.segment`.
- Database generation runs node-probability scoring and linking.
- Database browser inspects `ultrack_workdir/data.db`.
- Tracking solves an existing linked DB and exports `tracked_labels.tif`.
- Resolve from validated works from canonical segmentation inputs without
  `hypotheses.h5`.
- Extend uses candidates from `data.db`.
- Retrack remains available without `hypotheses.h5`.
- Every top-level section states required inputs, outputs, local status, local
  progress, run control, and terminal-run control.

