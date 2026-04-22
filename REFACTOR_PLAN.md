# CellFlow Refactor Implementation Plan

This plan translates [REFACTOR_SPECS.md](/home/aruppel/Projects/CellFlow/REFACTOR_SPECS.md) into concrete implementation work.

## Planning Decisions

- `nucleus_ultrack` and `cell_ultrack` will both use the new shared IoU-capable linking abstraction.
- Nuclei will be the first production consumer of the IoU linker.
- The IoU linker will be developed on a separate branch in parallel, then merged before the new `nucleus_ultrack` stage is integrated.
- The current centroid-based 3D-to-2D projection remains the default projection algorithm for both workflows in this refactor.
- Local Cellpose execution is removed entirely from the production workflow. The plugin keeps only an informational `cellpose_cluster` stage with validation.

## Proposed Branching

### Main branch

Owns:

- stage taxonomy and path/layout migration
- direct hypothesis ingestion backend
- new nucleus and cell workflow stages
- Napari workflow rewrite
- final integration and verification

### Separate branch

Branch name:

- `feature/ultrack-iou-linking`

Owns:

- reusable Ultrack-side linking abstraction
- IoU-oriented linking implementation
- dedicated tests and debugging around linking behavior

Should avoid editing until merge time:

- `packages/napari-plugin/**`
- `pipeline_schema.json`
- `packages/napari-plugin/src/cellflow/napari/_plugin.py`
- other stage-taxonomy files likely to churn on main

Primary files on this branch:

- `packages/ultrack/src/cellflow/ultrack/config.py`
- `packages/ultrack/src/cellflow/ultrack/stages/tracking.py`
- likely new `packages/ultrack/src/cellflow/ultrack/linking.py`
- tests extending `tests/test_label_tracking.py`
- tests extending `tests/test_staged_pipeline.py`

Reference implementations to mine but not couple to directly:

- `packages/analysis/src/cellflow/backend/tracking.py`
- `packages/analysis/src/cellflow/backend/segmentation.py`

Merge point:

- after the core stage-taxonomy/path cleanup lands on main
- before the new `nucleus_ultrack` stage is implemented on main

## Dependency Order

1. Task A: stage taxonomy and bootstrap migration
2. In parallel:
   - Task B: direct hypothesis ingestion backend
   - Task C: IoU-linking branch
3. Merge Task C into main
4. Task D: `nucleus_ultrack`
5. Task E: `correction` handoff and load/save updates
6. Task F: `cell_ultrack`
7. Task G: UI integration sweep
8. Task H: verification and fresh-project smoke testing

## Task Board

### Task A: Stage Taxonomy And Project Bootstrap

Status:

- complete
- implemented in commit `eab882e`

Goal:

- replace the old stage list and directory layout with the new canonical workflow

Deliverables:

- authoritative stage keys:
  - `raw_import`
  - `cellpose_cluster`
  - `nucleus_ultrack`
  - `correction`
  - `cell_ultrack`
  - `analysis`
- authoritative directory mappings:
  - `raw_import -> 0_input`
  - `cellpose_cluster -> 1_cellpose`
  - `nucleus_ultrack -> 2_nucleus_ultrack`
  - `correction -> 3_correction`
  - `cell_ultrack -> 4_cell_ultrack`
  - `analysis -> 5_analysis`
- removal of legacy top-level stages from schema and UI:
  - `cellpose_nucleus`
  - `cellpose_cell`
  - `contours`
  - `tracking`
  - `project2d`
  - `cell_labels`
  - `seeded_watershed`
  - `cell_segmentation`

Primary files:

- `packages/core/src/cellflow/core/paths.py`
- `pipeline_schema.json`
- `packages/napari-plugin/src/cellflow/napari/_plugin.py`
- `packages/napari-plugin/src/cellflow/napari/new_project_dialog.py`
- `packages/napari-plugin/src/cellflow/napari/project_panel.py`
- `packages/core/src/cellflow/core/schema.py`
- `tests/core/test_core.py`

Notes:

- this is the main blocker because nearly every downstream widget and stage wrapper resolves paths through this layer
- after this lands, the rest of the refactor can work against the correct stage names and directories
- completed: the repo now uses the new six-stage contract and directory layout

### Task B: Shared Direct Hypothesis Ingestion Backend

Goal:

- add the shared backend path that ingests explicit label hypotheses directly into Ultrack without going through `foreground.tif` and `contours.tif`

Deliverables:

- shared ingestion API for per-timepoint label hypotheses
- persisted `hypotheses_manifest.json`
- persisted `labelmaps/labelmap_*.tif`
- reusable path for both nucleus and cell workflows

Primary files:

- `packages/ultrack/src/cellflow/ultrack/stages/tracking.py`
- `packages/ultrack/src/cellflow/ultrack/config.py`
- likely new `packages/ultrack/src/cellflow/ultrack/ingestion.py`
- likely new tests under `tests/`

Upstream logic to extract and reuse:

- `packages/cellpose/src/cellflow/cellpose/stages/contours.py`
- `packages/cellpose/src/cellflow/cellpose/stages/seeded_watershed.py`

Notes:

- this is the core backend refactor target from the spec
- this task should be kept separate from Napari work

### Task C: Shared IoU Linking Abstraction

Branch:

- `feature/ultrack-iou-linking`

Goal:

- develop and harden the IoU-capable linker early, then merge it in before `nucleus_ultrack`

Deliverables:

- configurable linking abstraction in `ultrack`
- `linking_mode` or equivalent workflow-level configuration
- IoU-oriented linking implementation suitable for nuclei first and cells later
- tests for:
  - stable one-to-one matches
  - unmatched objects
  - large motion rejection
  - area-change rejection
  - empty frames
  - split/merge edge cases

Primary files:

- `packages/ultrack/src/cellflow/ultrack/config.py`
- `packages/ultrack/src/cellflow/ultrack/stages/tracking.py`
- likely new `packages/ultrack/src/cellflow/ultrack/linking.py`
- `tests/test_label_tracking.py`
- `tests/test_staged_pipeline.py`

Reference code:

- `packages/analysis/src/cellflow/backend/tracking.py`
- `packages/analysis/src/cellflow/backend/segmentation.py`

Notes:

- this item is intentionally separate because it will likely need focused debugging and iteration
- nuclei should be the first integration target after merge

### Task D: Raw Import Contract Migration

Goal:

- make `raw_import` fully own the new flat `0_input/` contract, including z-averaged outputs

Deliverables:

- write:
  - `0_input/nucleus_4d.tif`
  - `0_input/cell_4d.tif`
  - `0_input/nucleus_zavg.tif`
  - `0_input/cell_zavg.tif`
  - `0_input/z_shift.csv`
- downstream code stops depending on nested `0_input/nucleus/` and `0_input/cell/`

Primary files:

- `packages/cellpose/src/cellflow/cellpose/stages/raw_import.py`
- `packages/napari-plugin/src/cellflow/napari/ultrack_widgets/data_prep.py`
- `packages/napari-plugin/src/cellflow/napari/tracking_correction_widget.py`
- `tests/test_raw_import_zshift.py`

Notes:

- some of this already exists, but validation and UI file-status surfaces still need to be aligned to the new contract

### Task E: Informational `cellpose_cluster` Stage

Goal:

- replace local Cellpose execution with an informational and validation-only cluster stage

Deliverables:

- stage UI explains:
  - what runs externally
  - which `0_input/` files are consumed
  - which `1_cellpose/` files must exist
- validation uses the stage config as the authoritative expected file contract
- all local run controls are removed from the UI
- all local execution paths are removed from the backend for this workflow

Primary files:

- `packages/cellpose/pyproject.toml`
- `packages/cellpose/src/cellflow/cellpose/config.py`
- likely new `packages/cellpose/src/cellflow/cellpose/stages/cellpose_cluster.py`
- retire or remove production usage of:
  - `packages/cellpose/src/cellflow/cellpose/stages/nucleus_3d.py`
  - `packages/cellpose/src/cellflow/cellpose/stages/cell_2d.py`
- `packages/napari-plugin/src/cellflow/napari/ultrack_widgets/cellpose.py`
- `packages/napari-plugin/src/cellflow/napari/analysis_widget.py`
- `packages/napari-plugin/src/cellflow/napari/project_panel.py`

Notes:

- this is mostly a contract and UI simplification task after Task A

### Task F: New `nucleus_ultrack` Workflow

Goal:

- implement the nucleus workflow on top of direct hypothesis ingestion plus the merged IoU linker

Inputs:

- `1_cellpose/nucleus_dp.tif`
- `1_cellpose/nucleus_prob.tif`
- `1_cellpose/nucleus_dp_zavg.tif`
- `1_cellpose/nucleus_prob_zavg.tif`

Outputs:

- `2_nucleus_ultrack/data.db`
- `2_nucleus_ultrack/tracks.csv`
- `2_nucleus_ultrack/tracked_labels.tif`
- `2_nucleus_ultrack/nuclear_labels_2d.tif`
- `2_nucleus_ultrack/hypotheses_manifest.json`
- `2_nucleus_ultrack/labelmaps/labelmap_*.tif`

Primary files:

- `packages/ultrack/src/cellflow/ultrack/config.py`
- `packages/ultrack/src/cellflow/ultrack/stages/tracking.py`
- `packages/ultrack/src/cellflow/ultrack/stages/project2d.py`
- likely new `packages/ultrack/src/cellflow/ultrack/stages/nucleus_ultrack.py`
- `packages/ultrack/pyproject.toml`
- `packages/napari-plugin/src/cellflow/napari/ultrack_widgets/ultrack_widget.py`
- `packages/napari-plugin/src/cellflow/napari/project_panel.py`
- `packages/napari-plugin/src/cellflow/napari/tracking_correction_widget.py`

Notes:

- reuse the current Cellpose mask-generation logic from `contours.py`
- do not keep contour-driven Ultrack watershed in the canonical path
- this task depends on Task B and merged Task C

### Task G: Correction Workflow Handoff

Goal:

- keep correction as the explicit bridge between nucleus and cell workflows

Inputs:

- `2_nucleus_ultrack/nuclear_labels_2d.tif`

Outputs:

- `3_correction/nuclear_labels_corrected.tif`

Primary files:

- `packages/napari-plugin/src/cellflow/napari/tracking_correction_widget.py`
- `packages/napari-plugin/src/cellflow/napari/correction_widget.py`

Notes:

- correction tooling itself should stay mostly intact
- the load/save locations and file-status wiring need to follow the new stage names

### Task H: New `cell_ultrack` Workflow

Goal:

- replace terminal `cell_segmentation` output with a second Ultrack workflow driven by corrected nuclei and Cellpose cell flow/prob maps

Inputs:

- `3_correction/nuclear_labels_corrected.tif`
- `1_cellpose/cell_dp.tif` or `1_cellpose/cell_dp_zavg.tif`
- `1_cellpose/cell_prob.tif` or `1_cellpose/cell_prob_zavg.tif`

Outputs:

- `4_cell_ultrack/data.db`
- `4_cell_ultrack/tracks.csv`
- `4_cell_ultrack/tracked_labels.tif`
- `4_cell_ultrack/cell_labels_2d.tif`
- `4_cell_ultrack/hypotheses_manifest.json`
- `4_cell_ultrack/labelmaps/labelmap_*.tif`

Primary files:

- `packages/cellpose/src/cellflow/cellpose/stages/seeded_watershed.py`
- retire or replace production usage of `packages/cellpose/src/cellflow/cellpose/stages/cell_segmentation.py`
- `packages/ultrack/src/cellflow/ultrack/config.py`
- likely new `packages/ultrack/src/cellflow/ultrack/stages/cell_ultrack.py`
- `packages/ultrack/src/cellflow/ultrack/stages/project2d.py`
- `packages/ultrack/pyproject.toml`
- `packages/napari-plugin/src/cellflow/napari/ultrack_widgets/seeded_watershed.py`
- `packages/napari-plugin/src/cellflow/napari/ultrack_widgets/cell_segmentation.py`
- `packages/napari-plugin/src/cellflow/napari/project_panel.py`
- `packages/napari-plugin/src/cellflow/napari/tracking_correction_widget.py`

Notes:

- seeded watershed becomes a hypothesis generator, not a terminal output stage
- cell workflow must not depend on the nucleus Ultrack database

### Task I: Generic Projection Utility

Goal:

- make the projection utility generic for both workflows while preserving current behavior

Deliverables:

- generic projection writer for:
  - `nuclear_labels_2d.tif`
  - `cell_labels_2d.tif`
- same centroid-based conflict resolver as today

Primary files:

- `packages/ultrack/src/cellflow/ultrack/stages/project2d.py`
- callers in new nucleus and cell stages

Notes:

- this can be developed alongside Tasks F and H, but should be owned by one backend implementer to avoid churn

### Task J: Final UI Integration Sweep

Goal:

- make the user-facing workflow read exactly as:
  - Data Prep
  - Cluster Cellpose
  - Nucleus Ultrack
  - Nuclear Correction
  - Cell Ultrack
  - Analysis

Primary files:

- `packages/napari-plugin/src/cellflow/napari/analysis_widget.py`
- `packages/napari-plugin/src/cellflow/napari/_plugin.py`
- `packages/napari-plugin/src/cellflow/napari/project_panel.py`
- `packages/napari-plugin/src/cellflow/napari/new_project_dialog.py`
- `packages/napari-plugin/src/cellflow/napari/widgets.py`

Notes:

- this should be one owner near the end because these files are likely merge hotspots

### Task K: Verification And Acceptance

Goal:

- verify that a fresh project exposes the new stage order, new paths, and new workflow behavior end to end

Primary files:

- `tests/core/test_core.py`
- `tests/test_raw_import_zshift.py`
- new tests for direct ingestion
- new tests for IoU linking integration
- new tests for stage-1 artifact validation
- any new integration tests around nucleus and cell workflow outputs

Verification targets:

- fresh project writes the new stage list and directory mappings
- raw import writes the flat `0_input/` artifacts including z-averaged stacks
- cluster Cellpose stage validates external outputs without local inference
- nucleus workflow produces `tracked_labels.tif` and `nuclear_labels_2d.tif` without contour-driven watershed
- nucleus workflow persists hypothesis labelmaps
- correction loads projected nuclei and saves corrected nuclei
- cell workflow consumes corrected nuclei plus cell flow/prob artifacts and writes `cell_labels_2d.tif`
- both workflows use the same direct-ingestion mechanism
- both workflows use the same projection utility

## Parallelization Map

### Parallel wave 1

- Task A only

Reason:

- it rewrites the vocabulary and directory layout for the whole project

### Parallel wave 2

- Task B on main
- Task C on `feature/ultrack-iou-linking`
- Task D can start once Task A lands
- Task E can start once Task A lands

### Parallel wave 3

- merge Task C
- Task F
- Task G

### Parallel wave 4

- Task H
- Task I

### Parallel wave 5

- Task J
- Task K

## Merge Hotspots To Keep Single-Owned

- `packages/napari-plugin/src/cellflow/napari/analysis_widget.py`
- `packages/napari-plugin/src/cellflow/napari/project_panel.py`
- `packages/napari-plugin/src/cellflow/napari/tracking_correction_widget.py`
- `packages/napari-plugin/src/cellflow/napari/_plugin.py`
- `packages/cellpose/src/cellflow/cellpose/config.py`
- `packages/ultrack/src/cellflow/ultrack/config.py`
- `packages/cellpose/pyproject.toml`
- `packages/ultrack/pyproject.toml`

## Immediate Next Steps

1. Task A is complete.
2. Task B is now wired into `nucleus_ultrack`, so the next implementation target is Task J: the UI integration sweep.
3. Start the UI work by rewriting the Cluster Cellpose widget into an informational/validation-only panel.
4. Then split or simplify the Ultrack widgets so nucleus and cell workflows are distinct and match the new stage layout.
5. Keep Task C on the separate `feature/ultrack-iou-linking` branch and merge it only when the UI/backend work needs the shared linker.
