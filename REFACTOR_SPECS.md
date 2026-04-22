# CellFlow Refactor Specs

## Goal

Refactor the plugin around the new production workflow:

1. data prep writes the new 4D input stacks and the z-averaged raw stacks;
2. Cellpose flow/prob generation happens outside the plugin on a cluster;
3. nuclei go through an Ultrack pass first;
4. the projected 2D nuclei are corrected manually;
5. corrected nuclei seed a second Ultrack pass for cells;
6. both Ultrack passes project 3D results back to 2D.

This document is the implementation target for the next two phases: planning and coding.

## Core Decisions

- The plugin removes local Cellpose execution entirely from both the UI and backend.
- The plugin consumes cluster-produced Cellpose flow/prob outputs as stage inputs.
- The plugin keeps only a descriptive UI panel for the cluster Cellpose step: what runs externally, which inputs it expects, and which outputs it must produce.
- The nuclei Ultrack step no longer goes through `foreground.tif` and `contours.tif` as required pipeline artifacts.
- Ultrack watershed-based segmentation must be bypassed for both nucleus and cell workflows whenever explicit segmentation hypotheses already exist.
- Manual correction remains a 2D step between the nucleus and cell workflows.
- The current 3D-to-2D centroid-based conflict resolver remains the default projection algorithm for both nuclei and cells.
- Hypothesis labelmaps are required persisted artifacts for both nucleus and cell workflows.

## Out Of Scope

- Cluster job submission, scheduling, or monitoring.
- Finalizing the exact cell-linking cost beyond requiring an IoU-based linking path.
- Designing a new cell-specific 2D projection algorithm in this refactor.
- Migration support for legacy project layouts.

## Target Pipeline

### Stage order

1. `raw_import`
2. `cellpose_cluster`
3. `nucleus_ultrack`
4. `correction`
5. `cell_ultrack`
6. `analysis`

### Target directory layout

```text
<project_root>/
├── project.json
├── pipeline_schema.json
└── pos00/
    ├── pipeline_manifest.json
    ├── pipeline.log
    ├── 0_input/
    │   ├── nucleus_4d.tif
    │   ├── cell_4d.tif
    │   ├── nucleus_zavg.tif
    │   ├── cell_zavg.tif
    │   └── z_shift.csv
    ├── 1_cellpose/
    │   ├── nucleus_dp.tif
    │   ├── nucleus_prob.tif
    │   ├── nucleus_dp_zavg.tif
    │   ├── nucleus_prob_zavg.tif
    │   ├── cell_dp.tif
    │   ├── cell_prob.tif
    │   ├── cell_dp_zavg.tif
    │   └── cell_prob_zavg.tif
    ├── 2_nucleus_ultrack/
    │   ├── data.db
    │   ├── tracks.csv
    │   ├── tracked_labels.tif
    │   ├── nuclear_labels_2d.tif
    │   ├── hypotheses_manifest.json
    │   └── labelmaps/
    │       └── labelmap_*.tif
    ├── 3_correction/
    │   └── nuclear_labels_corrected.tif
    ├── 4_cell_ultrack/
    │   ├── data.db
    │   ├── tracks.csv
    │   ├── tracked_labels.tif
    │   ├── cell_labels_2d.tif
    │   ├── hypotheses_manifest.json
    │   └── labelmaps/
    │       └── labelmap_*.tif
    └── 5_analysis/
        ├── graph.h5
        └── topology.npz
```

## Stage Specs

### 0. `raw_import`

Responsibilities:

- export the z-corrected source data into the new flat `0_input/` contract;
- write:
  - `nucleus_4d.tif`
  - `cell_4d.tif`
  - `nucleus_zavg.tif`
  - `cell_zavg.tif`
  - `z_shift.csv`
- keep position-local output under `posXX/0_input/`.

Requirements:

- the rest of the plugin must stop depending on the old nested `0_input/nucleus/` and `0_input/cell/` layout;
- the z-averaged raw stacks are produced here, not later in the cluster stage and not by downstream widgets;
- file-status widgets, project previews, and validation must use the new flat files only.

### 1. `cellpose_cluster`

Responsibilities:

- represent the external cluster-side Cellpose step in the pipeline UI;
- document what runs on the cluster;
- document which files from `0_input/` are consumed;
- document which files must be written into `1_cellpose/`;
- validate that the expected artifacts for the current position exist.

Requirements:

- all local Cellpose execution paths are removed from the UI;
- all local Cellpose execution paths are removed from the backend for this workflow;
- the stage UI is reduced to explanatory text plus file-status/validation only;
- the pipeline config must be the authoritative source of the expected stage-1 files;
- the widget must not assume the legacy local-run semantics.

Default file contract:

- inputs from data prep:
  - `0_input/nucleus_4d.tif`
  - `0_input/cell_4d.tif`
  - `0_input/nucleus_zavg.tif`
  - `0_input/cell_zavg.tif`
- required nucleus outputs:
  - `1_cellpose/nucleus_dp.tif`
  - `1_cellpose/nucleus_prob.tif`
  - `1_cellpose/nucleus_dp_zavg.tif`
  - `1_cellpose/nucleus_prob_zavg.tif`
- required cell outputs:
  - `1_cellpose/cell_dp.tif`
  - `1_cellpose/cell_prob.tif`
  - `1_cellpose/cell_dp_zavg.tif`
  - `1_cellpose/cell_prob_zavg.tif`

### 2. `nucleus_ultrack`

Inputs:

- `1_cellpose/nucleus_dp.tif`
- `1_cellpose/nucleus_prob.tif`
- `1_cellpose/nucleus_dp_zavg.tif`
- `1_cellpose/nucleus_prob_zavg.tif`

Responsibilities:

- generate nucleus segmentation hypotheses from Cellpose flow/prob maps via parameter sweep;
- ingest those hypotheses directly into the Ultrack database;
- run linking and solving;
- export the 3D tracked labels;
- project the 3D tracked labels to `nuclear_labels_2d.tif`.

Explicit non-requirements:

- no required `foreground.tif`;
- no required `contours.tif`;
- no Ultrack watershed over contour maps in the canonical nucleus workflow.

Implementation requirements:

- reuse the current Cellpose-based mask generation logic for parameter sweeps;
- replace the current "mask -> contours -> Ultrack watershed" path with "mask -> direct candidate ingestion";
- make hypothesis provenance inspectable through `hypotheses_manifest.json`;
- persist the per-hypothesis label stacks under `2_nucleus_ultrack/labelmaps/` as `labelmap_*.tif`;
- keep existing linking and solving behavior unless the plan decides nucleus linking should also move to the new IoU-capable abstraction.

### 3. `correction`

Inputs:

- `2_nucleus_ultrack/nuclear_labels_2d.tif`

Outputs:

- `3_correction/nuclear_labels_corrected.tif`

Responsibilities:

- load the projected 2D nucleus labels;
- allow manual correction with the existing correction tooling;
- save the corrected 2D nucleus labels for the cell workflow.

Requirements:

- the correction step remains between nucleus tracking and cell hypothesis generation;
- the saved corrected labels are the authoritative seeds for the cell workflow.

### 4. `cell_ultrack`

Inputs:

- `3_correction/nuclear_labels_corrected.tif`
- `1_cellpose/cell_dp.tif` or `1_cellpose/cell_dp_zavg.tif`
- `1_cellpose/cell_prob.tif` or `1_cellpose/cell_prob_zavg.tif`

Responsibilities:

- generate cell segmentation hypotheses using corrected nucleus labels as seeds;
- reuse the seeded-watershed sweep logic that already exists later in the pipeline;
- ingest those cell hypotheses directly into a second Ultrack database;
- link candidates using an IoU-oriented strategy;
- solve the final segmentation;
- export `tracked_labels.tif`;
- project to `cell_labels_2d.tif`.

Requirements:

- the seeded-watershed code becomes a hypothesis generator, not a terminal output stage;
- the cell workflow must not depend on the nucleus Ultrack database;
- the cell workflow has its own working directory and database;
- persist the per-hypothesis label stacks under `4_cell_ultrack/labelmaps/` as `labelmap_*.tif`;
- the current centroid-based projection algorithm is kept for now.

Open design point:

- exact IoU-based linking behavior is still under development;
- the plan must explicitly decide whether the same linking abstraction should already be applied to nuclei or whether nuclei keep the current behavior in this refactor.

## Cross-Cutting Backend Specs

### Direct hypothesis ingestion

Add a shared Ultrack-side API for:

- accepting per-timepoint label hypotheses directly;
- populating the Ultrack database without going through watershed on `foreground` and `contours`;
- reusing the same ingestion path for:
  - nucleus hypotheses from Cellpose mask sweeps;
  - cell hypotheses from seeded-watershed sweeps.

This API is the main backend refactor target.

### Linking abstraction

Ultrack integration must support at least two linking modes:

- current/default linking for nuclei;
- IoU-oriented linking for cells.

The linking mode must be configurable per workflow rather than hard-coded globally.
The plan must resolve whether nuclei should stay on the current mode initially or move onto the shared abstraction immediately.

### Projection abstraction

The existing projection code in `project2d.py` becomes a generic utility that can write:

- `nuclear_labels_2d.tif`
- `cell_labels_2d.tif`

The algorithm stays the same in this refactor:

- if only one non-zero label occupies a 2D pixel column, keep it;
- if multiple labels contend for that pixel, assign it to the label whose 2D centroid is closest.

## UI Specs

The user-facing workflow should read as:

1. Data Prep
2. Cluster Cellpose
3. Nucleus Ultrack
4. Nuclear Correction
5. Cell Ultrack
6. Analysis

Requirements:

- the current Ultrack widget must be split or redesigned so nucleus and cell workflows are distinct;
- the current seeded-watershed widget must move under the cell workflow or be absorbed into it;
- the current Cellpose widget and Cellpose execution controls must be removed;
- the replacement Cellpose stage UI is informational only: it explains cluster-side execution, expected inputs, and required outputs;
- file-status widgets must reflect the new directories and filenames;
- project overview and new-project text must stop advertising the contour-driven nucleus workflow.

## Config And Path Specs

The following mappings must become authoritative:

- `raw_import -> 0_input`
- `cellpose_cluster -> 1_cellpose`
- `nucleus_ultrack -> 2_nucleus_ultrack`
- `correction -> 3_correction`
- `cell_ultrack -> 4_cell_ultrack`
- `analysis -> 5_analysis`

Requirements:

- `core.paths.STAGE_DIRS` must match this layout;
- `pipeline_schema.json` and any project bootstrap templates must use the new stage list;
- manifest badges, tab ordering, and validation logic must use the same stage keys;
- legacy references to `contours`, `tracking`, `project2d`, `cell_labels`, `seeded_watershed`, `cellpose_nucleus`, and `cellpose_cell` as top-level pipeline stages must be removed rather than kept transitional.

## Acceptance Criteria

- A fresh project exposes the new stage order and directories.
- Data prep writes the flat `0_input/` artifacts required downstream, including `nucleus_zavg.tif` and `cell_zavg.tif`.
- The plugin exposes an informational Cellpose cluster step with no local execution controls.
- The plugin can validate cluster-produced Cellpose nucleus and cell outputs without running local inference.
- The nucleus workflow produces `2_nucleus_ultrack/tracked_labels.tif` and `2_nucleus_ultrack/nuclear_labels_2d.tif` without using contour-driven Ultrack watershed.
- The nucleus workflow persists its hypothesis labelmaps under `2_nucleus_ultrack/labelmaps/`.
- The correction workflow loads `nuclear_labels_2d.tif` and saves `nuclear_labels_corrected.tif`.
- The cell workflow consumes corrected nuclei plus cell flow/prob maps, runs seeded-watershed hypothesis generation, links, solves, and exports `4_cell_ultrack/cell_labels_2d.tif`.
- The cell workflow persists its hypothesis labelmaps under `4_cell_ultrack/labelmaps/`.
- Both workflows use the same direct-hypothesis ingestion mechanism.
- Both workflows use the same current centroid-based 3D-to-2D projection implementation.

## Open Questions To Resolve In The Plan

- Whether nucleus linking should stay on the current behavior initially or move immediately onto the shared IoU-capable abstraction.
- Exact IoU-linking semantics for the cell workflow.
