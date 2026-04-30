# Flow Magnitude Cell Workflow Findings

Date: 2026-04-30

## Symptom

Selecting `Flow Magnitude` in the cell workflow does not produce a usable preview or segmentation output. The flow magnitude map is not available through the normal project file contract, and the segmentation path depends on a dp stack that is not advertised as a Cellpose output.

## Evidence

- `CellWorkflowWidget` exposes `Flow Magnitude` as a basin option and says it is computed from dp vectors on the fly.
  - `src/cellflow/napari/cell_workflow_widget.py:123-129`
- The cell workflow looks specifically for `1_cellpose/cell_dp_3dt.tif` when `Flow Magnitude` is selected.
  - `src/cellflow/napari/cell_workflow_widget.py:468-469`
  - `src/cellflow/napari/cell_workflow_widget.py:535-540`
- The Cellpose panel does not list `1_cellpose/cell_dp_3dt.tif` as an expected output. It lists nucleus dp, cell prob, and z-average prob files, but no cell dp file.
  - `src/cellflow/napari/cellpose_widget.py:39-45`
- The project status panel also does not track `1_cellpose/cell_dp_3dt.tif`, so the missing dependency is not visible at project level.
  - `src/cellflow/napari/data_panel_widget.py:29-34`
- The repository search found nucleus dp references and examples, but no local `cell_dp` fixtures or scripts. Existing scripts document nucleus dp as `(T, Z, 2, Y, X)`.
  - `scripts/test_cellpose_native.py:13`
  - `scripts/test_contour_sweep.py:31`
- Once loaded, the seeded watershed DB path expects dp shape `(T, Z, 2, Y, X)` and slices it as `dp_t[z]`.
  - `src/cellflow/database/hypotheses.py:472-488`
- Preview and DB generation both pass per-z dp slices into `compute_seeded_watershed`.
  - `src/cellflow/napari/cell_workflow_widget.py:748-759`
  - `src/cellflow/database/hypotheses.py:472-474`

## Likely Root Causes

1. The upstream file contract is incomplete: `cell_dp_3dt.tif` is required for the flow-magnitude cell basin, but the Cellpose/data panels do not advertise or track that file.
2. There appears to be no repository-side generation, fixture, or documented external output for `cell_dp_3dt.tif`; only nucleus dp is consistently referenced.
3. Flow-vector shape handling is narrow. The DB path documents `(T, Z, 2, Y, X)`, but the workflow loader does not normalize alternate common layouts such as `(T, 2, Z, Y, X)` or channel-last layouts. If external Cellpose writes a different axis order, the flow magnitude preview/segmentation path can fail even when a dp file exists.

## Suggested Fix Direction

- Decide and document the canonical cell dp file produced by external Cellpose: `1_cellpose/cell_dp_3dt.tif`.
- Add `cell_dp_3dt.tif` to the Cellpose output/status panels, preferably marked as required only when the cell workflow basin is `Flow Magnitude`.
- Add focused tests for `_load_inputs` and the preview path with `Flow Magnitude` selected.
- Normalize dp stack axes after loading so the cell workflow and DB generator receive a canonical `(T, Z, C, Y, X)` stack, then pass `(C, Y, X)` slices into `compute_seeded_watershed`.

## Resolution

Implemented after this investigation:

- `1_cellpose/cell_dp_3dt.tif` is now shown in the Cellpose panel and project status panel.
- Flow-vector dp stacks are normalized to canonical `(T, Z, C, Y, X)` before preview, Save to DB, and Sweep.
- Common layouts such as `(T, Z, C, Y, X)`, `(T, C, Z, Y, X)`, `(T, Z, Y, X, C)`, and single-time 4D equivalents are accepted.
- Regression coverage was added for Flow Magnitude preview and seeded-watershed DB record generation.
