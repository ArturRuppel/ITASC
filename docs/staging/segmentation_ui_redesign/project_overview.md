# CellFlow — Project Overview for Segmentation UI Redesign

## What is CellFlow?

CellFlow is a napari plugin for fast, interactive, hypothesis-driven cell tracking. It
processes 2D+t (+ optional z) microscopy data through a pipeline:

1. **Data Preparation** — import TIFF stacks
2. **Cellpose** — run cellpose to get cell probability maps and flow fields
3. **Nucleus Segmentation & Tracking** — watershed + ultrack tracking
4. **Cell Segmentation** ← the target of this redesign
5. **Analysis**
6. **Meta Analyzer**

The UI is structured as collapsible sections in a scrollable napari dock widget.

## File Map

```
src/cellflow/
├── napari/
│   ├── main_widget.py              # Top-level dock widget, holds all collapsible sections
│   ├── cell_workflow_widget.py     # "4. Cell Segmentation" — parent widget
│   ├── cell_boundary_workflow_widget.py  # ★ THE TARGET — "Segmentation" collapsible
│   ├── widgets.py                  # CollapsibleSection, PipelineFilesWidget
│   ├── ui_style.py                 # Layout helpers: block_grid, add_block_pair_row, etc.
│   ├── correction_widget.py        # Interactive label correction (painting, merging, splitting)
│   ├── nucleus_workflow_widget.py  # Nucleus workflow (cross-references selection callbacks)
│   └── ...
├── segmentation/
│   ├── __init__.py                 # Public API, CellLabelICMParams re-export
│   ├── cell_label_icm.py           # ICM solver + geodesic unary pipeline
│   ├── contour_filtering.py        # Median/Gaussian filtering of contour maps
│   └── flow_following.py           # Flow-following cell segmentation (alternative method)
├── database/
│   └── tracked.py                  # TIFF read/write for tracked label stacks
├── correction/
│   ├── __init__.py                 # Re-exports from labels module
│   └── labels.py                   # Label manipulation ops (expand, merge, split, etc.)
└── core/
    └── paths.py                    # Path utilities (if any)
```

Key external script:
- `scripts/experiment_cell_2d_t_multilabel_graphcut.py` — The graphcut experiment CLI.
  The ICM solver in `cell_label_icm.py` was extracted from this script. The graphcut
  path (alpha-expansion via PyMaxflow) is only available via subprocess call from the
  widget; the ICM path runs in-process.

## The "Segmentation" Widget — Current State

The widget lives in `CellBoundaryWorkflowWidget` (`cell_boundary_workflow_widget.py`).
It's embedded inside `CellWorkflowWidget` with the title "Segmentation" and contains
three sub-sections:

### 1. Contour Maps (collapsible)
Builds consensus boundary maps from Cellpose outputs. Runs cellpose dynamics across
a sweep of cellprob thresholds and gamma values, then averages the results.

**Inputs**: `1_cellpose/cell_prob_3dt.tif`, `3_cell/filtered_dp.tif`
**Outputs**: `3_cell/contour_maps.tif`, `3_cell/foreground_scores.tif`, `3_cell/foreground_masks.tif`

### 2. Track-Conditioned Boundary Selection (collapsible) ← THE TARGET SECTION
This is the section the user wants to split into Initialize→Refine→Commit stages.

**Current flow** (single "Run" button):
1. Loads nucleus tracks, contour maps, foreground scores/masks, flow field
2. Computes geodesic unary costs via MCP (Minimum Cost Path) on contour-weighted cost field
3. Computes pairwise Potts weights (spatial + temporal)
4. Runs ICM solver (or graphcut via subprocess)
5. Saves results to `3_cell/tracked_labels.tif`
6. Displays in napari

**Parameters exposed in the UI**:
- Solver: graphcut (subprocess) | icm (in-process)
- Unary mode: flow | geodesic_flow | geodesic | euclidean
- Boundary mode: contour | foreground_inverse
- Iters, Workers (graphcut-only), alpha_unary, lambda_s, beta_s, lambda_t,
  lambda_geodesic, lambda_flow, lambda_contour, init_mode, min_round_flips

**What's broken / misleading**:
- graphcut is exposed in the combo box but many parameters are disabled for ICM.
  The graphcut path calls a subprocess to the experiment script, which is slow
  and fragile.
- The ICM solver does **all** computation (unary + pairwise + solver) in one
  button press. There's no way to preview the unary costs.
- There's no separation between "compute the energy terms" and "run the solver".
- Results are always written to file; there's no in-place refinement.

### 3. Correction (collapsible, hidden in CellWorkflowWidget embedding)
Hand-painting tools for label correction. Not relevant to this redesign except
that the correction widget could be used to refine solver outputs.

## The ICM Solver (`cell_label_icm.py`)

**Entry points**:
- `run_cell_icm_from_pos_dir(pos_dir, params)` — loads TIFFs from disk, runs pipeline
- `segment_cells_icm(nuc_tracks, fg_mask, contours, params)` — runs on in-memory arrays

**Pipeline steps**:
1. `_compute_pairwise_weights()` — spatial (h, v, dr, dl) + temporal (tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l) Potts weights
2. `_compute_geodesic_unaries()` — per-(frame, label) MCP_Geometric distance, normalized
3. `_apply_nucleus_anchors()` — hard constraints at nucleus pixels
4. `_dict_to_dense_unary()` — sparse dict → dense (T, Y, X, K) float32
5. `_run_icm()` — Numba JIT-compiled Gauss-Seidel ICM, `n_iters` rounds

**Key data structures**:
- `CellLabelICMParams` dataclass: alpha_unary, lambda_s, beta_s, lambda_t, n_iters, min_round_flips
- Pairwise weights: 9 arrays of shape (T, Y, X) — h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l
- Unary costs: dict[(t, k)] → (Y, X) float32, then densified to (T, Y, X, K)

## The Target Redesign

Split the "Track-Conditioned Boundary Selection" section into three stages:

### Stage 1: Initialize
- Compute unary costs (geodesic MCP)
- Compute pairwise/binary weights
- Save both to a cache (HDF5 or similar)
- Initialize masks from unary argmin (or watershed)
- Display masks in viewer

### Stage 2: Refine
- "Refine" button runs ICM solver iterations
- Mutates the label layer in-place in the napari viewer
- Does NOT write to disk
- User can press repeatedly for more iterations
- User can also hand-correct between rounds using the Correction tools

### Stage 3: Commit
- "Commit" button writes current labels to `3_cell/tracked_labels.tif`
- Could also save the cached unary/pairwise arrays

## Key Considerations

1. **Memory**: The unary dense array is (T, Y, X, K) float32 — for 100 frames, 512×512, 50 labels → ~5 GB. The sparse dict representation is more memory-efficient. Caching to HDF5 is essential.

2. **ICM vs Graphcut**: The graphcut solver is exposed but not functional from the UI
   (it launches a subprocess). The redesign should focus on ICM (which is what the user
   said is actually used) and potentially hide or remove the graphcut option.

3. **Correction Integration**: The correction widget (`CorrectionWidget`) already
   supports in-place label editing with undo history. The Refine stage should work
   alongside it — the user refines with ICM, then hand-corrects, then refines more.

4. **Viewer State**: The napari Labels layer is the source of truth during refinement.
   ICM reads from it, mutates it, and refreshes it. The CorrectionWidget needs to
   be aware of programmatic changes (or ICM needs to use the same history mechanism).

5. **Caching Strategy**: Unary computation (MCP_Geometric) is the slowest step. The
   experiment script already has an HDF5 unary cache (`_read_unary_cache` / 
   `_write_unary_cache`). The widget should adopt this pattern — compute once,
   cache, reuse across refinement rounds.
