# Graph Cut Cell Segmentation Experiment

**Date:** 2026-05-04
**Status:** Approved, ready for implementation

## Problem

Geodesic Voronoi assigns each foreground voxel to the nearest nucleus seed by travel cost. For elongated cells, the distal tip can be closer in geodesic cost to a neighboring nucleus than to its own centroid, causing neighboring cells to take over elongated arms even with high boundary weights.

## Goal

Test whether α-expansion graph cut — which minimizes a global energy simultaneously across all pixels — holds elongated cell boundaries better than distance-based front propagation.

## Approach

2D per-frame α-expansion graph cut using `PyMaxflow` (Boykov-Kolmogorov max-flow). Process each of the 50 frames independently, then stack results into a (T, Y, X) volume.

## Inputs

Same inputs as `experiment_cell_3d_geodesic_voronoi.py`:

- `--contours`: (T, Y, X) float32 contour probability volume
- `--foreground-mask`: (T, Y, X) binary foreground domain
- `--markers`: (T, Y, X) uint32 tracked nuclear labels; centroid seeds computed from these

## Energy Formulation

**Data term:**
- Seed pixels: cost 0 for their label, ∞ for all other labels (hard constraint)
- All other foreground pixels: cost 0 for every label (uniform — smoothness does all the work)
- Background pixels: excluded from the graph, assigned label 0

**Smoothness term:**
- 4-connected pixel pairs within the foreground mask
- Edge weight: `smoothness_weight * (1 - mean(contour[i], contour[j]))`
- High contour value → low cost to disagree (boundary expected)
- Low contour value → high cost to disagree (cell interior)

**Parameter sweep:** `--smoothness-weight` values, e.g. `5 20 50 100`

## Algorithm: α-expansion

Per frame, repeat up to 5 rounds (or until no pixel changes):

For each label α (130 labels):
1. Build binary graph: source = label α, sink = "not α"
2. Hard-pin seed pixels for label α to source; all other seed pixels to sink
3. Add n-links between 4-connected neighbor pairs with smoothness weight
4. Run max-flow; pixels cut to source get label α, sink pixels keep current label

## Output

Directory: `3_cell/graphcut_experiment/<timestamp>/`

- `seed_markers.tif` — centroid seed volume (T, Y, X) uint32
- `tracked_labels_sw_{X}.tif` — one per smoothness-weight value, (T, Y, X) uint32
- `parameters.json` — all inputs and parameter values
- `summaries.json` — per-run stats: n_output_ids, missing_marker_ids, unlabeled_fg_voxels, elapsed_s
- `summary_{suffix}.json` — per-run summary file (same structure as geodesic script)

## Script

`scripts/experiment_cell_2d_graphcut.py`

Follows the same structure as `experiment_cell_3d_geodesic_voronoi.py`:
- `_parse_args()`, `_load_inputs()`, `_build_graph(frame, markers_frame, smoothness_weight)`, `_run_alpha_expansion(graph, n_labels, max_rounds)`, `_summarize_labels()`, `_write_json()`, `main()`

## Dependencies

- `PyMaxflow` — install into `cellflow` conda env before running
- All other dependencies already present (numpy, tifffile, scikit-image for centroid seeds)

## Success Criteria

- All 130 cell IDs present in output (0 missing marker IDs)
- Elongated cells visually hold their arms better than geodesic bw_128_alpha_0p25_tspace_10
- Runtime acceptable: target under 10 minutes for all parameter combos across 50 frames
