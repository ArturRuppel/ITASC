# Cell Segmentation — Contour Maps: Status Report

Date: 2026-05-04

## The Problem

We have Cellpose outputs (probability logits `cell_prob_3dt.tif`, displacement fields
`cell_dp_3dt.tif`) for a 3D+T spinning-disk dataset (50 frames × 8 z-planes × 512 × 512).
We also have curated tracked nuclear labels (`2_nucleus/tracked_labels.tif`, 130 unique
tracks). The goal is to produce per-frame 2D cell label maps that:

1. Follow real cell boundaries faithfully when the contour signal is present and consistent
2. Interpolate / fall back to a prior when the signal is spurious or absent

This is harder than it sounds because Cellpose contour maps are temporally inconsistent:
a boundary may be sharp in frame t but absent or shifted in frame t+1, even when the
underlying cell hasn't changed. Naive segmentation methods either follow every contour
including spurious ones, or ignore the contour geometry entirely.

---

## Contour Maps Generated

**Script**: `scripts/experiment_cell_contour_maps.py`  
**Backend**: `cellflow.segmentation.build_mean_z_consensus_boundary` (and the older
`build_consensus_boundary` for per-slice runs)

Five runs in `3_cell/contour_experiment/`:

| Directory | Method | Thresholds | Used downstream? |
|---|---|---|---|
| `20260503-231730` | Per-slice (z × threshold) | [-2, 0] | No (incomplete run) |
| `20260503-231753` | Per-slice | [-2, 0] | No |
| `20260503-232245-thr-8-to-0-maxfg` | Per-slice | [-8..0] | Only graphcut pilots |
| `20260504-contours-thr-m5-to-5-maxfg` | Per-slice | [-5..+5] | No |
| `20260504-contours-meanz-thr-m5-to-5` | **Mean-Z** | [-5..+5], gamma=1.0 | All current experiments |

**Mean-Z method** (`build_mean_z_consensus_boundary`): mean-projects prob and dp across Z
first (Z,Y,X → Y,X), then sweeps 11 threshold values, runs Cellpose `compute_masks` on each
2D projected image, and averages `find_boundaries` across all 11 results. Output shape:
`(T, Y, X)` float32 boundary density + foreground sigmoid map. Completed in ~18s for 50
frames.

**Per-slice method** (`build_consensus_boundary`): runs Cellpose for every
(threshold × z-slice) pair — 88 calls/frame at 11 thresholds × 8 z-slices — then averages
boundaries. Slower and never used in downstream experiments after mean-Z was available.

**Unexplored**: the `gammas` parameter in `build_mean_z_consensus_boundary` supports gamma
correction before mean-projection but all runs use `gammas=[1.0]`. A gamma sweep is a cheap
untried dimension. Per-slice contours were also never compared against mean-Z downstream.

---

## Downstream Label Experiments

All experiments use the same inputs: mean-Z contours (`20260504-contours-meanz-thr-m5-to-5`),
a binary foreground mask (`20260503-232245-thr-8-to-0-maxfg/foreground_masks.tif`), and
centroid seeds from tracked nuclear labels.

### Key finding: aggregate metrics are insensitive to contour quality

Every experiment produces **130 IDs, 0 missing markers** regardless of contour preprocessing.
The reason: hard centroid seeds guarantee one label per tracked nucleus; the contour map only
affects where basin boundaries fall, not how many labels are produced. Our metrics
(`n_output_ids`, `n_missing_marker_ids`, `unlabeled_foreground_voxels`) cannot detect
boundary placement quality. 50+ parameter sweeps all return the same numbers.

### 2D Seeded Watershed (baseline)

Script: `experiment_cell_2d_seeded_watershed.py`

Two runs (full-label seeds and centroid seeds), both: 130 IDs, 0 missing, ~16K unlabeled
foreground voxels (~0.23% of total). Fast: ~1s per 50-frame stack.

### Temporal Agreement Watershed

Script: `experiment_cell_2d_temporal_agreement_watershed.py`

Filters contours temporally before watershed. Two strategies:
- `max_agreement`: `(1-α)·contour_t + α·max(contour_{t-1}, contour_{t+1})`, α ∈ {0..1}
- `asymmetric`: independent boost/suppress parameters for recovering vs. suppressing boundaries

11 parameter combinations run. **All produce identical results**: 130 IDs, 0 missing,
16,588 unlabeled. Temporal filtering changes contour values but not watershed basins when
seeds are hard-pinned.

### Voronoi Fusion Watershed

Script: `experiment_cell_2d_voronoi_fusion_watershed.py`

Blends Cellpose contours with a soft Euclidean Voronoi boundary derived from nuclear labels:
`fused = α·cellpose + (1-α)·voronoi`. 8 runs across alpha=[0..1] and sigma=[2,5].

**All produce identical results**: 130 IDs, 0 missing, 16,588 unlabeled. The Voronoi
boundary is topologically consistent (captures which cells are adjacent) but doesn't capture
actual boundary geometry or cell deformation — the blend still feeds a hard-seeded watershed
that is insensitive to contour input.

### α-Expansion Graph Cut

Scripts: `experiment_cell_2d_graphcut.py` and `experiment_cell_2d_graphcut_fast.py`

Potts-model α-expansion with centroid seeds hard-pinned via infinite t-links. Smoothness
cost = `w × (1 - mean_contour)`. Weights: [5, 20, 50, 100].

| Variant | sw | Unlabeled fg | Time |
|---|---|---|---|
| Standard (compact foreground nodes) | 5–100 | 0 | ~9 min |
| Fast (full grid, vectorized edges) | 5–100 | 0 | ~15 min |

Graph cut labels every foreground pixel (0 unlabeled vs watershed's ~16K gap). The "fast"
variant is actually slower because it allocates 512×512 node arrays instead of ~7K foreground
nodes. Aggregate label metrics again identical across all smoothness weights.

### Not Run

- **Temporal smooth watershed** (`experiment_cell_2d_temporal_smooth_watershed.py`): script
  exists (Gaussian sigma 1-4, median kernel 3-9), no outputs generated
- **3D experiments** (`experiment_cell_3d_seeded_watershed.py`,
  `experiment_cell_3d_geodesic_voronoi.py`): scripts exist, no full-scale outputs
- **`compute_contour_watershed`**: EDT-based seeding from the contour map itself (no nuclear
  markers). Only method where contour quality would change the label count. Never benchmarked.

---

## Visual Inspection Findings

Visual inspection of the output label maps in napari revealed the fundamental failure modes
that the aggregate metrics cannot detect:

- **Watershed / all temporal variants**: does not follow real contours faithfully when they
  are present. Basin boundaries drift away from actual cell edges.
- **Graph cut**: follows every contour including spurious ones. A boundary that flickers on
  for one frame gets incorporated as a real cell boundary.
- **No method** implements the desired behavior: *trust a contour when it is temporally
  consistent; interpolate its position from neighboring frames when it is not.*

---

## The Core Unsolved Problem

We need a mechanism that can distinguish:

- **Real boundary**: present and geometrically stable across multiple frames → follow it
- **Spurious boundary**: appears in one frame, absent in neighbors → suppress it or
  interpolate its position from surrounding frames

All current methods treat the contour map as ground truth at each frame independently, with
no mechanism to identify or suppress temporally inconsistent signal.

---

## Ideas Discussed

### Optical flow — most promising, not yet tried

Warp `contours[t-1]` into frame t using optical flow estimated from the raw images, then
blend with the actual `contours[t]`:

```
warped[t] = warp(contours[t-1], flow(raw[t-1] → raw[t]))
robust[t]  = α·contours[t] + (1-α)·warped[t]
```

Can be iterated (incorporate t-2, t-3 etc.) and run bidirectionally (past and future frames).

**Why it addresses the problem**: a spurious boundary in frame t will disagree with the
motion-compensated prediction from t-1, suppressing it in the blend. A real boundary will
agree with the prediction and be reinforced.

**Why not yet implemented**: the complexity is not in the algorithm itself (~50 lines using
`skimage.registration.optical_flow_ilk` or similar) but in the validation loop — optical
flow has smoothness, window size, and pyramid-level parameters that are hard to tune without
careful visual inspection of the flow field itself, before you can even evaluate the blended
contours.

### Nuclear-centroid motion compensation — simpler but insufficient

Use the nuclear centroid displacement `d_k = centroid(k, t) - centroid(k, t-1)` per tracked
cell as a motion model, warp each cell's Voronoi region of `contours[t-1]` by `d_k`, then
blend with `contours[t]`.

**Rejected**: translation per cell is not a good motion model. Cell contours change shape
between frames (elongation, division, squeezing) — not just translate. A rigid shift per
cell would warp the contour region incorrectly wherever the cell is deforming, and those are
precisely the regions where temporal consistency matters most.

### Gamma sweep — cheap, unexplored

`build_mean_z_consensus_boundary` accepts a `gammas` list applied before mean-projection.
Sweeping gamma changes which boundaries are emphasized. Has never been tested with `gamma ≠ 1.0`.
Low-hanging fruit before investing in optical flow.

### Boundary-quality proxy metric — no ground truth needed

Measure what fraction of high-contour-map pixels fall on label boundaries vs. inside labels.
A good label map should have its boundaries aligned with high-contour pixels. This would let
us rank existing outputs without ground truth annotations and determine whether the problem
is in the contour maps or in the label assignment step.

---

## State Summary

| Layer | Status |
|---|---|
| Cellpose outputs (prob, dp) | Available, 50×8×512×512 |
| Mean-Z consensus boundary map | Generated, one gamma-1.0 run |
| Per-slice consensus boundary maps | Generated, never used downstream |
| Downstream label experiments | Extensive sweeps; all produce identical aggregate metrics |
| Visual quality | Poor — no method faithful-yet-robust |
| Ground truth / proxy metric | None |
| Optical flow temporal blending | Not tried; most promising next direction |
| Gamma sweep | Not tried |

## Recommended Next Steps (in order of increasing complexity)

1. **Visual inspection of contour maps across time** (30 min): in napari, scrub through
   `contours.tif` for a few cells. Classify failures as "flickering" (optical flow would
   help) vs "uniformly absent/weak" (need better contour generation).

2. **Gamma sweep** (1-2h): add `gammas=[0.5, 0.75, 1.0, 1.25, 1.5]` to contour map
   generation, visually compare outputs.

3. **Boundary-quality proxy metric** (2-4h): compute
   `mean(contours[label_boundaries]) / mean(contours[label_interior])` for each method's
   output. Lets us rank existing results without ground truth.

4. **Optical flow contour blending** (1-2 days including validation): implement single-
   lookback warp-and-blend, spend time validating the flow field visually before evaluating
   blended contours.
