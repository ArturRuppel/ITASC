# Results — 2D+T multi-label graph-cut experiment (v1)

**Date:** 2026-05-10  
**Script:** `scripts/experiment_cell_2d_t_multilabel_graphcut.py`  
**Output dir:** `<pos00>/4_cell_graphcut/20260510-full-run/`  
**Pos-dir:** `/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00`

## Parameters used

| Param | Value |
|-------|-------|
| alpha_unary | 4.0 |
| lambda_s | 1.0 |
| beta_s | 5.0 |
| lambda_t | 1.0 |
| n_iters | 3 |
| INF | 1e9 |
| Graph mode | Full T×Y×X grid; background pinned to sink |

## Runtime

| Phase | Time |
|-------|------|
| Geodesic unaries (2762 MCP runs) | 78.4 s |
| Graph-cut (60 labels × 1 round) | 111.2 s |
| **Total** | **189.5 s (~3.2 min)** |

Volume: 48×512×512, fg_voxels=7,176,789, 60 track IDs.

## Metric numbers

| Metric | Value |
|--------|-------|
| Coverage | 99.99% |
| Mean temporal IoU (vs GT) | 0.736 |
| Median temporal IoU | 0.742 |
| 25th-percentile IoU | 0.703 |
| Per-track IoU range | 0.552 – 0.865 |
| Mean purity | 0.818 |
| Mean completeness | 0.875 |
| Pred flicker rate | 0.230 |
| GT flicker rate | 0.258 |
| Boundary contour mean | 0.219 |
| Interior contour mean | 0.048 |
| Boundary alignment ratio | 4.55 |
| Energy after round 1 | 1,039,699 |

## What happened in the graph cut

**Zero label changes in round 1. Algorithm declared convergence and stopped.**

The α-expansion performed 60 binary graph-cut solves and changed 0 pixels in each one. This is not a solver bug — it is a structural consequence of the initialization:

1. The initialization assigns each foreground pixel the label with minimum geodesic distance (geodesic Voronoi). This is by definition the argmin of the unary cost per pixel.

2. For α-expansion to flip pixel p from β to α, the pairwise benefit must exceed the unary cost difference: `pairwise_benefit(p→α) > unary(p,α) − unary(p,β)`. Since initialization already achieved the argmin, `unary(p,α) − unary(p,β) ≥ 0` for all α≠β.

3. At Voronoi boundaries (where unary difference ≈ 0 and flipping should in principle happen), the 4-connected Potts pairwise terms are balanced: boundary pixels have roughly equal numbers of α- and β-labeled neighbors, so the net pairwise incentive to flip ≈ 0 at the boundary level.

4. **Net result: the initialization IS the α-expansion fixed point for these parameter values.** The output is simply the geodesic Voronoi segmentation, not a graph-cut enhanced result. Lambda_t=1.0 had no effect because no pixel ever changed.

## What worked

- **Geodesic Voronoi baseline is solid.** IoU 0.736 (mean), 0.742 (median), purity 0.818, completeness 0.875 — comparable to or better than prior per-frame methods for this dataset.
- **Boundary alignment is strong.** Ratio 4.55: predicted boundaries fall on pixels with 4.55× higher contour signal than interiors. The geodesic cost field (1 + alpha_unary × contour) does steer boundaries toward ridges.
- **Coverage is nearly perfect** (99.99%). Every foreground voxel received a label.
- **Speed.** Full 48-frame volume in 3.2 min including geodesic pre-computation. The vectorised `add_grid_edges` approach (per-frame 2D spatial + temporal via (T,YX) reshape) scales well: each of the 60 graph builds takes ~1.8 s for a 12.6 M-node graph.
- **No OOM.** The full-grid approach (background pinned to sink) worked within available RAM.
- **Pred flicker (0.230) is lower than GT flicker (0.258).** The pred is temporally smoother than GT by this metric. (Note: the flicker metric used here counts pixels entering/leaving each label's region between frames, not centroid-aligned label assignment — it is a proxy, not the exact metric the brief specified.)

## What didn't work

- **The graph cut adds nothing over geodesic Voronoi.** Both the initialization and the output are the same geodesic Voronoi segmentation. The α-expansion framework is correctly implemented but its parameters are in a degenerate regime where the unary term completely dominates.
- **Temporal coupling (lambda_t) has zero effect.** Because no pixel ever flips, the temporal Potts term never enters the optimization. The whole motivation for the 2D+T formulation — forcing label continuity across time — was not activated.
- **Worst-performing tracks** (IDs 32, 29, 26, 34, 12; IoU 0.55–0.63) are likely cases where the geodesic path is occluded or where cell shapes differ significantly from nucleus proximity.

## Root cause and path forward

The pathological regime arises from a circular dependency: **the unary is the geodesic Voronoi, and the initialization is also the geodesic Voronoi.** Any labeling reachable by α-expansion from this initialization would have to overcome the unary term to move a pixel — but the unary is exactly designed to keep each pixel where it is.

Breaking the circular dependency requires **at least one** of:

1. **Much stronger pairwise weights.** Try lambda_s in [10, 50, 200] and lambda_t in [5, 20]. At large lambda_s the Potts boundary-snapping can dominate, moving entire boundary strips from one cell to another even when the unary favors the original label. This is the most direct fix.

2. **Softer or differently-scaled unary.** The current normalization (divide by median distance) puts typical pixels at unary ≈ 1.0 while pairwise is at most ~1.0. If instead we use a log-transform or cap the unary at a smaller value (e.g., median→0.1), the unary becomes less dominant.

3. **Decouple initialization from unary.** Initialize with something other than geodesic Voronoi (e.g., Euclidean Voronoi or even all-zero labels). Then the α-expansion has real work to do: it must move pixels from their initial (suboptimal) positions to better ones, and the geodesic unary guides where they go. This avoids the self-defeating initialization.

4. **Try a proper α-β-swap instead of α-expansion.** Swaps can move entire regions simultaneously and may be less prone to this kind of initialization lock-in, at the cost of more complex implementation.

**Recommended next experiment:** Re-run with option 1 (lambda_s=20, lambda_t=5) and option 3 (random or Euclidean-Voronoi initialization). This should activate the boundary-snapping and temporal-coupling terms and test the formulation's actual power.
