# Results — Run 4 flow endpoint unary only

**Date:** 2026-05-10  
**Script:** `scripts/experiment_cell_2d_t_multilabel_graphcut.py`  
**Output:** `/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/4_cell_graphcut/20260510-run4-flow-only-strong-pair`

## Configuration

Run 4 removes the direct pixel-to-nucleus geodesic distance penalty from the unary by setting `lambda_geodesic = 0`.

```text
unary(p, k) = 0.0 * geodesic_unary(p, k) + 1.0 * flow_endpoint_unary(p, k)
init = unary argmin
lambda_s = 20
beta_s = 5
lambda_t = 5
n_iters = 5
min_round_flips = 100
```

Since `lambda_geodesic=0` and `init=unary`, the run skips geodesic MCP computation entirely. It uses the flow endpoint unary with the stronger 2A/Run 3 pairwise and temporal settings.

## Speed changes

- Added persistent HDF5 unary caches under `4_cell_graphcut/unary_cache/`.
- Full flow unary cache written at `4_cell_graphcut/unary_cache/flow_shape-48x512x512_crop-full_alpha-0.h5`.
- Added `--min-round-flips`; Run 4 stopped after round 2 because round 2 had only `85` flips.
- Added round-level best-energy guard: if a later tiny round worsens energy, the script reverts to the best previous labeling and stops.

Run 4 timing:

```text
Unary time:    74.9s  (includes first-time full flow cache write)
Graphcut time: 280.1s
Total time:    355.0s
```

For comparison, Run 3 took `546.7s` total (`150.6s` unary + `396.1s` graphcut).

## Results table

| Run | Unary | Init | λ_s | λ_t | Rounds | Final energy | Mean IoU | Median IoU | Purity | Completeness | Pred flicker | Boundary ratio |
|-----|-------|------|-----|-----|--------|--------------|----------|------------|--------|--------------|--------------|----------------|
| 2A | geodesic | geodesic | 20 | 5 | 2 (+osc.) | 3,576,469 | 0.780 | 0.776 | 0.868 | 0.875 | **0.154** | 7.90 |
| 3 | geodesic + flow endpoint | geodesic | 20 | 5 | 3 | 3,926,716 | 0.794 | 0.791 | 0.878 | 0.884 | 0.158 | 8.27 |
| **4** | **flow endpoint only** | **flow unary** | **20** | **5** | **2** | **3,468,001** | **0.820** | **0.830** | **0.899** | **0.898** | 0.167 | **8.71** |

GT flicker rate: 0.258. Run 4 coverage: 1.000.

Energy values are only directly comparable between runs with the same unary definition.

## Run 4 energy log

| Round | Energy | Flips | Note |
|-------|--------|-------|------|
| 1 | 3,468,366.4068 | 1,008,569 | Main optimization step |
| 2 | 3,468,000.6387 | 85 | Stopped by `min_round_flips=100` |

## Takeaway

Removing the direct geodesic distance term improved every identity and boundary metric relative to Run 3:

- Mean IoU: `0.794 → 0.820`
- Purity: `0.878 → 0.899`
- Completeness: `0.884 → 0.898`
- Boundary ratio: `8.27 → 8.71`

The tradeoff is flicker: `0.158 → 0.167`, still well below GT flicker `0.258` by this metric. Flow endpoint unary plus strong contour/temporal pairwise is now the best configuration tested.
