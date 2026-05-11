# Results — Run 5 flow endpoint unary with lower temporal coupling

**Date:** 2026-05-10  
**Script:** `scripts/experiment_cell_2d_t_multilabel_graphcut.py`  
**Output:** `/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/4_cell_graphcut/20260510-run5-flow-only-low-temporal`

## Configuration

Same as Run 4, except temporal pairwise was reduced by 10x:

```text
unary(p, k) = flow_endpoint_unary(p, k)
lambda_geodesic = 0
lambda_flow = 1
lambda_s = 20
beta_s = 5
lambda_t = 0.5
n_iters = 5
min_round_flips = 100
```

The full flow unary was loaded from cache.

## Results table

| Run | Unary | λ_s | λ_t | Final energy | Mean IoU | Median IoU | Purity | Completeness | Pred flicker | Boundary ratio | Total time |
|-----|-------|-----|-----|--------------|----------|------------|--------|--------------|--------------|----------------|------------|
| 4 | flow endpoint | 20 | 5.0 | 3,468,001 | 0.820 | 0.830 | 0.899 | 0.898 | **0.167** | 8.71 | 355.0s |
| **5** | **flow endpoint** | **20** | **0.5** | **1,589,367** | **0.882** | **0.895** | **0.935** | **0.938** | 0.251 | **12.35** | **224.9s** |

GT flicker rate: 0.258. Run 5 coverage: 1.000.

Energy values are comparable only between runs with the same pairwise weights and unary definition; changing `lambda_t` changes the objective scale.

## Energy log

| Round | Energy | Flips | Note |
|-------|--------|-------|------|
| 1 | 1,589,387.6420 | 790,271 | Main optimization step |
| 2 | 1,589,366.6860 | 25 | Stopped by `min_round_flips=100` |

## Takeaway

Reducing temporal coupling by 10x is a large improvement on the tracked-label benchmark:

- Mean IoU: `0.820 → 0.882`
- Purity: `0.899 → 0.935`
- Completeness: `0.898 → 0.938`
- Boundary ratio: `8.71 → 12.35`

The tradeoff is flicker: `0.167 → 0.251`, almost matching GT flicker `0.258`. By the current metrics, `lambda_t=0.5` is the best tested setting.

## pos04 replication

**Output:** `/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos04/4_cell_graphcut/20260510-pos04-run5-flow-only-low-temporal`

Same configuration as Run 5, applied to `pos04`.

```text
unary(p, k) = flow_endpoint_unary(p, k)
lambda_geodesic = 0
lambda_flow = 1
lambda_s = 20
beta_s = 5
lambda_t = 0.5
n_iters = 5
min_round_flips = 100
```

First `pos04` run wrote the full flow unary cache:

`/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos04/4_cell_graphcut/unary_cache/flow_shape-50x512x512_crop-full_alpha-0.h5`

| Position | Tracks | Frames | Final energy | Mean IoU | Median IoU | Purity | Completeness | Pred flicker | GT flicker | Boundary ratio | Unary time | Graphcut time | Total time |
|----------|--------|--------|--------------|----------|------------|--------|--------------|--------------|------------|----------------|------------|---------------|------------|
| pos00 | 68 | 48 | 1,589,367 | 0.882 | 0.895 | 0.935 | 0.938 | 0.251 | 0.258 | 12.35 | 6.5s | 218.4s | 224.9s |
| pos04 | 69 | 50 | 5,316,702 | 0.710 | 0.725 | 0.837 | 0.836 | 0.322 | 0.355 | 2.58 | 73.1s | 313.2s | 386.3s |

Energy log for `pos04`:

| Round | Energy | Flips | Note |
|-------|--------|-------|------|
| 1 | 5,316,883.8385 | 1,001,336 | Main optimization step |
| 2 | 5,316,701.9908 | 72 | Stopped by `min_round_flips=100` |

`pos04` does not replicate the strong `pos00` result. The flow-only low-temporal setting still covers all foreground and flicker remains below GT, but identity and boundary metrics are much weaker, especially boundary alignment ratio (`2.58` vs `12.35` on `pos00`).
