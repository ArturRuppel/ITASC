# Results — Run 3 geodesic + flow endpoint unary

**Date:** 2026-05-10  
**Script:** `scripts/experiment_cell_2d_t_multilabel_graphcut.py`  
**Output:** `/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/4_cell_graphcut/20260510-run3-geodesic-flow-unary`

## Configuration

Run 3 keeps the winning 2A pairwise/temporal settings and adds flow endpoint distance to the unary:

```text
unary(p, k) = geodesic_unary(p, k) + 1.0 * flow_endpoint_unary(p, k)
init = geodesic Voronoi
alpha_unary = 4.0
lambda_s = 20
beta_s = 5
lambda_t = 5
n_iters = 5
```

Hard nucleus anchors remain INF constraints. If either the geodesic or flow term is INF, the hybrid unary stays INF.

## Results table

| Run | Unary | Init | λ_s | λ_t | Round-1 flips | Final energy | Mean IoU | Median IoU | Purity | Completeness | Pred flicker | Boundary ratio |
|-----|-------|------|-----|-----|---------------|--------------|----------|------------|--------|--------------|--------------|----------------|
| 2A | geodesic | geodesic | 20 | 5 | 1,424,015 | 3,576,469 | 0.780 | 0.776 | 0.868 | 0.875 | **0.154** | 7.90 |
| 2C | flow endpoint | euclidean | 10 | 3 | 2,116,909 | 2,134,732 | 0.765 | 0.760 | 0.854 | 0.872 | 0.149 | 7.47 |
| **3** | **geodesic + flow endpoint** | **geodesic** | **20** | **5** | **1,293,977** | **3,926,716** | **0.794** | **0.791** | **0.878** | **0.884** | 0.158 | **8.27** |

GT flicker rate: 0.258. Run 3 coverage: 0.999879.

Energy values are comparable to 2A only up to the added flow unary term; they are not directly comparable to 2C because the unary scale/objective differs.

## Run 3 energy log

| Round | Energy | Flips |
|-------|--------|-------|
| 1 | 3,926,732.5355 | 1,293,977 |
| 2 | 3,926,715.8992 | 3 |
| 3 | 3,926,715.8992 | 0 |

Converged cleanly after round 3.

## Comparison vs 2A

| Metric | 2A | Run 3 | Delta |
|--------|----|-------|-------|
| Mean IoU | 0.780 | **0.794** | +0.014 |
| Median IoU | 0.776 | **0.791** | +0.015 |
| Purity | 0.868 | **0.878** | +0.010 |
| Completeness | 0.875 | **0.884** | +0.009 |
| Pred flicker | **0.154** | 0.158 | +0.004 |
| Boundary ratio | 7.90 | **8.27** | +0.37 |

## Takeaway

The flow endpoint unary improves the 2A objective in the intended way: better cell identity metrics and stronger contour-boundary alignment, with only a small flicker increase. This is now the best configuration by IoU, purity, completeness, and boundary alignment.

Recommended next step: tune `lambda_flow` around this point (`0.25`, `0.5`, `1.0`, `2.0`) while keeping 2A pairwise/temporal settings fixed.
