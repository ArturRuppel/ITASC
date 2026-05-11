# Results — Run 6 foreground-inverse pairwise signal

**Date:** 2026-05-10  
**Script:** `scripts/experiment_cell_2d_t_multilabel_graphcut.py`  
**Position:** `pos04`  
**Output:** `/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos04/4_cell_graphcut/20260510-pos04-run6-foreground-inverse-pairwise`

## Configuration

Same flow-only unary as Run 5, but spatial pairwise uses `1 - foreground_scores.tif`
instead of `contour_maps.tif`.

```text
unary(p, k) = flow_endpoint_unary(p, k)
spatial_pairwise_signal = 1 - foreground_score
lambda_geodesic = 0
lambda_flow = 1
lambda_s = 20
beta_s = 5
lambda_t = 0.5
n_iters = 5
min_round_flips = 100
```

Command shape:

```text
--pos-dir .../analysis/pos04
--unary-mode geodesic_flow
--init-mode unary
--lambda-geodesic 0
--lambda-flow 1.0
--lambda-s 20
--lambda-t 0.5
--boundary-mode foreground_inverse
--min-round-flips 100
--n-iters 5
```

## Results

| Run | Pairwise signal | Mean IoU vs old labels | Median IoU | Purity | Completeness | Pred flicker | GT flicker | Boundary ratio vs contours | Total time |
|-----|-----------------|------------------------|------------|--------|--------------|--------------|------------|----------------------------|------------|
| pos04 Run 5 | contour map | 0.710 | 0.725 | 0.837 | 0.836 | 0.322 | 0.355 | 2.58 | 386.3s |
| pos04 Run 6 | 1 - foreground score | 0.717 | 0.744 | 0.844 | 0.843 | 0.236 | 0.355 | 1.72 | 372.3s |

The "IoU" values are agreement with the previous pipeline labels, not biological ground truth.

## Energy log

| Round | Energy | Flips | Note |
|-------|--------|-------|------|
| 1 | 980,967.2728 | 856,886 | Main optimization step |
| 2 | 980,755.6372 | 1,965 | Continued improvement |
| 3 | 980,755.6372 | 0 | Converged |

## Notes

The foreground-inverse pairwise signal slightly improves agreement with the old labels and
substantially lowers flicker relative to the contour-pairwise `pos04` run. The contour-based
boundary-alignment metric drops, but that metric is no longer the optimized boundary signal,
so it should not be interpreted as a direct quality regression by itself.

Only `pos04` currently has `3_cell/foreground_scores.tif` in this dataset. Other positions
need that file before this run can be replicated there.
