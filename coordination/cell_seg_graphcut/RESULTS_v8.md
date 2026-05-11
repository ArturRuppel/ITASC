# Results — Run 8 Numba ICM solver

**Date:** 2026-05-10  
**Script:** `scripts/experiment_cell_2d_t_multilabel_graphcut.py`  
**Position:** `pos04`  
**Output:** `/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos04/4_cell_graphcut/20260510-pos04-run8-icm`

## Configuration

Same pairwise parameters as Run 6, but solver switched to Numba ICM. Unary is computed
via direct Numba flow integration + nucleus-pixel distance (no `distance_transform_edt`).

```text
solver = icm
unary(p, k) = nb_flow_unary(p, k)   # Numba: flow endpoint → nearest nucleus pixel
lambda_geodesic = 0
lambda_flow = 1
lambda_s = 20
beta_s = 5
lambda_t = 0.5
n_iters = 10
min_round_flips = 100
boundary_mode = foreground_inverse
```

Command shape:

```text
--pos-dir .../analysis/pos04
--solver icm
--unary-mode geodesic_flow
--init-mode unary
--lambda-geodesic 0
--lambda-flow 1.0
--lambda-s 20
--lambda-t 0.5
--boundary-mode foreground_inverse
--n-iters 10
--no-unary-cache
```

## Results

| Run | Solver | Mean IoU | Median IoU | Purity | Completeness | Pred flicker | GT flicker | Boundary ratio | Total time |
|-----|--------|----------|------------|--------|--------------|--------------|------------|----------------|------------|
| pos04 Run 6 | graphcut (Graph[int]) | 0.717 | 0.744 | 0.844 | 0.843 | 0.236 | 0.355 | 1.72 | 372.3s |
| pos04 Run 8 | ICM (Numba) | 0.702 | 0.726 | 0.854 | 0.814 | 0.297 | 0.355 | 1.976 | ~31.3s |

## Timing breakdown (Run 8)

| Phase | Time |
|-------|------|
| Numba flow integration + unary | 27.4s |
| ICM solver (10 rounds) | 3.9s |
| Total | ~31.3s |

Compare to Run 6: unary cache load 7.4s + graphcut 364.9s = 372.3s total. **~12× speedup.**

## Energy log

ICM did not fully converge at 10 rounds (30K+ flips remaining in final round).

## Notes

ICM is dramatically faster than α-expansion (31s vs 372s) but shows quality trade-offs
at 10 rounds:

- **Better:** purity 0.854 vs 0.844 (+0.010)
- **Worse:** completeness 0.814 vs 0.843 (−0.029), pred flicker 0.297 vs 0.236 (+0.061),
  mean IoU 0.702 vs 0.717 (−0.015)

ICM has not fully converged (synchronous Jacobi updates can oscillate). Options to
improve quality or convergence:
1. More rounds (>10) or energy-tolerance stopping
2. Checkerboard (asynchronous) update pattern to reduce oscillation
3. Tighter `lambda_t` tuning to reduce flicker
4. Use ICM as a warm-start initializer for graphcut (fast approximate solution → fewer
   graphcut rounds needed)
