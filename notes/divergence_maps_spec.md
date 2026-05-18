# Spec: Divergence-based segmentation maps

Replace the Cellpose-mask sweep that currently builds `contours.tif` and
`foreground_scores.tif` with a direct computation from `prob_3dt` and
`dp_3dt`. The downstream threshold-sweep that produces `*_sources.tif` for
Ultrack is unchanged.

## Background

Current `2_nucleus/foreground_scores.tif` is `mean(masks > 0)` over a
(`cellprob_threshold` × `z`) sweep of Cellpose's `compute_masks`. Current
`contours.tif` is `mean(find_boundaries(masks))` over the same sweep. The
sweep is expensive and parametrised by 4 thresholds × 20 z-slices per frame.

Experiment on `pos00` of
`2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk`
showed:

- `sigmoid(prob).mean(z)` matches the existing `foreground_scores` with
  pearson r = 0.98.
- `clip(div(dp), 0, ∞).mean(z) * foreground` produces visibly sharper
  contour rings than the existing `contour_maps`, with much less spurious
  structure than a max-z reduction.

See `scripts/experiment_nucleus_divergence_contours.py` and the panels in
`.../pos00/2_nucleus/divergence_experiment/`.

## What

A new napari widget that runs after Nucleus Cellpose and Cell Cellpose. It
replaces both `*_prob_zavg.tif` outputs and the mask-sweep step
(`build_nucleus_averaged_maps`).

## Outputs

Written to `1_cellpose/`:

- `nucleus_contours.tif`, `nucleus_foreground.tif`
- `cell_contours.tif`, `cell_foreground.tif`

All are `T × Y × X` float32 stacks.

Dropped:

- `1_cellpose/nucleus_prob_zavg.tif` and `cell_prob_zavg.tif` — replaced by
  `*_foreground.tif`. Note this is a semantic change (logit-mean →
  sigmoid-mean), not a rename: anything thresholding the zavg files needs
  threshold recalibration.
- The legacy filename `2_nucleus/contour_maps.tif` — not supported; only the
  new location is read.

## Signals (per channel)

```
foreground = reduce_z_fg(sigmoid(prob))

for each z:
    flow = dp[z]                              # (2, Y, X), channels [dy, dx]
    if median_radius > 0: flow = median(flow, radius)        # per channel
    if smoothing_sigma > 0: flow = gaussian(flow, sigma)     # per channel
    div_z = ∂dy/∂y + ∂dx/∂x                   # central differences
    pos_z = clip(div_z, 0, ∞)
contours = reduce_z_contour(pos_z over z)
```

Fixed (not exposed):

- Sigmoid is applied before the z-reduction (so `foreground ∈ [0, 1]`
  reads as occupancy; logit-mean was rejected — see "Activation choice"
  below).
- Filter order is median → Gaussian → divergence.
- Rectification of `div` is `clip(·, 0, ∞)`. Negative-divergence regions are
  cell interiors (flow converges to centre); positive-divergence regions
  are boundaries between touching cells (flow flips direction).
- No foreground gating in this version. Deferred.

## UI parameters

Nucleus and cell channels get independent controls (same names per
channel):

| Name                     | Type       | Default | Range          |
| ------------------------ | ---------- | ------- | -------------- |
| `foreground_z_reduction` | enum       | `mean`  | `mean` / `max` |
| `contour_z_reduction`    | enum       | `mean`  | `mean` / `max` |
| `smoothing_sigma`        | float      | `1.0`   | `≥ 0`, 0 = off |
| `median_radius`          | int        | `0`     | `≥ 0`, 0 = off |

## Activation choice (sigmoid-then-mean)

For `z_reduction = max` the activation choice doesn't matter (sigmoid is
monotonic). For `z_reduction = mean` it matters: logit-mean punishes pixels
that are "cell" in only a few in-focus slices because out-of-focus slices
drag the average logit down. Sigmoid-then-mean is closer to "fraction of z
that this pixel sits inside a nucleus" — an occupancy proxy in [0, 1] that
matches the semantics of the current `foreground_scores` (also in [0, 1]).

## Code changes

### New

- `segmentation/divergence_maps.py` with:

```python
def build_divergence_maps(
    prob_path: str | Path,
    dp_path: str | Path,
    contours_out: str | Path,
    foreground_out: str | Path,
    *,
    foreground_z_reduction: Literal["mean", "max"],
    contour_z_reduction: Literal["mean", "max"],
    smoothing_sigma: float,
    median_radius: int,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> DivergenceMapsReport
```

- `napari/divergence_maps_widget.py`, registered in `napari.yaml`. Runs
  per-channel sequentially (nucleus first, then cell) and reports progress
  the same way the cellpose widget does.

### Removed

- From `segmentation/nucleus_segmentation.py`: `build_nucleus_averaged_maps`,
  `build_consensus_boundary`, `apply_gamma`. `compute_contour_watershed`
  and the small helpers around it stay.
- From `napari/nucleus_pipeline_widget.py`: `_on_build_nucleus_maps`,
  `_on_preview_contour_maps`, and the "build maps" branch of
  `_on_build_segmentation_inputs`. The cellprob-threshold / z-indices
  controls go with them. Only the threshold-sweep step that produces
  `*_sources.tif` from `*_contours.tif` and `*_foreground.tif` remains.
- From `napari/cellpose_widget.py` and `scripts/`: zavg-generation code
  (including `precompute_cellpose_probability_zavgs.py`).

### Updated

- `napari/_paths.py`: `contours` and `foreground_scores` (rename to
  `foreground`) properties point to the new `1_cellpose/{channel}_*.tif`
  locations. Per-channel: `nucleus_contours`, `nucleus_foreground`,
  `cell_contours`, `cell_foreground`.
- `napari/data_panel_widget.py`: file entries updated to the new locations.
- `napari/radial_refinement_widget.py`: foreground reference updated.
- `tracking_ultrack/reseed.py`: contour-map path updated.
- Any other call sites surfaced by grepping
  `foreground_scores|contour_maps|prob_zavg`.

## Open items / deferred

- Foreground gating (multiply `contours` by `foreground`) is deferred. Will
  revisit when the new maps are wired in and we can judge whether
  background divergence noise actually hurts the Ultrack source threshold
  sweep.
- If the new `foreground` ends up being a good seed signal for the cell
  workflow's greedy propagator, consider switching that pipeline too.
  Out of scope here.
