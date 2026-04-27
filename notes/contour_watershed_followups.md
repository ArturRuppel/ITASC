# Contour-watershed follow-ups

Items deferred after the EDT-on-carved-mask seeding change in
`compute_contour_watershed`.

## 1. Restore noise impact on hypothesis diversity

`noise_scale` / `noise_blur_sigma` lost most of their effect because:

- the seeding image is now `EDT(fg_mask & (boundary < ridge_threshold))`, and the
  EDT peak sits at the *interior centroid* of each component — wiggling the
  mask outline by ±1–2 px barely moves it;
- `boundary` is never perturbed, so the topology of `core` (component count,
  connectivity, where carving lines fall) is deterministic across runs.

Fix: apply the same correlated-noise perturbation to `boundary` before the
carving step, using the existing `noise_scale` / `noise_blur_sigma` knobs (so
their meaning stays "perturbation magnitude," no new parameter). Ridges right
at `ridge_threshold` will then sometimes fall below (more aggressive split)
and sometimes above (cells merge), which is exactly the structural
uncertainty the sweep should explore.

Optional complement: also jitter `ridge_threshold` itself within a small band
each run.

## 2. Simplify the foreground / mask pathway

Foreground now only serves to build `fg_mask`; it is no longer the seeding
image. So the per-frame `foreground_maps.tif` step is overkill — we don't need
the smooth sigmoid surface anymore, just a binary mask.

Replace it with a mask built directly during the consensus-boundary stage:
union of all Cellpose-generated labels across (threshold × z), with an outlier
filter (drop tiny / improbably-shaped components before unioning). This:

- removes the `sigmoid(z-mean of gamma-corrected logits)` computation,
- removes the `foreground_threshold` parameter from the watershed step,
- replaces the threshold-on-a-soft-map mask with a consensus mask that
  already reflects "where Cellpose thinks there are cells across many
  configurations."

`build_consensus_boundary` would return `(boundary, fg_mask)` instead of
`(boundary, foreground)`. Storage becomes `2_nucleus/foreground_mask.tif`
(uint8/bool) instead of `foreground_maps.tif` (float32).

Implications to handle:
- `ContourWatershedParams.foreground_threshold` becomes obsolete — drop it
  (and its sweep dimension in `ContourWatershedSweepSpec` and the widget).
- Outlier filter needs at least a min-size and probably a min-circularity
  knob; pick conservative defaults so we don't lose real cells.
- `fg_mask = binary_fill_holes(...)` and the `opening(disk(2))` under
  `noise_scale > 0` move into the consensus stage (or get dropped if the
  union is already clean).
