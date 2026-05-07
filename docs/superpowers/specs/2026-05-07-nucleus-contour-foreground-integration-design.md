# Nucleus Contour Foreground Integration Design

## Context

The nucleus segmentation and tracking workflow currently has a standalone
Foreground Mask section in `NucleusWorkflowWidget`. That section runs a separate
Cellpose pass on z-averaged probability and flow maps, then writes
`2_nucleus/foreground_masks.tif`. Ultrack database generation and validated
re-solve consume that file alongside `2_nucleus/contour_maps.tif`.

The replacement is to make foreground generation part of the Contour Maps step.
Contour map generation already creates Cellpose label masks for every sampled
gamma, cellprob threshold, and z-slice. Those same masks should define the
foreground estimate, so foreground and contour artifacts come from the same
candidate sweep.

## Goals

- Remove the standalone nucleus Foreground Mask widget.
- Remove backend code that only exists to support that standalone widget.
- Make the Contour Maps step produce contour maps, fuzzy foreground scores, and
  thresholded foreground masks in one pass.
- Keep downstream Ultrack database generation and re-solve consuming
  `foreground_masks.tif` so the downstream file contract stays stable.
- Save the fuzzy foreground score as a diagnostic artifact for threshold tuning.

## Non-Goals

- Do not change the cell workflow foreground mask pipeline.
- Do not redesign Ultrack database construction, solve, export, or DB browsing.
- Do not remove deprecated foreground-related segmentation APIs unless they are
  only used by the removed nucleus foreground widget.
- Do not change the meaning of existing contour-map thresholds or gamma sweep
  controls.

## User Workflow

The `1. Contour Maps` section becomes the single producer for the required
Ultrack segmentation inputs:

- `2_nucleus/contour_maps.tif`
- `2_nucleus/foreground_scores.tif`
- `2_nucleus/foreground_masks.tif`

Add a `Foreground Threshold` control to the Contour Maps section. It thresholds
the fuzzy foreground score and should default to `0.5`, with range `0.0..1.0`
and step `0.01`.

Remove the current `2. Foreground Mask` section, including its threshold, gamma,
niter, preview, build, cancel, status, progress, and file-output UI. DB
generation missing-file messages should direct the user to run Contour Maps when
`foreground_masks.tif` is absent.

## Backend Data Flow

Use the existing contour-mask sweep as the source of foreground votes:

1. For each time frame, iterate over all selected gamma values.
2. For each gamma, iterate over all selected Cellpose cellprob thresholds.
3. For each threshold, run Cellpose mask generation for every z-slice, as the
   contour builder already does.
4. Accumulate boundary votes from `find_boundaries(masks)`, preserving the
   current contour-map behavior.
5. Accumulate foreground votes from `(masks > 0)`.
6. Divide both accumulators by the same total number of generated label masks.

For one frame, the outputs are:

- `boundary`: `float32`, shape `(Y, X)`, values in `[0, 1]`
- `foreground_score`: `float32`, shape `(Y, X)`, values in `[0, 1]`
- `foreground_mask`: `uint8`, shape `(Y, X)`, values `0` or `1`

The foreground mask is:

```python
foreground_mask = (foreground_score >= foreground_threshold).astype(np.uint8)
```

For the full movie, the Contour Maps worker stacks each frame and writes:

- `contour_maps.tif`: `float32`, shape `(T, Y, X)`
- `foreground_scores.tif`: `float32`, shape `(T, Y, X)`
- `foreground_masks.tif`: `uint8`, shape `(T, Y, X)`

If a single-frame input is normalized to a time stack internally, the output
should still use the existing contour output convention for that path.

## API Shape

Prefer accumulator-style APIs over returning all label masks. Keeping the full
generated label set with axes `(gamma, threshold, z, y, x)` in memory is
unnecessary for the main pipeline and can be large. Use a per-frame foreground
accumulator instead.

The consensus boundary helper should return foreground score with the boundary,
for example:

```python
boundary, foreground_score = build_consensus_boundary(...)
```

or, if clearer for call sites:

```python
boundary, foreground_score, foreground_mask = build_consensus_boundary(..., foreground_threshold=...)
```

The implementation plan should choose the smaller signature change after
checking call sites. The important contract is that foreground scores are
accumulated during the existing Cellpose mask loop, not by rerunning Cellpose.

The existing optional source-label saving should continue to work. It may still
receive the generated label masks through the existing callback mechanism.

## Removal Scope

Remove from `src/cellflow/napari/nucleus_workflow_widget.py`:

- foreground section construction
- foreground worker state
- foreground preview/build/cancel handlers
- foreground status/progress helpers
- signal connections for foreground controls
- state persistence for the old `foreground_mask` section
- refresh calls for the old foreground output widget

Remove from `src/cellflow/segmentation/__init__.py`:

- `compute_cellpose_foreground_mask` if no non-widget callers remain
- any foreground helper that is only used by the removed nucleus foreground
  widget

Keep `_foreground_masks_path()` because downstream DB generation, DB browser,
solve, and resolve still refer to the stable `2_nucleus/foreground_masks.tif`
artifact.

## Downstream Behavior

Ultrack database generation remains file-based and continues to require:

- `2_nucleus/contour_maps.tif`
- `2_nucleus/foreground_masks.tif`
- `2_nucleus/nucleus_prob_zavg.tif`

The DB browser can continue loading contour maps and foreground masks as image
layers. It may optionally load `foreground_scores.tif` later, but that is not
required for this change.

Validated re-solve continues to pass `foreground_masks_path` into
`resolve_with_canonical_segment`. Missing foreground messages should mention
that the file is produced by Contour Maps.

## Testing

Backend tests:

- Verify that foreground score averages binary label occupancy across every
  generated mask in the contour sweep.
- Verify that thresholding the score produces the expected `uint8` foreground
  mask.
- Verify that contour boundary behavior is unchanged for a small controlled
  mask set.

Napari tests:

- Assert the standalone `Foreground Mask` section is gone.
- Assert the Contour Maps section exposes `Foreground Threshold`.
- Assert the Contour Maps output file list includes `contour_maps.tif`,
  `foreground_scores.tif`, and `foreground_masks.tif`.
- Assert state persistence stores the new contour foreground threshold and no
  longer stores old foreground widget controls.
- Update DB generation and re-solve missing-file messages to refer to Contour
  Maps.

Integration-level tests:

- Keep existing Ultrack DB build tests focused on consuming
  `foreground_masks.tif`.
- Add or adjust a contour-build test that stubs Cellpose masks and verifies all
  three output stacks are written with the expected shape and dtype.

## Risks

The main risk is accidentally changing the contour map itself while adding
foreground accumulation. Tests should pin a small boundary output before and
after the change.

The second risk is memory use. The implementation should accumulate foreground
votes as `float32` or integer counts per frame and avoid retaining all generated
label masks unless `save_source_check` is enabled.

The third risk is stale UI state. Loading older saved widget state that contains
`foreground_mask` should ignore that block without failing.
