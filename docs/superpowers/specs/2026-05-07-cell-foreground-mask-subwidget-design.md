# Cell Foreground Mask Subwidget Design

## Context

The cell segmentation workflow currently has two main sections:

- `Filtered Flow`, which reads Cellpose flow output and writes `3_cell/filtered_dp.tif`
  plus `3_cell/filtered_flow_mag.tif`.
- `Tracked Cell Labels`, which consumes `3_cell/foreground_masks.tif`,
  `3_cell/filtered_dp.tif`, and `2_nucleus/tracked_labels.tif`.

The workflow already expects `3_cell/foreground_masks.tif`, but the cell widget
does not generate it. The new section should create that foreground mask from
the filtered flow field and Cellpose cell probability output, using Cellpose's
native mask dynamics rather than direct probability thresholding.

## Goals

- Add a `Foreground Mask` subwidget to the cell segmentation section.
- Use the z-averaged Cellpose probability logits from
  `1_cellpose/cell_prob_3dt.tif`.
- Use the filtered 2D flow vectors from `3_cell/filtered_dp.tif`.
- Run Cellpose mask generation with one user-selected `cellprob_threshold`.
- Save `3_cell/foreground_masks.tif` as the stable downstream artifact.
- Keep the existing tracked-label step unchanged except for status/UI refreshes.

## Non-Goals

- Do not add a threshold or gamma sweep.
- Do not use raw `1_cellpose/cell_dp_3dt.tif` directly for this foreground step.
- Do not replace the flow-following tracked-label algorithm.
- Do not change the nucleus segmentation foreground pipeline.

## UI

Add a new collapsible `Foreground Mask` section between `Filtered Flow` and
`Tracked Cell Labels` in `src/cellflow/napari/cell_workflow_widget.py`.

Controls:

- `Cellprob threshold`: `QDoubleSpinBox`, default `0.0`, range
  `-10.0..10.0`, step `0.1`.
- `Flow threshold`: `QDoubleSpinBox`, default `0.0`, range `0.0..10.0`,
  step `0.1`. A value of `0.0` leaves Cellpose flow-error filtering disabled.
- `Min size`: `QSpinBox`, default `15`, range `0..100000`.
- `Niter`: `QSpinBox`, default `200`, range `1..2000`.
- Button: `Create foreground_masks`.

The existing shared cancel button and progress/status area can be reused. While
the foreground worker is running, the filtered-flow, foreground, and label
creation buttons should be disabled and `Cancel` enabled.

## Data Flow

Inputs:

- `1_cellpose/cell_prob_3dt.tif`: expected as `(T, Z, Y, X)`, with a missing
  time axis accepted as `(Z, Y, X)`.
- `3_cell/filtered_dp.tif`: expected as `(T, 2, Y, X)`.

Processing:

1. Load the probability stack as `float32`.
2. Add a time axis if the probability stack is 3D.
3. Compute z-averaged logits per frame with `prob_tyx = prob_tzyx.mean(axis=1)`.
4. Load filtered flow vectors as `float32`.
5. Validate that probability and filtered flow agree on `T, Y, X`.
6. For each frame, call `cellpose.dynamics.compute_masks` with:
   - `dp_tcyx[t]`
   - `prob_tyx[t]`
   - the selected `cellprob_threshold`
   - the selected `flow_threshold`
   - the selected `min_size`
   - the selected `niter`
   - `do_3D=False`
7. Convert Cellpose labels to foreground with `(masks > 0).astype(np.uint8)`.
8. Stack frames and write `3_cell/foreground_masks.tif`.

The helper should handle Cellpose return variants by accepting either a mask
array or a tuple whose first item is the mask array.

## Segmentation API

Add a small helper to `src/cellflow/segmentation/__init__.py`, for example:

```python
def compute_cellpose_foreground_masks(
    prob_tzyx: np.ndarray,
    filtered_dp_tcyx: np.ndarray,
    *,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.0,
    min_size: int = 15,
    niter: int = 200,
    progress_cb: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    ...
```

Return contract: `uint8` array shaped `(T, Y, X)` with values `0` or `1`.

This keeps Cellpose-specific mask generation testable outside the widget and
keeps the widget responsible only for file I/O, UI state, and layer updates.

## State Persistence

Extend `CellWorkflowWidget.get_state()` and `set_state()` with a new
`foreground_mask` block:

```json
{
  "foreground_mask": {
    "cellprob_threshold": 0.0,
    "flow_threshold": 0.0,
    "min_size": 15,
    "niter": 200
  }
}
```

Existing `flow_following` state keys remain unchanged.

## Output Display

After a successful foreground build:

- Refresh the input and output file widgets.
- Update the input status label so `foreground` changes to present.
- Add or update a napari labels layer named `Foreground Mask` with the saved
  mask stack.
- Set status to `Foreground masks complete.`

## Error Handling

- If no project is open, report `No project open.`
- If `cell_prob_3dt.tif` is missing, report it as missing.
- If `filtered_dp.tif` is missing, report it as missing and guide the user to
  run `Filtered Flow` first.
- If input shapes do not match, raise a clear `ValueError` that includes both
  shapes.
- If Cellpose or torch is unavailable, raise an `ImportError` explaining that
  Cellpose foreground generation requires those packages.

## Tests

Segmentation tests:

- Stub `cellpose.dynamics.compute_masks` and verify z-averaged logits and
  filtered flow are passed per frame.
- Verify the returned foreground is `uint8`, shape `(T, Y, X)`, and binary.
- Verify shape mismatch errors include useful shape information.

Widget tests:

- Assert the new `Foreground Mask` section exists between `Filtered Flow` and
  `Tracked Cell Labels`.
- Assert default foreground controls and button text.
- Assert state persistence round-trips the `foreground_mask` block.
- With synchronous workers and a stubbed segmentation helper, assert
  `_on_create_foreground_masks()` writes only `3_cell/foreground_masks.tif`,
  updates the viewer layer, and refreshes status.
- Update existing missing-input tests so foreground generation and tracked-label
  generation each fail before compute when required files are absent.

## Risks

The main risk is shape interpretation. The implementation should not guess a
channel axis for `filtered_dp.tif`; that file is produced by the existing
filtered-flow step as `(T, 2, Y, X)`, so the foreground helper should validate
that contract explicitly. Raw Cellpose `cell_dp_3dt.tif` normalization remains
owned by the filtered-flow step.

The second risk is Cellpose API variation. Existing code already handles
Cellpose returning either masks directly or a tuple; the new helper should use
the same pattern.
