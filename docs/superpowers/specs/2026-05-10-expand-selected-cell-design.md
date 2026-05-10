# Expand Selected Cell Design

## Context

The cell workflow correction panel already loads tracked cell labels and reference images for manual correction. The previous cell workflow step also produces `3_cell/foreground_masks.tif`, which marks pixels considered cell foreground for each frame.

The new correction action should expand the currently selected cell into nearby foreground background pixels. This is meant as a targeted repair for cells whose tracked label underfills the visible cell area.

## User-Facing Behavior

Add an `Expand Selected Cell` action to the cell correction section. The action operates on the current napari frame and the cell currently selected by the correction widget.

The action requires:

- A selected non-zero cell label.
- An active tracked cell labels layer.
- A foreground mask stack, either already loaded as the `Foreground Mask` layer or readable from `3_cell/foreground_masks.tif`.
- A configurable `Max expansion px` value.

The action should:

- Expand only within the current frame.
- Fill only pixels currently labeled as background.
- Never overwrite other cell labels.
- Stop expansion at the configured maximum distance from the original selected cell mask.
- Record a normal correction undo step.
- Refresh the label layer and selected-cell highlight.
- Report how many pixels were added, or why no change was made.

## Algorithm

For frame `t`, selected label `label_id`, labels frame `seg2d`, and foreground frame `foreground2d`:

1. Build the selected-cell seed:

   ```python
   seed = seg2d == label_id
   ```

2. Build the allowed expansion mask:

   ```python
   allowed = (foreground2d > 0) & ((seg2d == 0) | seed)
   ```

3. Label connected components of `allowed` with 8-connectivity.

4. Keep only the connected component or components touching `seed`.

5. Apply the distance cap from the original seed:

   ```python
   dist = distance_transform_edt(~seed)
   expanded = touching_component & (dist <= max_expansion_px)
   ```

6. Assign only new background pixels:

   ```python
   added = expanded & (seg2d == 0)
   seg2d[added] = label_id
   ```

If `max_expansion_px` is `0`, treat that as unlimited and skip the distance filter.

## Architecture

Place the pure label operation in `cellflow.correction.labels`, for example:

```python
def expand_label_to_foreground(
    seg: np.ndarray,
    foreground: np.ndarray,
    label: int,
    *,
    max_distance: int,
) -> int:
    ...
```

The function mutates `seg` in place and returns the number of newly labeled pixels. It raises `ValueError` when `foreground` does not match `seg` shape. It returns `0` when the label is missing, the selected cell does not touch foreground, or no background pixels can be added.

Wire the action in `CellWorkflowWidget`, because foreground-mask discovery is specific to the cell workflow. The widget should read the active `CorrectionWidget` state, load or find the foreground mask, call the pure operation on the current frame, and use the existing correction history/highlight path.

## UI

In the cell correction section, add:

- `Max expansion px` integer spinbox, default `25`, range `0..999`.
- `Expand Selected Cell` button.

`0` means unlimited expansion within the connected foreground component touching the selected cell. The default is finite to avoid filling large connected foreground regions caused by bridges between touching cells.

## Error Handling

Show a correction status message and make no label changes when:

- No project is open.
- No tracked cell labels layer is loaded.
- No cell is selected.
- The selected cell is not present in the current frame.
- The foreground mask cannot be found.
- The foreground mask shape does not match the label stack.
- The selected cell does not touch any foreground region.
- The expansion adds no pixels.

## Testing

Add focused tests for the pure operation:

- Expands selected label into foreground background pixels.
- Respects `max_distance`.
- Does not overwrite neighboring labels.
- Does not cross foreground background gaps.
- Treats `max_distance=0` as unlimited.
- Raises on shape mismatch.
- Returns `0` without mutation when the selected label is absent or disconnected from foreground.

Add widget-level tests in the cell workflow napari tests:

- The correction section exposes the new spinbox and button.
- Clicking the button expands only the current frame.
- The action uses an already-loaded `Foreground Mask` layer.
- The action falls back to `3_cell/foreground_masks.tif` when the layer is absent.
- Status messages cover missing selection and missing foreground mask.
