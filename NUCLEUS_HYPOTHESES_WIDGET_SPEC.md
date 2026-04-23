# Nucleus Hypotheses Widget Spec

## Goal

Replace the old nucleus Ultrack widget with a nucleus hypothesis preview widget.
The new widget is a selector-driven watershed preview for generating hypothesis
labelmaps from `1_cellpose/nucleus_dp_4d.tif` and `1_cellpose/nucleus_prob_4d.tif`.

The widget should support previewing a single `(frame, z)` slice and later be
extended to batch hypothesis generation over `z` and parameter sweeps.

## Inputs

- `1_cellpose/nucleus_dp_4d.tif`
- `1_cellpose/nucleus_prob_4d.tif`
- A seed label source for watershed markers

Seed source priority should be configurable, with the current default following:

- active viewer labels layer
- corrected nucleus labels on disk
- flattened nucleus labels from `2_nucleus_ultrack/nuclear_labels_2d.tif`

## Processing Rules

- Compute flow magnitude from the DP stack with an L2 norm over the channel axis.
- Normalize the selected basin image to `[0, 1]` before thresholding.
- Apply the threshold as a percentage in the range `0..100`.
- Threshold the selected basin image, not just the probability map.
- Run `skimage.segmentation.watershed` on the selected basin image.
- Use the selected seed labels as markers.
- Apply optional smoothing to the selected basin image before watershed.
- Preserve preview behavior for a single `(frame, z)` selection.

## UI

The widget should expose:

- frame selector
- `z` selector
- basin selector:
  - `prob`
  - `flow_mag`
- seed source selector
- basin threshold slider/spinbox shown as percent `0..100`
- compactness control
- smooth sigma control
- `Preview` button

## Preview Behavior

- Clicking `Preview` should create a new napari labels layer for the result.
- If preview layers already exist, remove them first and recreate them.
- The preview layers should be:
  - `Preview: Nucleus prob`
  - `Preview: Nucleus flow mag`
  - `Preview: Nucleus hypotheses`
- Re-clicking preview should overwrite the previous preview by deleting the old
  layers first.
- The labels layer should be emitted to downstream correction widgets.

## Filesystem Layout

The preview widget should continue using the existing nucleus stage directory.
The batch sweep will persist to a single HDF5 file under the stage root:

- output root: `2_nucleus_ultrack/`
- sweep file: `2_nucleus_ultrack/hypotheses.h5`

The HDF5 hierarchy is ordered as:

- `hypotheses/t###/z###/p###/labels`

Where:

- `t###` is the timepoint index
- `z###` is the z-slice index
- `p###` is the parameter-set index

Each `p###` group stores the parameter metadata used to generate that label
array. The batch path is intended to write all hypotheses into this one file,
while the preview widget remains a single `(frame, z)` slice preview.

## Integration

Update the following UI entry points:

- `packages/napari-plugin/src/cellflow/napari/analysis_widget.py`
- `packages/napari-plugin/src/cellflow/napari/project_panel.py`
- `packages/napari-plugin/src/cellflow/napari/_plugin.py`
- `packages/napari-plugin/src/cellflow/napari/new_project_dialog.py`

## Future Extension

Batch generation should later:

- loop over `z`
- loop over parameter combinations
- save a set of 2D label TIFFs under a folder
- write a manifest JSON describing each hypothesis

The batch path should be built on top of the same basin normalization and
threshold semantics used by the preview widget.
