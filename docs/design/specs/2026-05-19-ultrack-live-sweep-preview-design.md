# Ultrack Live Sweep Preview

**Date:** 2026-05-19
**Scope:** Nucleus Ultrack input preview and database generation flow.

## Problem

The nucleus workflow currently materializes threshold-expanded Ultrack inputs as
`2_nucleus/contour_sources.tif` and `2_nucleus/foreground_sources.tif`. These
files duplicate data derived from `1_cellpose/nucleus_contours.tif` and
`1_cellpose/nucleus_foreground.tif`, then the database-generation step reads the
files back to run Ultrack segmentation per threshold source.

The files were useful as a way to inspect what the threshold controls mean, but
they make the pipeline heavier than necessary. The user wants the threshold
sweep to be visible in napari without creating persistent intermediate TIFFs.

## Goals

- Replace the source-stack file generation stage with a live in-memory preview.
- Use one preview/run button for the Ultrack Inputs row.
- Show contours and foreground together as two napari layers.
- Preview the whole threshold sweep, not just one threshold pair.
- Stop writing and depending on `contour_sources.tif` and
  `foreground_sources.tif` in the canonical workflow.
- Build the Ultrack database directly from the canonical contour/foreground
  maps and the current threshold controls.

## Non-goals

- No change to how divergence maps are created.
- No change to Ultrack's temporary working databases or merged `data.db`.
- No new persistent preview/export artifact in the normal pipeline.
- No attempt to reuse preview layer data as DB-build input.

## User Experience

The Nucleus workflow keeps the existing three stage rows:

```
Ultrack Inputs      ⚙   ▶
Ultrack database    ⚙   ▶
Ultrack solve       ⚙   ▶
```

The Ultrack Inputs row no longer means "write source TIFFs." Clicking its run
button computes the full threshold sweep in memory and adds or updates two
viewer layers:

- `Ultrack Sweep: Contours`
- `Ultrack Sweep: Foreground`

Both layers have the same leading sweep axis. Each sweep plane corresponds to
one `(contour_threshold, foreground_threshold)` pair in the same ordering used
for database generation. The napari layer controls can scrub both sweep index
and time index.

The layer metadata records the threshold pair list so downstream UI code and
debugging can identify which sweep index represents which parameter pair.

## Data Flow

### Preview

1. The user selects a position directory and adjusts the source sweep controls.
2. Clicking the Ultrack Inputs button validates that
   `1_cellpose/nucleus_contours.tif` and `1_cellpose/nucleus_foreground.tif`
   exist.
3. The worker reads both maps from disk.
4. The worker calls the existing threshold-sweep construction logic in
   `cellflow.tracking_ultrack.multi_threshold`.
5. The widget adds or updates the two preview layers.
6. No files are written.

The in-memory preview shape remains equivalent to the old files:

```
contours:   P x T x Y x X, float32
foreground: P x T x Y x X, uint8
```

where `P = len(contour_thresholds) * len(foreground_thresholds)`.

### Database Generation

The Ultrack database row reads the canonical maps directly:

- `1_cellpose/nucleus_contours.tif`
- `1_cellpose/nucleus_foreground.tif`

It applies the same threshold sweep in its worker and runs the same per-source
Ultrack segmentation and merge process that currently happens after loading the
source TIFFs. The database worker does not consume preview layers. This avoids
stale-preview coupling: changing thresholds or source files before DB generation
always affects the DB build through the current controls and current files.

## Architecture

### Core Compute

`src/cellflow/tracking_ultrack/multi_threshold.py` keeps the existing pure
threshold helpers:

- `build_ultrack_source_stacks`
- `threshold_contour_sources`
- `threshold_foreground_sources`

Add or expose an array-based DB entry point that takes the canonical contour and
foreground arrays plus threshold lists, builds the thresholded source arrays in
memory, and then runs the existing per-source segmentation/merge path.

The old file-based functions may remain temporarily for compatibility, but the
nucleus GUI should stop calling them.

### Napari Widget

`src/cellflow/napari/nucleus_pipeline_widget.py` changes ownership of the
Ultrack Inputs row:

- `_on_build_segmentation_inputs` becomes a preview-sweep handler.
- It reads source maps, computes the full sweep in a worker, and updates both
  napari layers.
- Its status text should describe preview work, not file generation.
- Its error handling should still cover missing source maps, invalid threshold
  ranges, cancellation, and worker errors.

Layer-update helper behavior:

- If the target layer already exists as an image layer, replace its data and
  metadata.
- If a layer with the same name exists but has an incompatible type, remove and
  recreate it.
- Contours should be an image layer.
- Foreground can be an image layer using binary `uint8` data; avoid labels if
  that makes the sweep dimension harder to inspect.

### Artifact Paths and File Tracking

`NucleusArtifactPaths.contour_sources` and `.foreground_sources` become
non-canonical for the GUI. They may be removed or left as deprecated aliases if
other code/tests still need a compatibility path during the transition.

The Pipeline Files panel removes:

- `2_nucleus/contour_sources.tif`
- `2_nucleus/foreground_sources.tif`

The Intermediates section should show `2_nucleus/ultrack_workdir/data.db` as
the database artifact after the canonical contour/foreground maps.

## Error Handling

- Missing position directory: `No project open.`
- Missing contours: `Missing: nucleus_contours.tif - build divergence maps first.`
- Missing foreground: `Missing: nucleus_foreground.tif - build divergence maps first.`
- Invalid threshold ranges: surface the existing `ValueError` message.
- Preview cancellation: clear progress, restore row state, show `Cancelled.`
- Memory pressure: for the first implementation, fail with the worker exception
  and show it in status. A later improvement can estimate bytes before compute
  and warn when the sweep is too large.

## Testing

Add or update focused tests for:

- Ultrack Inputs no longer calls `write_ultrack_source_stacks`.
- Ultrack Inputs reads canonical maps and updates two viewer layers.
- Preview layers have matching shapes, names, and threshold metadata.
- DB generation no longer requires `contour_sources.tif` or
  `foreground_sources.tif`.
- DB generation calls the new array/direct threshold DB builder with canonical
  contour and foreground paths or arrays plus threshold lists.
- Pipeline Files no longer lists source-stack TIFFs.
- Missing canonical maps produce clear status messages.

Core compute tests should cover the new array-based DB entry point with stubs
for Ultrack segmentation, so the test suite does not require a real Ultrack
install for the control-flow contract.

## Migration Notes

Existing project directories may still contain old source-stack TIFFs. The new
GUI ignores them. Users can delete them manually if they want to reclaim disk
space, but the workflow should not delete old files automatically.

Any external scripts that call `write_ultrack_source_stacks` can continue to do
so if the function remains. The GUI contract changes from "build files, then
build DB from files" to "preview in memory, then build DB from canonical maps."
