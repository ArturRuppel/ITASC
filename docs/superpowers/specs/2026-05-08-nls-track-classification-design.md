# NLS Track Classification H5 Patch Script Design

## Goal

Add a standalone script/CLI that patches an existing step-5 position analysis HDF5 artifact with automatic NLS-high/NLS-low track classifications.

The tool uses:

- `0_input/NLS_zavg.tif` as the NLS intensity image.
- `2_nucleus/tracked_labels.tif` as the nuclear track labels.
- An existing analysis H5 file produced by the position analysis step.

It classifies each nuclear track as NLS-high or NLS-low, then writes biological labels into the H5:

- NLS-high tracks become `ctrl`.
- NLS-low tracks become `vimentin_ko`.

## User-Facing Shape

The first implementation should be a script or module-level CLI, not a napari widget. The workflow is an explicit post-processing step for already-built analysis artifacts:

```bash
cellflow-classify-nls path/to/5_analysis/position_analysis.h5 \
  --nls-zavg path/to/0_input/NLS_zavg.tif \
  --nucleus-labels path/to/2_nucleus/tracked_labels.tif
```

If the project does not yet expose console scripts, the same behavior can initially be provided through a small Python module entry point, for example:

```bash
python -m cellflow.analysis.nls_classification path/to/5_analysis/position_analysis.h5
```

When paths are omitted, the tool may infer them from the H5 provenance or from the position directory layout, but explicit paths should always be accepted.

## Measurement Semantics

Classification is track-first.

The tool must not classify each frame independently. It must first collect all NLS intensity values for all frames and all track IDs, then reduce each track to one scalar value before splitting tracks into two groups.

For each nonzero label ID in the nuclear tracked-label stack:

1. Find every pixel in every frame where `nucleus_tracked_labels[t, y, x] == track_id`.
2. Collect the corresponding values from `NLS_zavg[t, y, x]`.
3. Treat those collected values as the track's NLS intensity distribution.
4. Reduce that distribution to one scalar, defaulting to median intensity.

The resulting classification input is one scalar per track:

```text
track_id -> nls_track_median_intensity
```

The automatic split operates only on this per-track scalar distribution.

## Automatic Split

The default split method should be deterministic and fully automatic. Use an Otsu threshold on the per-track median intensities as the first implementation because it directly targets a bimodal histogram and requires no user parameters.

Tracks with scalar intensity greater than the threshold are `high`; tracks with scalar intensity less than or equal to the threshold are `low`.

The implementation should fail with a clear error rather than silently writing questionable classifications when:

- The NLS image and nuclear label stack shapes do not match.
- There are fewer than two nonzero tracks with sampled pixels.
- All track intensity scalars are identical.
- The computed threshold assigns all tracks to one group.
- The H5 `cells/table/cell_id` values do not overlap any classified nuclear track IDs.

## H5 Writes

The script patches the existing H5 in place by default, with an option to write a copied output file if that is safer for batch use.

For every row in `cells/table`, the row's `cell_id` receives the classification for the matching nuclear track ID. Because CellFlow's analysis artifact already assumes `cell_id == nucleus_id`, this keeps cell rows and nuclear measurements aligned.

Write or replace these `cells/table` columns:

- `class_label`: `ctrl` for NLS-high tracks, `vimentin_ko` for NLS-low tracks.
- `nls_status`: `high` or `low`.
- `nls_track_median_intensity`: the per-track scalar used for thresholding.
- `nls_track_pixel_count`: total sampled nuclear pixels across all frames.
- `nls_track_frame_count`: number of frames where the track contributed at least one nuclear pixel.

Rows whose `cell_id` cannot be classified should receive:

- `class_label`: empty string.
- `nls_status`: empty string.
- numeric audit fields: `NaN` for intensities, `0` for counts.

Store run metadata under `cells/measurements/nls_classification`:

- `method`: `otsu_track_median`.
- `threshold`.
- `high_label`: `ctrl`.
- `low_label`: `vimentin_ko`.
- `nls_zavg_path`.
- `nucleus_tracked_labels_path`.
- `classified_track_count`.
- `high_track_count`.
- `low_track_count`.
- `created_at`.

## Error Handling

Validation should happen before mutating the H5 unless the user explicitly chooses overwrite behavior. When patching in place, compute all classifications and validate target columns before deleting or replacing any H5 datasets.

The CLI should print a concise summary:

- number of tracks measured
- threshold
- number of high tracks
- number of low tracks
- H5 path written

## Components

Add a small analysis helper module, for example `cellflow.analysis.nls_classification`, with testable pure functions:

- `measure_track_nls_intensity(nls_zavg, nucleus_labels)`: returns per-track median, pixel count, and frame count.
- `split_tracks_otsu(track_medians)`: returns threshold and high/low assignments.
- `patch_position_artifact_nls_classes(h5_path, nls_zavg_path, nucleus_labels_path, ...)`: performs validation, measurement, split, and H5 writes.

Keep the CLI thin. It should parse paths and options, call the helper, and print the summary.

## Testing

Unit tests should cover:

- Track-level aggregation across multiple frames.
- Median-based scalar computation from all pixels in all frames for each track.
- Automatic high/low split on a synthetic bimodal distribution.
- H5 patching of `cells/table/class_label` and audit columns.
- Replacement of existing NLS classification columns on rerun.
- Shape mismatch and invalid split errors.
- Unclassified H5 cell rows receiving empty labels and null audit fields.

The tests should use small synthetic arrays and temporary H5 files, avoiding napari.

## Open Decisions

None for the initial spec. The first implementation is a standalone script/CLI, fully automatic, using per-track median NLS intensity and an Otsu split.
