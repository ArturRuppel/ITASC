# TODO: Multi-tissue napari widget integration

## Problem

The napari widget currently works with a single Labels layer or Points layer.
It needs to support loading multiple movies (tissues) and concatenating them
into a 4D dataset for batch processing via `build_from_labels_4d` / `build_from_tracks_4d`.

Movies will have different lengths (number of frames), so we cannot simply
`np.stack` them into a (N_tissues, T, H, W) array — we need to handle ragged
time dimensions.

## Tasks

### 1. Handle variable-length movies in the batch API

- [x] Change `build_from_labels_4d` to accept a list of 3D stacks (List[np.ndarray])
      instead of requiring a single (N, T, H, W) array, since movies may differ in
      frame count
- [x] Similarly update `build_from_tracks_4d` if needed
      (Not needed — already handles variable lengths via positions array)
- [x] Update `TissueGraphDataset` or conventions so that per-tissue frame counts
      are independent (Already supported — each TissueGraphTimeSeries has its own frames dict)
- [x] Add tests with tissues of different lengths (e.g. 5 frames and 8 frames)

### 2. Multi-file loading in the widget

- [x] Add a "Load Labels" button that opens a file dialog for selecting multiple
      .tif files (one per tissue)
- [x] Load each .tif as a separate 3D stack (T, H, W)
- [x] Store the list of stacks internally (do not force them into a single 4D array)
- [x] Display loaded files in a list widget with tissue count and per-tissue frame info
- [x] Add input fields for pixel size (µm/px), time interval (s), condition name

### 3. Update the build pipeline in the widget

- [x] Wire "Build Graph" to call `build_from_labels_4d` (list version) or
      `build_from_tracks_4d` with the loaded multi-tissue data
- [x] Update `GraphBuildWorker` to produce a `TissueGraphDataset` instead of
      a single `TissueGraphTimeSeries`
- [x] Run `detect_all_t1_events` on the full dataset after building
- [x] Update status display to show per-tissue summaries

### 4. Tissue inspection in the widget

- [x] Add a tissue selector (dropdown or slider) to pick which tissue to visualize
- [x] Switching tissues should update the napari layers (junctions, centroids, T1 markers)
      for that tissue's frames
- [x] Add a "Remove tissue" button for QC (calls `dataset.remove_tissue()`)

### 5. Test on real data

- [ ] Test with 2-3 segmentation movies of different lengths
- [ ] Verify frame slider works correctly per tissue
- [ ] Verify T1 detection and trajectory construction work across the dataset
- [ ] Check that removing a tissue updates the display correctly
