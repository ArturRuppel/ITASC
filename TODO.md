# CellFlow TODO

## Tracking

- ~~**Tracks layer**: After tracking completes, create a dedicated napari `Tracks` layer
  showing the cell trajectories. This layer should be overwritten (not duplicated) when
  tracking is re-run. It should also be regenerated when the segmentation labels change
  (e.g. after corrections), by wiring a callback on the labels layer or by exposing a
  `rebuild_tracks_layer()` function that other widgets can call.~~ **Done** (commit 5b88fb7)

---

## Data IO / Shared State

- ~~**Correction widget**: `CorrectionWidget` loads data by picking existing napari layers
  via a `QComboBox`; it does not participate in the shared `ViewerState` / registry
  system used by the other widgets. Audit and align its data-loading pattern with the
  rest of the plugin.~~ **Done** — `CorrectionWidget` now receives `seg_tab`, auto-selects
  `seg_tab._seg_layer` on `showEvent`/refresh, matching the `TrackingTab` pattern.

- **Common data IO widget**: Consider extracting a shared "Data IO" panel (load image,
  load labels, pixel size, time interval) that all tabs — segmentation, tracking,
  correction, edge analysis — embed rather than each reinventing the same UI.

- ~~**Metadata**: Metadata fields (pixel size, time interval, condition) currently live
  only in the Database tab. Move them (or expose them) in a common widget so every tab
  can read them without going through the Database.~~ **Done** — `pixel_size`,
  `time_interval`, `condition` added to `ViewerState`; `DataBankWidget` writes to the
  state on edit/load; `analysis_widget` reads from `self._state` directly.

---

## DataBank → Database

- **Common widget**: Evaluate whether the Database widget should become the single
  shared hub for data IO + metadata, replacing the per-widget load/clear buttons. All
  other tabs would read the active dataset/labels from there.

- **Pointer-based storage**: Instead of copying edge data into the dataset, the Database
  could store pointers (file paths) to the raw data and read them on demand. This
  keeps memory usage low and makes it easy for the user to make corrections outside
  the plugin and reload without re-running the full pipeline.

---

## Batch Mode

- Design a batch processing mode: run segmentation → tracking → graph extraction on a
  list of files/directories without manual intervention.

- This requires first settling on the data IO story (see above): how intermediate
  results are saved between stages, and where the Database stores its pointers.

- The user should be able to interrupt the batch at any stage, make manual corrections
  (via the correction widget), save, and resume.

---

## UI / UX


