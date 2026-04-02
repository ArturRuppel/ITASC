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

- ~~**Common data IO widget**: Project bar and Database tab merged into a single always-visible
  `ProjectPanel` above the tab widget. Handles Load / Save / Save As, metadata (px, dt,
  condition), tissue table, and dashboard launch. `New` button removed.~~ **Done**

- ~~**Metadata**: Metadata fields (pixel size, time interval, condition) currently live
  only in the Database tab. Move them (or expose them) in a common widget so every tab
  can read them without going through the Database.~~ **Done** — `pixel_size`,
  `time_interval`, `condition` added to `ViewerState`; now live in `ProjectPanel` and
  sync through `ViewerState` to all tabs.

---

## DataBank → Database

- ~~**Common widget**: Evaluate whether the Database widget should become the single
  shared hub for data IO + metadata, replacing the per-widget load/clear buttons. All
  other tabs would read the active dataset/labels from there.~~ **Done** — merged into
  `ProjectPanel`.

- ~~**Pointer-based storage**~~ **Reconsidered** — keeping the single `.h5` format for
  integrity (segmentation and graph data stay atomically linked). Image paths are not
  stored because images are typically loaded from napari layers whose source path is
  unreliable.

- **Multi-file dataset**: Add a meta/project file (e.g. a lightweight JSON or TOML) that
  assembles several `.h5` files into a single logical dataset. The panel should show
  which `.h5` is currently *active* (the one being worked on), and any newly added
  tissues or metadata edits should be written back to that file. Switching the active
  file should be a single click. This enables multi-condition or multi-replicate
  experiments to live as separate `.h5` files but be analysed together.

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

---

## Tagging

- **Tag cells (not only edges)**: Currently tagging is only supported for edges. Extend
  the tagging feature to also support cells (nodes). Where exactly in the codebase /
  UI this feature should live still needs to be determined.
