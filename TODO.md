# CellFlow TODO

## Tracking

- **Tracks layer**: After tracking completes, create a dedicated napari `Tracks` layer
  showing the cell trajectories. This layer should be overwritten (not duplicated) when
  tracking is re-run. It should also be regenerated when the segmentation labels change
  (e.g. after corrections), by wiring a callback on the labels layer or by exposing a
  `rebuild_tracks_layer()` function that other widgets can call.

---

## Data IO / Shared State

- **Correction widget**: `CorrectionWidget` loads data by picking existing napari layers
  via a `QComboBox`; it does not participate in the shared `ViewerState` / registry
  system used by the other widgets. Audit and align its data-loading pattern with the
  rest of the plugin.

- **Common data IO widget**: Consider extracting a shared "Data IO" panel (load image,
  load labels, pixel size, time interval) that all tabs — segmentation, tracking,
  correction, edge analysis — embed rather than each reinventing the same UI.

- **Metadata**: Metadata fields (pixel size, time interval, condition) currently live
  only in the Database tab. Move them (or expose them) in a common widget so every tab
  can read them without going through the Database.

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

## Correction Widget

- **`draw_cell_path` (Shift+Left-drag) — rework behaviour and line thickness**:
  Current implementation (`backend/labels.py:draw_cell_path`) thickens the stroke with
  `binary_dilation(radius=3)` and assigns those pixels directly. Intended behaviour:
  1. Draw a **thin 1-px line** (identical to `split_draw` / `_draw_line`) — no dilation.
  2. Use the line as a **barrier**: flood-fill outward from the existing selected cell's
     pixels, blocked by the line and by any other non-zero label. Assign all reached
     background pixels to the cell label (so the enclosed region between the drawn line
     and the cell boundary is filled in).
  3. When no cell is selected (`curlabel=None`), keep creating a new cell — seeding from
     the drawn path itself and flood-filling the enclosed area, or falling back to a
     thin-stroke new cell if the path doesn't enclose anything.
  - The draw-preview layer in `correction_widget.py` already uses `edge_width=1`; no
    change needed there — the preview visually matches the thin line.

---

## UI / UX

- **Scrollable widgets**: All tab widgets should be wrapped in a `QScrollArea` so
  content is accessible when the napari dock panel is narrow. `ForcesWidget` and parts
  of `SegmentationTab` already do this — apply the same pattern everywhere.

- **Collapsible parameter sections**: Replace the `QGroupBox.setCheckable(True)` pattern
  (checkbox toggle) used for parameter panels with a disclosure triangle / arrow button
  (`QToolButton` with `setArrowType`) for a more conventional look.

