# Nucleus Workflow Implementation Plan

## Approach

Port the archive's nucleus hypothesis backend into three new v2 modules, keeping the z-axis since v2 consumes full 3D stacks (`nucleus_prob_3dt.tif`, shape T×Z×Y×X). The HDF5 schema simplifies from the archive's `t/z/p` (2D slices) to `t/p` (3D volumes) — each `(t, p)` entry stores the full `(Z, Y, X)` label volume computed by running watershed per z-slice then stacking. This matches the v2 UI which has no z-browser (z is always driven by the napari viewer's current step). The greedy IoU propagator is new — no archive equivalent exists. The `flow_mag` basin is dropped since no dp/flow file is tracked in v2; `basin` is fixed to `"prob"`.

---

## Files Affected

**New backend files:**
- `src/cellflow/segmentation/__init__.py` — watershed core; ported from archive `hypotheses.py`
- `src/cellflow/database/hypotheses.py` — `t/p` HDF5 schema for hypothesis pool; adapted from archive (drops z-axis hierarchy, stores 3D volumes per entry)
- `src/cellflow/database/tracked.py` — `t/labels` HDF5 schema for tracked output; new
- `src/cellflow/tracking/propagator.py` — greedy binary-IoU propagator; new

**Modified:**
- `src/cellflow/database/__init__.py` — re-export public API
- `src/cellflow/tracking/__init__.py` — re-export public API
- `src/cellflow/napari/nucleus_workflow_widget.py` — add `refresh()`, wire all button callbacks
- `src/cellflow/napari/main_widget.py` — call `nucleus_workflow_widget.refresh(pos_dir)` in `_refresh_all()`

---

## Implementation Steps

### Step 1 — `cellflow.segmentation`: Port watershed core

Create `src/cellflow/segmentation/__init__.py`. Port directly from
`archive/v1/packages/ultrack/src/cellflow/ultrack/hypotheses.py` — no logic changes:

| Symbol | Archive lines | Notes |
|--------|--------------|-------|
| `NucleusHypothesisParams` | 31–43 | Keep `basin` field; default `"prob"`. Flow-mag path left in logic but never called since dp=None always. |
| `_normalize_01()` | 150–158 | Port as-is |
| `_flow_magnitude()` | 161–193 | Port as-is (kept for completeness) |
| `_remove_small_labels()` | 196–205 | Port as-is |
| `_peak_local_max_markers()` | 208–217 | Port as-is |
| `compute_hypothesis_labels(prob_2d, dp_2d, markers, params)` | 220–269 | Port as-is — pure 2D computation, no z awareness |

`compute_hypothesis_labels` always receives a 2D slice; the caller (database layer) is responsible for iterating over z and stacking.

### Step 2 — `cellflow.database`: HDF5 I/O with `t/p` schema (3D volumes)

**Schema:** `hypotheses/t{t:03d}/p{p:03d}/labels` where `labels` has shape `(Z, Y, X)` and dtype `uint32`. Parameters stored as group attributes.

Create `src/cellflow/database/hypotheses.py`. Port from archive, replacing the `t/z/p` + 2D slice approach with `t/p` + 3D volume:

| Symbol | Source | Change |
|--------|--------|--------|
| `NucleusHypothesisSweepSpec` | archive lines 47–68 | Port as-is |
| `build_parameter_sets(spec)` | archive lines 98–125 | Port as-is |
| `HypothesisRecord` | archive lines 71–79 | Drop `z` field; `labels` is now 3D `(Z, Y, X)` |
| `write_hypothesis_record(h5, record)` | archive lines 364–390 | Path: `hypotheses/t{t:03d}/p{p:03d}/labels`; no z group |
| `write_hypothesis_sweep_h5(path, records, ...)` | archive lines 394–417 | Remove `n_z`; adapt for no-z |
| `iter_hypothesis_records(path)` | archive lines 420–455 | Adapt to flat `t/p` keys; labels loaded as 3D array |
| `read_hypothesis_labels(path, t, p) -> ndarray` | archive lines 458–461 | Adapt path |
| `list_hypotheses(path) -> (n_p, params_by_p)` | new | Returns count and metadata dict for UI |

Port and adapt `iter_hypothesis_records_from_stacks(prob_stack, dp_stack, seed_stack, spec)` from archive (lines 272–328):
- Input: `prob_stack` shape `(T, Z, Y, X)` (from `nucleus_prob_3dt.tif`)
- `dp_stack=None` always in v2
- Inner loop: for each `(t, p)`, iterate z-slices, call `compute_hypothesis_labels()` per slice, stack into `(Z, Y, X)` volume
- Yield `HypothesisRecord(t, p, labels_3d, params)`

Create `src/cellflow/database/tracked.py` (~50 lines, new):
- `write_tracked_frame(path, t, labels)` — writes `t{t:03d}/labels` with shape `(Z, Y, X)`, gzip compressed
- `read_tracked_frame(path, t) -> ndarray` — reads one 3D frame
- `tracked_n_frames(path) -> int` — counts `t###` groups present

Update `src/cellflow/database/__init__.py` to re-export the public surface.

### Step 3 — `cellflow.tracking`: Greedy IoU propagator

Create `src/cellflow/tracking/propagator.py` (~90 lines, no archive equivalent):

**`find_best_hypothesis(current_labels, candidates, iou_threshold, max_dist_px) -> int | None`**

- Inputs: `current_labels` shape `(Z, Y, X)`, `candidates` list of `(Z, Y, X)` arrays
- Compute centroids of all labelled cells in `current_labels` via `skimage.measure.regionprops` on the max-Z-projection (or iterate per slice — max-projection keeps it 2D and fast)
- For each candidate p: compute its cell centroids on max-projection
- Prefilter: for each candidate, compute mean min-distance from current centroids; skip if > `max_dist_px`
- For surviving candidates: compute 3D binary mask IoU = `((A>0) & (B>0)).sum() / ((A>0) | (B>0)).sum()`
- Return index of candidate with highest IoU ≥ `iou_threshold`, or `None`

**`propagate_one_frame(hypotheses_h5, tracked_h5, t_current, iou_threshold, max_dist_px) -> int | None`**

1. Read current tracked labels for `t_current` from `tracked_h5`
2. Load all `n_p` hypothesis label arrays for `t_current + 1` from `hypotheses_h5`
3. Call `find_best_hypothesis()`
4. If winner found: call `write_tracked_frame()` for `t_current + 1`; return winning p index
5. Return `None` if no suitable hypothesis found

Update `src/cellflow/tracking/__init__.py` to re-export.

### Step 4 — Wire `NucleusWorkflowWidget`

Add state fields:
```python
self._pos_dir: Path | None = None
self._stop_flag: bool = False
```

**`refresh(pos_dir: Path | None)`** — called by main widget on project change:
- Store `_pos_dir`
- If `2_nucleus/hypotheses.h5` exists: call `list_hypotheses()`, set `hyp_spin.setRange(0, n_p - 1)`; update `hyp_meta_lbl`
- If `2_nucleus/tracked_labels.h5` exists: load current-t frame into `Tracked: Nucleus` labels layer in viewer

**Button callbacks (all connected at end of `_setup_ui`):**

| Button | Handler | Thread worker? |
|--------|---------|---------------|
| `preview_btn` | `_on_preview()` | No |
| `save_db_btn` | `_on_save_db()` | Yes |
| `use_as_tracked_btn` | `_on_use_as_tracked()` | No |
| `run_sweep_btn` | `_on_run_sweep()` | Yes |
| `run_terminal_btn` | `_on_run_terminal()` | No (clipboard) |
| `hyp_spin.valueChanged` | `_on_hyp_changed(p)` | No |
| `set_seed_btn` | `_on_set_seed()` | No |
| `prop_next_btn` | `_on_propagate_next()` | No |
| `prop_all_btn` | `_on_propagate_all()` | Yes (checks `_stop_flag`) |
| `stop_btn` | `lambda: setattr(self, '_stop_flag', True)` | No |
| `jump_corr_btn` | stub — show message | No |

**`_on_preview()`:**
1. Guard: `_pos_dir` set; load `1_cellpose/nucleus_prob_3dt.tif` → `(T, Z, Y, X)`
2. Get `t, z` from `viewer.dims.current_step` (axes 0 and 1 respectively)
3. Slice `prob_2d = prob[t, z]` → `(Y, X)`
4. Resolve seed markers from `seed_source_combo`:
   - `Peak local max` → pass `markers=None` (auto-computed in `compute_hypothesis_labels`)
   - `Active Layer` → take current active Labels layer as markers for this z-slice
5. Build `NucleusHypothesisParams(basin="prob", threshold_pct, compactness, smooth_sigma, seed_distance)`
6. Call `compute_hypothesis_labels(prob_2d, None, markers, params)` → `(Y, X)` label array
7. Update/add `Preview: Nucleus` labels layer (2D, for the current z; napari shows it at the current step)

**`_on_save_db()` (thread_worker):**
- Load full `nucleus_prob_3dt.tif` → `(T, Z, Y, X)`
- Build single-param `NucleusHypothesisSweepSpec` from single tab values
- Call `iter_hypothesis_records_from_stacks()`, write to `2_nucleus/hypotheses.h5` via `write_hypothesis_sweep_h5()`
- On finish: call `refresh(self._pos_dir)`

**`_on_use_as_tracked()`:**
- Get all z-slices from `Preview: Nucleus` layer (or stack from viewer) → `(Z, Y, X)` volume
- `t = viewer.dims.current_step[0]`
- Call `write_tracked_frame(tracked_path, t, volume_3d)`
- Update/add `Tracked: Nucleus` labels layer

**`_on_hyp_changed(p)`:**
- Guard: `_pos_dir` set and `hypotheses.h5` exists
- `t = viewer.dims.current_step[0]`
- Call `read_hypothesis_labels(hyp_path, t, p)` → `(Z, Y, X)` volume
- Update/add `Hypothesis: Nucleus` labels layer
- Read params metadata from `list_hypotheses()` → update `hyp_meta_lbl`

**`_on_set_seed()`:**
- Read labels from `Hypothesis: Nucleus` layer for current t (full 3D volume)
- Call `write_tracked_frame(tracked_path, t, volume_3d)`
- Update/add `Tracked: Nucleus` layer

**`_on_propagate_next()`:**
- `t = viewer.dims.current_step[0]`
- Call `propagate_one_frame(hyp_path, tracked_path, t, iou_thr, max_dist)`
- Load t+1 tracked frame, update viewer

**`_on_propagate_all()` (thread_worker):**
- `_stop_flag = False`
- Loop from current t to end of movie: `propagate_one_frame()` then check `_stop_flag`; step viewer time dim after each frame

**`_on_run_sweep()` (thread_worker):**
- Like `_on_save_db()` but reads min/max/step values from batch sweep spinboxes to build a multi-parameter `NucleusHypothesisSweepSpec`

**`_on_run_terminal()`:**
- Generate CLI command string (pattern from `data_prep_widget._on_run_in_terminal()`)
- Copy to clipboard via `QApplication.clipboard().setText(...)`

### Step 5 — Hook into `main_widget.py`

In `_refresh_all()` (around line 286), add:
```python
self.nucleus_workflow_widget.refresh(pos_dir)
```

---

## Risks & Open Questions

1. **Preview shows a single z-slice** — The `Preview: Nucleus` layer only shows `(Y, X)` for the current `(t, z)`. `Save to DB` then computes all z and t. This is intentional (interactive tuning) but users must understand the preview is not the full volume. A status label should say "Previewing t={t}, z={z}".

2. **Viewer dim order** — `viewer.dims.current_step[0]` for t, `[1]` for z. This assumes layers are loaded with T as axis 0 and Z as axis 1, which matches `nucleus_prob_3dt.tif` loaded via tifffile. If any layer has a different order, the indices will be wrong. A guard (`len(viewer.dims.current_step) >= 2`) should be added.

3. **`iter_hypothesis_records_from_stacks()` z-loop** — The archive version yields one `HypothesisRecord` per `(t, z, p)`. The v2 version must accumulate z-slices per `(t, p)` and yield one record with a 3D volume. This is a non-trivial logic change from the archive — not a straight port; needs careful implementation.

4. **HDF5 schema incompatibility with archive** — Any `hypotheses.h5` generated by v1 (`t/z/p` with 2D slices) cannot be read by v2 (`t/p` with 3D volumes). Add a schema-version check: read the root `attrs["layout"]` field; if it contains `z{z:03d}`, show a clear error message.

5. **`write_tracked_frame` for 3D volumes** — A full 3D nucleus label volume per timepoint could be large (e.g., 50 z × 1024 × 1024 × 4 bytes = 200 MB per frame). Gzip compression will help substantially on sparse label data. This is acceptable but worth noting.

6. **Thread safety** — `prop_all_btn` writes to `tracked_labels.h5` from a worker thread. All HDF5 access must use `with h5py.File(...)` (open/close per call, no persistent handles). Verify this pattern in all database functions.

7. **`jump_corr_btn`** — The `CorrectionWidget` is also unwired with no public `activate()` API. This button is a stub for now (prints a message or scrolls) until the correction widget is wired separately.

8. **No dp/flow file** — `basin="flow_mag"` in `NucleusHypothesisParams` will raise a `ValueError` in `compute_hypothesis_labels` since `dp=None`. The UI doesn't expose basin selection, so this will never trigger — but the guard in the segmentation module should have a clear error message in case it's called programmatically.

---

## Testing Strategy

**Unit tests (new file `tests/test_nucleus_workflow.py`):**
- `test_compute_hypothesis_labels`: synthetic 50×50 prob array with 3 Gaussian peaks → verify non-empty uint32 output, correct number of regions
- `test_hypothesis_h5_roundtrip`: write 2 params × 2 timepoints, read back, assert label array equality and all metadata attributes
- `test_tracked_h5_roundtrip`: `write_tracked_frame()` + `read_tracked_frame()` for 3D volumes at t=0 and t=3
- `test_find_best_hypothesis_identical`: candidate[0] identical to current → IoU=1.0, returns 0
- `test_find_best_hypothesis_no_match`: all candidates empty array → returns None
- `test_list_hypotheses`: write sweep, call `list_hypotheses()`, verify n_p and param dict

**Manual napari integration test:**
1. Open plugin, set project dir with `1_cellpose/nucleus_prob_3dt.tif`
2. Preview → `Preview: Nucleus` layer appears at current (t, z)
3. Save to DB → `2_nucleus/hypotheses.h5` created; `hyp_spin` range updates
4. Browse `hyp_spin` → `Hypothesis: Nucleus` layer updates; `hyp_meta_lbl` shows params
5. Set as Tracking Seed → `2_nucleus/tracked_labels.h5` created; `Tracked: Nucleus` layer appears
6. Propagate Next → t+1 written to tracked file; viewer advances

**Regression:** `pytest tests/` — existing 179 tests cover `core.data_prep` and paths; none should be affected since we only add modules and one call in `_refresh_all()`.
