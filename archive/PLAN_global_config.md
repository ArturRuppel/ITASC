# Plan: Global Parameter Config + Shared Log

## Goal

Replace all per-widget "Save Parameters / Load Parameters" buttons with a single,
project-scoped config file (`cellflow_config.json`). The config is automatically
written to that file every time any run starts, so every run is reproducible. A
shared log viewer (one instance, at the bottom of the plugin) replaces the
per-widget log viewers that currently clutter every subwidget.

---

## Config File

**Location:** `<project_root>/cellflow_config.json`

Auto-saved (overwritten) each time any pipeline stage is run. Also written
explicitly when the user clicks "Save Config" in the project strip.

Loaded automatically when a project is opened (if the file exists). Also loadable
explicitly via "Load Config" button or "Load Config From…" (any path).

### Schema (version 1)

```json
{
  "version": 1,
  "data_prep": {
    "ndtiff_path": "",
    "positions": "0",
    "xy_downsample": 3,
    "overwrite": false
  },
  "cellpose_nucleus": {
    "model": "nuclei",
    "diameter": 17.0,
    "anisotropy": 1.0,
    "min_size": 500,
    "use_gpu": true,
    "gamma": null,
    "overwrite": false
  },
  "cellpose_cell": {
    "model": "cyto3",
    "diameter": 30.0,
    "min_size": 200,
    "use_gpu": true,
    "gamma": null,
    "overwrite": false
  },
  "ultrack": {
    "cp_contours": {
      "cellprob_min": 0.0,
      "cellprob_max": 0.0,
      "cellprob_step": 1.0,
      "do_3D": true,
      "smooth_sigma": 0.5,
      "device": "cuda",
      "save_masks": false,
      "overwrite": false
    },
    "tracking": {
      "min_area": 100,
      "max_area": 1000000,
      "min_frontier": 0.0,
      "threshold": 0.5,
      "ws_hierarchy": "area",
      "anisotropy_penalization": 0.0,
      "n_workers": 1,
      "max_distance": 15.0,
      "max_neighbors": 5,
      "distance_weight": 0.0,
      "link_n_workers": 1,
      "appear_weight": -0.001,
      "disappear_weight": -0.001,
      "division_weight": -0.001,
      "link_function": "power",
      "power": 4.0,
      "bias": 0.0,
      "solution_gap": 0.001,
      "time_limit": 36000,
      "window_size": 0,
      "overwrite_segmentation": false,
      "overwrite_linking": false,
      "overwrite_solve": false
    }
  },
  "flow_watershed": {
    "foreground_sigma": 2.0,
    "foreground_threshold": 0.1,
    "foreground_postprocess_steps": [],
    "flow_scale": 1.0,
    "cellpose_prob_threshold": 0.0,
    "flow_smoothing_sigma": 0.0,
    "max_iterations": 50,
    "uniform_growth_rate": 0.2,
    "flow_mag_scale": 3.0,
    "seg_overwrite": false,
    "postprocess_steps": [
      {"type": "open", "radius": 2, "sigma": 0.0},
      {"type": "close", "radius": 2, "sigma": 0.0},
      {"type": "fill_holes", "radius": 0, "sigma": 0.0},
      {"type": "smooth_boundary", "radius": 0, "sigma": 1.0}
    ],
    "pp_overwrite": false
  }
}
```

**Notes:**
- `version` field allows future migration via `from_dict()`-style logic.
- `flow_watershed` schema is a flattened merge of the existing `FlowWatershedConfig`
  fields plus the two overwrite flags that currently live only in the UI.
- `ultrack.cp_contours` omits `cellprob_threshold` (it is a derived/display value, not
  an input parameter).
- `data_prep.ndtiff_path` saves the last-used path; re-loading it is advisory only (the
  field is editable).

---

## Shared Log Viewer

One `StageLogViewer` instance lives at the bottom of `CellFlowWidget`. It is passed
into every subwidget constructor. Each subwidget stops owning a `_log_viewer` and
instead keeps a reference to the shared one.

The existing `StageLogViewer` API does not need to change. The widget is just created
once and re-used.

---

## Widget Interface: `get_params` / `set_params`

Each pipeline widget gains two public methods:

```python
def get_params(self) -> dict:
    """Return all current UI values as a plain dict (JSON-serialisable)."""

def set_params(self, data: dict) -> None:
    """Apply a params dict to the UI; unknown keys are silently ignored."""
```

These replace the existing per-widget `_on_*_save_params` / `_on_*_load_params`
handlers and the internal `_build_config` / `_apply_config` plumbing (which remains
for run time; these methods just wrap them).

---

## Auto-save Mechanism

Each widget emits a **`run_started`** Qt signal (`Signal()`) immediately before
starting any background worker. `CellFlowWidget` connects each widget's `run_started`
to a single `_autosave_config()` slot that:

1. Calls `get_params()` on every subwidget.
2. Assembles the global dict.
3. Writes `<project_root>/cellflow_config.json`.
4. Logs "Config saved." to the shared log.

If no project is open, the auto-save is skipped silently.

---

## Auto-load on Project Open

`CellFlowWidget._connect_signals()` connects `self._state.project_changed` (or
equivalent) to `_load_config_if_exists()`, which:

1. Reads `<project_root>/cellflow_config.json`.
2. Calls `set_params(section)` on each widget.
3. Logs "Config loaded." to the shared log.

If the file does not exist, no-op.

---

## UI Changes

### Removed from every subwidget
- "Save Parameters…" button and `_on_*_save_params` handler.
- "Load Parameters…" button and `_on_*_load_params` handler.
- Own `StageLogViewer` construction (`self._log_viewer = StageLogViewer()`).
- Constructor-side `_log_section = CollapsibleSection("Log", self._log_viewer)` wrapper.

### Added to `ProjectPanel` (or immediately below it in `CellFlowWidget`)
A compact row of two buttons:
- **"Save Config"** — writes current params to `<project_root>/cellflow_config.json` immediately (same as auto-save, but explicit).
- **"Load Config"** — reads from `<project_root>/cellflow_config.json`.

Optional (lower priority):
- **"Save Config As…"** — QFileDialog, saves to user-chosen path (for cross-project sharing).
- **"Load Config From…"** — QFileDialog, loads from user-chosen path.

### Added to `CellFlowWidget`
- One `StageLogViewer` at the very bottom of the accordion (below ForSys section),
  wrapped in a `CollapsibleSection("Log", self._log_viewer, expanded=True)`.

---

## Files Affected

### `analysis_widget.py` (CellFlowWidget)
- Create shared `_log_viewer = StageLogViewer()` before constructing subwidgets.
- Pass `log_viewer=self._log_viewer` to every subwidget constructor.
- Add "Save Config" / "Load Config" buttons (in a `QHBoxLayout` below `ProjectPanel`).
- Add `_autosave_config()` slot; connect to each widget's `run_started`.
- Add `_load_config_if_exists()` slot; connect to project open event.
- Add Log `CollapsibleSection` at the bottom.

### `ultrack_widgets/data_prep.py` (DataPrepWidget)
- Accept `log_viewer` kwarg; store as `self._log_viewer`.
- Add `get_params() -> dict` and `set_params(data)`.
- Emit `run_started = Signal()` before calling `thread_worker`.

### `ultrack_widgets/cellpose.py` (CellposeWidget)
- Accept `log_viewer` kwarg; remove own log viewer construction.
- Remove s01a and s01b Save/Load buttons and handlers.
- Add `get_params() -> dict` and `set_params(data)`.
- Emit `run_started = Signal()` before any run (s01a and s01b).

### `ultrack_widgets/ultrack_widget.py` (UltrackAnalysisWidget)
- Accept `log_viewer` kwarg; remove own log viewer construction.
- Remove "Save All Parameters" / "Load All Parameters" buttons and handlers.
- Add `get_params() -> dict` (wraps existing `_cp_ct_build_config()` and `_tr_build_config()` into a dict).
- Add `set_params(data)` (wraps existing `_cp_ct_apply_config()` and `_tr_apply_config()`).
- Emit `run_started = Signal()` before any run.

### `ultrack_widgets/flow_watershed.py` (FlowGuidedSegmentationWidget)
- Accept `log_viewer` kwarg; remove own log viewer construction.
- Remove "Save All Parameters" / "Load All Parameters" buttons and handlers.
- Add `get_params() -> dict` (wraps `_build_config().model_dump()` + overwrite flags).
- Add `set_params(data)` (wraps `_apply_config()` + sets overwrite flags).
- Emit `run_started = Signal()` before any run.

### `log_viewer.py` — no changes needed.

### `project_panel.py` — possibly add Save/Load Config buttons here; alternatively they live in `analysis_widget.py`.

---

## Implementation Steps

1. **Add `run_started = Signal()` to each widget** and emit it inside the existing
   run handlers before launching the background worker.

2. **Add `get_params()` / `set_params()` to each widget** using existing
   `_build_config()` / `_apply_config()` internals as the implementation.

3. **Create shared log viewer in `CellFlowWidget`**; update each subwidget
   constructor call to pass `log_viewer=...`; update each subwidget to accept
   and store it instead of constructing its own.

4. **Remove per-widget Save/Load buttons and handlers** (all four files).

5. **Add "Save Config" / "Load Config" row in `CellFlowWidget`** and implement
   `_save_config()` and `_load_config()`.

6. **Wire `run_started` signals** from all widgets to `_autosave_config()`.

7. **Wire project-open event** to `_load_config_if_exists()`.

8. **Add Log `CollapsibleSection`** at the bottom of the accordion in `CellFlowWidget`.

---

## Open Questions / Decisions Needed

1. **"Save Config" button placement**: In the `ProjectPanel` strip (project-level action,
   logically fits) or in its own row between ProjectPanel and the first accordion section?
   Recommendation: below `ProjectPanel`, before the accordion, in its own compact row.

2. **Save Config As / Load Config From**: Include now or defer? These are useful for
   sharing configs across datasets. Recommendation: defer to a follow-up.

3. **What happens to per-widget `_on_*_save_params` for Run in Terminal?**
   Currently, "Run in Terminal" writes a temp JSON to pass params to the CLI. That path
   uses the same `_build_config()` method and is unaffected — keep it as-is.

4. **`data_prep` positions field**: It's a freeform string like `"0,1,2"`. Save and
   restore verbatim — no parsing needed at config level.

5. **Overwrite flags in `flow_watershed`**: Currently not part of `FlowWatershedConfig`.
   They need to be added to `get_params()` explicitly and applied in `set_params()`.
   `FlowWatershedConfig` itself does not need to change.

6. **Log section**: Shared log replaces all four individual log viewers. Should it be
   always expanded (default) or collapsed? Recommendation: expanded by default.

---

## Non-Goals (explicitly out of scope)

- Timepoint customization removal from `data_prep.py` (separate plan exists).
- "Run in Terminal" wiring for Foreground Mask section.
- EdgeAnalysisWidget / ForcesWidget parameter persistence (no params yet).
- Per-run sidecar JSON next to output files (separate from the live config file).
