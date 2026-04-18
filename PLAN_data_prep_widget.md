# Plan: Prepare Input Data Widget Improvements

## Approach

Surface ndtiff metadata in the UI before and after export, remove the timepoints
customization entirely (always export all), add position discovery by auto-populating
the positions field and triggering on path change, add a "Pull Metadata" button,
validate positions exist before exporting, store the downsample-corrected pixel size,
and add a CLI entrypoint to `raw_import.py` to enable "Run in Terminal". All changes
are isolated to two files plus a minor fix in `DatasetConfig`.

---

## Files Affected

- `packages/cellpose/src/cellflow/cellpose/stages/raw_import.py`
  - Store downsample-corrected pixel size in `run_params.json`
  - Add `__main__` CLI entrypoint
  - Add `discover_metadata()` helper callable from the widget without running the full export
  - Add position validation before any files are written

- `packages/cellpose/src/cellflow/cellpose/config.py`
  - Remove `timepoints` field from `DatasetConfig` (run function already handles None → all timepoints)

- `packages/napari-plugin/src/cellflow/napari/ultrack_widgets/data_prep.py`
  - Remove timepoints UI
  - Add "Pull Metadata" button + auto-trigger on path change (300 ms debounce)
  - Show discovered px size, time interval, and positions
  - Add "Run in Terminal" (all positions sequential in one terminal)
  - Fix `_try_load_metadata` to always overwrite state
  - Add position validation

---

## Implementation Steps

1. **`raw_import.py` — add `discover_metadata()` helper**
   - Opens `Dataset(path)`, returns a dict:
     `{"positions": [...], "pixel_size_um": float|None, "time_interval_s": float|None}`
   - Reuses existing extraction logic, no side effects, safe to call from background thread
   - Position key fallback: `axes.get("position", axes.get("p", [0]))`

2. **`raw_import.py` — fix pixel size stored in `run_params.json`**
   - Change `run_params["pixel_size_um"] = pixel_size_um`
     to `pixel_size_um * config.xy_downsample`
   - Stored value now reflects actual exported (downsampled) resolution

3. **`raw_import.py` — add position validation in `run()`**
   - After opening `ds`, check `pos in axes.get("position", axes.get("p", [0]))`
   - Raise `ValueError(f"Position {pos} not found in dataset (available: {available})")`
     before any file is written

4. **`raw_import.py` — add `__main__` CLI entrypoint**
   - Arguments: `--ndtiff-path`, `--root-dir`, `--pos` (int, repeatable for multiple positions),
     `--xy-downsample` (int, default 3), `--overwrite`
   - Loops over all `--pos` values sequentially, printing `[done/total] label` per step

5. **`config.py` — remove `timepoints` from `DatasetConfig`**
   - Delete the `timepoints: Optional[list[int]] = None` field
   - Update `_build_config` in widget (remove `tp = ...` logic)

6. **`data_prep.py` — remove timepoints UI**
   - Remove `_tp_all_check`, `_tp_edit`, `_on_tp_toggle`
   - Remove `tp = None if ...` from `_build_config`

7. **`data_prep.py` — add metadata display labels**
   - Add two read-only `QLabel`s in a row below NDTiff path:
     "Pixel size: —" and "Interval: —"
   - Updated after successful metadata pull

8. **`data_prep.py` — add "Pull Metadata" button and auto-trigger**
   - Add "Pull Metadata" button in NDTiff path row (after "Browse…")
   - Connect `_ndtiff_edit.textChanged` to a 300 ms single-shot `QTimer` debounce
     that calls the same `_pull_metadata` slot
   - `_pull_metadata` runs `discover_metadata()` in a `thread_worker`
   - On result: populate positions text field, update px/interval labels

9. **`data_prep.py` — add "Run in Terminal" button**
   - Place beside "Run Export" in `QHBoxLayout`
   - Builds a single shell command that runs all selected positions sequentially:
     one `python -m cellflow.cellpose.stages.raw_import` invocation per position,
     chained with `&&` in a single terminal
   - Command includes: `--ndtiff-path`, `--root-dir`, `--pos N`, `--xy-downsample N`,
     optionally `--overwrite`

10. **`data_prep.py` — fix `_try_load_metadata` to always overwrite**
    - Remove `and self._state.pixel_size is None` guard
    - Remove `and self._state.time_interval is None` guard
    - Always push discovered values to state after export

11. **`data_prep.py` — surface per-position run errors**
    - Position existence is validated inside `run()` and raises `ValueError`
    - Existing `_on_error` callback surfaces it in the status label — no extra work needed

---

## Risks & Open Questions

- **`timepoints` removal from `DatasetConfig`**: confirm field is not referenced outside
  `raw_import.py` and `data_prep.py`. Serialized `run_params.json` files won't break
  at runtime but the field will no longer be written going forward.

- **ndtiff `axes` position key name**: may be `"position"` or `"p"` depending on ndtiff
  version. `discover_metadata()` and `run()` both use the same fallback chain.

- **Debounce pileup**: if path is on a slow mount, `textChanged` could queue multiple
  background workers. Mitigate by cancelling any in-flight metadata worker before
  starting a new one.

- **Sequential terminal command length**: if many positions are selected, the `&&`-chained
  command string could get long but is not a practical issue for typical position counts.

---

## Testing Strategy

- `python -m cellflow.cellpose.stages.raw_import --help` — confirm CLI loads
- Manually type a valid NDTiff path — verify positions/px/dt labels auto-populate within ~300 ms
- Click "Pull Metadata" explicitly — verify same result
- Export a known dataset, open `run_params.json` — confirm `pixel_size_um = raw_px * downsample`
- Enter a non-existent position, click Run — verify error message, no empty folder created
- `pytest packages/cellpose` — confirm no regressions

---

## Decisions Recorded

- "Run in Terminal" launches **one terminal with all positions chained sequentially** (not
  one terminal per position), per user preference.
- Custom timepoints UI removed entirely; all timepoints always exported.
- Metadata auto-pulled on path change (300 ms debounce) AND via explicit button.
- Channel selection (nucleus/cell channel indices) is out of scope.
- `_try_load_metadata` always overwrites state; no `is None` guards.
