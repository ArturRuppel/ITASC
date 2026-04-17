# Ultrareview Findings — 2026-04-17

## Normal Severity

### ~~bug_018 — Cross-position data corruption~~ ✅ FIXED
**File:** `packages/napari-plugin/src/cellflow/napari/tracking_correction_widget.py:60-89`

When the user loads labels for position 0, switches the position spinbox to position 1, and clicks Load again, the widget silently reuses the pos-0 label layer (because nothing clears it on position change). The background image correctly shows pos-1, but the label data being edited is still from pos-0. On Save, pos-0 labels are written to pos-1's output file with no warning.

**Fix:** Subscribe to `position_changed` in `TrackingCorrectionWidget.__init__` and clear `self._data_layer`, remove the "Nuclear Labels" viewer layer, and reset the status label — forcing an explicit Load on the new position.

---

### ~~bug_005 — Infinite recursion in `CorrectionWidget._deactivate`~~ ✅ FIXED
**File:** `packages/napari-plugin/src/cellflow/napari/correction_widget.py:412-420`

`_deactivate` emits `set_tissue_nuclear_labels` *before* clearing `self._layer` (line 375) or calling `_activate_btn.setChecked(False)` (line 385). This triggers `TrackingCorrectionWidget._on_nuclear_labels_changed` → `set_data_layer` → `_deactivate` again in a loop (~140 levels) until Python raises RecursionError — which the surrounding `except Exception: pass` silently swallows. The widget ends up in an inconsistent half-deactivated state.

**Fix:** Move `self._layer = None` and/or `_activate_btn.setChecked(False)` to *before* the emit on line 371, or add an `_in_deactivate` re-entrancy guard at the top of `_deactivate`.

---

### ~~bug_003 — Silent all-zero output when `cellprob_min > cellprob_max`~~ ✅ FIXED
**File:** `packages/cellpose/src/cellflow/cellpose/stages/contours.py:224-228`

If a user sets `cellprob_min > cellprob_max` in the UI (two independent spinboxes with no cross-validation), `np.arange(min, max, step)` returns an empty list. The threshold loop is skipped, `np.mean([], axis=0)` raises, the outer `try/except` catches it, and all-zero frames are written to `foreground.tif` and `contours.tif`. The widget reports "Done" with no error. The pre-PR code had an explicit empty-thresholds guard that was not carried over.

**Fix:** Restore the guard after building `thresholds`:
```python
if not thresholds:
    yield (0, 0, "Error: no cellprob thresholds generated (check cellprob_min/max/step)")
    return
```
Or add a pydantic `@model_validator` on `CellposeContoursConfig` that raises when `cellprob_min > cellprob_max`.

---

## Nit Severity

### ~~bug_007 — `labels_loaded` signal has no subscribers (dead code)~~ ✅ FIXED
**File:** `packages/napari-plugin/src/cellflow/napari/ultrack_widgets/ultrack_widget.py:63-64`

`labels_loaded = Signal(object)` is declared and emitted after Ultrack loads tracked labels, but there are zero `.labels_loaded.connect(...)` calls anywhere in the repo. The likely intent was to auto-feed the tracked labels layer into `TrackingCorrectionWidget._set_data_layer` via `CellFlowWidget`, but that wiring was never added.

**Fix (preferred):** In `CellFlowWidget._build_ui`, add:
```python
self._ultrack_tab.labels_loaded.connect(self._tracking_correction_widget._set_data_layer)
```
**Or:** Delete the signal, its emit, and the `Signal` import.

---

### ~~bug_017 — `_try_load_from_disk` aborts on corrupt file instead of trying fallback~~ ✅ FIXED
**File:** `packages/napari-plugin/src/cellflow/napari/tracking_correction_widget.py:140-147`

The method builds an ordered candidate list (`nuclear_labels_corrected.tif` first, then `nuclear_labels_2d.tif` as fallback). If the first file is corrupt, the `except` block does `return None` instead of `continue`, so the fallback is never tried. The error message is also immediately overwritten by the caller with a generic message, leaving the user no hint that a fallback exists.

**Fix:** Change `return None` to `continue` in the except block. Optionally preserve the error as a warning when the fallback succeeds (e.g., "Corrected file unreadable — loaded 2d fallback").

---

### ~~bug_002 — Loading pre-sweep configs silently zeros cellprob spinboxes~~ ✅ FIXED
**File:** `packages/napari-plugin/src/cellflow/napari/ultrack_widgets/ultrack_widget.py:313-320`

Any config JSON saved before the cellprob-sweep feature was added contains only `cellprob_threshold`, not `cellprob_min/max`. Pydantic fills missing fields with defaults (0.0/0.0). `_cp_ct_apply_config` then sets both spinboxes to 0.0, silently discarding the user's saved threshold.

**Fix:** Add a `@model_validator(mode="before")` on `CellposeContoursConfig` that seeds `cellprob_min = cellprob_max = cellprob_threshold` when `min`/`max` are absent — mirroring the existing `_migrate_legacy_fields` pattern on `FlowWatershedConfig`.

---

### ~~bug_008 — `_deactivate` unconditionally overwrites `state.nuclear_labels`~~ ✅ FIXED
**File:** `packages/napari-plugin/src/cellflow/napari/correction_widget.py:375-383`

"Load from active layer" accepts any `napari.layers.Labels` without checking whether it corresponds to the nuclear labels layer. On deactivate, `_deactivate` unconditionally writes whatever layer was loaded into `state.tissue.nuclear_labels`. If the user accidentally loaded a cell-segmentation layer, the real nuclear labels in state are silently overwritten with no undo.

**Fix:** Gate the write on `self._layer.name == self._state.tissue.nuclear_labels_layer`, or validate at `_on_load_from_layer` that the active layer matches the expected nuclear labels layer.
