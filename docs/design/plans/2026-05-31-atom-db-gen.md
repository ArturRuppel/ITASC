# Spec ② Implementation Plan — Candidate DB-Gen from atoms.tif

**Date:** 2026-05-31
**Branch state:** spec ①-cleanup changes uncommitted; partial spec ② changes also uncommitted (see diff state below)

---

## Diff state entering this plan

Already landed in the working tree (not yet committed):

| File | Status |
|---|---|
| `atoms.py` | `atom_adjacency` + `enum_connected_unions` added ✅; `extract_atoms_stack_with_maps` accidentally widened to 4 returns — **regression, must fix** |
| `config.py` | `atom_union_max_atoms: int = 3`, `atom_union_max_area: int = 8000` added ✅ |
| `db_build.py` | `AtomUnionDatabaseBuildReport` + `build_atom_union_database` added ✅ |
| `nucleus_tracking_inputs_widget.py` | `atom_union_max_atoms_spin` / `atom_union_max_area_spin` controls added ✅; grid layout and `db_gen_config()` **not yet updated** |
| `nucleus_atom_extraction_widget.py` | spec ①-cleanup UX changes (pre-existing) |
| `nucleus_workflow_widget.py` | spec ①-cleanup changes (pre-existing) |
| `test_nucleus_atom_extraction_widget.py` | spec ①-cleanup tests (pre-existing) |
| `test_atoms.py` | spec ①-cleanup tests (pre-existing) |

---

## Step 1 — Fix `atoms.py` regression

`extract_atoms_stack_with_maps` was widened to return 4 values
`(atoms, territory, residual_foreground, residual_contour)`.
This breaks:
- `nucleus_atom_extraction_widget.py` line 313 — unpacks exactly 3
- `test_atoms.py::test_extract_atoms_stack_with_maps_dtypes_and_consistency` — unpacks exactly 3

**Fix:** revert to 3-return signature `(atoms, territory, residual_contour)`, matching the
spec ①-cleanup design and all existing callsites.

---

## Step 2 — Wire new controls into the tracking inputs widget

**File:** `napari/nucleus_tracking_inputs_widget.py`

### Grid layout
Replace the old "Source Thresholds" + higra "Candidates" rows in `db_grid` with:

```
[Candidates header]
Min area:   [db_gen_min_area_spin]     Max atoms: [atom_union_max_atoms_spin]
Max union area: [atom_union_max_area_spin]
```

The dead backward-compat controls (`db_gen_min_frontier_spin`, `db_gen_ws_hierarchy_combo`,
`db_gen_n_workers_spin`, `db_gen_max_area_spin`) stay as widget attributes but are not added
to the grid. The Source Thresholds section (threshold pair table, preview checkbox,
add/remove/clear buttons) is removed from the grid entirely.

### `db_gen_config()`
Update to set:
- `min_area=self.db_gen_min_area_spin.value()` (was mapping to `seg_min_area` only)
- `seg_min_area=self.db_gen_min_area_spin.value()` (keep for compat)
- `atom_union_max_atoms=self.atom_union_max_atoms_spin.value()`
- `atom_union_max_area=self.atom_union_max_area_spin.value()`

Remove from config call: `seg_min_frontier`, `seg_ws_hierarchy`, `seg_n_workers`, `seg_max_area`.

---

## Step 3 — Update workflow widget aliases + signals

**File:** `napari/nucleus_workflow_widget.py`

### `_alias_tracking_inputs_controls`
- **Remove** aliases: `db_gen_max_area_spin`, `db_gen_min_frontier_spin`,
  `db_gen_ws_hierarchy_combo`, `db_gen_n_workers_spin`,
  `source_contour_threshold_spin`, `source_foreground_threshold_spin`,
  `source_threshold_preview_check`, `source_threshold_add_btn`,
  `source_threshold_remove_btn`, `source_threshold_clear_btn`,
  `source_threshold_pairs_table`, `source_threshold_status_lbl`,
  `current_threshold_pair`, `threshold_pairs`, `set_threshold_pairs`,
  `add_threshold_pair`, `remove_selected_threshold_pair`, `clear_threshold_pairs`
- **Add** aliases: `atom_union_max_atoms_spin`, `atom_union_max_area_spin`

### `_connect_signals`
Remove the three source-threshold-preview connections:
```python
self.source_threshold_preview_check.toggled.connect(...)
self.source_contour_threshold_spin.valueChanged.connect(...)
self.source_foreground_threshold_spin.valueChanged.connect(...)
```

### `_active_viewer_activity`
Remove `("source_preview", self.source_threshold_preview_check.isChecked())` entry.

### `_sync_viewer_activity_controls`
Remove all `source_threshold_preview_check` references and the entire `source_active` branch.
Remove `_on_guarded_source_threshold_preview_toggled` and `_cancel_source_threshold_preview`.

---

## Step 4 — Replace `_on_run_db_generation` in the pipeline widget

**File:** `napari/nucleus_pipeline_widget.py`

### Imports
- Add: `from cellflow.tracking_ultrack.db_build import build_atom_union_database`
- Remove: `build_ultrack_database_from_threshold_pairs` from the multi_threshold import
  (keep `build_ultrack_source_stacks_from_pairs` — still used by seg-preview handler)

### New helper
```python
def _atoms_path(self) -> Path | None:
    return self._paths.nucleus_atoms if self._paths else None
```

### `_on_run_db_generation` — new logic
1. Guard: `atoms_path = self._atoms_path()` → if not found or not exists:
   `self._status("Missing: atoms.tif — run Atom Extraction first."); return`
2. Remove: threshold-pairs non-empty check
3. Worker calls:
   ```python
   build_atom_union_database(atoms_path, working_dir, cfg, progress_cb)
   ```
   then (if foreground exists):
   ```python
   apply_annotations_and_score(working_dir, cfg, score_signal_path=foreground_path, ...)
   ```
   Skip scoring if `foreground_path` is `None` or does not exist.

### Remove (no longer needed)
- `_on_preview_threshold_pair` and all its inner functions
- `_on_threshold_preview_toggled`
- `_on_threshold_preview_params_changed`
- `_on_contour_worker_error`
- `_contour_worker` / `_contour_cancel` state attributes
- Update `_on_cancel` to remove contour-cancel logic
- Update `_alias_pipeline_controls` to remove the aliased preview-handler names

---

## Step 5 — Update `_state.py`

**File:** `napari/_state.py`

### `dump_state`
Under `"db_generation"`:
- Replace `"max_area": w.db_gen_max_area_spin.value()` with `"atom_union_max_area": w.atom_union_max_area_spin.value()`
- Add `"max_atoms": w.atom_union_max_atoms_spin.value()`
- Remove: `"threshold_pairs"`, `"min_frontier"`, `"ws_hierarchy"`, `"n_workers"`

### `load_state`
- Load `"atom_union_max_area"` → `w.atom_union_max_area_spin.setValue(...)`
- Load `"max_atoms"` → `w.atom_union_max_atoms_spin.setValue(...)`
- Silently ignore legacy keys: `"max_area"`, `"threshold_pairs"`, `"min_frontier"`,
  `"ws_hierarchy"`, `"n_workers"`

---

## Step 6 — Tests

### `tests/tracking_ultrack/test_atoms.py` — add 5 tests
- `test_atom_adjacency_two_adjacent_labels` — 2-label image, checks symmetric edges, no background entry
- `test_atom_adjacency_diagonal_not_adjacent` — diagonally touching labels → no shared edge (4-conn)
- `test_enum_connected_unions_single_atom` — 1 atom → exactly 1 union
- `test_enum_connected_unions_two_adjacent` — 2 adjacent atoms → {A}, {B}, {A,B}
- `test_enum_connected_unions_respects_max_atoms_and_max_area`

### `tests/tracking_ultrack/test_db_build.py` — add 1 test
- `test_build_atom_union_database_segments_then_links(monkeypatch, tmp_path)`
  Monkeypatch `tifffile.imread` (return a tiny 2-frame atoms stack),
  `_build_ultrack_config`, all ultrack DB symbols, `run_linking`.
  Verify call order and that an `AtomUnionDatabaseBuildReport` is returned.

### `tests/napari/test_nucleus_tracking_inputs_widget.py` — update
- **Remove:** `test_tracking_inputs_widget_threshold_pair_list_starts_empty_and_adds_pairs`,
  `test_tracking_inputs_widget_preview_control_is_checkbox`,
  `test_tracking_inputs_widget_threshold_pair_list_rejects_duplicates`,
  `test_tracking_inputs_widget_removes_and_clears_threshold_pairs`
- **Update:** `test_tracking_inputs_widget_db_gen_config_applies_all_controls` —
  change `cfg.seg_min_area == 500` assertion; add `cfg.min_area == 500`,
  `cfg.atom_union_max_atoms`, `cfg.atom_union_max_area`
- **Add:** `test_tracking_inputs_widget_has_atom_union_controls` —
  checks `atom_union_max_atoms_spin` default 3, `atom_union_max_area_spin` default 8000

### `tests/napari/test_nucleus_pipeline_widget.py` — update
- **Remove:** threshold-pair-state tests:
  `test_nucleus_workflow_exposes_threshold_pair_controls_in_db_parameters`,
  `test_nucleus_state_persists_explicit_threshold_pairs`,
  `test_nucleus_state_loads_legacy_sweep_state_as_empty_threshold_pairs`
- **Remove:** preview-handler tests:
  `test_preview_threshold_pair_updates_layers_without_mutating_pair_list`,
  `test_preview_checkbox_auto_updates_when_threshold_changes`,
  `test_source_preview_parameter_update_does_not_show_progress_bar`,
  `test_unchecked_source_auto_preview_ignores_late_worker_result`
- **Remove:** old DB-gen guard tests:
  `test_run_db_generation_reports_empty_threshold_pair_list`,
  `test_run_db_generation_reports_missing_canonical_contours`,
  `test_run_db_generation_reports_missing_canonical_foreground`
- **Replace:** `test_run_db_generation_calls_build_database` →
  `test_run_db_generation_calls_build_atom_union_database`:
  monkeypatch `build_atom_union_database`, write a placeholder `atoms.tif`,
  verify `atoms_path`, `working_dir`, `cfg.atom_union_max_atoms`, `cfg.atom_union_max_area`
- **Add:** `test_run_db_generation_reports_missing_atoms_tif`
- **Add:** `test_nucleus_state_persists_atom_union_params` — round-trip `max_atoms` / `atom_union_max_area`
- **Update stub dicts:** replace `build_ultrack_database_from_threshold_pairs` with
  `build_atom_union_database` in `_install_import_stubs`

---

## Files touched (summary)

| File | Nature of change |
|---|---|
| `tracking_ultrack/atoms.py` | Fix 4→3 return regression |
| `tracking_ultrack/config.py` | Already done |
| `tracking_ultrack/db_build.py` | Already done |
| `napari/nucleus_tracking_inputs_widget.py` | Wire grid + update `db_gen_config()` |
| `napari/nucleus_workflow_widget.py` | Update aliases + remove source-preview signals/guards |
| `napari/nucleus_pipeline_widget.py` | Replace DB-gen handler; remove preview handlers |
| `napari/_state.py` | Update dump/load |
| `tests/tracking_ultrack/test_atoms.py` | Add graph function tests |
| `tests/tracking_ultrack/test_db_build.py` | Add atom-union build test |
| `tests/napari/test_nucleus_tracking_inputs_widget.py` | Remove threshold-pair tests; update/add controls tests |
| `tests/napari/test_nucleus_pipeline_widget.py` | Replace DB-gen handler tests; remove preview tests |
