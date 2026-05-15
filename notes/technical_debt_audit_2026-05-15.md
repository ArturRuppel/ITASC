# Technical Debt & Dead Code Audit — CellFlow

_Date: 2026-05-15_

_Status update: 2026-05-15 evening_
- [DONE] Broken scripts named below were archived under `notes/archived_scripts/2026-05-15-dead-imports/`.
- [DONE] `correction.apply_gamma` re-export, deprecated `db_gen_power_spin`, the test-only unconstrained `retrack_frame`, and the old `tracking/` source package were removed or folded into `tracking_ultrack`.
- [DONE] Shared node geometry was extracted to `tracking_ultrack/_node_geometry.py`, and cross-module private imports from `validation_nodes._node_bbox_and_mask` were replaced.
- [PARTIAL] `nucleus_workflow_widget.py` is still large, but the Ultrack DB browser was extracted to `napari/nucleus_db_browser_widget.py`. The widget is now ~2333 LOC, the new browser module is ~762 LOC, and the companion layout test is still ~3994 LOC.
- [DONE] Ruff/lint configuration was added to `pyproject.toml` with initial `F401`/`F811` checks; the baseline passes after unused-import cleanup. `E501` is deferred to avoid formatting churn.

_Status update: 2026-05-15 late evening_
- [DONE] First section split completed: `NucleusUltrackDbBrowserWidget` now owns the Ultrack database browser controls and behavior, while `NucleusWorkflowWidget` keeps compatibility aliases for existing tests/callers.
- [DONE] Extraction seam test was added to prove `NucleusWorkflowWidget.ultrack_db_browser_widget` exists and owns the legacy DB-browser controls.
- [VERIFIED] `python -m py_compile` and `python -m ruff check` passed for the touched Python files. Focused DB-browser tests passed: `18 passed, 2 skipped`.
- [DONE] DB-browser tests were moved into `tests/napari/test_nucleus_db_browser_widget.py`; focused DB-browser suite passes in the default environment (`18 passed, 4 skipped`) and in the `cellflow` conda env (`22 passed`).
- [DONE] Correction section behavior moved to `napari/nucleus_correction_widget.py`; `NucleusWorkflowWidget` now composes `NucleusCorrectionWidget`, keeps compatibility aliases for existing tests/callers, and only coordinates correction with the DB-browser dims refresh. The workflow widget is now ~1476 LOC and the correction widget is ~1024 LOC.
- [DONE] Correction-focused behavior tests moved into `tests/napari/test_nucleus_correction_widget.py`; the new suite instantiates `NucleusCorrectionWidget` directly and passes (`16 passed`). Correction button signal wiring and default path/viewer providers now live in the child widget.
- [VERIFIED] Full `tests/napari/test_nucleus_tracking_correction_layout.py -q` now passes (`82 passed, 2 skipped`) after moving/narrowing the grouped anchor tests and restoring tracked-label viewer refresh on Ultrack completion.

_Status update: 2026-05-15 night_
- [DONE] Correction/widget boundary tightened: `NucleusCorrectionWidget` now takes explicit `pos_dir_provider`, `refresh_refinement_callback`, and dependency callbacks instead of a broad workflow fallback.
- [DONE] Next self-contained workflow section extracted: segmentation input parameter controls now live in `napari/nucleus_segmentation_inputs_widget.py`, with workflow aliases kept for existing callers/tests.
- [VERIFIED] Focused correction suite passes (`17 passed`), focused workflow layout suite passes (`84 passed, 2 skipped`), and `python -m py_compile` / `python -m ruff check` pass for touched files.

_Status update: 2026-05-15 late night_
- [DONE] Ultrack tracking/database-generation parameter section extracted to `napari/nucleus_tracking_inputs_widget.py` (`NucleusTrackingInputsWidget`, 220 LOC). `NucleusWorkflowWidget` is now 1227 LOC (down from 1394). The child widget owns all DB-gen and Ultrack solver spinboxes/combos, the section `CollapsibleSection`, mode-change wiring, and `db_gen_config()`/`ultrack_config()` builder methods. `NucleusWorkflowWidget._db_gen_config_from_controls` and `_ultrack_config_from_controls` delegate to the child; all control attributes are aliased on the parent for backward compat.
- [DONE] Focused test file `tests/napari/test_nucleus_tracking_inputs_widget.py` added (9 tests, 310 LOC); delegation seam test added to `test_nucleus_tracking_correction_layout.py` (now 87 passed in layout suite, 96 total across both files).
- [VERIFIED] `python -m py_compile` and `python -m ruff check` pass for all touched files. Combined test run: `96 passed`.
- [NEXT] Extract the remaining pipeline action/worker coordination (segmentation-input build, DB-gen run, Ultrack solve, cancel) out of `nucleus_workflow_widget.py` into a `nucleus_pipeline_widget.py`, leaving `NucleusWorkflowWidget` as a thin compositor only.

_Status update: 2026-05-16 morning_
- [DONE] Pipeline action/worker-coordination section extracted to `napari/nucleus_pipeline_widget.py` (`NucleusPipelineWidget`, ~804 LOC). `NucleusWorkflowWidget` is now ~529 LOC (down from 1227) — a thin compositor only. The new child widget owns all pipeline handler methods (`_on_build_segmentation_inputs`, `_on_build_contour_maps`, `_on_preview_contour_maps`, `_on_run_db_generation`, `_on_db_gen_done`, `_on_run_ultrack`, `_on_run_ultrack_done`, `_on_cancel`, error handlers), all three background workers, all status/progress helpers, all pipeline buttons, and all pipeline-scoped path/viewer helpers. `NucleusWorkflowWidget` composes it via `nucleus_pipeline_widget`, aliases buttons/status/progress/handlers for backward compat, and provides compat path-delegate methods for existing seam tests.
- [DONE] Focused test file `tests/napari/test_nucleus_pipeline_widget.py` added (13 tests, ~470 LOC); 9 occurrences of `module = sys.modules[widget_class.__module__]` in `test_nucleus_tracking_correction_layout.py` updated to patch `cellflow.napari.nucleus_pipeline_widget` (where handlers now live).
- [VERIFIED] `python -m py_compile` and `python -m ruff check` pass for all touched files. Combined suite: `100 passed` (`87 layout` + `13 pipeline`).
- [NEXT] `NucleusWorkflowWidget` is now a thin compositor at ~529 LOC. The remaining structural candidate is to split `cell_workflow_widget.py` (still ~1344 LOC) following the same pattern, or to address the `tracking_ultrack/multi_threshold.py` / `linking.py` parallel-pipeline debt noted in section 3.

_Status update: 2026-05-16 midday — §3 structural sweep_
- [DONE] `napari/cellpose_widget.py` (68-line passthrough) folded into `main_widget.py` as a private `_CellposePanel` class. Public `_cellpose_widget` / `hpc_cellpose_widget` attributes on `CellFlowMainWidget` preserved. `cellpose_widget.py` deleted; `test_cellpose_file_contract.py` and `test_nucleus_tracking_correction_layout.py` (stub cleanup) updated. `95 passed` across affected tests.
- [DONE] `tracking_ultrack/linking.py` — shared per-pair gate extracted as `_shape_pair_score(...)` and called from both `compute_edge_weight` (shape branch) and the `_run_shape_linking` inner loop. No behavior change; the only surface difference between the two original sites was `return None` vs `continue`, normalized in the helper. `2 passed` on linking tests.
- [VERIFIED-AS-ALREADY-DONE] `tracking_ultrack/multi_threshold.py` private `_node_*` imports — already cleaned up in commit `f524612`. Shared geometry now lives in `_node_geometry` as public functions (`node_bbox_and_mask`, `node_pickle_ndim`, `make_node_pickle`, `intersects`, `raw_iou`); `multi_threshold.py`, `validation_nodes.py`, `swap_candidate.py`, `extend.py`, `seed_prior.py` all import them at module level. Remaining in-function imports are legitimate: lazy `ultrack` (heavy optional dep) and a real `db_query` ↔ `multi_threshold` cycle break. Smoke import + `96 passed, 1 pre-existing failure`. The audit's §3 line on this is now stale.
- [NEXT] Split `cell_workflow_widget.py` (~1344 LOC) following the nucleus widget pattern, OR sweep the remaining §4 naming/organization items (`tracking` package merge, `scripts/test_*` rename, README pointer).

## Codebase shape
- ~21k LOC across `src/cellflow/` (8 packages), ~14k LOC of tests, ~8.8k LOC of scripts.
- Single napari entry point (`CellFlowWidget` → `CellFlowMainWidget`) wires together six sub-widgets.

## 1. Dead / broken scripts (the worst offender)
Many `scripts/*.py` import symbols that **no longer exist** in `src/`. They'll fail at import — they are stranded experimental code:

| Script | Missing import |
|---|---|
| `scripts/experiment_consensus_movie.py` | `cellflow.tracking.consensus_movie` (module doesn't exist) |
| `scripts/experiment_frame_selector.py` | `cellflow.tracking.frame_selector` (module doesn't exist) |
| `scripts/test_cellpose_preproc_sweep.py` | `cellflow.database.hypotheses.HypothesisRecord`, `write_hypothesis_sweep_h5` |
| `scripts/experiment_cell_contour_maps.py` | `cellflow.database.hypotheses.normalize_seeded_watershed_dp_stack` |
| `scripts/test_cellpose_native.py` | `cellflow.segmentation.CellposeFlowHypothesisParams`, `compute_cellpose_flow_hypothesis` |
| `scripts/experiment_cell_2d_*` (7 files) | `cellflow.segmentation.centroid_markers_from_labels` |
| `scripts/test_validate_and_resolve.py` | `cellflow.tracking_ultrack.reseed.resolve_with_validation` |
| `scripts/probe_ultrack_db.py`, `benchmark_ultrack_phase2.py`, `run_ultrack_*.py` | `cellflow.tracking_ultrack.ingest.ingest_hypotheses_to_db` |

Recommendation: delete (or move to `scripts/_archive/`) anything that doesn't import cleanly. There's also no script-level CI to keep them honest.

## 2. Concrete dead code in `src/`
- **`src/cellflow/correction/labels.py:666`** — circular re-export hack: `from cellflow.segmentation import apply_gamma  # noqa: F401 — re-exported from here`. Nothing actually imports `apply_gamma` from `cellflow.correction`; callers import it from `cellflow.segmentation`. The `correction/__init__.py` also re-exports it. Pure backward-compat shim with no remaining caller — drop it.
- **`db_gen_power_spin` (`nucleus_workflow_widget.py:437–441`)** — explicitly commented "Hidden/deprecated — needed for state persistence only". `_state.py` round-trips a `"power"` key. If the param isn't used by the solver anymore, the state shim and tests covering it (`test_nucleus_tracking_correction_layout.py:676,3641,3669`) are pure dead weight kept alive by themselves.
- **`tracking/retracker.retrack_frame`** — only called from `tests/tracking/test_retracker.py`. Production code uses `retrack_frame_constrained`. The unconstrained version exists solely to be tested.
- **`tracking/__init__.py`** is empty save for a docstring, and the whole `tracking/` package contains one production-used function. The mismatch suggests the package was bigger before; the residual `__init__` could be folded into `tracking_ultrack` or the file moved.

## 3. Structural debt
- **`napari/nucleus_workflow_widget.py` (~1452 LOC after DB-browser and correction extraction)** and **`cell_workflow_widget.py` (1344 LOC)** are still large god-widgets. The companion layout test is down to ~2629 LOC, with ~656 LOC of correction behavior coverage now isolated in `tests/napari/test_nucleus_correction_widget.py`. The `_state.py`/`_paths.py`/`_thresholds.py`/`_widget_helpers.py` helpers and the new `nucleus_db_browser_widget.py` / `nucleus_correction_widget.py` extractions show the split is feasible; the next cleanup should extract another self-contained section and keep its tests isolated.
- **`napari/cellpose_widget.py`** is a 68-line passthrough that wraps `HpcCellposeWidget` plus two `PipelineFilesWidget`s. It looks like a leftover scaffold between when Cellpose ran in-process and now (when "Cellpose runs externally"). Could be merged into `main_widget` or `HpcCellposeWidget`.
- **`tracking_ultrack/linking.py`** has two parallel pipelines (`"default"` and `"shape"` modes) with `compute_edge_weight` duplicating filter logic inline at lines 131–147 and 244–258. Worth pulling the per-pair gate into one function.
- **`tracking_ultrack/multi_threshold.py` (861 LOC)** does **four** local `from cellflow.tracking_ultrack.validation_nodes import _node_*` inside function bodies — private-symbol traffic between siblings suggests the module boundary is wrong.
- **Lazy/local imports for cycle avoidance**: `corrections.py:208` imports `linking` inside a function; `db_query.py:188,264` does the same for `multi_threshold`; `extend.py:300`, `swap_candidate.py:44`, `seed_prior.py:12` all reach into `validation_nodes._node_bbox_and_mask`. These are smells of a package that wants a small shared `_node_geometry.py`.

## 4. Naming / organization
- The package split `tracking` vs `tracking_ultrack` is confusing — the former has one file (`retracker.py`) used in one place. Consider merging or renaming.
- `test_*` scripts in `scripts/` (e.g. `test_param_sweep.py`, `test_validate_and_resolve.py`) are *experiments*, not tests — pytest discovery in that directory would be surprising. Rename to `experiment_*` or `bench_*`.
- `notebooks/` and `notes/` together hold ~3 design docs; `docs/superpowers/` looks like vendored tooling. Worth a README pointer.

## 5. Lower-priority observations
- `pyproject.toml` declares `requires-python = ">=3.9"` but the code uses `str | None` (PEP 604) and `from __future__ import annotations` selectively — actually fine, but pin to 3.10+ if any module forgoes the future import.
- `tool.ruff` now covers initial import-drift checks (`F401`/`F811`). `tool.pytest` config and broader Ruff rules remain future cleanup.
- `napari/_napari_compat.py` patches private napari internals (`napari._qt.containers._layer_delegate`) — a known fragility that should be marked for periodic re-check when bumping napari.

## Suggested order to pay it down
1. [DONE] **Delete or archive broken scripts** (15+ files) — pure win, zero risk.
2. [DONE] **Drop the `correction.apply_gamma` re-export** and the `_state` "power" round-trip — and remove the tests that only exist to keep them alive.
3. [DONE] **Collapse `tracking/` into `tracking_ultrack/`** (or rename to `relabel/`), and drop the test-only `retrack_frame`.
4. [PARTIAL] **Split `nucleus_workflow_widget.py`** by section. Ultrack DB browser, correction behavior, and correction tests are now extracted; next tighten the correction provider boundary and then extract another self-contained section.
5. [DONE] **Extract a `tracking_ultrack/_node_geometry.py`** so `validation_nodes._node_bbox_and_mask` stops being privately imported across five modules.
6. [DONE] **Add `ruff`** with `F401`/`F811` to catch the next round of import drift automatically; defer `E501` until line-length cleanup is worth the churn.
