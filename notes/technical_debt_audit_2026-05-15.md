# Technical Debt & Dead Code Audit — CellFlow

_Date: 2026-05-15_

_Status update: 2026-05-15 evening_
- [DONE] Broken scripts named below were archived under `notes/archived_scripts/2026-05-15-dead-imports/`.
- [DONE] `correction.apply_gamma` re-export, deprecated `db_gen_power_spin`, the test-only unconstrained `retrack_frame`, and the old `tracking/` source package were removed or folded into `tracking_ultrack`.
- [DONE] Shared node geometry was extracted to `tracking_ultrack/_node_geometry.py`, and cross-module private imports from `validation_nodes._node_bbox_and_mask` were replaced.
- [OPEN] `nucleus_workflow_widget.py` is still monolithic; it is now ~3032 LOC with the companion layout test still ~3975 LOC.
- [DONE] Ruff/lint configuration was added to `pyproject.toml` with initial `F401`/`F811` checks; the baseline passes after unused-import cleanup. `E501` is deferred to avoid formatting churn.

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
- **`napari/nucleus_workflow_widget.py` (2999 LOC, 164 methods, 1 class)** and **`cell_workflow_widget.py` (1344 LOC)** are giant god-widgets. The companion test file `tests/napari/test_nucleus_tracking_correction_layout.py` is 3966 LOC — implies the widget is hard to test in isolation. The `_state.py`/`_paths.py`/`_thresholds.py`/`_widget_helpers.py` private helpers are a step toward splitting, but only `nucleus_workflow_widget.py` consumes them; further extraction (per-section sub-widgets) is feasible.
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
4. [OPEN] **Split `nucleus_workflow_widget.py`** by section (already partially done via `_widget_helpers`); the test file shrinks naturally.
5. [DONE] **Extract a `tracking_ultrack/_node_geometry.py`** so `validation_nodes._node_bbox_and_mask` stops being privately imported across five modules.
6. [DONE] **Add `ruff`** with `F401`/`F811` to catch the next round of import drift automatically; defer `E501` until line-length cleanup is worth the churn.
