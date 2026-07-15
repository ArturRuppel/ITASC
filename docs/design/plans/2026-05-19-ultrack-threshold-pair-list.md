# Ultrack Threshold Pair List Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the nucleus Ultrack threshold sweep with an explicit preview-and-add threshold-pair list used by DB generation.

**Architecture:** Add an explicit pair-list API in `multi_threshold.py`, move threshold controls into `NucleusTrackingInputsWidget`, and make `NucleusPipelineWidget` preview/build from ordered pairs. Keep source-stack arrays in memory and preserve pair order through metadata and DB source order.

**Tech Stack:** Python, Qt via qtpy/superqt, napari layer APIs, pytest.

---

## File Structure

- Modify `src/cellflow/tracking_ultrack/multi_threshold.py`: add pair normalization and source-stack generation from explicit ordered pairs; keep old array sweep functions for compatibility where useful.
- Modify `src/cellflow/napari/nucleus_tracking_inputs_widget.py`: add source threshold controls, preview/add/remove/clear buttons, pair table, and pair-list accessor/mutators.
- Modify `src/cellflow/napari/nucleus_pipeline_widget.py`: remove the separate Ultrack Inputs stage row, preview only the current pair, and build DB from the explicit pair list.
- Modify `src/cellflow/napari/nucleus_workflow_widget.py`: stop aliasing old segmentation sweep controls; expose new DB threshold controls and list helpers.
- Modify `src/cellflow/napari/_state.py`: persist `threshold_pairs`; old min/max/step state loads to an empty list.
- Modify tests in `tests/tracking_ultrack/test_multi_threshold.py`, `tests/napari/test_nucleus_tracking_inputs_widget.py`, and `tests/napari/test_nucleus_pipeline_widget.py`.

## Task 1: Backend Explicit Pair API

**Files:**
- Modify: `src/cellflow/tracking_ultrack/multi_threshold.py`
- Test: `tests/tracking_ultrack/test_multi_threshold.py`

- [ ] **Step 1: Write failing backend tests**

Add tests proving explicit pairs preserve order and DB generation does not require source-stack TIFFs.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/tracking_ultrack/test_multi_threshold.py::test_build_ultrack_source_stacks_from_pairs_preserves_pair_order -q`

Expected: fail because `build_ultrack_source_stacks_from_pairs` does not exist.

- [ ] **Step 3: Implement backend API**

Add `normalize_threshold_pairs`, `build_ultrack_source_stacks_from_pairs`, and `build_ultrack_database_from_threshold_pairs`. Existing `build_ultrack_source_stacks` delegates by expanding the Cartesian product into pairs.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/tracking_ultrack/test_multi_threshold.py::test_build_ultrack_source_stacks_from_pairs_preserves_pair_order -q`

Expected: pass.

## Task 2: DB Parameter Pair-List UI

**Files:**
- Modify: `src/cellflow/napari/nucleus_tracking_inputs_widget.py`
- Test: `tests/napari/test_nucleus_tracking_inputs_widget.py`

- [ ] **Step 1: Write failing widget tests**

Add tests covering empty initial list, add/remove/clear, duplicate rejection, and current-pair accessor.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/napari/test_nucleus_tracking_inputs_widget.py::test_tracking_inputs_widget_threshold_pair_list_starts_empty_and_adds_pairs -q`

Expected: fail because pair-list controls and methods do not exist.

- [ ] **Step 3: Implement widget controls**

Add `source_contour_threshold_spin`, `source_foreground_threshold_spin`, `source_threshold_preview_btn`, `source_threshold_add_btn`, `source_threshold_remove_btn`, `source_threshold_clear_btn`, and `source_threshold_pairs_table`. Implement `current_threshold_pair()`, `threshold_pairs()`, `set_threshold_pairs()`, `add_threshold_pair()`, `remove_selected_threshold_pair()`, and `clear_threshold_pairs()`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/napari/test_nucleus_tracking_inputs_widget.py -q`

Expected: pass.

## Task 3: Pipeline Preview and DB Generation

**Files:**
- Modify: `src/cellflow/napari/nucleus_pipeline_widget.py`
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_pipeline_widget.py`

- [ ] **Step 1: Write failing pipeline tests**

Update preview and DB tests to use current pair/list APIs. Add DB generation tests proving an empty list refuses to start and a non-empty list calls `build_ultrack_database_from_threshold_pairs` with exact pair order.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/napari/test_nucleus_pipeline_widget.py::test_preview_threshold_pair_updates_layers_without_mutating_pair_list -q`

Expected: fail because the preview handler and new builder call are missing.

- [ ] **Step 3: Implement pipeline behavior**

Remove the visible "Ultrack Inputs" row from `build_pipeline_block`; keep the method compatible with an optional `seg_section` argument. Add `_on_preview_threshold_pair`, wire the tracking widget preview button to it, read `threshold_pairs()` for DB generation, show `Add at least one threshold pair before DB generation.` when empty, and call `build_ultrack_database_from_threshold_pairs`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/napari/test_nucleus_pipeline_widget.py -q`

Expected: pass.

## Task 4: State Persistence and Compatibility Cleanup

**Files:**
- Modify: `src/cellflow/napari/_state.py`
- Modify: `src/cellflow/napari/nucleus_segmentation_inputs_widget.py`
- Test: `tests/napari/test_nucleus_pipeline_widget.py`

- [ ] **Step 1: Write failing state tests**

Add tests proving `get_state()` includes `db_generation["threshold_pairs"]`, old sweep keys are absent from new dumps, and old sweep-only state loads to an empty pair list.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/napari/test_nucleus_pipeline_widget.py::test_nucleus_state_persists_explicit_threshold_pairs -q`

Expected: fail because state still persists old sweep controls.

- [ ] **Step 3: Implement state changes**

Persist `threshold_pairs` via the tracking widget. Load `threshold_pairs` when present. Do not reconstruct pairs from `threshold_min`, `threshold_max`, or `threshold_step`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/napari/test_nucleus_pipeline_widget.py -q`

Expected: pass.

## Task 5: Final Verification

**Files:**
- All touched implementation and test files.

- [ ] **Step 1: Run focused verification**

Run: `pytest tests/tracking_ultrack/test_multi_threshold.py tests/napari/test_nucleus_tracking_inputs_widget.py tests/napari/test_nucleus_pipeline_widget.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run compile check**

Run: `python -m py_compile src/cellflow/tracking_ultrack/multi_threshold.py src/cellflow/napari/nucleus_tracking_inputs_widget.py src/cellflow/napari/nucleus_pipeline_widget.py src/cellflow/napari/nucleus_workflow_widget.py src/cellflow/napari/_state.py`

Expected: exit code 0.

- [ ] **Step 3: Review diff and status**

Run: `git diff --stat` and `git status --short`

Expected: only the planned files and this plan are changed.

