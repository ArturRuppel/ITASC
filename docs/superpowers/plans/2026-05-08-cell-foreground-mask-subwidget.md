# Cell Foreground Mask Subwidget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cell workflow foreground-mask section that builds `3_cell/foreground_masks.tif` from z-averaged Cellpose probabilities and filtered flow vectors.

**Architecture:** Put Cellpose mask dynamics in a testable segmentation helper and keep the napari widget responsible for UI, file I/O, worker lifecycle, and layer refresh. The new widget section sits between `Filtered Flow` and `Tracked Cell Labels` so users naturally create filtered flow, then foreground masks, then labels.

**Tech Stack:** Python, numpy, tifffile, qtpy/napari, Cellpose `cellpose.dynamics.compute_masks`, pytest.

---

### Task 1: Segmentation Helper

**Files:**
- Modify: `tests/segmentation/test_foreground_masks.py`
- Modify: `src/cellflow/segmentation/__init__.py`

- [ ] **Step 1: Write failing helper tests**

Add tests that stub `cellpose.dynamics.compute_masks`, call `compute_cellpose_foreground_masks`, and assert z-averaged logits, filtered flow, binary `uint8` output, progress callbacks, and shape validation.

- [ ] **Step 2: Run red tests**

Run:

```bash
pytest tests/segmentation/test_foreground_masks.py -q
```

Expected: fail because `compute_cellpose_foreground_masks` does not exist.

- [ ] **Step 3: Implement helper**

Add `compute_cellpose_foreground_masks(prob_tzyx, filtered_dp_tcyx, ..., progress_cb=None)` to `src/cellflow/segmentation/__init__.py`. It normalizes a missing time axis on probability input, validates `filtered_dp_tcyx == (T, 2, Y, X)`, averages probability over z, runs Cellpose masks per frame, and returns `(masks > 0).astype(np.uint8)` stacked as `(T, Y, X)`.

- [ ] **Step 4: Run green helper tests**

Run:

```bash
pytest tests/segmentation/test_foreground_masks.py -q
```

Expected: pass.

### Task 2: Cell Workflow Widget

**Files:**
- Modify: `tests/napari/test_cell_workflow_widget.py`
- Modify: `src/cellflow/napari/cell_workflow_widget.py`

- [ ] **Step 1: Write failing widget tests**

Add tests asserting:

- `Foreground Mask` section exists between `Filtered Flow` and `Tracked Cell Labels`.
- Default controls are `cellprob_threshold=0.0`, `flow_threshold=0.0`, `min_size=15`, `niter=200`.
- State round-trips the `foreground_mask` block.
- `_on_create_foreground_masks()` reads `cell_prob_3dt.tif` and `filtered_dp.tif`, calls the segmentation helper, writes `3_cell/foreground_masks.tif`, and adds a `Foreground Mask` labels layer.

- [ ] **Step 2: Run red widget tests**

Run:

```bash
pytest tests/napari/test_cell_workflow_widget.py -q
```

Expected: fail because the new section and handler do not exist.

- [ ] **Step 3: Implement widget section**

In `CellWorkflowWidget`, add foreground controls and a new section between filtered flow and tracked labels. Add `_on_create_foreground_masks`, `_foreground_params_from_ui`, foreground state persistence, button signal connection, and running-state handling that disables all three action buttons while a worker is active.

- [ ] **Step 4: Run green widget tests**

Run:

```bash
pytest tests/napari/test_cell_workflow_widget.py -q
```

Expected: pass.

### Task 3: Focused Verification

**Files:**
- Verify: `tests/segmentation/test_foreground_masks.py`
- Verify: `tests/napari/test_cell_workflow_widget.py`
- Verify: `tests/segmentation/test_flow_following.py`

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/segmentation/test_foreground_masks.py tests/napari/test_cell_workflow_widget.py tests/segmentation/test_flow_following.py -q
```

Expected: pass.

- [ ] **Step 2: Check diff**

Run:

```bash
git status --short
git diff -- src/cellflow/segmentation/__init__.py src/cellflow/napari/cell_workflow_widget.py tests/segmentation/test_foreground_masks.py tests/napari/test_cell_workflow_widget.py
```

Expected: only the planned files and this plan are changed.
