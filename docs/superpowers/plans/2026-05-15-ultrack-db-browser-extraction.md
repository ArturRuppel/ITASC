# Ultrack DB Browser Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the Ultrack DB browser from the monolithic nucleus workflow widget while preserving existing behavior and compatibility.

**Architecture:** Add `NucleusUltrackDbBrowserWidget` as a focused child widget. `NucleusWorkflowWidget` creates it, inserts its `CollapsibleSection`, aliases existing public controls, and forwards existing private DB-browser methods to the child.

**Tech Stack:** Python, Qt via `qtpy`, napari layers, pytest.

---

### Task 1: Add Extraction Seam Test

**Files:**
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] Add `test_nucleus_workflow_delegates_ultrack_db_browser_to_child_widget` near existing DB-browser layout tests.
- [ ] Run:
  `pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_nucleus_workflow_delegates_ultrack_db_browser_to_child_widget -q`
- [ ] Expected red result: import or attribute failure because `NucleusUltrackDbBrowserWidget` / `ultrack_db_browser_widget` does not exist yet.

### Task 2: Extract Widget

**Files:**
- Create: `src/cellflow/napari/nucleus_db_browser_widget.py`
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`

- [ ] Move DB-browser constants/imports needed only by the browser.
- [ ] Implement `NucleusUltrackDbBrowserWidget` with context callbacks for DB path, current time, and viewer frame updates.
- [ ] Replace `_build_db_browser_section` with child creation.
- [ ] Alias legacy attributes from child to parent.
- [ ] Keep parent forwarding methods for existing tests and call sites.

### Task 3: Verify Focused Behavior

**Files:**
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] Run the new extraction seam test and fix failures.
- [ ] Run focused DB-browser tests:
  `pytest tests/napari/test_nucleus_tracking_correction_layout.py -k 'ultrack_db_browser or hierarchy_slider' -q`
- [ ] Run the full layout file:
  `pytest tests/napari/test_nucleus_tracking_correction_layout.py -q`

