# UI Style Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small shared style/layout helper module for CellFlow's napari UI and migrate the most visible ad hoc styling to it.

**Architecture:** Keep all existing widget behavior and layout structure. Introduce `cellflow.napari.ui_style` as the single source for compact margins, button sizing, muted/status labels, and semantic danger/success button styling, then migrate current widgets incrementally.

**Tech Stack:** Python 3.13, napari, qtpy, pytest

---

## File Map

- Create: `src/cellflow/napari/ui_style.py`
  Responsibility: semantic Qt widget styling helpers and shared layout constants.
- Create: `tests/napari/test_ui_style.py`
  Responsibility: focused unit tests for helper behavior.
- Modify: `src/cellflow/napari/widgets.py`
  Responsibility: use helpers for collapsible sections and pipeline file status rows.
- Modify: `src/cellflow/napari/main_widget.py`
  Responsibility: use helpers for top project/config controls.
- Modify: `src/cellflow/napari/data_prep_widget.py`
  Responsibility: use helpers for muted metadata and status text.
- Modify: `src/cellflow/napari/cellpose_widget.py`
  Responsibility: use helpers for descriptive muted text.
- Modify: `src/cellflow/napari/correction_widget.py`
  Responsibility: use helpers for checked success, warning/danger, and muted status text.
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
  Responsibility: use helpers for local compact controls and danger buttons.
- Modify: `src/cellflow/napari/cell_workflow_widget.py`
  Responsibility: use helpers for local compact controls and danger buttons.

## Tasks

### Task 1: Add Helper Module With Tests

- [ ] Write failing tests in `tests/napari/test_ui_style.py` for compact spinboxes, action buttons, tiny buttons, muted/status labels, danger buttons, and checked-success buttons.
- [ ] Run `pytest tests/napari/test_ui_style.py -v` and confirm it fails because `cellflow.napari.ui_style` does not exist.
- [ ] Create `src/cellflow/napari/ui_style.py` with the tested helpers.
- [ ] Run `pytest tests/napari/test_ui_style.py -v` and confirm it passes.
- [ ] Commit the helper module and tests.

### Task 2: Migrate Shared Widgets

- [ ] Add tests or extend existing assertions where practical so `PipelineFilesWidget` continues to construct and reflect present/missing states.
- [ ] Replace inline style strings and magic sizing in `src/cellflow/napari/widgets.py` with `ui_style` helpers.
- [ ] Run `pytest tests/napari/test_ui_style.py tests/napari/test_nucleus_tracking_correction_layout.py -v -q`.
- [ ] Commit the shared widget migration.

### Task 3: Migrate Visible Widget Surfaces

- [ ] Replace obvious inline button/label styling in `main_widget.py`, `data_prep_widget.py`, `cellpose_widget.py`, `correction_widget.py`, `nucleus_workflow_widget.py`, and `cell_workflow_widget.py` with `ui_style` helpers.
- [ ] Keep existing layout semantics intact: do not redesign section hierarchy, remove controls, or change backend wiring.
- [ ] Run `python -m py_compile` on all modified source files.
- [ ] Run `pytest tests/napari/test_ui_style.py tests/napari/test_nucleus_tracking_correction_layout.py -v -q`.
- [ ] Commit the visible-widget migration.

## Self-Review

- The plan covers all requirements in the spec.
- The plan has no placeholders.
- Helper names are concrete and scoped to the current Qt widgets.

