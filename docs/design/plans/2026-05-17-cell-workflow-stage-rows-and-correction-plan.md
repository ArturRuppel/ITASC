# Cell Workflow Stage Rows and Correction Header Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the cell workflow UI to the nucleus-style stage rows and correction header while preserving existing pipeline and correction behavior.

**Architecture:** `CellParamsWidget` remains the owner of all pipeline parameter controls, but exposes one `CollapsibleSection` per pipeline stage. `CellWorkflowWidget` owns the pipeline stage rows, run/cancel state, and aliases. `CellCorrectionWidget` owns correction behavior and exposes nucleus-style header/content widgets for the workflow to reparent into the main layout.

**Tech Stack:** Python, qtpy/PyQt, napari test stubs, pytest.

---

### Task 1: Pipeline Stage Row Tests

**Files:**
- Modify: `tests/napari/test_cell_workflow_widget.py`

- [ ] **Step 1: Write failing tests for stage rows and params toggles**

Add tests that instantiate `CellWorkflowWidget` and assert these controls exist:

```python
def test_widget_exposes_stage_rows_with_inline_params(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    for name in (
        "flow_params_btn", "flow_run_btn",
        "foreground_params_btn", "foreground_run_btn",
        "contour_params_btn", "contour_preview_btn", "contour_run_btn",
        "segmentation_params_btn", "segmentation_run_btn",
    ):
        assert isinstance(getattr(widget, name), QToolButton)

    assert widget.preview_contour_btn is widget.contour_preview_btn
    assert widget.flow_filter_section.is_expanded is False
    assert widget.foreground_section.is_expanded is False
    assert widget.contour_section.is_expanded is False
    assert widget.segmentation_section.is_expanded is False

    widget.flow_params_btn.setChecked(True)
    assert widget.flow_filter_section.is_expanded is True
    assert widget.foreground_section.is_expanded is False
    widget.contour_params_btn.setChecked(True)
    assert widget.contour_section.is_expanded is True

    widget.deleteLater()
    app.processEvents()
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `pytest tests/napari/test_cell_workflow_widget.py::test_widget_exposes_stage_rows_with_inline_params -q`

Expected: FAIL because the new `flow_params_btn` and stage section aliases do not exist yet.

- [ ] **Step 3: Implement only enough UI aliases/rows to pass later tasks**

No production code in this task. Continue to Task 2 for implementation.

### Task 2: Split Cell Parameter Sections

**Files:**
- Modify: `src/cellflow/napari/cell_params_widget.py`
- Test: `tests/napari/test_cell_workflow_widget.py`

- [ ] **Step 1: Refactor `CellParamsWidget` into stage sections**

Build four inner widgets and four `CollapsibleSection`s:

```python
self.flow_filter_section = CollapsibleSection("Flow filtering", flow_inner, expanded=False)
self.foreground_section = CollapsibleSection("Foreground masks", foreground_inner, expanded=False)
self.contour_section = CollapsibleSection("Contours", contour_inner, expanded=False)
self.segmentation_section = CollapsibleSection("Segmentation", segmentation_inner, expanded=False)
```

Keep all existing control attributes and helper methods unchanged.

- [ ] **Step 2: Preserve `section` compatibility**

Create a compatibility container:

```python
container = QWidget(self)
lay = QVBoxLayout(container)
lay.setContentsMargins(0, 0, 0, 0)
lay.setSpacing(6)
for section in (
    self.flow_filter_section,
    self.foreground_section,
    self.contour_section,
    self.segmentation_section,
):
    lay.addWidget(section)
self.section = CollapsibleSection("Parameters", container, expanded=False)
```

- [ ] **Step 3: Run the focused stage-row test**

Run: `pytest tests/napari/test_cell_workflow_widget.py::test_widget_exposes_stage_rows_with_inline_params -q`

Expected: still FAIL until Task 3 adds workflow aliases and row buttons.

### Task 3: Build Cell Pipeline Stage Rows

**Files:**
- Modify: `src/cellflow/napari/cell_workflow_widget.py`
- Test: `tests/napari/test_cell_workflow_widget.py`

- [ ] **Step 1: Replace old QPushButton grid with QToolButton rows**

Import `QHBoxLayout`, `QToolButton`, `_widget_helpers.tool_btn`, `_widget_helpers.make_status`, `_widget_helpers.make_progress`, and `stage_header_label`.

Create buttons:

```python
self.flow_params_btn = _tool_btn("⚙", "Show parameters for this stage.", checkable=True)
self.flow_run_btn = _tool_btn("▶", "Run flow filtering.")
self.foreground_params_btn = _tool_btn("⚙", "Show parameters for this stage.", checkable=True)
self.foreground_run_btn = _tool_btn("▶", "Run foreground mask generation.")
self.contour_params_btn = _tool_btn("⚙", "Show parameters for this stage.", checkable=True)
self.contour_preview_btn = _tool_btn("▷", "Preview contours for the current frame.")
self.contour_run_btn = _tool_btn("▶", "Run contour map generation.")
self.segmentation_params_btn = _tool_btn("⚙", "Show parameters for this stage.", checkable=True)
self.segmentation_run_btn = _tool_btn("▶", "Run cell segmentation.")
self.preview_contour_btn = self.contour_preview_btn
```

- [ ] **Step 2: Add row builder**

Add helpers on `CellWorkflowWidget`:

```python
def _stage_label(self, text: str) -> QLabel:
    return stage_header_label(QLabel(text), "cell")

def _stage_row(self, label: QLabel, *trailing: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(label)
    row.addStretch(1)
    for widget in trailing:
        row.addWidget(widget)
    return row
```

- [ ] **Step 3: Add sections inline**

In `_setup_ui`, hide stage section headers, collapse each section, connect gear buttons to section toggles, and add each row followed by its section.

- [ ] **Step 4: Run the focused stage-row test**

Run: `pytest tests/napari/test_cell_workflow_widget.py::test_widget_exposes_stage_rows_with_inline_params -q`

Expected: PASS.

### Task 4: Run/Cancel State

**Files:**
- Modify: `tests/napari/test_cell_workflow_widget.py`
- Modify: `src/cellflow/napari/cell_workflow_widget.py`

- [ ] **Step 1: Write failing test for running state**

Add:

```python
def test_cell_stage_running_state_disables_other_rows(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    widget._set_running_stage("contour")

    assert widget.contour_run_btn.text() == "✕"
    assert widget.contour_run_btn.isEnabled() is True
    assert widget.contour_params_btn.isEnabled() is True
    assert widget.flow_run_btn.isEnabled() is False
    assert widget.foreground_params_btn.isEnabled() is False
    assert widget.segmentation_run_btn.isEnabled() is False
    assert widget.contour_preview_btn.isEnabled() is False

    widget._set_running_stage(None)
    assert widget.contour_run_btn.text() == "▶"
    assert widget.flow_run_btn.isEnabled() is True
    assert widget.contour_preview_btn.isEnabled() is True

    widget.deleteLater()
    app.processEvents()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/napari/test_cell_workflow_widget.py::test_cell_stage_running_state_disables_other_rows -q`

Expected: FAIL because `_set_running_stage` does not exist or does not update the new buttons.

- [ ] **Step 3: Implement `_set_running_stage`, dispatchers, and `_on_cancel`**

Add `_running_stage` in `__init__`. Add click dispatchers that cancel when a stage is active or call the existing handler when idle. Add `_on_cancel()` that calls `.quit()` on the current worker if present, clears progress, resets buttons, and sets an informative status.

- [ ] **Step 4: Replace old enable/disable calls**

Change worker start paths from `_set_pipeline_buttons_enabled(False)` to `_set_running_stage("<stage>")`, and completion/error paths from `_set_pipeline_buttons_enabled(True)` to `_set_running_stage(None)`. Keep `_set_pipeline_buttons_enabled` as a compatibility shim.

- [ ] **Step 5: Run the focused test**

Run: `pytest tests/napari/test_cell_workflow_widget.py::test_cell_stage_running_state_disables_other_rows -q`

Expected: PASS.

### Task 5: Correction Header Tests and Implementation

**Files:**
- Modify: `tests/napari/test_cell_workflow_widget.py`
- Modify: `tests/napari/test_cell_correction_widget.py`
- Modify: `src/cellflow/napari/cell_correction_widget.py`
- Modify: `src/cellflow/napari/cell_workflow_widget.py`

- [ ] **Step 1: Write failing correction header test**

Add a workflow-level test:

```python
def test_cell_correction_uses_stage_style_header(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    assert widget.correction_header_lbl.text() == "Correction"
    assert widget.correction_shortcuts_btn.text() == "📖"
    assert widget.correction_params_btn.text() == "⚙"
    assert widget.correction_active_btn.text() == "⏻"
    assert isinstance(widget.correction_shortcuts_btn, QToolButton)
    assert isinstance(widget.correction_params_btn, QToolButton)
    assert isinstance(widget.correction_active_btn, QToolButton)
    assert widget.cell_correction_widget.isVisible() is False

    widget.correction_params_btn.setChecked(True)
    assert widget.correction_mode_section.is_expanded is True
    widget.correction_params_btn.setChecked(False)
    assert widget.correction_mode_section.is_expanded is False

    widget.deleteLater()
    app.processEvents()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/napari/test_cell_workflow_widget.py::test_cell_correction_uses_stage_style_header -q`

Expected: FAIL because the header aliases and icon buttons do not exist.

- [ ] **Step 3: Implement nucleus-style correction header**

In `CellCorrectionWidget`, create `header`, `header_lbl`, `shortcuts_btn`,
`params_btn`, and icon `active_btn` using `_tool_btn`. Hide params/shortcuts/
active content by default. Add `_sync_correction_panel_visibility()` and
button toggled handlers, following the nucleus correction widget pattern.

- [ ] **Step 4: Reparent correction pieces in workflow**

In `CellWorkflowWidget._setup_ui`, hide `cell_correction_widget` and add
`correction_header` plus `correction_mode_section` to the workflow layout.
Extend correction aliases with the new header controls.

- [ ] **Step 5: Run correction header tests**

Run: `pytest tests/napari/test_cell_workflow_widget.py::test_cell_correction_uses_stage_style_header tests/napari/test_cell_correction_widget.py -q`

Expected: PASS.

### Task 6: Update Existing Layout Tests and Verify

**Files:**
- Modify: `tests/napari/test_cell_workflow_widget.py`
- Modify: `tests/napari/test_cell_correction_widget.py`

- [ ] **Step 1: Update old flat-layout assertions**

Replace checks for `QPushButton` pipeline controls with `QToolButton` row
controls. Remove assertions that there are no stage sections. Keep existing
state, file widget, and correction behavior tests intact.

- [ ] **Step 2: Run focused cell workflow suite**

Run: `pytest tests/napari/test_cell_workflow_widget.py tests/napari/test_cell_correction_widget.py -q`

Expected: PASS.

- [ ] **Step 3: Run adjacent style/layout tests**

Run: `pytest tests/napari/test_cell_params_widget.py tests/napari/test_ui_style.py -q`

Expected: PASS.

- [ ] **Step 4: Final status check**

Run: `git status --short`

Expected: only intentional source, test, and plan changes are present.
