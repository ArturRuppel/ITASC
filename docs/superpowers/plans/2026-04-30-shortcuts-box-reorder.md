# Shortcuts Box Reorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the correction shortcuts info box to the bottom of the tracking/correction widget so it appears after the show-outlines control and cell inspector.

**Architecture:** Keep the standalone correction widget behavior unchanged. Adjust only the `NucleusWorkflowWidget` composition so the shortcuts section is inserted after the embedded correction widget inside the correction collapsible section. Update the layout test to assert the new order using the section's inner layout.

**Tech Stack:** Python, Qt via `qtpy`, `pytest`

---

### Task 1: Reorder the correction section children

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write the failing test**

```python
def test_tracking_correction_shell_exposes_stable_section_attributes():
    ...
    inner_layout = widget.correction_section._content_frame.layout()
    assert inner_layout.itemAt(0).widget() is widget.correction_widget
    assert inner_layout.itemAt(1).widget() is widget.correction_shortcuts_section
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_tracking_correction_shell_exposes_stable_section_attributes -v -q`
Expected: FAIL because `correction_shortcuts_section` is currently added before `correction_widget`.

- [ ] **Step 3: Make the minimal implementation**

```python
self.correction_widget = CorrectionWidget(
    self.viewer,
    show_activate_btn=False,
    show_shortcuts=False,
    inspector_first=True,
)
self.correction_widget.set_edit_callback(self._on_cells_edited)
self._corr_inner_lay.addWidget(self.correction_widget)
self.correction_shortcuts_section = CollapsibleSection(
    "Correction Shortcuts",
    self.correction_widget.build_shortcuts_widget(),
    expanded=True,
)
self._corr_inner_lay.addWidget(self.correction_shortcuts_section)
```

- [ ] **Step 4: Run the targeted test to verify it passes**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_tracking_correction_shell_exposes_stable_section_attributes -v -q`
Expected: PASS.

- [ ] **Step 5: Run the relevant layout test file**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py -q`
Expected: PASS.

---
