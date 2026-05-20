# Stage Header Button Cluster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move compact stage-header action buttons next to their stage title and style them as matching pill controls.

**Architecture:** Add one shared style helper in `cellflow.napari.ui_style`, then apply it at the existing header construction points. Keep all buttons as normal `QToolButton` instances; only layout order and visual styling change.

**Tech Stack:** Python, Qt via `qtpy`, napari plugin widgets, pytest Qt widget tests.

---

## File Structure

- `src/cellflow/napari/ui_style.py`: Owns the new `stage_header_action_button` helper and stage-pill QSS generation.
- `src/cellflow/napari/widgets.py`: Applies the helper to the shared Pipeline Files header and moves its toggle next to the label.
- `src/cellflow/napari/cell_workflow_widget.py`: Applies the helper to cell stage-row controls and leaves them directly after the title.
- `src/cellflow/napari/cell_correction_widget.py`: Applies the helper to correction header controls and moves them next to the title.
- `src/cellflow/napari/nucleus_pipeline_widget.py`: Applies the helper to nucleus pipeline stage-row controls and leaves them directly after the title.
- `src/cellflow/napari/nucleus_db_browser_widget.py`: Applies the helper to the database-browser active button and moves it next to the title.
- `src/cellflow/napari/nucleus_correction_widget.py`: Applies the helper to correction header controls and moves them next to the title.
- `tests/napari/test_ui_style.py`: Covers shared button styling.
- `tests/napari/test_cell_workflow_widget.py`: Covers cell row/header layout order.
- `tests/napari/test_nucleus_tracking_correction_layout.py`: Covers nucleus row/header layout order.

---

### Task 1: Shared Stage-Header Action Button Style

**Files:**
- Modify: `src/cellflow/napari/ui_style.py`
- Test: `tests/napari/test_ui_style.py`

- [ ] **Step 1: Write the failing style-helper test**

Add `stage_header_action_button` to the import list in `tests/napari/test_ui_style.py`:

```python
from cellflow.napari.ui_style import (
    DEFAULT_SPIN_WIDTH,
    FIELD_NOTES,
    SECTION_MARGIN,
    SOLARIZED_DARK,
    TIGHT_SPACING,
    TINY_MARGIN,
    action_button,
    checked_success_button,
    compact_spinbox,
    danger_button,
    icon_button,
    muted_label,
    muted_stage_accent,
    parameter_heading,
    stage_header_action_button,
    stage_header_label,
    stage_header_pill_background,
    status_label,
    tiny_button,
)
```

Add this test near `test_stage_header_label_uses_accent_pill_style`:

```python
def test_stage_header_action_button_uses_matching_pill_style(_app):
    button = QToolButton()
    button.setText("⚙")
    button.setCheckable(True)

    assert stage_header_action_button(button, "nucleus") is button

    style = button.styleSheet()
    assert f"color: {muted_stage_accent('nucleus')};" in style
    assert f"background-color: {stage_header_pill_background('nucleus')};" in style
    assert "QToolButton:checked" in style
    assert "border-radius: 4px" in style
    assert button.property("cellflow_stage_key") == "nucleus"
    assert button.property("cellflow_stage_header_action") is True
    assert button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
    assert button.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed
```

- [ ] **Step 2: Run the failing style-helper test**

Run:

```bash
pytest tests/napari/test_ui_style.py::test_stage_header_action_button_uses_matching_pill_style -v
```

Expected: FAIL with an import error for `stage_header_action_button`.

- [ ] **Step 3: Implement the shared helper**

In `src/cellflow/napari/ui_style.py`, add `QToolButton` to the imports:

```python
from qtpy.QtWidgets import QFormLayout, QGridLayout, QLabel, QSizePolicy, QToolButton
```

Add this helper after `stage_header_label`:

```python
def stage_header_action_button(button: QToolButton, stage_key: str, size_px: int = 22):
    button.setProperty("cellflow_stage_key", stage_key)
    button.setProperty("cellflow_stage_header_action", True)
    button.setFixedSize(size_px, size_px)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    color = muted_stage_accent(stage_key)
    background = stage_header_pill_background(stage_key)
    button.setStyleSheet(
        "QToolButton { "
        "font-weight: bold; "
        "font-size: 9pt; "
        f"color: {color}; "
        f"background-color: {background}; "
        f"border: 1px solid {color}; "
        "border-radius: 4px; "
        "padding: 1px 4px; "
        "} "
        "QToolButton:hover { "
        f"background-color: {stage_header_pill_background(stage_key, alpha=58)}; "
        "} "
        "QToolButton:checked { "
        f"background-color: {stage_header_pill_background(stage_key, alpha=82)}; "
        f"border: 1px solid {stage_accent(stage_key)}; "
        "} "
        "QToolButton:disabled { "
        "color: palette(mid); "
        "border-color: palette(mid); "
        "background-color: transparent; "
        "}"
    )
    return button
```

Update `stage_header_pill_background` to accept the alpha used above:

```python
def stage_header_pill_background(stage_key: str, alpha: int = 38) -> str:
    color = QColor(muted_stage_accent(stage_key))
    red, green, blue, _ = color.getRgb()
    return f"rgba({red}, {green}, {blue}, {alpha})"
```

- [ ] **Step 4: Run the style tests**

Run:

```bash
pytest tests/napari/test_ui_style.py::test_stage_header_label_uses_accent_pill_style tests/napari/test_ui_style.py::test_stage_header_action_button_uses_matching_pill_style -v
```

Expected: PASS.

- [ ] **Step 5: Commit the shared helper**

```bash
git add src/cellflow/napari/ui_style.py tests/napari/test_ui_style.py
git commit -m "Add stage header action button style"
```

---

### Task 2: Cell Workflow Header Button Clusters

**Files:**
- Modify: `src/cellflow/napari/cell_workflow_widget.py`
- Modify: `src/cellflow/napari/cell_correction_widget.py`
- Test: `tests/napari/test_cell_workflow_widget.py`

- [ ] **Step 1: Write failing cell layout tests**

Add `QWidget` to the Qt imports in `tests/napari/test_cell_workflow_widget.py`:

```python
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QWidget,
)
```

Add these helpers after `_progress_bars`:

```python
def _layout_widgets(layout):
    return [
        layout.itemAt(i).widget()
        for i in range(layout.count())
        if layout.itemAt(i).widget() is not None
    ]


def _layout_for_row_containing(root, *widgets):
    wanted = set(widgets)
    for layout in root.findChildren(QVBoxLayout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            row = item.layout()
            if row is None:
                continue
            row_widgets = _layout_widgets(row)
            if wanted <= set(row_widgets):
                return row_widgets
    raise AssertionError("No layout row contained all requested widgets")
```

Also import `QVBoxLayout`:

```python
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
```

Add this test after `test_widget_exposes_stage_rows_with_inline_params`:

```python
def test_cell_stage_row_buttons_are_clustered_next_to_title(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    flow_label = next(child for child in widget.findChildren(QLabel) if child.text() == "Flow filtering")
    contour_label = next(child for child in widget.findChildren(QLabel) if child.text() == "Contours")

    flow_widgets = _layout_for_row_containing(widget, flow_label, widget.flow_params_btn, widget.flow_run_btn)
    assert flow_widgets[:3] == [flow_label, widget.flow_params_btn, widget.flow_run_btn]

    contour_widgets = _layout_for_row_containing(
        widget,
        contour_label,
        widget.contour_params_btn,
        widget.contour_preview_btn,
        widget.contour_run_btn,
    )
    assert contour_widgets[:4] == [
        contour_label,
        widget.contour_params_btn,
        widget.contour_preview_btn,
        widget.contour_run_btn,
    ]

    for button in (widget.flow_params_btn, widget.flow_run_btn, widget.contour_preview_btn):
        assert button.property("cellflow_stage_header_action") is True
        assert "border-radius: 4px" in button.styleSheet()

    widget.deleteLater()
    app.processEvents()
```

Update `test_cell_correction_uses_stage_style_header` to assert the correction controls are clustered:

```python
    header_widgets = _layout_widgets(widget.correction_header.layout())
    assert header_widgets[:4] == [
        widget.correction_header_lbl,
        widget.correction_shortcuts_btn,
        widget.correction_params_btn,
        widget.correction_active_btn,
    ]
    for button in (
        widget.correction_shortcuts_btn,
        widget.correction_params_btn,
        widget.correction_active_btn,
    ):
        assert button.property("cellflow_stage_header_action") is True
        assert "border-radius: 4px" in button.styleSheet()
```

- [ ] **Step 2: Run the failing cell layout tests**

Run:

```bash
pytest tests/napari/test_cell_workflow_widget.py::test_cell_stage_row_buttons_are_clustered_next_to_title tests/napari/test_cell_workflow_widget.py::test_cell_correction_uses_stage_style_header -v
```

Expected: FAIL because cell stage-row buttons and correction buttons are not yet styled with `cellflow_stage_header_action`; correction may also fail if the current layout still inserts stretch before the controls.

- [ ] **Step 3: Implement cell workflow button styling and clustering**

In `src/cellflow/napari/cell_workflow_widget.py`, add `stage_header_action_button` to the import from `cellflow.napari.ui_style`:

```python
from cellflow.napari.ui_style import (
    stage_header_action_button,
    stage_header_label,
    status_label,
)
```

After each `_tool_btn` assignment in `_build_pipeline_stage_rows`, apply the helper:

```python
        for button in (
            self.flow_params_btn,
            self.flow_run_btn,
            self.foreground_params_btn,
            self.foreground_run_btn,
            self.contour_params_btn,
            self.contour_preview_btn,
            self.contour_run_btn,
            self.segmentation_params_btn,
            self.segmentation_run_btn,
        ):
            stage_header_action_button(button, "cell")
```

Keep `_stage_row` in this order so the trailing widgets are placed before the stretch:

```python
    @staticmethod
    def _stage_row(label: QLabel, *trailing: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(label)
        for widget in trailing:
            row.addWidget(widget)
        row.addStretch(1)
        return row
```

In `src/cellflow/napari/cell_correction_widget.py`, add `stage_header_action_button` to the import from `cellflow.napari.ui_style`:

```python
from cellflow.napari.ui_style import (
    add_block_pair_row,
    block_grid,
    compact_spinbox,
    stage_header_action_button,
    stage_header_label,
)
```

In `_build_correction_header`, style the header controls and move the stretch after them:

```python
        self.correction_header_lbl = QLabel("Correction")
        stage_header_label(self.correction_header_lbl, "cell")
        for button in (self.shortcuts_btn, self.params_btn, self.active_btn):
            stage_header_action_button(button, "cell")
        row.addWidget(self.correction_header_lbl)
        row.addWidget(self.shortcuts_btn)
        row.addWidget(self.params_btn)
        row.addWidget(self.active_btn)
        row.addStretch(1)
        return header
```

- [ ] **Step 4: Run the cell layout tests**

Run:

```bash
pytest tests/napari/test_cell_workflow_widget.py::test_cell_stage_row_buttons_are_clustered_next_to_title tests/napari/test_cell_workflow_widget.py::test_cell_correction_uses_stage_style_header -v
```

Expected: PASS.

- [ ] **Step 5: Commit the cell workflow changes**

```bash
git add src/cellflow/napari/cell_workflow_widget.py src/cellflow/napari/cell_correction_widget.py tests/napari/test_cell_workflow_widget.py
git commit -m "Cluster cell stage header buttons"
```

---

### Task 3: Nucleus and Shared Header Button Clusters

**Files:**
- Modify: `src/cellflow/napari/widgets.py`
- Modify: `src/cellflow/napari/nucleus_pipeline_widget.py`
- Modify: `src/cellflow/napari/nucleus_db_browser_widget.py`
- Modify: `src/cellflow/napari/nucleus_correction_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write failing nucleus layout tests**

Add `QVBoxLayout` to the Qt imports in `tests/napari/test_nucleus_tracking_correction_layout.py`:

```python
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
```

Add these helpers near `_make_viewer`:

```python
def _layout_widgets(layout):
    return [
        layout.itemAt(i).widget()
        for i in range(layout.count())
        if layout.itemAt(i).widget() is not None
    ]


def _layout_for_row_containing(root, *widgets):
    wanted = set(widgets)
    for layout in root.findChildren(QVBoxLayout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            row = item.layout()
            if row is None:
                continue
            row_widgets = _layout_widgets(row)
            if wanted <= set(row_widgets):
                return row_widgets
    raise AssertionError("No layout row contained all requested widgets")
```

Add this test before `test_correction_section_uses_stage_header_params_activate_and_active_toolbar`:

```python
def test_nucleus_stage_row_buttons_are_clustered_next_to_title():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_label = next(child for child in widget.findChildren(QLabel) if child.text() == "Ultrack database")
    solve_label = next(child for child in widget.findChildren(QLabel) if child.text() == "Ultrack solve")

    db_widgets = _layout_for_row_containing(widget, db_label, widget.db_params_btn, widget.db_run_btn)
    assert db_widgets[:3] == [db_label, widget.db_params_btn, widget.db_run_btn]

    solve_widgets = _layout_for_row_containing(widget, solve_label, widget.solve_params_btn, widget.solve_run_btn)
    assert solve_widgets[:3] == [solve_label, widget.solve_params_btn, widget.solve_run_btn]

    pipeline_widgets = _layout_widgets(widget.pipeline_files_header.layout())
    assert pipeline_widgets[:2] == [widget.pipeline_files_header_lbl, widget.pipeline_files_toggle_btn]

    db_browser_widgets = _layout_widgets(widget.ultrack_db_browser_header.layout())
    assert db_browser_widgets[:2] == [
        widget.ultrack_db_browser_header_lbl,
        widget.ultrack_db_active_btn,
    ]

    for button in (
        widget.db_params_btn,
        widget.db_run_btn,
        widget.solve_params_btn,
        widget.solve_run_btn,
        widget.pipeline_files_toggle_btn,
        widget.ultrack_db_active_btn,
    ):
        assert button.property("cellflow_stage_header_action") is True
        assert "border-radius: 4px" in button.styleSheet()

    widget.deleteLater()
    viewer.close()
```

`test_correction_section_uses_stage_header_params_activate_and_active_toolbar` already asserts correction header order. Extend it with style assertions:

```python
    for button in (
        widget.correction_shortcuts_btn,
        widget.correction_params_btn,
        widget.correction_active_btn,
    ):
        assert button.property("cellflow_stage_header_action") is True
        assert "border-radius: 4px" in button.styleSheet()
```

- [ ] **Step 2: Run the failing nucleus layout tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_nucleus_stage_row_buttons_are_clustered_next_to_title tests/napari/test_nucleus_tracking_correction_layout.py::test_correction_section_uses_stage_header_params_activate_and_active_toolbar -v
```

Expected: FAIL because nucleus/shared header buttons are not yet styled and some rows still add stretch before their controls.

- [ ] **Step 3: Implement shared Pipeline Files header clustering**

In `src/cellflow/napari/widgets.py`, add `stage_header_action_button` to the import from `.ui_style`:

```python
from .ui_style import (
    SECTION_MARGIN,
    TIGHT_SPACING,
    TINY_MARGIN,
    icon_button,
    muted_accent,
    muted_label,
    stage_header_action_button,
    stage_header_label,
    stage_status_color,
    status_label,
)
```

In `make_pipeline_files_header`, style the button and move the stretch after it:

```python
    button = tool_btn("🔍", "Show pipeline files.", checkable=True)
    stage_header_action_button(button, stage_key)
    button.setChecked(section.is_expanded)
```

Replace the layout tail with:

```python
    layout.addWidget(label)
    layout.addWidget(button)
    layout.addStretch(1)
    return header, label, button
```

- [ ] **Step 4: Implement nucleus pipeline row clustering**

In `src/cellflow/napari/nucleus_pipeline_widget.py`, import the helper:

```python
from cellflow.napari.ui_style import (
    stage_header_action_button as _stage_header_action_button,
    stage_header_label as _stage_header_label,
)
```

After the six stage buttons are created in `__init__`, style them:

```python
        for button in (
            self.seg_params_btn,
            self.seg_run_btn,
            self.db_params_btn,
            self.db_run_btn,
            self.solve_params_btn,
            self.solve_run_btn,
        ):
            _stage_header_action_button(button, "nucleus")
```

Update `_stage_row` in `build_pipeline_block` so the trailing controls stay next to the label:

```python
        def _stage_row(label: QLabel, *trailing: QWidget) -> QHBoxLayout:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(label)
            for w in trailing:
                row.addWidget(w)
            row.addStretch(1)
            return row
```

- [ ] **Step 5: Implement nucleus browser and correction header clustering**

In `src/cellflow/napari/nucleus_db_browser_widget.py`, import the helper:

```python
from cellflow.napari.ui_style import (
    stage_header_action_button as _stage_header_action_button,
    stage_header_label as _stage_header_label,
)
```

In `__init__`, after creating `self.active_btn`, style it and move the stretch:

```python
        self.active_btn.setChecked(False)
        _stage_header_action_button(self.active_btn, "nucleus")
        header_lay.addWidget(self.header_lbl)
        header_lay.addWidget(self.active_btn)
        header_lay.addStretch(1)
```

In `src/cellflow/napari/nucleus_correction_widget.py`, add `stage_header_action_button` to the import from `cellflow.napari.ui_style`:

```python
from cellflow.napari.ui_style import (
    add_block_checkbox_row,
    add_block_pair_row,
    block_grid,
    compact_spinbox,
    danger_button,
    stage_header_action_button,
    stage_header_label,
)
```

In `_build_correction_header`, style the header controls and move the stretch after them:

```python
        self.header_lbl = QLabel("Correction")
        stage_header_label(self.header_lbl, "nucleus")
        for button in (self.shortcuts_btn, self.params_btn, self.active_btn):
            stage_header_action_button(button, "nucleus")
        row.addWidget(self.header_lbl)
        row.addWidget(self.shortcuts_btn)
        row.addWidget(self.params_btn)
        row.addWidget(self.active_btn)
        row.addStretch(1)
        return header
```

- [ ] **Step 6: Run the nucleus layout tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_nucleus_stage_row_buttons_are_clustered_next_to_title tests/napari/test_nucleus_tracking_correction_layout.py::test_correction_section_uses_stage_header_params_activate_and_active_toolbar -v
```

Expected: PASS.

- [ ] **Step 7: Commit the nucleus/shared header changes**

```bash
git add src/cellflow/napari/widgets.py src/cellflow/napari/nucleus_pipeline_widget.py src/cellflow/napari/nucleus_db_browser_widget.py src/cellflow/napari/nucleus_correction_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "Cluster nucleus stage header buttons"
```

---

### Task 4: Focused Regression Run

**Files:**
- Verify only.

- [ ] **Step 1: Run the focused style and workflow tests**

Run:

```bash
pytest tests/napari/test_ui_style.py tests/napari/test_cell_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py -q
```

Expected: PASS.

- [ ] **Step 2: Check the final diff and status**

Run:

```bash
git status --short
git diff --stat HEAD
```

Expected: no unstaged changes from this plan except any pre-existing local edits that were present before implementation.

- [ ] **Step 3: Commit any final test-only adjustment if needed**

Only if Step 1 required a small test correction, commit the exact touched paths:

```bash
git add tests/napari/test_ui_style.py tests/napari/test_cell_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "Test stage header button clusters"
```

If Step 1 passes without changes, skip this commit.
