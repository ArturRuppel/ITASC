# Correction Mode Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make correction mode visibly active and easy to exit in both nucleus and cell correction widgets without blocking other workflow actions.

**Architecture:** Add small Qt helper factories in `_widget_helpers.py` for the prominent checkable mode button and active banner. Wire those helpers into `NucleusCorrectionWidget` and `CellCorrectionWidget` while preserving the existing `active_btn` activation/deactivation path. Update focused widget and workflow layout tests to assert visible state, exit behavior, and permissive controls.

**Tech Stack:** Python, Qt via `qtpy`, napari test viewers, pytest.

---

## File Structure

- Modify `src/cellflow/napari/_widget_helpers.py`
  - Add shared helper functions for a prominent correction-mode toggle, active-state banner, and toggle text/style synchronization.
- Modify `src/cellflow/napari/nucleus_correction_widget.py`
  - Replace the compact power icon with the prominent `active_btn`.
  - Add active banner and `Exit Correction` button inside active correction content.
  - Keep cleanup centralized by toggling `active_btn`.
- Modify `src/cellflow/napari/cell_correction_widget.py`
  - Apply the same header and banner pattern as nucleus correction.
- Modify `tests/napari/test_nucleus_correction_widget.py`
  - Add direct coverage for the prominent toggle, active banner, and exit button.
- Modify `tests/napari/test_cell_correction_widget.py`
  - Add direct coverage for the same behavior in the cell correction widget.
- Modify `tests/napari/test_nucleus_tracking_correction_layout.py`
  - Update workflow-level expectations from the old `⏻` `QToolButton` to the prominent toggle.
- Modify `tests/napari/test_cell_workflow_widget.py`
  - Update workflow-level expectations from the old `⏻` `QToolButton` to the prominent toggle.

Do not modify `src/cellflow/napari/ui_style.py`; it has unrelated local edits in the current worktree.

---

### Task 1: Shared Correction Mode UI Helpers

**Files:**
- Modify: `src/cellflow/napari/_widget_helpers.py:5-18`
- Modify: `src/cellflow/napari/_widget_helpers.py:248-255`

- [ ] **Step 1: Add imports for the helper widgets**

In `src/cellflow/napari/_widget_helpers.py`, extend the `qtpy.QtWidgets` import block so it includes `QHBoxLayout`:

```python
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleOption,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
```

- [ ] **Step 2: Add correction-mode helper functions**

Append these functions after `tool_btn()` in `src/cellflow/napari/_widget_helpers.py`:

```python
def correction_mode_btn(tooltip: str = "") -> QPushButton:
    """Prominent checkable button used as the correction mode on/off switch."""
    b = QPushButton("Correction Mode")
    b.setToolTip(tooltip)
    b.setCheckable(True)
    b.setMinimumWidth(132)
    b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    sync_correction_mode_btn(b)
    return b


def sync_correction_mode_btn(button: QPushButton) -> None:
    """Update correction mode button text and checked-state styling."""
    if button.isChecked():
        button.setText("Correction Active")
        button.setStyleSheet(
            "QPushButton { font-weight: 700; padding: 3px 8px; "
            "border: 1px solid #f9e2af; background: rgba(249, 226, 175, 45); }"
            "QPushButton:checked { color: palette(text); }"
        )
    else:
        button.setText("Correction Mode")
        button.setStyleSheet(
            "QPushButton { font-weight: 600; padding: 3px 8px; }"
        )


def correction_active_banner(parent: QWidget | None = None) -> tuple[QWidget, QLabel, QPushButton]:
    """Build the active correction banner and exit button."""
    banner = QWidget(parent)
    row = QHBoxLayout(banner)
    row.setContentsMargins(6, 4, 6, 4)
    row.setSpacing(6)
    label = QLabel("Correction mode active")
    label.setStyleSheet("font-weight: 700;")
    exit_btn = QPushButton("Exit Correction")
    exit_btn.setToolTip("Turn off correction mode and restore the previous viewer state.")
    exit_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    row.addWidget(label)
    row.addStretch(1)
    row.addWidget(exit_btn)
    banner.setStyleSheet(
        "QWidget { border: 1px solid #f9e2af; background: rgba(249, 226, 175, 35); }"
    )
    banner.setVisible(False)
    return banner, label, exit_btn
```

- [ ] **Step 3: Run syntax check**

Run:

```bash
python -m py_compile src/cellflow/napari/_widget_helpers.py
```

Expected: exits `0`.

- [ ] **Step 4: Commit**

```bash
git add src/cellflow/napari/_widget_helpers.py
git commit -m "feat: add correction mode UI helpers"
```

---

### Task 2: Nucleus Correction Visibility

**Files:**
- Modify: `tests/napari/test_nucleus_correction_widget.py`
- Modify: `src/cellflow/napari/nucleus_correction_widget.py`

- [ ] **Step 1: Write failing nucleus widget test**

Add this test near the existing correction activation tests in `tests/napari/test_nucleus_correction_widget.py`:

```python
def test_nucleus_correction_mode_has_prominent_toggle_and_exit(monkeypatch):
    _app, viewer = _make_viewer()
    widget, _module = _make_widget(viewer)
    viewer.add_labels(np.zeros((1, 4, 4), dtype=np.uint32), name="[Correction] Nucleus Labels")
    states = []

    monkeypatch.setattr(widget, "_capture_correction_view_state", lambda: None)
    monkeypatch.setattr(widget, "_load_correction_layers_from_disk", lambda: True)
    monkeypatch.setattr(widget, "_restore_correction_view_state", lambda: None)
    monkeypatch.setattr(widget, "_refresh_tracked_layer_from_disk", lambda: None)
    monkeypatch.setattr(widget, "_remove_correction_owned_layers", lambda: None)
    monkeypatch.setattr(widget, "_refresh_refinement_widget", lambda: None)
    monkeypatch.setattr(
        widget.correction_widget,
        "activate_layer",
        lambda layer: states.append(("activate", layer.name)),
    )
    monkeypatch.setattr(
        widget.correction_widget,
        "deactivate",
        lambda: states.append(("deactivate", None)),
    )

    assert widget.correction_active_btn.text() == "Correction Mode"
    assert widget.correction_active_banner.isVisible() is False
    assert widget.exit_correction_btn.text() == "Exit Correction"

    widget.correction_active_btn.setChecked(True)

    assert widget.correction_active_btn.text() == "Correction Active"
    assert widget.correction_active_banner.isVisible() is True
    assert widget.correction_mode_section.is_expanded is True

    widget.exit_correction_btn.click()

    assert widget.correction_active_btn.isChecked() is False
    assert widget.correction_active_btn.text() == "Correction Mode"
    assert widget.correction_active_banner.isVisible() is False
    assert states == [
        ("activate", "[Correction] Nucleus Labels"),
        ("deactivate", None),
    ]

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/napari/test_nucleus_correction_widget.py::test_nucleus_correction_mode_has_prominent_toggle_and_exit -q
```

Expected: FAIL with an `AttributeError` for `correction_active_banner` or with the old active button text `⏻`.

- [ ] **Step 3: Import helper functions**

In `src/cellflow/napari/nucleus_correction_widget.py`, extend the `_widget_helpers` import:

```python
from cellflow.napari._widget_helpers import (
    btn as _btn,
    correction_active_banner,
    correction_mode_btn,
    dspin as _dspin,
    heading as _heading,
    ispin as _ispin,
    make_status as _make_status,
    sync_correction_mode_btn,
    tool_btn as _tool_btn,
)
```

- [ ] **Step 4: Replace active button construction**

Replace the current `self.active_btn = _tool_btn(...)` block with:

```python
self.active_btn = correction_mode_btn(
    "Activate correction mode and show correction layers and controls."
)
```

Leave `params_btn` and `shortcuts_btn` as `_tool_btn(...)`.

- [ ] **Step 5: Add active banner to the active content**

In `_setup_ui()`, immediately before `active_lay.addWidget(self.toolbar)`, add:

```python
(
    self.correction_active_banner,
    self.correction_active_banner_lbl,
    self.exit_correction_btn,
) = correction_active_banner(self.active_content)
active_lay.addWidget(self.correction_active_banner)
```

- [ ] **Step 6: Put the mode toggle before compact icons in the header**

In `_build_correction_header()`, keep `Correction` on the left and then add the prominent active button before the compact controls:

```python
row.addWidget(self.header_lbl)
row.addStretch(1)
row.addWidget(self.active_btn)
row.addWidget(self.shortcuts_btn)
row.addWidget(self.params_btn)
return header
```

- [ ] **Step 7: Wire exit button and sync active-state UI**

In `_connect_signals()`, after the active button connection, add:

```python
self.exit_correction_btn.clicked.connect(lambda: self.active_btn.setChecked(False))
```

In `_sync_correction_panel_visibility()`, after `self.active_content.setVisible(show_active)`, add:

```python
self.correction_active_banner.setVisible(show_active)
sync_correction_mode_btn(self.active_btn)
```

- [ ] **Step 8: Run focused nucleus test**

Run:

```bash
pytest tests/napari/test_nucleus_correction_widget.py::test_nucleus_correction_mode_has_prominent_toggle_and_exit -q
```

Expected: PASS.

- [ ] **Step 9: Run nucleus correction suite**

Run:

```bash
pytest tests/napari/test_nucleus_correction_widget.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/cellflow/napari/nucleus_correction_widget.py tests/napari/test_nucleus_correction_widget.py
git commit -m "feat: make nucleus correction mode visible"
```

---

### Task 3: Cell Correction Visibility

**Files:**
- Modify: `tests/napari/test_cell_correction_widget.py`
- Modify: `src/cellflow/napari/cell_correction_widget.py`

- [ ] **Step 1: Write failing cell widget test**

Add this test in the UI structure section of `tests/napari/test_cell_correction_widget.py`:

```python
def test_cell_correction_mode_has_prominent_toggle_and_exit(monkeypatch):
    _app, viewer = _make_viewer()
    widget, _module = _make_widget(viewer)
    viewer.add_labels(np.zeros((1, 4, 4), dtype=np.uint32), name="[Correction] Cell Labels")
    states = []

    monkeypatch.setattr(widget, "_capture_correction_view_state", lambda: None)
    monkeypatch.setattr(widget, "_load_correction_layers_from_disk", lambda: True)
    monkeypatch.setattr(widget, "_restore_correction_view_state", lambda: None)
    monkeypatch.setattr(widget, "_refresh_tracked_layer_from_disk", lambda: None)
    monkeypatch.setattr(widget, "_remove_correction_owned_layers", lambda: None)
    monkeypatch.setattr(
        widget.correction_widget,
        "activate_layer",
        lambda layer: states.append(("activate", layer.name)),
    )
    monkeypatch.setattr(
        widget.correction_widget,
        "deactivate",
        lambda: states.append(("deactivate", None)),
    )

    assert widget.correction_active_btn.text() == "Correction Mode"
    assert widget.correction_active_banner.isVisible() is False
    assert widget.exit_correction_btn.text() == "Exit Correction"

    widget.correction_active_btn.setChecked(True)

    assert widget.correction_active_btn.text() == "Correction Active"
    assert widget.correction_active_banner.isVisible() is True
    assert widget.correction_mode_section.is_expanded is True

    widget.exit_correction_btn.click()

    assert widget.correction_active_btn.isChecked() is False
    assert widget.correction_active_btn.text() == "Correction Mode"
    assert widget.correction_active_banner.isVisible() is False
    assert states == [
        ("activate", "[Correction] Cell Labels"),
        ("deactivate", None),
    ]

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/napari/test_cell_correction_widget.py::test_cell_correction_mode_has_prominent_toggle_and_exit -q
```

Expected: FAIL with an `AttributeError` for `correction_active_banner` or with the old active button text `⏻`.

- [ ] **Step 3: Import helper functions**

In `src/cellflow/napari/cell_correction_widget.py`, extend the `_widget_helpers` import:

```python
from cellflow.napari._widget_helpers import (
    btn as _btn,
    button_grid as _button_grid,
    correction_active_banner,
    correction_mode_btn,
    make_status as _make_status,
    sync_correction_mode_btn,
    tool_btn as _tool_btn,
)
```

- [ ] **Step 4: Replace active button construction**

Replace the current `self.active_btn = _tool_btn(...)` block with:

```python
self.active_btn = correction_mode_btn(
    "Activate correction mode and show correction controls."
)
```

- [ ] **Step 5: Add active banner to active content**

In `_setup_ui()`, immediately before `active_lay.addWidget(self.correction_widget)`, add:

```python
(
    self.correction_active_banner,
    self.correction_active_banner_lbl,
    self.exit_correction_btn,
) = correction_active_banner(self.active_content)
active_lay.addWidget(self.correction_active_banner)
```

- [ ] **Step 6: Put the mode toggle before compact icons in the header**

In `_build_correction_header()`, keep `Correction` on the left and then add:

```python
row.addWidget(self.correction_header_lbl)
row.addStretch(1)
row.addWidget(self.active_btn)
row.addWidget(self.shortcuts_btn)
row.addWidget(self.params_btn)
return header
```

- [ ] **Step 7: Wire exit button and sync active-state UI**

In `_connect_signals()`, after the active button connection, add:

```python
self.exit_correction_btn.clicked.connect(lambda: self.active_btn.setChecked(False))
```

In `_sync_correction_panel_visibility()`, after `self.active_content.setVisible(show_active)`, add:

```python
self.correction_active_banner.setVisible(show_active)
sync_correction_mode_btn(self.active_btn)
```

- [ ] **Step 8: Run focused cell test**

Run:

```bash
pytest tests/napari/test_cell_correction_widget.py::test_cell_correction_mode_has_prominent_toggle_and_exit -q
```

Expected: PASS.

- [ ] **Step 9: Run cell correction suite**

Run:

```bash
pytest tests/napari/test_cell_correction_widget.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/cellflow/napari/cell_correction_widget.py tests/napari/test_cell_correction_widget.py
git commit -m "feat: make cell correction mode visible"
```

---

### Task 4: Workflow Layout Test Updates

**Files:**
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`
- Modify: `tests/napari/test_cell_workflow_widget.py`

- [ ] **Step 1: Update nucleus workflow expectations**

In `test_correction_section_uses_stage_header_params_activate_and_active_toolbar`, replace the old active button expectations:

```python
assert widget.correction_active_btn.text() == "Correction Mode"
assert widget.correction_active_btn.isCheckable() is True
assert widget.correction_active_btn.minimumWidth() >= 120
assert isinstance(widget.correction_shortcuts_btn, QToolButton)
assert isinstance(widget.correction_params_btn, QToolButton)
```

Remove the assertion that `widget.correction_active_btn` is a `QToolButton`.

- [ ] **Step 2: Update nucleus active-state assertions**

Where the test activates correction with `widget.correction_active_btn.setChecked(True)`, add:

```python
assert widget.correction_active_btn.text() == "Correction Active"
assert widget.correction_active_banner.isVisible() is True
```

After deactivation assertions, add:

```python
assert widget.correction_active_btn.text() == "Correction Mode"
assert widget.correction_active_banner.isVisible() is False
```

- [ ] **Step 3: Update cell workflow expectations**

In the cell correction header test in `tests/napari/test_cell_workflow_widget.py`, replace the old active button expectations:

```python
assert widget.correction_active_btn.text() == "Correction Mode"
assert widget.correction_active_btn.isCheckable() is True
assert widget.correction_active_btn.minimumWidth() >= 120
assert isinstance(widget.correction_shortcuts_btn, QToolButton)
assert isinstance(widget.correction_params_btn, QToolButton)
```

Remove the assertion that `widget.correction_active_btn` is a `QToolButton`.

- [ ] **Step 4: Run workflow layout tests to verify failures are resolved**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_correction_section_uses_stage_header_params_activate_and_active_toolbar tests/napari/test_cell_workflow_widget.py::test_cell_correction_uses_stage_style_header -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/napari/test_nucleus_tracking_correction_layout.py tests/napari/test_cell_workflow_widget.py
git commit -m "test: update correction mode layout expectations"
```

---

### Task 5: Final Verification

**Files:**
- Verify only; no file edits expected.

- [ ] **Step 1: Run focused correction and layout tests**

Run:

```bash
pytest tests/napari/test_nucleus_correction_widget.py tests/napari/test_cell_correction_widget.py tests/napari/test_nucleus_tracking_correction_layout.py tests/napari/test_cell_workflow_widget.py -q
```

Expected: PASS.

- [ ] **Step 2: Run syntax checks for modified source files**

Run:

```bash
python -m py_compile src/cellflow/napari/_widget_helpers.py src/cellflow/napari/nucleus_correction_widget.py src/cellflow/napari/cell_correction_widget.py
```

Expected: exits `0`.

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected: only unrelated pre-existing local changes remain, or a clean tree if those were handled separately.

---

## Spec Coverage Review

- Prominent always-visible correction state: Task 1 helper plus Tasks 2 and 3 header integration.
- Large active exit affordance: Task 1 banner plus Tasks 2 and 3 exit button wiring.
- Permissive behavior: no task disables pipeline, DB browser, file panel, or other workflow controls.
- Centralized activation/deactivation: Tasks 2 and 3 wire `Exit Correction` through `active_btn.setChecked(False)`.
- Both widgets: Task 2 covers nucleus; Task 3 covers cell.
- Existing params and shortcuts preserved: Tasks 2 and 3 leave `_tool_btn` params/shortcuts controls intact; Task 4 keeps workflow assertions for those compact controls.
- Tests: Tasks 2, 3, 4, and 5 cover direct widget behavior and workflow aliases.

---

## Follow-Up TODO: Louder Exclusive Viewer States

Preview mode, correction mode, and database-browser mode are mutually exclusive
viewer states. When one is active, the workflow disables buttons that would
conflict with the active state. That behavior is correct, but the UI needs to
make the active state louder so users can immediately tell why other buttons are
disabled and what they need to exit first.

- [ ] Add a persistent active-state banner for source preview mode.
  - Label it clearly as preview/source-preview active.
  - Include an obvious exit/stop action.
  - Update disabled button tooltips to say they are unavailable while preview mode is active.

- [ ] Add or keep a persistent active-state banner for correction mode.
  - Label it clearly as correction mode active.
  - Include an obvious exit action.
  - Update disabled button tooltips to say they are unavailable while correction mode is active.
  - When correction mode has unsaved changes, prompt before deactivation with a file dialog asking whether to save the corrections.
  - Keep correction mode active if the user cancels the save/deactivation prompt.

- [ ] Add a persistent active-state banner for database-browser mode.
  - Label it clearly as database browser active.
  - Include an obvious exit/close action.
  - Update disabled button tooltips to say they are unavailable while database browser mode is active.

- [ ] Make Ultrack database preview layers behave like correction-mode layers.
  - Add a `[Preview]` prefix/tag to layers created by Ultrack database preview mode.
  - Disable or hide unrelated viewer layers while preview mode is active so the preview state is unambiguous.
  - Restore previously visible/enabled layers after preview mode is deactivated.
  - Remove all `[Preview]` layers when preview mode exits so stale preview overlays do not remain in the viewer.

- [ ] Centralize the disabled-reason text for exclusive viewer states.
  - Use the same wording from banners, button tooltips, and status messages.
  - Cover `source_threshold_preview_check`, `ultrack_db_active_btn`, `correction_active_btn`, and the pipeline run/parameter buttons disabled by `_sync_viewer_activity_controls()`.
  - Prefer a small helper over repeating ad hoc strings across widgets.

- [ ] Remove right-side status indicators from stage headers.
  - Keep stage-header rows focused on the title and its action buttons.
  - Put state in the shared status/progress area, active-state banners, checked button state, and disabled-reason tooltips instead.
  - Update layout tests so no passive status label is expected on the right side.

- [ ] Add focused Qt tests for each active state.
  - Assert the correct banner is visible.
  - Assert conflicting controls are disabled.
  - Assert each disabled conflicting control explains the active mode in its tooltip.
