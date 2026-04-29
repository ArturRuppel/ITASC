# Nucleus Tracking & Correction UI Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the nucleus napari workflow so tracking and correction appear as one coherent `Tracking & Correction` stage, while preserving existing backend behavior and exposing the missing correction parameters required by the April 29, 2026 spec.

**Architecture:** Keep the existing backend handlers and signal wiring wherever possible. The work is primarily a UI composition refactor in `src/cellflow/napari/nucleus_workflow_widget.py`, with one small presentational extension in `src/cellflow/napari/correction_widget.py` and one minimal parameter-plumbing change for extend/retrack controls.

**Tech Stack:** Python 3.9+, napari, qtpy, numpy, pytest, Qt widgets

---

## File Map

- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
  Responsibility: merge the old tracking and manual-correction sections into the new top-level `Tracking & Correction` layout; move controls; add route-selection and correction parameter widgets; preserve existing handlers.
- Modify: `src/cellflow/napari/correction_widget.py`
  Responsibility: make the correction shortcut reference embeddable inside a nested `CollapsibleSection` without changing correction behavior.
- Modify: `src/cellflow/napari/widgets.py` only if a tiny shared layout helper is truly needed.
  Responsibility: optional shared helper only; do not refactor this file unless the widget code becomes substantially cleaner.
- Create: `tests/napari/test_nucleus_tracking_correction_layout.py`
  Responsibility: smoke-test the new section titles, moved controls, and parameter widgets using narrow widget-construction assertions.

### Task 1: Add a Narrow Widget Smoke Test Harness

**Files:**
- Create: `tests/napari/test_nucleus_tracking_correction_layout.py`
- Modify: none
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import napari
from qtpy.QtWidgets import QApplication

from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget


def _viewer():
    app = QApplication.instance() or QApplication([])
    viewer = napari.Viewer(show=False)
    return app, viewer


def test_tracking_and_correction_shell_exposes_new_section_titles():
    _app, viewer = _viewer()
    widget = NucleusWorkflowWidget(viewer)

    assert widget.tracking_correction_section.title == "4. Tracking & Correction"
    assert widget.ultrack_section.title == "Ultrack Tracking"
    assert widget.correction_section.title == "Correction"

    viewer.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_tracking_and_correction_shell_exposes_new_section_titles -v`

Expected: FAIL because `tracking_correction_section` and `ultrack_section` do not exist yet.

- [ ] **Step 3: Write minimal implementation support**

Add the smallest new attributes in `src/cellflow/napari/nucleus_workflow_widget.py` needed for the shell refactor so the test can target stable names:

```python
self.tracking_correction_section = CollapsibleSection(
    "4. Tracking & Correction", _tracking_correction_inner, expanded=False
)
self.ultrack_section = CollapsibleSection(
    "Ultrack Tracking", _ultrack_inner, expanded=False
)
self.correction_section = CollapsibleSection(
    "Correction", _correction_inner, expanded=False
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_tracking_and_correction_shell_exposes_new_section_titles -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/napari/test_nucleus_tracking_correction_layout.py src/cellflow/napari/nucleus_workflow_widget.py
git commit -m "test: add nucleus tracking correction layout smoke test"
```

### Task 2: Refactor the Top-Level Tracking/Correction Shell

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py:520-680`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Extend the failing test for the merged shell**

Append these assertions:

```python
def test_tracking_and_correction_shell_hides_old_top_level_sections():
    _app, viewer = _viewer()
    widget = NucleusWorkflowWidget(viewer)

    assert widget.tracking_correction_section.title == "4. Tracking & Correction"
    assert widget.correction_section.title == "Correction"
    assert not hasattr(widget, "tracking_section")

    viewer.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py -v`

Expected: FAIL because the widget still creates `self.tracking_section` and labels correction as `5. Manual Correction`.

- [ ] **Step 3: Write minimal implementation**

Replace the old layout with one top-level section containing two nested collapsibles. Keep the old buttons and widgets alive; only move them.

```python
_tracking_correction_inner = QWidget()
tracking_correction_lay = QVBoxLayout(_tracking_correction_inner)
tracking_correction_lay.setContentsMargins(0, 0, 0, 0)
tracking_correction_lay.setSpacing(4)

_ultrack_inner = QWidget()
ultrack_lay = QVBoxLayout(_ultrack_inner)
ultrack_lay.setContentsMargins(0, 0, 0, 0)
ultrack_lay.setSpacing(4)

_correction_inner = QWidget()
correction_lay = QVBoxLayout(_correction_inner)
correction_lay.setContentsMargins(0, 0, 0, 0)
correction_lay.setSpacing(4)

self.ultrack_section = CollapsibleSection("Ultrack Tracking", _ultrack_inner, expanded=False)
self.correction_section = CollapsibleSection("Correction", _correction_inner, expanded=False)
tracking_correction_lay.addWidget(self.ultrack_section)
tracking_correction_lay.addWidget(self.correction_section)

self.tracking_correction_section = CollapsibleSection(
    "4. Tracking & Correction", _tracking_correction_inner, expanded=False
)
layout.addWidget(self.tracking_correction_section)
```

Delete creation of the old `self.tracking_section` and old top-level `5. Manual Correction` wrapper.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py -v`

Expected: PASS

- [ ] **Step 5: Run a syntax smoke check**

Run: `python -m py_compile src/cellflow/napari/nucleus_workflow_widget.py`

Expected: no output

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "refactor: merge nucleus tracking and correction shell"
```

### Task 3: Move Controls Into Their Specified Subsections

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py:520-730`
- Modify: `src/cellflow/napari/correction_widget.py:88-180`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write the failing test for moved controls**

Add assertions that the correction subsection owns persistence and correction affordances:

```python
def test_correction_section_contains_persistence_and_retrack_controls():
    _app, viewer = _viewer()
    widget = NucleusWorkflowWidget(viewer)

    assert widget.save_tracked_btn.parent() is not None
    assert widget.load_tracked_btn.parent() is not None
    assert widget.reassign_ids_btn.parent() is not None
    assert widget.extend_back_btn.parent() is not None
    assert widget.retrack_fwd_btn.parent() is not None

    viewer.close()
```

Also add a shortcut presentation assertion after introducing a stable attribute:

```python
assert widget.correction_shortcuts_section.title == "Correction Shortcuts"
assert widget.correction_shortcuts_section.is_expanded is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py -v`

Expected: FAIL because save/load/reassign still live in the ultrack area and shortcuts are still hardcoded as a `QGroupBox`.

- [ ] **Step 3: Write minimal implementation**

In `src/cellflow/napari/correction_widget.py`, extract the shortcut reference into an embeddable widget factory:

```python
def build_shortcuts_widget(self) -> QWidget:
    container = QWidget()
    lay = QVBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    for key, desc in [...]:
        lay.addWidget(QLabel(f"<tt>{key}</tt>  -  {desc}"))
    return container
```

In `src/cellflow/napari/nucleus_workflow_widget.py`, create and place:

```python
self.correction_shortcuts_section = CollapsibleSection(
    "Correction Shortcuts",
    self.correction_widget.build_shortcuts_widget(),
    expanded=True,
)
```

Move:
- `save_tracked_btn`
- `load_tracked_btn`
- `reassign_ids_btn`
- `extend_back_btn`
- `extend_fwd_btn`
- `retrack_back_btn`
- `retrack_fwd_btn`

into the correction subsection layout, and remove them from the ultrack subsection.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py -v`

Expected: PASS

- [ ] **Step 5: Run syntax smoke checks**

Run: `python -m py_compile src/cellflow/napari/nucleus_workflow_widget.py src/cellflow/napari/correction_widget.py`

Expected: no output

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py src/cellflow/napari/correction_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "refactor: move nucleus correction controls into correction subsection"
```

### Task 4: Unify Ultrack and Resolve-From-Validated

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py:520-730`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write the failing test for route selection and local ultrack status**

Add assertions for the new surface:

```python
def test_ultrack_section_exposes_route_selector_and_local_status():
    _app, viewer = _viewer()
    widget = NucleusWorkflowWidget(viewer)

    assert widget.ultrack_route_check.text() == "Resolve from validated"
    assert widget.ultrack_status_lbl.text() == ""
    assert widget.ultrack_progress_bar.isVisible() is False

    viewer.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_section_exposes_route_selector_and_local_status -v`

Expected: FAIL because there is no route checkbox and no local ultrack status label.

- [ ] **Step 3: Write minimal implementation**

Create:

```python
self.ultrack_route_check = QCheckBox("Resolve from validated")
self.ultrack_status_lbl = QLabel("")
self.ultrack_status_lbl.setWordWrap(True)
```

Use one run row for both routes:

```python
ultrack_lay.addWidget(self.ultrack_route_check)
ultrack_lay.addLayout(ultrack_run_row)
ultrack_lay.addWidget(self.ultrack_status_lbl)
ultrack_lay.addWidget(self.ultrack_progress_bar)
```

Then route the existing actions through small dispatchers:

```python
def _on_run_tracking_route(self) -> None:
    if self.ultrack_route_check.isChecked():
        self._on_resolve_with_validation()
    else:
        self._on_run_ultrack()


def _on_run_tracking_route_terminal(self) -> None:
    if self.ultrack_route_check.isChecked():
        self._on_resolve_terminal()
    else:
        self._on_ultrack_terminal()
```

Wire `run_ultrack_btn` and `ultrack_terminal_btn` to those dispatchers instead of directly to the old handlers. Preserve the old handlers unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py -v`

Expected: PASS

- [ ] **Step 5: Run syntax smoke check**

Run: `python -m py_compile src/cellflow/napari/nucleus_workflow_widget.py`

Expected: no output

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "refactor: unify ultrack and resolve routes in nucleus UI"
```

### Task 5: Expose Retrack and Extend Parameters

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py:625-730, 2083-2214`
- Modify: `src/cellflow/tracking_ultrack/extend.py:27-35` only if a parameter name or helper extraction is needed
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`
- Test: `tests/tracking_ultrack/test_extend.py`

- [ ] **Step 1: Write the failing tests**

Add a widget smoke test:

```python
def test_correction_section_exposes_extend_and_retrack_parameter_sections():
    _app, viewer = _viewer()
    widget = NucleusWorkflowWidget(viewer)

    assert widget.extend_params_section.title == "Extend Parameters"
    assert widget.extend_params_section.is_expanded is False
    assert widget.retrack_params_section.title == "Retrack Parameters"
    assert widget.retrack_params_section.is_expanded is False
    assert widget.retrack_max_dist_spin.value() == 20.0

    viewer.close()
```

If backend coverage is needed for extend distance plumbing, add a focused unit test in `tests/tracking_ultrack/test_extend.py`:

```python
def test_extend_track_respects_custom_d_max(...):
    result = extend_track(..., d_max=5.0)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py -v`

Expected: FAIL because the parameter sections and spins do not exist.

Run: `pytest tests/tracking_ultrack/test_extend.py -v`

Expected: FAIL only if the new backend test was added.

- [ ] **Step 3: Write minimal implementation**

Create nested parameter sections in the correction area:

```python
self.extend_max_dist_spin = QDoubleSpinBox()
self.extend_max_dist_spin.setRange(0.0, 500.0)
self.extend_max_dist_spin.setValue(40.0)

self.retrack_max_dist_spin = QDoubleSpinBox()
self.retrack_max_dist_spin.setRange(0.0, 500.0)
self.retrack_max_dist_spin.setValue(20.0)
```

Wrap them in collapsibles:

```python
self.extend_params_section = CollapsibleSection("Extend Parameters", extend_params_inner, expanded=False)
self.retrack_params_section = CollapsibleSection("Retrack Parameters", retrack_params_inner, expanded=False)
```

Pass the widget values into the existing handlers:

```python
result = extend_track(
    source_id=source_id,
    source_frame=t,
    direction=direction,
    tracked_labels=tracked,
    hypotheses_path=hyp_path,
    d_max=float(self.extend_max_dist_spin.value()),
)
```

```python
stack[t] = retrack_frame_constrained(
    ref,
    tgt,
    locked,
    max_dist_px=float(self.retrack_max_dist_spin.value()),
)
```

Update the “No hypothesis” status string so it reports the configured extend distance rather than hardcoded `40px`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py tests/tracking_ultrack/test_extend.py -v`

Expected: PASS

- [ ] **Step 5: Run targeted syntax and regression checks**

Run: `python -m py_compile src/cellflow/napari/nucleus_workflow_widget.py src/cellflow/napari/correction_widget.py src/cellflow/tracking_ultrack/extend.py`

Expected: no output

Run: `pytest tests/tracking_ultrack/test_extend.py tests/tracking/test_retracker.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py src/cellflow/napari/correction_widget.py src/cellflow/tracking_ultrack/extend.py tests/napari/test_nucleus_tracking_correction_layout.py tests/tracking_ultrack/test_extend.py
git commit -m "feat: expose nucleus correction extend and retrack parameters"
```

### Task 6: Final Verification and Cleanup

**Files:**
- Modify: only if review-driven cleanup is required
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`
- Test: `tests/tracking_ultrack/test_extend.py`
- Test: `tests/tracking/test_retracker.py`

- [ ] **Step 1: Run the focused verification suite**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py tests/tracking_ultrack/test_extend.py tests/tracking/test_retracker.py -v
```

Expected: PASS

- [ ] **Step 2: Run syntax verification**

Run:

```bash
python -m py_compile src/cellflow/napari/nucleus_workflow_widget.py src/cellflow/napari/correction_widget.py src/cellflow/tracking_ultrack/extend.py
```

Expected: no output

- [ ] **Step 3: Manual smoke-check in napari**

Run:

```bash
python -c "import napari; from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget; v = napari.Viewer(show=False); w = NucleusWorkflowWidget(v); print(w.tracking_correction_section.title); v.close()"
```

Expected output:

```text
4. Tracking & Correction
```

- [ ] **Step 4: Commit any final cleanup**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py src/cellflow/napari/correction_widget.py src/cellflow/tracking_ultrack/extend.py tests/napari/test_nucleus_tracking_correction_layout.py tests/tracking_ultrack/test_extend.py
git commit -m "test: finalize nucleus tracking correction UI refactor"
```

## Self-Review

- Spec coverage:
  - Top-level merge covered by Task 2.
  - Nested `Ultrack Tracking` and `Correction` covered by Task 2.
  - Moved persistence/reassign and shortcuts covered by Task 3.
  - Shared ultrack route selector plus local status/progress covered by Task 4.
  - Extend/retrack parameter exposure covered by Task 5.
  - Responsiveness is preserved by reusing the existing `_compact_btn` / `_compact` helpers throughout Tasks 2-5.
- Placeholder scan:
  - No `TODO` or `TBD` placeholders remain.
- Type consistency:
  - Stable attribute names in tests and implementation are `tracking_correction_section`, `ultrack_section`, `correction_section`, `correction_shortcuts_section`, `ultrack_route_check`, `ultrack_status_lbl`, `extend_params_section`, `retrack_params_section`, `extend_max_dist_spin`, and `retrack_max_dist_spin`.

Plan complete and saved to `docs/superpowers/plans/2026-04-29-nucleus-tracking-correction-ui-refactor.md`. Recommended execution approach: subagent-driven, one slice at a time with local review between slices.
