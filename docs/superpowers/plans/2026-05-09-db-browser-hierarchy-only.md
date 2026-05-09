# DB Browser Hierarchy-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Ultrack Database Browser use hierarchy preview as its only mode, keep summary statistics visible without widening the widget, and stop activation from adding contour/foreground image layers.

**Architecture:** Keep the existing DB browser rendering pipeline, but remove the mode selector and the summary-only branch. Summary text remains informational in `ultrack_db_info_lbl`; hierarchy rendering remains the only behavior after activation. Browser-owned layers should be limited to the hierarchy preview labels, annotation labels, and selection highlight.

**Tech Stack:** Python, Qt via `qtpy`, napari layer APIs, pytest.

---

### Task 1: Make DB Browser Summary Text Wrap

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write the failing layout test**

Add this test near the existing Ultrack DB browser layout tests:

```python
def test_ultrack_db_browser_summary_label_wraps_instead_of_widening():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_db_info_lbl.wordWrap() is True
    assert widget.ultrack_db_info_lbl.sizePolicy().horizontalPolicy() != QSizePolicy.Policy.Expanding

    widget.deleteLater()
    viewer.close()
```

If `QSizePolicy` is not already imported in the test file, extend the existing QtWidgets import with:

```python
from qtpy.QtWidgets import QSizePolicy
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_summary_label_wraps_instead_of_widening -q
```

Expected: FAIL because `ultrack_db_info_lbl.wordWrap()` is currently false.

- [ ] **Step 3: Implement wrapping and a non-expanding horizontal policy**

In `NucleusWorkflowWidget.__init__`, change the DB browser summary label setup:

```python
self.ultrack_db_info_lbl = QLabel("—")
self.ultrack_db_info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
self.ultrack_db_info_lbl.setWordWrap(True)
self.ultrack_db_info_lbl.setSizePolicy(
    QSizePolicy.Policy.Preferred,
    QSizePolicy.Policy.Minimum,
)
ultrack_db_browser_lay.addWidget(self.ultrack_db_info_lbl)
```

If `QSizePolicy` is not imported at the top of `nucleus_workflow_widget.py`, add it to the existing `qtpy.QtWidgets` import list.

- [ ] **Step 4: Run the wrapping test**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_summary_label_wraps_instead_of_widening -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "fix: wrap ultrack db browser summary label"
```

### Task 2: Remove Summary-Only Mode From the Browser UI

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Replace the two-mode test with hierarchy-only expectations**

Replace `test_ultrack_db_browser_exposes_two_modes` with:

```python
def test_ultrack_db_browser_exposes_hierarchy_only_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "ultrack_db_mode_combo")
    assert widget.ultrack_db_hierarchy_slider.minimum() == 0
    assert widget.ultrack_db_hierarchy_slider.maximum() == 100
    assert widget.ultrack_db_hierarchy_slider.value() == 50
    assert widget._ultrack_db_slider_row.isVisible() is True

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Update tests that currently force `Hierarchy cut`**

In DB browser tests such as `test_ultrack_db_browser_hierarchy_cut_caches_by_frame_and_slider` and `test_ultrack_db_browser_probability_transparency_renders_rgba_preview`, remove lines like:

```python
widget.ultrack_db_mode_combo.setCurrentText("Hierarchy cut")
```

The tests should rely on hierarchy being the only behavior.

- [ ] **Step 3: Run the changed layout test to verify it fails**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_exposes_hierarchy_only_controls -q
```

Expected: FAIL because `ultrack_db_mode_combo` still exists and `_ultrack_db_slider_row` starts hidden.

- [ ] **Step 4: Remove the mode combo from construction**

In the DB browser constructor block, delete this UI setup:

```python
self.ultrack_db_mode_combo = QComboBox()
self.ultrack_db_mode_combo.addItems([
    "Summary only",
    "Hierarchy cut",
])
add_block_pair_row(
    ultrack_db_grid,
    0,
    "Mode:",
    self.ultrack_db_mode_combo,
    "",
    QWidget(),
)
```

Keep the hierarchy slider and make its row visible by default:

```python
self._ultrack_db_slider_row = QWidget()
_slider_lay = QHBoxLayout(self._ultrack_db_slider_row)
_slider_lay.setContentsMargins(0, 0, 0, 0)
_slider_lay.addWidget(self.ultrack_db_hierarchy_slider)
_slider_lay.addWidget(self.ultrack_db_height_lbl)
ultrack_db_grid.addWidget(self._ultrack_db_slider_row, 0, 0, 1, 4)
self._ultrack_db_slider_row.setVisible(True)
```

- [ ] **Step 5: Remove mode signal wiring and activation toggling**

In `_connect_signals`, delete:

```python
self.ultrack_db_mode_combo.currentTextChanged.connect(self._on_ultrack_db_mode_changed)
```

In `_on_ultrack_db_activate`, delete:

```python
self.ultrack_db_mode_combo.setEnabled(checked)
```

Delete the now-unused `_on_ultrack_db_mode_changed` method:

```python
def _on_ultrack_db_mode_changed(self, mode: str) -> None:
    self._ultrack_db_preview_cache.clear()
    self._ultrack_db_slider_row.setVisible(mode == "Hierarchy cut")
```

- [ ] **Step 6: Run focused DB browser layout tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_exposes_hierarchy_only_controls -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "fix: make ultrack db browser hierarchy only"
```

### Task 3: Always Show Summary Statistics While Rendering Hierarchy

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Replace the summary-only behavior test**

Replace `test_ultrack_db_browser_summary_mode_does_not_render` with:

```python
def test_ultrack_db_browser_shows_summary_while_rendering_hierarchy(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True

    calls = []
    labels = np.zeros((5, 5), dtype=np.uint32)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda path, frame: "summary stats")
    monkeypatch.setattr(widget, "_query_distinct_heights", lambda path, mtime_ns: (0.5,))
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda path, frame, height: calls.append((path, frame, height))
        or (labels, "rendered hierarchy cut"),
    )

    widget._refresh_ultrack_db_browser()

    assert calls == [(db_path, 0, 0.5)]
    assert widget.ultrack_db_info_lbl.text() == "summary stats"
    assert widget.ultrack_db_section_status_lbl.text() == "rendered hierarchy cut"
    assert "Ultrack DB Preview" in viewer.layers

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Run the replacement test to verify it fails before implementation**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_shows_summary_while_rendering_hierarchy -q
```

Expected: this should fail before Task 2 is implemented, or fail if the old summary-only branch still exists.

- [ ] **Step 3: Remove the summary-only branch from refresh**

In `_refresh_ultrack_db_browser`, keep:

```python
self.ultrack_db_info_lbl.setText(self._ultrack_db_summary_text(db_path, frame))
```

Delete:

```python
mode = self.ultrack_db_mode_combo.currentText()
if mode == "Summary only":
    self._set_ultrack_db_status("Summary refreshed.")
    return
```

The next executed block should always be:

```python
mtime_ns = db_path.stat().st_mtime_ns
states = self._configure_ultrack_db_hierarchy_slider(db_path, mtime_ns, frame)
```

- [ ] **Step 4: Run hierarchy rendering tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_shows_summary_while_rendering_hierarchy tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_hierarchy_cut_caches_by_frame_and_slider -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "fix: render hierarchy with db browser summary"
```

### Task 4: Stop Loading Contour and Foreground Image Layers

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write a failing test for browser-owned layers**

Add this test near the DB browser tests:

```python
def test_ultrack_db_browser_does_not_add_contour_or_foreground_layers(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    db_path = pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    tifffile.imwrite(pos_dir / "2_nucleus" / "contour_maps.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_masks.tif", np.zeros((1, 4, 4), dtype=np.uint8))
    widget._pos_dir = pos_dir
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True

    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda path, frame: "summary")
    monkeypatch.setattr(widget, "_query_distinct_heights", lambda path, mtime_ns: (0.5,))
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *args: (np.zeros((4, 4), dtype=np.uint32), "rendered hierarchy cut"),
    )

    widget._refresh_ultrack_db_browser()

    assert "Ultrack DB Preview" in viewer.layers
    assert "Ultrack DB Annotation" not in viewer.layers
    assert "Contour Maps (DB)" not in viewer.layers
    assert "Foreground Masks (DB)" not in viewer.layers

    widget.deleteLater()
    viewer.close()
```

If the project constants use different display names, inspect `_CONTOUR_MAPS_DB_LAYER` and `_FOREGROUND_MASKS_DB_LAYER` and use those exact string values in the assertions.

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_does_not_add_contour_or_foreground_layers -q
```

Expected: FAIL because `_refresh_ultrack_db_browser()` currently calls `_ensure_ultrack_db_browser_layers_loaded()`.

- [ ] **Step 3: Remove automatic contour/foreground loading**

In `_refresh_ultrack_db_browser`, delete:

```python
self._ensure_ultrack_db_browser_layers_loaded()
```

Delete `_ensure_ultrack_db_browser_layers_loaded` if no references remain:

```python
def _ensure_ultrack_db_browser_layers_loaded(self) -> None:
    ...
```

- [ ] **Step 4: Keep layer cleanup focused on DB browser label/selection layers**

In `_remove_ultrack_db_browser_layers`, remove contour and foreground constants from the tuple:

```python
for name in (
    _ULTRACK_DB_PREVIEW_LAYER,
    _ULTRACK_DB_ANNOTATION_LAYER,
):
    if name in self.viewer.layers:
        self.viewer.layers.remove(name)
```

Keep selection cleanup:

```python
if _ULTRACK_DB_SELECTION_LAYER in self.viewer.layers:
    self.viewer.layers.remove(_ULTRACK_DB_SELECTION_LAYER)
```

- [ ] **Step 5: Run the layer test**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_does_not_add_contour_or_foreground_layers -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "fix: limit db browser layers to labels"
```

### Task 5: Final Verification

**Files:**
- Verify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Verify: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Run all nucleus workflow layout tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py -q
```

Expected: PASS.

- [ ] **Step 2: Run UI style tests**

Run:

```bash
pytest tests/napari/test_ui_style.py -q
```

Expected: PASS.

- [ ] **Step 3: Inspect changed files**

Run:

```bash
git diff -- src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
```

Expected: only DB browser hierarchy-only UI, wrapped summary label, layer-loading removal, and matching tests changed.

- [ ] **Step 4: Commit any final fixes**

If verification required fixes, commit them:

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "test: verify db browser hierarchy behavior"
```

Skip this commit if no final fixes were needed after Task 4.

### Self-Review

- Spec coverage: The plan covers line wrapping for the DB browser summary label, hierarchy-only mode, always-visible summary statistics, and no contour/foreground layers from DB browser activation.
- Placeholder scan: No TBD/TODO placeholders remain. The only conditional note is to use the exact contour/foreground layer names if constants differ from the displayed strings.
- Type consistency: The plan keeps existing widget attribute names for the hierarchy slider, status label, preview layer, annotation layer, and selection layer. It removes `ultrack_db_mode_combo` consistently from construction, signal wiring, tests, and refresh behavior.
