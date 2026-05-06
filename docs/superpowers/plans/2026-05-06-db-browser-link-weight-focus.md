# Database Browser Link Weight Focus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DB-browser connected-node focus so selecting an Ultrack database node shows only its temporal neighbors and modulates neighbor opacity by link weights and/or node probability.

**Architecture:** Keep the feature inside `NucleusWorkflowWidget`, following the existing DB browser refresh path and correction-widget highlight pattern. Extend hierarchy rendering to return stable `NodeDB.id` metadata, use `LinkDB` queries for predecessor/successor focus, and update the existing `Ultrack DB Preview` layer in place.

**Tech Stack:** Python, qtpy, napari layers, SQLAlchemy, Ultrack `NodeDB`/`LinkDB`, NumPy, pytest.

---

## File Structure

- Modify `src/cellflow/napari/nucleus_workflow_widget.py`
  - Add connected-focus state fields.
  - Add DB browser controls and signal wiring.
  - Preserve display-label to `NodeDB.id` metadata during hierarchy rendering.
  - Add click selection callbacks for the DB preview layer.
  - Add focused link-query rendering and cyan contour highlight helpers.
- Modify `tests/napari/test_nucleus_tracking_correction_layout.py`
  - Add focused tests beside the existing Ultrack DB browser tests.
  - Use monkeypatching where a real sqlite Ultrack schema is unnecessary.
  - Add one real/structured metadata test for label-to-node mapping behavior.

## Task 1: Controls And Metadata Contract

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting:

```python
def test_ultrack_db_browser_exposes_connected_focus_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_db_connected_focus_check.text() == "Connected focus"
    assert widget.ultrack_db_edge_alpha_check.text() == "Edge weight transparency"
    assert widget.ultrack_db_prob_alpha_check.text() == "Node prob transparency"
    assert not widget.ultrack_db_connected_focus_check.isEnabled()
    assert not widget.ultrack_db_edge_alpha_check.isEnabled()

    widget.deleteLater()
    viewer.close()
```

Add a test for the normalized preview contract:

```python
def test_ultrack_db_browser_normalizes_preview_metadata():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    labels = np.array([[1, 0]], dtype=np.uint32)
    normalized = widget._normalize_ultrack_db_preview(
        (labels, "status", {1: 0.5}, {1: 101}, {101: 1})
    )

    assert normalized[0] is labels
    assert normalized[1] == "status"
    assert normalized[2] == {1: 0.5}
    assert normalized[3] == {1: 101}
    assert normalized[4] == {101: 1}

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_exposes_connected_focus_controls tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_normalizes_preview_metadata -v -q
```

Expected: FAIL because the new controls and five-field normalization do not exist yet.

- [ ] **Step 3: Implement controls and metadata normalization**

In `__init__`, extend `_ultrack_db_preview_cache` to accept the new tuple shape, and add state:

```python
self._ultrack_db_selected_node_id: int | None = None
self._ultrack_db_selected_frame: int | None = None
self._ultrack_db_label_to_node_id: dict[int, int] = {}
self._ultrack_db_node_id_to_label: dict[int, int] = {}
self._ultrack_db_preview_mouse_callback = None
```

In UI setup, add `ultrack_db_connected_focus_check` and `ultrack_db_edge_alpha_check` next to the existing node-prob checkbox.

Update `_normalize_ultrack_db_preview` to return `(labels, status, prob_dict, label_to_node_id, node_id_to_label)`, preserving backward compatibility for two- and three-field cached values.

- [ ] **Step 4: Run tests and verify green**

Run the same two tests. Expected: PASS.

## Task 2: Selection And Highlight

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting:

```python
def test_ultrack_db_browser_click_selects_node_id_from_display_label(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_mode_combo.setCurrentText("Hierarchy cut")
    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    monkeypatch.setattr(widget, "_query_distinct_heights", lambda *a: (0.5,))
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *a: (
            np.array([[0, 2], [0, 0]], dtype=np.uint32),
            "rendered",
            {2: 0.8},
            {2: 222},
            {222: 2},
        ),
    )

    widget._refresh_ultrack_db_browser()
    widget._select_ultrack_db_preview_label(2, frame=4)

    assert widget._ultrack_db_selected_node_id == 222
    assert widget._ultrack_db_selected_frame == 4
    assert "Selected node 222" in widget.ultrack_db_section_status_lbl.text()

    widget.deleteLater()
    viewer.close()
```

Add a contour helper test using a tiny label image and asserting the highlight layer becomes visible.

- [ ] **Step 2: Run tests and verify red**

Run the new selection/highlight tests. Expected: FAIL because selection and highlight helpers do not exist.

- [ ] **Step 3: Implement selection and highlight**

Add helpers:

- `_install_ultrack_db_preview_selector()`
- `_remove_ultrack_db_preview_selector()`
- `_select_ultrack_db_preview_label(label, frame=None)`
- `_get_ultrack_db_highlight_layer()`
- `_update_ultrack_db_highlight(labels, display_label)`
- `_clear_ultrack_db_highlight()`

Use a napari `Shapes` layer with cyan edge and transparent face, matching `CorrectionWidget._update_highlight`. Use `skimage.measure.find_contours` inside the helper to avoid a top-level import if preferred.

- [ ] **Step 4: Run tests and verify green**

Run the new selection/highlight tests. Expected: PASS.

## Task 3: Focused Link Rendering

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting:

- with connected focus enabled and the viewer frame at selected frame, only the selected node is rendered at alpha `1.0`
- at `t - 1`, only predecessor nodes are rendered
- at `t + 1`, only successor nodes are rendered
- the selected node is excluded from edge-weight modulation

Use monkeypatches for `_query_ultrack_db_connected_nodes` to avoid real DB setup in most tests.

- [ ] **Step 2: Run tests and verify red**

Run the focused rendering tests. Expected: FAIL because connected focus is not implemented.

- [ ] **Step 3: Implement focused rendering**

Add:

- `_query_ultrack_db_connected_nodes(db_path, selected_node_id)` returning predecessor/successor node rows and per-neighbor edge-weight products.
- `_render_ultrack_db_connected_focus(db_path, frame, h_actual, labels, prob_dict, label_to_node_id, node_id_to_label)` to filter labels and compute alpha inputs.
- alpha composition helper for selected node, edge-weight checkbox, node-prob checkbox, and readable alpha clamp.

In `_refresh_ultrack_db_browser`, after normal hierarchy render metadata is available, branch to focused rendering when `ultrack_db_connected_focus_check.isChecked()`.

- [ ] **Step 4: Run tests and verify green**

Run the focused rendering tests. Expected: PASS.

## Task 4: Integration And Regression Coverage

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Add regression tests**

Add tests asserting:

- disabling connected focus preserves existing all-node hierarchy rendering
- edge-weight and node-prob transparency multiply for connected nodes
- activation enables/disables all three DB browser checkboxes
- removing DB browser layers also removes the DB selection highlight layer and mouse callback

- [ ] **Step 2: Run targeted DB browser tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py -k "ultrack_db_browser" -v -q
```

Expected: PASS.

- [ ] **Step 3: Run full napari layout test file**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py -v -q
```

Expected in a fully provisioned environment: PASS. In the current local environment,
this broader file has unrelated failures outside the DB browser slice, including
missing `sqlalchemy` for tests that import real Ultrack modules and existing
foreground/layout assertions that fail when run alone.

- [ ] **Step 4: Commit**

Stage only:

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py docs/superpowers/plans/2026-05-06-db-browser-link-weight-focus.md
git commit -m "Add DB browser connected link focus"
```
