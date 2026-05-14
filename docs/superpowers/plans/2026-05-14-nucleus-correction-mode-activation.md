# Nucleus Correction Mode Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the nucleus workflow Correction section an explicit activate/deactivate mode that loads its own correction layers from disk, hides unrelated layers while active, and cleans up only the layers it owns.

**Architecture:** Keep the implementation inside `NucleusWorkflowWidget` because the behavior coordinates widget state, napari viewer state, file paths, and the embedded `CorrectionWidget`. Add a small internal registry for correction-owned layer names and a captured viewer-state snapshot; all correction actions resolve their working labels layer through one helper so the existing edit/save/extend/retrack commands operate on `[Correction] Tracked: Nucleus`.

**Tech Stack:** Python, Qt/qtpy, napari layers, tifffile, numpy, pytest.

---

## Files

- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
  - Remove the separate `Load Labels` button from the Correction section.
  - Add correction-owned layer constants, NLS z-avg path helper, layer registry, and viewer-state snapshot helpers.
  - Change Correction activation to load fresh `[Correction]` layers from disk.
  - Change Correction deactivation to remove only registered correction layers and restore prior viewer state.
  - Route save, extend, retrack, reassign, and remove-unvalidated actions through the active correction labels layer.
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`
  - Update layout assertions for the simplified active correction controls.
  - Add activation/deactivation tests for owned layer loading, reference image loading, contrast limits, registry cleanup, and viewer-state restoration.
- No new production modules.

## Task 1: Add Failing Layout Tests For The Simplified Correction Section

**Files:**
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [x] **Step 1: Update the stable-section layout test**

In `test_tracking_correction_shell_exposes_stable_section_attributes`, replace the correction button assertions with:

```python
    assert "Activate Correction" in correction_button_texts
    assert "Save tracked" in correction_button_texts
    assert "Load Labels" not in correction_button_texts
    assert "Save Labels" not in correction_button_texts
    assert "Extend selected" in correction_button_texts
    assert "Retrack selected" in correction_button_texts
    assert "Reassign ID" in correction_button_texts
    assert "Remove unvalidated" in correction_button_texts
    assert "Clean Holes / Islands" not in correction_button_texts
    assert "◀ Extend (A)" not in correction_button_texts
    assert "Extend (D) ▶" not in correction_button_texts
    assert "◀ Retrack (Q)" not in correction_button_texts
    assert "Retrack (E) ▶" not in correction_button_texts
```

Also update the Ultrack assertions in the same test to keep:

```python
    assert "Save tracked" not in ultrack_button_texts
    assert "Load Labels" not in ultrack_button_texts
    assert "Reassign ID" not in ultrack_button_texts
```

- [x] **Step 2: Update the button expansion test**

In `test_tracking_correction_action_buttons_expand_horizontally`, replace `tracked_buttons` with:

```python
    tracked_buttons = [
        widget.run_ultrack_btn,
        widget.extend_selected_btn,
        widget.retrack_selected_btn,
        widget.save_tracked_btn,
        widget.reassign_ids_btn,
        widget.remove_unvalidated_btn,
    ]
```

- [x] **Step 3: Update duplicate correction-section tests**

In `test_correction_section_is_top_level`, replace the correction button assertions with the same simplified set from Step 1:

```python
    assert "Save tracked" in correction_button_texts
    assert "Load Labels" not in correction_button_texts
    assert "Extend selected" in correction_button_texts
    assert "Retrack selected" in correction_button_texts
```

In `test_tracking_correction_restores_two_column_button_and_parameter_layouts`, replace old direction-button geometry checks with:

```python
    assert widget.extend_selected_btn.y() == widget.retrack_selected_btn.y()
    assert widget.reassign_ids_btn.y() == widget.remove_unvalidated_btn.y()
```

- [x] **Step 4: Run the focused layout tests and verify they fail**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_tracking_correction_shell_exposes_stable_section_attributes tests/napari/test_nucleus_tracking_correction_layout.py::test_tracking_correction_action_buttons_expand_horizontally tests/napari/test_nucleus_tracking_correction_layout.py::test_correction_section_is_top_level -q
```

Expected: FAIL because `load_tracked_btn`, direction-specific correction buttons, and old labels still exist.

- [x] **Step 5: Commit the failing tests**

```bash
git add tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "test: specify simplified nucleus correction controls"
```

## Task 2: Simplify The Correction Section Controls

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py:675`
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py:840`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [x] **Step 1: Replace old correction buttons in `_build_correction_section`**

Replace the current `load_tracked_btn`, `save_tracked_btn`, direction-specific extend/retrack buttons, and button grid with:

```python
        self.save_tracked_btn = _btn(
            "Save tracked", "Save corrected tracked nucleus labels to disk."
        )
        self.extend_selected_btn = _btn(
            "Extend selected", "Extend the selected track before or after the current frame."
        )
        self.retrack_selected_btn = _btn(
            "Retrack selected", "Retrack labels around the current frame."
        )
        self.reassign_ids_btn = _btn(
            "Reassign ID", "Reassign cell IDs to contiguous range 1-N."
        )
        self.remove_unvalidated_btn = _btn(
            "Remove unvalidated",
            "Remove nucleus label pixels not marked validated for their frame.",
        )
        danger_button(self.remove_unvalidated_btn)

        group_lay.addLayout(_button_grid(
            (self.save_tracked_btn,),
            (self.extend_selected_btn, self.retrack_selected_btn),
            (self.reassign_ids_btn, self.remove_unvalidated_btn),
        ))
```

- [x] **Step 2: Add elevated common parameters**

Immediately after the status and validation labels, add a compact grid for the frequently tuned parameters:

```python
        main_params = block_grid(horizontal_spacing=12)
        self.extend_before_spin = _ispin(0, 50, 0, 1, "Frames before the current frame to extend.")
        self.extend_after_spin = _ispin(0, 50, 1, 1, "Frames after the current frame to extend.")
        self.retrack_radius_spin = _dspin(0, 500, 20.0, 1.0, 1, "Maximum centroid distance for retracking.")
        self.retrack_window_spin = _ispin(1, 100, 1, 1, "Frames before and after the current frame to retrack.")
        self.extend_greedy_overwrite_check = QCheckBox("Greedy overwrite")
        add_block_pair_row(
            main_params, 0,
            "Extend before/after:", compact_spinbox(self.extend_before_spin),
            "", compact_spinbox(self.extend_after_spin),
        )
        add_block_pair_row(
            main_params, 1,
            "Retrack radius/window:", compact_spinbox(self.retrack_radius_spin),
            "", compact_spinbox(self.retrack_window_spin),
        )
        add_block_checkbox_row(main_params, 2, self.extend_greedy_overwrite_check)
        group_lay.addLayout(main_params)
```

- [x] **Step 3: Rename the advanced parameter section**

Keep the existing lower-frequency extend weights, but remove the duplicated `self.extend_greedy_overwrite_check` creation from the advanced section. Rename the section construction to:

```python
        self.advanced_correction_params_section = CollapsibleSection(
            "Advanced Correction Params",
            advanced_inner,
            expanded=False,
            title_role="params",
            title_level=2,
        )
        self.extend_params_section = self.advanced_correction_params_section
        self.retrack_params_section = self.advanced_correction_params_section
        group_lay.addWidget(self.advanced_correction_params_section)
```

Use one `advanced_inner`/`advanced_lay` container containing the existing extend weight controls. Do not keep a separate "Retrack Parameters" collapsible section after `retrack_radius_spin` has moved into the main active section.

- [x] **Step 4: Update signal wiring**

In `_connect_signals`, remove:

```python
        self.load_tracked_btn.clicked.connect(self._on_load_tracked)
        self.extend_back_btn.clicked.connect(self._on_extend_backward)
        self.extend_fwd_btn.clicked.connect(self._on_extend_forward)
        self.retrack_back_btn.clicked.connect(self._on_retrack_backward)
        self.retrack_fwd_btn.clicked.connect(self._on_retrack_forward)
```

Add:

```python
        self.extend_selected_btn.clicked.connect(self._on_extend_selected)
        self.retrack_selected_btn.clicked.connect(self._on_retrack_selected)
```

- [x] **Step 5: Add command handlers**

Add these methods near the existing extend/retrack wrappers so the new buttons execute the existing one-frame correction operations:

```python
    def _on_extend_selected(self) -> None:
        before = int(self.extend_before_spin.value())
        after = int(self.extend_after_spin.value())
        if before <= 0 and after <= 0:
            self._correction_status("Choose at least one frame to extend.")
            return
        for _ in range(before):
            self._on_extend(direction="backward")
        for _ in range(after):
            self._on_extend(direction="forward")

    def _on_retrack_selected(self) -> None:
        window = int(self.retrack_window_spin.value())
        if window <= 0:
            self._correction_status("Retrack window must be at least 1.")
            return
        self._on_retrack_backward()
        self._on_retrack_forward()
```

- [x] **Step 6: Run the focused layout tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_tracking_correction_shell_exposes_stable_section_attributes tests/napari/test_nucleus_tracking_correction_layout.py::test_tracking_correction_action_buttons_expand_horizontally tests/napari/test_nucleus_tracking_correction_layout.py::test_correction_section_is_top_level -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat: simplify nucleus correction controls"
```

## Task 3: Add Failing Tests For Correction Activation Loading Owned Layers

**Files:**
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [x] **Step 1: Add a synchronous activation test**

Add this test near `test_correction_activate_button_expands_activates_and_deactivates_layers`:

```python
def test_correction_activation_loads_owned_layers_from_disk(monkeypatch, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "Position_1"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "0_input").mkdir(parents=True)
    tracked = np.zeros((2, 4, 5), dtype=np.uint32)
    tracked[0, 1:3, 1:3] = 7
    cell = np.arange(20, dtype=np.float32).reshape(4, 5)
    nucleus = np.full((4, 5), 3, dtype=np.float32)
    nls = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", tracked)
    tifffile.imwrite(pos_dir / "0_input" / "cell_zavg.tif", cell)
    tifffile.imwrite(pos_dir / "0_input" / "nucleus_zavg.tif", nucleus)
    tifffile.imwrite(pos_dir / "0_input" / "NLS_zavg.tif", nls)

    stale = viewer.add_labels(np.ones((1, 2, 2), dtype=np.uint32), name="[Correction] stale")
    stale.visible = True
    existing = viewer.add_image(np.ones((4, 5), dtype=np.float32), name="Existing")
    existing.visible = True
    widget.refresh(pos_dir)

    widget.correction_active_btn.setChecked(True)

    assert "[Correction] stale" not in viewer.layers
    assert "[Correction] Tracked: Nucleus" in viewer.layers
    assert "[Correction] Cell z-avg" in viewer.layers
    assert "[Correction] Nucleus z-avg" in viewer.layers
    assert "[Correction] NLS z-avg" in viewer.layers
    assert viewer.layers["Existing"].visible is False
    assert widget.correction_widget._layer is viewer.layers["[Correction] Tracked: Nucleus"]
    assert widget.correction_mode_section.is_expanded is True
    assert set(widget._correction_owned_layers) == {
        "[Correction] Tracked: Nucleus",
        "[Correction] Cell z-avg",
        "[Correction] Nucleus z-avg",
        "[Correction] NLS z-avg",
    }
    np.testing.assert_array_equal(viewer.layers["[Correction] Tracked: Nucleus"].data, tracked)
    assert viewer.layers["[Correction] Cell z-avg"].blending == "additive"
    assert viewer.layers["[Correction] Nucleus z-avg"].blending == "additive"
    assert viewer.layers["[Correction] NLS z-avg"].blending == "additive"
    assert viewer.layers["[Correction] NLS z-avg"].colormap.name == "bop_blue"

    widget.deleteLater()
    viewer.close()
```

- [x] **Step 2: Add a deactivation and restore test**

Add:

```python
def test_correction_deactivation_removes_registered_layers_and_restores_viewer_state(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "Position_1"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    tracked = np.zeros((1, 4, 4), dtype=np.uint32)
    tracked[0, 1:3, 1:3] = 5
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", tracked)

    existing = viewer.add_image(np.ones((4, 4), dtype=np.float32), name="Existing")
    unrelated_prefixed = viewer.add_image(np.ones((4, 4), dtype=np.float32), name="[Correction] User Layer")
    existing.visible = True
    unrelated_prefixed.visible = True
    viewer.layers.selection.active = existing
    widget.refresh(pos_dir)

    widget.correction_active_btn.setChecked(True)
    widget.correction_active_btn.setChecked(False)

    assert "[Correction] Tracked: Nucleus" not in viewer.layers
    assert "[Correction] User Layer" in viewer.layers
    assert viewer.layers["Existing"].visible is True
    assert viewer.layers["[Correction] User Layer"].visible is True
    assert viewer.layers.selection.active is existing
    assert widget._correction_owned_layers == set()
    assert widget.correction_widget._layer is None
    assert widget.correction_mode_section.is_expanded is False

    widget.deleteLater()
    viewer.close()
```

- [x] **Step 3: Run the new tests and verify they fail**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_correction_activation_loads_owned_layers_from_disk tests/napari/test_nucleus_tracking_correction_layout.py::test_correction_deactivation_removes_registered_layers_and_restores_viewer_state -q
```

Expected: FAIL because activation still depends on a selected existing labels layer and has no owned-layer registry.

- [x] **Step 4: Commit the failing tests**

```bash
git add tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "test: specify nucleus correction mode layer ownership"
```

## Task 4: Implement Correction-Owned Layer Loading And Registry

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [x] **Step 1: Add correction constants**

Near the existing layer-name constants, add:

```python
_CORRECTION_TRACKED_LAYER = "[Correction] Tracked: Nucleus"
_CORRECTION_CELL_ZAVG_LAYER = "[Correction] Cell z-avg"
_CORRECTION_NUC_ZAVG_LAYER = "[Correction] Nucleus z-avg"
_CORRECTION_NLS_ZAVG_LAYER = "[Correction] NLS z-avg"
```

- [x] **Step 2: Initialize correction mode state**

In `NucleusWorkflowWidget.__init__`, before `_setup_ui()`, add:

```python
        self._correction_owned_layers: set[str] = set()
        self._correction_view_state: dict | None = None
```

- [x] **Step 3: Add the NLS path helper**

After `_nucleus_zavg_path`, add:

```python
    def _nls_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "NLS_zavg.tif" if self._pos_dir else None
```

- [x] **Step 4: Add contrast and layer helper methods**

Add these methods before `_on_save_tracked`:

```python
    def _correction_tracked_layer(self):
        if _CORRECTION_TRACKED_LAYER in self.viewer.layers:
            return self.viewer.layers[_CORRECTION_TRACKED_LAYER]
        if _TRACKED_LAYER in self.viewer.layers:
            return self.viewer.layers[_TRACKED_LAYER]
        return None

    def _contrast_limits_for_image(self, data: np.ndarray):
        arr = np.asarray(data, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        lo, hi = np.percentile(finite, [0.05, 99.5])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return (float(lo), float(hi))
        data_min = float(np.min(finite))
        data_max = float(np.max(finite))
        if data_max > data_min:
            return (data_min, data_max)
        return None

    def _capture_correction_view_state(self) -> None:
        selected = [layer.name for layer in self.viewer.layers.selection]
        active = self.viewer.layers.selection.active
        self._correction_view_state = {
            "visibility": {layer.name: bool(layer.visible) for layer in self.viewer.layers},
            "active": active.name if active is not None else None,
            "selected": selected,
        }

    def _restore_correction_view_state(self) -> None:
        state = self._correction_view_state or {}
        visibility = state.get("visibility", {})
        for name, visible in visibility.items():
            if name in self.viewer.layers:
                self.viewer.layers[name].visible = bool(visible)
        self.viewer.layers.selection.clear()
        for name in state.get("selected", ()):
            if name in self.viewer.layers:
                self.viewer.layers.selection.add(self.viewer.layers[name])
        active_name = state.get("active")
        if active_name in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[active_name]
        self._correction_view_state = None

    def _remove_correction_owned_layers(self) -> None:
        for name in list(self._correction_owned_layers):
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])
        self._correction_owned_layers.clear()

    def _add_correction_image_layer(self, data: np.ndarray, name: str, colormap: str) -> None:
        arr = np.asarray(data, dtype=np.float32)
        kwargs = {"name": name, "colormap": colormap, "blending": "additive"}
        limits = self._contrast_limits_for_image(arr)
        if limits is not None:
            kwargs["contrast_limits"] = limits
        self.viewer.add_image(arr, **kwargs)
        self._correction_owned_layers.add(name)
```

- [x] **Step 5: Implement synchronous correction layer loading**

Replace `_on_load_tracked` and `_on_load_tracked_done` with a private loader that returns `True` on success:

```python
    def _load_correction_layers_from_disk(self) -> bool:
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            self._correction_status("No tracked labels file found.")
            return False

        self._remove_correction_owned_layers()
        stack = read_full_tracked_stack(tracked_path)
        self.viewer.add_labels(stack, name=_CORRECTION_TRACKED_LAYER)
        self._correction_owned_layers.add(_CORRECTION_TRACKED_LAYER)

        for path, name, cmap in (
            (self._cell_zavg_path(), _CORRECTION_CELL_ZAVG_LAYER, "gray"),
            (self._nucleus_zavg_path(), _CORRECTION_NUC_ZAVG_LAYER, "gray"),
            (self._nls_zavg_path(), _CORRECTION_NLS_ZAVG_LAYER, "bop_blue"),
        ):
            if path is None or not path.exists():
                continue
            self._add_correction_image_layer(
                np.asarray(tifffile.imread(str(path)), dtype=np.float32),
                name,
                cmap,
            )

        self._correction_status(f"Loaded tracked stack {stack.shape} into correction mode.")
        return True
```

Delete the old worker-based `_on_load_tracked_done` method. If any tests still call `_on_load_tracked`, keep this compatibility shim:

```python
    def _on_load_tracked(self) -> None:
        self._load_correction_layers_from_disk()
```

- [x] **Step 6: Rewrite activation and deactivation**

Replace `_on_correction_active_button_toggled` with:

```python
    def _on_correction_active_button_toggled(self, active: bool) -> None:
        if active:
            self._capture_correction_view_state()
            for layer in list(self.viewer.layers):
                layer.visible = False
            if not self._load_correction_layers_from_disk():
                self._restore_correction_view_state()
                old = self.correction_active_btn.blockSignals(True)
                try:
                    self.correction_active_btn.setChecked(False)
                finally:
                    self.correction_active_btn.blockSignals(old)
                self.correction_widget.deactivate()
                self.correction_mode_section.collapse()
                return
            layer = self.viewer.layers[_CORRECTION_TRACKED_LAYER]
            layer.visible = True
            self.viewer.layers.selection.active = layer
            self.correction_widget.activate_layer(layer)
            self.correction_mode_section.expand()
            return

        self.correction_widget.deactivate()
        for sc in getattr(self, "_correction_shortcuts", []):
            sc.setEnabled(False)
        self._remove_correction_owned_layers()
        self._restore_correction_view_state()
        self.correction_mode_section.collapse()
```

- [x] **Step 7: Keep embedded CorrectionWidget activation one-way**

Remove this signal connection from `_connect_signals`:

```python
        self.correction_widget._activate_btn.toggled.connect(
            self.correction_active_btn.setChecked
        )
```

Keep `_on_correction_mode_toggled` connected so shortcuts follow the embedded widget's active state.

- [x] **Step 8: Run the new activation tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_correction_activation_loads_owned_layers_from_disk tests/napari/test_nucleus_tracking_correction_layout.py::test_correction_deactivation_removes_registered_layers_and_restores_viewer_state -q
```

Expected: PASS.

- [x] **Step 9: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat: load owned nucleus correction layers"
```

## Task 5: Route Correction Actions Through The Active Correction Layer

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [x] **Step 1: Replace direct `_TRACKED_LAYER` checks in correction actions**

In `_on_save_tracked`, `_on_reassign_ids`, `_on_extend`, `_on_retrack_forward`, `_on_retrack_backward`, and `_on_remove_unvalidated_labels`, replace checks like:

```python
        if _TRACKED_LAYER not in self.viewer.layers:
            self._correction_status("No tracked layer loaded."); return
        layer = self.viewer.layers[_TRACKED_LAYER]
```

with:

```python
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
```

In `_on_save_tracked`, keep the project-open check first, then use the helper:

```python
        tracked_path = self._tracked_path()
        if tracked_path is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer to save."); return
```

- [x] **Step 2: Replace `_TRACKED_LAYER` updates in callbacks**

In `_on_reassign_ids_done`, replace:

```python
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = remapped
```

with:

```python
        layer = self._correction_tracked_layer()
        if layer is not None:
            layer.data = remapped
```

- [x] **Step 3: Add a focused save test**

Add this test:

```python
def test_correction_save_writes_correction_owned_tracked_layer(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "Position_1"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    original = np.zeros((2, 4, 4), dtype=np.uint32)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", original)
    widget.refresh(pos_dir)
    widget.correction_active_btn.setChecked(True)

    edited = np.zeros((2, 4, 4), dtype=np.uint32)
    edited[1, 1:3, 1:3] = 9
    viewer.layers["[Correction] Tracked: Nucleus"].data = edited
    widget._on_save_tracked()

    np.testing.assert_array_equal(
        tifffile.imread(pos_dir / "2_nucleus" / "tracked_labels.tif"),
        edited,
    )
    assert "Saved 2 frame(s)" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()
```

- [x] **Step 4: Run the save test**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_correction_save_writes_correction_owned_tracked_layer -q
```

Expected: PASS.

- [x] **Step 5: Run previous correction action tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_nucleus_correction_button_removes_unvalidated_label_instances tests/napari/test_nucleus_tracking_correction_layout.py::test_correction_save_writes_correction_owned_tracked_layer -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "fix: route nucleus correction actions through active layer"
```

## Task 6: Tighten Activation Error Handling And Refresh Behavior

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [x] **Step 1: Add missing-file test**

Add:

```python
def test_correction_activation_without_tracked_file_restores_button_and_viewer(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "Position_1"
    pos_dir.mkdir()
    existing = viewer.add_image(np.ones((4, 4), dtype=np.float32), name="Existing")
    existing.visible = True
    widget.refresh(pos_dir)

    widget.correction_active_btn.setChecked(True)

    assert widget.correction_active_btn.isChecked() is False
    assert widget.correction_mode_section.is_expanded is False
    assert viewer.layers["Existing"].visible is True
    assert widget._correction_owned_layers == set()
    assert "No tracked labels file found" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()
```

- [x] **Step 2: Make `refresh(None)` leave correction mode cleanly**

In `refresh`, replace the `pos_dir is None` branch with:

```python
        if pos_dir is None:
            if self.correction_active_btn.isChecked():
                self.correction_active_btn.setChecked(False)
            else:
                self.correction_widget.deactivate()
                self._remove_correction_owned_layers()
            return
```

- [x] **Step 3: Ensure status label visibility**

Confirm `_correction_status` still sets text and visibility:

```python
    def _correction_status(self, msg: str) -> None:
        self.correction_status_lbl.setText(msg)
        self.correction_status_lbl.setVisible(bool(msg))
```

No change is needed if the method already matches.

- [x] **Step 4: Run missing-file test**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_correction_activation_without_tracked_file_restores_button_and_viewer -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "fix: restore viewer when nucleus correction activation fails"
```

## Task 7: Final Regression Verification

**Files:**
- No code edits unless verification exposes a regression.

- [x] **Step 1: Run the focused correction/layout suite**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py -q
```

Expected: PASS. If optional dependencies skip some tests, record the skip count and rerun in the project environment if available.

- [x] **Step 2: Compile touched production module**

Run:

```bash
python -m py_compile src/cellflow/napari/nucleus_workflow_widget.py
```

Expected: no output and exit code 0.

- [x] **Step 3: Inspect the diff**

Run:

```bash
git diff -- src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
```

Expected: diff is limited to the correction-mode behavior and tests described in this plan.

- [x] **Step 4: Final commit if any verification fixes were required**

Only if Step 1 or Step 2 required additional edits:

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "test: verify nucleus correction mode activation"
```

## Self-Review Notes

- Spec coverage: The plan covers the line-270 Correction Mode requirements: single `Activate Correction` entry point, no separate `Load Labels`, fresh disk-loaded `[Correction]` layers, optional z-avg layers, additive blending, percentile contrast limits, `bop_blue` NLS colormap, internal registry cleanup, viewer visibility restoration, active/selected layer restoration, and collapse on deactivation.
- Intentional scope boundary: This plan does not implement the database-browser mode or artifact-status table because those are separate design sections in the same spec.
- Compatibility: `_on_load_tracked` remains as a shim for any stale tests or external calls, but the visible UI no longer exposes it.
