# Cell Segmentation Subwidget Status Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the cell segmentation widget so each stage owns its input file rows, output file rows, status label, progress bar, and viewer-load buttons, while removing the global status/progress area and making parameter controls compact two-column rows.

**Architecture:** Keep all changes in `CellWorkflowWidget` and its napari tests. Reuse `PipelineFilesWidget` for per-stage input/output file rows because it already renders status, shape/dtype, and load buttons when constructed with `viewer=self.viewer`. Use small local helpers in `cell_workflow_widget.py` to avoid duplicating file-widget, progress-bar, and parameter-grid setup across the three subwidgets.

**Tech Stack:** Python, qtpy/Qt widgets, napari viewer layer APIs, tifffile, pytest.

---

## File Structure

- Modify: `src/cellflow/napari/cell_workflow_widget.py`
  - Split the current global input/output/status/progress controls into stage-specific controls.
  - Add a foreground-mask preview action that computes and displays the current-frame mask without writing `foreground_masks.tif`.
  - Replace one-column parameter grids based on `sweep_parameter_grid()` with two-column field grids.
- Modify: `tests/napari/test_cell_workflow_widget.py`
  - Update layout tests to assert per-stage file widgets, status labels, progress bars, and absence of global labels/bars.
  - Add preview behavior tests.
  - Update existing run tests to assert the correct stage status/progress/file widgets are refreshed.

---

### Task 1: Lock Down The New Widget Structure

**Files:**
- Modify: `tests/napari/test_cell_workflow_widget.py`

- [ ] **Step 1: Add structure assertions to the existing default-layout test**

In `test_widget_exposes_flow_following_section_with_default_params`, replace assertions for the global file/status controls with this concrete structure check:

```python
from qtpy.QtWidgets import QLabel, QProgressBar


def _label_texts(widget):
    return [child.text() for child in widget.findChildren(QLabel)]


def _progress_bars(widget):
    return widget.findChildren(QProgressBar)
```

Then add these assertions inside the test, after the section-order assertions:

```python
    assert not hasattr(widget, "input_files")
    assert not hasattr(widget, "ff_files")
    assert not hasattr(widget, "ff_input_lbl")
    assert not hasattr(widget, "ff_status_lbl")
    assert not hasattr(widget, "ff_progress_bar")

    assert hasattr(widget, "filtered_flow_input_files")
    assert hasattr(widget, "filtered_flow_output_files")
    assert hasattr(widget, "filtered_flow_status_lbl")
    assert hasattr(widget, "filtered_flow_progress_bar")

    assert hasattr(widget, "foreground_mask_input_files")
    assert hasattr(widget, "foreground_mask_output_files")
    assert hasattr(widget, "foreground_mask_status_lbl")
    assert hasattr(widget, "foreground_mask_progress_bar")

    assert hasattr(widget, "tracked_labels_input_files")
    assert hasattr(widget, "tracked_labels_output_files")
    assert hasattr(widget, "tracked_labels_status_lbl")
    assert hasattr(widget, "tracked_labels_progress_bar")

    assert widget.filtered_flow_input_files.parent() is widget.filtered_flow_params_widget
    assert widget.filtered_flow_output_files.parent() is widget.filtered_flow_params_widget
    assert widget.filtered_flow_status_lbl.parent() is widget.filtered_flow_params_widget
    assert widget.filtered_flow_progress_bar.parent() is widget.filtered_flow_params_widget

    assert widget.foreground_mask_input_files.parent() is widget.foreground_mask_params_widget
    assert widget.foreground_mask_output_files.parent() is widget.foreground_mask_params_widget
    assert widget.foreground_mask_status_lbl.parent() is widget.foreground_mask_params_widget
    assert widget.foreground_mask_progress_bar.parent() is widget.foreground_mask_params_widget

    assert widget.tracked_labels_input_files.parent() is widget.tracked_labels_params_widget
    assert widget.tracked_labels_output_files.parent() is widget.tracked_labels_params_widget
    assert widget.tracked_labels_status_lbl.parent() is widget.tracked_labels_params_widget
    assert widget.tracked_labels_progress_bar.parent() is widget.tracked_labels_params_widget

    assert widget.filtered_flow_progress_bar.isVisible() is False
    assert widget.foreground_mask_progress_bar.isVisible() is False
    assert widget.tracked_labels_progress_bar.isVisible() is False

    texts = _label_texts(widget)
    assert "min" not in texts
    assert "max" not in texts
    assert "step" not in texts
```

- [ ] **Step 2: Add button assertions for foreground preview**

In the same test, add:

```python
    assert widget.preview_fg_masks_btn.text() == "Preview"
    assert widget.preview_fg_masks_btn.parent() is widget.foreground_mask_params_widget
```

- [ ] **Step 3: Run the red test**

Run:

```bash
pytest tests/napari/test_cell_workflow_widget.py::test_widget_exposes_flow_following_section_with_default_params -q
```

Expected: fail because the widget still exposes global `input_files`, `ff_files`, `ff_input_lbl`, `ff_status_lbl`, and `ff_progress_bar`; per-stage controls and `preview_fg_masks_btn` do not exist yet.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/napari/test_cell_workflow_widget.py
git commit -m "test: specify cell workflow per-stage layout"
```

---

### Task 2: Build Per-Stage File, Status, And Progress Controls

**Files:**
- Modify: `src/cellflow/napari/cell_workflow_widget.py`
- Modify: `tests/napari/test_cell_workflow_widget.py`

- [ ] **Step 1: Add imports and local UI helpers**

In `src/cellflow/napari/cell_workflow_widget.py`, extend the imports from `cellflow.napari.ui_style`:

```python
from cellflow.napari.ui_style import (
    add_block_button_row,
    add_parameter_grid_row,
    block_grid,
    status_label,
)
```

Remove `sweep_parameter_grid` from that import.

Inside `_setup_ui`, before the spinbox helpers, add:

```python
        def _stage_files(group_label: str, entries: list[tuple[str, str]]) -> PipelineFilesWidget:
            return PipelineFilesWidget([(group_label, entries)], viewer=self.viewer)

        def _stage_status() -> QLabel:
            label = QLabel("")
            label.setWordWrap(True)
            label.setVisible(False)
            status_label(label)
            return label

        def _stage_progress() -> QProgressBar:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setVisible(False)
            return bar

        def _param_grid():
            grid = block_grid(horizontal_spacing=12, vertical_spacing=4)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            return grid
```

- [ ] **Step 2: Replace the top-level global input widget**

Delete this current block:

```python
        self.input_files = PipelineFilesWidget([
            ("Inputs", [
                ("1_cellpose/cell_prob_3dt.tif",   "Cell prob 3D+t"),
                ("1_cellpose/cell_dp_3dt.tif",     "Cell dp 3D+t"),
                ("3_cell/foreground_masks.tif",    "Foreground masks"),
                ("2_nucleus/tracked_labels.tif",   "Nucleus tracked labels"),
            ]),
        ], viewer=self.viewer)
        layout.addWidget(self.input_files)
```

- [ ] **Step 3: Update the Filtered Flow section**

At the top of `filtered_flow_params_widget`, before the parameter grid, add:

```python
        self.filtered_flow_input_files = _stage_files("Inputs", [
            ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
            ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
        ])
        filter_lay.addWidget(self.filtered_flow_input_files)
```

Replace:

```python
        filter_grid = sweep_parameter_grid(spin_width=_FF_SPIN_WIDTH)
```

with:

```python
        filter_grid = _param_grid()
```

Replace the four `filter_grid.addWidget(...)` parameter rows with:

```python
        add_parameter_grid_row(filter_grid, 0, 0, "Median t kernel:", self.ff_median_time_spin)
        add_parameter_grid_row(filter_grid, 0, 1, "Median xy kernel:", self.ff_median_space_spin)
        add_parameter_grid_row(filter_grid, 1, 0, "Gaussian t sigma:", self.ff_gauss_time_spin)
        add_parameter_grid_row(filter_grid, 1, 1, "Gaussian xy sigma:", self.ff_gauss_space_spin)
```

After the filtered-flow button row, add:

```python
        self.filtered_flow_status_lbl = _stage_status()
        filter_lay.addWidget(self.filtered_flow_status_lbl)
        self.filtered_flow_progress_bar = _stage_progress()
        filter_lay.addWidget(self.filtered_flow_progress_bar)
        self.filtered_flow_output_files = _stage_files("Outputs", [
            ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
            ("3_cell/filtered_flow_mag.tif", "Filtered flow magnitude"),
        ])
        filter_lay.addWidget(self.filtered_flow_output_files)
```

- [ ] **Step 4: Update the Foreground Mask section**

At the top of `foreground_mask_params_widget`, before the parameter grid, add:

```python
        self.foreground_mask_input_files = _stage_files("Inputs", [
            ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
            ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
        ])
        fg_lay.addWidget(self.foreground_mask_input_files)
```

Replace:

```python
        fg_grid = sweep_parameter_grid(spin_width=_FF_SPIN_WIDTH)
```

with:

```python
        fg_grid = _param_grid()
```

Replace the four `fg_grid.addWidget(...)` parameter rows with:

```python
        add_parameter_grid_row(fg_grid, 0, 0, "Cellprob threshold:", self.fg_cellprob_threshold_spin)
        add_parameter_grid_row(fg_grid, 0, 1, "Flow threshold:", self.fg_flow_threshold_spin)
        add_parameter_grid_row(fg_grid, 1, 0, "Min size:", self.fg_min_size_spin)
        add_parameter_grid_row(fg_grid, 1, 1, "Niter:", self.fg_niter_spin)
```

Replace the foreground button setup:

```python
        self.fg_masks_btn = QPushButton("Create foreground_masks")
        self.fg_masks_btn.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
        self.fg_masks_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        fg_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(fg_btn_row, 0, self.fg_masks_btn)
        fg_lay.addLayout(fg_btn_row)
```

with:

```python
        self.preview_fg_masks_btn = QPushButton("Preview")
        self.fg_masks_btn = QPushButton("Create foreground_masks")
        for button in (self.preview_fg_masks_btn, self.fg_masks_btn):
            button.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        fg_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(fg_btn_row, 0, self.preview_fg_masks_btn, self.fg_masks_btn)
        fg_lay.addLayout(fg_btn_row)
```

After the foreground button row, add:

```python
        self.foreground_mask_status_lbl = _stage_status()
        fg_lay.addWidget(self.foreground_mask_status_lbl)
        self.foreground_mask_progress_bar = _stage_progress()
        fg_lay.addWidget(self.foreground_mask_progress_bar)
        self.foreground_mask_output_files = _stage_files("Outputs", [
            ("3_cell/foreground_masks.tif", "Foreground masks"),
        ])
        fg_lay.addWidget(self.foreground_mask_output_files)
```

- [ ] **Step 5: Update the Tracked Cell Labels section**

At the top of `tracked_labels_params_widget`, before the parameter grid, add:

```python
        self.tracked_labels_input_files = _stage_files("Inputs", [
            ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
            ("3_cell/foreground_masks.tif", "Foreground masks"),
            ("2_nucleus/tracked_labels.tif", "Nucleus tracked labels"),
        ])
        labels_lay.addWidget(self.tracked_labels_input_files)
```

Replace:

```python
        labels_grid = sweep_parameter_grid(spin_width=_FF_SPIN_WIDTH)
```

with:

```python
        labels_grid = _param_grid()
```

Replace the four `labels_grid.addWidget(...)` parameter rows with:

```python
        add_parameter_grid_row(labels_grid, 0, 0, "Flow weight:", self.ff_flow_weight_spin)
        add_parameter_grid_row(labels_grid, 0, 1, "Step scale:", self.ff_step_scale_spin)
        add_parameter_grid_row(labels_grid, 1, 0, "Max iterations:", self.ff_max_iter_spin)
        add_parameter_grid_row(labels_grid, 1, 1, "Capture radius:", self.ff_capture_radius_spin)
```

After the tracked-labels button row, add:

```python
        self.tracked_labels_status_lbl = _stage_status()
        labels_lay.addWidget(self.tracked_labels_status_lbl)
        self.tracked_labels_progress_bar = _stage_progress()
        labels_lay.addWidget(self.tracked_labels_progress_bar)
        self.tracked_labels_output_files = _stage_files("Outputs", [
            ("3_cell/tracked_labels.tif", "Cell labels"),
        ])
        labels_lay.addWidget(self.tracked_labels_output_files)
```

- [ ] **Step 6: Delete global status/progress/output widgets**

Delete the bottom-of-layout global controls:

```python
        self.ff_input_lbl = QLabel("")
        self.ff_input_lbl.setWordWrap(True)
        layout.addWidget(self.ff_input_lbl)

        self.ff_status_lbl = QLabel("")
        self.ff_status_lbl.setWordWrap(True)
        self.ff_status_lbl.setVisible(False)
        layout.addWidget(self.ff_status_lbl)

        self.ff_progress_bar = QProgressBar()
        self.ff_progress_bar.setRange(0, 100)
        self.ff_progress_bar.setValue(0)
        self.ff_progress_bar.setVisible(False)
        layout.addWidget(self.ff_progress_bar)

        self._update_ff_status_labels()

        self.ff_files = PipelineFilesWidget([
            ("Outputs", [
                ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
                ("3_cell/filtered_flow_mag.tif", "Filtered flow magnitude"),
                ("3_cell/tracked_labels.tif",    "Cell labels"),
            ]),
        ], viewer=self.viewer)
        layout.addWidget(self.ff_files)
```

- [ ] **Step 7: Update refresh to refresh every stage file widget**

Replace `refresh()` with:

```python
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._refresh_stage_files(pos_dir)
```

Add:

```python
    def _refresh_stage_files(self, pos_dir: Path | None = None) -> None:
        if pos_dir is None:
            pos_dir = self._pos_dir
        for files_widget in (
            self.filtered_flow_input_files,
            self.filtered_flow_output_files,
            self.foreground_mask_input_files,
            self.foreground_mask_output_files,
            self.tracked_labels_input_files,
            self.tracked_labels_output_files,
        ):
            files_widget.refresh(pos_dir)
```

- [ ] **Step 8: Replace generic status/progress helpers with stage-aware helpers**

Replace `_update_ff_status_labels`, `_set_ff_status`, `_set_ff_buttons_running`, `_on_ff_progress`, and `_on_ff_worker_error` with:

```python
    def _set_stage_status(self, stage: str, msg: str) -> None:
        label = self._stage_status_label(stage)
        label.setText(msg)
        label.setVisible(bool(msg))
        logger.info(msg)

    def _stage_status_label(self, stage: str) -> QLabel:
        return {
            "filtered_flow": self.filtered_flow_status_lbl,
            "foreground_mask": self.foreground_mask_status_lbl,
            "tracked_labels": self.tracked_labels_status_lbl,
        }[stage]

    def _stage_progress_bar(self, stage: str) -> QProgressBar:
        return {
            "filtered_flow": self.filtered_flow_progress_bar,
            "foreground_mask": self.foreground_mask_progress_bar,
            "tracked_labels": self.tracked_labels_progress_bar,
        }[stage]

    def _set_ff_buttons_running(self, running: bool) -> None:
        self.ff_flow_mag_btn.setEnabled(not running)
        self.preview_fg_masks_btn.setEnabled(not running)
        self.fg_masks_btn.setEnabled(not running)
        self.ff_labels_btn.setEnabled(not running)
        self.ff_cancel_btn.setEnabled(running)
        if not running:
            for bar in (
                self.filtered_flow_progress_bar,
                self.foreground_mask_progress_bar,
                self.tracked_labels_progress_bar,
            ):
                bar.setValue(0)
                bar.setVisible(False)

    def _on_stage_progress(self, stage: str, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            bar = self._stage_progress_bar(stage)
            if total > 0:
                bar.setVisible(True)
                bar.setRange(0, total)
                bar.setValue(done)
            self._set_stage_status(stage, msg)
        else:
            self._set_stage_status(stage, str(data))

    def _on_stage_worker_error(self, stage: str, exc: Exception) -> None:
        if self._ff_worker is None:
            return
        self._ff_worker = None
        self._set_ff_buttons_running(False)
        self._set_stage_status(stage, f"Error: {exc}")
        logger.exception("Cell workflow worker error", exc_info=exc)
```

- [ ] **Step 9: Update existing handlers to use stage-aware helpers**

In `_on_create_flow_mag`, replace calls:

```python
self._set_ff_status(...)
self._on_ff_progress
self._on_ff_worker_error
self.ff_files.refresh(pos_dir)
self._update_ff_status_labels()
```

with:

```python
self._set_stage_status("filtered_flow", ...)
lambda data: self._on_stage_progress("filtered_flow", data)
lambda exc: self._on_stage_worker_error("filtered_flow", exc)
self._refresh_stage_files(pos_dir)
```

In `_on_create_foreground_masks`, use stage `"foreground_mask"` and refresh all stage file widgets after writing.

In `_on_create_tracked_labels`, use stage `"tracked_labels"` and refresh all stage file widgets after writing.

In `_on_cancel_flow_following`, replace:

```python
        self._set_ff_status("Flow-following cancelled.")
```

with:

```python
        self._set_stage_status("filtered_flow", "Cancelled.")
        self._set_stage_status("foreground_mask", "Cancelled.")
        self._set_stage_status("tracked_labels", "Cancelled.")
```

- [ ] **Step 10: Update tests that read the removed global input label**

Replace `test_widget_input_status_label_shows_check_for_each_required_file` with a stage-file-widget test:

```python
def test_widget_stage_file_widgets_show_present_and_missing_files(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "3_cell").mkdir()
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif",
                     np.zeros((1, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif",
                     np.zeros((1, 2, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "filtered_dp.tif",
                     np.zeros((1, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif",
                     np.zeros((1, 4, 4), dtype=np.uint8))
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif",
                     np.zeros((1, 4, 4), dtype=np.uint32))

    widget.refresh(pos_dir)

    texts = _label_texts(widget)
    assert texts.count("✓") >= 7
    assert "missing" in texts
    assert widget.filtered_flow_input_files.parent() is widget.filtered_flow_params_widget
    assert widget.foreground_mask_output_files.parent() is widget.foreground_mask_params_widget
    assert widget.tracked_labels_input_files.parent() is widget.tracked_labels_params_widget

    widget.deleteLater()
    app.processEvents()
```

Replace `test_widget_input_status_label_shows_cross_when_files_missing` with:

```python
def test_widget_stage_file_widgets_show_missing_when_files_are_absent(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget.refresh(pos_dir)

    texts = _label_texts(widget)
    assert texts.count("✗") >= 9
    assert texts.count("missing") >= 9

    widget.deleteLater()
    app.processEvents()
```

- [ ] **Step 11: Run the focused layout tests**

Run:

```bash
pytest tests/napari/test_cell_workflow_widget.py::test_widget_exposes_flow_following_section_with_default_params tests/napari/test_cell_workflow_widget.py::test_widget_stage_file_widgets_show_present_and_missing_files tests/napari/test_cell_workflow_widget.py::test_widget_stage_file_widgets_show_missing_when_files_are_absent -q
```

Expected: pass.

- [ ] **Step 12: Commit the UI structure implementation**

```bash
git add src/cellflow/napari/cell_workflow_widget.py tests/napari/test_cell_workflow_widget.py
git commit -m "refactor: give cell workflow stages their own file status"
```

---

### Task 3: Add Foreground Mask Preview

**Files:**
- Modify: `src/cellflow/napari/cell_workflow_widget.py`
- Modify: `tests/napari/test_cell_workflow_widget.py`

- [ ] **Step 1: Add preview constants and signal connection**

Near the existing layer constants in `cell_workflow_widget.py`, add:

```python
_FOREGROUND_MASK_PREVIEW_LAYER = "Preview: Foreground Mask"
```

In `_connect_signals`, add:

```python
        self.preview_fg_masks_btn.clicked.connect(self._on_preview_foreground_masks)
```

- [ ] **Step 2: Add the preview handler**

Add this method before `_on_create_foreground_masks`:

```python
    def _current_time_index(self, max_t: int) -> int:
        step = getattr(getattr(self.viewer, "dims", None), "current_step", ())
        if not step:
            return 0
        return min(max(int(step[0]), 0), max(max_t - 1, 0))

    def _on_preview_foreground_masks(self) -> None:
        if self._pos_dir is None:
            self._set_stage_status("foreground_mask", "No project open.")
            return

        prob_path = self._prob_path()
        filtered_dp_path = self._filtered_dp_out_path()
        for path, name in [
            (prob_path, "cell_prob_3dt.tif"),
            (filtered_dp_path, "filtered_dp.tif (run Filtered Flow first)"),
        ]:
            if path is None or not path.exists():
                self._set_stage_status("foreground_mask", f"Missing: {name}")
                return

        from cellflow.segmentation import compute_cellpose_foreground_masks

        prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
        filtered_dp = np.asarray(tifffile.imread(str(filtered_dp_path)), dtype=np.float32)
        if prob.ndim == 3:
            prob = prob[np.newaxis]
        if filtered_dp.ndim == 3:
            filtered_dp = filtered_dp[np.newaxis]

        t_idx = self._current_time_index(min(prob.shape[0], filtered_dp.shape[0]))
        params_snapshot = self._foreground_params_from_ui()
        self._set_stage_status("foreground_mask", f"Previewing foreground mask at t={t_idx}...")
        self.foreground_mask_progress_bar.setVisible(True)
        self.foreground_mask_progress_bar.setRange(0, 1)
        self.foreground_mask_progress_bar.setValue(0)
        try:
            preview = compute_cellpose_foreground_masks(
                prob[t_idx:t_idx + 1],
                filtered_dp[t_idx:t_idx + 1],
                **params_snapshot,
                progress_cb=None,
            )[0].astype(np.uint8, copy=False)
        except Exception as exc:
            self.foreground_mask_progress_bar.setVisible(False)
            self._set_stage_status("foreground_mask", f"Error: {exc}")
            logger.exception("Foreground mask preview error", exc_info=exc)
            return

        self.foreground_mask_progress_bar.setValue(1)
        self._show_layer(_FOREGROUND_MASK_PREVIEW_LAYER, preview, {}, self.viewer.add_labels)
        self._set_stage_status("foreground_mask", f"Previewed foreground mask at t={t_idx}.")
```

- [ ] **Step 3: Add preview test**

Add this test to `tests/napari/test_cell_workflow_widget.py`:

```python
def test_widget_preview_foreground_masks_uses_current_frame_without_writing(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()

    T, Z, H, W = 3, 2, 5, 5
    prob = np.arange(T * Z * H * W, dtype=np.float32).reshape(T, Z, H, W)
    filtered_dp = np.zeros((T, 2, H, W), dtype=np.float32)
    expected_preview = np.zeros((1, H, W), dtype=np.uint8)
    expected_preview[:, 2:4, 1:4] = 1

    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "3_cell" / "filtered_dp.tif", filtered_dp)

    viewer = _FakeViewer()
    viewer.dims.current_step = (2,)
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    captured: dict[str, object] = {}

    def fake_foreground(prob_arg, dp_arg, **kwargs):
        captured["prob"] = np.asarray(prob_arg).copy()
        captured["dp"] = np.asarray(dp_arg).copy()
        captured.update(kwargs)
        return expected_preview

    with patch("cellflow.segmentation.compute_cellpose_foreground_masks", fake_foreground):
        widget._on_preview_foreground_masks()

    np.testing.assert_array_equal(captured["prob"], prob[2:3])
    np.testing.assert_array_equal(captured["dp"], filtered_dp[2:3])
    assert captured["progress_cb"] is None
    assert not (pos_dir / "3_cell" / "foreground_masks.tif").exists()
    assert "Preview: Foreground Mask" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["Preview: Foreground Mask"].data, expected_preview[0])
    assert "Previewed foreground mask at t=2." in widget.foreground_mask_status_lbl.text()

    widget.deleteLater()
    app.processEvents()
```

- [ ] **Step 4: Run preview test**

Run:

```bash
pytest tests/napari/test_cell_workflow_widget.py::test_widget_preview_foreground_masks_uses_current_frame_without_writing -q
```

Expected: pass.

- [ ] **Step 5: Commit foreground preview**

```bash
git add src/cellflow/napari/cell_workflow_widget.py tests/napari/test_cell_workflow_widget.py
git commit -m "feat: preview cell foreground masks"
```

---

### Task 4: Update Existing Behavior Tests For Stage Status

**Files:**
- Modify: `tests/napari/test_cell_workflow_widget.py`

- [ ] **Step 1: Update flow-magnitude test assertions**

In `test_widget_create_flow_mag_writes_filtered_dp_and_flow_mag`, add:

```python
    assert "Flow magnitude complete." in widget.filtered_flow_status_lbl.text()
    assert widget.filtered_flow_progress_bar.isVisible() is False
```

Remove any assertion that reads `widget.ff_status_lbl`, `widget.ff_input_lbl`, or `widget.ff_files`.

- [ ] **Step 2: Update foreground-mask create test assertions**

In `test_widget_create_foreground_masks_uses_cellprob_and_filtered_dp`, replace:

```python
    assert "foreground" in widget.ff_input_lbl.text()
    assert "Foreground masks complete." in widget.ff_status_lbl.text()
```

with:

```python
    assert "Foreground masks complete." in widget.foreground_mask_status_lbl.text()
    assert widget.foreground_mask_progress_bar.isVisible() is False
```

- [ ] **Step 3: Update tracked-labels test assertions**

In `test_widget_create_tracked_labels_calls_compute_flow_following_movie_and_writes_only_labels`, add:

```python
    assert "Tracked labels complete." in widget.tracked_labels_status_lbl.text()
    assert widget.tracked_labels_progress_bar.isVisible() is False
```

- [ ] **Step 4: Update missing-input test assertions**

Replace:

```python
    assert "Missing" in widget.ff_status_lbl.text()
```

with:

```python
    assert "Missing" in widget.tracked_labels_status_lbl.text()
```

- [ ] **Step 5: Add load-button smoke test for section-local file widgets**

Add this test:

```python
def test_widget_section_file_load_buttons_load_files_into_viewer(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "3_cell").mkdir(parents=True)
    flow_mag = np.ones((2, 4, 4), dtype=np.float32)
    foreground = np.ones((2, 4, 4), dtype=np.uint8)
    labels = np.ones((2, 4, 4), dtype=np.uint32)
    tifffile.imwrite(pos_dir / "3_cell" / "filtered_flow_mag.tif", flow_mag)
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif", foreground)
    tifffile.imwrite(pos_dir / "3_cell" / "tracked_labels.tif", labels)

    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    for files_widget in (
        widget.filtered_flow_output_files,
        widget.foreground_mask_output_files,
        widget.tracked_labels_output_files,
    ):
        for row in files_widget._rows:
            if row._full_path is not None:
                row._on_load_clicked()

    assert "3_cell_filtered_flow_mag" in viewer.layers
    assert "3_cell_foreground_masks" in viewer.layers
    assert "3_cell_tracked_labels" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["3_cell_filtered_flow_mag"].data, flow_mag)
    np.testing.assert_array_equal(viewer.layers["3_cell_foreground_masks"].data, foreground)
    np.testing.assert_array_equal(viewer.layers["3_cell_tracked_labels"].data, labels)

    widget.deleteLater()
    app.processEvents()
```

- [ ] **Step 6: Run the full cell widget test file**

Run:

```bash
pytest tests/napari/test_cell_workflow_widget.py -q
```

Expected: pass.

- [ ] **Step 7: Commit behavior test updates**

```bash
git add tests/napari/test_cell_workflow_widget.py
git commit -m "test: cover cell workflow stage-local status"
```

---

### Task 5: Focused Verification And Diff Review

**Files:**
- Verify: `src/cellflow/napari/cell_workflow_widget.py`
- Verify: `tests/napari/test_cell_workflow_widget.py`

- [ ] **Step 1: Run focused napari widget tests**

Run:

```bash
pytest tests/napari/test_cell_workflow_widget.py tests/napari/test_ui_style.py -q
```

Expected: pass.

- [ ] **Step 2: Run related segmentation tests**

Run:

```bash
pytest tests/segmentation/test_foreground_masks.py tests/segmentation/test_flow_following.py -q
```

Expected: pass.

- [ ] **Step 3: Inspect exact diff**

Run:

```bash
git diff -- src/cellflow/napari/cell_workflow_widget.py tests/napari/test_cell_workflow_widget.py
```

Expected:
- `CellWorkflowWidget` has no top-level global `input_files`, `ff_files`, `ff_input_lbl`, `ff_status_lbl`, or `ff_progress_bar`.
- Each stage has its own input `PipelineFilesWidget`, output `PipelineFilesWidget`, status label, and progress bar.
- `PipelineFilesWidget(..., viewer=self.viewer)` is used for every stage file widget so load buttons remain available.
- `sweep_parameter_grid()` is no longer used in `cell_workflow_widget.py`, so the `min`, `max`, and `step` header labels are gone from this widget.
- Parameter spinboxes are arranged through `add_parameter_grid_row(..., column=0/1, ...)` in two columns.
- Foreground preview displays `Preview: Foreground Mask` and does not write `3_cell/foreground_masks.tif`.

- [ ] **Step 4: Check worktree paths before final handoff**

Run:

```bash
git status --short
```

Expected: only planned files are modified, plus this plan file if it was not committed separately.

