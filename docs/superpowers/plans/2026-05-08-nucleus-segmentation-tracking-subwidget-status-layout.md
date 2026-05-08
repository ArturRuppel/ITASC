# Nucleus Segmentation And Tracking Subwidget Status Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `NucleusWorkflowWidget` so the contour-map, Ultrack database-generation, and Ultrack tracking stages each own their input file rows, output file rows, status label, progress bar, and viewer-load buttons, while removing the remaining global input file widget and making the contour/filter parameter controls compact two-column rows.

**Architecture:** Keep changes in `NucleusWorkflowWidget` and its existing napari layout/behavior tests. Reuse `PipelineFilesWidget(..., viewer=self.viewer)` for stage-local file rows so present/missing state, shape/dtype metadata, and load buttons come from the shared file-row implementation. Add small local helpers in `_setup_ui` for stage file widgets, status labels, progress bars, and two-column parameter grids, matching the cell workflow refactor.

**Tech Stack:** Python, qtpy/Qt widgets, napari viewer layer APIs, tifffile, pytest.

---

## File Structure

- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
  - Remove top-level `self.input_files`.
  - Add per-stage `PipelineFilesWidget`s:
    - `contour_input_files`, `contour_output_files`
    - `db_gen_input_files`, `db_gen_output_files`
    - `ultrack_input_files`, `ultrack_output_files`
  - Keep existing section-local status labels/progress bars, but create them via shared local helpers and style them consistently.
  - Refresh every stage file widget from one helper.
  - Convert contour map and contour filter controls from `sweep_parameter_grid()` to two-column `add_parameter_grid_row(...)` rows.
  - Remove `_update_contour_status_labels()` once no call sites remain.
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`
  - Replace source-text checks for the old global `input_files`.
  - Extend canonical layout tests to assert stage-local input/output files.
  - Add file-widget refresh and load-button smoke tests.
  - Update contour/DB/tracking behavior tests to assert the correct stage file widgets are refreshed.
- Optional verify only: `tests/napari/test_nucleus_sweep_streaming.py`
  - Run because it checks source structure around nucleus sweep behavior.

---

### Task 1: Lock Down The New Nucleus Stage Layout

**Files:**
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Add `QProgressBar` import and local helpers**

Change the Qt import near the top from:

```python
from qtpy.QtWidgets import QApplication, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget
```

to:

```python
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
```

Add these helpers after `_install_sync_thread_worker`:

```python
def _label_texts(widget):
    return [child.text() for child in widget.findChildren(QLabel)]


def _progress_bars(widget):
    return widget.findChildren(QProgressBar)
```

- [ ] **Step 2: Replace the obsolete source-text input test**

Replace `test_cell_workflow_required_inputs_exclude_optional_flow_vectors` with:

```python
def test_nucleus_workflow_uses_stage_local_file_widgets():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "input_files")

    assert hasattr(widget, "contour_input_files")
    assert hasattr(widget, "contour_output_files")
    assert hasattr(widget, "contour_status_lbl")
    assert hasattr(widget, "build_progress_bar")

    assert hasattr(widget, "db_gen_input_files")
    assert hasattr(widget, "db_gen_output_files")
    assert hasattr(widget, "db_gen_status_lbl")
    assert hasattr(widget, "db_gen_progress_bar")

    assert hasattr(widget, "ultrack_input_files")
    assert hasattr(widget, "ultrack_output_files")
    assert hasattr(widget, "ultrack_status_lbl")
    assert hasattr(widget, "ultrack_progress_bar")

    assert widget.contour_input_files in widget.contour_section.findChildren(type(widget.contour_input_files))
    assert widget.contour_output_files in widget.contour_section.findChildren(type(widget.contour_output_files))
    assert widget.db_gen_input_files in widget.db_gen_section.findChildren(type(widget.db_gen_input_files))
    assert widget.db_gen_output_files in widget.db_gen_section.findChildren(type(widget.db_gen_output_files))
    assert widget.ultrack_input_files in widget.ultrack_section.findChildren(type(widget.ultrack_input_files))
    assert widget.ultrack_output_files in widget.ultrack_section.findChildren(type(widget.ultrack_output_files))

    assert widget.build_progress_bar.isVisible() is False
    assert widget.db_gen_progress_bar.isVisible() is False
    assert widget.ultrack_progress_bar.isVisible() is False

    texts = _label_texts(widget.contour_section)
    assert "min" not in texts
    assert "max" not in texts
    assert "step" not in texts

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 3: Update canonical element assertions**

In `test_canonical_sections_expose_required_elements`, add these assertions in the existing section blocks:

```python
    # Section 1: Contour Maps
    assert hasattr(widget, "contour_input_files")
    assert hasattr(widget, "contour_output_files")
```

```python
    # Section 3: Ultrack Database Generation
    assert hasattr(widget, "db_gen_input_files")
    assert hasattr(widget, "db_gen_output_files")
```

```python
    # Section 5: Ultrack Tracking
    assert hasattr(widget, "ultrack_input_files")
    assert hasattr(widget, "ultrack_output_files")
```

- [ ] **Step 4: Update contour output test name and references**

Rename `test_contour_maps_section_exposes_foreground_threshold_and_outputs` to:

```python
def test_contour_maps_section_exposes_stage_files_and_foreground_threshold():
```

Inside it, replace:

```python
    output_text = " ".join(
        label.text()
        for label in widget.contour_files.findChildren(QLabel)
    )
```

with:

```python
    input_text = " ".join(
        label.text()
        for label in widget.contour_input_files.findChildren(QLabel)
    )
    output_text = " ".join(
        label.text()
        for label in widget.contour_output_files.findChildren(QLabel)
    )
```

Then assert both inputs and outputs:

```python
    assert "1_cellpose/nucleus_prob_3dt.tif" in input_text
    assert "1_cellpose/nucleus_dp_3dt.tif" in input_text
    assert "2_nucleus/foreground_masks.tif" in output_text
    assert "2_nucleus/foreground_scores.tif" in output_text
    assert "2_nucleus/contour_maps.tif" in output_text
```

- [ ] **Step 5: Add DB generation and tracking file-row assertions**

Add this test near the contour section layout tests:

```python
def test_nucleus_db_gen_and_ultrack_sections_expose_stage_file_rows():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_input_text = " ".join(
        label.text()
        for label in widget.db_gen_input_files.findChildren(QLabel)
    )
    db_output_text = " ".join(
        label.text()
        for label in widget.db_gen_output_files.findChildren(QLabel)
    )
    ultrack_input_text = " ".join(
        label.text()
        for label in widget.ultrack_input_files.findChildren(QLabel)
    )
    ultrack_output_text = " ".join(
        label.text()
        for label in widget.ultrack_output_files.findChildren(QLabel)
    )

    assert "2_nucleus/contour_maps.tif" in db_input_text
    assert "2_nucleus/foreground_masks.tif" in db_input_text
    assert "1_cellpose/nucleus_prob_zavg.tif" in db_input_text
    assert "2_nucleus/ultrack_workdir/data.db" in db_output_text
    assert "2_nucleus/ultrack_workdir/data.db" in ultrack_input_text
    assert "2_nucleus/tracked_labels.tif" in ultrack_output_text

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 6: Run the red layout tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_nucleus_workflow_uses_stage_local_file_widgets tests/napari/test_nucleus_tracking_correction_layout.py::test_canonical_sections_expose_required_elements tests/napari/test_nucleus_tracking_correction_layout.py::test_contour_maps_section_exposes_stage_files_and_foreground_threshold tests/napari/test_nucleus_tracking_correction_layout.py::test_nucleus_db_gen_and_ultrack_sections_expose_stage_file_rows -q
```

Expected: fail because `input_files` still exists and `contour_input_files`, `db_gen_input_files`, `db_gen_output_files`, `ultrack_input_files`, and `ultrack_output_files` do not exist yet.

- [ ] **Step 7: Commit the failing layout tests**

```bash
git add tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "test: specify nucleus stage-local file layout"
```

---

### Task 2: Build Stage-Local File Widgets And Refreshing

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Extend UI-style imports**

In `src/cellflow/napari/nucleus_workflow_widget.py`, change the `cellflow.napari.ui_style` import to include `add_parameter_grid_row` and `status_label`:

```python
from cellflow.napari.ui_style import (
    add_block_button_row,
    add_block_checkbox_row,
    add_block_pair_row,
    add_parameter_grid_row,
    add_sweep_parameter_row,
    block_grid,
    compact_spinbox,
    danger_button,
    muted_label,
    status_label,
    sweep_parameter_grid,
)
```

- [ ] **Step 2: Add local helper functions in `_setup_ui`**

Inside `_setup_ui`, immediately after `SPIN_MAX_W = 70`, add:

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

- [ ] **Step 3: Delete the top-level global input file widget**

Remove this block:

```python
        # ── Inputs ────────────────────────────────────────────────────────
        self.input_files = PipelineFilesWidget([
            ("Inputs", [
                ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
                ("1_cellpose/nucleus_dp_3dt.tif",  "Nucleus dp 3D+t"),
            ]),
        ])
        layout.addWidget(self.input_files)
```

- [ ] **Step 4: Add Contour Maps input files**

After `cp_params_lay.setAlignment(Qt.AlignmentFlag.AlignTop)`, add:

```python
        self.contour_input_files = _stage_files("Inputs", [
            ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
            ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ])
        cp_params_lay.addWidget(self.contour_input_files)
```

- [ ] **Step 5: Keep contour output files section-local and viewer-enabled**

Replace:

```python
        self.contour_files = PipelineFilesWidget([
            ("Outputs", [
                ("2_nucleus/contour_maps.tif", "Contour maps"),
                ("2_nucleus/foreground_scores.tif", "Foreground scores"),
                ("2_nucleus/foreground_masks.tif", "Foreground masks"),
            ]),
        ])
        cp_params_lay.addWidget(self.build_progress_bar)
        cp_params_lay.addWidget(self.contour_files)
```

with:

```python
        self.contour_output_files = _stage_files("Outputs", [
            ("2_nucleus/contour_maps.tif", "Contour maps"),
            ("2_nucleus/foreground_scores.tif", "Foreground scores"),
            ("2_nucleus/foreground_masks.tif", "Foreground masks"),
        ])
        cp_params_lay.addWidget(self.build_progress_bar)
        cp_params_lay.addWidget(self.contour_output_files)
```

- [ ] **Step 6: Add DB Generation input/output files**

At the top of the DB generation section, immediately after `db_gen_lay.setAlignment(Qt.AlignmentFlag.AlignTop)`, add:

```python
        self.db_gen_input_files = _stage_files("Inputs", [
            ("2_nucleus/contour_maps.tif", "Contour maps"),
            ("2_nucleus/foreground_masks.tif", "Foreground masks"),
            ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
        ])
        db_gen_lay.addWidget(self.db_gen_input_files)
```

After `db_gen_lay.addWidget(self.db_gen_progress_bar)`, add:

```python
        self.db_gen_output_files = _stage_files("Outputs", [
            ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
        ])
        db_gen_lay.addWidget(self.db_gen_output_files)
```

- [ ] **Step 7: Add Ultrack Tracking input/output files**

At the top of the Ultrack Tracking section, immediately after `ultrack_lay.setAlignment(Qt.AlignmentFlag.AlignTop)`, add:

```python
        self.ultrack_input_files = _stage_files("Inputs", [
            ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
        ])
        ultrack_lay.addWidget(self.ultrack_input_files)
```

Replace:

```python
        self.tracking_files = PipelineFilesWidget([
            ("Outputs", [
                ("2_nucleus/tracked_labels.tif", "Tracked labels"),
            ]),
        ])
        ultrack_lay.addWidget(self.tracking_files)
```

with:

```python
        self.ultrack_output_files = _stage_files("Outputs", [
            ("2_nucleus/tracked_labels.tif", "Tracked labels"),
        ])
        ultrack_lay.addWidget(self.ultrack_output_files)
```

- [ ] **Step 8: Replace `refresh()` file widget calls**

Replace:

```python
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self.input_files.refresh(pos_dir)
        self.contour_files.refresh(pos_dir)
        self.tracking_files.refresh(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()
            return
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
```

with:

```python
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._refresh_stage_files(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()
            return
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _refresh_stage_files(self, pos_dir: Path | None = None) -> None:
        if pos_dir is None:
            pos_dir = self._pos_dir
        for files_widget in (
            self.contour_input_files,
            self.contour_output_files,
            self.db_gen_input_files,
            self.db_gen_output_files,
            self.ultrack_input_files,
            self.ultrack_output_files,
        ):
            files_widget.refresh(pos_dir)
```

- [ ] **Step 9: Update all file-widget refresh call sites**

Replace:

```python
        self.contour_files.refresh(pos_dir)
        self._update_contour_status_labels()
```

in `_on_build_done` with:

```python
        self._refresh_stage_files(pos_dir)
```

Replace:

```python
        self._update_contour_status_labels()
```

in `_on_cancel_build` with nothing. The method should end:

```python
        self._set_build_buttons_running(False)
        self._set_contour_status("Build cancelled.")
```

Replace:

```python
            self.contour_files.refresh(pos_dir)
```

in `_on_run_contour_filter`’s `_on_filter_done` with:

```python
            self._refresh_stage_files(pos_dir)
```

In `_on_db_gen_done`, add a file refresh before refreshing the browser:

```python
        self._refresh_stage_files(pos_dir)
```

In `_on_run_ultrack_done`, after updating/adding the tracked layer and before setting final status, add:

```python
        self._refresh_stage_files()
```

- [ ] **Step 10: Remove `_update_contour_status_labels`**

Delete:

```python
    def _update_contour_status_labels(self) -> None:
        """(placeholder — file status now handled by PipelineFilesWidget rows)"""
        pass
```

Also remove the call to `_update_contour_status_labels()` at the end of `_setup_ui`.

- [ ] **Step 11: Update tests for renamed file widgets**

In `tests/napari/test_nucleus_tracking_correction_layout.py`, replace every remaining `widget.contour_files` reference with `widget.contour_output_files`.

Replace every remaining `widget.tracking_files` reference with `widget.ultrack_output_files`.

- [ ] **Step 12: Add stage file refresh test**

Add this test near the layout tests:

```python
def test_nucleus_stage_file_widgets_show_present_and_missing_files(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus" / "ultrack_workdir").mkdir(parents=True)
    import tifffile

    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif", np.zeros((1, 1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif", np.zeros((1, 1, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "2_nucleus" / "contour_maps.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_masks.tif", np.zeros((1, 4, 4), dtype=np.uint8))
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", np.zeros((1, 4, 4), dtype=np.uint32))
    (pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db").write_bytes(b"sqlite placeholder")

    widget.refresh(pos_dir)

    texts = _label_texts(widget)
    assert texts.count("✓") >= 8
    assert "missing" in texts
    assert widget.contour_input_files in widget.contour_section.findChildren(type(widget.contour_input_files))
    assert widget.db_gen_output_files in widget.db_gen_section.findChildren(type(widget.db_gen_output_files))
    assert widget.ultrack_output_files in widget.ultrack_section.findChildren(type(widget.ultrack_output_files))

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 13: Run focused layout tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_nucleus_workflow_uses_stage_local_file_widgets tests/napari/test_nucleus_tracking_correction_layout.py::test_canonical_sections_expose_required_elements tests/napari/test_nucleus_tracking_correction_layout.py::test_contour_maps_section_exposes_stage_files_and_foreground_threshold tests/napari/test_nucleus_tracking_correction_layout.py::test_nucleus_db_gen_and_ultrack_sections_expose_stage_file_rows tests/napari/test_nucleus_tracking_correction_layout.py::test_nucleus_stage_file_widgets_show_present_and_missing_files -q
```

Expected: pass.

- [ ] **Step 14: Commit stage-local file widgets**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "refactor: give nucleus stages their own file widgets"
```

---

### Task 3: Compact Contour Parameter Grids

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Update the contour layout test expectations**

In `test_contour_maps_parameters_expand_and_scroll_when_narrow`, replace this assertion:

```python
    assert params_scroll.horizontalScrollBar().maximum() > 0
```

with:

```python
    texts = _label_texts(widget.contour_section)
    assert "min" not in texts
    assert "max" not in texts
    assert "step" not in texts
    assert params_scroll.horizontalScrollBar().maximum() >= 0
```

Keep the existing spin-width and button-width assertions so compactness remains covered.

- [ ] **Step 2: Run the red contour layout test**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_contour_maps_parameters_expand_and_scroll_when_narrow -q
```

Expected: fail because `sweep_parameter_grid()` still creates `min`, `max`, and `step` labels.

- [ ] **Step 3: Convert contour sweep controls to two-column rows**

Replace:

```python
        contour_sweep_grid = sweep_parameter_grid(spin_width=_CONTOUR_SWEEP_WIDTH)
```

with:

```python
        contour_sweep_grid = _param_grid()
```

Replace the manual `contour_sweep_grid.addWidget(...)` block for Cellprob, Gamma, Foreground Threshold, and Save label images with:

```python
        add_parameter_grid_row(contour_sweep_grid, 0, 0, "Cellprob min:", self.cp_min_spin)
        add_parameter_grid_row(contour_sweep_grid, 0, 1, "Cellprob max:", self.cp_max_spin)
        add_parameter_grid_row(contour_sweep_grid, 1, 0, "Cellprob step:", self.cp_step_spin)
        add_parameter_grid_row(contour_sweep_grid, 1, 1, "FG threshold:", self.contour_fg_threshold_spin)
        add_parameter_grid_row(contour_sweep_grid, 2, 0, "Gamma min:", self.cp_gamma_min_spin)
        add_parameter_grid_row(contour_sweep_grid, 2, 1, "Gamma max:", self.cp_gamma_max_spin)
        add_parameter_grid_row(contour_sweep_grid, 3, 0, "Gamma step:", self.cp_gamma_step_spin)
        contour_sweep_grid.addWidget(
            self.save_source_check,
            3,
            2,
            1,
            2,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
```

Keep all existing spinbox setup, tooltips, ranges, default values, and `save_source_check` setup.

- [ ] **Step 4: Convert contour filter controls to two-column rows**

Replace:

```python
        contour_filter_grid = sweep_parameter_grid(spin_width=_CONTOUR_SWEEP_WIDTH)
```

with:

```python
        contour_filter_grid = _param_grid()
```

Replace the four manual filter rows with:

```python
        add_parameter_grid_row(contour_filter_grid, 0, 0, "Median t kernel:", self.contour_filter_median_time_spin)
        add_parameter_grid_row(contour_filter_grid, 0, 1, "Median xy kernel:", self.contour_filter_median_space_spin)
        add_parameter_grid_row(contour_filter_grid, 1, 0, "Gaussian t sigma:", self.contour_filter_gauss_time_spin)
        add_parameter_grid_row(contour_filter_grid, 1, 1, "Gaussian xy sigma:", self.contour_filter_gauss_space_spin)
```

- [ ] **Step 5: Remove unused sweep import only if no longer referenced**

Run:

```bash
rg -n "sweep_parameter_grid|add_sweep_parameter_row" src/cellflow/napari/nucleus_workflow_widget.py
```

If the command only shows import lines, remove `sweep_parameter_grid` and `add_sweep_parameter_row` from the import block.

- [ ] **Step 6: Run contour layout and state tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_contour_maps_parameters_expand_and_scroll_when_narrow tests/napari/test_nucleus_tracking_correction_layout.py::test_contour_foreground_threshold_persists_without_old_foreground_state tests/napari/test_nucleus_tracking_correction_layout.py::test_contour_filter_controls_persist_through_state -q
```

Expected: pass.

- [ ] **Step 7: Commit compact contour grids**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "refactor: compact nucleus contour parameter grids"
```

---

### Task 4: Verify Stage Outputs Refresh And Load Into Viewer

**Files:**
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py` only if tests reveal a missing refresh

- [ ] **Step 1: Update contour build behavior test**

In `test_contour_maps_build_writes_contour_scores_and_thresholded_masks`, after the existing file assertions, add:

```python
    output_texts = _label_texts(widget.contour_output_files)
    assert "✓" in output_texts
    assert "Contour maps and foreground masks built." in widget.contour_status_lbl.text()
    assert widget.build_progress_bar.isVisible() is False
```

- [ ] **Step 2: Update contour filter run behavior test**

In `test_contour_filter_run_overwrites_contour_maps`, after the viewer layer assertions, add:

```python
    assert "Filtered contour maps written to contour_maps.tif." in widget.contour_status_lbl.text()
    assert "✓" in _label_texts(widget.contour_output_files)
```

- [ ] **Step 3: Update DB generation behavior test**

In `test_db_gen_section_calls_ultrack_segment_on_run`, after the status assertion, add:

```python
    assert widget.db_gen_progress_bar.isVisible() is False
    assert "✓" in _label_texts(widget.db_gen_output_files)
```

If the fake DB generation path does not create `data.db`, add this monkeypatch before `widget._on_run_db_generation()`:

```python
    def fake_build_database(**kwargs):
        data_db = kwargs["working_dir"] / "data.db"
        data_db.parent.mkdir(parents=True, exist_ok=True)
        data_db.write_bytes(b"sqlite placeholder")
        return {"database": str(data_db)}

    monkeypatch.setattr(module, "build_ultrack_database", fake_build_database)
```

- [ ] **Step 4: Update Ultrack tracking behavior tests**

Add this new test near `test_ultrack_tracking_solve_fails_clearly_if_db_missing`:

```python
def test_ultrack_tracking_refreshes_stage_output_files(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus" / "ultrack_workdir").mkdir(parents=True)
    (pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db").write_bytes(b"sqlite placeholder")
    widget._pos_dir = pos_dir

    labels = np.ones((2, 4, 4), dtype=np.uint32)

    def fake_export(_working_dir, _cfg, tracked_path, **_kwargs):
        tracked_path.parent.mkdir(parents=True, exist_ok=True)
        import tifffile
        tifffile.imwrite(tracked_path, labels)
        return labels

    monkeypatch.setattr(module, "run_solve", lambda *a, **kw: iter([(1, 1, "solved")]))
    monkeypatch.setattr(module, "export_tracked_labels", fake_export)

    widget._on_run_ultrack()

    assert "Tracked: Nucleus" in viewer.layers
    assert (pos_dir / "2_nucleus" / "tracked_labels.tif").exists()
    assert "✓" in _label_texts(widget.ultrack_output_files)
    assert widget.ultrack_progress_bar.isVisible() is False

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 5: Add file load-button smoke test**

Add this test near the stage file layout tests:

```python
def test_nucleus_stage_file_load_buttons_load_files_into_viewer(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    contours = np.ones((2, 4, 4), dtype=np.float32)
    masks = np.ones((2, 4, 4), dtype=np.uint8)
    labels = np.ones((2, 4, 4), dtype=np.uint32)
    import tifffile

    tifffile.imwrite(pos_dir / "2_nucleus" / "contour_maps.tif", contours)
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_masks.tif", masks)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", labels)

    widget.refresh(pos_dir)

    for files_widget in (
        widget.contour_output_files,
        widget.ultrack_output_files,
    ):
        for row in files_widget._rows:
            if row._full_path is not None:
                row._on_load_clicked()

    assert "2_nucleus_contour_maps" in viewer.layers
    assert "2_nucleus_foreground_masks" in viewer.layers
    assert "2_nucleus_tracked_labels" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["2_nucleus_contour_maps"].data, contours)
    np.testing.assert_array_equal(viewer.layers["2_nucleus_foreground_masks"].data, masks)
    np.testing.assert_array_equal(viewer.layers["2_nucleus_tracked_labels"].data, labels)

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 6: Run behavior tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_contour_maps_build_writes_contour_scores_and_thresholded_masks tests/napari/test_nucleus_tracking_correction_layout.py::test_contour_filter_run_overwrites_contour_maps tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_section_calls_ultrack_segment_on_run tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_tracking_refreshes_stage_output_files tests/napari/test_nucleus_tracking_correction_layout.py::test_nucleus_stage_file_load_buttons_load_files_into_viewer -q
```

Expected: pass.

- [ ] **Step 7: Commit behavior coverage**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "test: cover nucleus stage-local file refresh"
```

---

### Task 5: Focused Verification And Diff Review

**Files:**
- Verify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Verify: `tests/napari/test_nucleus_tracking_correction_layout.py`
- Verify: `tests/napari/test_nucleus_sweep_streaming.py`

- [ ] **Step 1: Run focused nucleus tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py tests/napari/test_nucleus_sweep_streaming.py -q
```

Expected: pass.

- [ ] **Step 2: Run related shared widget tests if collection is fixed**

Run:

```bash
pytest tests/napari/test_ui_style.py -q
```

Expected: pass if the existing `write_hypothesis_record` import issue has been fixed. If it still fails during collection with:

```text
ImportError: cannot import name 'write_hypothesis_record' from 'cellflow.database.hypotheses'
```

record that as a pre-existing blocker and do not change unrelated database code in this plan.

- [ ] **Step 3: Inspect exact diff**

Run:

```bash
git diff -- src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
```

Expected:
- `NucleusWorkflowWidget` has no top-level `self.input_files`.
- Contour Maps has `contour_input_files`, `contour_output_files`, `contour_status_lbl`, and `build_progress_bar`.
- Ultrack Database Generation has `db_gen_input_files`, `db_gen_output_files`, `db_gen_status_lbl`, and `db_gen_progress_bar`.
- Ultrack Tracking has `ultrack_input_files`, `ultrack_output_files`, `ultrack_status_lbl`, and `ultrack_progress_bar`.
- Every stage file widget is constructed through `PipelineFilesWidget(..., viewer=self.viewer)`.
- `refresh()` calls `_refresh_stage_files(pos_dir)`.
- `_on_build_done`, `_on_run_contour_filter`, `_on_db_gen_done`, and `_on_run_ultrack_done` refresh stage file widgets after writes.
- `sweep_parameter_grid()` is no longer used for the contour and contour-filter grids.
- The `min`, `max`, and `step` sweep header labels are absent from the contour section.

- [ ] **Step 4: Check worktree paths before handoff**

Run:

```bash
git status --short
```

Expected: only planned files are modified, plus this plan file if it is not committed separately. Do not stage or revert unrelated worktree changes.

- [ ] **Step 5: Commit the plan if requested**

If the user wants the plan committed before execution:

```bash
git add docs/superpowers/plans/2026-05-08-nucleus-segmentation-tracking-subwidget-status-layout.md
git commit -m "docs: plan nucleus stage-local layout refactor"
```
