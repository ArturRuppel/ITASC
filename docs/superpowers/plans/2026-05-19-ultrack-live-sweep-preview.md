# Ultrack Live Sweep Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace persisted Ultrack source-stack TIFFs with an in-memory whole-sweep napari preview and direct threshold-based database generation.

**Architecture:** Keep the pure threshold sweep logic in `tracking_ultrack/multi_threshold.py`, add a path-based builder that thresholds canonical contour/foreground maps in memory, and make the nucleus pipeline widget preview the same sweep as two napari image layers. The GUI stops writing and requiring `2_nucleus/contour_sources.tif` / `foreground_sources.tif`; DB generation recomputes from canonical maps and current controls.

**Tech Stack:** Python, NumPy, tifffile, napari layers, qtpy, pytest.

---

## File Structure

### Modified

- `src/cellflow/tracking_ultrack/multi_threshold.py` - add `build_ultrack_database_from_thresholds(...)` and share the existing per-source segmentation/merge logic with `build_ultrack_database_from_sources(...)`.
- `src/cellflow/napari/nucleus_pipeline_widget.py` - change the Ultrack Inputs row from file generation to preview generation; switch DB generation to the new threshold builder; add two image-layer helpers.
- `src/cellflow/napari/nucleus_workflow_widget.py` - remove source-stack rows from Pipeline Files and adjust top-level comments/path helper compatibility.
- `src/cellflow/napari/data_panel_widget.py` - remove source-stack rows from the project-wide status panel.
- `tests/tracking_ultrack/test_multi_threshold.py` - cover the new threshold-based DB builder.
- `tests/napari/test_nucleus_pipeline_widget.py` - cover preview layers and DB generation without source TIFFs.
- `tests/napari/test_nucleus_tracking_inputs_widget.py`, `tests/napari/test_nucleus_tracking_correction_layout.py`, `tests/napari/test_nucleus_db_browser_widget.py`, `tests/napari/test_nucleus_correction_widget.py` - update import stubs to expose the new builder.

### Left Compatible

- `write_ultrack_source_stacks(...)` stays for scripts and old tests.
- `build_ultrack_database_from_sources(...)` stays for scripts and old tests.
- `NucleusArtifactPaths.contour_sources` and `.foreground_sources` can stay as deprecated compatibility properties, but the GUI should not call them.

---

## Task 1: Add Threshold-Based DB Builder

**Files:**
- Modify: `src/cellflow/tracking_ultrack/multi_threshold.py`
- Test: `tests/tracking_ultrack/test_multi_threshold.py`

- [ ] **Step 1: Write the failing test**

Add this test after `test_build_ultrack_database_from_sources_segments_sources_in_order` in `tests/tracking_ultrack/test_multi_threshold.py`:

```python
def test_build_ultrack_database_from_thresholds_segments_generated_sources(
    tmp_path, monkeypatch
):
    import cellflow.tracking_ultrack.multi_threshold as mt

    contours = np.array(
        [
            [[0.1, 0.6], [0.8, 0.2]],
            [[0.4, 0.7], [0.3, 1.0]],
        ],
        dtype=np.float32,
    )
    foreground_scores = np.array(
        [
            [[0.1, 0.9], [0.6, 0.2]],
            [[0.8, 0.4], [0.7, 1.0]],
        ],
        dtype=np.float32,
    )
    contours_path = tmp_path / "nucleus_contours.tif"
    foreground_path = tmp_path / "nucleus_foreground.tif"
    tifffile.imwrite(contours_path, contours)
    tifffile.imwrite(foreground_path, foreground_scores)

    segmented: list[tuple[np.ndarray, np.ndarray, str]] = []
    merge_calls = []
    progress_messages = []

    def fake_build_config(cfg, tmp_dir):
        class DummyConfig:
            pass

        dummy = DummyConfig()
        dummy.tmp_dir = tmp_dir
        return dummy

    def fake_segment(foreground_source, contour_source, ultrack_cfg, cfg):
        segmented.append(
            (foreground_source.copy(), contour_source.copy(), ultrack_cfg.tmp_dir.name)
        )
        (ultrack_cfg.tmp_dir / "data.db").write_bytes(b"source")

    def fake_merge(temp_dbs, output_path, frame_shape, progress_cb=None):
        merge_calls.append((list(temp_dbs), output_path, frame_shape))
        output_path.write_bytes(b"merged")

    monkeypatch.setattr(mt, "_build_ultrack_config", fake_build_config)
    monkeypatch.setattr(mt, "_run_ultrack_segment", fake_segment)
    monkeypatch.setattr(mt, "merge_ultrack_databases", fake_merge)
    monkeypatch.setattr(mt, "run_linking", lambda working_dir, cfg: [])

    report = mt.build_ultrack_database_from_thresholds(
        contours_path,
        foreground_path,
        tmp_path / "work",
        TrackingConfig(),
        contour_thresholds=[0.5, 0.75],
        foreground_thresholds=[0.6],
        progress_cb=progress_messages.append,
    )

    assert report == UltrackDatabaseBuildReport()
    assert [item[2] for item in segmented] == ["_source_tmp_0", "_source_tmp_1"]
    np.testing.assert_allclose(
        segmented[0][1],
        np.where(contours >= 0.5, contours, 0.0),
    )
    np.testing.assert_array_equal(
        segmented[0][0],
        (foreground_scores >= 0.6).astype(np.float32),
    )
    np.testing.assert_allclose(
        segmented[1][1],
        np.where(contours >= 0.75, contours, 0.0),
    )
    assert merge_calls[0][2] == (2, 2)
    assert any("Building threshold source sweep" in msg for msg in progress_messages)
    assert not (tmp_path / "contour_sources.tif").exists()
    assert not (tmp_path / "foreground_sources.tif").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/tracking_ultrack/test_multi_threshold.py::test_build_ultrack_database_from_thresholds_segments_generated_sources -v
```

Expected: fail with `AttributeError` because `build_ultrack_database_from_thresholds` does not exist.

- [ ] **Step 3: Extract the shared array DB builder**

In `src/cellflow/tracking_ultrack/multi_threshold.py`, add this helper just above `build_ultrack_database_from_sources`:

```python
def _build_ultrack_database_from_source_arrays(
    contour_sources: np.ndarray,
    foreground_sources: np.ndarray,
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> UltrackDatabaseBuildReport:
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    contour_sources = _normalize_source_stack(contour_sources, "contour_sources")
    foreground_sources = _normalize_source_stack(
        foreground_sources,
        "foreground_sources",
    )
    if contour_sources.shape != foreground_sources.shape:
        raise ValueError("contour_sources and foreground_sources must have the same shape.")
    _validate_source_stacks(contour_sources, foreground_sources)
    source_count, _frame_count, h, w = contour_sources.shape

    temp_dirs: list[Path] = []
    temp_dbs: list[Path] = []

    try:
        for source_index in range(source_count):
            _notify(
                progress_cb,
                f"Segmenting source {source_index + 1}/{source_count} …",
            )
            tmp_dir = working_dir / f"_source_tmp_{source_index}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            temp_dirs.append(tmp_dir)

            ultrack_cfg = _build_ultrack_config(cfg, tmp_dir)
            _run_ultrack_segment(
                foreground_sources[source_index],
                contour_sources[source_index],
                ultrack_cfg,
                cfg,
            )

            db_path = tmp_dir / "data.db"
            if not db_path.exists():
                raise RuntimeError(
                    f"Ultrack segment did not create {db_path} for source {source_index}"
                )
            temp_dbs.append(db_path)

        _notify(progress_cb, "Merging source databases …")
        merge_ultrack_databases(
            temp_dbs,
            working_dir / "data.db",
            frame_shape=(int(h), int(w)),
            progress_cb=progress_cb,
        )

        _notify(progress_cb, "Linking candidates …")
        for step, total, label in run_linking(working_dir, cfg):
            _notify(progress_cb, f"[link {step}/{total}] {label}")

    finally:
        for td in temp_dirs:
            if td.exists():
                shutil.rmtree(td, ignore_errors=True)

    return UltrackDatabaseBuildReport()
```

- [ ] **Step 4: Rewire the file-based builder through the helper**

Replace the body of `build_ultrack_database_from_sources(...)` after `working_dir.mkdir(...)` with:

```python
    _notify(progress_cb, "Loading Ultrack source stacks …")
    contour_sources, foreground_sources = _load_ultrack_source_stacks(
        contour_sources_path,
        foreground_sources_path,
    )
    return _build_ultrack_database_from_source_arrays(
        contour_sources,
        foreground_sources,
        working_dir,
        cfg,
        progress_cb=progress_cb,
    )
```

Keep the existing docstring and function signature.

- [ ] **Step 5: Add the new threshold-based public builder**

Add this function immediately after `build_ultrack_database_from_sources(...)`:

```python
def build_ultrack_database_from_thresholds(
    contours_path: str | Path,
    foreground_scores_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    contour_thresholds: Sequence[float],
    foreground_thresholds: Sequence[float],
    progress_cb: Callable[[str], None] | None = None,
) -> UltrackDatabaseBuildReport:
    """Build candidate ``data.db`` from canonical maps and threshold controls.

    Unlike :func:`build_ultrack_database_from_sources`, this does not read or
    write threshold-expanded source-stack TIFFs. It loads the canonical
    contour/foreground maps, builds the full source sweep in memory, segments
    each source, merges candidates, and links them.
    """
    _notify(progress_cb, "Loading contour maps and foreground scores …")
    contours = np.asarray(tifffile.imread(str(contours_path)), dtype=np.float32)
    foreground_scores = np.asarray(
        tifffile.imread(str(foreground_scores_path)),
        dtype=np.float32,
    )

    _notify(progress_cb, "Building threshold source sweep …")
    contour_sources, foreground_sources, _metadata = build_ultrack_source_stacks(
        contours,
        foreground_scores,
        contour_thresholds=contour_thresholds,
        foreground_thresholds=foreground_thresholds,
    )
    return _build_ultrack_database_from_source_arrays(
        contour_sources,
        foreground_sources,
        working_dir,
        cfg,
        progress_cb=progress_cb,
    )
```

- [ ] **Step 6: Run focused tracking tests**

Run:

```bash
pytest tests/tracking_ultrack/test_multi_threshold.py::test_build_ultrack_database_from_thresholds_segments_generated_sources tests/tracking_ultrack/test_multi_threshold.py::test_build_ultrack_database_from_sources_segments_sources_in_order tests/tracking_ultrack/test_multi_threshold.py::test_build_ultrack_database_from_sources_normalizes_single_source_input -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/cellflow/tracking_ultrack/multi_threshold.py tests/tracking_ultrack/test_multi_threshold.py
git commit -m "Add threshold-based Ultrack database builder"
```

---

## Task 2: Preview Whole Sweep as Two Napari Layers

**Files:**
- Modify: `src/cellflow/napari/nucleus_pipeline_widget.py`
- Test: `tests/napari/test_nucleus_pipeline_widget.py`

- [ ] **Step 1: Update the test stubs for the new import**

In `_install_import_stubs()` in `tests/napari/test_nucleus_pipeline_widget.py`, replace the `cellflow.tracking_ultrack.multi_threshold` stub mapping with:

```python
        "cellflow.tracking_ultrack.multi_threshold": {
            "build_ultrack_database_from_sources": lambda *args, **kwargs: None,
            "build_ultrack_database_from_thresholds": lambda *args, **kwargs: None,
            "build_ultrack_source_stacks": lambda *args, **kwargs: (
                np.zeros((1, 1, 1, 1), dtype=np.float32),
                np.zeros((1, 1, 1, 1), dtype=np.uint8),
                [],
            ),
            "write_ultrack_source_stacks": lambda *args, **kwargs: [],
        },
```

- [ ] **Step 2: Replace the old source-stack handler test with a preview test**

Replace `test_build_segmentation_inputs_calls_write_source_stacks_from_divergence_maps` with:

```python
def test_build_segmentation_inputs_previews_source_sweep_without_writing(
    tmp_path, monkeypatch
):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    _install_sync_thread_worker(monkeypatch, pipeline_module)

    pos_dir = tmp_path / "pos00"
    widget._pos_dir = pos_dir
    widget.db_gen_threshold_min_spin.setValue(0.2)
    widget.db_gen_threshold_max_spin.setValue(0.4)
    widget.db_gen_threshold_step_spin.setValue(0.2)
    widget.source_foreground_threshold_min_spin.setValue(0.3)
    widget.source_foreground_threshold_max_spin.setValue(0.5)
    widget.source_foreground_threshold_step_spin.setValue(0.2)

    import tifffile
    (pos_dir / "1_cellpose").mkdir(parents=True)
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_contours.tif",
        np.ones((2, 3, 3), dtype=np.float32),
    )
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_foreground.tif",
        np.ones((2, 3, 3), dtype=np.float32),
    )

    calls = []
    contour_preview = np.full((4, 2, 3, 3), 0.75, dtype=np.float32)
    foreground_preview = np.ones((4, 2, 3, 3), dtype=np.uint8)
    metadata = [
        {"contour_threshold": 0.2, "foreground_threshold": 0.3},
        {"contour_threshold": 0.2, "foreground_threshold": 0.5},
        {"contour_threshold": 0.4, "foreground_threshold": 0.3},
        {"contour_threshold": 0.4, "foreground_threshold": 0.5},
    ]

    def fake_build(contours, foreground_scores, **kwargs):
        calls.append((contours.copy(), foreground_scores.copy(), kwargs))
        return contour_preview, foreground_preview, metadata

    def fail_write(*args, **kwargs):
        raise AssertionError("source-stack TIFF writer should not be called")

    monkeypatch.setattr(pipeline_module, "build_ultrack_source_stacks", fake_build)
    monkeypatch.setattr(
        pipeline_module,
        "write_ultrack_source_stacks",
        fail_write,
        raising=False,
    )

    widget._on_build_segmentation_inputs()

    assert len(calls) == 1
    _contours, _foreground_scores, kwargs = calls[0]
    np.testing.assert_allclose(kwargs["contour_thresholds"], np.array([0.2, 0.4]))
    np.testing.assert_allclose(kwargs["foreground_thresholds"], np.array([0.3, 0.5]))
    assert "Ultrack Sweep: Contours" in viewer.layers
    assert "Ultrack Sweep: Foreground" in viewer.layers
    np.testing.assert_allclose(viewer.layers["Ultrack Sweep: Contours"].data, contour_preview)
    np.testing.assert_array_equal(
        viewer.layers["Ultrack Sweep: Foreground"].data,
        foreground_preview,
    )
    assert viewer.layers["Ultrack Sweep: Contours"].metadata["thresholds"] == metadata
    assert viewer.layers["Ultrack Sweep: Foreground"].metadata["thresholds"] == metadata
    assert not (pos_dir / "2_nucleus" / "contour_sources.tif").exists()
    assert "preview" in widget.pipeline_status_lbl.text().lower()
    assert not widget.pipeline_progress_bar.isVisible()

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 3: Run the preview test to verify it fails**

Run:

```bash
pytest tests/napari/test_nucleus_pipeline_widget.py::test_build_segmentation_inputs_previews_source_sweep_without_writing -v
```

Expected: fail because `nucleus_pipeline_widget.py` still imports and calls `write_ultrack_source_stacks`.

- [ ] **Step 4: Change the imports**

In `src/cellflow/napari/nucleus_pipeline_widget.py`, replace:

```python
from cellflow.tracking_ultrack.multi_threshold import (
    build_ultrack_database_from_sources,
    write_ultrack_source_stacks,
)
```

with:

```python
from cellflow.tracking_ultrack.multi_threshold import (
    build_ultrack_database_from_thresholds,
    build_ultrack_source_stacks,
)
```

Do not import `write_ultrack_source_stacks` in this widget after this change.

If `build_ultrack_database_from_sources` is still needed by other modules, keep it
there, but remove it from `nucleus_pipeline_widget.py`.

- [ ] **Step 5: Add preview layer constants**

Near `_TRACKED_LAYER = "Tracked: Nucleus"`, add:

```python
_SWEEP_CONTOURS_LAYER = "Ultrack Sweep: Contours"
_SWEEP_FOREGROUND_LAYER = "Ultrack Sweep: Foreground"
```

- [ ] **Step 6: Add an image-layer helper**

Add this method after `_update_labels_layer(...)`:

```python
    def _update_image_layer(
        self,
        name: str,
        data: np.ndarray,
        *,
        metadata: dict | None = None,
    ) -> None:
        from napari.layers import Image

        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Image):
            layer = self.viewer.layers[name]
            layer.data = data
            layer.metadata = dict(metadata or {})
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_image(data, name=name, metadata=dict(metadata or {}))
```

- [ ] **Step 7: Replace `_on_build_segmentation_inputs` with preview behavior**

In `src/cellflow/napari/nucleus_pipeline_widget.py`, keep the existing early validation through threshold parsing, then replace the worker and done callback with:

```python
        cancel_event = threading.Event()
        self._contour_cancel = cancel_event

        def _done(result):
            contour_preview, foreground_preview, metadata = result
            self._contour_worker = None
            self._contour_cancel = None
            self._clear_progress()
            layer_metadata = {"thresholds": metadata}
            self._update_image_layer(
                _SWEEP_CONTOURS_LAYER,
                contour_preview,
                metadata=layer_metadata,
            )
            self._update_image_layer(
                _SWEEP_FOREGROUND_LAYER,
                foreground_preview,
                metadata=layer_metadata,
            )
            self._status(f"Ultrack source sweep preview ready ({len(metadata)} sources).")
            self._set_running_stage(None)

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation.nucleus_segmentation import _check_cancel

            yield (0, 3, "Loading Ultrack input maps...")
            contours = np.asarray(tifffile.imread(str(contours_path)), dtype=np.float32)
            _check_cancel(cancel_event.is_set)
            foreground_scores = np.asarray(
                tifffile.imread(str(score_path)),
                dtype=np.float32,
            )
            _check_cancel(cancel_event.is_set)
            yield (1, 3, "Building Ultrack source sweep preview...")
            contour_preview, foreground_preview, metadata = build_ultrack_source_stacks(
                contours,
                foreground_scores,
                contour_thresholds=contour_thresholds,
                foreground_thresholds=foreground_thresholds,
            )
            _check_cancel(cancel_event.is_set)
            yield (3, 3, "Loaded Ultrack source sweep preview.")
            return contour_preview, foreground_preview, metadata

        n_sources = len(contour_thresholds) * len(foreground_thresholds)
        self._status(f"Building Ultrack source sweep preview ({n_sources} sources)...")
        self._set_running_stage("seg")
        self._contour_worker = _worker()
```

- [ ] **Step 8: Run the preview test**

Run:

```bash
pytest tests/napari/test_nucleus_pipeline_widget.py::test_build_segmentation_inputs_previews_source_sweep_without_writing -v
```

Expected: pass.

- [ ] **Step 9: Commit**

Run:

```bash
git add src/cellflow/napari/nucleus_pipeline_widget.py tests/napari/test_nucleus_pipeline_widget.py
git commit -m "Preview Ultrack source sweep in napari"
```

---

## Task 3: Build DB Directly From Canonical Maps

**Files:**
- Modify: `src/cellflow/napari/nucleus_pipeline_widget.py`
- Test: `tests/napari/test_nucleus_pipeline_widget.py`

- [ ] **Step 1: Replace the DB generation test**

Replace `test_run_db_generation_calls_build_database` with:

```python
def test_run_db_generation_builds_from_canonical_maps_without_source_tiffs(
    tmp_path, monkeypatch
):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    _install_sync_thread_worker(monkeypatch, pipeline_module)

    calls = []

    def fake_build_database(**kwargs):
        calls.append(kwargs)
        data_db = kwargs["working_dir"] / "data.db"
        data_db.parent.mkdir(parents=True, exist_ok=True)
        data_db.write_bytes(b"sqlite db")
        return {"database": str(data_db)}

    monkeypatch.setattr(
        pipeline_module,
        "build_ultrack_database_from_thresholds",
        fake_build_database,
    )
    monkeypatch.setattr(pipeline_module, "_ultrack_available", lambda: True)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir(parents=True)
    import tifffile
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_contours.tif",
        np.ones((2, 4, 4), dtype=np.float32),
    )
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_foreground.tif",
        np.ones((2, 4, 4), dtype=np.float32),
    )
    widget._pos_dir = pos_dir
    widget.db_gen_threshold_min_spin.setValue(0.2)
    widget.db_gen_threshold_max_spin.setValue(0.4)
    widget.db_gen_threshold_step_spin.setValue(0.2)
    widget.source_foreground_threshold_min_spin.setValue(0.3)
    widget.source_foreground_threshold_max_spin.setValue(0.5)
    widget.source_foreground_threshold_step_spin.setValue(0.2)

    widget._on_run_db_generation()

    assert len(calls) == 1
    call = calls[0]
    assert call["contours_path"] == pos_dir / "1_cellpose" / "nucleus_contours.tif"
    assert call["foreground_scores_path"] == pos_dir / "1_cellpose" / "nucleus_foreground.tif"
    np.testing.assert_allclose(call["contour_thresholds"], np.array([0.2, 0.4]))
    np.testing.assert_allclose(call["foreground_thresholds"], np.array([0.3, 0.5]))
    assert call["working_dir"] == pos_dir / "2_nucleus" / "ultrack_workdir"
    assert "score_signal_path" not in call
    assert "complete" in widget.pipeline_status_lbl.text().lower()
    assert not widget.pipeline_progress_bar.isVisible()

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Add missing canonical map status tests**

Add these two tests after the DB generation test:

```python
def test_run_db_generation_reports_missing_canonical_contours(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    monkeypatch.setattr(pipeline_module, "_ultrack_available", lambda: True)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    widget._pos_dir = pos_dir

    widget._on_run_db_generation()

    assert "nucleus_contours.tif" in widget.pipeline_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_run_db_generation_reports_missing_canonical_foreground(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    monkeypatch.setattr(pipeline_module, "_ultrack_available", lambda: True)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    import tifffile
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_contours.tif",
        np.ones((2, 4, 4), dtype=np.float32),
    )
    widget._pos_dir = pos_dir

    widget._on_run_db_generation()

    assert "nucleus_foreground.tif" in widget.pipeline_status_lbl.text()

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 3: Run the DB tests to verify they fail**

Run:

```bash
pytest tests/napari/test_nucleus_pipeline_widget.py::test_run_db_generation_builds_from_canonical_maps_without_source_tiffs tests/napari/test_nucleus_pipeline_widget.py::test_run_db_generation_reports_missing_canonical_contours tests/napari/test_nucleus_pipeline_widget.py::test_run_db_generation_reports_missing_canonical_foreground -v
```

Expected: fail because `_on_run_db_generation` still requires source-stack TIFFs.

- [ ] **Step 4: Change `_on_run_db_generation` validation**

In `src/cellflow/napari/nucleus_pipeline_widget.py`, replace the initial source-stack path checks with:

```python
        paths = self._paths
        if paths is None:
            self._status("No project open."); return
        contours_path = paths.nucleus_contours
        score_path = paths.nucleus_foreground
        if not contours_path.exists():
            self._status(
                "Missing: nucleus_contours.tif - build divergence maps first."
            ); return
        if not score_path.exists():
            self._status(
                "Missing: nucleus_foreground.tif - build divergence maps first."
            ); return
```

Then parse thresholds before starting the worker:

```python
        try:
            contour_thresholds = self._source_contour_thresholds_from_controls()
            foreground_thresholds = self._source_foreground_thresholds_from_controls()
        except ValueError as exc:
            self._status(str(exc)); return
```

- [ ] **Step 5: Change the DB worker call**

Inside `_run()` in `_on_run_db_generation`, replace `build_ultrack_database_from_sources(...)` with:

```python
                    build_ultrack_database_from_thresholds(
                        contours_path=contours_path,
                        foreground_scores_path=score_path,
                        working_dir=working_dir,
                        cfg=cfg,
                        contour_thresholds=contour_thresholds,
                        foreground_thresholds=foreground_thresholds,
                        progress_cb=_progress_cb,
                    )
```

Keep the existing `apply_annotations_and_score(...)` call after the build; it should still use `score_path = self._foreground_path()`.

- [ ] **Step 6: Run the DB tests**

Run:

```bash
pytest tests/napari/test_nucleus_pipeline_widget.py::test_run_db_generation_builds_from_canonical_maps_without_source_tiffs tests/napari/test_nucleus_pipeline_widget.py::test_run_db_generation_reports_missing_canonical_contours tests/napari/test_nucleus_pipeline_widget.py::test_run_db_generation_reports_missing_canonical_foreground -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/cellflow/napari/nucleus_pipeline_widget.py tests/napari/test_nucleus_pipeline_widget.py
git commit -m "Build Ultrack database from threshold controls"
```

---

## Task 4: Remove Source Stack Files From UI Tracking

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `src/cellflow/napari/data_panel_widget.py`
- Test: `tests/napari/test_nucleus_pipeline_widget.py`

- [ ] **Step 1: Add Pipeline Files assertions**

Add this test after `test_nucleus_pipeline_files_header_uses_magnifier_button`:

```python
def test_nucleus_pipeline_files_omit_source_stack_artifacts():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    tracked_paths = [
        path
        for group in widget._files_widget._groups
        for path, _label in group[1]
    ]

    assert "2_nucleus/contour_sources.tif" not in tracked_paths
    assert "2_nucleus/foreground_sources.tif" not in tracked_paths
    assert "2_nucleus/ultrack_workdir/data.db" in tracked_paths

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Run the Pipeline Files test to verify it fails**

Run:

```bash
pytest tests/napari/test_nucleus_pipeline_widget.py::test_nucleus_pipeline_files_omit_source_stack_artifacts -v
```

Expected: fail because the workflow still lists source-stack TIFFs.

- [ ] **Step 3: Update `nucleus_workflow_widget.py` file groups and comments**

In the module docstring, replace:

```python
  2. Source stacks → ``contour_sources.tif`` / ``foreground_sources.tif``
  3. Ultrack database + solve → ``data.db`` / ``tracked_labels.tif``
```

with:

```python
  2. Source sweep preview → in-memory napari layers
  3. Ultrack database + solve → ``data.db`` / ``tracked_labels.tif``
```

In the `PipelineFilesWidget` config, replace the Intermediates group:

```python
                ("Intermediates", [
                    ("2_nucleus/contour_sources.tif", "Contour sources"),
                    ("2_nucleus/foreground_sources.tif", "Foreground sources"),
                    ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
                ]),
```

with:

```python
                ("Intermediates", [
                    ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
                ]),
```

- [ ] **Step 4: Update `data_panel_widget.py` file groups**

In `_TRACKED_FILE_GROUPS`, replace the `Nucleus Workflow` group:

```python
    ("Nucleus Workflow", [
        ("2_nucleus/contour_sources.tif", "Contour sources"),
        ("2_nucleus/foreground_sources.tif", "Foreground sources"),
        ("2_nucleus/ultrack_workdir/data.db", "Ultrack DB"),
        ("2_nucleus/tracked_labels.tif", "Tracked labels"),
    ]),
```

with:

```python
    ("Nucleus Workflow", [
        ("2_nucleus/ultrack_workdir/data.db", "Ultrack DB"),
        ("2_nucleus/tracked_labels.tif", "Tracked labels"),
    ]),
```

- [ ] **Step 5: Run the Pipeline Files test**

Run:

```bash
pytest tests/napari/test_nucleus_pipeline_widget.py::test_nucleus_pipeline_files_omit_source_stack_artifacts -v
```

Expected: pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py src/cellflow/napari/data_panel_widget.py tests/napari/test_nucleus_pipeline_widget.py
git commit -m "Remove Ultrack source stacks from file tracking"
```

---

## Task 5: Update Remaining Test Stubs and Compatibility Surfaces

**Files:**
- Modify: `tests/napari/test_nucleus_tracking_inputs_widget.py`
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`
- Modify: `tests/napari/test_nucleus_db_browser_widget.py`
- Modify: `tests/napari/test_nucleus_correction_widget.py`
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`

- [ ] **Step 1: Update all napari multi-threshold stubs**

In each test file listed above, find stub mappings for `"cellflow.tracking_ultrack.multi_threshold"` and ensure they include:

```python
        "cellflow.tracking_ultrack.multi_threshold": {
            "build_ultrack_database_from_sources": lambda *args, **kwargs: None,
            "build_ultrack_database_from_thresholds": lambda *args, **kwargs: None,
            "preview_ultrack_source_stack_frame": lambda *args, **kwargs: (None, None, 0, []),
            "build_ultrack_source_stacks": lambda *args, **kwargs: (
                np.zeros((1, 1, 1, 1), dtype=np.float32),
                np.zeros((1, 1, 1, 1), dtype=np.uint8),
                [],
            ),
            "write_ultrack_source_stacks": lambda *args, **kwargs: [],
        },
```

If a file does not import `numpy as np`, add:

```python
import numpy as np
```

Keep only the keys that existing imports require in that file plus the new builder and `build_ultrack_source_stacks`.

- [ ] **Step 2: Remove legacy workflow path delegates**

In `src/cellflow/napari/nucleus_workflow_widget.py`, delete these methods if no tests or code still call them:

```python
    def _contour_sources_path(self) -> Path | None:
        return self.nucleus_pipeline_widget._contour_sources_path()

    def _foreground_sources_path(self) -> Path | None:
        return self.nucleus_pipeline_widget._foreground_sources_path()
```

Run this search first:

```bash
rg -n "_contour_sources_path|_foreground_sources_path" src tests
```

Expected before deletion: references only in `nucleus_workflow_widget.py` and `nucleus_pipeline_widget.py`. If tests still reference the workflow delegates, update those tests to use canonical map paths instead, then delete the delegates.

- [ ] **Step 3: Run affected napari smoke tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_inputs_widget.py tests/napari/test_nucleus_tracking_correction_layout.py tests/napari/test_nucleus_db_browser_widget.py tests/napari/test_nucleus_correction_widget.py -v
```

Expected: all selected tests pass or skip for missing optional Qt/napari dependencies in the same way they did before this task.

- [ ] **Step 4: Commit**

Run:

```bash
git add tests/napari/test_nucleus_tracking_inputs_widget.py tests/napari/test_nucleus_tracking_correction_layout.py tests/napari/test_nucleus_db_browser_widget.py tests/napari/test_nucleus_correction_widget.py src/cellflow/napari/nucleus_workflow_widget.py
git commit -m "Update nucleus workflow test stubs for live sweep preview"
```

---

## Task 6: Final Verification and Cleanup

**Files:**
- Verify only.

- [ ] **Step 1: Search for unwanted GUI calls to source-stack files**

Run:

```bash
rg -n "write_ultrack_source_stacks|contour_sources_path|foreground_sources_path|Missing: contour_sources|Missing: foreground_sources|2_nucleus/contour_sources.tif|2_nucleus/foreground_sources.tif" src/cellflow/napari tests/napari
```

Expected: no GUI references that write or require source-stack TIFFs. Remaining references in compatibility tests or path properties are acceptable only outside GUI flow.

- [ ] **Step 2: Run focused tracking tests**

Run:

```bash
pytest tests/tracking_ultrack/test_multi_threshold.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Run focused nucleus pipeline tests**

Run:

```bash
pytest tests/napari/test_nucleus_pipeline_widget.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Run broader napari nucleus tests**

Run:

```bash
pytest tests/napari/test_nucleus_pipeline_widget.py tests/napari/test_nucleus_tracking_inputs_widget.py tests/napari/test_nucleus_tracking_correction_layout.py tests/napari/test_nucleus_db_browser_widget.py tests/napari/test_nucleus_correction_widget.py -v
```

Expected: all tests pass or skip only for pre-existing optional dependency reasons.

- [ ] **Step 5: Inspect final diff**

Run:

```bash
git status --short
git diff --stat
git diff -- src/cellflow/tracking_ultrack/multi_threshold.py src/cellflow/napari/nucleus_pipeline_widget.py src/cellflow/napari/nucleus_workflow_widget.py src/cellflow/napari/data_panel_widget.py
```

Expected: only live-sweep preview changes are present. Existing unrelated dirty files should not be staged or modified by this plan.

- [ ] **Step 6: Commit final verification fixes if any were needed**

If Step 1-5 required code or test changes, stage only the files changed by this plan:

```bash
git add src/cellflow/tracking_ultrack/multi_threshold.py src/cellflow/napari/nucleus_pipeline_widget.py src/cellflow/napari/nucleus_workflow_widget.py src/cellflow/napari/data_panel_widget.py tests/tracking_ultrack/test_multi_threshold.py tests/napari/test_nucleus_pipeline_widget.py tests/napari/test_nucleus_tracking_inputs_widget.py tests/napari/test_nucleus_tracking_correction_layout.py tests/napari/test_nucleus_db_browser_widget.py tests/napari/test_nucleus_correction_widget.py
git commit -m "Finish Ultrack live sweep preview"
```

If no changes were needed, do not create an empty commit.
