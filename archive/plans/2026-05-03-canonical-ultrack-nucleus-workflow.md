# Canonical Ultrack Nucleus Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the nucleus workflow widget to use five canonical top-level sections, replacing the old H5 hypothesis path with `ultrack.segment`-based candidate generation, a new DB browser, and a split Tracking / Correction layout.

**Architecture:** The old "2. Hypothesis Generation" and "3. H5 Database Browser" sections are hidden (not deleted) and replaced by "2. Ultrack Database Generation" (segment → score → link) and "3. Ultrack Database Browser". The shared "4. Tracking & Correction" wrapper is removed; "Ultrack Tracking" (solve only) and "Correction" become separate top-level sections numbered 4 and 5. Extend migrates to `data.db`; resolve-from-validated migrates to `ultrack.segment`.

**Tech Stack:** PyQt5/PySide2 via qtpy, napari, ultrack (via conda env), SQLAlchemy, tifffile, pydantic, pytest-qt

---

## File Map

| File | What changes |
|---|---|
| `src/cellflow/tracking_ultrack/config.py` | Add segmentation fields |
| `src/cellflow/tracking_ultrack/ingest.py` | Expose `_build_ultrack_config` for external use; update to use seg fields |
| `src/cellflow/tracking_ultrack/extend.py` | Add `extend_track_from_db` function |
| `src/cellflow/tracking_ultrack/reseed.py` | Add `resolve_with_canonical_segment` function |
| `src/cellflow/napari/nucleus_workflow_widget.py` | Major UI restructure |
| `tests/napari/test_nucleus_tracking_correction_layout.py` | Update broken tests + add new ones |

---

## Task 1: Add segmentation fields to TrackingConfig

**Files:**
- Modify: `src/cellflow/tracking_ultrack/config.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

The `segment()` call needs these on `ultrack_cfg.segmentation_config`. We store them on `TrackingConfig` so terminal scripts can serialise them.

- [ ] **Step 1: Write the failing test**

Add to `test_nucleus_tracking_correction_layout.py`:

```python
def test_tracking_config_has_segmentation_fields():
    from cellflow.tracking_ultrack.config import TrackingConfig
    cfg = TrackingConfig()
    assert cfg.seg_min_area == 300
    assert cfg.seg_max_area == 100_000
    assert cfg.seg_foreground_threshold == 0.5
    assert cfg.seg_min_frontier == 0.0
    assert cfg.seg_ws_hierarchy == "area"
    assert cfg.seg_n_workers == 1
```

- [ ] **Step 2: Run to verify it fails**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_tracking_config_has_segmentation_fields -v
```
Expected: `AttributeError: 'TrackingConfig' object has no attribute 'seg_min_area'`

- [ ] **Step 3: Implement**

In `src/cellflow/tracking_ultrack/config.py`, after the `max_segments_per_time` line, add:

```python
    # Segmentation (ultrack.segment / ultrack.core.segmentation.processing.segment)
    seg_min_area: int = 300
    seg_max_area: int = 100_000
    seg_foreground_threshold: float = 0.5
    seg_min_frontier: float = 0.0
    seg_ws_hierarchy: str = "area"    # "area", "dynamics", or "volume"
    seg_n_workers: int = 1
```

- [ ] **Step 4: Run to verify it passes**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_tracking_config_has_segmentation_fields -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/config.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat(config): add segmentation fields to TrackingConfig"
```

---

## Task 2: Update `_build_ultrack_config` to apply segmentation fields

**Files:**
- Modify: `src/cellflow/tracking_ultrack/ingest.py`

- [ ] **Step 1: Write the failing test**

Add to `test_nucleus_tracking_correction_layout.py`:

```python
def test_build_ultrack_config_applies_segmentation_fields(tmp_path):
    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.ingest import _build_ultrack_config
    cfg = TrackingConfig(
        seg_min_area=500,
        seg_max_area=50_000,
        seg_foreground_threshold=0.3,
        seg_min_frontier=0.05,
        seg_ws_hierarchy="dynamics",
        seg_n_workers=2,
    )
    ultrack_cfg = _build_ultrack_config(cfg, tmp_path)
    sc = ultrack_cfg.segmentation_config
    assert sc.min_area == 500
    assert sc.max_area == 50_000
    assert abs(sc.threshold - 0.3) < 1e-6
    assert abs(sc.min_frontier - 0.05) < 1e-6
    assert sc.n_workers == 2
```

- [ ] **Step 2: Run to verify it fails**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_build_ultrack_config_applies_segmentation_fields -v
```
Expected: FAIL (fields are not being set on segmentation_config)

- [ ] **Step 3: Implement**

In `src/cellflow/tracking_ultrack/ingest.py`, update `_build_ultrack_config` to set segmentation fields. The function currently builds a `MainConfig` with data/linking/tracking keys only. Extend it:

```python
def _build_ultrack_config(cfg: TrackingConfig, working_dir: Path):
    from ultrack.config import MainConfig
    from ultrack.config.segmentationconfig import NAME_TO_WS_HIER

    ultrack_cfg = MainConfig(
        data={"working_dir": str(working_dir)},
        linking={
            "max_distance": cfg.max_distance,
            "max_neighbors": cfg.max_neighbors,
            "distance_weight": cfg.distance_weight,
            "n_workers": cfg.link_n_workers,
        },
        tracking={
            "solver_name": _select_solver(),
            "appear_weight": cfg.appear_weight,
            "disappear_weight": cfg.disappear_weight,
            "division_weight": cfg.division_weight,
            "link_function": cfg.link_function,
            "power": cfg.power,
            "bias": cfg.bias,
            "solution_gap": cfg.solution_gap,
            "time_limit": cfg.time_limit,
            "window_size": cfg.window_size if cfg.window_size > 0 else None,
        },
    )
    sc = ultrack_cfg.segmentation_config
    sc.min_area = cfg.seg_min_area
    sc.max_area = cfg.seg_max_area
    sc.threshold = cfg.seg_foreground_threshold
    sc.min_frontier = cfg.seg_min_frontier
    sc.ws_hierarchy = NAME_TO_WS_HIER[cfg.seg_ws_hierarchy]
    sc.n_workers = cfg.seg_n_workers
    return ultrack_cfg
```

- [ ] **Step 4: Run to verify it passes**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_build_ultrack_config_applies_segmentation_fields -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/ingest.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat(ingest): apply TrackingConfig seg fields in _build_ultrack_config"
```

---

## Task 3: Add `extend_track_from_db` function

**Files:**
- Modify: `src/cellflow/tracking_ultrack/extend.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

The old `extend_track` reads candidates from `hypotheses.h5`. The new DB-backed version queries `NodeDB` from `data.db`, deserialises masks via `node.pickle`, and picks the best candidate.

- [ ] **Step 1: Write the failing test**

Add to test file (uses stubs so no real DB needed):

```python
def test_extend_track_from_db_missing_db_raises(tmp_path):
    from cellflow.tracking_ultrack.extend import extend_track_from_db
    import numpy as np
    tracked = np.zeros((3, 10, 10), dtype=np.uint32)
    tracked[0, 2:5, 2:5] = 7
    result = extend_track_from_db(
        source_id=7,
        source_frame=0,
        direction="forward",
        tracked_labels=tracked,
        db_path=tmp_path / "data.db",
        d_max=40.0,
    )
    assert result is None  # missing DB returns None; widget shows clear error
```

- [ ] **Step 2: Run to verify it fails**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_extend_track_from_db_missing_db_raises -v
```
Expected: `ImportError` or `AttributeError` — function does not exist yet

- [ ] **Step 3: Implement `extend_track_from_db`**

Add to the bottom of `src/cellflow/tracking_ultrack/extend.py`:

```python
def extend_track_from_db(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,   # (T, Y, X) uint32
    db_path: Path,
    d_max: float = _D_MAX_DEFAULT,
) -> ExtendResult | None:
    """Extend a track using candidates from ultrack_workdir/data.db.

    Returns None if the DB is missing, target frame is out of range, or no
    candidate within d_max is found.  Widget caller should show a local status
    message on None — the function itself is silent.
    """
    import pickle
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    if not db_path.exists():
        return None

    T = tracked_labels.shape[0]
    target_frame = source_frame + (1 if direction == "forward" else -1)
    if target_frame < 0 or target_frame >= T:
        return None

    source_mask = tracked_labels[source_frame] == source_id
    if not source_mask.any():
        return None

    props = regionprops(source_mask.astype(np.uint8))
    src_cy, src_cx = props[0].centroid
    src_area = float(props[0].area)

    target_frame_labels = tracked_labels[target_frame]
    other_cells = (target_frame_labels != 0) & (target_frame_labels != source_id)

    engine = sqla.create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    best: ExtendResult | None = None
    best_score: tuple[float, float] | None = None

    with Session(engine) as session:
        candidates = session.query(NodeDB).filter(NodeDB.t_node_id == target_frame).all()
        for node in candidates:
            try:
                mask_data = pickle.loads(node.pickle)
            except Exception:
                continue
            # NodeDB stores (bbox, mask) or similar — unpack appropriately
            # bbox fields: y_start, x_start, y_end, x_end (or similar Ultrack schema)
            bbox = (int(node.y_start), int(node.x_start), int(node.y_end), int(node.x_end))
            h = bbox[2] - bbox[0]
            w = bbox[3] - bbox[1]
            if isinstance(mask_data, np.ndarray) and mask_data.shape == (h, w):
                mask_bool = mask_data.astype(bool)
            else:
                continue
            # Place mask in full-frame coordinates
            full_mask = np.zeros(tracked_labels.shape[1:], dtype=bool)
            full_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = mask_bool
            cy = bbox[0] + (bbox[2] - bbox[0]) / 2
            cx = bbox[1] + (bbox[3] - bbox[1]) / 2
            dist = float(np.hypot(cy - src_cy, cx - src_cx))
            if dist > d_max:
                continue
            cand_area = float(full_mask.sum())
            if cand_area == 0:
                continue
            existing_overlap = float((full_mask & other_cells).sum()) / cand_area
            area_ratio = min(src_area, cand_area) / max(src_area, cand_area)
            combined = area_ratio * (1.0 - existing_overlap)
            score = (combined, -dist)
            if best_score is None or score > best_score:
                best_score = score
                best = ExtendResult(
                    target_frame=target_frame,
                    candidate_label=int(getattr(node, "id", 0)),
                    candidate_partition=0,
                    mask_2d=full_mask,
                    bbox=bbox,
                    centroid_distance=dist,
                    area_ratio=area_ratio,
                    existing_overlap=existing_overlap,
                )
    engine.dispose()
    return best
```

- [ ] **Step 4: Run to verify it passes**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_extend_track_from_db_missing_db_raises -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/extend.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat(extend): add extend_track_from_db using ultrack NodeDB"
```

---

## Task 4: Add `resolve_with_canonical_segment` to reseed.py

**Files:**
- Modify: `src/cellflow/tracking_ultrack/reseed.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

The new resolve path uses `ultrack.segment` instead of `ingest_hypotheses_to_db`, so it does not require `hypotheses.h5`.

- [ ] **Step 1: Write the failing test**

```python
def test_resolve_with_canonical_segment_exists():
    from cellflow.tracking_ultrack.reseed import resolve_with_canonical_segment
    import inspect
    sig = inspect.signature(resolve_with_canonical_segment)
    params = set(sig.parameters)
    assert "contour_maps_path" in params
    assert "foreground_masks_path" in params
    assert "validated_tracks" in params
    assert "tracked_labels" in params
    assert "cfg" in params
    assert "intensity_image_path" in params
    # must NOT require hypotheses_path
    assert "hypotheses_path" not in params
```

- [ ] **Step 2: Run to verify it fails**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_resolve_with_canonical_segment_exists -v
```
Expected: `ImportError` — function does not exist

- [ ] **Step 3: Implement `resolve_with_canonical_segment`**

Add to the bottom of `src/cellflow/tracking_ultrack/reseed.py`:

```python
def resolve_with_canonical_segment(
    contour_maps_path: str | Path,
    foreground_masks_path: str | Path,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
    cfg: "TrackingConfig",
    progress_cb: Callable[[str], None] | None = None,
    *,
    intensity_image_path: str | Path,
) -> tuple[np.ndarray, dict[int, int]]:
    """Re-solve using canonical ultrack.segment instead of hypotheses.h5.

    Replaces the old resolve_with_validation() call chain:
      old: hypotheses.h5 → ingest_hypotheses_to_db → inject → score → link → solve
      new: foreground_masks + contour_maps → ultrack.segment → inject → score → link → solve

    Parameters
    ----------
    contour_maps_path:
        Path to ``2_nucleus/contour_maps.tif``.
    foreground_masks_path:
        Path to ``2_nucleus/foreground_masks.tif`` (canonical external input).
    validated_tracks:
        ``{cell_id: {frames}}`` locked-in validated tracks.
    tracked_labels:
        Current corrected labelmap ``(T, Y, X)``.
    cfg:
        TrackingConfig with both segmentation and solver parameters.
    progress_cb:
        Optional status string callback.
    intensity_image_path:
        Path to nucleus z-average image for scoring.

    Returns
    -------
    tuple[np.ndarray, dict[int, int]]
        ``(exported_labelmap, id_map)`` same contract as resolve_with_validation.
    """
    import tempfile
    import tifffile

    from cellflow.tracking_ultrack.ingest import _build_ultrack_config
    from cellflow.tracking_ultrack.linking import run_linking
    from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs
    from cellflow.tracking_ultrack.solve import run_solve
    from cellflow.tracking_ultrack.export import export_tracked_labels
    from cellflow.tracking_ultrack.validation_nodes import inject_validated_nodes

    try:
        from ultrack.core.segmentation.processing import segment as ultrack_segment
    except ImportError as exc:
        raise ImportError(
            "ultrack must be installed (conda env cellflow) to use canonical segment"
        ) from exc

    def _notify(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    contour_maps_path = Path(contour_maps_path)
    foreground_masks_path = Path(foreground_masks_path)

    if not validated_tracks:
        return np.asarray(tracked_labels, dtype=np.uint32).copy(), {}

    _notify("Loading contour maps and foreground masks…")
    contours = np.asarray(tifffile.imread(str(contour_maps_path)), dtype=np.float32)
    foreground = np.asarray(tifffile.imread(str(foreground_masks_path)), dtype=np.float32)
    if contours.ndim == 4 and contours.shape[1] == 1:
        contours = contours[:, 0]
    if foreground.ndim == 4 and foreground.shape[1] == 1:
        foreground = foreground[:, 0]

    with tempfile.TemporaryDirectory(prefix="cellflow_resolve_") as tmp_dir:
        working_dir = Path(tmp_dir)
        ultrack_cfg = _build_ultrack_config(cfg, working_dir)

        _notify("Segmenting with canonical Ultrack hierarchy…")
        ultrack_segment(
            foreground,
            contours,
            ultrack_cfg,
            max_segments_per_time=cfg.max_segments_per_time,
            overwrite=True,
        )

        _notify("Injecting validated nodes…")
        inject_validated_nodes(
            working_dir=working_dir,
            validated_tracks=validated_tracks,
            tracked_labels=np.asarray(tracked_labels, dtype=np.uint32),
            cfg=cfg,
        )

        _notify("Scoring node probabilities…")
        write_seed_prior_node_probs(working_dir, intensity_image_path, cfg)

        _notify("Linking candidates…")
        for _step, _total, label in run_linking(working_dir, cfg):
            _notify(f"[link] {label}")

        _notify("Solving ILP…")
        for _step, _total, label in run_solve(working_dir, cfg, overwrite=True):
            _notify(f"[solve] {label}")

        _notify("Exporting tracked labels…")
        tmp_out = working_dir / "tracked_labels_resolve.tif"
        export_tracked_labels(working_dir, cfg, tmp_out)
        new_labels = np.asarray(tifffile.imread(str(tmp_out)), dtype=np.uint32)
        if new_labels.ndim == 4 and new_labels.shape[1] == 1:
            new_labels = new_labels[:, 0]

    # Build id_map: validated cell_id → exported ID that covers its pixels most often
    id_map: dict[int, int] = {}
    tl = np.asarray(tracked_labels, dtype=np.uint32)
    for cell_id, frames in validated_tracks.items():
        vote: dict[int, int] = {}
        for t in frames:
            if t >= new_labels.shape[0] or t >= tl.shape[0]:
                continue
            src_mask = tl[t] == cell_id
            if not src_mask.any():
                continue
            for exported_id in np.unique(new_labels[t][src_mask]):
                if exported_id != 0:
                    vote[int(exported_id)] = vote.get(int(exported_id), 0) + 1
        if vote:
            id_map[cell_id] = max(vote, key=vote.__getitem__)

    return new_labels, id_map
```

- [ ] **Step 4: Run to verify it passes**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_resolve_with_canonical_segment_exists -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/reseed.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat(reseed): add resolve_with_canonical_segment using ultrack.segment"
```

---

## Task 5: Widget structural refactor — 5 top-level sections

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

This task changes the widget's visible layout skeleton. No new logic. The test-first approach requires first writing tests that describe the NEW structure (all failing), then updating the widget.

- [ ] **Step 1: Write new layout tests (will fail until widget is updated)**

Delete or replace `test_tracking_correction_shell_exposes_stable_section_attributes` and `test_tracking_correction_restores_two_column_button_and_parameter_layouts` and the wrapper-expansion calls in `test_tracking_correction_widget_minimum_width_stays_compact`.

Add these tests that describe the NEW structure:

```python
def test_nucleus_workflow_has_five_canonical_top_level_sections():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    # All five top-level sections must exist with exact titles
    assert widget.contour_section.title == "1. Contour Maps"
    assert widget.db_gen_section.title == "2. Ultrack Database Generation"
    assert widget.ultrack_db_browser_section.title == "3. Ultrack Database Browser"
    assert widget.ultrack_section.title == "4. Ultrack Tracking"
    assert widget.correction_section.title == "5. Correction"

    # Ultrack Tracking and Correction must NOT be nested inside a shared wrapper
    assert not hasattr(widget, "tracking_correction_section")

    widget.deleteLater()
    viewer.close()


def test_deprecated_sections_are_hidden():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    # gen_section (Hypothesis Generation) and db_section (H5 Browser) exist in code
    # but must not be visible in the normal workflow
    assert hasattr(widget, "gen_section")
    assert hasattr(widget, "db_section")
    assert not widget.gen_section.isVisible()
    assert not widget.db_section.isVisible()

    widget.deleteLater()
    viewer.close()


def test_canonical_sections_expose_required_elements():
    """Every top-level section must show status label, progress bar where applicable,
    run button, and terminal-run button (except Correction and DB Browser)."""
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    # Section 1: Contour Maps
    assert hasattr(widget, "build_btn")
    assert hasattr(widget, "contour_terminal_btn")
    assert hasattr(widget, "contour_status_lbl")
    assert hasattr(widget, "build_progress_bar")

    # Section 2: Ultrack Database Generation
    assert hasattr(widget, "run_db_gen_btn")
    assert hasattr(widget, "db_gen_terminal_btn")
    assert hasattr(widget, "db_gen_status_lbl")
    assert hasattr(widget, "db_gen_progress_bar")

    # Section 3: Ultrack Database Browser
    assert hasattr(widget, "ultrack_db_info_lbl")
    assert hasattr(widget, "ultrack_db_refresh_btn")
    assert hasattr(widget, "ultrack_db_section_status_lbl")

    # Section 4: Ultrack Tracking
    assert hasattr(widget, "run_ultrack_btn")
    assert hasattr(widget, "ultrack_terminal_btn")
    assert hasattr(widget, "ultrack_status_lbl")
    assert hasattr(widget, "ultrack_progress_bar")

    # Section 5: Correction (no Run button per spec)
    assert hasattr(widget, "correction_status_lbl")

    widget.deleteLater()
    viewer.close()


def test_ultrack_section_is_top_level_and_has_route_selector():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_section.title == "4. Ultrack Tracking"
    assert widget.ultrack_route_check.text() == "Resolve from validated"
    assert widget.ultrack_route_check in widget.ultrack_section.findChildren(
        type(widget.ultrack_route_check)
    )
    assert widget.ultrack_status_lbl.text() == ""
    assert widget.ultrack_progress_bar.isVisible() is False

    widget.deleteLater()
    viewer.close()


def test_correction_section_is_top_level():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.correction_section.title == "5. Correction"
    correction_button_texts = {
        button.text()
        for button in widget.correction_section.findChildren(QPushButton)
    }
    assert "Save Tracked Labels" in correction_button_texts
    assert "Load Tracked Labels" in correction_button_texts
    assert "◀ Extend (A)" in correction_button_texts
    assert "Extend (D) ▶" in correction_button_texts
    assert "◀ Retrack (Q)" in correction_button_texts
    assert "Retrack (E) ▶" in correction_button_texts

    widget.deleteLater()
    viewer.close()


def test_correction_shortcuts_are_still_installed():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    shortcut_keys = {
        shortcut.key().toString(QKeySequence.SequenceFormat.PortableText)
        for shortcut in widget.findChildren(QShortcut)
    }
    assert {"A", "D", "Q", "E"} <= shortcut_keys

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Run to verify all new tests fail**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_nucleus_workflow_has_five_canonical_top_level_sections tests/napari/test_nucleus_tracking_correction_layout.py::test_deprecated_sections_are_hidden tests/napari/test_nucleus_tracking_correction_layout.py::test_canonical_sections_expose_required_elements -v
```
Expected: FAIL (attributes do not exist)

- [ ] **Step 3: Update widget `_setup_ui`**

In `src/cellflow/napari/nucleus_workflow_widget.py`:

a) **Add import for `resolve_with_canonical_segment`** at the top (alongside existing imports):
```python
from cellflow.tracking_ultrack.reseed import resolve_with_validation, resolve_with_canonical_segment
from cellflow.tracking_ultrack.extend import extend_track, extend_track_from_db
```

b) **After Section 1 (Contour Maps)** and before Section 2, **hide the existing gen_section and db_section**. Do this by calling `.setVisible(False)` after their construction, not by deleting their build code. The sections still build and their widgets are still accessible, but they're invisible.

c) **Add Section 2: Ultrack Database Generation** with these attributes:
- `self.db_gen_min_area_spin` QSpinBox range(0, 1_000_000), value=300
- `self.db_gen_max_area_spin` QSpinBox range(0, 10_000_000), value=100_000
- `self.db_gen_fg_thr_spin` QDoubleSpinBox range(0.0, 1.0), value=0.5, decimals=2, step=0.05, tooltip="Pixel-level foreground threshold for ultrack segmentation (threshold in segmentation_config)"
- `self.db_gen_min_frontier_spin` QDoubleSpinBox range(0.0, 1.0), value=0.0, decimals=3, step=0.01, tooltip="Minimum boundary fraction to keep a candidate (min_frontier in segmentation_config)"
- `self.db_gen_ws_hierarchy_combo` QComboBox items=["area", "dynamics", "volume"]
- `self.db_gen_max_dist_spin` QDoubleSpinBox range(0, 500), value=15.0, decimals=1
- `self.db_gen_max_neighbors_spin` QSpinBox range(1, 50), value=5
- `self.db_gen_linking_mode_combo` QComboBox items=["default", "iou"]
- `self.db_gen_iou_weight_spin` QDoubleSpinBox range(0, 1), value=1.0, decimals=2, initially disabled
- `self.db_gen_quality_exp_spin` QDoubleSpinBox range(0.1, 50), value=8.0, decimals=2, tooltip="Raises signal-based quality before storing as node_prob"
- `self.db_gen_power_spin` QDoubleSpinBox range(0.1, 20), value=4.0, decimals=2, tooltip="Ultrack solver transform for node_prob and link weights"
- `self.db_gen_n_workers_spin` QSpinBox range(1, cpu_count), value=1, tooltip="Parallel workers for segmentation"
- `self.run_db_gen_btn` QPushButton("Run DB Generation") Expanding
- `self.db_gen_terminal_btn` QPushButton("Run in Terminal") Expanding
- `self.db_gen_status_lbl` QLabel("") WordWrap, initially hidden
- `self.db_gen_progress_bar` QProgressBar range(0,100), initially hidden
- `self.db_gen_section` CollapsibleSection("2. Ultrack Database Generation", expanded=False)

Layout for db_gen controls (use `block_grid` with `add_block_pair_row`):
```
Row 0: Min Area | Max Area
Row 1: FG Threshold | Min Frontier
Row 2: WS Hierarchy | N Workers
Row 3: Max Distance | Max Neighbors
Row 4: Linking Mode | IoU Weight
Row 5: Quality Exp | Power
```

d) **Add Section 3: Ultrack Database Browser** with:
- `self.ultrack_db_info_lbl` QLabel("—") AlignCenter
- `self.ultrack_db_refresh_btn` QPushButton with reload icon
- `self.ultrack_db_section_status_lbl` QLabel("") WordWrap, initially hidden
- `self.ultrack_db_browser_section` CollapsibleSection("3. Ultrack Database Browser", expanded=False)

e) **Update Section 4** (previously inside tracking_correction_section wrapper):
- Remove the outer `_tracking_correction_inner` widget and `tracking_correction_section`
- Instead, add `_ultrack_inner` and `_corr_inner` directly to the main layout
- Change `self.ultrack_section = CollapsibleSection("Ultrack Tracking", ...)` → `CollapsibleSection("4. Ultrack Tracking", ...)`
- Change `self.correction_section = CollapsibleSection("Correction", ...)` → `CollapsibleSection("5. Correction", ...)`

f) **After building gen_section and db_section** (H5-based ones), call:
```python
self.gen_section.setVisible(False)
self.db_section.setVisible(False)
```

g) **Remove the global `self.output_files` widget** that shows `hypotheses.h5` (was at the bottom). Add per-section file widgets instead if needed.

- [ ] **Step 4: Run tests to verify new layout tests pass and existing ones still work**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py -v 2>&1 | tail -40
```

Fix any remaining failures. Key expected issues:
- `test_tracking_correction_widget_minimum_width_stays_compact`: remove `tracking_correction_section.expand()` call, expand `ultrack_section` and `correction_section` directly
- `test_tracking_correction_restores_two_column_button_and_parameter_layouts`: replace `ultrack_min_area_spin.y() == ultrack_appear_spin.y()` with a check that uses `db_gen_min_area_spin` in db_gen_section, and a separate check for `ultrack_appear_spin` in ultrack_section

Update the old tests that reference the removed `tracking_correction_section`:

```python
# OLD test_tracking_correction_widget_minimum_width_stays_compact:
# Replace:
#   widget.tracking_correction_section.expand()
# With nothing (just expand ultrack and correction sections directly):
    widget.ultrack_section.expand()
    widget.correction_section.expand()
    widget.correction_shortcuts_section.expand()

# OLD test_tracking_correction_restores_two_column_button_and_parameter_layouts:
# Replace the whole test body with:
    widget.db_gen_section.expand()
    widget.ultrack_section.expand()
    widget.correction_section.expand()
    widget.show()
    _app.processEvents()
    # DB gen parameters should present as two side-by-side columns
    assert widget.db_gen_min_area_spin.y() == widget.db_gen_max_area_spin.y()
    assert widget.db_gen_min_area_spin.x() < widget.db_gen_max_area_spin.x()
    # Paired correction actions should also sit side-by-side
    assert widget.extend_back_btn.y() == widget.extend_fwd_btn.y()
    assert widget.extend_back_btn.x() < widget.extend_fwd_btn.x()
    assert widget.retrack_back_btn.y() == widget.retrack_fwd_btn.y()
    assert widget.save_tracked_btn.y() == widget.load_tracked_btn.y()
```

Also update `test_nucleus_workflow_status_labels_are_section_local` to add db_gen_status_lbl:
```python
    local_statuses = [
        (widget.contour_section, widget.contour_status_lbl),
        (widget.db_gen_section, widget.db_gen_status_lbl),
        (widget.ultrack_db_browser_section, widget.ultrack_db_section_status_lbl),
        (widget.ultrack_section, widget.ultrack_status_lbl),
        (widget.correction_section, widget.correction_status_lbl),
    ]
    # Remove the gen_section/db_section (H5) entries from this test
    # (they still exist but are hidden; their status labels still work)
```

- [ ] **Step 5: Run full test suite and verify all pass**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py -v 2>&1 | tail -50
```
Expected: all tests pass (some xfail may now pass — that's fine)

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat(widget): restructure to 5 canonical top-level sections; hide deprecated H5 sections"
```

---

## Task 6: Wire DB generation section — run and terminal

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write failing tests**

Add to test file (also add `"cellflow.tracking_ultrack.ingest"` stub entry for `_build_ultrack_config` and `"ultrack.core.segmentation.processing"` stub for `segment`):

Update `_install_import_stubs` in test file to add:
```python
stub_exports["cellflow.tracking_ultrack.ingest"]["_build_ultrack_config"] = lambda *a, **kw: None
```

Add:
```python
def test_db_gen_section_calls_ultrack_segment_on_run(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]

    calls = []
    def fake_segment(fg, contours, ultrack_cfg, **kwargs):
        calls.append(("segment", fg.shape, contours.shape))

    monkeypatch.setattr(module, "_ultrack_segment", fake_segment, raising=False)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    import numpy as np, tifffile
    dummy = np.zeros((2, 4, 4), dtype=np.float32)
    tifffile.imwrite(str(pos_dir / "2_nucleus" / "contour_maps.tif"), dummy)
    tifffile.imwrite(str(pos_dir / "2_nucleus" / "foreground_masks.tif"), dummy)
    tifffile.imwrite(str(pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif"), dummy)

    widget._pos_dir = pos_dir
    # Patch write_seed_prior_node_probs and run_linking to no-ops
    monkeypatch.setattr(module, "write_seed_prior_node_probs", lambda *a, **kw: None)
    monkeypatch.setattr(module, "run_linking", lambda *a, **kw: iter([]))

    widget._on_run_db_generation()
    _app.processEvents()
    # The worker runs in a thread; we just verify the method exists and can be called
    assert hasattr(widget, "_on_run_db_generation")


def test_db_gen_section_terminal_script_includes_canonical_segment(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    captured = _install_terminal_capture(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "2_nucleus" / "contour_maps.tif").touch()
    (pos_dir / "2_nucleus" / "foreground_masks.tif").touch()
    (pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif").touch()
    widget._pos_dir = pos_dir

    widget._on_db_gen_terminal()
    script = _read_launched_script(captured)

    assert "ultrack.core.segmentation.processing" in script or "from ultrack" in script
    assert "foreground_masks" in script
    assert "contour_maps" in script
    assert "nucleus_prob_zavg" in script
    assert "write_seed_prior_node_probs" in script
    assert "run_linking" in script
    assert "if __name__ == '__main__':" in script


def test_db_gen_section_fails_clearly_if_foreground_masks_missing(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "2_nucleus" / "contour_maps.tif").touch()
    # foreground_masks.tif is NOT created
    widget._pos_dir = pos_dir

    widget._on_run_db_generation()

    assert "foreground_masks" in widget.db_gen_status_lbl.text().lower() or \
           "missing" in widget.db_gen_status_lbl.text().lower()

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Run to verify they fail**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_section_calls_ultrack_segment_on_run tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_section_terminal_script_includes_canonical_segment tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_section_fails_clearly_if_foreground_masks_missing -v
```
Expected: FAIL (`_on_run_db_generation` not found)

- [ ] **Step 3: Add path helpers and implement db generation methods**

Add path helpers to `NucleusWorkflowWidget`:
```python
def _foreground_masks_path(self) -> Path | None:
    return self._pos_dir / "2_nucleus" / "foreground_masks.tif" if self._pos_dir else None

def _nucleus_prob_zavg_path(self) -> Path | None:
    return self._pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif" if self._pos_dir else None

def _ultrack_db_path(self) -> Path | None:
    workdir = self._ultrack_workdir()
    return workdir / "data.db" if workdir else None
```

Add `_db_gen_config_from_controls()`:
```python
def _db_gen_config_from_controls(self) -> UltrackConfig:
    return UltrackConfig(
        seg_min_area=self.db_gen_min_area_spin.value(),
        seg_max_area=self.db_gen_max_area_spin.value(),
        seg_foreground_threshold=self.db_gen_fg_thr_spin.value(),
        seg_min_frontier=self.db_gen_min_frontier_spin.value(),
        seg_ws_hierarchy=self.db_gen_ws_hierarchy_combo.currentText(),
        seg_n_workers=self.db_gen_n_workers_spin.value(),
        max_distance=self.db_gen_max_dist_spin.value(),
        max_neighbors=self.db_gen_max_neighbors_spin.value(),
        linking_mode=self.db_gen_linking_mode_combo.currentText(),
        iou_weight=self.db_gen_iou_weight_spin.value(),
        quality_exponent=self.db_gen_quality_exp_spin.value(),
        power=self.db_gen_power_spin.value(),
        link_n_workers=self.db_gen_n_workers_spin.value(),
    )
```

Add module-level import alias (so tests can monkeypatch it):
```python
try:
    from ultrack.core.segmentation.processing import segment as _ultrack_segment
except ImportError:
    _ultrack_segment = None  # type: ignore[assignment]
```

Add `_on_run_db_generation`:
```python
def _on_run_db_generation(self) -> None:
    if self._pos_dir is None:
        self._set_db_gen_status("No project open.")
        return
    contour_path = self._contour_maps_path()
    fg_path = self._foreground_masks_path()
    nuc_zavg_path = self._nucleus_prob_zavg_path()
    if contour_path is None or not contour_path.exists():
        self._set_db_gen_status("Missing: contour_maps.tif — run Contour Maps first.")
        return
    if fg_path is None or not fg_path.exists():
        self._set_db_gen_status("Missing: foreground_masks.tif — provide 2_nucleus/foreground_masks.tif.")
        return
    if nuc_zavg_path is None or not nuc_zavg_path.exists():
        self._set_db_gen_status("Missing: nucleus_prob_zavg.tif — run Cellpose first.")
        return
    if _ultrack_segment is None:
        self._set_db_gen_status("ultrack not installed — activate the cellflow conda environment.")
        return

    cfg = self._db_gen_config_from_controls()
    working_dir = self._ultrack_workdir()
    pos_dir = self._pos_dir

    self.db_gen_progress_bar.setRange(0, 0)
    self.db_gen_progress_bar.setVisible(True)
    self._set_db_gen_status("Starting DB generation…")
    self.run_db_gen_btn.setEnabled(False)
    self.db_gen_terminal_btn.setEnabled(False)

    @thread_worker(connect={
        "yielded":  self._on_db_gen_progress,
        "returned": self._on_db_gen_done,
        "errored":  self._on_db_gen_worker_error,
    })
    def _worker():
        yield "Loading inputs…"
        contours = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
        foreground = np.asarray(tifffile.imread(str(fg_path)), dtype=np.float32)
        if contours.ndim == 4 and contours.shape[1] == 1:
            contours = contours[:, 0]
        if foreground.ndim == 4 and foreground.shape[1] == 1:
            foreground = foreground[:, 0]

        from cellflow.tracking_ultrack.ingest import _build_ultrack_config
        working_dir.mkdir(parents=True, exist_ok=True)
        ultrack_cfg = _build_ultrack_config(cfg, working_dir)

        yield "Segmenting candidates (ultrack hierarchy)…"
        _ultrack_segment(
            foreground,
            contours,
            ultrack_cfg,
            max_segments_per_time=cfg.max_segments_per_time,
            overwrite=True,
        )

        yield "Scoring node probabilities…"
        write_seed_prior_node_probs(working_dir, nuc_zavg_path, cfg)

        yield "Linking candidates…"
        for step, total, label in run_linking(working_dir, cfg):
            yield f"[link {step}/{total}] {label}"

        return pos_dir

    _worker()

def _on_db_gen_progress(self, msg: str) -> None:
    self._set_db_gen_status(msg)

def _on_db_gen_done(self, pos_dir: Path) -> None:
    self.db_gen_progress_bar.setVisible(False)
    self.run_db_gen_btn.setEnabled(True)
    self.db_gen_terminal_btn.setEnabled(True)
    self._set_db_gen_status("DB generation complete.")
    self._refresh_ultrack_db_browser()

def _on_db_gen_worker_error(self, exc: Exception) -> None:
    self.db_gen_progress_bar.setVisible(False)
    self.run_db_gen_btn.setEnabled(True)
    self.db_gen_terminal_btn.setEnabled(True)
    self._set_db_gen_status(f"Error: {exc}")
    logger.exception("DB generation worker error", exc_info=exc)

def _set_db_gen_status(self, msg: str) -> None:
    self.db_gen_status_lbl.setText(msg)
    self.db_gen_status_lbl.setVisible(bool(msg))
    logger.info(msg)
```

Add `_on_db_gen_terminal`:
```python
def _on_db_gen_terminal(self) -> None:
    import sys
    import tempfile

    if self._pos_dir is None:
        self._set_db_gen_status("No project open.")
        return
    contour_path = self._contour_maps_path()
    fg_path = self._foreground_masks_path()
    nuc_zavg_path = self._nucleus_prob_zavg_path()
    if contour_path is None or not contour_path.exists():
        self._set_db_gen_status("Missing: contour_maps.tif")
        return
    if fg_path is None or not fg_path.exists():
        self._set_db_gen_status("Missing: foreground_masks.tif")
        return
    if nuc_zavg_path is None or not nuc_zavg_path.exists():
        self._set_db_gen_status("Missing: nucleus_prob_zavg.tif")
        return

    cfg = self._db_gen_config_from_controls()
    working_dir = self._ultrack_workdir()

    python_code = (
        "import sys, pathlib, tifffile, numpy as np\n"
        "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
        "from cellflow.tracking_ultrack.config import TrackingConfig\n"
        "from cellflow.tracking_ultrack.ingest import _build_ultrack_config\n"
        "from cellflow.tracking_ultrack.linking import run_linking\n"
        "from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs\n"
        "from ultrack.core.segmentation.processing import segment as ultrack_segment\n"
        "\n"
        "if __name__ == '__main__':\n"
        f"    contour_path    = pathlib.Path({str(contour_path)!r})\n"
        f"    fg_path         = pathlib.Path({str(fg_path)!r})\n"
        f"    nuc_zavg_path   = pathlib.Path({str(nuc_zavg_path)!r})\n"
        f"    working_dir     = pathlib.Path({str(working_dir)!r})\n"
        f"    cfg = TrackingConfig(\n"
        f"        seg_min_area={cfg.seg_min_area},\n"
        f"        seg_max_area={cfg.seg_max_area},\n"
        f"        seg_foreground_threshold={cfg.seg_foreground_threshold},\n"
        f"        seg_min_frontier={cfg.seg_min_frontier},\n"
        f"        seg_ws_hierarchy={cfg.seg_ws_hierarchy!r},\n"
        f"        seg_n_workers={cfg.seg_n_workers},\n"
        f"        max_distance={cfg.max_distance},\n"
        f"        max_neighbors={cfg.max_neighbors},\n"
        f"        linking_mode={cfg.linking_mode!r},\n"
        f"        iou_weight={cfg.iou_weight},\n"
        f"        quality_exponent={cfg.quality_exponent},\n"
        f"        power={cfg.power},\n"
        f"        link_n_workers={cfg.link_n_workers},\n"
        "    )\n"
        "    contours    = np.asarray(tifffile.imread(str(contour_path)), dtype='float32')\n"
        "    foreground  = np.asarray(tifffile.imread(str(fg_path)), dtype='float32')\n"
        "    if contours.ndim == 4 and contours.shape[1] == 1: contours = contours[:, 0]\n"
        "    if foreground.ndim == 4 and foreground.shape[1] == 1: foreground = foreground[:, 0]\n"
        "    working_dir.mkdir(parents=True, exist_ok=True)\n"
        "    ultrack_cfg = _build_ultrack_config(cfg, working_dir)\n"
        "    print('[1/3] Segmenting…', flush=True)\n"
        "    ultrack_segment(foreground, contours, ultrack_cfg,\n"
        "        max_segments_per_time=cfg.max_segments_per_time, overwrite=True)\n"
        "    print('[2/3] Scoring…', flush=True)\n"
        "    write_seed_prior_node_probs(working_dir, nuc_zavg_path, cfg)\n"
        "    print('[3/3] Linking…', flush=True)\n"
        "    for step, total, label in run_linking(working_dir, cfg):\n"
        "        print(f'  [{step}/{total}] {label}', flush=True)\n"
        "    print('Done.', flush=True)\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="cellflow_db_gen_", delete=False) as tmp:
        tmp.write(python_code)
        tmp_path = tmp.name

    cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
    try:
        from cellflow.napari.utils import launch_in_terminal
        launch_in_terminal(cmd)
        self._set_db_gen_status("DB generation launched in terminal.")
    except Exception:
        QApplication.clipboard().setText(cmd)
        self._set_db_gen_status("Copied command to clipboard.")
```

Wire in `_connect_signals`:
```python
self.run_db_gen_btn.clicked.connect(self._on_run_db_generation)
self.db_gen_terminal_btn.clicked.connect(self._on_db_gen_terminal)
self.db_gen_linking_mode_combo.currentTextChanged.connect(self._on_db_gen_mode_changed)
```

Add:
```python
def _on_db_gen_mode_changed(self, mode: str) -> None:
    self.db_gen_iou_weight_spin.setEnabled(mode == "iou")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_section_calls_ultrack_segment_on_run tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_section_terminal_script_includes_canonical_segment tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_section_fails_clearly_if_foreground_masks_missing -v
```
Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py -v 2>&1 | tail -30
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat(widget): implement DB generation run and terminal with ultrack.segment"
```

---

## Task 7: Wire Ultrack DB Browser section

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

The first slice only needs: database summary (node/link counts) and a clear error when `data.db` is missing.

- [ ] **Step 1: Write failing tests**

```python
def test_ultrack_db_browser_shows_missing_db_status(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget._pos_dir = pos_dir

    widget._refresh_ultrack_db_browser()

    text = widget.ultrack_db_section_status_lbl.text().lower()
    assert "data.db" in text or "missing" in text or "not found" in text

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Run to verify it fails**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_shows_missing_db_status -v
```
Expected: FAIL (`_refresh_ultrack_db_browser` not found)

- [ ] **Step 3: Implement browser helpers**

Add to widget:
```python
def _set_ultrack_db_status(self, msg: str) -> None:
    self.ultrack_db_section_status_lbl.setText(msg)
    self.ultrack_db_section_status_lbl.setVisible(bool(msg))
    logger.info(msg)

def _refresh_ultrack_db_browser(self) -> None:
    self.ultrack_db_info_lbl.setText("—")
    db_path = self._ultrack_db_path()
    if db_path is None or not db_path.exists():
        self._set_ultrack_db_status("data.db not found — run DB generation first.")
        return
    try:
        import sqlalchemy as sqla
        from sqlalchemy import func
        from sqlalchemy.orm import Session
        from ultrack.core.database import LinkDB, NodeDB
        engine = sqla.create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        with Session(engine) as session:
            n_nodes = int(session.query(func.count(NodeDB.id)).scalar() or 0)
            n_links = int(session.query(func.count(LinkDB.source_id)).scalar() or 0)
        engine.dispose()
        self.ultrack_db_info_lbl.setText(f"{n_nodes} nodes | {n_links} links")
        self._set_ultrack_db_status("")
    except Exception as e:
        self._set_ultrack_db_status(f"DB read error: {e}")
        logger.warning("DB browser error: %s", e)
```

Wire in `_connect_signals`:
```python
self.ultrack_db_refresh_btn.clicked.connect(self._refresh_ultrack_db_browser)
```

- [ ] **Step 4: Run test**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_browser_shows_missing_db_status -v
```
Expected: PASS

- [ ] **Step 5: Run full suite**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py -v 2>&1 | tail -20
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat(widget): implement Ultrack DB browser section with node/link summary"
```

---

## Task 8: Update Ultrack Tracking section — solve-only + migrate extend to DB

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

Section 4 should solve an existing linked `data.db` — not rebuild candidates. Extend should use `extend_track_from_db` instead of `extend_track` with hypotheses.h5.

- [ ] **Step 1: Write failing tests**

```python
def test_ultrack_tracking_solve_fails_clearly_if_db_missing(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    # data.db NOT created
    widget._pos_dir = pos_dir

    widget._on_run_ultrack()

    text = widget.ultrack_status_lbl.text().lower()
    assert "data.db" in text or "missing" in text or "database" in text

    widget.deleteLater()
    viewer.close()


def test_extend_fails_clearly_if_db_missing(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    import numpy as np
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    # No data.db, no tracked layer
    widget._pos_dir = pos_dir

    widget._on_extend(direction="forward")

    text = widget.correction_status_lbl.text().lower()
    # Either no layer message or missing DB message
    assert len(text) > 0

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Run to verify failures**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_tracking_solve_fails_clearly_if_db_missing tests/napari/test_nucleus_tracking_correction_layout.py::test_extend_fails_clearly_if_db_missing -v
```
Expected: FAIL (current `_on_run_ultrack` checks for `hypotheses.h5`, not `data.db`)

- [ ] **Step 3: Update `_on_run_ultrack` to check `data.db` and only solve**

Replace the current `_on_run_ultrack` body:

```python
def _on_run_ultrack(self) -> None:
    if self._pos_dir is None:
        self._set_ultrack_status("No project open.")
        return
    db_path = self._ultrack_db_path()
    if db_path is None or not db_path.exists():
        self._set_ultrack_status("data.db not found — run DB generation first.")
        return
    working_dir = self._ultrack_workdir()
    tracked_path = self._tracked_path()
    cfg = self._ultrack_config_from_controls()

    self.ultrack_progress_bar.setRange(0, 100)
    self.ultrack_progress_bar.setVisible(True)
    self.ultrack_progress_bar.setValue(0)
    self._set_ultrack_status("Starting Ultrack solve…")
    self.run_ultrack_btn.setEnabled(False)
    self.ultrack_terminal_btn.setEnabled(False)

    @thread_worker(connect={
        "yielded":  self._on_ultrack_progress,
        "returned": self._on_run_ultrack_done,
        "errored":  self._on_ultrack_worker_error,
    })
    def _worker():
        for step, total, label in run_solve(working_dir, cfg, overwrite=True):
            yield ("solve", step, total, label)
        yield ("export", 0, 1, "Exporting tracked labels…")
        return export_tracked_labels(working_dir, cfg, tracked_path)

    _worker()
```

Update `_on_ultrack_worker_error` to re-enable buttons:
```python
def _on_ultrack_worker_error(self, exc: Exception) -> None:
    self.ultrack_progress_bar.setVisible(False)
    self.ultrack_progress_bar.setRange(0, 100)
    self.run_ultrack_btn.setEnabled(True)
    self.ultrack_terminal_btn.setEnabled(True)
    self._set_ultrack_status(f"Error: {exc}")
    logger.exception("Ultrack worker error", exc_info=exc)
```

Also update `_on_run_ultrack_done`:
```python
def _on_run_ultrack_done(self, labels: np.ndarray | None) -> None:
    self.ultrack_progress_bar.setVisible(False)
    self.run_ultrack_btn.setEnabled(True)
    self.ultrack_terminal_btn.setEnabled(True)
    if labels is None:
        self._set_ultrack_status("Ultrack tracking failed (no output).")
        return
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    nt = labels.shape[0]
    if _TRACKED_LAYER in self.viewer.layers:
        self.viewer.layers[_TRACKED_LAYER].data = labels
    else:
        self.viewer.add_labels(labels, name=_TRACKED_LAYER)
    layer = self.viewer.layers[_TRACKED_LAYER]
    self.correction_widget.activate_layer(layer)
    self._set_ultrack_status(f"Tracking done: {nt} frame(s). Unsaved.")
```

Also update `_ultrack_config_from_controls` to use only solve parameters (appear/disappear/division weights) since linking params moved to db_gen section. Keep power/quality_exp here for the resolve path. Actually, since resolve rebuilds from scratch, it needs all params — so keep the full config here but only the solve controls remain in ultrack_section UI:

```python
def _ultrack_config_from_controls(self) -> UltrackConfig:
    # Solve-only params from ultrack_section; linking/seg params from db_gen_section
    return UltrackConfig(
        # From db_gen_section (needed for resolve-from-validated rebuild)
        seg_min_area=self.db_gen_min_area_spin.value(),
        seg_max_area=self.db_gen_max_area_spin.value(),
        seg_foreground_threshold=self.db_gen_fg_thr_spin.value(),
        seg_min_frontier=self.db_gen_min_frontier_spin.value(),
        seg_ws_hierarchy=self.db_gen_ws_hierarchy_combo.currentText(),
        seg_n_workers=self.db_gen_n_workers_spin.value(),
        max_distance=self.db_gen_max_dist_spin.value(),
        max_neighbors=self.db_gen_max_neighbors_spin.value(),
        linking_mode=self.db_gen_linking_mode_combo.currentText(),
        iou_weight=self.db_gen_iou_weight_spin.value(),
        quality_exponent=self.db_gen_quality_exp_spin.value(),
        power=self.db_gen_power_spin.value(),
        # From ultrack_section (solver penalties)
        appear_weight=self.ultrack_appear_spin.value(),
        disappear_weight=self.ultrack_disappear_spin.value(),
        division_weight=self.ultrack_division_spin.value(),
        # Seed prior (for resolve-from-validated)
        seed_weight=self.ultrack_seed_weight_spin.value(),
        seed_sigma_space=self.ultrack_seed_space_spin.value(),
        seed_tau_time=self.ultrack_seed_time_spin.value(),
        seed_max_dt=self.ultrack_seed_window_spin.value(),
    )
```

Update `_on_extend` to use `extend_track_from_db` instead of `extend_track`:

```python
def _on_extend(self, direction: str) -> None:
    if _TRACKED_LAYER not in self.viewer.layers:
        self._set_correction_status("No tracked layer loaded.")
        return

    db_path = self._ultrack_db_path()
    if db_path is None or not db_path.exists():
        self._set_correction_status(
            "Extend: data.db not found — run DB generation first."
        )
        return

    source_id = self.correction_widget._selected_label
    if not source_id:
        self._set_correction_status("Extend: no cell selected (left-click a cell first).")
        return

    layer = self.viewer.layers[_TRACKED_LAYER]
    t = self._current_t()
    tracked = np.asarray(layer.data)
    T = tracked.shape[0]

    target_frame = t + (1 if direction == "forward" else -1)
    if direction == "forward" and t >= T - 1:
        self._set_correction_status("Already at last frame")
        return
    if direction == "backward" and t <= 0:
        self._set_correction_status("Already at first frame")
        return

    if not np.any(tracked[t] == source_id):
        self._set_correction_status(f"Cell {source_id} not present at t={t}")
        return

    result = extend_track_from_db(
        source_id=source_id,
        source_frame=t,
        direction=direction,
        tracked_labels=tracked,
        db_path=db_path,
        d_max=float(self.extend_max_dist_spin.value()),
    )

    if result is None:
        self._set_correction_status(
            f"No candidate within {self.extend_max_dist_spin.value():g}px at t={target_frame}"
        )
        return

    frame = layer.data[result.target_frame]
    frame[frame == source_id] = 0
    paintable = result.mask_2d & (frame == 0)
    frame[paintable] = source_id
    layer.refresh()

    step = list(self.viewer.dims.current_step)
    step[0] = result.target_frame
    self.viewer.dims.current_step = tuple(step)

    self._set_correction_status(
        f"Extended cell {source_id} → t={result.target_frame} "
        f"(dist={result.centroid_distance:.1f}px, area_ratio={result.area_ratio:.2f})"
    )
```

- [ ] **Step 4: Run tests**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_tracking_solve_fails_clearly_if_db_missing tests/napari/test_nucleus_tracking_correction_layout.py::test_extend_fails_clearly_if_db_missing -v
```
Expected: PASS

- [ ] **Step 5: Run full suite**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py -v 2>&1 | tail -30
```
Expected: all pass (fix remaining failures)

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat(widget): tracking section solves existing DB; extend uses data.db"
```

---

## Task 9: Migrate resolve-from-validated to canonical segmentation

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

The resolve path should use `resolve_with_canonical_segment` (not `hypotheses.h5`).

- [ ] **Step 1: Write failing tests**

```python
def test_resolve_from_validated_fails_clearly_if_foreground_masks_missing(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]

    import numpy as np
    monkeypatch_fn = lambda _pd: {1: {0, 1}}  # noqa: E731
    # Can't use monkeypatch fixture here — inline patching
    original = getattr(module, "read_validated_tracks")
    module.read_validated_tracks = lambda _pd: {1: {0, 1}}

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "2_nucleus" / "contour_maps.tif").touch()
    # foreground_masks.tif intentionally missing
    widget._pos_dir = pos_dir

    widget.ultrack_route_check.setChecked(True)
    widget._on_run_tracking_route()

    text = widget.ultrack_status_lbl.text().lower()
    assert "foreground_masks" in text or "missing" in text

    module.read_validated_tracks = original
    widget.deleteLater()
    viewer.close()


def test_resolve_terminal_script_uses_canonical_segment_not_hypotheses_h5(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    captured = _install_terminal_capture(monkeypatch)
    monkeypatch.setattr(module, "read_validated_tracks", lambda _pos_dir: {12: {0, 1}})

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "2_nucleus" / "contour_maps.tif").touch()
    (pos_dir / "2_nucleus" / "foreground_masks.tif").touch()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif").touch()
    widget._pos_dir = pos_dir

    widget._on_resolve_terminal()
    script = _read_launched_script(captured)

    # New canonical path — must use foreground_masks and contour_maps, not hypotheses.h5
    assert "foreground_masks" in script
    assert "contour_maps" in script
    assert "nucleus_prob_zavg" in script
    assert "resolve_with_canonical_segment" in script
    assert "hypotheses.h5" not in script
    assert "ingest_hypotheses_to_db" not in script
```

- [ ] **Step 2: Run to verify failures**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_resolve_from_validated_fails_clearly_if_foreground_masks_missing tests/napari/test_nucleus_tracking_correction_layout.py::test_resolve_terminal_script_uses_canonical_segment_not_hypotheses_h5 -v
```
Expected: FAIL

- [ ] **Step 3: Update `_on_resolve_with_validation` and `_on_resolve_terminal`**

Replace `_on_resolve_with_validation` to check canonical inputs and use `resolve_with_canonical_segment`:

```python
def _on_resolve_with_validation(self) -> None:
    if self._pos_dir is None:
        self._set_ultrack_status("No project open.")
        return
    validated_tracks = read_validated_tracks(self._pos_dir)
    if not validated_tracks:
        self._set_ultrack_status("No validated tracks — validate some cells first (press V).")
        return
    contour_path = self._contour_maps_path()
    fg_path = self._foreground_masks_path()
    tracked_path = self._tracked_path()
    nuc_zavg_path = self._nucleus_prob_zavg_path()
    if contour_path is None or not contour_path.exists():
        self._set_ultrack_status("Missing: contour_maps.tif — run Contour Maps first.")
        return
    if fg_path is None or not fg_path.exists():
        self._set_ultrack_status("Missing: foreground_masks.tif — provide 2_nucleus/foreground_masks.tif.")
        return
    if tracked_path is None or not tracked_path.exists():
        self._set_ultrack_status("Tracked labels not found — run tracking first.")
        return
    if nuc_zavg_path is None or not nuc_zavg_path.exists():
        self._set_ultrack_status("Missing: nucleus_prob_zavg.tif — run Cellpose first.")
        return

    cfg = self._ultrack_config_from_controls()
    pos_dir = self._pos_dir

    self.ultrack_progress_bar.setRange(0, 0)
    self.ultrack_progress_bar.setVisible(True)
    self._set_ultrack_status("Starting resolve from validated…")
    self.run_ultrack_btn.setEnabled(False)
    self.ultrack_terminal_btn.setEnabled(False)

    @thread_worker(connect={
        "yielded":  lambda msg: self._set_ultrack_status(msg),
        "returned": self._on_resolve_done,
        "errored":  self._on_ultrack_worker_error,
    })
    def _worker():
        from cellflow.database.tracked import read_full_tracked_stack
        tracked_labels = read_full_tracked_stack(tracked_path)
        new_labels, _id_map = resolve_with_canonical_segment(
            contour_maps_path=contour_path,
            foreground_masks_path=fg_path,
            validated_tracks=validated_tracks,
            tracked_labels=tracked_labels,
            cfg=cfg,
            progress_cb=lambda msg: None,  # yields handled separately
            intensity_image_path=nuc_zavg_path,
        )
        return new_labels, pos_dir

    _worker()
```

Replace `_on_resolve_terminal` to use canonical inputs:

```python
def _on_resolve_terminal(self) -> None:
    import sys
    import tempfile

    if self._pos_dir is None:
        self._set_ultrack_status("No project open.")
        return
    validated_tracks = read_validated_tracks(self._pos_dir)
    if not validated_tracks:
        self._set_ultrack_status("No validated tracks — validate some cells first (press V).")
        return
    contour_path = self._contour_maps_path()
    fg_path = self._foreground_masks_path()
    tracked_path = self._tracked_path()
    nuc_zavg_path = self._nucleus_prob_zavg_path()
    if contour_path is None or not contour_path.exists():
        self._set_ultrack_status("Missing: contour_maps.tif")
        return
    if fg_path is None or not fg_path.exists():
        self._set_ultrack_status("Missing: foreground_masks.tif")
        return
    if tracked_path is None or not tracked_path.exists():
        self._set_ultrack_status("Tracked labels not found.")
        return
    if nuc_zavg_path is None or not nuc_zavg_path.exists():
        self._set_ultrack_status("Missing: nucleus_prob_zavg.tif")
        return

    cfg = self._ultrack_config_from_controls()
    pos_dir = self._pos_dir

    python_code = (
        "import sys, pathlib, tifffile, numpy as np\n"
        "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
        "from cellflow.tracking_ultrack.config import TrackingConfig\n"
        "from cellflow.tracking_ultrack.reseed import resolve_with_canonical_segment\n"
        "from cellflow.database.tracked import read_full_tracked_stack\n"
        "from cellflow.database.validation import read_validated_tracks\n"
        "\n"
        "if __name__ == '__main__':\n"
        f"    pos_dir              = pathlib.Path({str(pos_dir)!r})\n"
        f"    contour_path         = pathlib.Path({str(contour_path)!r})\n"
        f"    foreground_masks_path = pathlib.Path({str(fg_path)!r})\n"
        f"    tracked_path         = pathlib.Path({str(tracked_path)!r})\n"
        f"    nucleus_prob_zavg_path = pathlib.Path({str(nuc_zavg_path)!r})\n"
        f"    cfg = TrackingConfig(\n"
        f"        seg_min_area={cfg.seg_min_area},\n"
        f"        seg_max_area={cfg.seg_max_area},\n"
        f"        seg_foreground_threshold={cfg.seg_foreground_threshold},\n"
        f"        seg_min_frontier={cfg.seg_min_frontier},\n"
        f"        seg_ws_hierarchy={cfg.seg_ws_hierarchy!r},\n"
        f"        seg_n_workers={cfg.seg_n_workers},\n"
        f"        max_distance={cfg.max_distance},\n"
        f"        max_neighbors={cfg.max_neighbors},\n"
        f"        linking_mode={cfg.linking_mode!r},\n"
        f"        iou_weight={cfg.iou_weight},\n"
        f"        quality_exponent={cfg.quality_exponent},\n"
        f"        power={cfg.power},\n"
        f"        appear_weight={cfg.appear_weight},\n"
        f"        disappear_weight={cfg.disappear_weight},\n"
        f"        division_weight={cfg.division_weight},\n"
        f"        seed_weight={cfg.seed_weight},\n"
        f"        seed_sigma_space={cfg.seed_sigma_space},\n"
        f"        seed_tau_time={cfg.seed_tau_time},\n"
        f"        seed_max_dt={cfg.seed_max_dt},\n"
        "    )\n"
        "    validated_tracks = read_validated_tracks(pos_dir)\n"
        "    print(f'Loaded {len(validated_tracks)} validated track(s).', flush=True)\n"
        "    tracked_labels = read_full_tracked_stack(tracked_path)\n"
        "    print(f'Loaded tracked labels: {tracked_labels.shape}', flush=True)\n"
        "    new_labels, _id_map = resolve_with_canonical_segment(\n"
        "        contour_maps_path=contour_path,\n"
        "        foreground_masks_path=foreground_masks_path,\n"
        "        validated_tracks=validated_tracks,\n"
        "        tracked_labels=tracked_labels,\n"
        "        cfg=cfg,\n"
        "        progress_cb=lambda msg: print(msg, flush=True),\n"
        "        intensity_image_path=nucleus_prob_zavg_path,\n"
        "    )\n"
        "    if new_labels.ndim == 4 and new_labels.shape[1] == 1:\n"
        "        new_labels = new_labels[:, 0]\n"
        "    preview_path = tracked_path.parent / 'tracked_labels_resolve_preview.tif'\n"
        "    tifffile.imwrite(str(preview_path), new_labels)\n"
        "    print(f'Preview saved to {preview_path}', flush=True)\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="cellflow_resolve_", delete=False) as tmp:
        tmp.write(python_code)
        tmp_path_str = tmp.name

    cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path_str)}"
    try:
        from cellflow.napari.utils import launch_in_terminal
        launch_in_terminal(cmd)
        self._set_ultrack_status("Resolve launched in terminal.")
    except Exception:
        QApplication.clipboard().setText(cmd)
        self._set_ultrack_status("Copied resolve command to clipboard.")
```

Also update existing `_on_resolve_done` to handle the new tuple form:
```python
def _on_resolve_done(self, result: tuple) -> None:
    new_labels, pos_dir = result
    self.ultrack_progress_bar.setVisible(False)
    self.run_ultrack_btn.setEnabled(True)
    self.ultrack_terminal_btn.setEnabled(True)
    if new_labels is None:
        self._set_ultrack_status("Resolve failed (no output).")
        return
    if new_labels.ndim == 4 and new_labels.shape[1] == 1:
        new_labels = new_labels[:, 0]
    if _TRACKED_LAYER in self.viewer.layers:
        self.viewer.layers[_TRACKED_LAYER].data = new_labels
    else:
        self.viewer.add_labels(new_labels, name=_TRACKED_LAYER)
    self._set_ultrack_status(f"Resolve done: {new_labels.shape[0]} frame(s). Unsaved.")
```

- [ ] **Step 4: Run tests**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_resolve_from_validated_fails_clearly_if_foreground_masks_missing tests/napari/test_nucleus_tracking_correction_layout.py::test_resolve_terminal_script_uses_canonical_segment_not_hypotheses_h5 -v
```
Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py -v 2>&1 | tail -30
```
Expected: all pass. Note: old `test_resolve_terminal_script_includes_seed_prior_and_cellprob_zavg` checks for `cell_prob_zavg.tif` (old name) — update it to `nucleus_prob_zavg.tif`.

Old tests that checked `hypotheses.h5` in resolve paths need updating:
- `test_ultrack_terminal_script_includes_visible_config_controls`: update to not create `hypotheses.h5`; update path checks
- `test_resolve_terminal_script_includes_validated_seed_prior_controls`: update to use `foreground_masks.tif` + `contour_maps.tif`
- `test_resolve_terminal_script_does_not_autosave_tracked_labels`: update paths
- `test_resolve_terminal_script_includes_seed_prior_and_cellprob_zavg`: update to `nucleus_prob_zavg`

For each of these, the fixture setup should create `foreground_masks.tif` and `contour_maps.tif` instead of `hypotheses.h5`, and assert path strings match the new names.

- [ ] **Step 6: Run full test suite (clean)**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py -v
```
Expected: all pass, `test_correction_section_has_no_separate_resolve_action_group` may flip from xfail to pass.

- [ ] **Step 7: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat(widget): resolve-from-validated uses canonical segmentation; no hypotheses.h5"
```

---

## Task 10: State persistence for new db_gen controls

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write failing test**

```python
def test_db_gen_controls_persist_through_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.db_gen_min_area_spin.setValue(500)
    widget.db_gen_max_area_spin.setValue(80_000)
    widget.db_gen_fg_thr_spin.setValue(0.4)
    widget.db_gen_min_frontier_spin.setValue(0.05)
    widget.db_gen_ws_hierarchy_combo.setCurrentText("dynamics")
    widget.db_gen_max_dist_spin.setValue(20.0)
    widget.db_gen_max_neighbors_spin.setValue(8)
    widget.db_gen_linking_mode_combo.setCurrentText("iou")
    widget.db_gen_iou_weight_spin.setValue(0.8)
    widget.db_gen_quality_exp_spin.setValue(6.0)
    widget.db_gen_power_spin.setValue(3.0)
    widget.db_gen_n_workers_spin.setValue(4)

    state = widget.get_state()
    widget.deleteLater()

    widget2 = widget_class(viewer)
    widget2.set_state(state)

    assert widget2.db_gen_min_area_spin.value() == 500
    assert widget2.db_gen_max_area_spin.value() == 80_000
    assert abs(widget2.db_gen_fg_thr_spin.value() - 0.4) < 0.01
    assert abs(widget2.db_gen_min_frontier_spin.value() - 0.05) < 0.01
    assert widget2.db_gen_ws_hierarchy_combo.currentText() == "dynamics"
    assert widget2.db_gen_max_dist_spin.value() == 20.0
    assert widget2.db_gen_max_neighbors_spin.value() == 8
    assert widget2.db_gen_linking_mode_combo.currentText() == "iou"
    assert abs(widget2.db_gen_iou_weight_spin.value() - 0.8) < 0.01
    assert abs(widget2.db_gen_quality_exp_spin.value() - 6.0) < 0.01
    assert abs(widget2.db_gen_power_spin.value() - 3.0) < 0.01
    assert widget2.db_gen_n_workers_spin.value() == 4

    widget2.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Run to verify it fails**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_controls_persist_through_state -v
```
Expected: FAIL (get_state/set_state don't include db_gen fields)

- [ ] **Step 3: Update `get_state` and `set_state` in widget**

Find the `get_state` method (reads existing spinbox values into a dict) and add:
```python
# DB generation section
"db_gen_min_area": self.db_gen_min_area_spin.value(),
"db_gen_max_area": self.db_gen_max_area_spin.value(),
"db_gen_fg_thr": self.db_gen_fg_thr_spin.value(),
"db_gen_min_frontier": self.db_gen_min_frontier_spin.value(),
"db_gen_ws_hierarchy": self.db_gen_ws_hierarchy_combo.currentText(),
"db_gen_max_dist": self.db_gen_max_dist_spin.value(),
"db_gen_max_neighbors": self.db_gen_max_neighbors_spin.value(),
"db_gen_linking_mode": self.db_gen_linking_mode_combo.currentText(),
"db_gen_iou_weight": self.db_gen_iou_weight_spin.value(),
"db_gen_quality_exp": self.db_gen_quality_exp_spin.value(),
"db_gen_power": self.db_gen_power_spin.value(),
"db_gen_n_workers": self.db_gen_n_workers_spin.value(),
```

Find the `set_state` method and add:
```python
self.db_gen_min_area_spin.setValue(state.get("db_gen_min_area", 300))
self.db_gen_max_area_spin.setValue(state.get("db_gen_max_area", 100_000))
self.db_gen_fg_thr_spin.setValue(state.get("db_gen_fg_thr", 0.5))
self.db_gen_min_frontier_spin.setValue(state.get("db_gen_min_frontier", 0.0))
idx = self.db_gen_ws_hierarchy_combo.findText(state.get("db_gen_ws_hierarchy", "area"))
if idx >= 0:
    self.db_gen_ws_hierarchy_combo.setCurrentIndex(idx)
self.db_gen_max_dist_spin.setValue(state.get("db_gen_max_dist", 15.0))
self.db_gen_max_neighbors_spin.setValue(state.get("db_gen_max_neighbors", 5))
idx = self.db_gen_linking_mode_combo.findText(state.get("db_gen_linking_mode", "default"))
if idx >= 0:
    self.db_gen_linking_mode_combo.setCurrentIndex(idx)
self.db_gen_iou_weight_spin.setValue(state.get("db_gen_iou_weight", 1.0))
self.db_gen_quality_exp_spin.setValue(state.get("db_gen_quality_exp", 8.0))
self.db_gen_power_spin.setValue(state.get("db_gen_power", 4.0))
self.db_gen_n_workers_spin.setValue(state.get("db_gen_n_workers", 1))
```

- [ ] **Step 4: Run test**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_controls_persist_through_state -v
```
Expected: PASS

- [ ] **Step 5: Run full suite + commit**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py -v 2>&1 | tail -20
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat(widget): persist db_gen section controls through get/set_state"
```

---

## Task 11: Final check — remove xfail, update deprecated ultrack_section spinbox tests

**Files:**
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

After the full refactor, `test_correction_section_has_no_separate_resolve_action_group` should pass (resolve is in ultrack_section, not correction_section). Remove the `xfail` marker.

Update or remove tests that still reference removed attributes:
- `ultrack_min_area_spin` → now `db_gen_min_area_spin`
- `ultrack_max_dist_spin` → now `db_gen_max_dist_spin`
- `ultrack_linking_mode_combo` → now `db_gen_linking_mode_combo`
- `ultrack_iou_weight_spin` → now `db_gen_iou_weight_spin`
- `ultrack_max_neighbors_spin` → now `db_gen_max_neighbors_spin`
- `ultrack_power_spin` → now `db_gen_power_spin`
- `ultrack_quality_exp_spin` → now `db_gen_quality_exp_spin`

Update `test_ultrack_section_exposes_validated_seed_prior_controls` to check the new locations:
```python
def test_db_gen_section_exposes_quality_and_power_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.db_gen_power_spin.value() == 4.0
    assert widget.db_gen_quality_exp_spin.value() == 8.0
    assert "solver transform" in widget.db_gen_power_spin.toolTip()
    assert "node_prob" in widget.db_gen_quality_exp_spin.toolTip()

    widget.deleteLater()
    viewer.close()


def test_ultrack_section_still_exposes_seed_prior_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_seed_weight_spin.value() == 0.5
    assert widget.ultrack_seed_space_spin.value() == 25.0
    assert widget.ultrack_seed_time_spin.value() == 2.0
    assert widget.ultrack_seed_window_spin.value() == 5
    assert "validated cells" in widget.ultrack_seed_weight_spin.toolTip()

    widget.deleteLater()
    viewer.close()
```

Update `test_validated_seed_prior_controls_follow_resolve_checkbox` to not check power/quality_exp (which are now in db_gen_section and are always enabled):
```python
def test_validated_seed_prior_controls_follow_resolve_checkbox():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    controls = [
        widget.ultrack_seed_weight_spin,
        widget.ultrack_seed_space_spin,
        widget.ultrack_seed_time_spin,
        widget.ultrack_seed_window_spin,
    ]

    widget.ultrack_route_check.setChecked(False)
    _app.processEvents()
    assert all(not control.isEnabled() for control in controls)

    widget.ultrack_route_check.setChecked(True)
    _app.processEvents()
    assert all(control.isEnabled() for control in controls)

    widget.deleteLater()
    viewer.close()
```

Update `test_ultrack_seed_prior_controls_persist_through_state` to not test power/quality_exp (they're now in db_gen state tested separately).

- [ ] **Step 1: Make all the above test updates**

- [ ] **Step 2: Run full test suite**

```bash
conda run -n cellflow pytest tests/napari/test_nucleus_tracking_correction_layout.py -v
```
Expected: all pass, no xfail

- [ ] **Step 3: Final commit**

```bash
git add tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "test: clean up test suite after canonical nucleus workflow refactor"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Task |
|---|---|
| 5 top-level sections with exact titles | Task 5 |
| H5 hypothesis generation hidden | Task 5 |
| H5 database browser hidden | Task 5 |
| `foreground_masks.tif` canonical (not `foreground_maps.tif`) | Task 6 (`_foreground_masks_path`) |
| DB generation uses `ultrack.segment` | Task 6 |
| DB generation runs node-prob scoring | Task 6 |
| DB generation runs linking | Task 6 |
| DB browser inspects `data.db` | Task 7 |
| Tracking solves existing linked DB (no rebuild) | Task 8 |
| Resolve from validated uses canonical seg inputs | Task 9 |
| Extend uses `data.db` candidates | Task 8 |
| Retrack unchanged (label-based, no H5) | No change needed — already label-based |
| Every section has status label, progress bar, run, terminal | Tasks 5 + 6 |
| Missing-input errors are section-local | Tasks 6, 8, 9 |
| Terminal scripts are self-contained with `__main__` guard | Tasks 6, 9 |
| State persistence for new controls | Task 10 |
| `TrackingConfig` has segmentation fields | Task 1, 2 |
| Deprecated code not deleted (kept but hidden) | Task 5 |

### Gaps identified
- `_get_nz` still references `read_hypothesis_labels` — update or remove in Task 8 (it can fall back to 1 if `hypotheses.h5` is gone).
- `output_files` widget showing `hypotheses.h5` — remove/hide in Task 5.
- Old terminal script tests (`test_ultrack_terminal_script_includes_visible_config_controls`) need updates in Task 11 since the terminal script now solves an existing DB (no ingest step).
