# Validated Seed Prior Ultrack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace resolve-from-validated prune-and-paste behavior with annotation-backed validated Ultrack nodes and a resolve-only additive seed prior.

**Architecture:** Keep normal Ultrack tracking unchanged, and route the new behavior only through `resolve_with_validation()`. Put validated-node database mutation in `validation_nodes.py`, node-prior scoring in `seed_prior.py`, and keep `reseed.py` as orchestration. The nucleus intensity image for `drop_frac` is `pos_dir / "0_input" / "nucleus_zavg.tif"`, passed into the core resolve API as an explicit path.

**Tech Stack:** Python, NumPy, SciPy/skimage, tifffile, SQLAlchemy, Ultrack `NodeDB`/`OverlapDB`/`VarAnnotation`, pytest, Qt/napari widget tests.

---

## File Structure

- Modify `src/cellflow/tracking_ultrack/config.py`: add resolve-only seed-prior config defaults.
- Create `src/cellflow/tracking_ultrack/validation_nodes.py`: extract validated masks, insert synthetic `NodeDB` rows, mark overlaps as annotations, and report inserted/skipped counts.
- Create `src/cellflow/tracking_ultrack/seed_prior.py`: load/normalize nucleus zavg image, compute `drop_frac`, compute best seed affinity, and write `NodeDB.node_prob`.
- Modify `src/cellflow/tracking_ultrack/reseed.py`: remove prune-and-paste from the success path, call validation injection and scoring, solve with annotations.
- Modify `src/cellflow/napari/nucleus_workflow_widget.py`: surface controls, enable them only for resolve, persist state, pass `nucleus_zavg.tif` and new config into GUI and terminal resolve paths.
- Modify `tests/tracking_ultrack/test_reseed.py`: add orchestration tests for annotation solve and no paste-back success path.
- Add `tests/tracking_ultrack/test_validation_nodes.py`: focused DB injection tests.
- Add `tests/tracking_ultrack/test_seed_prior.py`: focused scoring tests.
- Add `tests/tracking_ultrack/test_config.py`: config default tests.
- Modify `tests/napari/test_nucleus_tracking_correction_layout.py`: UI controls, enable/disable, state persistence, terminal command values.

---

### Task 1: TrackingConfig Defaults

**Files:**
- Modify: `src/cellflow/tracking_ultrack/config.py`
- Test: `tests/tracking_ultrack/test_config.py`

- [ ] **Step 1: Write the failing config defaults test**

Create `tests/tracking_ultrack/test_config.py`:

```python
from cellflow.tracking_ultrack.config import TrackingConfig


def test_tracking_config_exposes_seed_prior_defaults():
    cfg = TrackingConfig()

    assert cfg.power == 4.0
    assert cfg.quality_exponent == 8.0
    assert cfg.seed_weight == 0.5
    assert cfg.seed_sigma_space == 25.0
    assert cfg.seed_tau_time == 2.0
    assert cfg.seed_max_dt == 5
    assert cfg.seed_sigma_area == 0.5
```

- [ ] **Step 2: Run the failing test**

Run: `pytest tests/tracking_ultrack/test_config.py -q`

Expected: FAIL with `AttributeError` for `quality_exponent`.

- [ ] **Step 3: Add config fields**

In `src/cellflow/tracking_ultrack/config.py`, add this block after solver fields:

```python
    # Resolve-from-validated node prior
    quality_exponent: float = 8.0
    seed_weight: float = 0.5
    seed_sigma_space: float = 25.0
    seed_tau_time: float = 2.0
    seed_max_dt: int = 5
    seed_sigma_area: float = 0.5
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/tracking_ultrack/test_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/config.py tests/tracking_ultrack/test_config.py
git commit -m "feat: add validated resolve seed prior config"
```

---

### Task 2: Napari Resolve Controls And Config Plumbing

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write the failing UI exposure test**

Add to `tests/napari/test_nucleus_tracking_correction_layout.py`:

```python
def test_ultrack_section_exposes_validated_seed_prior_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_power_spin.value() == 4.0
    assert widget.ultrack_quality_exp_spin.value() == 8.0
    assert widget.ultrack_seed_weight_spin.value() == 0.5
    assert widget.ultrack_seed_space_spin.value() == 25.0
    assert widget.ultrack_seed_time_spin.value() == 2.0
    assert widget.ultrack_seed_window_spin.value() == 5

    assert "solver transform" in widget.ultrack_power_spin.toolTip()
    assert "node_prob" in widget.ultrack_quality_exp_spin.toolTip()
    assert "validated cells" in widget.ultrack_seed_weight_spin.toolTip()

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Write the failing enable/disable test**

Add:

```python
def test_validated_seed_prior_controls_follow_resolve_checkbox():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    controls = [
        widget.ultrack_quality_exp_spin,
        widget.ultrack_seed_weight_spin,
        widget.ultrack_seed_space_spin,
        widget.ultrack_seed_time_spin,
        widget.ultrack_seed_window_spin,
    ]

    widget.ultrack_route_check.setChecked(False)
    _app.processEvents()
    assert all(not control.isEnabled() for control in controls)
    assert widget.ultrack_power_spin.isEnabled()

    widget.ultrack_route_check.setChecked(True)
    _app.processEvents()
    assert all(control.isEnabled() for control in controls)
    assert widget.ultrack_power_spin.isEnabled()

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 3: Write the failing state persistence test**

Add:

```python
def test_ultrack_seed_prior_controls_persist_through_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.ultrack_route_check.setChecked(True)
    widget.ultrack_power_spin.setValue(3.5)
    widget.ultrack_quality_exp_spin.setValue(9.0)
    widget.ultrack_seed_weight_spin.setValue(0.75)
    widget.ultrack_seed_space_spin.setValue(30.0)
    widget.ultrack_seed_time_spin.setValue(3.0)
    widget.ultrack_seed_window_spin.setValue(7)

    state = widget.get_state()
    widget.deleteLater()

    widget = widget_class(viewer)
    widget.set_state(state)

    assert widget.ultrack_route_check.isChecked()
    assert widget.ultrack_power_spin.value() == 3.5
    assert widget.ultrack_quality_exp_spin.value() == 9.0
    assert widget.ultrack_seed_weight_spin.value() == 0.75
    assert widget.ultrack_seed_space_spin.value() == 30.0
    assert widget.ultrack_seed_time_spin.value() == 3.0
    assert widget.ultrack_seed_window_spin.value() == 7

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 4: Run the failing UI tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_section_exposes_validated_seed_prior_controls -q
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_validated_seed_prior_controls_follow_resolve_checkbox -q
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_seed_prior_controls_persist_through_state -q
```

Expected: FAIL with missing widget attributes.

- [ ] **Step 5: Add widgets and helper methods**

In `src/cellflow/napari/nucleus_workflow_widget.py`, add spinboxes near the existing Ultrack spinbox creation:

```python
        self.ultrack_power_spin = QDoubleSpinBox()
        self.ultrack_power_spin.setRange(0.1, 20.0)
        self.ultrack_power_spin.setValue(4.0)
        self.ultrack_power_spin.setSingleStep(0.5)
        self.ultrack_power_spin.setDecimals(2)
        self.ultrack_power_spin.setToolTip(
            "Ultrack's solver transform for node_prob and link weights. "
            "With link_function=power, stored weights are raised to this power during solving."
        )

        self.ultrack_quality_exp_spin = QDoubleSpinBox()
        self.ultrack_quality_exp_spin.setRange(0.1, 50.0)
        self.ultrack_quality_exp_spin.setValue(8.0)
        self.ultrack_quality_exp_spin.setSingleStep(0.5)
        self.ultrack_quality_exp_spin.setDecimals(2)
        self.ultrack_quality_exp_spin.setToolTip(
            "Raises the signal-based segmentation quality before storing it as node_prob. "
            "Higher values favor high-confidence whole-object candidates over fragments."
        )

        self.ultrack_seed_weight_spin = QDoubleSpinBox()
        self.ultrack_seed_weight_spin.setRange(0.0, 10.0)
        self.ultrack_seed_weight_spin.setValue(0.5)
        self.ultrack_seed_weight_spin.setSingleStep(0.1)
        self.ultrack_seed_weight_spin.setDecimals(2)
        self.ultrack_seed_weight_spin.setToolTip(
            "Additive reward for candidates similar to nearby validated cells. "
            "Zero disables the seed-local bonus."
        )

        self.ultrack_seed_space_spin = QDoubleSpinBox()
        self.ultrack_seed_space_spin.setRange(1.0, 500.0)
        self.ultrack_seed_space_spin.setValue(25.0)
        self.ultrack_seed_space_spin.setSingleStep(5.0)
        self.ultrack_seed_space_spin.setDecimals(1)
        self.ultrack_seed_space_spin.setToolTip(
            "Spatial decay scale for seed proximity. Larger values let validated cells influence candidates farther away."
        )

        self.ultrack_seed_time_spin = QDoubleSpinBox()
        self.ultrack_seed_time_spin.setRange(0.1, 50.0)
        self.ultrack_seed_time_spin.setValue(2.0)
        self.ultrack_seed_time_spin.setSingleStep(0.5)
        self.ultrack_seed_time_spin.setDecimals(1)
        self.ultrack_seed_time_spin.setToolTip(
            "Temporal decay scale in frames. Larger values let validated cells influence more distant frames within the seed window."
        )

        self.ultrack_seed_window_spin = QSpinBox()
        self.ultrack_seed_window_spin.setRange(0, 100)
        self.ultrack_seed_window_spin.setValue(5)
        self.ultrack_seed_window_spin.setToolTip(
            "Maximum frame distance from a validated cell used for seed affinity."
        )
```

Add these rows below the existing route checkbox row:

```python
        resolve_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            resolve_grid,
            0,
            "Ultrack Power:",
            _compact(self.ultrack_power_spin, 80),
            "Quality Exp:",
            _compact(self.ultrack_quality_exp_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            resolve_grid,
            1,
            "Seed Weight:",
            _compact(self.ultrack_seed_weight_spin, 80),
            "Seed Space (px):",
            _compact(self.ultrack_seed_space_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            resolve_grid,
            2,
            "Seed Time:",
            _compact(self.ultrack_seed_time_spin, 80),
            "Seed Window:",
            _compact(self.ultrack_seed_window_spin, 80),
            field_width=80,
        )
        ultrack_lay.addLayout(resolve_grid)
```

Add helper methods near `_on_ultrack_mode_changed`:

```python
    def _set_resolve_prior_controls_enabled(self, enabled: bool) -> None:
        for control in (
            self.ultrack_quality_exp_spin,
            self.ultrack_seed_weight_spin,
            self.ultrack_seed_space_spin,
            self.ultrack_seed_time_spin,
            self.ultrack_seed_window_spin,
        ):
            control.setEnabled(enabled)

    def _ultrack_config_from_controls(self) -> UltrackConfig:
        return UltrackConfig(
            min_area=self.ultrack_min_area_spin.value(),
            max_distance=self.ultrack_max_dist_spin.value(),
            linking_mode=self.ultrack_linking_mode_combo.currentText(),
            iou_weight=self.ultrack_iou_weight_spin.value(),
            appear_weight=self.ultrack_appear_spin.value(),
            disappear_weight=self.ultrack_disappear_spin.value(),
            division_weight=self.ultrack_division_spin.value(),
            max_neighbors=self.ultrack_max_neighbors_spin.value(),
            power=self.ultrack_power_spin.value(),
            quality_exponent=self.ultrack_quality_exp_spin.value(),
            seed_weight=self.ultrack_seed_weight_spin.value(),
            seed_sigma_space=self.ultrack_seed_space_spin.value(),
            seed_tau_time=self.ultrack_seed_time_spin.value(),
            seed_max_dt=self.ultrack_seed_window_spin.value(),
        )
```

Connect the checkbox after it is created:

```python
        self.ultrack_route_check.toggled.connect(self._set_resolve_prior_controls_enabled)
        self._set_resolve_prior_controls_enabled(self.ultrack_route_check.isChecked())
```

- [ ] **Step 6: Use the helper in normal and resolve config creation**

Replace duplicated `UltrackConfig(...)` construction in `_on_run_ultrack()` and `_on_resolve_with_validation()` with:

```python
        cfg = self._ultrack_config_from_controls()
```

Keep `max_partitions` and `n_frames` capture separate because they are ingest options, not `TrackingConfig` fields.

- [ ] **Step 7: Persist state fields**

Add to `get_state()["ultrack"]`:

```python
                "power":            self.ultrack_power_spin.value(),
                "quality_exponent": self.ultrack_quality_exp_spin.value(),
                "seed_weight":      self.ultrack_seed_weight_spin.value(),
                "seed_sigma_space": self.ultrack_seed_space_spin.value(),
                "seed_tau_time":    self.ultrack_seed_time_spin.value(),
                "seed_max_dt":      self.ultrack_seed_window_spin.value(),
```

Add to `set_state()` under `if "ultrack" in state`:

```python
            if "power"            in ul: self.ultrack_power_spin.setValue(ul["power"])
            if "quality_exponent" in ul: self.ultrack_quality_exp_spin.setValue(ul["quality_exponent"])
            if "seed_weight"      in ul: self.ultrack_seed_weight_spin.setValue(ul["seed_weight"])
            if "seed_sigma_space" in ul: self.ultrack_seed_space_spin.setValue(ul["seed_sigma_space"])
            if "seed_tau_time"    in ul: self.ultrack_seed_time_spin.setValue(ul["seed_tau_time"])
            if "seed_max_dt"      in ul: self.ultrack_seed_window_spin.setValue(ul["seed_max_dt"])
```

- [ ] **Step 8: Run UI tests**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat: surface validated resolve prior controls"
```

---

### Task 3: Validated Node Injection Module

**Files:**
- Create: `src/cellflow/tracking_ultrack/validation_nodes.py`
- Test: `tests/tracking_ultrack/test_validation_nodes.py`

- [ ] **Step 1: Write failing injection tests**

Create `tests/tracking_ultrack/test_validation_nodes.py`:

```python
from pathlib import Path

import numpy as np
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.validation_nodes import inject_validated_nodes
from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row


def test_inject_validated_nodes_creates_real_node_with_reserved_hierarchy(tmp_path):
    from ultrack.core.database import NodeDB, VarAnnotation

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        candidate = _make_node_row(1_000_001, 0, 30, 30, 40, 40)
        candidate.t_node_id = 1
        session.add(candidate)
        session.commit()

    tracked = np.zeros((1, 64, 64), dtype=np.uint32)
    tracked[0, 5:15, 7:17] = 42

    report = inject_validated_nodes(tmp_path, {42: {0}}, tracked, TrackingConfig())

    assert report.inserted == 1
    assert report.skipped_missing == 0
    with Session(engine) as session:
        rows = session.query(NodeDB).order_by(NodeDB.id).all()

    injected = [row for row in rows if row.t_hier_id == 0][0]
    assert injected.t == 0
    assert injected.t_node_id == 2
    assert injected.node_annot == VarAnnotation.REAL
    assert injected.node_prob == 1.0
    assert injected.area == 100


def test_inject_validated_nodes_marks_intersecting_candidates_fake_and_adds_overlap(tmp_path):
    from ultrack.core.database import NodeDB, OverlapDB, VarAnnotation

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        overlap_candidate = _make_node_row(1_000_001, 0, 0, 0, 10, 10)
        overlap_candidate.t_node_id = 1
        unrelated_candidate = _make_node_row(1_000_002, 0, 30, 30, 40, 40)
        unrelated_candidate.t_node_id = 2
        session.add_all([overlap_candidate, unrelated_candidate])
        session.commit()

    tracked = np.zeros((1, 64, 64), dtype=np.uint32)
    tracked[0, 5:15, 5:15] = 7

    report = inject_validated_nodes(tmp_path, {7: {0}}, tracked, TrackingConfig())

    assert report.inserted == 1
    with Session(engine) as session:
        rows = {row.id: row for row in session.query(NodeDB).all()}
        injected_id = [row.id for row in rows.values() if row.t_hier_id == 0][0]
        assert rows[1_000_001].node_annot == VarAnnotation.FAKE
        assert rows[1_000_002].node_annot == VarAnnotation.UNKNOWN
        assert rows[injected_id].node_annot == VarAnnotation.REAL
        overlap_pairs = {
            (row.node_id, row.ancestor_id)
            for row in session.query(OverlapDB).all()
        }

    assert (max(injected_id, 1_000_001), min(injected_id, 1_000_001)) in overlap_pairs


def test_inject_validated_nodes_reports_missing_cell_frames(tmp_path):
    _make_engine(tmp_path / "data.db")
    tracked = np.zeros((1, 32, 32), dtype=np.uint32)

    report = inject_validated_nodes(tmp_path, {9: {0}}, tracked, TrackingConfig())

    assert report.inserted == 0
    assert report.skipped_missing == 1
    assert report.skipped == [(9, 0)]
```

- [ ] **Step 2: Run the failing tests**

Run: `pytest tests/tracking_ultrack/test_validation_nodes.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement validation node injection**

Create `src/cellflow/tracking_ultrack/validation_nodes.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlalchemy as sqla
from scipy.ndimage import find_objects
from sqlalchemy.orm import Session

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.ingest import _generate_id, _make_node_pickle


@dataclass(frozen=True)
class ValidationInjectionReport:
    inserted: int
    skipped_missing: int
    skipped: list[tuple[int, int]]
    faked: int
    overlaps_added: int


@dataclass(frozen=True)
class _MaskRecord:
    cell_id: int
    t: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area: int
    y: float
    x: float


def _frame_2d_for_cell(tracked_labels: np.ndarray, t: int, cell_id: int) -> np.ndarray:
    frame = np.asarray(tracked_labels)[t]
    if frame.ndim == 3:
        return np.asarray(frame == cell_id).any(axis=0)
    if frame.ndim == 2:
        return np.asarray(frame == cell_id)
    raise ValueError(f"Expected tracked frame to be 2D or 3D, got shape {frame.shape}")


def _validated_mask_records(
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> tuple[list[_MaskRecord], list[tuple[int, int]]]:
    records: list[_MaskRecord] = []
    skipped: list[tuple[int, int]] = []
    n_frames = int(np.asarray(tracked_labels).shape[0])

    for cell_id, frames in sorted(validated_tracks.items()):
        for raw_t in sorted(frames):
            t = int(raw_t)
            if t < 0 or t >= n_frames:
                skipped.append((int(cell_id), t))
                continue
            mask_2d = _frame_2d_for_cell(tracked_labels, t, int(cell_id))
            if not mask_2d.any():
                skipped.append((int(cell_id), t))
                continue
            rows = np.where(mask_2d.any(axis=1))[0]
            cols = np.where(mask_2d.any(axis=0))[0]
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
            crop = np.ascontiguousarray(mask_2d[y0:y1, x0:x1], dtype=bool)
            ys, xs = np.nonzero(crop)
            records.append(
                _MaskRecord(
                    cell_id=int(cell_id),
                    t=t,
                    bbox=(y0, x0, y1, x1),
                    mask=crop,
                    area=int(crop.sum()),
                    y=float(y0 + ys.mean()),
                    x=float(x0 + xs.mean()),
                )
            )
    return records, skipped


def _node_mask_record(node_id: int, node) -> tuple[tuple[int, int, int, int], np.ndarray]:
    bbox = np.asarray(node.bbox)
    ndim = len(bbox) // 2
    if ndim == 3:
        y0, x0, y1, x1 = int(bbox[1]), int(bbox[2]), int(bbox[4]), int(bbox[5])
    elif ndim == 2:
        y0, x0, y1, x1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    else:
        raise ValueError(f"Unexpected bbox for node {node_id}: {bbox}")
    mask = np.asarray(node.mask, dtype=bool)
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] == 1 else mask.any(axis=0)
    return (y0, x0, y1, x1), np.ascontiguousarray(mask, dtype=bool)


def _intersects(lhs_bbox: tuple[int, int, int, int], lhs_mask: np.ndarray, rhs_bbox: tuple[int, int, int, int], rhs_mask: np.ndarray) -> bool:
    ly0, lx0, ly1, lx1 = lhs_bbox
    ry0, rx0, ry1, rx1 = rhs_bbox
    oy0, ox0 = max(ly0, ry0), max(lx0, rx0)
    oy1, ox1 = min(ly1, ry1), min(lx1, rx1)
    if oy0 >= oy1 or ox0 >= ox1:
        return False
    lhs_crop = lhs_mask[oy0 - ly0: oy1 - ly0, ox0 - lx0: ox1 - lx0]
    rhs_crop = rhs_mask[oy0 - ry0: oy1 - ry0, ox0 - rx0: ox1 - rx0]
    return bool(np.logical_and(lhs_crop, rhs_crop).any())


def inject_validated_nodes(
    working_dir: str | Path,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
    cfg: TrackingConfig,
) -> ValidationInjectionReport:
    from ultrack.core.database import NodeDB, OverlapDB, VarAnnotation

    working_dir = Path(working_dir)
    records, skipped = _validated_mask_records(validated_tracks, tracked_labels)
    if not records:
        return ValidationInjectionReport(0, len(skipped), skipped, 0, 0)

    engine = sqla.create_engine(f"sqlite:///{working_dir / 'data.db'}")
    inserted = 0
    faked: set[int] = set()
    overlap_pairs: set[tuple[int, int]] = set()

    with Session(engine) as session:
        next_t_node_id: dict[int, int] = {}
        for t in {record.t for record in records}:
            current_max = (
                session.query(sqla.func.max(NodeDB.t_node_id))
                .where(NodeDB.t == t)
                .scalar()
            )
            next_t_node_id[t] = int(current_max or 0) + 1

        for record in records:
            t_node_id = next_t_node_id[record.t]
            if t_node_id >= cfg.max_segments_per_time:
                raise ValueError(
                    f"Cannot inject validated node at t={record.t}: "
                    f"t_node_id={t_node_id} exceeds max_segments_per_time={cfg.max_segments_per_time}"
                )
            next_t_node_id[record.t] += 1
            node_id = _generate_id(t_node_id, record.t, cfg.max_segments_per_time)
            bbox_arr = np.array(record.bbox, dtype=np.int32)
            session.add(
                NodeDB(
                    id=node_id,
                    t=record.t,
                    t_node_id=t_node_id,
                    t_hier_id=0,
                    z=0,
                    y=record.y,
                    x=record.x,
                    area=record.area,
                    node_prob=1.0,
                    node_annot=VarAnnotation.REAL,
                    pickle=_make_node_pickle(record.t, record.mask, bbox_arr, node_id),
                )
            )
            inserted += 1

            candidates = (
                session.query(NodeDB.id, NodeDB.pickle, NodeDB.node_annot)
                .where(NodeDB.t == record.t, NodeDB.t_hier_id != 0)
                .all()
            )
            for candidate_id, candidate_node, candidate_annot in candidates:
                candidate_bbox, candidate_mask = _node_mask_record(int(candidate_id), candidate_node)
                if not _intersects(record.bbox, record.mask, candidate_bbox, candidate_mask):
                    continue
                faked.add(int(candidate_id))
                hi = max(int(node_id), int(candidate_id))
                lo = min(int(node_id), int(candidate_id))
                overlap_pairs.add((hi, lo))

        if faked:
            session.query(NodeDB).where(NodeDB.id.in_(sorted(faked))).update(
                {NodeDB.node_annot: VarAnnotation.FAKE},
                synchronize_session=False,
            )
        for node_id, ancestor_id in sorted(overlap_pairs):
            exists = (
                session.query(OverlapDB)
                .where(OverlapDB.node_id == node_id, OverlapDB.ancestor_id == ancestor_id)
                .first()
            )
            if exists is None:
                session.add(OverlapDB(node_id=node_id, ancestor_id=ancestor_id))
        session.commit()

    return ValidationInjectionReport(
        inserted=inserted,
        skipped_missing=len(skipped),
        skipped=skipped,
        faked=len(faked),
        overlaps_added=len(overlap_pairs),
    )
```

- [ ] **Step 4: Run validation node tests**

Run: `pytest tests/tracking_ultrack/test_validation_nodes.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/validation_nodes.py tests/tracking_ultrack/test_validation_nodes.py
git commit -m "feat: inject validated masks as ultrack annotations"
```

---

### Task 4: Seed Prior Scoring

**Files:**
- Create: `src/cellflow/tracking_ultrack/seed_prior.py`
- Test: `tests/tracking_ultrack/test_seed_prior.py`

- [ ] **Step 1: Write failing scoring tests**

Create `tests/tracking_ultrack/test_seed_prior.py`:

```python
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.seed_prior import compute_drop_frac, write_seed_prior_node_probs
from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row


def test_compute_drop_frac_counts_outer_ring_pixels_below_inside_median():
    image = np.full((12, 12), 10.0, dtype=np.float32)
    image[3:7, 3:7] = 100.0
    image[2:8, 2:8] = np.minimum(image[2:8, 2:8], 5.0)
    mask = np.ones((4, 4), dtype=bool)

    assert compute_drop_frac(image, (3, 3, 7, 7), mask) == 1.0


def test_write_seed_prior_node_probs_uses_drop_quality_and_best_seed_affinity(tmp_path):
    from ultrack.core.database import NodeDB, VarAnnotation

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        candidate = _make_node_row(1_000_001, 0, 3, 3, 7, 7)
        seed = _make_node_row(2_000_001, 1, 3, 3, 7, 7)
        seed.t_hier_id = 0
        seed.node_annot = VarAnnotation.REAL
        session.add_all([candidate, seed])
        session.commit()

    image = np.full((2, 12, 12), 10.0, dtype=np.float32)
    image[:, 3:7, 3:7] = 100.0
    image[:, 2:8, 2:8] = np.minimum(image[:, 2:8, 2:8], 5.0)
    image_path = tmp_path / "nucleus_zavg.tif"
    tifffile.imwrite(image_path, image)

    cfg = TrackingConfig(
        quality_exponent=2.0,
        seed_weight=0.5,
        seed_sigma_space=25.0,
        seed_tau_time=2.0,
        seed_max_dt=5,
        seed_sigma_area=0.5,
    )

    report = write_seed_prior_node_probs(tmp_path, image_path, cfg)

    assert report.scored == 1
    with Session(engine) as session:
        candidate = session.get(NodeDB, 1_000_001)
        seed = session.get(NodeDB, 2_000_001)

    assert candidate.node_prob == 1.5
    assert seed.node_prob == 1.0
```

- [ ] **Step 2: Run the failing tests**

Run: `pytest tests/tracking_ultrack/test_seed_prior.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement seed-prior scoring**

Create `src/cellflow/tracking_ultrack/seed_prior.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlalchemy as sqla
import tifffile
from scipy.ndimage import binary_dilation
from sqlalchemy.orm import Session

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.validation_nodes import _node_mask_record


@dataclass(frozen=True)
class SeedPriorReport:
    scored: int
    seeds: int


@dataclass(frozen=True)
class _NodeScoreRecord:
    node_id: int
    t: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area: int
    y: float
    x: float


def _load_signal_stack(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nucleus zavg image not found: {path}")
    arr = np.asarray(tifffile.imread(path), dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError(f"Expected nucleus zavg image to be 2D, 3D, or singleton-Z 4D, got {arr.shape}")


def compute_drop_frac(frame: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray) -> float:
    y0, x0, y1, x1 = bbox
    inside = np.asarray(frame[y0:y1, x0:x1], dtype=np.float32)[mask]
    if inside.size == 0:
        return 0.0
    inside_median = float(np.median(inside))

    pad_y0 = max(0, y0 - 1)
    pad_x0 = max(0, x0 - 1)
    pad_y1 = min(frame.shape[0], y1 + 1)
    pad_x1 = min(frame.shape[1], x1 + 1)

    expanded = np.zeros((pad_y1 - pad_y0, pad_x1 - pad_x0), dtype=bool)
    inner_y0 = y0 - pad_y0
    inner_x0 = x0 - pad_x0
    expanded[inner_y0:inner_y0 + mask.shape[0], inner_x0:inner_x0 + mask.shape[1]] = mask
    ring = binary_dilation(expanded) & ~expanded
    if not ring.any():
        return 0.0
    ring_values = np.asarray(frame[pad_y0:pad_y1, pad_x0:pad_x1], dtype=np.float32)[ring]
    return float(np.mean(ring_values < inside_median))


def _affinity(node: _NodeScoreRecord, seed: _NodeScoreRecord, cfg: TrackingConfig) -> float:
    if node.area <= 0 or seed.area <= 0:
        return 0.0
    dt = abs(int(node.t) - int(seed.t))
    if dt > cfg.seed_max_dt:
        return 0.0
    area_ratio = float(node.area) / float(seed.area)
    size_similarity = np.exp(-abs(np.log(area_ratio)) / cfg.seed_sigma_area)
    dist = float(np.hypot(node.y - seed.y, node.x - seed.x))
    spatial_decay = np.exp(-((dist / cfg.seed_sigma_space) ** 2))
    temporal_decay = np.exp(-(dt / cfg.seed_tau_time))
    return float(size_similarity * spatial_decay * temporal_decay)


def write_seed_prior_node_probs(
    working_dir: str | Path,
    intensity_image_path: str | Path,
    cfg: TrackingConfig,
) -> SeedPriorReport:
    from ultrack.core.database import NodeDB, VarAnnotation

    signal = _load_signal_stack(intensity_image_path)
    engine = sqla.create_engine(f"sqlite:///{Path(working_dir) / 'data.db'}")

    with Session(engine) as session:
        rows = session.query(
            NodeDB.id,
            NodeDB.t,
            NodeDB.pickle,
            NodeDB.area,
            NodeDB.y,
            NodeDB.x,
            NodeDB.node_annot,
        ).all()
        records: list[_NodeScoreRecord] = []
        seed_records: list[_NodeScoreRecord] = []
        for node_id, t, node, area, y, x, annot in rows:
            bbox, mask = _node_mask_record(int(node_id), node)
            record = _NodeScoreRecord(
                node_id=int(node_id),
                t=int(t),
                bbox=bbox,
                mask=mask,
                area=int(area),
                y=float(y),
                x=float(x),
            )
            if annot == VarAnnotation.REAL:
                seed_records.append(record)
            else:
                records.append(record)

        scored = 0
        for record in records:
            if record.t >= signal.shape[0]:
                raise ValueError(
                    f"Nucleus zavg image has {signal.shape[0]} frame(s), cannot score node at t={record.t}"
                )
            drop_frac = compute_drop_frac(signal[record.t], record.bbox, record.mask)
            best_affinity = max((_affinity(record, seed, cfg) for seed in seed_records), default=0.0)
            node_prob = float((drop_frac ** cfg.quality_exponent) + cfg.seed_weight * best_affinity)
            session.query(NodeDB).where(NodeDB.id == record.node_id).update(
                {NodeDB.node_prob: node_prob},
                synchronize_session=False,
            )
            scored += 1

        for seed in seed_records:
            session.query(NodeDB).where(NodeDB.id == seed.node_id).update(
                {NodeDB.node_prob: 1.0},
                synchronize_session=False,
            )
        session.commit()

    return SeedPriorReport(scored=scored, seeds=len(seed_records))
```

- [ ] **Step 4: Run scoring tests**

Run: `pytest tests/tracking_ultrack/test_seed_prior.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/seed_prior.py tests/tracking_ultrack/test_seed_prior.py
git commit -m "feat: score resolve nodes with validated seed prior"
```

---

### Task 5: Resolve Orchestration With Annotations

**Files:**
- Modify: `src/cellflow/tracking_ultrack/reseed.py`
- Test: `tests/tracking_ultrack/test_reseed.py`

- [ ] **Step 1: Update orchestration tests**

In `tests/tracking_ultrack/test_reseed.py`, add:

```python
def test_resolve_with_validation_solves_with_annotations(monkeypatch, tmp_path):
    from cellflow.tracking_ultrack.reseed import resolve_with_validation

    calls = {"solve": [], "merge": 0}
    tracked = np.zeros((1, 16, 16), dtype=np.uint32)
    tracked[0, 1:5, 1:5] = 7
    image_path = tmp_path / "nucleus_zavg.tif"
    import tifffile
    tifffile.imwrite(image_path, np.ones((1, 16, 16), dtype=np.float32))

    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.ingest_hypotheses_to_db",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.inject_validated_nodes",
        lambda *args, **kwargs: type("Report", (), {"inserted": 1, "skipped_missing": 0, "faked": 0})(),
        raising=False,
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.write_seed_prior_node_probs",
        lambda *args, **kwargs: type("Report", (), {"scored": 1, "seeds": 1})(),
        raising=False,
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.run_linking",
        lambda *args, **kwargs: iter([(1, 1, "linked")]),
        raising=False,
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.run_solve",
        lambda *args, **kwargs: calls["solve"].append(kwargs) or iter([(1, 1, "solved")]),
        raising=False,
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.export_tracked_labels",
        lambda *args, **kwargs: tracked.copy(),
        raising=False,
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.merge_validated_into_export",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("paste-back should not run")),
    )

    result, id_map = resolve_with_validation(
        tmp_path / "hypotheses.h5",
        {7: {0}},
        tracked,
        TrackingConfig(),
        intensity_image_path=image_path,
    )

    assert result.shape == tracked.shape
    assert id_map == {7: 7}
    assert calls["solve"] == [{"overwrite": True, "use_annotations": True}]
```

- [ ] **Step 2: Run the failing orchestration test**

Run: `pytest tests/tracking_ultrack/test_reseed.py::test_resolve_with_validation_solves_with_annotations -q`

Expected: FAIL because `resolve_with_validation()` lacks `intensity_image_path` and still uses paste-back.

- [ ] **Step 3: Refactor imports for monkeypatchable orchestration**

At module top in `src/cellflow/tracking_ultrack/reseed.py`, add:

```python
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs
from cellflow.tracking_ultrack.solve import run_solve
from cellflow.tracking_ultrack.validation_nodes import inject_validated_nodes
```

Remove the matching local imports inside `resolve_with_validation()`.

- [ ] **Step 4: Change `resolve_with_validation()` signature**

Update the signature:

```python
def resolve_with_validation(
    hypotheses_path: str | Path,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
    cfg: TrackingConfig,
    progress_cb: "Callable[[str], None] | None" = None,
    *,
    intensity_image_path: str | Path,
) -> tuple[np.ndarray, dict[int, int]]:
```

- [ ] **Step 5: Replace orchestration body**

Replace the current temporary-directory body with:

```python
    if not validated_tracks:
        return np.asarray(tracked_labels, dtype=np.uint32).copy(), {}

    with tempfile.TemporaryDirectory(prefix="cellflow_resolve_workdir_") as tmp_workdir:
        working_dir = Path(tmp_workdir)

        _notify("Ingesting hypotheses…")
        ingest_hypotheses_to_db(hypotheses_path, working_dir, cfg, overwrite=True)

        _notify("Injecting validated masks…")
        injection = inject_validated_nodes(working_dir, validated_tracks, tracked_labels, cfg)
        if injection.skipped_missing:
            _notify(f"Skipped {injection.skipped_missing} validated cell-frame(s) missing from tracked labels.")
        if injection.inserted == 0:
            raise ValueError("No validated masks could be injected; resolve aborted before solve.")
        _notify(
            f"Injected {injection.inserted} validated node(s); "
            f"marked {injection.faked} overlapping candidate(s) false."
        )

        _notify("Scoring node probabilities…")
        score_report = write_seed_prior_node_probs(working_dir, intensity_image_path, cfg)
        _notify(f"Scored {score_report.scored} candidate node(s) from {score_report.seeds} validated seed node(s).")

        _notify("Linking hypotheses…")
        for _step, _total, label in run_linking(working_dir, cfg):
            _notify(label)

        _notify("Solving ILP with annotations…")
        for _step, _total, label in run_solve(
            working_dir,
            cfg,
            overwrite=True,
            use_annotations=True,
        ):
            _notify(label)

        _notify("Exporting tracks…")
        with tempfile.TemporaryDirectory(prefix="cellflow_resolve_export_") as tmpdir:
            out_path = Path(tmpdir) / "tracked_labels.tif"
            exported = export_tracked_labels(working_dir, cfg, out_path)

    exported = np.asarray(exported, dtype=np.uint32)
    id_map = _validated_export_id_map(exported, validated_tracks, tracked_labels)
    return exported, id_map
```

Add this helper above `resolve_with_validation()`:

```python
def _validated_export_id_map(
    exported_labels: np.ndarray,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> dict[int, int]:
    id_map: dict[int, int] = {}
    for cell_id, frames in validated_tracks.items():
        seen: list[int] = []
        for t in sorted(frames):
            frame_src = tracked_labels[int(t)]
            frame_out = exported_labels[int(t)]
            mask = (frame_src == cell_id).any(axis=0) if frame_src.ndim == 3 else (frame_src == cell_id)
            if not mask.any():
                continue
            out_values = frame_out[mask]
            out_values = out_values[out_values != 0]
            if out_values.size:
                values, counts = np.unique(out_values, return_counts=True)
                seen.append(int(values[int(np.argmax(counts))]))
        if seen:
            values, counts = np.unique(np.asarray(seen, dtype=np.int64), return_counts=True)
            id_map[int(cell_id)] = int(values[int(np.argmax(counts))])
    return id_map
```

- [ ] **Step 6: Run orchestration tests**

Run:

```bash
pytest tests/tracking_ultrack/test_reseed.py::test_resolve_with_validation_solves_with_annotations -q
pytest tests/tracking_ultrack/test_reseed.py -q
```

Expected: The new test passes. Existing tests that assert paste-back behavior may fail and should be updated only where they describe the old success path.

- [ ] **Step 7: Update obsolete paste-back integration expectations**

For tests whose names assert validated pixels are preserved by paste-back, change the assertion focus to:

```python
assert id_map
assert np.asarray(result).shape[:1] == tracked.shape[:1]
```

Keep unit tests for `merge_validated_into_export()` intact, because the helper remains available even though the normal success path no longer calls it.

- [ ] **Step 8: Commit**

```bash
git add src/cellflow/tracking_ultrack/reseed.py tests/tracking_ultrack/test_reseed.py
git commit -m "feat: resolve validated tracks through ultrack annotations"
```

---

### Task 6: Resolve GUI And Terminal Use Nucleus Zavg

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Write failing terminal command test**

Add to `tests/napari/test_nucleus_tracking_correction_layout.py`:

```python
def test_resolve_terminal_script_includes_seed_prior_and_nucleus_zavg(monkeypatch, tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    pos_dir = tmp_path / "pos00"
    (pos_dir / "0_input").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "2_nucleus" / "validated_cells.json").write_text(
        '{"version": 1, "tracks": {"7": [0]}}'
    )
    (pos_dir / "2_nucleus" / "hypotheses.h5").write_bytes(b"stub")
    (pos_dir / "2_nucleus" / "tracked_labels.tif").write_bytes(b"stub")
    (pos_dir / "0_input" / "nucleus_zavg.tif").write_bytes(b"stub")
    widget._pos_dir = pos_dir

    captured = {}
    monkeypatch.setattr("cellflow.napari.utils.launch_in_terminal", lambda cmd: captured.setdefault("cmd", cmd))

    widget.ultrack_route_check.setChecked(True)
    widget.ultrack_power_spin.setValue(3.5)
    widget.ultrack_quality_exp_spin.setValue(9.0)
    widget.ultrack_seed_weight_spin.setValue(0.75)
    widget.ultrack_seed_space_spin.setValue(30.0)
    widget.ultrack_seed_time_spin.setValue(3.0)
    widget.ultrack_seed_window_spin.setValue(7)

    widget._on_resolve_terminal()

    command = captured["cmd"]
    script = Path(command.split()[-1]).read_text()
    assert "nucleus_zavg.tif" in script
    assert "power=3.5" in script
    assert "quality_exponent=9.0" in script
    assert "seed_weight=0.75" in script
    assert "seed_sigma_space=30.0" in script
    assert "seed_tau_time=3.0" in script
    assert "seed_max_dt=7" in script
    assert "intensity_image_path=nucleus_zavg_path" in script

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Run the failing terminal test**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_resolve_terminal_script_includes_seed_prior_and_nucleus_zavg -q`

Expected: FAIL because the script omits new fields and intensity path.

- [ ] **Step 3: Pass nucleus zavg path into GUI resolve**

In `_on_resolve_with_validation()`, before starting the worker, add:

```python
        nucleus_zavg_path = self._nucleus_zavg_path()
        if nucleus_zavg_path is None or not nucleus_zavg_path.exists():
            self._set_ultrack_status("Nucleus z-avg image not found — prepare project inputs first.")
            return
```

In the worker call to `resolve_with_validation()`, add:

```python
                            intensity_image_path=nucleus_zavg_path,
```

- [ ] **Step 4: Update terminal resolve capture and script**

In `_on_resolve_terminal()`, capture:

```python
        nucleus_zavg_path = self._nucleus_zavg_path()
        if nucleus_zavg_path is None or not nucleus_zavg_path.exists():
            self._set_ultrack_status("Nucleus z-avg image not found — prepare project inputs first.")
            return
        cfg = self._ultrack_config_from_controls()
```

Emit config fields in the generated script:

```python
            f"    nucleus_zavg_path = pathlib.Path({str(nucleus_zavg_path)!r})\n"
            f"    cfg = TrackingConfig(\n"
            f"        min_area={cfg.min_area},\n"
            f"        max_distance={cfg.max_distance},\n"
            f"        linking_mode={cfg.linking_mode!r},\n"
            f"        iou_weight={cfg.iou_weight},\n"
            f"        appear_weight={cfg.appear_weight},\n"
            f"        disappear_weight={cfg.disappear_weight},\n"
            f"        division_weight={cfg.division_weight},\n"
            f"        max_neighbors={cfg.max_neighbors},\n"
            f"        power={cfg.power},\n"
            f"        quality_exponent={cfg.quality_exponent},\n"
            f"        seed_weight={cfg.seed_weight},\n"
            f"        seed_sigma_space={cfg.seed_sigma_space},\n"
            f"        seed_tau_time={cfg.seed_tau_time},\n"
            f"        seed_max_dt={cfg.seed_max_dt},\n"
            f"    )\n"
```

Update the script call:

```python
            "        intensity_image_path=nucleus_zavg_path,\n"
```

- [ ] **Step 5: Update normal Ultrack terminal config for `power`**

In `_on_ultrack_terminal()`, include:

```python
        power = self.ultrack_power_spin.value()
```

and emit:

```python
            f"        power={power},\n"
```

Normal Ultrack does not pass seed-prior fields into behavior, but passing them through `TrackingConfig` is harmless. Do not score seed priors unless `Resolve from validated` is checked.

- [ ] **Step 6: Run napari tests**

Run: `pytest tests/napari/test_nucleus_tracking_correction_layout.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat: pass nucleus zavg into validated resolve"
```

---

### Task 7: Final Verification

**Files:**
- No new source files unless previous tasks reveal a focused fix.

- [ ] **Step 1: Run focused tracking tests**

Run:

```bash
pytest tests/tracking_ultrack/test_config.py tests/tracking_ultrack/test_validation_nodes.py tests/tracking_ultrack/test_seed_prior.py tests/tracking_ultrack/test_reseed.py -q
```

Expected: PASS.

- [ ] **Step 2: Run focused napari tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py -q
```

Expected: PASS.

- [ ] **Step 3: Run ingest regression tests**

Run:

```bash
pytest tests/tracking/test_tracking_ultrack_ingest.py -q
```

Expected: PASS.

- [ ] **Step 4: Run style smoke test**

Run:

```bash
pytest tests/napari/test_ui_style.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit any verification fixes**

If fixes were needed:

```bash
git add src tests
git commit -m "fix: stabilize validated seed prior resolve"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review

- Spec coverage: config defaults, UI controls, validation injection, fake overlap annotations, missing validated frames, scoring formula, `use_annotations=True`, no paste-back success path, terminal parity, and nucleus zavg plumbing are all assigned to tasks.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: the plan uses `intensity_image_path`, `quality_exponent`, `seed_weight`, `seed_sigma_space`, `seed_tau_time`, `seed_max_dt`, and `seed_sigma_area` consistently across config, widget, terminal, scoring, and resolve orchestration.
- Scope check: this is one coherent feature with two focused new modules and does not change the normal Ultrack route except surfacing/passing `power`.
