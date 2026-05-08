# Ultrack Node Probability Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose explicit Ultrack database node-probability weights and enable circularity-based candidate preference by default.

**Architecture:** Keep scoring centralized in `cellflow.tracking_ultrack.seed_prior`. Add configuration fields to `TrackingConfig`, thread them through the DB-generation widget and terminal script, and verify the final `NodeDB.node_prob` formula with focused tests.

**Tech Stack:** Python, Pydantic, NumPy, tifffile, SQLAlchemy/Ultrack database models, Qt/napari widget tests, pytest.

---

## File Structure

- Modify `src/cellflow/tracking_ultrack/config.py`
  Add explicit node-probability weighting fields.
- Modify `src/cellflow/tracking_ultrack/seed_prior.py`
  Add circularity helper and update node-probability formula.
- Modify `src/cellflow/napari/nucleus_workflow_widget.py`
  Add UI controls, pass config values to in-process and terminal DB generation, and persist state.
- Modify `tests/tracking_ultrack/test_seed_prior.py`
  Test circularity helper and weighted node-probability formula.
- Modify `tests/napari/test_nucleus_tracking_correction_layout.py`
  Test control exposure, config pass-through, and state persistence.
- Modify `tests/tracking_ultrack/test_config.py`
  Add default config assertions for the new `TrackingConfig` fields.

---

### Task 1: Add Config Defaults

**Files:**
- Modify: `src/cellflow/tracking_ultrack/config.py`
- Test: `tests/tracking_ultrack/test_config.py`

- [ ] **Step 1: Write or extend a config default test**

If `tests/tracking_ultrack/test_config.py` already contains `TrackingConfig` default assertions, add:

```python
def test_tracking_config_exposes_node_probability_weights():
    from cellflow.tracking_ultrack.config import TrackingConfig

    cfg = TrackingConfig()

    assert cfg.quality_weight == 1.0
    assert cfg.circularity_weight == 0.25
    assert cfg.quality_exponent == 8.0
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
pytest tests/tracking_ultrack/test_config.py::test_tracking_config_exposes_node_probability_weights -q
```

Expected: fail with `AttributeError` or Pydantic validation/model-field failure because the fields do not exist yet.

- [ ] **Step 3: Add fields to `TrackingConfig`**

In `src/cellflow/tracking_ultrack/config.py`, update the resolve-from-validated node prior section:

```python
    # Resolve-from-validated node prior
    quality_weight: float = 1.0
    quality_exponent: float = 8.0
    circularity_weight: float = 0.25
    seed_weight: float = 0.5
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
pytest tests/tracking_ultrack/test_config.py::test_tracking_config_exposes_node_probability_weights -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/config.py tests/tracking_ultrack/test_config.py
git commit -m "feat: add ultrack node probability weight config"
```

---

### Task 2: Add Circularity Scoring

**Files:**
- Modify: `src/cellflow/tracking_ultrack/seed_prior.py`
- Test: `tests/tracking_ultrack/test_seed_prior.py`

- [ ] **Step 1: Add circularity helper tests**

In `tests/tracking_ultrack/test_seed_prior.py`, extend the import:

```python
from cellflow.tracking_ultrack.seed_prior import (
    compute_drop_frac,
    compute_mask_circularity,
    write_seed_prior_node_probs,
)
```

Add:

```python
def test_compute_mask_circularity_prefers_compact_masks():
    yy, xx = np.ogrid[:21, :21]
    disk = ((yy - 10) ** 2 + (xx - 10) ** 2) <= 6**2
    bar = np.zeros((21, 21), dtype=bool)
    bar[9:12, 2:19] = True

    disk_circularity = compute_mask_circularity(disk)
    bar_circularity = compute_mask_circularity(bar)

    assert 0.0 <= bar_circularity <= 1.0
    assert 0.0 <= disk_circularity <= 1.0
    assert disk_circularity > bar_circularity


def test_compute_mask_circularity_handles_empty_masks():
    assert compute_mask_circularity(np.zeros((5, 5), dtype=bool)) == 0.0
```

- [ ] **Step 2: Run the helper tests and verify they fail**

Run:

```bash
pytest tests/tracking_ultrack/test_seed_prior.py::test_compute_mask_circularity_prefers_compact_masks tests/tracking_ultrack/test_seed_prior.py::test_compute_mask_circularity_handles_empty_masks -q
```

Expected: fail because `compute_mask_circularity` does not exist.

- [ ] **Step 3: Implement `compute_mask_circularity`**

In `src/cellflow/tracking_ultrack/seed_prior.py`, add `import math` near the top and add this helper near `compute_drop_frac`:

```python
def compute_mask_circularity(mask: np.ndarray) -> float:
    from skimage.measure import perimeter

    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area == 0:
        return 0.0

    perimeter_px = float(perimeter(mask, neighborhood=4))
    if perimeter_px <= 0.0:
        return 0.0

    circularity = 4.0 * math.pi * float(area) / (perimeter_px * perimeter_px)
    return float(np.clip(circularity, 0.0, 1.0))
```

- [ ] **Step 4: Run the helper tests and verify they pass**

Run:

```bash
pytest tests/tracking_ultrack/test_seed_prior.py::test_compute_mask_circularity_prefers_compact_masks tests/tracking_ultrack/test_seed_prior.py::test_compute_mask_circularity_handles_empty_masks -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/seed_prior.py tests/tracking_ultrack/test_seed_prior.py
git commit -m "feat: compute ultrack node circularity"
```

---

### Task 3: Apply Weighted Node-Probability Formula

**Files:**
- Modify: `src/cellflow/tracking_ultrack/seed_prior.py`
- Test: `tests/tracking_ultrack/test_seed_prior.py`

- [ ] **Step 1: Update the existing scoring test expectation**

In `test_write_seed_prior_node_probs_uses_drop_quality_and_best_seed_affinity`, construct the config with explicit weights:

```python
    cfg = TrackingConfig(
        quality_weight=0.7,
        quality_exponent=2.0,
        circularity_weight=0.25,
        seed_weight=0.5,
        seed_sigma_space=25.0,
        seed_tau_time=2.0,
        seed_max_dt=5,
        seed_sigma_area=0.5,
    )
```

Update the expected value:

```python
    circularity = compute_mask_circularity(np.ones((4, 4), dtype=bool))
    expected_affinity = np.exp(-(1 / cfg.seed_tau_time))
    expected = (
        cfg.quality_weight * 1.0
        + cfg.circularity_weight * circularity
        + cfg.seed_weight * expected_affinity
    )
    assert candidate.node_prob == pytest.approx(expected)
```

- [ ] **Step 2: Add a zero-weight regression test**

Add:

```python
def test_write_seed_prior_node_probs_respects_zero_quality_and_circularity_weights(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        candidate = _make_node_row(1_000_001, 0, 3, 3, 7, 7)
        session.add(candidate)
        session.commit()

    image = np.full((1, 12, 12), 10.0, dtype=np.float32)
    image[0, 2:8, 2:8] = 5.0
    image[0, 3:7, 3:7] = 100.0
    image_path = tmp_path / "nucleus_zavg.tif"
    tifffile.imwrite(image_path, image)

    cfg = TrackingConfig(
        quality_weight=0.0,
        circularity_weight=0.0,
        seed_weight=0.0,
    )

    report = write_seed_prior_node_probs(tmp_path, image_path, cfg)

    assert report.scored == 1
    with Session(engine) as session:
        candidate = session.get(NodeDB, 1_000_001)
    assert candidate.node_prob == pytest.approx(0.0)
```

- [ ] **Step 3: Run the focused tests and verify they fail**

Run:

```bash
pytest tests/tracking_ultrack/test_seed_prior.py::test_write_seed_prior_node_probs_uses_drop_quality_and_best_seed_affinity tests/tracking_ultrack/test_seed_prior.py::test_write_seed_prior_node_probs_respects_zero_quality_and_circularity_weights -q
```

Expected: first test fails because the formula does not include explicit weights or circularity; second test fails because quality still contributes even when `quality_weight=0.0`.

- [ ] **Step 4: Update node-probability formula**

In `write_seed_prior_node_probs()`, replace:

```python
            node_prob = float(
                (drop_frac ** cfg.quality_exponent) + cfg.seed_weight * best_affinity
            )
```

with:

```python
            circularity = compute_mask_circularity(record.mask)
            node_prob = float(
                cfg.quality_weight * (drop_frac ** cfg.quality_exponent)
                + cfg.circularity_weight * circularity
                + cfg.seed_weight * best_affinity
            )
```

- [ ] **Step 5: Run all seed-prior tests**

Run:

```bash
pytest tests/tracking_ultrack/test_seed_prior.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/tracking_ultrack/seed_prior.py tests/tracking_ultrack/test_seed_prior.py
git commit -m "feat: weight ultrack node probability components"
```

---

### Task 4: Expose DB-Generation Controls

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Add UI exposure and config pass-through test**

In `tests/napari/test_nucleus_tracking_correction_layout.py`, add a test near the DB-generation tests:

```python
def test_db_gen_exposes_node_probability_weight_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert hasattr(widget, "db_gen_quality_weight_spin")
    assert hasattr(widget, "db_gen_quality_exp_spin")
    assert hasattr(widget, "db_gen_circularity_weight_spin")
    assert widget.db_gen_quality_weight_spin.value() == pytest.approx(1.0)
    assert widget.db_gen_quality_exp_spin.value() == pytest.approx(8.0)
    assert widget.db_gen_circularity_weight_spin.value() == pytest.approx(0.25)

    widget.db_gen_quality_weight_spin.setValue(0.8)
    widget.db_gen_quality_exp_spin.setValue(6.0)
    widget.db_gen_circularity_weight_spin.setValue(0.35)

    cfg = widget._db_gen_config_from_controls()

    assert cfg.quality_weight == pytest.approx(0.8)
    assert cfg.quality_exponent == pytest.approx(6.0)
    assert cfg.circularity_weight == pytest.approx(0.35)

    widget.deleteLater()
    viewer.close()
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_exposes_node_probability_weight_controls -q
```

Expected: fail because the new spin boxes do not exist.

- [ ] **Step 3: Add the controls**

In `src/cellflow/napari/nucleus_workflow_widget.py`, near the existing `db_gen_quality_exp_spin`, add:

```python
        self.db_gen_quality_weight_spin = QDoubleSpinBox()
        self.db_gen_quality_weight_spin.setRange(0.0, 10.0)
        self.db_gen_quality_weight_spin.setValue(1.0)
        self.db_gen_quality_weight_spin.setDecimals(2)
        self.db_gen_quality_weight_spin.setSingleStep(0.05)
        self.db_gen_quality_weight_spin.setToolTip(
            "Weight applied to signal-based segmentation quality before storing node_prob"
        )

        self.db_gen_circularity_weight_spin = QDoubleSpinBox()
        self.db_gen_circularity_weight_spin.setRange(0.0, 10.0)
        self.db_gen_circularity_weight_spin.setValue(0.25)
        self.db_gen_circularity_weight_spin.setDecimals(2)
        self.db_gen_circularity_weight_spin.setSingleStep(0.05)
        self.db_gen_circularity_weight_spin.setToolTip(
            "Weight applied to shape circularity before storing node_prob"
        )
```

Replace the DB grid row for quality with two rows:

```python
        add_block_pair_row(db_gen_grid, 5, "Quality Weight:", _compact(self.db_gen_quality_weight_spin), "Quality Exp:", _compact(self.db_gen_quality_exp_spin))
        add_block_pair_row(db_gen_grid, 6, "Circularity Weight:", _compact(self.db_gen_circularity_weight_spin), "", QWidget())
```

- [ ] **Step 4: Pass controls into `TrackingConfig`**

In `_db_gen_config_from_controls()`, include:

```python
            quality_weight=self.db_gen_quality_weight_spin.value(),
            quality_exponent=self.db_gen_quality_exp_spin.value(),
            circularity_weight=self.db_gen_circularity_weight_spin.value(),
```

- [ ] **Step 5: Run the focused test and verify it passes**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_exposes_node_probability_weight_controls -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat: expose ultrack node probability weights"
```

---

### Task 5: Persist UI State And Terminal Script Values

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Test: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Extend state persistence test**

In `test_db_gen_controls_persist_through_state`, add before `state = widget.get_state()`:

```python
    widget.db_gen_quality_weight_spin.setValue(0.8)
    widget.db_gen_circularity_weight_spin.setValue(0.35)
```

Add after restore:

```python
    assert abs(widget.db_gen_quality_weight_spin.value() - 0.8) < 0.01
    assert abs(widget.db_gen_circularity_weight_spin.value() - 0.35) < 0.01
```

- [ ] **Step 2: Add terminal script assertions**

In the existing terminal DB-generation test that reads the launched script, add:

```python
    assert "quality_weight=" in script
    assert "circularity_weight=" in script
```

- [ ] **Step 3: Run the focused tests and verify they fail**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_controls_persist_through_state tests/napari/test_nucleus_tracking_correction_layout.py::test_resolve_terminal_script_uses_canonical_segment_not_hypotheses_h5 -q
```

Expected: fail because state and terminal script do not include the new fields yet.

- [ ] **Step 4: Persist the new controls**

In `get_state()`, update the `"db_generation"` dict:

```python
                "quality_weight":   self.db_gen_quality_weight_spin.value(),
                "quality_exponent": self.db_gen_quality_exp_spin.value(),
                "circularity_weight": self.db_gen_circularity_weight_spin.value(),
```

In `set_state()`, update the `"db_generation"` restore block:

```python
            if "quality_weight" in dbg: self.db_gen_quality_weight_spin.setValue(dbg["quality_weight"])
            if "quality_exponent" in dbg: self.db_gen_quality_exp_spin.setValue(dbg["quality_exponent"])
            if "circularity_weight" in dbg: self.db_gen_circularity_weight_spin.setValue(dbg["circularity_weight"])
```

- [ ] **Step 5: Update terminal script config**

In `_on_db_gen_terminal()`, add the fields to the generated `TrackingConfig` block:

```python
            f"        quality_weight={cfg.quality_weight},\n"
            f"        quality_exponent={cfg.quality_exponent},\n"
            f"        circularity_weight={cfg.circularity_weight},\n"
```

Keep `quality_exponent` present only once.

- [ ] **Step 6: Run focused tests and verify they pass**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_db_gen_controls_persist_through_state tests/napari/test_nucleus_tracking_correction_layout.py::test_resolve_terminal_script_uses_canonical_segment_not_hypotheses_h5 -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "feat: persist ultrack node probability weights"
```

---

### Task 6: Final Verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run focused tracking tests**

Run:

```bash
pytest tests/tracking_ultrack/test_config.py tests/tracking_ultrack/test_seed_prior.py tests/tracking_ultrack/test_db_build.py -q
```

Expected: pass.

- [ ] **Step 2: Run focused napari DB-generation tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py -k "db_gen or db_generation or resolve_terminal" -q
```

Expected: pass. If unrelated existing tests in the selected set fail, record the exact failures and verify the new tests individually.

- [ ] **Step 3: Inspect git diff**

Run:

```bash
git diff -- src/cellflow/tracking_ultrack/config.py src/cellflow/tracking_ultrack/seed_prior.py src/cellflow/napari/nucleus_workflow_widget.py tests/tracking_ultrack/test_config.py tests/tracking_ultrack/test_seed_prior.py tests/napari/test_nucleus_tracking_correction_layout.py
```

Expected: diff only contains config fields, circularity scoring, UI/control pass-through, state persistence, and tests for this feature.

- [ ] **Step 4: Commit any final fixes**

If final verification required small fixes:

```bash
git add src/cellflow/tracking_ultrack/config.py src/cellflow/tracking_ultrack/seed_prior.py src/cellflow/napari/nucleus_workflow_widget.py tests/tracking_ultrack/test_config.py tests/tracking_ultrack/test_seed_prior.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "test: verify ultrack node probability weighting"
```

If no fixes were needed, no final commit is required.

---

## Self-Review

- Spec coverage: The plan covers explicit quality weighting, enabled circularity weighting, UI exposure, terminal script parity, state persistence, and tests.
- Placeholder scan: No placeholder steps remain.
- Type consistency: The config fields are consistently named `quality_weight`, `quality_exponent`, and `circularity_weight` across config, scoring, UI state, terminal script, and tests.
