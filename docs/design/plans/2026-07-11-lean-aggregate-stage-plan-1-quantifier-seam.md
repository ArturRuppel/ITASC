# Lean Aggregate Stage — Plan 1: Quantifier Seam (compute at pool time)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the cheap quantifiers compute their pooled table in memory at aggregate time instead of persisting a per-position file, so only `contacts` (`contact_analysis.h5`) is written per position.

**Architecture:** Add one seam method, `Quantifier.compute_object_table(inputs, *, params) -> Mapping | None`, implemented on each pooled quantifier by extracting the in-memory compute its `build()` already runs before writing. The pooling layer (`shape_tables`) calls `compute_object_table` instead of reading a per-position artifact, threading the shared build params. `build_quantities` persists only *producers* (`bool(q.produces)` — in practice `contacts`). The existing per-position writers are **kept** (the live napari studio still calls them, per the spec's A5 scope boundary); this plan changes the canonical pipeline only.

**Tech Stack:** Python, numpy, pandas, pytest, tifffile, skimage.

**Spec:** `docs/superpowers/specs/2026-07-11-lean-aggregate-stage-design.md` (Workstream A).

---

## File Structure

- `src/cellflow/contact_analysis/quantifier.py` — add `compute_object_table` to the `Quantifier` base.
- `src/cellflow/contact_analysis/shape/core.py` — extract `compute_object_shape` (pure compute) out of `build_object_shape`.
- `src/cellflow/contact_analysis/shape/relational.py` — extract `compute_relational_table` out of `build_relational_shape`.
- `src/cellflow/contact_analysis/quantifiers/{cell_shape,nucleus_shape,shape_relational,cell_dynamics,nucleus_dynamics,cell_density,neighbor_count,signed_contact_length}.py` — implement `compute_object_table`.
- `src/cellflow/contact_analysis/shape_tables.py` — pool via `compute_object_table`, add `params` to `build_table` / `aggregate`, relocate the required-build-param gate.
- `src/cellflow/contact_analysis/pipeline.py` — `build_quantities` persists only producers; `run()` threads `cfg.params` into `aggregate`.
- Tests: one `test_compute_object_table_*` per quantifier (golden: equals the existing `build`→`object_table` round-trip); rewrite `tests/contact_analysis/test_shape_tables.py` to drive pooling through `compute_object_table`; extend `tests/contact_analysis/test_pipeline.py`.

**Note on nucleus twins:** `nucleus_shape` mirrors `cell_shape` and `nucleus_dynamics` mirrors `cell_dynamics` (they differ only by which label field they read and the output filename). Each twin gets the same treatment in the same task as its cell counterpart.

---

## Task 1: Add the `compute_object_table` seam method

**Files:**
- Modify: `src/cellflow/contact_analysis/quantifier.py` (the `Quantifier` class, after `object_table`, around line 149-156)
- Test: `tests/contact_analysis/test_quantifier.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/contact_analysis/test_quantifier.py`:

```python
def test_compute_object_table_default_raises():
    from cellflow.contact_analysis.quantifier import PositionInputs, Quantifier

    class _Bare(Quantifier):
        quantity_id = ""  # not registered

    import pytest
    with pytest.raises(NotImplementedError):
        _Bare().compute_object_table(PositionInputs(position_dir=__import__("pathlib").Path(".")))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/contact_analysis/test_quantifier.py::test_compute_object_table_default_raises -v`
Expected: FAIL with `AttributeError: 'Quantifier' object has no attribute 'compute_object_table'`.

- [ ] **Step 3: Add the method**

In `src/cellflow/contact_analysis/quantifier.py`, add after the `object_table` method (keep the imports `Mapping`, `Any` already present):

```python
    def compute_object_table(
        self, inputs: PositionInputs, *, params: dict | None = None
    ) -> Mapping[str, Any] | None:
        """The pooled tidy table for one position, computed directly from *inputs*.

        A **pooled** quantifier (one that declares ``table_keys``) implements this:
        it returns the same column-major table that ``object_table`` used to return
        after a disk round-trip, but never touches disk. ``None`` when this position
        yields no rows. Producers (``contacts``) are not pooled and do not implement
        it. *params* carries the shared build knobs (e.g. ``fov_area_mm2``) for the
        quantifiers that opt in; per-position values (pixel size, frame interval)
        arrive via *inputs*.
        """
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/contact_analysis/test_quantifier.py::test_compute_object_table_default_raises -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/contact_analysis/quantifier.py tests/contact_analysis/test_quantifier.py
git commit -m "feat(quantifier): add compute_object_table seam method"
```

---

## Task 2: Extract `compute_object_shape` from `build_object_shape`

**Files:**
- Modify: `src/cellflow/contact_analysis/shape/core.py:59-104`
- Test: `tests/contact_analysis/test_shape_core.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/contact_analysis/test_shape_core.py` (imports: `numpy as np`, `tifffile`, `from pathlib import Path`):

```python
def test_compute_object_shape_matches_build_roundtrip(tmp_path):
    from cellflow.contact_analysis.shape.core import (
        build_object_shape,
        compute_object_shape,
        read_shape_table,
    )
    frame = np.zeros((10, 10), dtype=np.uint16)
    frame[1:5, 1:5] = 1
    frame[5:9, 5:9] = 2
    labels = tmp_path / "cells.tif"
    tifffile.imwrite(labels, frame[np.newaxis, ...])

    out = tmp_path / "cell_shape.csv"
    build_object_shape(labels, out, pixel_size_um=0.5, object_key="cell_id")
    expected = read_shape_table(out)

    got = compute_object_shape(labels, pixel_size_um=0.5, object_key="cell_id")

    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_allclose(np.asarray(got[key], float), np.asarray(expected[key], float))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/contact_analysis/test_shape_core.py::test_compute_object_shape_matches_build_roundtrip -v`
Expected: FAIL with `ImportError: cannot import name 'compute_object_shape'`.

- [ ] **Step 3: Extract the compute**

In `src/cellflow/contact_analysis/shape/core.py`, add a new function and make `build_object_shape` call it. The compute is the existing lines that read the stack and extract columns:

```python
def compute_object_shape(
    label_path: str | Path,
    *,
    pixel_size_um: float,
    object_key: str = "cell_id",
) -> dict[str, np.ndarray]:
    """Per-object shape descriptors for every frame, as a column-major table.

    The pure compute behind :func:`build_object_shape` — no file written. Columns:
    ``frame``, *object_key*, then :data:`DESCRIPTOR_COLUMNS`."""
    pixel_size_um = float(pixel_size_um)
    if not pixel_size_um > 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um!r}")
    label_stack = read_label_stack(Path(label_path))
    return _extract_shape_columns(label_stack, pixel_size_um, object_key)
```

Then replace the body of `build_object_shape` between reading and writing so it reuses the extract. Change lines 86-90 (the `label_stack = ...` / `_report_progress` / `columns = _extract_shape_columns(...)` / `_report_progress`) to:

```python
    label_stack = read_label_stack(label_path)
    _report_progress(progress_cb, 1, total, "read labels")

    columns = _extract_shape_columns(label_stack, pixel_size_um, object_key)
    _report_progress(progress_cb, 2, total, "extract shape")
```

(No behavioural change to `build_object_shape`; `compute_object_shape` just exposes the same `_extract_shape_columns` result for the in-memory path. Leave `build_object_shape` otherwise untouched.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/contact_analysis/test_shape_core.py -v`
Expected: PASS (the new test plus all existing shape-core tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/contact_analysis/shape/core.py tests/contact_analysis/test_shape_core.py
git commit -m "refactor(shape): extract compute_object_shape from build_object_shape"
```

---

## Task 3: `compute_object_table` for cell_shape and nucleus_shape

**Files:**
- Modify: `src/cellflow/contact_analysis/quantifiers/cell_shape.py`, `src/cellflow/contact_analysis/quantifiers/nucleus_shape.py`
- Test: `tests/contact_analysis/test_shape_quantifier.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/contact_analysis/test_shape_quantifier.py`:

```python
"""compute_object_table for the shape quantifiers matches the build round-trip."""
import numpy as np
import tifffile

from cellflow.contact_analysis.quantifier import PositionInputs
from cellflow.contact_analysis.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.contact_analysis.quantifiers.nucleus_shape import NucleusShapeQuantifier


def _two_cell_stack(tmp_path, name):
    frame = np.zeros((12, 12), dtype=np.uint16)
    frame[1:5, 1:5] = 1
    frame[6:10, 6:10] = 2
    path = tmp_path / name
    tifffile.imwrite(path, frame[np.newaxis, ...])
    return path


def test_cell_shape_compute_matches_build(tmp_path):
    labels = _two_cell_stack(tmp_path, "cells.tif")
    q = CellShapeQuantifier()
    inputs = PositionInputs(position_dir=tmp_path, cell_labels_path=labels, pixel_size_um=0.5)
    out = q.default_output(inputs)
    q.build(inputs, out)
    expected = q.object_table(out)

    got = q.compute_object_table(inputs)

    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_allclose(np.asarray(got[key], float), np.asarray(expected[key], float))


def test_nucleus_shape_compute_matches_build(tmp_path):
    labels = _two_cell_stack(tmp_path, "nuclei.tif")
    q = NucleusShapeQuantifier()
    inputs = PositionInputs(position_dir=tmp_path, nucleus_labels_path=labels, pixel_size_um=0.5)
    out = q.default_output(inputs)
    q.build(inputs, out)
    expected = q.object_table(out)

    got = q.compute_object_table(inputs)

    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_allclose(np.asarray(got[key], float), np.asarray(expected[key], float))
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/contact_analysis/test_shape_quantifier.py -v`
Expected: FAIL with `NotImplementedError` (base method).

- [ ] **Step 3: Implement**

In `src/cellflow/contact_analysis/quantifiers/cell_shape.py`, add the import and method (keep `build`/`read`/`object_table` as-is):

```python
from cellflow.contact_analysis.shape import build_object_shape, compute_object_shape, read_shape_table
```

```python
    def compute_object_table(self, inputs, *, params=None):
        return compute_object_shape(
            inputs.cell_labels_path,
            pixel_size_um=inputs.pixel_size_um,
            object_key="cell_id",
        )
```

(Check that `compute_object_shape` is exported from `shape/__init__.py`; if not, add it to that module's imports/`__all__` alongside `build_object_shape`.)

In `src/cellflow/contact_analysis/quantifiers/nucleus_shape.py`, the same but reading `inputs.nucleus_labels_path`:

```python
    def compute_object_table(self, inputs, *, params=None):
        return compute_object_shape(
            inputs.nucleus_labels_path,
            pixel_size_um=inputs.pixel_size_um,
            object_key="cell_id",
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/contact_analysis/test_shape_quantifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/contact_analysis/quantifiers/cell_shape.py src/cellflow/contact_analysis/quantifiers/nucleus_shape.py src/cellflow/contact_analysis/shape/__init__.py tests/contact_analysis/test_shape_quantifier.py
git commit -m "feat(shape): compute_object_table for cell/nucleus shape"
```

---

## Task 4: `compute_relational_table` + `compute_object_table` for shape_relational

**Files:**
- Modify: `src/cellflow/contact_analysis/shape/relational.py` (the `build_relational_shape` function around lines 60-105) and its `shape/__init__.py` export
- Modify: `src/cellflow/contact_analysis/quantifiers/shape_relational.py`
- Test: `tests/contact_analysis/test_shape_relational.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/contact_analysis/test_shape_relational.py` (match the existing fixture style in that file; if it already has a paired cell+nucleus stack helper, reuse it):

```python
def test_relational_compute_matches_build(tmp_path):
    import numpy as np, tifffile
    from cellflow.contact_analysis.quantifier import PositionInputs
    from cellflow.contact_analysis.quantifiers.shape_relational import ShapeRelationalQuantifier

    cell = np.zeros((12, 12), dtype=np.uint16); cell[1:6, 1:6] = 1
    nuc = np.zeros((12, 12), dtype=np.uint16); nuc[2:5, 2:5] = 1
    cp = tmp_path / "cells.tif"; np_path = tmp_path / "nuclei.tif"
    tifffile.imwrite(cp, cell[np.newaxis, ...]); tifffile.imwrite(np_path, nuc[np.newaxis, ...])

    q = ShapeRelationalQuantifier()
    inputs = PositionInputs(
        position_dir=tmp_path, cell_labels_path=cp, nucleus_labels_path=np_path, pixel_size_um=0.5
    )
    out = q.default_output(inputs)
    q.build(inputs, out)
    expected = q.object_table(out)

    got = q.compute_object_table(inputs)

    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_allclose(np.asarray(got[key], float), np.asarray(expected[key], float))
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/contact_analysis/test_shape_relational.py::test_relational_compute_matches_build -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Extract compute + implement**

In `src/cellflow/contact_analysis/shape/relational.py`, add a pure-compute function that returns the columns `build_relational_shape` computes at lines 79-92 (read both stacks, props, join, `_columns_from_rows`):

```python
def compute_relational_table(
    cell_labels_path: str | Path,
    nucleus_labels_path: str | Path,
    *,
    pixel_size_um: float,
) -> dict[str, np.ndarray]:
    """The relational per-(frame, id) table, computed in memory (no file written)."""
    pixel_size_um = float(pixel_size_um)
    if not pixel_size_um > 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um!r}")
    cell_stack = read_label_stack(Path(cell_labels_path))
    nucleus_stack = read_label_stack(Path(nucleus_labels_path))
    cell_props = _object_props(cell_stack, pixel_size_um)
    nucleus_props = _object_props(nucleus_stack, pixel_size_um)
    rows, _dropped = _join_rows(cell_props, nucleus_props)
    return _columns_from_rows(rows)
```

Refactor `build_relational_shape` to call it (replacing its lines 79-92 read/props/join/columns block):

```python
    cell_stack = read_label_stack(cell_labels_path)
    nucleus_stack = read_label_stack(nucleus_labels_path)
    _report_progress(progress_cb, 1, total, "read labels")
    cell_props = _object_props(cell_stack, pixel_size_um)
    nucleus_props = _object_props(nucleus_stack, pixel_size_um)
    _report_progress(progress_cb, 2, total, "extract shape")
    rows, dropped = _join_rows(cell_props, nucleus_props)
    _report_progress(progress_cb, 3, total, f"join ({len(rows)} paired, {dropped} unpaired dropped)")
    columns = _columns_from_rows(rows)
```

Export `compute_relational_table` from `shape/__init__.py` (add to its imports/`__all__`).

In `src/cellflow/contact_analysis/quantifiers/shape_relational.py`, import and implement:

```python
from cellflow.contact_analysis.shape import compute_relational_table
```

```python
    def compute_object_table(self, inputs, *, params=None):
        return compute_relational_table(
            inputs.cell_labels_path,
            inputs.nucleus_labels_path,
            pixel_size_um=inputs.pixel_size_um,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/contact_analysis/test_shape_relational.py -v`
Expected: PASS (new test + existing ones).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/contact_analysis/shape/relational.py src/cellflow/contact_analysis/shape/__init__.py src/cellflow/contact_analysis/quantifiers/shape_relational.py tests/contact_analysis/test_shape_relational.py
git commit -m "feat(shape): compute_relational_table + compute_object_table for shape_relational"
```

---

## Task 5: `compute_object_table` for cell_dynamics and nucleus_dynamics

**Files:**
- Modify: `src/cellflow/contact_analysis/quantifiers/cell_dynamics.py`, `src/cellflow/contact_analysis/quantifiers/nucleus_dynamics.py`
- Test: `tests/contact_analysis/test_dynamics_quantifier.py`

The dynamics `object_table` is only the instantaneous table, which is
`instantaneous_table(extract_trajectories(labels, pixel_size_um=...), time_interval_s=...)` —
computed in memory without the msd/dac/collective sub-tables or the h5.

- [ ] **Step 1: Write the failing test**

Add to `tests/contact_analysis/test_dynamics_quantifier.py` (reuse the file's `_moving_disk_stack`):

```python
def test_cell_dynamics_compute_matches_build(tmp_path):
    centers = [(40, 10 + 2 * i) for i in range(16)]
    labels = tmp_path / "cells.tif"
    tifffile.imwrite(labels, _moving_disk_stack(centers))
    q = CellDynamicsQuantifier()
    inputs = PositionInputs(
        position_dir=tmp_path, cell_labels_path=labels, pixel_size_um=0.5, time_interval_s=2.0
    )
    out = q.default_output(inputs)
    q.build(inputs, out)
    expected = q.object_table(out)

    got = q.compute_object_table(inputs)

    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_allclose(
            np.asarray(got[key], float), np.asarray(expected[key], float), equal_nan=True
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/contact_analysis/test_dynamics_quantifier.py::test_cell_dynamics_compute_matches_build -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement**

In `src/cellflow/contact_analysis/quantifiers/cell_dynamics.py`, add imports and method:

```python
from cellflow.contact_analysis.dynamics import (
    TrackDynamics,
    build_track_dynamics,
    read_instantaneous_table,
    read_track_dynamics,
)
from cellflow.contact_analysis.dynamics.kinematics import instantaneous_table
from cellflow.contact_analysis.dynamics.trajectories import extract_trajectories
```

```python
    def compute_object_table(self, inputs, *, params=None):
        trajectories = extract_trajectories(
            inputs.cell_labels_path, pixel_size_um=inputs.pixel_size_um
        )
        return instantaneous_table(trajectories, time_interval_s=inputs.time_interval_s)
```

In `src/cellflow/contact_analysis/quantifiers/nucleus_dynamics.py`, the same reading `inputs.nucleus_labels_path`. (Verify `extract_trajectories` and `instantaneous_table` are importable at those paths; the dynamics `store.build_track_dynamics` imports them the same way — reuse whatever import path `store.py` uses.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/contact_analysis/test_dynamics_quantifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/contact_analysis/quantifiers/cell_dynamics.py src/cellflow/contact_analysis/quantifiers/nucleus_dynamics.py tests/contact_analysis/test_dynamics_quantifier.py
git commit -m "feat(dynamics): compute_object_table (instantaneous table) for cell/nucleus dynamics"
```

---

## Task 6: `compute_object_table` for cell_density

**Files:**
- Modify: `src/cellflow/contact_analysis/quantifiers/cell_density.py`
- Test: `tests/contact_analysis/test_cell_density_quantifier.py`

`cell_density` reads `fov_area_mm2` from `params` (it sets `wants_build_params = True`).

- [ ] **Step 1: Write the failing test**

Add to `tests/contact_analysis/test_cell_density_quantifier.py` (reuse `_inputs` / `_labels_stack`):

```python
def test_cell_density_compute_matches_build(tmp_path):
    q = _quantifier()
    inputs = _inputs(tmp_path)
    out = q.default_output(inputs)
    q.build(inputs, out, params={"fov_area_mm2": 2.0})
    expected = q.object_table(out)

    got = q.compute_object_table(inputs, params={"fov_area_mm2": 2.0})

    assert set(got) == set(expected)
    assert dict(zip(got["label"].tolist(), got["n_cells"].tolist()))["all"] == 4


def test_cell_density_compute_missing_fov_raises(tmp_path):
    import pytest
    q = _quantifier()
    with pytest.raises(ValueError, match="field-of-view"):
        q.compute_object_table(_inputs(tmp_path), params={})
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/contact_analysis/test_cell_density_quantifier.py -v -k compute`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement**

In `src/cellflow/contact_analysis/quantifiers/cell_density.py`, add a `compute_object_table` that reuses the existing compute (extract the `build` body's compute, keep `build` calling `persist`):

```python
    def compute_object_table(self, inputs, *, params=None):
        fov = (params or {}).get("fov_area_mm2")
        if not (isinstance(fov, (int, float)) and fov > 0):
            raise ValueError(
                "Cell density requires a field-of-view area — set 'FOV area (mm²)' "
                "in Parameters before building."
            )
        frame_cells = _frame_cells_from_labels(inputs.cell_labels_path)
        return cell_density(frame_cells, fov_area_mm2=float(fov))
```

Optionally simplify `build` to `return persist(output_path, self.compute_object_table(inputs, params=params))` (DRY; keeps behaviour identical). Do so only if it does not change the raised message.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/contact_analysis/test_cell_density_quantifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/contact_analysis/quantifiers/cell_density.py tests/contact_analysis/test_cell_density_quantifier.py
git commit -m "feat(density): compute_object_table for cell_density"
```

---

## Task 7: `compute_object_table` for neighbor_count and signed_contact_length

**Files:**
- Modify: `src/cellflow/contact_analysis/quantifiers/neighbor_count.py`, `src/cellflow/contact_analysis/quantifiers/signed_contact_length.py`
- Test: `tests/contact_analysis/test_contacts_derived_quantifiers.py`

These read the persisted `contact_analysis.h5` via `derived.load_analysis(inputs)`.

- [ ] **Step 1: Write the failing test**

Look at `tests/contact_analysis/test_contacts_derived_quantifiers.py` for its existing fixture that produces a `contact_analysis.h5` and a `PositionInputs` pointing at it (call it `inputs_with_contacts(tmp_path)` here — reuse whatever that file already builds). Add:

```python
def test_neighbor_count_compute_matches_build(tmp_path):
    from cellflow.contact_analysis.quantifiers.neighbor_count import NeighborCountQuantifier
    inputs = _inputs_with_contacts(tmp_path)  # existing helper in this file
    q = NeighborCountQuantifier()
    out = q.default_output(inputs)
    q.build(inputs, out)
    expected = q.object_table(out)

    got = q.compute_object_table(inputs)

    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(np.asarray(got[key]), np.asarray(expected[key]))


def test_signed_contact_length_compute_matches_build(tmp_path):
    from cellflow.contact_analysis.quantifiers.signed_contact_length import (
        SignedContactLengthQuantifier,
    )
    inputs = _inputs_with_contacts(tmp_path)
    q = SignedContactLengthQuantifier()
    out = q.default_output(inputs)
    q.build(inputs, out)
    expected = q.object_table(out)

    got = q.compute_object_table(inputs)

    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(np.asarray(got[key]), np.asarray(expected[key]))
```

If the file has no reusable `_inputs_with_contacts`, build one from its existing setup (it must already create a contacts h5 to test the current `build`). Match that file's idiom exactly.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/contact_analysis/test_contacts_derived_quantifiers.py -v -k compute`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement**

`neighbor_count.py` — add:

```python
from cellflow.contact_analysis.contacts.neighborhood import cell_neighbor_counts
```

```python
    def compute_object_table(self, inputs, *, params=None):
        return dict(cell_neighbor_counts(derived.load_analysis(inputs)))
```

`signed_contact_length.py` — factor the build's compute (including the `contact_type` blank→`"unlabelled"` normalization) into `compute_object_table`, and have `build` persist its result:

```python
    def compute_object_table(self, inputs, *, params=None):
        analysis = derived.load_analysis(inputs)
        table = dict(
            signed_central_junction_lengths(analysis, pixel_size_um=inputs.pixel_size_um)
        )
        if "contact_type" in table:
            ct = np.asarray(table["contact_type"], dtype=object)
            ct[ct == ""] = "unlabelled"
            table["contact_type"] = ct
        return table
```

Then simplify `build` to:

```python
    def build(self, inputs, output_path, *, params=None, progress_cb=None):
        return derived.persist(output_path, self.compute_object_table(inputs, params=params))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/contact_analysis/test_contacts_derived_quantifiers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/contact_analysis/quantifiers/neighbor_count.py src/cellflow/contact_analysis/quantifiers/signed_contact_length.py tests/contact_analysis/test_contacts_derived_quantifiers.py
git commit -m "feat(contacts-derived): compute_object_table for neighbor_count + signed_contact_length"
```

---

## Task 8: Pool via `compute_object_table`, threading params

**Files:**
- Modify: `src/cellflow/contact_analysis/shape_tables.py` (`_position_frame` lines 158-185; `build_table` line 103; `aggregate` lines 227-248)
- Test: `tests/contact_analysis/test_shape_tables.py` (rewrite the injection helper)

This is the integration task: pooling stops reading per-position files and calls `compute_object_table` instead, so the shared build params must reach it here.

- [ ] **Step 1: Rewrite the test injection helper (write the failing form first)**

In `tests/contact_analysis/test_shape_tables.py`, replace `_write_object_table` (which wrote a per-position CSV) with a monkeypatch-based stub. Add a `monkeypatch` fixture parameter to each test and install the stub. New helper:

```python
from pathlib import Path

def _stub_compute(monkeypatch, quantifier_cls, tables_by_pid):
    """Make the pooled quantifier return a fixed table per position folder name."""
    def compute(self, inputs, *, params=None):
        return tables_by_pid.get(Path(inputs.position_dir).name)
    monkeypatch.setattr(quantifier_cls, "compute_object_table", compute)
```

**Important — the relocated build-param gate:** `_position_frame` now checks
`missing_build_params` *before* calling `compute_object_table` (the gate moved out of
`build_quantities`). `CellShapeQuantifier` declares `required_build_params =
{"pixel_size_um": ...}`, so a record without `pixel_size_um` gets gated out and pools
empty even with the stub installed. Stamp it in the `_record` helper so cell-shape
records pass the gate — change `_record` to include `"pixel_size_um": 0.5` in the dict
it returns. (`NeighborCountQuantifier` has no `required_build_params`, so it is
unaffected.)

Mechanical transform for **every** test in the file: where it previously did
`_write_object_table(cs, rec_a, table_a)` / `_write_object_table(cs, rec_b, table_b)`,
instead collect `{Path(rec_a["position_path"]).name: table_a, Path(rec_b["position_path"]).name: table_b}`
and call `_stub_compute(monkeypatch, CellShapeQuantifier, that_dict)` once (same for
`NeighborCountQuantifier`). Assertions are unchanged. Worked example (the rewrite of
`test_two_positions_pool_into_one_table`):

```python
def test_two_positions_pool_into_one_table(tmp_path, monkeypatch):
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _stub_compute(monkeypatch, CellShapeQuantifier, {
        "a": _cell_shape_table([1, 2], [0, 1]),
        "b": _cell_shape_table([1], [0]),
    })

    df = build_table("cell_shape", [rec_a, rec_b])

    assert set(df["position_id"]) == {"a", "b"}
    assert {"condition", "date", "position_id", "frame", "cell_id"} <= set(df.columns)
    assert "cell_shape.area_um2" in df.columns
    assert len(df) == 4 + 1
```

Apply the same swap to the other pooling tests (`test_experiment_id_broadcast_onto_pooled_rows`, `test_free_form_column_broadcast_onto_pooled_rows`, `test_build_table_assigns_deterministic_row_id`, `test_row_id_distinguishes_same_cell_across_positions`, `test_each_quantity_is_its_own_table`, `test_quantity_built_for_only_some_positions`, `test_ids_do_not_collide_across_positions`, `test_pooled_table_has_no_class_label`, `test_aggregate_writes_csv_and_rewrites_whole`, `test_read_table_restores_integer_keys`). `test_aggregate_skips_empty_tables`, `test_registry_*`, and `test_catalogue_root_*` need no stub (they assert on the empty / registry / path cases). Delete the now-unused `output_for_record` import and the old `_write_object_table`.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/contact_analysis/test_shape_tables.py -v`
Expected: FAIL — `build_table` still reads files via `object_table`, so the stubbed compute is ignored and tables come back empty.

- [ ] **Step 3: Rewrite `_position_frame` + thread params**

In `src/cellflow/contact_analysis/shape_tables.py`:

Replace the imports at line 42-43:

```python
from .quantifier import Quantifier, available_quantifiers
from .records import available_fields, position_inputs_from_record, record_build_params
```

Also add `Mapping` to the `collections.abc` import at line 35 (currently
`from collections.abc import Iterable, Sequence` → add `Mapping`).

Rewrite `_position_frame` to compute in memory (drop `output_for_record` / `is_built`),
using the shared helpers `available_fields` / `record_build_params` moved to
`records.py` below:

```python
def _position_frame(
    record: dict,
    quantifiers: list[Quantifier],
    keys: tuple[str, ...],
    params: Mapping[str, object] | None,
) -> pd.DataFrame | None:
    """Outer-join (within one position) every co-targeting quantity on the table's
    keys, namespacing each quantity's value columns. ``None`` when no targeting
    quantity yields a table for this position."""
    inputs = position_inputs_from_record(record)
    merged: pd.DataFrame | None = None
    for quantifier in quantifiers:
        if not set(quantifier.requires) <= available_fields(inputs):
            continue
        if quantifier.missing_build_params(record_build_params(quantifier, record, params)):
            continue
        table = quantifier.compute_object_table(inputs, params=dict(params) if params else None)
        if not table:
            continue
        df = pd.DataFrame({k: np.asarray(v) for k, v in dict(table).items()})
        present_keys = [k for k in keys if k in df.columns]
        value_cols = [c for c in df.columns if c not in present_keys]
        df = df[[*present_keys, *value_cols]].rename(
            columns={c: f"{quantifier.quantity_id}.{c}" for c in value_cols}
        )
        if merged is None:
            merged = df
        else:
            on = [k for k in keys if k in merged.columns and k in df.columns]
            merged = merged.merge(df, on=on, how="outer") if on else pd.concat(
                [merged, df], ignore_index=True
            )
    return merged
```

`_position_frame` uses `available_fields` and `record_build_params`, which today live as private `_available_fields` / `_record_build_params` in `pipeline.py`. To share them without a circular import, **move** them into `records.py` (Qt-free, already imported by both `pipeline.py` and `shape_tables.py`) as public functions, and add them to `records.__all__`:

```python
# records.py — add (moved from pipeline.py, made public):
from collections.abc import Mapping
from dataclasses import fields as _dataclass_fields

def available_fields(inputs: PositionInputs) -> set[str]:
    """Populated (non-None) PositionInputs field names — the satisfied prerequisites."""
    return {f.name for f in _dataclass_fields(inputs) if getattr(inputs, f.name) is not None}

def record_build_params(quantifier, record, params):
    """Shared *params* overlaid with the record's own required-build-param values."""
    merged = dict(params or {})
    for key in quantifier.required_build_params:
        value = record.get(key)
        if value is not None:
            merged[key] = value
    return merged
```

Then update `pipeline.py`: delete its private `_available_fields` / `_record_build_params` and import `available_fields` / `record_build_params` from `.records`, updating its call sites (in Task 9's `build_quantities` rewrite they are already referenced by the public names).

Thread `params` through `build_table` and `aggregate`:

```python
def build_table(name: str, records: Iterable[dict], *, params: Mapping[str, object] | None = None) -> pd.DataFrame:
    ...
    for record in records:
        merged = _position_frame(record, quantifiers, spec.keys, params)
        ...

def aggregate(records, out_dir=None, *, params: Mapping[str, object] | None = None) -> dict[str, Path]:
    ...
    for name in shape_table_registry():
        df = build_table(name, records, params=params)
        ...
```

(Existing calls `build_table("cell_shape", [rec])` still work — `params` defaults to `None`.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/contact_analysis/test_shape_tables.py tests/contact_analysis/test_pipeline.py -v`
Expected: PASS (shape-tables rewritten; pipeline still green because `run()` passes params in Task 9 — until then `aggregate` params defaults to None, and existing pipeline tests that build real per-position files still read them via the compute path, which recomputes from the same inputs).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/contact_analysis/shape_tables.py src/cellflow/contact_analysis/records.py src/cellflow/contact_analysis/pipeline.py tests/contact_analysis/test_shape_tables.py
git commit -m "refactor(aggregate): pool via compute_object_table, thread build params"
```

---

## Task 9: `build_quantities` persists only producers; `run()` threads params to aggregate

**Files:**
- Modify: `src/cellflow/contact_analysis/pipeline.py` (`build_quantities` lines 75-147; `run` lines 299-325)
- Test: `tests/contact_analysis/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/contact_analysis/test_pipeline.py`. Reuse the file's existing catalog/labels fixtures (it already exercises `run()` end to end). The new assertions: after a run, each position folder holds `contact_analysis.h5` but **no** pooled per-position file (`cell_shape.csv`, `cell_dynamics.h5`, etc.), and the pooled tables still contain the cheap quantities. Sketch (adapt fixture calls to the file's helpers):

```python
def test_run_persists_only_contacts_per_position(tmp_path):
    from cellflow.contact_analysis.pipeline import run, author_config

    records = _catalog_two_positions(tmp_path)   # existing helper: real cell+nucleus labels
    cfg = author_config(
        tmp_path / "proj", records,
        quantities=("contacts", "cell_shape"),
        params={"pixel_size_um": 0.5, "time_interval_s": 1.0},
    )
    tables = run(cfg)

    for rec in records:
        pos = Path(rec["position_path"])
        assert (pos / "aggregate_quantification" / "contact_analysis.h5").exists()
        assert not (pos / "aggregate_quantification" / "cell_shape.csv").exists()
    # cheap quantity lives only in the pooled table
    assert "cell_shape" in tables
    import pandas as pd
    assert "cell_shape.area_um2" in pd.read_csv(tables["cell_shape"]).columns
```

(If `author_config` / a two-position labelled fixture does not yet exist in the test file, build the catalog inline the way the file's existing `run()` test does; do not invent a helper that is not there.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/contact_analysis/test_pipeline.py::test_run_persists_only_contacts_per_position -v`
Expected: FAIL — `cell_shape.csv` is currently written per position by `build_quantities`.

- [ ] **Step 3: Implement**

In `pipeline.py` `build_quantities`, restrict the build loop to producers (quantifiers whose artifact another consumes) and drop the now-unused dependency-planning for pooled leaves. Replace the job-collection loop body so only `bool(quantifier.produces)` quantifiers are built:

```python
    quants = [q for q in quants if q.produces]  # only producers persist (contacts)
    records = list(catalog)
    jobs: list[tuple[Quantifier, dict, dict | None]] = []
    for quantifier in quants:
        q_params = dict(params) if (params and quantifier.wants_build_params) else None
        for record in records:
            inputs = position_inputs_from_record(record)
            if not set(quantifier.requires) <= available_fields(inputs):
                continue
            if quantifier.missing_build_params(record_build_params(quantifier, record, params)):
                continue
            jobs.append((quantifier, record, q_params))

    total = len(jobs)
    for index, (quantifier, record, q_params) in enumerate(jobs, start=1):
        inputs = position_inputs_from_record(record)
        if progress_cb is not None:
            progress_cb(index, total, inputs.position_dir.name)
        quantifier.build(inputs, output_for_record(quantifier, record), params=q_params)
```

Because only producers build and today's only producer (`contacts`) has no producer dependency, `_dependency_order` is no longer needed to sequence *pooled* builds; keep the call `quants = _dependency_order(quants)` before the `q.produces` filter (harmless, still correct if a future producer depends on another). Delete `_produced_field_for` and the `available` / produced-field growth logic if they become unused (grep first; `_dependency_order` may still reference none of them).

In `run()`, pass params to aggregate (line 325):

```python
    return aggregate(catalog, cfg.out_dir, params=cfg.params or None)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/contact_analysis/test_pipeline.py -v`
Expected: PASS (new test + existing run tests; pooled quantities now come from `compute_object_table` inside `aggregate`, which recomputes from the same labels/h5).

- [ ] **Step 5: Full suite + commit**

Run: `pytest tests/contact_analysis/ -q`
Expected: PASS. Then:

```bash
git add src/cellflow/contact_analysis/pipeline.py tests/contact_analysis/test_pipeline.py
git commit -m "feat(pipeline): build only producers per position, pool cheap quantities in aggregate"
```

---

## Task 10: Verification (no new code)

- [ ] **Step 1: Run the contact_analysis + napari suites**

Run: `pytest tests/contact_analysis/ tests/napari/ -q`
Expected: PASS. (The napari studio still calls the retained `build` / `object_table` writers, so `BuildArea` and `studio_plugins` tests are unaffected.)

- [ ] **Step 2: Lint**

Run: `ruff check src/cellflow/contact_analysis/`
Expected: clean.

- [ ] **Step 3: Confirm the scope boundary held**

Grep to confirm no pooled quantifier lost its `build` / `read` / `object_table` (the studio needs them):

Run: `grep -L "def object_table" src/cellflow/contact_analysis/quantifiers/{cell_shape,nucleus_shape,shape_relational,cell_dynamics,nucleus_dynamics,cell_density,neighbor_count,signed_contact_length}.py`
Expected: prints nothing (every file still defines `object_table`).

No commit (verification only).

---

## Notes for the executor

- **Do not delete** the per-position writers (`build_object_shape`, `write_table_csv`, `write_provenance`, `_tidy_table.persist` / `read_derived_table`) or the quantifiers' `build` / `read` / `object_table`. The live napari studio (`studio_plugins.BuildArea`) still uses them; their removal is a separate, deferred task (see TODO.md, "Deferred from the lean-aggregate-stage change").
- The `date` column and the `4_contact_analysis` rename are **Plan 2**, not here. In this plan the built h5 still lands under `aggregate_quantification/`; the Task 9 test asserts that path deliberately.
- Golden-oracle tests (compute == build→object_table) mean you never hand-compute expected shape/dynamics values; if a compute test fails, the extract diverged from `build` — reconcile the two.
- **Relocated gate ripple:** because the `required_build_params` gate now runs in `_position_frame`, any *other* test that calls `aggregate` / `build_table` directly for a param-gated quantity (`cell_shape`, dynamics, `cell_density`) must stamp the param on its records (`pixel_size_um`, `time_interval_s`, `fov_area_mm2`) or pass it via the new `params=` argument, or that quantity now pools empty. `run()`-driven tests already stamp params on records, so they are unaffected. If a previously-green `test_reduce` / `test_author_config` / `test_plotting` case goes empty, this is why — add the param, do not weaken the gate.
