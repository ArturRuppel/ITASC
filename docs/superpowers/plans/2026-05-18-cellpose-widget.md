# Cellpose Widget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder `_CellposePanel` with a working in-app Cellpose-SAM runner that has one row per channel (Nucleus, Cell), with ⚙ params / ▷ preview / ▶ run-cancel buttons, sharing a status bar and progress bar.

**Architecture:** A Qt-free runner module (`segmentation/cellpose_runner.py`) vendors the proven logic from `cellpose_full.py` and exposes per-frame and per-stack helpers plus a canonical output writer. A Qt widget (`napari/cellpose_widget.py`) wraps the runner with the established per-stage row pattern (matching `CellWorkflowWidget`), embeds the existing `PipelineFilesWidget` and `CellposeZavgVizWidget`, and runs work in `thread_worker`s with progress and cancel callbacks. `main_widget.py` drops `_CellposePanel`, constructs the new widget, and persists its state under a new top-level `"cellpose"` key.

**Tech Stack:** Python 3.9+, Cellpose-SAM v4 (`cellpose>=4.0` with `pretrained_model="cpsam"`), PyTorch (transitive), napari `thread_worker`, qtpy, NumPy, tifffile.

**Spec:** `docs/superpowers/specs/2026-05-18-cellpose-widget-design.md`

---

## File Structure

- `src/cellflow/segmentation/cellpose_runner.py` *(new)* — Qt-free Cellpose-SAM runner. Owns the lazy model cache, the `NucleusParams`/`CellParams` dataclasses, the per-frame helpers (`run_nucleus_frame`, `run_cell_frame`), the per-stack drivers (`run_nucleus_stack`, `run_cell_stack`) with `progress_cb`/`cancel_cb`, the gamma pre-correction helper, and `write_outputs` (canonical names + z-avg). No imports of qtpy/napari.
- `tests/segmentation/test_cellpose_runner.py` *(new)* — Unit tests with `cellpose.models.CellposeModel` mocked so no torch/GPU is required.
- `src/cellflow/napari/cellpose_widget.py` *(new)* — `CellposeWidget(QWidget)`. Mirrors `CellWorkflowWidget` structure: `PipelineFilesWidget` panel + two stage rows (Nucleus, Cell) + shared status + progress + embedded `CellposeZavgVizWidget`. Public API: `refresh(pos_dir)`, `get_state()`, `set_state(state)`, attribute `output_files_tracker`.
- `tests/napari/test_cellpose_widget.py` *(new)* — Qt smoke tests using a fake viewer (mirroring `tests/napari/test_cell_workflow_widget.py`) and a stubbed `cellpose_runner`.
- `src/cellflow/napari/main_widget.py` *(modify)* — Remove `_CellposePanel` (lines 36–99). Replace construction at line 142 with `CellposeWidget`. Extend `get_state()`/`set_state()` with a `"cellpose"` key.
- `pyproject.toml` *(modify)* — Add `"cellpose>=4.0"` to `dependencies`.

---

## Task 1: Add cellpose dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

Edit `pyproject.toml`. Find the `dependencies = [ ... ]` block. After `"sqlalchemy",` add a new line so the list ends with:

```toml
dependencies = [
    "napari[all]>=0.4.18",
    "qtpy>=2.3.0",
    "h5py",
    "numpy",
    "scipy",
    "scikit-image",
    "pandas",
    "tifffile",
    "numba",
    "matplotlib>=3.9.4",
    "pymaxflow>=1.3.2",
    "pydantic",
    "sqlalchemy",
    "cellpose>=4.0",
]
```

- [ ] **Step 2: Install the dependency in the dev env**

Run: `pip install -e .`
Expected: pip resolves cellpose>=4.0 and pulls torch transitively. Look for `Successfully installed cellpose-...` near the end. No errors.

- [ ] **Step 3: Smoke-import**

Run: `python -c "from cellpose.models import CellposeModel; print(CellposeModel)"`
Expected: prints `<class 'cellpose.models.CellposeModel'>`. No import errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add cellpose>=4.0 for in-app Cellpose-SAM runner"
```

---

## Task 2: Runner module — dataclasses and apply_gamma helper

**Files:**
- Create: `src/cellflow/segmentation/cellpose_runner.py`
- Create: `tests/segmentation/test_cellpose_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/segmentation/test_cellpose_runner.py`:

```python
"""Tests for cellflow.segmentation.cellpose_runner.

The cellpose package is mocked at import time so these tests run without
torch/GPU.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _mock_cellpose(monkeypatch):
    """Install a fake cellpose package so the runner imports cleanly."""
    fake_cellpose = types.ModuleType("cellpose")
    fake_models = types.ModuleType("cellpose.models")

    class _FakeModel:
        def __init__(self, *_, **__):
            pass

        def eval(self, img, **_kwargs):
            img = np.asarray(img, dtype=np.float32)
            if img.ndim == 2:
                dp = np.zeros((2, *img.shape), dtype=np.float32)
                prob = np.zeros(img.shape, dtype=np.float32)
            else:
                dp = np.zeros((3, *img.shape), dtype=np.float32)
                prob = np.zeros(img.shape, dtype=np.float32)
            return None, (None, dp, prob), None

    fake_models.CellposeModel = _FakeModel
    fake_cellpose.models = fake_models
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)

    # Ensure the runner module is freshly imported under the mock.
    monkeypatch.delitem(sys.modules, "cellflow.segmentation.cellpose_runner", raising=False)
    yield


def _runner():
    import importlib

    import cellflow.segmentation.cellpose_runner as runner
    importlib.reload(runner)
    return runner


def test_dataclasses_have_expected_fields():
    r = _runner()
    n = r.NucleusParams(do_3d=True, anisotropy=1.5, diameter=25.0, min_size=15, gamma=1.0)
    c = r.CellParams(diameter=0.0, min_size=0, gamma=1.0)
    assert n.do_3d is True
    assert n.anisotropy == 1.5
    assert n.diameter == 25.0
    assert n.min_size == 15
    assert n.gamma == 1.0
    assert c.diameter == 0.0
    assert c.min_size == 0
    assert c.gamma == 1.0


def test_dataclasses_are_frozen():
    r = _runner()
    n = r.NucleusParams(do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0)
    with pytest.raises(Exception):
        n.do_3d = True


def test_apply_gamma_identity_when_one():
    r = _runner()
    img = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = r._apply_gamma(img, 1.0)
    np.testing.assert_array_equal(out, img)


def test_apply_gamma_handles_constant_image():
    r = _runner()
    img = np.full((4, 4), 7.0, dtype=np.float32)
    out = r._apply_gamma(img, 0.5)
    np.testing.assert_array_equal(out, img)


def test_apply_gamma_warps_dynamic_range():
    r = _runner()
    img = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    out = r._apply_gamma(img, 2.0)
    expected = img ** 2.0
    np.testing.assert_allclose(out, expected, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/segmentation/test_cellpose_runner.py -v`
Expected: All tests FAIL with `ModuleNotFoundError: No module named 'cellflow.segmentation.cellpose_runner'`.

- [ ] **Step 3: Create the runner module with dataclasses + gamma helper**

Create `src/cellflow/segmentation/cellpose_runner.py`:

```python
"""Local Cellpose-SAM runner — Qt-free, used by the napari Cellpose widget.

Vendored and adapted from /home/aruppel/Projects/HPC/cellpose_full/cellpose_full.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import tifffile

_NORMALIZE = {"tile_norm_blocksize": 128}


@dataclass(frozen=True)
class NucleusParams:
    do_3d: bool
    anisotropy: float
    diameter: float  # 0 means "let cpsam decide" (None passed to model)
    min_size: int
    gamma: float


@dataclass(frozen=True)
class CellParams:
    diameter: float
    min_size: int
    gamma: float


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Min/max-normalized gamma correction matching cellpose_full.py."""
    if gamma == 1.0:
        return np.asarray(img)
    img = np.asarray(img, dtype=np.float32)
    img_min = float(np.min(img))
    img_max = float(np.max(img))
    if img_max <= img_min:
        return img
    scaled = (img - img_min) / (img_max - img_min)
    return (scaled ** gamma) * (img_max - img_min) + img_min


def _diameter_kwarg(diameter: float) -> float | None:
    return None if diameter == 0 else float(diameter)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/segmentation/test_cellpose_runner.py -v`
Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/segmentation/cellpose_runner.py tests/segmentation/test_cellpose_runner.py
git commit -m "feat(cellpose_runner): add params dataclasses and gamma helper"
```

---

## Task 3: Runner module — lazy model cache

**Files:**
- Modify: `src/cellflow/segmentation/cellpose_runner.py`
- Modify: `tests/segmentation/test_cellpose_runner.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/segmentation/test_cellpose_runner.py`:

```python
def test_is_model_loaded_false_initially():
    r = _runner()
    assert r.is_model_loaded() is False


def test_get_model_caches_across_calls(monkeypatch):
    r = _runner()
    calls = {"n": 0}
    import cellpose.models as models

    real = models.CellposeModel

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(models, "CellposeModel", _counting)
    a = r.get_model()
    b = r.get_model()
    assert a is b
    assert calls["n"] == 1
    assert r.is_model_loaded() is True


def test_get_model_uses_cpu_when_cuda_unavailable(monkeypatch):
    r = _runner()
    received_kwargs = {}
    import cellpose.models as models

    class _Probe:
        def __init__(self, **kwargs):
            received_kwargs.update(kwargs)

    monkeypatch.setattr(models, "CellposeModel", _Probe)
    monkeypatch.setattr(r, "_cuda_available", lambda: False)
    r.get_model()
    assert received_kwargs["gpu"] is False
    assert received_kwargs["pretrained_model"] == "cpsam"
    assert received_kwargs["use_bfloat16"] is False


def test_get_model_uses_gpu_when_cuda_available(monkeypatch):
    r = _runner()
    received_kwargs = {}
    import cellpose.models as models

    class _Probe:
        def __init__(self, **kwargs):
            received_kwargs.update(kwargs)

    monkeypatch.setattr(models, "CellposeModel", _Probe)
    monkeypatch.setattr(r, "_cuda_available", lambda: True)
    r.get_model()
    assert received_kwargs["gpu"] is True
    assert received_kwargs["use_bfloat16"] is True


def test_device_label_reflects_cuda(monkeypatch):
    r = _runner()
    monkeypatch.setattr(r, "_cuda_available", lambda: True)
    assert r.device_label() == "cuda:0"
    monkeypatch.setattr(r, "_cuda_available", lambda: False)
    assert r.device_label() == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/segmentation/test_cellpose_runner.py -v`
Expected: New tests FAIL with `AttributeError: module 'cellflow.segmentation.cellpose_runner' has no attribute 'is_model_loaded'` (and similar for `get_model`, `_cuda_available`, `device_label`).

- [ ] **Step 3: Add the model cache + device helpers to the runner**

Append to `src/cellflow/segmentation/cellpose_runner.py`:

```python
_MODEL = None


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def device_label() -> str:
    return "cuda:0" if _cuda_available() else "cpu"


def is_model_loaded() -> bool:
    return _MODEL is not None


def get_model():
    """Lazy-load the cpsam model once per process; cached at module level."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from cellpose.models import CellposeModel

    use_gpu = _cuda_available()
    _MODEL = CellposeModel(
        gpu=use_gpu,
        pretrained_model="cpsam",
        use_bfloat16=use_gpu,
    )
    return _MODEL
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/segmentation/test_cellpose_runner.py -v`
Expected: 10 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/segmentation/cellpose_runner.py tests/segmentation/test_cellpose_runner.py
git commit -m "feat(cellpose_runner): add lazy cpsam model cache with cuda auto-detect"
```

---

## Task 4: Runner module — per-frame helpers

**Files:**
- Modify: `src/cellflow/segmentation/cellpose_runner.py`
- Modify: `tests/segmentation/test_cellpose_runner.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/segmentation/test_cellpose_runner.py`:

```python
def _install_recording_model(monkeypatch, r):
    """Replace the runner's model with one that records eval kwargs."""
    calls = []

    class _Recorder:
        def eval(self, img, **kwargs):
            calls.append({"shape": np.asarray(img).shape, **kwargs})
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 2:
                dp = np.ones((2, *arr.shape), dtype=np.float32)
                prob = np.full(arr.shape, 0.5, dtype=np.float32)
            else:
                dp = np.ones((3, *arr.shape), dtype=np.float32)
                prob = np.full(arr.shape, 0.5, dtype=np.float32)
            return None, (None, dp, prob), None

    monkeypatch.setattr(r, "_MODEL", _Recorder())
    return calls


def test_run_nucleus_frame_3d_passes_do_3d_and_anisotropy(monkeypatch):
    r = _runner()
    calls = _install_recording_model(monkeypatch, r)
    frame = np.zeros((4, 8, 8), dtype=np.float32)
    params = r.NucleusParams(do_3d=True, anisotropy=1.5, diameter=25.0, min_size=15, gamma=1.0)
    prob, dp = r.run_nucleus_frame(frame, z=None, params=params)
    assert prob.shape == (4, 8, 8)
    assert dp.shape == (3, 4, 8, 8)
    assert len(calls) == 1
    assert calls[0]["do_3D"] is True
    assert calls[0]["z_axis"] == 0
    assert calls[0]["anisotropy"] == 1.5
    assert calls[0]["diameter"] == 25.0
    assert calls[0]["min_size"] == 15


def test_run_nucleus_frame_2d_slices_z(monkeypatch):
    r = _runner()
    calls = _install_recording_model(monkeypatch, r)
    frame = np.zeros((4, 8, 8), dtype=np.float32)
    params = r.NucleusParams(do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0)
    prob, dp = r.run_nucleus_frame(frame, z=2, params=params)
    assert prob.shape == (8, 8)
    assert dp.shape == (2, 8, 8)
    assert len(calls) == 1
    assert calls[0]["shape"] == (8, 8)
    assert calls[0].get("do_3D", False) is False
    assert calls[0]["diameter"] is None


def test_run_cell_frame_runs_2d_slice(monkeypatch):
    r = _runner()
    calls = _install_recording_model(monkeypatch, r)
    frame = np.zeros((4, 8, 8), dtype=np.float32)
    params = r.CellParams(diameter=30.0, min_size=10, gamma=1.0)
    prob, dp = r.run_cell_frame(frame, z=1, params=params)
    assert prob.shape == (8, 8)
    assert dp.shape == (2, 8, 8)
    assert len(calls) == 1
    assert calls[0]["shape"] == (8, 8)
    assert calls[0]["diameter"] == 30.0
    assert calls[0]["min_size"] == 10


def test_run_nucleus_frame_applies_gamma(monkeypatch):
    r = _runner()
    received = {}

    class _Sniffer:
        def eval(self, img, **kwargs):
            received["img"] = np.asarray(img).copy()
            arr = np.asarray(img, dtype=np.float32)
            return None, (None, np.zeros((3, *arr.shape), dtype=np.float32), np.zeros(arr.shape, dtype=np.float32)), None

    monkeypatch.setattr(r, "_MODEL", _Sniffer())
    frame = np.linspace(0.0, 1.0, num=2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    params = r.NucleusParams(do_3d=True, anisotropy=1.0, diameter=0.0, min_size=0, gamma=2.0)
    r.run_nucleus_frame(frame, z=None, params=params)
    expected = r._apply_gamma(frame, 2.0)
    np.testing.assert_allclose(received["img"], expected, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/segmentation/test_cellpose_runner.py -v`
Expected: New tests FAIL with `AttributeError: module ... has no attribute 'run_nucleus_frame'`.

- [ ] **Step 3: Add per-frame helpers to the runner**

Append to `src/cellflow/segmentation/cellpose_runner.py`:

```python
def run_nucleus_frame(
    frame: np.ndarray,
    z: int | None,
    params: NucleusParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-frame nucleus inference.

    If ``z`` is None, runs full 3D over (Z, Y, X) and returns
    prob with shape (Z, Y, X) and dp with shape (3, Z, Y, X).
    If ``z`` is an integer, runs 2D on frame[z] and returns
    prob with shape (Y, X) and dp with shape (2, Y, X).
    """
    model = get_model()
    diameter = _diameter_kwarg(params.diameter)
    if z is None:
        volume = _apply_gamma(frame, params.gamma)
        _, flows, _ = model.eval(
            volume,
            do_3D=True,
            z_axis=0,
            diameter=diameter,
            anisotropy=params.anisotropy,
            min_size=params.min_size,
            normalize=_NORMALIZE,
        )
    else:
        slice_2d = _apply_gamma(frame[z], params.gamma)
        _, flows, _ = model.eval(
            slice_2d,
            diameter=diameter,
            min_size=params.min_size,
            normalize=_NORMALIZE,
        )
    dp = np.asarray(flows[1], dtype=np.float32)
    prob = np.asarray(flows[2], dtype=np.float32)
    return prob, dp


def run_cell_frame(
    frame: np.ndarray,
    z: int,
    params: CellParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Single 2D-slice cell inference. Returns (prob (Y,X), dp (2,Y,X))."""
    model = get_model()
    diameter = _diameter_kwarg(params.diameter)
    slice_2d = _apply_gamma(frame[z], params.gamma)
    _, flows, _ = model.eval(
        slice_2d,
        diameter=diameter,
        min_size=params.min_size,
        normalize=_NORMALIZE,
    )
    dp = np.asarray(flows[1], dtype=np.float32)
    prob = np.asarray(flows[2], dtype=np.float32)
    return prob, dp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/segmentation/test_cellpose_runner.py -v`
Expected: 14 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/segmentation/cellpose_runner.py tests/segmentation/test_cellpose_runner.py
git commit -m "feat(cellpose_runner): add per-frame nucleus and cell helpers"
```

---

## Task 5: Runner module — per-stack drivers with progress + cancel

**Files:**
- Modify: `src/cellflow/segmentation/cellpose_runner.py`
- Modify: `tests/segmentation/test_cellpose_runner.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/segmentation/test_cellpose_runner.py`:

```python
def test_run_nucleus_stack_3d_iterates_frames_and_reports_progress(monkeypatch):
    r = _runner()
    _install_recording_model(monkeypatch, r)
    stack = np.zeros((3, 4, 6, 6), dtype=np.float32)
    params = r.NucleusParams(do_3d=True, anisotropy=1.5, diameter=0.0, min_size=0, gamma=1.0)
    progress = []
    prob, dp = r.run_nucleus_stack(
        stack, params,
        progress_cb=lambda d, t, msg: progress.append((d, t, msg)),
    )
    assert prob.shape == (3, 4, 6, 6)
    assert dp.shape == (3, 3, 4, 6, 6)
    assert progress[0] == (0, 3, "Nucleus: frame 1/3...")
    assert progress[-1] == (3, 3, "Nucleus: frame 3/3...")


def test_run_nucleus_stack_2d_returns_per_slice_outputs(monkeypatch):
    r = _runner()
    _install_recording_model(monkeypatch, r)
    stack = np.zeros((2, 3, 6, 6), dtype=np.float32)
    params = r.NucleusParams(do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0)
    prob, dp = r.run_nucleus_stack(stack, params)
    assert prob.shape == (2, 3, 6, 6)
    assert dp.shape == (2, 3, 2, 6, 6)


def test_run_cell_stack_iterates_t_then_z(monkeypatch):
    r = _runner()
    calls = _install_recording_model(monkeypatch, r)
    stack = np.zeros((2, 3, 6, 6), dtype=np.float32)
    params = r.CellParams(diameter=0.0, min_size=0, gamma=1.0)
    progress = []
    prob, dp = r.run_cell_stack(
        stack, params,
        progress_cb=lambda d, t, msg: progress.append((d, t, msg)),
    )
    assert prob.shape == (2, 3, 6, 6)
    assert dp.shape == (2, 3, 2, 6, 6)
    assert len(calls) == 2 * 3
    assert progress[0] == (0, 2, "Cell: frame 1/2...")


def test_run_nucleus_stack_respects_cancel_between_frames(monkeypatch):
    r = _runner()
    _install_recording_model(monkeypatch, r)
    stack = np.zeros((5, 2, 4, 4), dtype=np.float32)
    params = r.NucleusParams(do_3d=True, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0)
    seen_frames = []

    def _cancel():
        return len(seen_frames) >= 2

    def _record(done, total, msg):
        seen_frames.append(done)

    with pytest.raises(r.CancelledError):
        r.run_nucleus_stack(stack, params, progress_cb=_record, cancel_cb=_cancel)


def test_run_cell_stack_respects_cancel_between_frames(monkeypatch):
    r = _runner()
    _install_recording_model(monkeypatch, r)
    stack = np.zeros((5, 2, 4, 4), dtype=np.float32)
    params = r.CellParams(diameter=0.0, min_size=0, gamma=1.0)
    seen = []

    def _cancel():
        return len(seen) >= 1

    def _record(done, total, msg):
        seen.append(done)

    with pytest.raises(r.CancelledError):
        r.run_cell_stack(stack, params, progress_cb=_record, cancel_cb=_cancel)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/segmentation/test_cellpose_runner.py -v`
Expected: New tests FAIL with `AttributeError: module ... has no attribute 'run_nucleus_stack'` (and similar).

- [ ] **Step 3: Add the per-stack drivers + CancelledError**

Append to `src/cellflow/segmentation/cellpose_runner.py`:

```python
class CancelledError(RuntimeError):
    """Raised by run_*_stack when cancel_cb returns True between frames."""


def _check_cancel(cancel_cb: Callable[[], bool] | None) -> None:
    if cancel_cb is not None and cancel_cb():
        raise CancelledError("cellpose run cancelled")


def run_nucleus_stack(
    stack: np.ndarray,
    params: NucleusParams,
    *,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a (T, Z, Y, X) stack frame-by-frame.

    Returns (prob_3dt, dp_3dt). For do_3d=True dp has shape (T, 3, Z, Y, X);
    for do_3d=False dp has shape (T, Z, 2, Y, X).
    """
    if stack.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {stack.shape}")
    T = stack.shape[0]
    prob_frames: list[np.ndarray] = []
    dp_frames: list[np.ndarray] = []
    for t in range(T):
        _check_cancel(cancel_cb)
        if progress_cb is not None:
            progress_cb(t, T, f"Nucleus: frame {t + 1}/{T}...")
        if params.do_3d:
            prob, dp = run_nucleus_frame(stack[t], z=None, params=params)
        else:
            slice_probs: list[np.ndarray] = []
            slice_dps: list[np.ndarray] = []
            for z in range(stack.shape[1]):
                p, d = run_nucleus_frame(stack[t], z=z, params=params)
                slice_probs.append(p)
                slice_dps.append(d)
            prob = np.stack(slice_probs, axis=0)
            dp = np.stack(slice_dps, axis=0)
        prob_frames.append(prob)
        dp_frames.append(dp)
        if progress_cb is not None:
            progress_cb(t + 1, T, f"Nucleus: frame {t + 1}/{T}...")
    return np.stack(prob_frames, axis=0), np.stack(dp_frames, axis=0)


def run_cell_stack(
    stack: np.ndarray,
    params: CellParams,
    *,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a (T, Z, Y, X) stack slice-by-slice in 2D.

    Returns (prob_3dt (T, Z, Y, X), dp_3dt (T, Z, 2, Y, X)).
    """
    if stack.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {stack.shape}")
    T, Z = stack.shape[:2]
    prob_frames: list[np.ndarray] = []
    dp_frames: list[np.ndarray] = []
    for t in range(T):
        _check_cancel(cancel_cb)
        if progress_cb is not None:
            progress_cb(t, T, f"Cell: frame {t + 1}/{T}...")
        slice_probs: list[np.ndarray] = []
        slice_dps: list[np.ndarray] = []
        for z in range(Z):
            p, d = run_cell_frame(stack[t], z=z, params=params)
            slice_probs.append(p)
            slice_dps.append(d)
        prob_frames.append(np.stack(slice_probs, axis=0))
        dp_frames.append(np.stack(slice_dps, axis=0))
        if progress_cb is not None:
            progress_cb(t + 1, T, f"Cell: frame {t + 1}/{T}...")
    return np.stack(prob_frames, axis=0), np.stack(dp_frames, axis=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/segmentation/test_cellpose_runner.py -v`
Expected: 19 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/segmentation/cellpose_runner.py tests/segmentation/test_cellpose_runner.py
git commit -m "feat(cellpose_runner): add stack drivers with progress and cancel"
```

---

## Task 6: Runner module — write_outputs (canonical names + z-avg)

**Files:**
- Modify: `src/cellflow/segmentation/cellpose_runner.py`
- Modify: `tests/segmentation/test_cellpose_runner.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/segmentation/test_cellpose_runner.py`:

```python
def test_write_outputs_nucleus(tmp_path: Path):
    r = _runner()
    prob = np.random.rand(2, 3, 4, 5).astype(np.float32)
    dp = np.random.rand(2, 3, 3, 4, 5).astype(np.float32)
    r.write_outputs(prob, dp, tmp_path, "nucleus")
    prob_3dt = tmp_path / "nucleus_prob_3dt.tif"
    dp_3dt = tmp_path / "nucleus_dp_3dt.tif"
    zavg = tmp_path / "nucleus_prob_zavg.tif"
    assert prob_3dt.exists() and dp_3dt.exists() and zavg.exists()
    written_prob = tifffile.imread(str(prob_3dt))
    written_zavg = tifffile.imread(str(zavg))
    np.testing.assert_allclose(written_prob, prob)
    np.testing.assert_allclose(written_zavg, prob.mean(axis=1), rtol=1e-6)
    assert written_zavg.shape == (2, 4, 5)
    assert written_zavg.dtype == np.float32


def test_write_outputs_cell(tmp_path: Path):
    r = _runner()
    prob = np.random.rand(2, 3, 4, 5).astype(np.float32)
    dp = np.random.rand(2, 3, 2, 4, 5).astype(np.float32)
    r.write_outputs(prob, dp, tmp_path, "cell")
    assert (tmp_path / "cell_prob_3dt.tif").exists()
    assert (tmp_path / "cell_dp_3dt.tif").exists()
    assert (tmp_path / "cell_prob_zavg.tif").exists()


def test_write_outputs_creates_missing_dir(tmp_path: Path):
    r = _runner()
    target = tmp_path / "1_cellpose"
    prob = np.zeros((1, 2, 3, 3), dtype=np.float32)
    dp = np.zeros((1, 2, 3, 2, 3), dtype=np.float32)
    r.write_outputs(prob, dp, target, "cell")
    assert (target / "cell_prob_3dt.tif").exists()


def test_write_outputs_rejects_bad_channel(tmp_path: Path):
    r = _runner()
    prob = np.zeros((1, 1, 2, 2), dtype=np.float32)
    dp = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        r.write_outputs(prob, dp, tmp_path, "blah")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/segmentation/test_cellpose_runner.py -v`
Expected: New tests FAIL with `AttributeError: module ... has no attribute 'write_outputs'`.

- [ ] **Step 3: Implement write_outputs**

Append to `src/cellflow/segmentation/cellpose_runner.py`:

```python
def write_outputs(
    prob_3dt: np.ndarray,
    dp_3dt: np.ndarray,
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
) -> None:
    """Write the three canonical TIFFs under output_dir.

    Writes ``{channel}_prob_3dt.tif``, ``{channel}_dp_3dt.tif``, and the
    z-averaged probability ``{channel}_prob_zavg.tif``.
    """
    if channel not in ("nucleus", "cell"):
        raise ValueError(f"channel must be 'nucleus' or 'cell', got {channel!r}")
    if prob_3dt.ndim != 4:
        raise ValueError(f"prob_3dt must be (T, Z, Y, X), got {prob_3dt.shape}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prob_path = output_dir / f"{channel}_prob_3dt.tif"
    dp_path = output_dir / f"{channel}_dp_3dt.tif"
    zavg_path = output_dir / f"{channel}_prob_zavg.tif"
    tifffile.imwrite(str(prob_path), prob_3dt.astype(np.float32), compression="zlib")
    tifffile.imwrite(str(dp_path), dp_3dt.astype(np.float32), compression="zlib")
    zavg = prob_3dt.mean(axis=1, dtype=np.float32).astype(np.float32)
    tifffile.imwrite(str(zavg_path), zavg, compression="zlib")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/segmentation/test_cellpose_runner.py -v`
Expected: 23 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/segmentation/cellpose_runner.py tests/segmentation/test_cellpose_runner.py
git commit -m "feat(cellpose_runner): add write_outputs for canonical 1_cellpose tifs"
```

---

## Task 7: Widget skeleton — params, stage rows, mutual exclusion, state I/O

**Files:**
- Create: `src/cellflow/napari/cellpose_widget.py`
- Create: `tests/napari/test_cellpose_widget.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/napari/test_cellpose_widget.py`:

```python
"""Tests for the local Cellpose widget."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QToolButton


class _LayerCollection(dict):
    def remove(self, layer):
        self.pop(layer.name, None)


class _FakeEvent:
    def connect(self, cb):
        pass

    def disconnect(self, cb):
        pass


class _FakeEvents:
    def __init__(self) -> None:
        self.data = _FakeEvent()
        self.paint = _FakeEvent()
        self.mode = _FakeEvent()
        self.removed = _FakeEvent()


class _FakeSelection:
    def __init__(self) -> None:
        self.active = None


class _FakeLayer:
    def __init__(self, data, name, **kwargs) -> None:
        self.data = np.asarray(data)
        self.name = name
        self.events = _FakeEvents()
        self.kwargs = kwargs


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()
        self.layers.selection = _FakeSelection()
        self.layers.events = _FakeEvents()
        self.dims = SimpleNamespace(
            current_step=(0, 0),
            events=SimpleNamespace(
                current_step=SimpleNamespace(connect=lambda cb: None)
            ),
        )

    def add_image(self, data, *, name, **kwargs):
        layer = _FakeLayer(data, name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_labels(self, data, *, name, **kwargs):
        layer = _FakeLayer(data, name, **kwargs)
        self.layers[name] = layer
        return layer


@pytest.fixture
def _mock_cellpose(monkeypatch):
    """Install a fake cellpose so the runner imports cleanly."""
    fake_cellpose = types.ModuleType("cellpose")
    fake_models = types.ModuleType("cellpose.models")

    class _FakeModel:
        def __init__(self, *_, **__):
            pass

        def eval(self, img, **_kwargs):
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 2:
                dp = np.zeros((2, *arr.shape), dtype=np.float32)
                prob = np.zeros(arr.shape, dtype=np.float32)
            else:
                dp = np.zeros((3, *arr.shape), dtype=np.float32)
                prob = np.zeros(arr.shape, dtype=np.float32)
            return None, (None, dp, prob), None

    fake_models.CellposeModel = _FakeModel
    fake_cellpose.models = fake_models
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)
    monkeypatch.delitem(sys.modules, "cellflow.segmentation.cellpose_runner", raising=False)


def _load_widget(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.cellpose_widget", None)
    return importlib.import_module("cellflow.napari.cellpose_widget")


def test_widget_exposes_stage_rows_and_buttons(_mock_cellpose, monkeypatch):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    for name in (
        "nucleus_params_btn",
        "nucleus_preview_btn",
        "nucleus_run_btn",
        "cell_params_btn",
        "cell_preview_btn",
        "cell_run_btn",
    ):
        btn = getattr(w, name)
        assert isinstance(btn, QToolButton), name
    assert w.nucleus_run_btn.text() == "▶"
    assert w.cell_run_btn.text() == "▶"
    w.deleteLater()


def test_params_button_toggles_section(_mock_cellpose, monkeypatch):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    assert not w.nucleus_section.is_expanded
    w.nucleus_params_btn.setChecked(True)
    assert w.nucleus_section.is_expanded
    w.nucleus_params_btn.setChecked(False)
    assert not w.nucleus_section.is_expanded
    w.deleteLater()


def test_get_set_state_round_trips(_mock_cellpose, monkeypatch):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    new_state = {
        "nucleus": {
            "do_3d": False,
            "anisotropy": 2.25,
            "diameter": 42.0,
            "min_size": 7,
            "gamma": 1.5,
        },
        "cell": {
            "diameter": 18.0,
            "min_size": 3,
            "gamma": 0.8,
        },
    }
    w.set_state(new_state)
    got = w.get_state()
    assert got == new_state
    w.deleteLater()


def test_set_running_stage_disables_other_row(_mock_cellpose, monkeypatch):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    w._set_running_stage("nucleus")
    assert w.nucleus_run_btn.text() == "✕"
    assert w.nucleus_run_btn.isEnabled()
    assert not w.cell_run_btn.isEnabled()
    assert not w.cell_params_btn.isEnabled()
    assert not w.cell_preview_btn.isEnabled()
    w._set_running_stage(None)
    assert w.nucleus_run_btn.text() == "▶"
    assert w.cell_run_btn.isEnabled()
    assert w.cell_params_btn.isEnabled()
    w.deleteLater()


def test_exposes_output_files_tracker(_mock_cellpose, monkeypatch):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    assert hasattr(w, "output_files_tracker")
    # Same attribute name kept for main_widget.pipeline_status_from_files.
    assert w.output_files_tracker is w._files_widget
    w.deleteLater()


def test_refresh_with_none_does_not_raise(_mock_cellpose, monkeypatch):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    w.refresh(None)
    w.deleteLater()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/napari/test_cellpose_widget.py -v`
Expected: All tests FAIL with `ModuleNotFoundError: No module named 'cellflow.napari.cellpose_widget'`.

- [ ] **Step 3: Create the widget skeleton**

Create `src/cellflow/napari/cellpose_widget.py`:

```python
"""Local Cellpose-SAM widget — per-channel rows with preview, run, cancel."""
from __future__ import annotations

import logging
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._widget_helpers import tool_btn as _tool_btn
from cellflow.napari.cellpose_zavg_viz_widget import CellposeZavgVizWidget
from cellflow.napari.ui_style import stage_header_label, status_label
from cellflow.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    make_pipeline_files_header,
)
from cellflow.segmentation import cellpose_runner

logger = logging.getLogger(__name__)


_PIPELINE_FILES = [
    ("Inputs", [
        ("0_input/nucleus_3dt.tif", "Nucleus 3D+t"),
        ("0_input/cell_3dt.tif", "Cell 3D+t"),
    ]),
    ("Outputs", [
        ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
        ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
        ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
        ("1_cellpose/cell_prob_zavg.tif", "Cell prob z-avg"),
        ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
    ]),
]


def _make_status() -> QLabel:
    lbl = QLabel("")
    lbl.setWordWrap(True)
    lbl.setVisible(False)
    status_label(lbl)
    return lbl


def _make_progress() -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setVisible(False)
    return bar


class CellposeWidget(QWidget):
    """Local Cellpose-SAM runner — two rows (Nucleus, Cell)."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._running_stage: str | None = None
        self._worker = None
        self._cancel_requested = False

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # ── Pipeline files ─────────────────────────────────────────────
        self._files_widget = PipelineFilesWidget(_PIPELINE_FILES, viewer=self.viewer)
        self.output_files_tracker = self._files_widget
        self.input_files_tracker = self._files_widget
        self._pipeline_files_section = CollapsibleSection(
            "Pipeline Files", self._files_widget, expanded=False,
        )
        (
            self.pipeline_files_header,
            self.pipeline_files_header_lbl,
            self.pipeline_files_toggle_btn,
        ) = make_pipeline_files_header(
            self._pipeline_files_section, stage_key="cellpose", parent=self,
        )
        root.addWidget(self.pipeline_files_header)
        root.addWidget(self._pipeline_files_section)

        # ── Nucleus row + params ───────────────────────────────────────
        self.nucleus_params_btn = _tool_btn(
            "⚙", "Show parameters for nucleus Cellpose.", checkable=True,
        )
        self.nucleus_preview_btn = _tool_btn("▷", "Preview on current frame.")
        self.nucleus_run_btn = _tool_btn("▶", "Run nucleus Cellpose on all frames.")
        self.nucleus_section = self._build_nucleus_params_section()
        self.nucleus_section.set_header_visible(False)
        self.nucleus_section.collapse()
        self.nucleus_params_btn.toggled.connect(
            lambda checked: self.nucleus_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Nucleus Cellpose"),
            self.nucleus_params_btn,
            self.nucleus_preview_btn,
            self.nucleus_run_btn,
        ))
        root.addWidget(self.nucleus_section)

        # ── Cell row + params ──────────────────────────────────────────
        self.cell_params_btn = _tool_btn(
            "⚙", "Show parameters for cell Cellpose.", checkable=True,
        )
        self.cell_preview_btn = _tool_btn("▷", "Preview on current frame/z-slice.")
        self.cell_run_btn = _tool_btn("▶", "Run cell Cellpose on all frames.")
        self.cell_section = self._build_cell_params_section()
        self.cell_section.set_header_visible(False)
        self.cell_section.collapse()
        self.cell_params_btn.toggled.connect(
            lambda checked: self.cell_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Cell Cellpose"),
            self.cell_params_btn,
            self.cell_preview_btn,
            self.cell_run_btn,
        ))
        root.addWidget(self.cell_section)

        # ── Status + progress (shared) ─────────────────────────────────
        self.status_lbl = _make_status()
        root.addWidget(self.status_lbl)
        self.progress_bar = _make_progress()
        root.addWidget(self.progress_bar)

        # ── Z-avg viz (unchanged) ──────────────────────────────────────
        self.zavg_viz_widget = CellposeZavgVizWidget()
        root.addWidget(self.zavg_viz_widget)

    def _build_nucleus_params_section(self) -> CollapsibleSection:
        body = QWidget(self)
        form = QFormLayout(body)
        form.setContentsMargins(8, 4, 4, 4)
        form.setSpacing(4)
        self.nuc_3d_chk = QCheckBox("3D mode")
        self.nuc_3d_chk.setChecked(True)
        self.nuc_anisotropy_spin = QDoubleSpinBox()
        self.nuc_anisotropy_spin.setRange(0.1, 20.0)
        self.nuc_anisotropy_spin.setSingleStep(0.1)
        self.nuc_anisotropy_spin.setDecimals(2)
        self.nuc_anisotropy_spin.setValue(1.5)
        self.nuc_diameter_spin = QDoubleSpinBox()
        self.nuc_diameter_spin.setRange(0.0, 500.0)
        self.nuc_diameter_spin.setDecimals(1)
        self.nuc_diameter_spin.setValue(25.0)
        self.nuc_min_size_spin = QSpinBox()
        self.nuc_min_size_spin.setRange(0, 100000)
        self.nuc_min_size_spin.setValue(15)
        self.nuc_gamma_spin = QDoubleSpinBox()
        self.nuc_gamma_spin.setRange(0.1, 5.0)
        self.nuc_gamma_spin.setSingleStep(0.1)
        self.nuc_gamma_spin.setDecimals(2)
        self.nuc_gamma_spin.setValue(1.0)
        form.addRow(self.nuc_3d_chk)
        form.addRow("Anisotropy", self.nuc_anisotropy_spin)
        form.addRow("Diameter", self.nuc_diameter_spin)
        form.addRow("Min size", self.nuc_min_size_spin)
        form.addRow("Gamma", self.nuc_gamma_spin)
        return CollapsibleSection("Nucleus parameters", body, expanded=False)

    def _build_cell_params_section(self) -> CollapsibleSection:
        body = QWidget(self)
        form = QFormLayout(body)
        form.setContentsMargins(8, 4, 4, 4)
        form.setSpacing(4)
        self.cell_diameter_spin = QDoubleSpinBox()
        self.cell_diameter_spin.setRange(0.0, 500.0)
        self.cell_diameter_spin.setDecimals(1)
        self.cell_diameter_spin.setValue(0.0)
        self.cell_min_size_spin = QSpinBox()
        self.cell_min_size_spin.setRange(0, 100000)
        self.cell_min_size_spin.setValue(0)
        self.cell_gamma_spin = QDoubleSpinBox()
        self.cell_gamma_spin.setRange(0.1, 5.0)
        self.cell_gamma_spin.setSingleStep(0.1)
        self.cell_gamma_spin.setDecimals(2)
        self.cell_gamma_spin.setValue(1.0)
        form.addRow("Diameter", self.cell_diameter_spin)
        form.addRow("Min size", self.cell_min_size_spin)
        form.addRow("Gamma", self.cell_gamma_spin)
        return CollapsibleSection("Cell parameters", body, expanded=False)

    @staticmethod
    def _stage_label(text: str) -> QLabel:
        return stage_header_label(QLabel(text), "cellpose")

    @staticmethod
    def _stage_row(label: QLabel, *trailing: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(label)
        row.addStretch(1)
        for w in trailing:
            row.addWidget(w)
        return row

    # ------------------------------------------------------------------
    # Signals (run/cancel handlers are filled in in later tasks)
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.nucleus_run_btn.clicked.connect(self._on_nucleus_run_clicked)
        self.cell_run_btn.clicked.connect(self._on_cell_run_clicked)
        self.nucleus_preview_btn.clicked.connect(self._on_nucleus_preview)
        self.cell_preview_btn.clicked.connect(self._on_cell_preview)

    def _on_nucleus_run_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()

    def _on_cell_run_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()

    def _on_nucleus_preview(self) -> None:
        pass

    def _on_cell_preview(self) -> None:
        pass

    def _on_cancel(self) -> None:
        self._cancel_requested = True
        worker = self._worker
        if worker is not None and hasattr(worker, "quit"):
            worker.quit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._files_widget.refresh(pos_dir)
        self.zavg_viz_widget.refresh(pos_dir)

    def get_state(self) -> dict:
        return {
            "nucleus": {
                "do_3d": self.nuc_3d_chk.isChecked(),
                "anisotropy": self.nuc_anisotropy_spin.value(),
                "diameter": self.nuc_diameter_spin.value(),
                "min_size": self.nuc_min_size_spin.value(),
                "gamma": self.nuc_gamma_spin.value(),
            },
            "cell": {
                "diameter": self.cell_diameter_spin.value(),
                "min_size": self.cell_min_size_spin.value(),
                "gamma": self.cell_gamma_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        nuc = state.get("nucleus", {})
        if isinstance(nuc, dict):
            if "do_3d" in nuc:
                self.nuc_3d_chk.setChecked(bool(nuc["do_3d"]))
            if "anisotropy" in nuc:
                self.nuc_anisotropy_spin.setValue(float(nuc["anisotropy"]))
            if "diameter" in nuc:
                self.nuc_diameter_spin.setValue(float(nuc["diameter"]))
            if "min_size" in nuc:
                self.nuc_min_size_spin.setValue(int(nuc["min_size"]))
            if "gamma" in nuc:
                self.nuc_gamma_spin.setValue(float(nuc["gamma"]))
        cel = state.get("cell", {})
        if isinstance(cel, dict):
            if "diameter" in cel:
                self.cell_diameter_spin.setValue(float(cel["diameter"]))
            if "min_size" in cel:
                self.cell_min_size_spin.setValue(int(cel["min_size"]))
            if "gamma" in cel:
                self.cell_gamma_spin.setValue(float(cel["gamma"]))

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _status(self, msg: str) -> None:
        self.status_lbl.setText(msg)
        self.status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _progress(self, done: int, total: int, msg: str) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self._status(msg)

    def _on_progress(self, data) -> None:
        if isinstance(data, tuple):
            self._progress(*data)
        else:
            self._status(str(data))

    def _clear_progress(self) -> None:
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def _set_running_stage(self, stage_key: str | None) -> None:
        """``None`` means idle; ``'nucleus'`` or ``'cell'`` claims the row."""
        self._running_stage = stage_key
        rows = {
            "nucleus": (
                self.nucleus_params_btn,
                self.nucleus_preview_btn,
                self.nucleus_run_btn,
                "Run nucleus Cellpose on all frames.",
            ),
            "cell": (
                self.cell_params_btn,
                self.cell_preview_btn,
                self.cell_run_btn,
                "Run cell Cellpose on all frames.",
            ),
        }
        if stage_key is None:
            for params_btn, preview_btn, run_btn, tooltip in rows.values():
                params_btn.setEnabled(True)
                preview_btn.setEnabled(True)
                run_btn.setEnabled(True)
                run_btn.setText("▶")
                run_btn.setToolTip(tooltip)
            self._cancel_requested = False
            return
        for key, (params_btn, preview_btn, run_btn, _tooltip) in rows.items():
            if key == stage_key:
                params_btn.setEnabled(True)
                preview_btn.setEnabled(False)
                run_btn.setEnabled(True)
                run_btn.setText("✕")
                run_btn.setToolTip("Cancel.")
            else:
                params_btn.setEnabled(False)
                preview_btn.setEnabled(False)
                run_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Layer helper (mirrors CellWorkflowWidget._show_layer)
    # ------------------------------------------------------------------
    def _show_layer(self, name, data, kwargs, adder):
        if name in self.viewer.layers:
            try:
                self.viewer.layers[name].data = data
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[name])
                adder(data, name=name, **kwargs)
        else:
            adder(data, name=name, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/napari/test_cellpose_widget.py -v`
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/cellpose_widget.py tests/napari/test_cellpose_widget.py
git commit -m "feat(napari): scaffold CellposeWidget with rows, params, state I/O"
```

---

## Task 8: Widget — run flow (nucleus + cell shared worker)

**Files:**
- Modify: `src/cellflow/napari/cellpose_widget.py`
- Modify: `tests/napari/test_cellpose_widget.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/napari/test_cellpose_widget.py`:

```python
def _make_sync_thread_worker():
    """Patch thread_worker so workers execute synchronously."""
    import inspect

    def fake_thread_worker(connect=None):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    if connect and "errored" in connect:
                        connect["errored"](exc)
                    return None
                if inspect.isgenerator(result):
                    return_value = None
                    while True:
                        try:
                            yielded = next(result)
                        except StopIteration as exc:
                            return_value = exc.value
                            break
                        if connect and "yielded" in connect:
                            connect["yielded"](yielded)
                    if connect and "returned" in connect:
                        connect["returned"](return_value)
                else:
                    if connect and "returned" in connect:
                        connect["returned"](result)
                return None
            return wrapper
        return decorator
    return fake_thread_worker


def _write_test_stack(path: Path, shape):
    arr = np.zeros(shape, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tifffile as tf

    tf.imwrite(str(path), arr)


def test_run_nucleus_writes_outputs_and_updates_status(_mock_cellpose, monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    _write_test_stack(tmp_path / "0_input" / "nucleus_3dt.tif", (2, 3, 6, 6))
    w.refresh(tmp_path)
    w.nucleus_run_btn.click()
    out = tmp_path / "1_cellpose"
    assert (out / "nucleus_prob_3dt.tif").exists()
    assert (out / "nucleus_dp_3dt.tif").exists()
    assert (out / "nucleus_prob_zavg.tif").exists()
    assert "complete" in w.status_lbl.text().lower()
    assert w.nucleus_run_btn.text() == "▶"
    w.deleteLater()


def test_run_cell_writes_outputs(_mock_cellpose, monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    _write_test_stack(tmp_path / "0_input" / "cell_3dt.tif", (2, 3, 6, 6))
    w.refresh(tmp_path)
    w.cell_run_btn.click()
    out = tmp_path / "1_cellpose"
    assert (out / "cell_prob_3dt.tif").exists()
    assert (out / "cell_dp_3dt.tif").exists()
    assert (out / "cell_prob_zavg.tif").exists()
    w.deleteLater()


def test_run_reports_missing_input(_mock_cellpose, monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    w.refresh(tmp_path)  # no input tif written
    w.nucleus_run_btn.click()
    assert "missing" in w.status_lbl.text().lower()
    w.deleteLater()


def test_run_with_no_project_reports_status(_mock_cellpose, monkeypatch):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    w.nucleus_run_btn.click()
    assert "no project" in w.status_lbl.text().lower()
    w.deleteLater()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/napari/test_cellpose_widget.py -v`
Expected: The four new tests FAIL — the run buttons currently do nothing for the idle case, so no files are written and no status is set.

- [ ] **Step 3: Implement the run flow**

In `src/cellflow/napari/cellpose_widget.py`, replace the `_on_nucleus_run_clicked` and `_on_cell_run_clicked` stubs (currently only handle the cancel branch) with:

```python
    def _on_nucleus_run_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
            return
        self._run_channel("nucleus")

    def _on_cell_run_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
            return
        self._run_channel("cell")
```

Then add these methods to `CellposeWidget` (place them after `_on_cancel`):

```python
    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _input_path(self, channel: str) -> Path | None:
        if self._pos_dir is None:
            return None
        name = "nucleus_3dt.tif" if channel == "nucleus" else "cell_3dt.tif"
        return self._pos_dir / "0_input" / name

    def _output_dir(self) -> Path | None:
        return None if self._pos_dir is None else self._pos_dir / "1_cellpose"

    # ------------------------------------------------------------------
    # Run flow
    # ------------------------------------------------------------------
    def _build_nucleus_params(self) -> "cellpose_runner.NucleusParams":
        return cellpose_runner.NucleusParams(
            do_3d=self.nuc_3d_chk.isChecked(),
            anisotropy=float(self.nuc_anisotropy_spin.value()),
            diameter=float(self.nuc_diameter_spin.value()),
            min_size=int(self.nuc_min_size_spin.value()),
            gamma=float(self.nuc_gamma_spin.value()),
        )

    def _build_cell_params(self) -> "cellpose_runner.CellParams":
        return cellpose_runner.CellParams(
            diameter=float(self.cell_diameter_spin.value()),
            min_size=int(self.cell_min_size_spin.value()),
            gamma=float(self.cell_gamma_spin.value()),
        )

    def _run_channel(self, channel: str) -> None:
        if self._pos_dir is None:
            self._status("No project open.")
            return
        in_path = self._input_path(channel)
        if in_path is None or not in_path.exists():
            self._status(f"Missing: {in_path.name if in_path else '(no path)'}")
            return
        out_dir = self._output_dir()
        params = (
            self._build_nucleus_params() if channel == "nucleus"
            else self._build_cell_params()
        )
        pos_dir = self._pos_dir
        self._cancel_requested = False

        def _done(result):
            self._worker = None
            self._set_running_stage(None)
            self._clear_progress()
            self._files_widget.refresh(pos_dir)
            label = "Nucleus" if channel == "nucleus" else "Cell"
            self._status(f"{label} Cellpose complete — wrote {channel}_*_3dt.tif")

        def _error(exc):
            self._worker = None
            self._set_running_stage(None)
            self._clear_progress()
            if isinstance(exc, cellpose_runner.CancelledError):
                self._status("Cancelled.")
            else:
                self._status(f"Error: {exc}")
                logger.exception("Cellpose run error", exc_info=exc)

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": _error,
        })
        def _worker():
            yield (0, 1, "Loading input...")
            stack = np.asarray(tifffile.imread(str(in_path)))
            if stack.ndim != 4:
                raise ValueError(
                    f"expected 4D (T,Z,Y,X) input, got shape {stack.shape}"
                )

            def _cb_progress(done, total, msg):
                # Forward into the worker's yield stream via the closure variable.
                _progress_queue.append((done, total, msg))

            def _cb_cancel():
                return self._cancel_requested

            _progress_queue: list[tuple[int, int, str]] = []
            if channel == "nucleus":
                prob, dp = cellpose_runner.run_nucleus_stack(
                    stack, params,
                    progress_cb=_cb_progress, cancel_cb=_cb_cancel,
                )
            else:
                prob, dp = cellpose_runner.run_cell_stack(
                    stack, params,
                    progress_cb=_cb_progress, cancel_cb=_cb_cancel,
                )
            # Replay queued progress updates so the UI receives them; in
            # practice this runs at the end because the callback fires inside
            # the heavy compute. That's fine for status text purposes — the
            # final state will be the last frame.
            for item in _progress_queue:
                yield item
            yield (1, 1, "Writing outputs...")
            cellpose_runner.write_outputs(prob, dp, out_dir, channel)
            return None

        self._set_running_stage(channel)
        self._status(
            f"Loading Cellpose-SAM model on {cellpose_runner.device_label()} "
            f"(~10s on first run)..." if not cellpose_runner.is_model_loaded()
            else f"Running {channel} Cellpose..."
        )
        self._worker = _worker()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/napari/test_cellpose_widget.py -v`
Expected: 10 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/cellpose_widget.py tests/napari/test_cellpose_widget.py
git commit -m "feat(napari): wire CellposeWidget run flow to runner with cancel"
```

---

## Task 9: Widget — preview flow

**Files:**
- Modify: `src/cellflow/napari/cellpose_widget.py`
- Modify: `tests/napari/test_cellpose_widget.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/napari/test_cellpose_widget.py`:

```python
def test_nucleus_preview_2d_creates_layers(_mock_cellpose, monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    viewer = _FakeViewer()
    viewer.dims.current_step = (1, 2)  # t=1, z=2
    w = mod.CellposeWidget(viewer)
    _write_test_stack(tmp_path / "0_input" / "nucleus_3dt.tif", (3, 4, 6, 6))
    w.refresh(tmp_path)
    w.nuc_3d_chk.setChecked(False)
    w.nucleus_preview_btn.click()
    assert "Preview: Nucleus prob" in viewer.layers
    assert "Preview: Nucleus flow" in viewer.layers
    prob = viewer.layers["Preview: Nucleus prob"].data
    assert prob.shape == (3, 6, 6)
    assert np.all(prob[0] == 0) and np.all(prob[2] == 0)  # only t=1 populated
    w.deleteLater()


def test_nucleus_preview_3d_creates_volume_layers(_mock_cellpose, monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    viewer = _FakeViewer()
    viewer.dims.current_step = (1, 0)
    w = mod.CellposeWidget(viewer)
    _write_test_stack(tmp_path / "0_input" / "nucleus_3dt.tif", (3, 4, 6, 6))
    w.refresh(tmp_path)
    w.nuc_3d_chk.setChecked(True)
    w.nucleus_preview_btn.click()
    prob = viewer.layers["Preview: Nucleus prob"].data
    flow = viewer.layers["Preview: Nucleus flow"].data
    assert prob.shape == (3, 4, 6, 6)
    assert flow.shape == (3, 4, 6, 6)
    w.deleteLater()


def test_cell_preview_creates_2d_layers(_mock_cellpose, monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    viewer = _FakeViewer()
    viewer.dims.current_step = (2, 1)
    w = mod.CellposeWidget(viewer)
    _write_test_stack(tmp_path / "0_input" / "cell_3dt.tif", (3, 4, 5, 5))
    w.refresh(tmp_path)
    w.cell_preview_btn.click()
    prob = viewer.layers["Preview: Cell prob"].data
    flow = viewer.layers["Preview: Cell flow"].data
    assert prob.shape == (3, 5, 5)
    assert flow.shape == (3, 5, 5)
    w.deleteLater()


def test_preview_reports_missing_input(_mock_cellpose, monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    w.refresh(tmp_path)
    w.nucleus_preview_btn.click()
    assert "missing" in w.status_lbl.text().lower()
    w.deleteLater()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/napari/test_cellpose_widget.py -v`
Expected: New preview tests FAIL — the preview handlers are still stubs.

- [ ] **Step 3: Implement the preview flow**

In `src/cellflow/napari/cellpose_widget.py`, replace the `_on_nucleus_preview` and `_on_cell_preview` stubs with:

```python
    def _on_nucleus_preview(self) -> None:
        self._preview_channel("nucleus")

    def _on_cell_preview(self) -> None:
        self._preview_channel("cell")
```

Then add (place after `_run_channel`):

```python
    # ------------------------------------------------------------------
    # Preview flow
    # ------------------------------------------------------------------
    def _current_tz(self) -> tuple[int, int]:
        step = getattr(getattr(self.viewer, "dims", None), "current_step", (0, 0))
        t = int(step[0]) if len(step) >= 1 else 0
        z = int(step[1]) if len(step) >= 2 else 0
        return t, z

    @staticmethod
    def _flow_magnitude(dp: np.ndarray) -> np.ndarray:
        # dp has shape (C, ...) — sum-of-squares over the channel axis.
        return np.sqrt(np.sum(np.asarray(dp, dtype=np.float32) ** 2, axis=0))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

    def _preview_channel(self, channel: str) -> None:
        if self._pos_dir is None:
            self._status("No project open.")
            return
        in_path = self._input_path(channel)
        if in_path is None or not in_path.exists():
            self._status(f"Missing: {in_path.name if in_path else '(no path)'}")
            return

        t, z = self._current_tz()
        stack = np.asarray(tifffile.imread(str(in_path)))
        if stack.ndim != 4:
            self._status(f"Expected 4D input (T,Z,Y,X), got {stack.shape}")
            return
        T, Z = stack.shape[:2]
        t = min(max(t, 0), T - 1)
        z = min(max(z, 0), Z - 1)

        if channel == "nucleus":
            params = self._build_nucleus_params()
            if params.do_3d:
                prob_logits, dp = cellpose_runner.run_nucleus_frame(
                    stack[t], z=None, params=params,
                )
                prob = self._sigmoid(prob_logits)
                flow = self._flow_magnitude(dp)  # (Z, Y, X)
                prob_full = np.zeros((T, Z, *prob.shape[-2:]), dtype=np.float32)
                flow_full = np.zeros_like(prob_full)
                prob_full[t] = prob
                flow_full[t] = flow
                self._status(
                    f"Preview: nucleus 3D t={t} (Z={Z}, anisotropy={params.anisotropy})"
                )
            else:
                prob_logits, dp = cellpose_runner.run_nucleus_frame(
                    stack[t], z=z, params=params,
                )
                prob = self._sigmoid(prob_logits)
                flow = self._flow_magnitude(dp)  # (Y, X)
                prob_full = np.zeros((T, *prob.shape), dtype=np.float32)
                flow_full = np.zeros_like(prob_full)
                prob_full[t] = prob
                flow_full[t] = flow
                self._status(
                    f"Preview: nucleus 2D t={t} z={z} (diameter={params.diameter})"
                )
            self._show_layer(
                "Preview: Nucleus prob", prob_full,
                {"colormap": "viridis", "blending": "additive"},
                self.viewer.add_image,
            )
            self._show_layer(
                "Preview: Nucleus flow", flow_full,
                {"colormap": "inferno", "blending": "additive"},
                self.viewer.add_image,
            )
            return

        # Cell preview — always 2D
        params = self._build_cell_params()
        prob_logits, dp = cellpose_runner.run_cell_frame(stack[t], z=z, params=params)
        prob = self._sigmoid(prob_logits)
        flow = self._flow_magnitude(dp)
        prob_full = np.zeros((T, *prob.shape), dtype=np.float32)
        flow_full = np.zeros_like(prob_full)
        prob_full[t] = prob
        flow_full[t] = flow
        self._status(
            f"Preview: cell t={t} z={z} (diameter={params.diameter})"
        )
        self._show_layer(
            "Preview: Cell prob", prob_full,
            {"colormap": "viridis", "blending": "additive"},
            self.viewer.add_image,
        )
        self._show_layer(
            "Preview: Cell flow", flow_full,
            {"colormap": "inferno", "blending": "additive"},
            self.viewer.add_image,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/napari/test_cellpose_widget.py -v`
Expected: 14 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/cellpose_widget.py tests/napari/test_cellpose_widget.py
git commit -m "feat(napari): add per-channel preview producing in-memory layers"
```

---

## Task 10: main_widget integration — replace placeholder + persist state

**Files:**
- Modify: `src/cellflow/napari/main_widget.py`
- Create: `tests/napari/test_main_widget_cellpose_integration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/napari/test_main_widget_cellpose_integration.py`:

```python
"""Integration test: CellFlowMainWidget uses the new CellposeWidget."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def _mock_cellpose(monkeypatch):
    fake_cellpose = types.ModuleType("cellpose")
    fake_models = types.ModuleType("cellpose.models")

    class _FakeModel:
        def __init__(self, *_, **__): pass

        def eval(self, img, **_kwargs):
            arr = np.asarray(img, dtype=np.float32)
            ndim = 3 if arr.ndim == 3 else 2
            chans = 3 if ndim == 3 else 2
            dp = np.zeros((chans, *arr.shape), dtype=np.float32)
            prob = np.zeros(arr.shape, dtype=np.float32)
            return None, (None, dp, prob), None

    fake_models.CellposeModel = _FakeModel
    fake_cellpose.models = fake_models
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)
    monkeypatch.delitem(sys.modules, "cellflow.segmentation.cellpose_runner", raising=False)
    monkeypatch.delitem(sys.modules, "cellflow.napari.cellpose_widget", raising=False)
    monkeypatch.delitem(sys.modules, "cellflow.napari.main_widget", raising=False)


def _fake_viewer():
    class _Sel:
        active = None

    class _Layers(dict):
        selection = _Sel()
        events = SimpleNamespace(removed=SimpleNamespace(connect=lambda cb: None))

        def remove(self, layer):
            self.pop(layer.name, None)

    viewer = SimpleNamespace()
    viewer.layers = _Layers()
    viewer.dims = SimpleNamespace(
        current_step=(0, 0),
        events=SimpleNamespace(current_step=SimpleNamespace(connect=lambda cb: None)),
    )
    viewer.add_image = MagicMock()
    viewer.add_labels = MagicMock()
    viewer.add_shapes = MagicMock()
    return viewer


def test_main_widget_constructs_new_cellpose_widget():
    QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    cellpose_mod = importlib.import_module("cellflow.napari.cellpose_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())
    assert isinstance(w._cellpose_widget, cellpose_mod.CellposeWidget)
    # Old placeholder class no longer exists.
    assert not hasattr(main_mod, "_CellposePanel")
    w.deleteLater()


def test_main_widget_state_round_trips_cellpose():
    QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())
    cellpose_state = {
        "nucleus": {
            "do_3d": False,
            "anisotropy": 1.25,
            "diameter": 33.0,
            "min_size": 9,
            "gamma": 1.2,
        },
        "cell": {"diameter": 17.0, "min_size": 4, "gamma": 0.9},
    }
    w.set_state({"cellpose": cellpose_state})
    got = w.get_state()
    assert "cellpose" in got
    assert got["cellpose"] == cellpose_state
    w.deleteLater()


def test_main_widget_pipeline_status_uses_output_files_tracker(tmp_path):
    QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())
    # Should reach pipeline_status_from_files without error (tracker exists).
    assert w._cellpose_widget.output_files_tracker is not None
    w._update_section_statuses()
    w.deleteLater()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/napari/test_main_widget_cellpose_integration.py -v`
Expected: First test FAILS — `_cellpose_widget` is still the old `_CellposePanel`. Second test FAILS — `get_state()` has no `"cellpose"` key.

- [ ] **Step 3: Modify main_widget.py**

In `src/cellflow/napari/main_widget.py`:

1. Replace the import block at lines 22–33 so it no longer imports the placeholder's helpers and adds the new widget. Find:

```python
from cellflow.napari.cellpose_zavg_viz_widget import CellposeZavgVizWidget
from cellflow.napari.contact_analysis_widget import ContactAnalysisWidget
from cellflow.napari.cell_workflow_widget import CellWorkflowWidget
from cellflow.napari.data_panel_widget import ProjectStatusPanel
from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget
from cellflow.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    make_pipeline_files_header,
    pipeline_status_from_files,
)
from cellflow.napari.ui_style import icon_button, muted_label, stage_accent, tiny_button
```

Replace with:

```python
from cellflow.napari.cellpose_widget import CellposeWidget
from cellflow.napari.contact_analysis_widget import ContactAnalysisWidget
from cellflow.napari.cell_workflow_widget import CellWorkflowWidget
from cellflow.napari.data_panel_widget import ProjectStatusPanel
from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget
from cellflow.napari.widgets import (
    CollapsibleSection,
    pipeline_status_from_files,
)
from cellflow.napari.ui_style import icon_button, muted_label, stage_accent, tiny_button
```

2. Delete the entire `_CellposePanel` class (currently lines 36–99, from `class _CellposePanel(QWidget):` up to and including its `refresh` method's last `self.zavg_viz_widget.refresh(pos_dir)` line). Leave one blank line before `class CellFlowMainWidget`.

3. In `CellFlowMainWidget.__init__`, find:

```python
        self._cellpose_widget = _CellposePanel(self.viewer)
```

Replace with:

```python
        self._cellpose_widget = CellposeWidget(self.viewer)
```

4. In `get_state`, find:

```python
    def get_state(self) -> dict:
        """Return the current UI state as a dictionary."""
        return {
            "metadata": {
                "pixel_size_um": self.px_edit.text(),
                "time_interval_s": self.dt_edit.text(),
                "condition": self.cond_edit.text(),
                "position": self.pos_spin.value(),
            },
            "nucleus": self.nucleus_workflow_widget.get_state(),
            "cell": self.cell_workflow_widget.get_state(),
        }
```

Replace with:

```python
    def get_state(self) -> dict:
        """Return the current UI state as a dictionary."""
        return {
            "metadata": {
                "pixel_size_um": self.px_edit.text(),
                "time_interval_s": self.dt_edit.text(),
                "condition": self.cond_edit.text(),
                "position": self.pos_spin.value(),
            },
            "cellpose": self._cellpose_widget.get_state(),
            "nucleus": self.nucleus_workflow_widget.get_state(),
            "cell": self.cell_workflow_widget.get_state(),
        }
```

5. In `set_state`, find:

```python
        if "nucleus" in state:
            self.nucleus_workflow_widget.set_state(state["nucleus"])
        
        if "cell" in state:
            self.cell_workflow_widget.set_state(state["cell"])
```

Replace with:

```python
        if "cellpose" in state:
            self._cellpose_widget.set_state(state["cellpose"])

        if "nucleus" in state:
            self.nucleus_workflow_widget.set_state(state["nucleus"])

        if "cell" in state:
            self.cell_workflow_widget.set_state(state["cell"])
```

6. Remove the now-unused `QLabel` import from the qtpy import block if it is no longer referenced after removing `_CellposePanel`. Run `python -c "import ast, sys; tree=ast.parse(open('src/cellflow/napari/main_widget.py').read()); print('QLabel used:', 'QLabel' in open('src/cellflow/napari/main_widget.py').read())"`. If `QLabel` still appears (it does, in `_setup_project_ui`), keep the import. Same for `PipelineFilesWidget` (now unused after the edit) — drop it from the `widgets` import as done in step 1, and drop `make_pipeline_files_header` too as shown.

- [ ] **Step 4: Run integration tests to verify they pass**

Run: `pytest tests/napari/test_main_widget_cellpose_integration.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Run the full napari test suite to catch regressions**

Run: `pytest tests/napari/ tests/segmentation/test_cellpose_runner.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/main_widget.py tests/napari/test_main_widget_cellpose_integration.py
git commit -m "feat(napari): wire CellposeWidget into CellFlowMainWidget and persist state"
```

---

## Self-Review

Spec coverage check:

| Spec requirement | Task |
|---|---|
| New `segmentation/cellpose_runner.py` with `NucleusParams`/`CellParams` | Task 2 |
| `get_model()` / `is_model_loaded()` with lazy cpsam cache | Task 3 |
| `_apply_gamma` matching cellpose_full | Task 2 |
| `run_nucleus_frame` (2D and 3D) | Task 4 |
| `run_cell_frame` | Task 4 |
| `run_nucleus_stack` / `run_cell_stack` with progress + cancel | Task 5 |
| `write_outputs` writing canonical `{channel}_prob_3dt.tif`, `{channel}_dp_3dt.tif`, `{channel}_prob_zavg.tif` | Task 6 |
| New widget `CellposeWidget` with two stage rows + shared status/progress + embedded `CellposeZavgVizWidget` + `PipelineFilesWidget` | Task 7 |
| `_set_running_stage` mutual exclusion | Task 7 |
| `get_state` / `set_state` for nucleus + cell sub-dicts | Task 7 |
| `refresh(pos_dir)` propagates to files + zavg widget | Task 7 |
| `output_files_tracker` attribute preserved | Task 7 |
| Run flow: load tif, build params, worker, write outputs, status messages, cancel | Task 8 |
| Device-aware first-load status message (`"... on cuda:0 ..."` / `"... on cpu ..."`) | Task 8 |
| Preview flow: 2D nucleus, 3D nucleus, 2D cell; sigmoid prob + flow magnitude; zero-padded full-T arrays | Task 9 |
| Replace `_CellposePanel` and construction site in `main_widget.py` | Task 10 |
| Persist under new top-level `"cellpose"` key | Task 10 |
| `pipeline_status_from_files(self._cellpose_widget.output_files_tracker, ...)` still works | Task 10 (covered by integration test) |
| Add `cellpose>=4.0` to `pyproject.toml` | Task 1 |

All spec requirements map to tasks. The runner tests cover the testing checklist (dataclasses, `get_model` caching, gamma, `write_outputs` filenames + dtypes + zavg axis, stack iteration order with `progress_cb`/`cancel_cb`, per-frame shapes). The widget tests cover params toggle, `get_state`/`set_state` round-trip, `refresh(None)`, preview wiring (mocked runner via the autouse `_mock_cellpose` fixture which makes the real runner return zero arrays), and `_set_running_stage` enable/disable + glyph swap.

No GPU-requiring code paths in tests; the cellpose mock fixture is autouse and forces the runner to use the `_FakeModel`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-18-cellpose-widget.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
