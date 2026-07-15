# Cellpose Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sigmoid-probability heatmap + flow-vector overlays for Cellpose output tifs (nucleus and cell channels, z-avg and 3D+t modes) via a new helper module and two QGroupBoxes in `HpcCellposeWidget`.

**Architecture:** Pure-numpy helper module `cellpose_visualization.py` (no Qt, mirrors `artifact_visualization.py`) holds all math + napari layer construction. `HpcCellposeWidget` gains two `QGroupBox` sections (nuclei viz, cells viz) with stride/scale spinboxes and Load buttons; it delegates entirely to the helper for layer operations.

**Tech Stack:** numpy, tifffile, napari (Viewer, Image, Vectors layers), qtpy (QGroupBox, QHBoxLayout, QSpinBox, QDoubleSpinBox, QPushButton, QLabel), pytest.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/cellflow/napari/cellpose_visualization.py` | All math (sigmoid, Z-mean, stride/scale, vector-array building) and napari layer operations |
| Modify | `src/cellflow/napari/hpc_cellpose_widget.py` | Add two `QGroupBox` viz sections; wire buttons to helper; extend `get_state`/`set_state` |
| Create | `tests/napari/test_cellpose_visualization.py` | Tests for all three public functions in the new module |
| Modify | `tests/napari/test_hpc_cellpose_widget.py` | Button enable/disable tests; extended `get_state`/`set_state` round-trip |

---

## Task 1: New helper module — load_sigmoid_prob

**Files:**
- Create: `src/cellflow/napari/cellpose_visualization.py`
- Create: `tests/napari/test_cellpose_visualization.py`

- [ ] **Step 1: Write the failing tests for `load_sigmoid_prob`**

```python
# tests/napari/test_cellpose_visualization.py
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _load_viz_mod():
    """Import cellpose_visualization without the rest of the package."""
    src_root = Path(__file__).resolve().parents[2] / "src" / "cellflow"
    package_root = src_root / "napari"

    if "cellflow.napari" not in sys.modules:
        pkg = types.ModuleType("cellflow.napari")
        pkg.__path__ = [str(package_root)]
        sys.modules["cellflow.napari"] = pkg

    sys.modules.pop("cellflow.napari.cellpose_visualization", None)
    return importlib.import_module("cellflow.napari.cellpose_visualization")


@pytest.fixture
def mod():
    return _load_viz_mod()


def _write_prob(tmp_path: Path, channel: str, data: np.ndarray) -> Path:
    p = tmp_path / f"{channel}_prob_3dt.tif"
    tifffile.imwrite(str(p), data)
    return p


def _write_dp(tmp_path: Path, channel: str, data: np.ndarray) -> Path:
    p = tmp_path / f"{channel}_dp_3dt.tif"
    tifffile.imwrite(str(p), data)
    return p


# ----- load_sigmoid_prob -----

def test_load_sigmoid_prob_zavg_shape(mod, tmp_path):
    T, Z, Y, X = 3, 4, 8, 8
    data = np.random.randn(T, Z, Y, X).astype(np.float32)
    _write_prob(tmp_path, "nucleus", data)
    result = mod.load_sigmoid_prob(tmp_path, "nucleus", "zavg")
    assert result.shape == (T, Y, X)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_load_sigmoid_prob_3dt_shape(mod, tmp_path):
    T, Z, Y, X = 2, 5, 6, 6
    data = np.random.randn(T, Z, Y, X).astype(np.float32)
    _write_prob(tmp_path, "cell", data)
    result = mod.load_sigmoid_prob(tmp_path, "cell", "3dt")
    assert result.shape == (T, Z, Y, X)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_load_sigmoid_prob_zavg_sigmoid_then_mean(mod, tmp_path):
    """Regression: sigmoid-then-mean must differ from mean-then-sigmoid."""
    T, Z, Y, X = 2, 4, 8, 8
    rng = np.random.default_rng(42)
    data = (rng.random((T, Z, Y, X)) * 10 - 5).astype(np.float32)
    _write_prob(tmp_path, "nucleus", data)

    result = mod.load_sigmoid_prob(tmp_path, "nucleus", "zavg")

    # mean-then-sigmoid would give a different result
    wrong = 1.0 / (1.0 + np.exp(-data.mean(axis=1)))
    assert not np.allclose(result, wrong), (
        "sigmoid-then-mean should differ from mean-then-sigmoid"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/aruppel/Projects/CellFlow && uv run pytest tests/napari/test_cellpose_visualization.py::test_load_sigmoid_prob_zavg_shape -x 2>&1 | tail -20
```
Expected: FAIL with `ModuleNotFoundError` or `AttributeError`.

- [ ] **Step 3: Create the module skeleton with `load_sigmoid_prob`**

```python
# src/cellflow/napari/cellpose_visualization.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import tifffile

__all__ = [
    "load_sigmoid_prob",
    "load_flow_vectors",
    "add_cellpose_viz_layers",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float32)))).astype(np.float32)


def _prob_filename(channel: str) -> str:
    return f"{channel}_prob_3dt.tif"


def _dp_filename(channel: str) -> str:
    return f"{channel}_dp_3dt.tif"


def load_sigmoid_prob(
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
    mode: Literal["zavg", "3dt"],
) -> np.ndarray:
    path = Path(output_dir) / _prob_filename(channel)
    raw = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    prob = _sigmoid(raw)  # (T, Z, Y, X)
    if mode == "zavg":
        return prob.mean(axis=1)  # (T, Y, X)
    return prob  # (T, Z, Y, X)
```

- [ ] **Step 4: Run all three sigmoid tests**

```bash
cd /home/aruppel/Projects/CellFlow && uv run pytest tests/napari/test_cellpose_visualization.py -k "sigmoid_prob" -x 2>&1 | tail -20
```
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/aruppel/Projects/CellFlow && git add src/cellflow/napari/cellpose_visualization.py tests/napari/test_cellpose_visualization.py && git commit -m "feat: add cellpose_visualization module with load_sigmoid_prob"
```

---

## Task 2: `load_flow_vectors` — z-avg and 3D+t modes

**Files:**
- Modify: `src/cellflow/napari/cellpose_visualization.py`
- Modify: `tests/napari/test_cellpose_visualization.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/napari/test_cellpose_visualization.py`:

```python
# ----- load_flow_vectors -----

def test_load_flow_vectors_zavg_shape(mod, tmp_path):
    T, Z, Y, X = 2, 4, 16, 16
    dp = np.random.randn(T, Z, 2, Y, X).astype(np.float32)
    _write_dp(tmp_path, "nucleus", dp)
    stride = 4
    result = mod.load_flow_vectors(tmp_path, "nucleus", "zavg", stride=stride, scale=1.0)
    N_expected = T * (Y // stride) * (X // stride)
    assert result.shape == (N_expected, 2, 3)


def test_load_flow_vectors_zavg_respects_stride(mod, tmp_path):
    T, Z, Y, X = 1, 2, 12, 12
    dp = np.ones((T, Z, 2, Y, X), dtype=np.float32)
    _write_dp(tmp_path, "nucleus", dp)
    stride = 3
    result = mod.load_flow_vectors(tmp_path, "nucleus", "zavg", stride=stride, scale=1.0)
    N_expected = T * (Y // stride) * (X // stride)
    assert result.shape == (N_expected, 2, 3)


def test_load_flow_vectors_3dt_shape_and_dz_zero(mod, tmp_path):
    T, Z, Y, X = 2, 3, 16, 16
    dp = np.random.randn(T, Z, 2, Y, X).astype(np.float32)
    _write_dp(tmp_path, "nucleus", dp)
    stride = 4
    result = mod.load_flow_vectors(tmp_path, "nucleus", "3dt", stride=stride, scale=1.0)
    N_expected = T * Z * (Y // stride) * (X // stride)
    assert result.shape == (N_expected, 2, 4)
    # dz component (index 3 of vector part, position [1, :, 3]) should all be 0
    # result[i, 1, :] is [dz, dy, dx] but spec says (t,z,y,x) → dz at index 1
    # Vector format: result[:, 0, :] = positions (t,z,y,x); result[:, 1, :] = (0, 0, dy*scale, dx*scale)
    dz_components = result[:, 1, 1]  # Z-delta is at index 1 of the delta vector
    assert np.all(dz_components == 0.0)


def test_load_flow_vectors_scale_applied(mod, tmp_path):
    T, Z, Y, X = 1, 2, 8, 8
    dy_val, dx_val = 2.0, 3.0
    dp = np.zeros((T, Z, 2, Y, X), dtype=np.float32)
    dp[:, :, 0, :, :] = dy_val  # dy channel
    dp[:, :, 1, :, :] = dx_val  # dx channel
    _write_dp(tmp_path, "nucleus", dp)
    stride = 2
    scale = 0.5
    result = mod.load_flow_vectors(tmp_path, "nucleus", "zavg", stride=stride, scale=scale)
    # After Z-mean, dy=2.0, dx=3.0. After scale: dy*0.5=1.0, dx*0.5=1.5
    dy_col = result[:, 1, 1]  # dy is at index 1 of the delta vector for zavg (t, dy, dx)
    dx_col = result[:, 1, 2]  # dx is at index 2
    assert np.allclose(dy_col, dy_val * scale)
    assert np.allclose(dx_col, dx_val * scale)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/aruppel/Projects/CellFlow && uv run pytest tests/napari/test_cellpose_visualization.py -k "flow_vectors" -x 2>&1 | tail -20
```
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `load_flow_vectors`**

Add to `src/cellflow/napari/cellpose_visualization.py` after `load_sigmoid_prob`:

```python
def load_flow_vectors(
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
    mode: Literal["zavg", "3dt"],
    *,
    stride: int,
    scale: float,
) -> np.ndarray:
    path = Path(output_dir) / _dp_filename(channel)
    dp = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    # dp shape: (T, Z, 2, Y, X) — channels are (dy, dx)
    T, Z, _, Y, X = dp.shape

    if mode == "zavg":
        # mean over Z → (T, 2, Y, X)
        dp_2d = dp.mean(axis=1)
        ys = np.arange(0, Y, stride)
        xs = np.arange(0, X, stride)
        ts = np.arange(T)
        tg, yg, xg = np.meshgrid(ts, ys, xs, indexing="ij")  # (T, nY, nX)
        nY, nX = len(ys), len(xs)
        tg = tg.ravel().astype(np.float32)
        yg_r = yg.ravel().astype(np.float32)
        xg_r = xg.ravel().astype(np.float32)
        dy = dp_2d[:, 0, :, :][:, ys[:, None], xs[None, :]].ravel() * scale
        dx = dp_2d[:, 1, :, :][:, ys[:, None], xs[None, :]].ravel() * scale
        N = len(tg)
        vectors = np.zeros((N, 2, 3), dtype=np.float32)
        vectors[:, 0, 0] = tg
        vectors[:, 0, 1] = yg_r
        vectors[:, 0, 2] = xg_r
        vectors[:, 1, 1] = dy
        vectors[:, 1, 2] = dx
        return vectors

    # 3D+t mode
    ys = np.arange(0, Y, stride)
    xs = np.arange(0, X, stride)
    ts = np.arange(T)
    zs = np.arange(Z)
    tg, zg, yg, xg = np.meshgrid(ts, zs, ys, xs, indexing="ij")  # (T, Z, nY, nX)
    tg = tg.ravel().astype(np.float32)
    zg_r = zg.ravel().astype(np.float32)
    yg_r = yg.ravel().astype(np.float32)
    xg_r = xg.ravel().astype(np.float32)
    dy = dp[:, :, 0, :, :][:, :, ys[:, None], xs[None, :]].ravel() * scale
    dx = dp[:, :, 1, :, :][:, :, ys[:, None], xs[None, :]].ravel() * scale
    N = len(tg)
    vectors = np.zeros((N, 2, 4), dtype=np.float32)
    vectors[:, 0, 0] = tg
    vectors[:, 0, 1] = zg_r
    vectors[:, 0, 2] = yg_r
    vectors[:, 0, 3] = xg_r
    # vectors[:, 1, 1] = 0  (dz = 0, already zeros)
    vectors[:, 1, 2] = dy
    vectors[:, 1, 3] = dx
    return vectors
```

- [ ] **Step 4: Run flow_vectors tests**

```bash
cd /home/aruppel/Projects/CellFlow && uv run pytest tests/napari/test_cellpose_visualization.py -k "flow_vectors" -x 2>&1 | tail -20
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/aruppel/Projects/CellFlow && git add src/cellflow/napari/cellpose_visualization.py tests/napari/test_cellpose_visualization.py && git commit -m "feat: add load_flow_vectors to cellpose_visualization"
```

---

## Task 3: `add_cellpose_viz_layers` — orchestration function

**Files:**
- Modify: `src/cellflow/napari/cellpose_visualization.py`
- Modify: `tests/napari/test_cellpose_visualization.py`

- [ ] **Step 1: Write the failing tests**

The `add_cellpose_viz_layers` tests use a real `napari.Viewer` (see pattern from `test_nucleus_db_browser_widget.py`). Add to `tests/napari/test_cellpose_visualization.py`:

```python
# ----- add_cellpose_viz_layers -----

import napari
from qtpy.QtWidgets import QApplication


def _make_viewer():
    app = QApplication.instance() or QApplication([])
    viewer = napari.Viewer(show=False)
    return app, viewer


def test_add_cellpose_viz_layers_adds_prob_and_flow(mod, tmp_path):
    T, Z, Y, X = 2, 3, 16, 16
    prob = np.random.randn(T, Z, Y, X).astype(np.float32)
    dp = np.random.randn(T, Z, 2, Y, X).astype(np.float32)
    _write_prob(tmp_path, "nucleus", prob)
    _write_dp(tmp_path, "nucleus", dp)

    app, viewer = _make_viewer()
    try:
        layers = mod.add_cellpose_viz_layers(
            viewer, tmp_path, "nucleus", "zavg", stride=4, scale=1.0
        )
        assert len(layers) == 2
        names = [la.name for la in viewer.layers]
        assert "Cellpose viz: nucleus prob (z-avg)" in names
        assert "Cellpose viz: nucleus flow (z-avg)" in names
    finally:
        viewer.close()
        app.processEvents()


def test_add_cellpose_viz_layers_replaces_existing_layers(mod, tmp_path):
    T, Z, Y, X = 2, 3, 16, 16
    prob = np.random.randn(T, Z, Y, X).astype(np.float32)
    dp = np.random.randn(T, Z, 2, Y, X).astype(np.float32)
    _write_prob(tmp_path, "nucleus", prob)
    _write_dp(tmp_path, "nucleus", dp)

    app, viewer = _make_viewer()
    try:
        mod.add_cellpose_viz_layers(
            viewer, tmp_path, "nucleus", "zavg", stride=4, scale=1.0
        )
        assert len(viewer.layers) == 2

        mod.add_cellpose_viz_layers(
            viewer, tmp_path, "nucleus", "zavg", stride=8, scale=2.0
        )
        # Should still be exactly 2 layers (old ones removed, new ones added)
        assert len(viewer.layers) == 2
    finally:
        viewer.close()
        app.processEvents()


def test_add_cellpose_viz_layers_missing_files_returns_empty(mod, tmp_path):
    # Neither prob nor dp files present
    app, viewer = _make_viewer()
    try:
        layers = mod.add_cellpose_viz_layers(
            viewer, tmp_path, "nucleus", "zavg", stride=4, scale=1.0
        )
        assert layers == []
        assert len(viewer.layers) == 0
    finally:
        viewer.close()
        app.processEvents()


def test_add_cellpose_viz_layers_3dt_mode(mod, tmp_path):
    T, Z, Y, X = 2, 3, 16, 16
    prob = np.random.randn(T, Z, Y, X).astype(np.float32)
    dp = np.random.randn(T, Z, 2, Y, X).astype(np.float32)
    _write_prob(tmp_path, "nucleus", prob)
    _write_dp(tmp_path, "nucleus", dp)

    app, viewer = _make_viewer()
    try:
        layers = mod.add_cellpose_viz_layers(
            viewer, tmp_path, "nucleus", "3dt", stride=4, scale=1.0
        )
        assert len(layers) == 2
        names = [la.name for la in viewer.layers]
        assert "Cellpose viz: nucleus prob (3D+t)" in names
        assert "Cellpose viz: nucleus flow (3D+t)" in names
    finally:
        viewer.close()
        app.processEvents()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/aruppel/Projects/CellFlow && uv run pytest tests/napari/test_cellpose_visualization.py -k "add_cellpose" -x 2>&1 | tail -20
```
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `add_cellpose_viz_layers`**

Add to `src/cellflow/napari/cellpose_visualization.py`:

```python
def _mode_label(mode: str) -> str:
    return "z-avg" if mode == "zavg" else "3D+t"


def _layer_names(channel: str, mode: str) -> tuple[str, str]:
    label = _mode_label(mode)
    return (
        f"Cellpose viz: {channel} prob ({label})",
        f"Cellpose viz: {channel} flow ({label})",
    )


def add_cellpose_viz_layers(
    viewer: Any,
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
    mode: Literal["zavg", "3dt"],
    *,
    stride: int,
    scale: float,
) -> list[Any]:
    from typing import Any as _Any  # local to avoid top-level Qt import guard issues

    output_dir = Path(output_dir)
    prob_path = output_dir / _prob_filename(channel)
    dp_path = output_dir / _dp_filename(channel)

    prob_name, flow_name = _layer_names(channel, mode)

    # Remove pre-existing layers for this (channel, mode)
    for name in (prob_name, flow_name):
        to_remove = [la for la in list(viewer.layers) if la.name == name]
        for la in to_remove:
            viewer.layers.remove(la)

    if not prob_path.is_file() or not dp_path.is_file():
        return []

    prob_data = load_sigmoid_prob(output_dir, channel, mode)
    flow_data = load_flow_vectors(output_dir, channel, mode, stride=stride, scale=scale)

    prob_layer = viewer.add_image(
        prob_data,
        name=prob_name,
        colormap="magma",
        contrast_limits=(0.0, 1.0),
        blending="translucent",
    )
    flow_layer = viewer.add_vectors(
        flow_data,
        name=flow_name,
        edge_width=1,
        length=1,
        vector_style="arrow",
        edge_color="cyan",
    )
    return [prob_layer, flow_layer]
```

Also add `from typing import Any` at the top of the file if not already present.

The full top-of-file imports should be:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile
```

- [ ] **Step 4: Run all cellpose_visualization tests**

```bash
cd /home/aruppel/Projects/CellFlow && uv run pytest tests/napari/test_cellpose_visualization.py -x 2>&1 | tail -20
```
Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/aruppel/Projects/CellFlow && git add src/cellflow/napari/cellpose_visualization.py tests/napari/test_cellpose_visualization.py && git commit -m "feat: add add_cellpose_viz_layers orchestration function"
```

---

## Task 4: Widget UI — add visualization QGroupBoxes

**Files:**
- Modify: `src/cellflow/napari/hpc_cellpose_widget.py`

- [ ] **Step 1: Add the imports at the top of the widget file**

The file already imports `QDoubleSpinBox`, `QHBoxLayout`, `QLabel`, `QSpinBox`, `QVBoxLayout`, `QWidget` from qtpy. Add `QGroupBox` to that import block:

```python
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
```

Also add the import of the new helper at the bottom of the imports section:

```python
from cellflow.napari.cellpose_visualization import add_cellpose_viz_layers
```

- [ ] **Step 2: Add the viz group boxes in `__init__` after `layout.addWidget(self.status_lbl)`**

After line `layout.addWidget(self.status_lbl)` (currently line 122), add:

```python
        # --- Nuclei visualization ---
        nuclei_viz_box = QGroupBox("Nuclei visualization")
        nuclei_viz_layout = QVBoxLayout(nuclei_viz_box)
        nuclei_viz_layout.setContentsMargins(4, 4, 4, 4)
        nuclei_viz_layout.setSpacing(4)

        nuclei_params_row = QWidget()
        nuclei_params_layout = QHBoxLayout(nuclei_params_row)
        nuclei_params_layout.setContentsMargins(0, 0, 0, 0)
        nuclei_params_layout.setSpacing(8)
        nuclei_params_layout.addWidget(QLabel("Stride:"))
        self.nuclei_viz_stride_spin = self._int_spin(1, 256, 16)
        nuclei_params_layout.addWidget(self.nuclei_viz_stride_spin)
        nuclei_params_layout.addWidget(QLabel("Vector scale:"))
        self.nuclei_viz_scale_spin = self._double_spin(0.01, 100.0, 1.0, 2)
        nuclei_params_layout.addWidget(self.nuclei_viz_scale_spin)
        nuclei_params_layout.addStretch()
        nuclei_viz_layout.addWidget(nuclei_params_row)

        nuclei_btn_row = QWidget()
        nuclei_btn_layout = QHBoxLayout(nuclei_btn_row)
        nuclei_btn_layout.setContentsMargins(0, 0, 0, 0)
        nuclei_btn_layout.setSpacing(4)
        self.nuclei_viz_zavg_btn = QPushButton("Load z-avg")
        action_button(self.nuclei_viz_zavg_btn, expand=True)
        self.nuclei_viz_3dt_btn = QPushButton("Load 3D+t")
        action_button(self.nuclei_viz_3dt_btn, expand=True)
        nuclei_btn_layout.addWidget(self.nuclei_viz_zavg_btn)
        nuclei_btn_layout.addWidget(self.nuclei_viz_3dt_btn)
        nuclei_viz_layout.addWidget(nuclei_btn_row)

        self.nuclei_viz_status_lbl = QLabel("")
        status_label(self.nuclei_viz_status_lbl, muted=True)
        nuclei_viz_layout.addWidget(self.nuclei_viz_status_lbl)

        layout.addWidget(nuclei_viz_box)

        # --- Cells visualization ---
        cells_viz_box = QGroupBox("Cells visualization")
        cells_viz_layout = QVBoxLayout(cells_viz_box)
        cells_viz_layout.setContentsMargins(4, 4, 4, 4)
        cells_viz_layout.setSpacing(4)

        cells_params_row = QWidget()
        cells_params_layout = QHBoxLayout(cells_params_row)
        cells_params_layout.setContentsMargins(0, 0, 0, 0)
        cells_params_layout.setSpacing(8)
        cells_params_layout.addWidget(QLabel("Stride:"))
        self.cells_viz_stride_spin = self._int_spin(1, 256, 16)
        cells_params_layout.addWidget(self.cells_viz_stride_spin)
        cells_params_layout.addWidget(QLabel("Vector scale:"))
        self.cells_viz_scale_spin = self._double_spin(0.01, 100.0, 1.0, 2)
        cells_params_layout.addWidget(self.cells_viz_scale_spin)
        cells_params_layout.addStretch()
        cells_viz_layout.addWidget(cells_params_row)

        cells_btn_row = QWidget()
        cells_btn_layout = QHBoxLayout(cells_btn_row)
        cells_btn_layout.setContentsMargins(0, 0, 0, 0)
        cells_btn_layout.setSpacing(4)
        self.cells_viz_zavg_btn = QPushButton("Load z-avg")
        action_button(self.cells_viz_zavg_btn, expand=True)
        self.cells_viz_3dt_btn = QPushButton("Load 3D+t")
        action_button(self.cells_viz_3dt_btn, expand=True)
        cells_btn_layout.addWidget(self.cells_viz_zavg_btn)
        cells_btn_layout.addWidget(self.cells_viz_3dt_btn)
        cells_viz_layout.addWidget(cells_btn_row)

        self.cells_viz_status_lbl = QLabel("")
        status_label(self.cells_viz_status_lbl, muted=True)
        cells_viz_layout.addWidget(self.cells_viz_status_lbl)

        layout.addWidget(cells_viz_box)
```

- [ ] **Step 3: Wire button signals and add `_update_viz_status` + click handlers in the `__init__` after the existing signal connections**

After `self._update_input_status()` at the bottom of `__init__`, add:

```python
        self.nuclei_viz_zavg_btn.clicked.connect(
            lambda: self._on_load_viz("nucleus", "zavg")
        )
        self.nuclei_viz_3dt_btn.clicked.connect(
            lambda: self._on_load_viz("nucleus", "3dt")
        )
        self.cells_viz_zavg_btn.clicked.connect(
            lambda: self._on_load_viz("cell", "zavg")
        )
        self.cells_viz_3dt_btn.clicked.connect(
            lambda: self._on_load_viz("cell", "3dt")
        )

        self.output_dir_edit.textChanged.connect(self._update_viz_status)

        self._update_viz_status()
```

- [ ] **Step 4: Add helper methods `_update_viz_status` and `_on_load_viz`**

Add these methods to the class (after `_update_input_status`):

```python
    def _update_viz_status(self) -> None:
        output_dir_text = self.output_dir_edit.text().strip()
        if not output_dir_text:
            for btn in (
                self.nuclei_viz_zavg_btn,
                self.nuclei_viz_3dt_btn,
                self.cells_viz_zavg_btn,
                self.cells_viz_3dt_btn,
            ):
                btn.setEnabled(False)
            self.nuclei_viz_status_lbl.setText("no output directory selected")
            self.cells_viz_status_lbl.setText("no output directory selected")
            return

        output_dir = Path(output_dir_text)
        for channel, zavg_btn, dt_btn, status_lbl in (
            (
                "nucleus",
                self.nuclei_viz_zavg_btn,
                self.nuclei_viz_3dt_btn,
                self.nuclei_viz_status_lbl,
            ),
            (
                "cell",
                self.cells_viz_zavg_btn,
                self.cells_viz_3dt_btn,
                self.cells_viz_status_lbl,
            ),
        ):
            prob_ok = (output_dir / f"{channel}_prob_3dt.tif").is_file()
            dp_ok = (output_dir / f"{channel}_dp_3dt.tif").is_file()
            enabled = prob_ok and dp_ok
            zavg_btn.setEnabled(enabled)
            dt_btn.setEnabled(enabled)
            if not enabled:
                missing = []
                if not prob_ok:
                    missing.append(f"{channel}_prob_3dt.tif")
                if not dp_ok:
                    missing.append(f"{channel}_dp_3dt.tif")
                status_lbl.setText(f"missing: {', '.join(missing)}")
            else:
                status_lbl.setText("")

    def _on_load_viz(self, channel: str, mode: str) -> None:
        output_dir_text = self.output_dir_edit.text().strip()
        if not output_dir_text:
            return
        output_dir = Path(output_dir_text)
        status_lbl = (
            self.nuclei_viz_status_lbl
            if channel == "nucleus"
            else self.cells_viz_status_lbl
        )
        stride = (
            self.nuclei_viz_stride_spin.value()
            if channel == "nucleus"
            else self.cells_viz_stride_spin.value()
        )
        scale = (
            self.nuclei_viz_scale_spin.value()
            if channel == "nucleus"
            else self.cells_viz_scale_spin.value()
        )
        try:
            layers = add_cellpose_viz_layers(
                self.viewer,
                output_dir,
                channel,
                mode,
                stride=stride,
                scale=scale,
            )
        except Exception as exc:
            status_lbl.setText(f"error: {exc}")
            return

        if not layers:
            prob_name = f"{channel}_prob_3dt.tif"
            dp_name = f"{channel}_dp_3dt.tif"
            missing = [
                n
                for n in (prob_name, dp_name)
                if not (output_dir / n).is_file()
            ]
            status_lbl.setText(f"missing: {', '.join(missing)}")
            return

        mode_label = "z-avg" if mode == "zavg" else "3D+t"
        # Count vectors: last layer is flow
        n_vectors = len(layers[-1].data) if hasattr(layers[-1], "data") else "?"
        status_lbl.setText(
            f"loaded {channel} {mode_label} ({n_vectors} vectors)"
        )
```

- [ ] **Step 5: Extend `get_state` and `set_state`**

In `get_state`, add the four new keys to the returned dict (after `"remote_host"`):

```python
            "nuclei_viz_stride": self.nuclei_viz_stride_spin.value(),
            "nuclei_viz_scale": self.nuclei_viz_scale_spin.value(),
            "cells_viz_stride": self.cells_viz_stride_spin.value(),
            "cells_viz_scale": self.cells_viz_scale_spin.value(),
```

In `set_state`, add the corresponding `if` blocks (after the `"remote_host"` block):

```python
        if "nuclei_viz_stride" in state:
            self.nuclei_viz_stride_spin.setValue(int(state["nuclei_viz_stride"]))
        if "nuclei_viz_scale" in state:
            self.nuclei_viz_scale_spin.setValue(float(state["nuclei_viz_scale"]))
        if "cells_viz_stride" in state:
            self.cells_viz_stride_spin.setValue(int(state["cells_viz_stride"]))
        if "cells_viz_scale" in state:
            self.cells_viz_scale_spin.setValue(float(state["cells_viz_scale"]))
```

- [ ] **Step 6: Verify the widget still imports cleanly**

```bash
cd /home/aruppel/Projects/CellFlow && uv run python -c "
import os; os.environ['QT_QPA_PLATFORM']='offscreen'
from qtpy.QtWidgets import QApplication
app = QApplication([])
from cellflow.napari.hpc_cellpose_widget import HpcCellposeWidget
print('import OK')
"
```
Expected: `import OK`

- [ ] **Step 7: Commit**

```bash
cd /home/aruppel/Projects/CellFlow && git add src/cellflow/napari/hpc_cellpose_widget.py && git commit -m "feat: add nuclei/cells viz QGroupBoxes to HpcCellposeWidget"
```

---

## Task 5: Widget tests — button enable/disable and state round-trip

**Files:**
- Modify: `tests/napari/test_hpc_cellpose_widget.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/napari/test_hpc_cellpose_widget.py`:

```python
def test_viz_buttons_disabled_when_output_dir_empty(monkeypatch):
    app, _mod, widget = _make_widget(monkeypatch)

    # ensure output_dir is empty (it is by default)
    widget.output_dir_edit.clear()
    widget._update_viz_status()

    assert not widget.nuclei_viz_zavg_btn.isEnabled()
    assert not widget.nuclei_viz_3dt_btn.isEnabled()
    assert not widget.cells_viz_zavg_btn.isEnabled()
    assert not widget.cells_viz_3dt_btn.isEnabled()

    widget.deleteLater()
    app.processEvents()


def test_viz_buttons_disabled_when_tifs_missing(monkeypatch, tmp_path):
    app, _mod, widget = _make_widget(monkeypatch)

    output_dir = tmp_path / "1_cellpose"
    output_dir.mkdir()
    # No tif files present
    widget.output_dir_edit.setText(str(output_dir))
    widget._update_viz_status()

    assert not widget.nuclei_viz_zavg_btn.isEnabled()
    assert not widget.nuclei_viz_3dt_btn.isEnabled()
    assert not widget.cells_viz_zavg_btn.isEnabled()
    assert not widget.cells_viz_3dt_btn.isEnabled()

    widget.deleteLater()
    app.processEvents()


def test_viz_buttons_enabled_when_tifs_present(monkeypatch, tmp_path):
    app, _mod, widget = _make_widget(monkeypatch)

    output_dir = tmp_path / "1_cellpose"
    output_dir.mkdir()
    (output_dir / "nucleus_prob_3dt.tif").write_bytes(b"x")
    (output_dir / "nucleus_dp_3dt.tif").write_bytes(b"x")
    (output_dir / "cell_prob_3dt.tif").write_bytes(b"x")
    (output_dir / "cell_dp_3dt.tif").write_bytes(b"x")

    widget.output_dir_edit.setText(str(output_dir))
    widget._update_viz_status()

    assert widget.nuclei_viz_zavg_btn.isEnabled()
    assert widget.nuclei_viz_3dt_btn.isEnabled()
    assert widget.cells_viz_zavg_btn.isEnabled()
    assert widget.cells_viz_3dt_btn.isEnabled()

    widget.deleteLater()
    app.processEvents()


def test_get_set_state_includes_viz_params(monkeypatch, tmp_path):
    app, _mod, widget = _make_widget(monkeypatch)

    state = {
        "input_dir": "",
        "output_dir": "",
        "config_path": str(DEFAULT_CONFIG),
        "nuclei_input": "nucleus_3dt.tif",
        "cells_input": "cell_3dt.tif",
        "frames": "all",
        "nuclei_do_3d": False,
        "nuclei_anisotropy": 1.5,
        "nuclei_diameter": 25,
        "nuclei_size": 0,
        "nuclei_gamma": 1.0,
        "cells_size": 0,
        "cells_gamma": 1.0,
        "max_concurrent_jobs": 4,
        "remote_user": "aruppel",
        "remote_host": "maestro.pasteur.fr",
        "nuclei_viz_stride": 8,
        "nuclei_viz_scale": 2.5,
        "cells_viz_stride": 32,
        "cells_viz_scale": 0.5,
    }

    widget.set_state(state)
    result = widget.get_state()

    assert result["nuclei_viz_stride"] == 8
    assert result["nuclei_viz_scale"] == 2.5
    assert result["cells_viz_stride"] == 32
    assert result["cells_viz_scale"] == 0.5

    widget.deleteLater()
    app.processEvents()
```

- [ ] **Step 2: Run the new widget tests to verify they fail**

```bash
cd /home/aruppel/Projects/CellFlow && uv run pytest tests/napari/test_hpc_cellpose_widget.py -k "viz" -x 2>&1 | tail -20
```
Expected: FAIL (attribute errors since the widget changes aren't in yet — but Task 4 should already be done at this point, so tests should pass or point to specific issues).

- [ ] **Step 3: Run all widget tests**

```bash
cd /home/aruppel/Projects/CellFlow && uv run pytest tests/napari/test_hpc_cellpose_widget.py -x 2>&1 | tail -20
```
Expected: all PASSED (Tasks 1-4 complete before this).

- [ ] **Step 4: Commit**

```bash
cd /home/aruppel/Projects/CellFlow && git add tests/napari/test_hpc_cellpose_widget.py && git commit -m "test: add viz button enable/disable and state round-trip tests for HpcCellposeWidget"
```

---

## Task 6: Full regression run

- [ ] **Step 1: Run entire napari test suite**

```bash
cd /home/aruppel/Projects/CellFlow && uv run pytest tests/napari/ -x 2>&1 | tail -30
```
Expected: all PASSED, no regressions.

- [ ] **Step 2: If any test fails, fix and re-run**

Common issues to watch for:
- `get_state` test in existing `test_get_set_state_round_trips_hpc_cellpose_controls` now fails because `get_state` returns more keys than the test's `state` dict. Fix: the test uses `assert widget.get_state() == state` — you need to either update that test's `state` dict to include the four new viz keys with their defaults, OR change the assertion to check a subset. The cleanest fix is to add the four new keys to the existing test's `state` dict with their defaults (`nuclei_viz_stride: 16`, `nuclei_viz_scale: 1.0`, `cells_viz_stride: 16`, `cells_viz_scale: 1.0`).

- [ ] **Step 3: Commit any fixes**

```bash
cd /home/aruppel/Projects/CellFlow && git add -p && git commit -m "fix: update existing state round-trip test with new viz params"
```

---

## Self-Review Checklist

**Spec coverage:**

| Spec requirement | Covered by |
|---|---|
| `load_sigmoid_prob` z-avg returns `(T, Y, X)` in [0,1] | Task 1, test `test_load_sigmoid_prob_zavg_shape` |
| `load_sigmoid_prob` 3D+t returns `(T, Z, Y, X)` in [0,1] | Task 1, test `test_load_sigmoid_prob_3dt_shape` |
| Sigmoid-then-mean regression test | Task 1, test `test_load_sigmoid_prob_zavg_sigmoid_then_mean` |
| `load_flow_vectors` zavg shape `(N, 2, 3)` with correct N | Task 2, tests `test_load_flow_vectors_zavg_shape`, `test_load_flow_vectors_zavg_respects_stride` |
| `load_flow_vectors` 3D+t shape `(N, 2, 4)` with dz=0 | Task 2, test `test_load_flow_vectors_3dt_shape_and_dz_zero` |
| Scale applied to vectors | Task 2, test `test_load_flow_vectors_scale_applied` |
| `add_cellpose_viz_layers` removes prior layers before adding new | Task 3, test `test_add_cellpose_viz_layers_replaces_existing_layers` |
| `add_cellpose_viz_layers` missing files → empty return | Task 3, test `test_add_cellpose_viz_layers_missing_files_returns_empty` |
| 3D+t mode layer names | Task 3, test `test_add_cellpose_viz_layers_3dt_mode` |
| QGroupBox nuclei/cells viz with stride/scale/buttons | Task 4 |
| Buttons disabled when output_dir empty | Task 5, `test_viz_buttons_disabled_when_output_dir_empty` |
| Buttons disabled when tifs missing | Task 5, `test_viz_buttons_disabled_when_tifs_missing` |
| Buttons enabled when tifs present | Task 5, `test_viz_buttons_enabled_when_tifs_present` |
| `get_state`/`set_state` includes viz params | Task 5, `test_get_set_state_includes_viz_params` |
| Layer naming: `Cellpose viz: <channel> prob (<mode>)` | Task 3, `_layer_names` function, verified in test |
| Prob layer: colormap=magma, contrast_limits=(0,1), blending=translucent | Task 3, `add_cellpose_viz_layers` impl |
| Vectors layer: edge_width=1, length=1, vector_style=arrow, edge_color=cyan | Task 3, `add_cellpose_viz_layers` impl |

**No placeholders detected** — all steps have actual code.

**Type consistency:** `load_sigmoid_prob`, `load_flow_vectors`, `add_cellpose_viz_layers` signatures are consistent across all tasks. `_layer_names` returns the canonical names used in tests and button wiring. `channel` parameter is `"nucleus"` / `"cell"` throughout (matching the spec's Literal type). Note: `_on_load_viz` uses `channel == "nucleus"` guard to select spin boxes — matches widget attribute names `nuclei_viz_stride_spin` / `cells_viz_stride_spin`.
