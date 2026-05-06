# Flow-Following Cell Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cell-segmentation Contour Maps + 3D Temporal Watershed pipeline with a per-frame flow-following segmentation that advects each foreground pixel along the (z-averaged, T-Y-X-filtered) Cellpose flow field with an EDT-direction gravity blend toward tracked nuclei, and Voronoi-fills the leftovers.

**Architecture:** New backend module `src/cellflow/segmentation/flow_following.py` containing `FlowFollowingParams`, a Numba `_flow_integrate` kernel ported from `archive/v1/.../gravity_flow.py`, an EDT-direction-only gravity computation, and a per-frame orchestration `compute_flow_following_movie` that does median+Gaussian filtering on the (T, 2, Y, X) flow stack before integration. The cell-workflow napari widget loses two collapsible sections ("Contour Maps", "3D Temporal Watershed") and gains a single "Flow-Following Segmentation" section that reads `cell_prob_3dt.tif`, `cell_dp_3dt.tif`, `foreground_masks.tif`, and `2_nucleus/tracked_labels.tif`, and writes `3_cell/filtered_flow_mag.tif` and `3_cell/tracked_labels.tif`. Cleanup is conservative — only cell-pipeline-exclusive symbols are deleted; nucleus-widget callers and the deprecated h5 hypothesis pipeline are left for a later cleanup PR.

**Tech Stack:** Python, NumPy, SciPy (`distance_transform_edt`, `median_filter`, `gaussian_filter`), Numba (`@njit(parallel=True, cache=True)`), tifffile, napari `@thread_worker`, Qt/qtpy.

---

## File Structure

- Modify `pyproject.toml`: add `numba` dependency.
- Create `src/cellflow/segmentation/flow_following.py`: `FlowFollowingParams`, `_flow_integrate` (Numba kernel), `_fill_foreground` (Voronoi helper), `compute_flow_following_movie` (per-frame orchestration with pre-integration filtering).
- Modify `src/cellflow/segmentation/__init__.py`: export `FlowFollowingParams` and `compute_flow_following_movie`; drop the `compute_3d_temporal_watershed` re-export and the `build_mean_z_consensus_boundary` definition (cell-only).
- Modify `src/cellflow/napari/cell_workflow_widget.py`: delete the "Contour Maps" and "3D Temporal Watershed" sections wholesale and replace with a single "Flow-Following Segmentation" section + `@thread_worker` Run/Cancel.
- Create `tests/segmentation/test_flow_following.py`: backend unit tests (params defaults, voronoi fill, kernel capture, orchestration).
- Delete `tests/napari/test_cell_workflow_preview.py`; create `tests/napari/test_cell_workflow_widget.py` covering the new section.
- Modify `tests/segmentation/test_label_postprocessing.py`: drop the three `build_mean_z_consensus_boundary` tests; keep `test_fill_and_close_labels_fills_per_label_bounding_boxes`.
- Delete `src/cellflow/segmentation/watershed_3d.py`.

**Conservative cleanup boundary:** `build_consensus_boundary`, `compute_cellpose_flow_hypothesis`, `compute_seeded_watershed`, `SeededWatershedParams`, and `compute_masks_for_threshold` remain in `segmentation/__init__.py`. They are still consumed by the live nucleus widget (`build_consensus_boundary`) and by the deprecated `database/hypotheses.py` h5 pipeline (the others), and a follow-up PR will retire them along with the rest of the h5 layer.

---

### Task 1: Add Numba Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `numba` to dependencies**

In `pyproject.toml`, locate the `[project]` `dependencies` list (lines 12–20) and append `"numba"` to it. After editing the list reads:

```toml
dependencies = [
    "napari[all]>=0.4.18",
    "qtpy>=2.3.0",
    "h5py",
    "numpy",
    "scipy",
    "pandas",
    "tifffile",
    "numba",
]
```

- [ ] **Step 2: Verify numba imports**

Run: `python -c "import numba; print(numba.__version__)"`

Expected: prints a version string (e.g. `0.65.0`) with no error.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add numba dependency for flow-following kernel"
```

---

### Task 2: FlowFollowingParams And _fill_foreground

**Files:**
- Create: `src/cellflow/segmentation/flow_following.py`
- Test: `tests/segmentation/test_flow_following.py`

- [ ] **Step 1: Write the failing dataclass + Voronoi tests**

Create `tests/segmentation/test_flow_following.py`:

```python
"""Tests for the flow-following cell segmentation backend."""
from __future__ import annotations

import numpy as np
import pytest

from cellflow.segmentation.flow_following import (
    FlowFollowingParams,
    _fill_foreground,
)


def test_flow_following_params_defaults_match_spec():
    p = FlowFollowingParams()
    assert p.median_kernel_time == 3
    assert p.median_kernel_space == 5
    assert p.gaussian_sigma_time == 0.0
    assert p.gaussian_sigma_space == 0.0
    assert p.flow_weight == 0.5
    assert p.flow_step_scale == 0.2
    assert p.max_iterations == 100
    assert p.capture_radius == 3.0


def test_fill_foreground_voronoi_assigns_unlabelled_foreground():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1, 1] = 7
    labels[4, 4] = 9
    fg = np.ones((6, 6), dtype=bool)

    out = _fill_foreground(labels, fg)

    # Closer to seed 7 (top-left) ⇒ label 7
    assert out[0, 0] == 7
    assert out[2, 2] == 7
    # Closer to seed 9 (bottom-right) ⇒ label 9
    assert out[5, 5] == 9
    assert out[3, 3] == 9
    # Original seeds preserved
    assert out[1, 1] == 7
    assert out[4, 4] == 9


def test_fill_foreground_skips_when_no_unlabelled_foreground():
    labels = np.array([[1, 1], [2, 2]], dtype=np.int32)
    fg = np.ones((2, 2), dtype=bool)

    out = _fill_foreground(labels, fg)
    np.testing.assert_array_equal(out, labels)


def test_fill_foreground_leaves_background_zero():
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0, 0] = 5
    fg = np.zeros((4, 4), dtype=bool)
    fg[0, 0] = True
    fg[0, 1] = True

    out = _fill_foreground(labels, fg)
    assert out[0, 0] == 5
    assert out[0, 1] == 5
    # Outside foreground stays zero
    assert out[2, 2] == 0
    assert out[3, 3] == 0
```

- [ ] **Step 2: Run the failing tests**

Run: `pytest tests/segmentation/test_flow_following.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'cellflow.segmentation.flow_following'`.

- [ ] **Step 3: Create the module skeleton**

Create `src/cellflow/segmentation/flow_following.py`:

```python
"""Flow-following cell segmentation: per-frame Euler integration of the
Cellpose flow field with an EDT-direction gravity blend toward tracked nuclei,
plus Voronoi fill for unconverged foreground pixels."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    median_filter,
)


@dataclass(frozen=True, slots=True)
class FlowFollowingParams:
    """Parameters for `compute_flow_following_movie`."""

    median_kernel_time: int = 3
    median_kernel_space: int = 5
    gaussian_sigma_time: float = 0.0
    gaussian_sigma_space: float = 0.0
    flow_weight: float = 0.5
    flow_step_scale: float = 0.2
    max_iterations: int = 100
    capture_radius: float = 3.0


def _fill_foreground(labels: np.ndarray, prob_mask: np.ndarray) -> np.ndarray:
    """Voronoi-fill any foreground pixels that the integrator did not assign."""
    missing = prob_mask & (labels == 0)
    if not missing.any():
        return labels
    _, (iy, ix) = distance_transform_edt(labels == 0, return_indices=True)
    out = labels.copy()
    out[missing] = labels[iy[missing], ix[missing]]
    return out
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/segmentation/test_flow_following.py -q`

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/segmentation/flow_following.py tests/segmentation/test_flow_following.py
git commit -m "feat(segmentation): add FlowFollowingParams and Voronoi fill helper"
```

---

### Task 3: Numba `_flow_integrate` Kernel

**Files:**
- Modify: `src/cellflow/segmentation/flow_following.py`
- Modify: `tests/segmentation/test_flow_following.py`

- [ ] **Step 1: Write the failing kernel test**

Append to `tests/segmentation/test_flow_following.py`:

```python
def test_flow_integrate_captures_foreground_pixel_into_seed_label():
    from cellflow.segmentation.flow_following import _flow_integrate

    H = W = 16
    nuclear_labels = np.zeros((H, W), dtype=np.int32)
    nuclear_labels[8, 8] = 5  # single seed at the centre
    prob_mask = np.ones((H, W), dtype=bool)

    # Pure-gravity setup: zero flow, gravity points each pixel toward the seed.
    flow = np.zeros((H, W, 2), dtype=np.float32)
    yi, xi = np.indices((H, W))
    dy = (8 - yi).astype(np.float32)
    dx = (8 - xi).astype(np.float32)
    norm = np.hypot(dy, dx)
    safe = np.where(norm > 0, norm, 1.0)
    grav_y = (dy / safe).astype(np.float32)
    grav_x = (dx / safe).astype(np.float32)
    grav_y[8, 8] = 0.0
    grav_x[8, 8] = 0.0

    dist = distance_transform_edt(nuclear_labels == 0).astype(np.float32)
    _, (ny, nx) = distance_transform_edt(nuclear_labels == 0, return_indices=True)

    result = _flow_integrate(
        nuclear_labels,
        flow, grav_y, grav_x,
        dist,
        ny.astype(np.int32), nx.astype(np.int32),
        prob_mask,
        n_steps=50,
        flow_step_scale=1.0,
        flow_weight=0.0,            # pure gravity
        capture_radius=1.5,
    )

    # Every foreground pixel should arrive at the seed and inherit label 5.
    assert (result == 5).all()


def test_flow_integrate_skips_pixels_outside_prob_mask():
    from cellflow.segmentation.flow_following import _flow_integrate

    H = W = 8
    nuclear_labels = np.zeros((H, W), dtype=np.int32)
    nuclear_labels[4, 4] = 3
    prob_mask = np.zeros((H, W), dtype=bool)
    prob_mask[4, 4] = True  # only the seed itself is foreground

    flow = np.zeros((H, W, 2), dtype=np.float32)
    grav_y = np.zeros((H, W), dtype=np.float32)
    grav_x = np.zeros((H, W), dtype=np.float32)
    dist = distance_transform_edt(nuclear_labels == 0).astype(np.float32)
    _, (ny, nx) = distance_transform_edt(nuclear_labels == 0, return_indices=True)

    result = _flow_integrate(
        nuclear_labels,
        flow, grav_y, grav_x,
        dist,
        ny.astype(np.int32), nx.astype(np.int32),
        prob_mask,
        n_steps=10,
        flow_step_scale=0.2,
        flow_weight=0.5,
        capture_radius=1.5,
    )

    assert result[4, 4] == 3
    # No assignments outside foreground.
    background = np.ones_like(result, dtype=bool)
    background[4, 4] = False
    assert (result[background] == 0).all()


def test_flow_integrate_preserves_existing_labels():
    from cellflow.segmentation.flow_following import _flow_integrate

    H = W = 6
    nuclear_labels = np.zeros((H, W), dtype=np.int32)
    nuclear_labels[2, 2] = 11
    nuclear_labels[3, 3] = 22
    prob_mask = np.ones((H, W), dtype=bool)

    flow = np.zeros((H, W, 2), dtype=np.float32)
    grav_y = np.zeros((H, W), dtype=np.float32)
    grav_x = np.zeros((H, W), dtype=np.float32)
    dist = distance_transform_edt(nuclear_labels == 0).astype(np.float32)
    _, (ny, nx) = distance_transform_edt(nuclear_labels == 0, return_indices=True)

    result = _flow_integrate(
        nuclear_labels,
        flow, grav_y, grav_x,
        dist,
        ny.astype(np.int32), nx.astype(np.int32),
        prob_mask,
        n_steps=5,
        flow_step_scale=0.2,
        flow_weight=0.5,
        capture_radius=1.5,
    )

    assert result[2, 2] == 11
    assert result[3, 3] == 22
```

The first test currently fails with `ImportError`; it will pass once the kernel is implemented.

- [ ] **Step 2: Run the failing tests**

Run: `pytest tests/segmentation/test_flow_following.py -q`

Expected: FAIL with `ImportError: cannot import name '_flow_integrate'`.

- [ ] **Step 3: Add the Numba kernel**

In `src/cellflow/segmentation/flow_following.py`, add `import numba` near the top imports (between `numpy` and `scipy.ndimage`):

```python
import numba
```

Then append the kernel definition below `_fill_foreground`:

```python
@numba.njit(parallel=True, cache=True)
def _flow_integrate(
    nuclear_labels: np.ndarray,    # (H, W) int32
    flow: np.ndarray,              # (H, W, 2) float32 — channel 0 = dY, channel 1 = dX
    grav_y: np.ndarray,            # (H, W) float32 — EDT-direction unit vector y
    grav_x: np.ndarray,            # (H, W) float32 — EDT-direction unit vector x
    dist_to_nucleus: np.ndarray,   # (H, W) float32 — EDT distance to nearest nuclear pixel
    nearest_y: np.ndarray,         # (H, W) int32 — y-index of nearest nuclear pixel
    nearest_x: np.ndarray,         # (H, W) int32
    prob_mask: np.ndarray,         # (H, W) bool — foreground mask
    n_steps: int,
    flow_step_scale: float,
    flow_weight: float,
    capture_radius: float,
) -> np.ndarray:
    H, W = nuclear_labels.shape
    result = nuclear_labels.copy()

    for i in numba.prange(H):
        for j in range(W):
            if result[i, j] > 0:
                continue
            if not prob_mask[i, j]:
                continue

            py = float(i)
            px = float(j)
            label = 0

            for _ in range(n_steps):
                iy0 = int(py)
                ix0 = int(px)
                iy0 = max(0, min(H - 2, iy0))
                ix0 = max(0, min(W - 2, ix0))

                fy = py - float(iy0)
                fx = px - float(ix0)

                flow_y = (flow[iy0,     ix0,     0] * (1.0 - fy) * (1.0 - fx) +
                          flow[iy0 + 1, ix0,     0] * fy          * (1.0 - fx) +
                          flow[iy0,     ix0 + 1, 0] * (1.0 - fy) * fx          +
                          flow[iy0 + 1, ix0 + 1, 0] * fy          * fx)

                flow_x = (flow[iy0,     ix0,     1] * (1.0 - fy) * (1.0 - fx) +
                          flow[iy0 + 1, ix0,     1] * fy          * (1.0 - fx) +
                          flow[iy0,     ix0 + 1, 1] * (1.0 - fy) * fx          +
                          flow[iy0 + 1, ix0 + 1, 1] * fy          * fx)

                w = flow_weight

                iy_nn = max(0, min(H - 1, int(py + 0.5)))
                ix_nn = max(0, min(W - 1, int(px + 0.5)))

                step_y = w * flow_y + (1.0 - w) * grav_y[iy_nn, ix_nn]
                step_x = w * flow_x + (1.0 - w) * grav_x[iy_nn, ix_nn]

                py = max(0.0, min(float(H - 1), py + step_y * flow_step_scale))
                px = max(0.0, min(float(W - 1), px + step_x * flow_step_scale))

                iy = max(0, min(H - 1, int(py + 0.5)))
                ix = max(0, min(W - 1, int(px + 0.5)))

                if dist_to_nucleus[iy, ix] <= capture_radius:
                    L = nuclear_labels[nearest_y[iy, ix], nearest_x[iy, ix]]
                    if L > 0:
                        label = L
                        break

            result[i, j] = label

    return result
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/segmentation/test_flow_following.py -q`

Expected: PASS (7 tests). Numba JIT will compile on first call — the run may take a few extra seconds.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/segmentation/flow_following.py tests/segmentation/test_flow_following.py
git commit -m "feat(segmentation): add _flow_integrate Numba kernel"
```

---

### Task 4: Per-Frame Orchestration `compute_flow_following_movie`

**Files:**
- Modify: `src/cellflow/segmentation/flow_following.py`
- Modify: `tests/segmentation/test_flow_following.py`

- [ ] **Step 1: Write the failing orchestration tests**

Append to `tests/segmentation/test_flow_following.py`:

```python
def _make_inward_flow(H: int, W: int, cy: float, cx: float) -> np.ndarray:
    """Flow vectors pointing each pixel toward (cy, cx), unit magnitude."""
    yi, xi = np.indices((H, W))
    dy = (cy - yi).astype(np.float32)
    dx = (cx - xi).astype(np.float32)
    norm = np.hypot(dy, dx)
    safe = np.where(norm > 0, norm, 1.0)
    return np.stack([dy / safe, dx / safe], axis=0).astype(np.float32)


def test_compute_flow_following_movie_assigns_foreground_pixels_to_nucleus():
    from cellflow.segmentation.flow_following import compute_flow_following_movie

    T, H, W = 2, 24, 24
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[0, 12, 12] = 7
    labels[1, 12, 12] = 7

    flow_t0 = _make_inward_flow(H, W, 12.0, 12.0)
    flow_t1 = _make_inward_flow(H, W, 12.0, 12.0)
    dp = np.stack([flow_t0, flow_t1], axis=0).astype(np.float32)

    params = FlowFollowingParams(
        median_kernel_time=1,
        median_kernel_space=1,
        gaussian_sigma_time=0.0,
        gaussian_sigma_space=0.0,
        flow_weight=0.5,
        flow_step_scale=0.5,
        max_iterations=100,
        capture_radius=3.0,
    )

    filtered_dp, cell_labels = compute_flow_following_movie(
        foreground, dp, labels, params
    )

    assert filtered_dp.shape == dp.shape
    assert filtered_dp.dtype == np.float32
    assert cell_labels.shape == (T, H, W)
    assert cell_labels.dtype == np.int32
    # Every foreground pixel collapses onto the single nucleus, so all are label 7.
    assert (cell_labels == 7).all()


def test_compute_flow_following_movie_voronoi_fills_zero_flow_foreground():
    from cellflow.segmentation.flow_following import compute_flow_following_movie

    T, H, W = 1, 12, 12
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[0, 2, 2] = 4
    labels[0, 9, 9] = 8

    # Zero flow → integrator never converges; Voronoi must fill.
    dp = np.zeros((T, 2, H, W), dtype=np.float32)

    params = FlowFollowingParams(
        median_kernel_time=1, median_kernel_space=1,
        gaussian_sigma_time=0.0, gaussian_sigma_space=0.0,
        flow_weight=1.0,        # ignore gravity → integrator will not move
        flow_step_scale=0.2,
        max_iterations=10,
        capture_radius=0.5,
    )

    _, cell_labels = compute_flow_following_movie(foreground, dp, labels, params)

    # Every foreground pixel must end up labelled, partitioned by EDT distance.
    assert (cell_labels[0] > 0).all()
    assert cell_labels[0, 0, 0] == 4   # closer to seed 4
    assert cell_labels[0, 11, 11] == 8 # closer to seed 8


def test_compute_flow_following_movie_returns_zeros_for_empty_foreground_frame():
    from cellflow.segmentation.flow_following import compute_flow_following_movie

    T, H, W = 2, 8, 8
    foreground = np.ones((T, H, W), dtype=bool)
    foreground[1] = False                          # second frame is empty
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[0, 4, 4] = 1
    dp = np.zeros((T, 2, H, W), dtype=np.float32)

    _, cell_labels = compute_flow_following_movie(
        foreground, dp, labels, FlowFollowingParams(
            median_kernel_time=1, median_kernel_space=1,
        ),
    )
    assert (cell_labels[1] == 0).all()


def test_compute_flow_following_movie_returns_zeros_for_no_nuclei_frame():
    from cellflow.segmentation.flow_following import compute_flow_following_movie

    T, H, W = 1, 8, 8
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)   # no nuclei in t=0
    dp = np.zeros((T, 2, H, W), dtype=np.float32)

    _, cell_labels = compute_flow_following_movie(
        foreground, dp, labels, FlowFollowingParams(
            median_kernel_time=1, median_kernel_space=1,
        ),
    )
    assert (cell_labels[0] == 0).all()


def test_compute_flow_following_movie_applies_median_and_gaussian_filters(monkeypatch):
    from cellflow.segmentation import flow_following as ff

    T, H, W = 2, 6, 6
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[0, 3, 3] = 1
    labels[1, 3, 3] = 1
    dp = np.ones((T, 2, H, W), dtype=np.float32)

    median_calls: list[tuple] = []
    gauss_calls: list[tuple] = []

    real_median = ff.median_filter
    real_gauss = ff.gaussian_filter

    def spy_median(arr, size):
        median_calls.append(tuple(size))
        return real_median(arr, size=size)

    def spy_gauss(arr, sigma):
        gauss_calls.append(tuple(sigma))
        return real_gauss(arr, sigma=sigma)

    monkeypatch.setattr(ff, "median_filter", spy_median)
    monkeypatch.setattr(ff, "gaussian_filter", spy_gauss)

    params = FlowFollowingParams(
        median_kernel_time=3,
        median_kernel_space=3,
        gaussian_sigma_time=1.0,
        gaussian_sigma_space=1.0,
    )
    ff.compute_flow_following_movie(foreground, dp, labels, params)

    # Channel axis is left at size 1 so the filter operates only on (T, Y, X).
    assert median_calls == [(1, 3, 3, 3)]
    assert gauss_calls == [(0, 1.0, 1.0, 1.0)]


def test_compute_flow_following_movie_skips_filter_when_kernels_off(monkeypatch):
    from cellflow.segmentation import flow_following as ff

    T, H, W = 1, 6, 6
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[0, 3, 3] = 1
    dp = np.zeros((T, 2, H, W), dtype=np.float32)

    median_calls: list[tuple] = []
    gauss_calls: list[tuple] = []
    monkeypatch.setattr(ff, "median_filter",
                        lambda arr, size: (median_calls.append(size), arr)[1])
    monkeypatch.setattr(ff, "gaussian_filter",
                        lambda arr, sigma: (gauss_calls.append(sigma), arr)[1])

    ff.compute_flow_following_movie(
        foreground, dp, labels,
        FlowFollowingParams(
            median_kernel_time=1, median_kernel_space=1,
            gaussian_sigma_time=0.0, gaussian_sigma_space=0.0,
        ),
    )

    assert median_calls == []
    assert gauss_calls == []


def test_compute_flow_following_movie_progress_callback_invoked_per_frame():
    from cellflow.segmentation.flow_following import compute_flow_following_movie

    T, H, W = 3, 6, 6
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[:, 3, 3] = 1
    dp = np.zeros((T, 2, H, W), dtype=np.float32)

    calls: list[tuple[int, int]] = []
    compute_flow_following_movie(
        foreground, dp, labels,
        FlowFollowingParams(median_kernel_time=1, median_kernel_space=1),
        progress_cb=lambda done, total: calls.append((done, total)),
    )
    assert calls == [(1, 3), (2, 3), (3, 3)]
```

- [ ] **Step 2: Run the failing tests**

Run: `pytest tests/segmentation/test_flow_following.py -q`

Expected: FAIL with `ImportError: cannot import name 'compute_flow_following_movie'`.

- [ ] **Step 3: Implement the orchestration**

Append to `src/cellflow/segmentation/flow_following.py`:

```python
def compute_flow_following_movie(
    foreground_tyx: np.ndarray,    # (T, Y, X) bool
    dp_tcyx: np.ndarray,           # (T, 2, Y, X) float32
    labels_tyx: np.ndarray,        # (T, Y, X) int32
    params: FlowFollowingParams,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame flow-following segmentation with pre-integration filtering.

    Returns
    -------
    filtered_dp_tcyx : (T, 2, Y, X) float32 — flow stack after median+Gaussian.
    cell_labels_tyx  : (T, Y, X) int32      — same labelling as input nuclei.
    """
    foreground = np.asarray(foreground_tyx, dtype=bool)
    dp = np.asarray(dp_tcyx, dtype=np.float32)
    labels = np.asarray(labels_tyx, dtype=np.int32)

    T = dp.shape[0]

    filtered = dp
    if params.median_kernel_time > 1 or params.median_kernel_space > 1:
        filtered = median_filter(
            filtered,
            size=(
                1,
                int(params.median_kernel_time),
                int(params.median_kernel_space),
                int(params.median_kernel_space),
            ),
        )
    if params.gaussian_sigma_time > 0.0 or params.gaussian_sigma_space > 0.0:
        filtered = gaussian_filter(
            filtered,
            sigma=(
                0.0,
                float(params.gaussian_sigma_time),
                float(params.gaussian_sigma_space),
                float(params.gaussian_sigma_space),
            ),
        )
    filtered = np.asarray(filtered, dtype=np.float32)

    out_labels = np.zeros_like(labels, dtype=np.int32)

    for t in range(T):
        prob_mask = foreground[t]
        nuclear_labels = labels[t]

        if not prob_mask.any() or not (nuclear_labels > 0).any():
            if progress_cb is not None:
                progress_cb(t + 1, T)
            continue

        flow_yx2 = np.stack(
            [filtered[t, 0], filtered[t, 1]], axis=-1
        ).astype(np.float32)
        mags = np.hypot(flow_yx2[..., 0], flow_yx2[..., 1])
        mean_mag = float(mags[prob_mask].mean()) if prob_mask.any() else 0.0
        if mean_mag > 1e-6:
            flow_yx2 = (flow_yx2 / mean_mag).astype(np.float32)

        dist, (ny, nx) = distance_transform_edt(
            nuclear_labels == 0, return_indices=True
        )
        H, W = nuclear_labels.shape
        yi, xi = np.indices((H, W))
        dy = (ny - yi).astype(np.float32)
        dx = (nx - xi).astype(np.float32)
        norm = np.hypot(dy, dx)
        safe = np.where(norm > 0, norm, 1.0)
        grav_y = (dy / safe).astype(np.float32)
        grav_x = (dx / safe).astype(np.float32)
        inside = nuclear_labels > 0
        grav_y[inside] = 0.0
        grav_x[inside] = 0.0

        integrated = _flow_integrate(
            nuclear_labels.astype(np.int32),
            np.ascontiguousarray(flow_yx2, dtype=np.float32),
            grav_y, grav_x,
            dist.astype(np.float32),
            ny.astype(np.int32), nx.astype(np.int32),
            prob_mask,
            int(params.max_iterations),
            float(params.flow_step_scale),
            float(params.flow_weight),
            float(params.capture_radius),
        )

        out_labels[t] = _fill_foreground(integrated, prob_mask)

        if progress_cb is not None:
            progress_cb(t + 1, T)

    return filtered, out_labels
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/segmentation/test_flow_following.py -q`

Expected: PASS (14 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/segmentation/flow_following.py tests/segmentation/test_flow_following.py
git commit -m "feat(segmentation): add compute_flow_following_movie per-frame orchestration"
```

---

### Task 5: Export New Symbols From Segmentation Package

**Files:**
- Modify: `src/cellflow/segmentation/__init__.py`

- [ ] **Step 1: Write the failing import test**

Append to `tests/segmentation/test_flow_following.py`:

```python
def test_flow_following_symbols_reexported_from_segmentation_package():
    from cellflow.segmentation import (
        FlowFollowingParams as PkgParams,
        compute_flow_following_movie as pkg_fn,
    )
    assert PkgParams().capture_radius == 3.0
    assert callable(pkg_fn)
```

- [ ] **Step 2: Run the failing test**

Run: `pytest tests/segmentation/test_flow_following.py::test_flow_following_symbols_reexported_from_segmentation_package -q`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Re-export the new symbols**

In `src/cellflow/segmentation/__init__.py`, immediately after the `from cellflow.segmentation.watershed_3d import compute_3d_temporal_watershed` line (line 4), add:

```python
from cellflow.segmentation.flow_following import (
    FlowFollowingParams,
    compute_flow_following_movie,
)
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/segmentation/test_flow_following.py -q`

Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/segmentation/__init__.py tests/segmentation/test_flow_following.py
git commit -m "feat(segmentation): export FlowFollowingParams and compute_flow_following_movie"
```

---

### Task 6: Replace Cell Workflow Widget With Flow-Following Section

**Files:**
- Modify: `src/cellflow/napari/cell_workflow_widget.py`
- Delete: `tests/napari/test_cell_workflow_preview.py`
- Create: `tests/napari/test_cell_workflow_widget.py`

- [ ] **Step 1: Write the failing widget tests**

Create `tests/napari/test_cell_workflow_widget.py`:

```python
"""Tests for the cell workflow widget — Flow-Following Segmentation section."""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication


class _LayerCollection(dict):
    def remove(self, layer):
        self.pop(layer.name, None)


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()
        self.dims = SimpleNamespace(
            current_step=(0,),
            events=SimpleNamespace(
                current_step=SimpleNamespace(connect=lambda cb: None)
            ),
        )

    def add_image(self, data, *, name, **kwargs):
        layer = SimpleNamespace(data=np.asarray(data), name=name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_labels(self, data, *, name, **kwargs):
        layer = SimpleNamespace(data=np.asarray(data), name=name, **kwargs)
        self.layers[name] = layer
        return layer


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.cell_workflow_widget", None)
    return importlib.import_module("cellflow.napari.cell_workflow_widget")


def _make_sync_thread_worker():
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


def test_widget_exposes_flow_following_section_with_default_params(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    assert widget.flow_section.title() == "Flow-Following Segmentation"

    assert widget.ff_median_time_spin.value() == 3
    assert widget.ff_median_space_spin.value() == 5
    assert widget.ff_gauss_time_spin.value() == 0.0
    assert widget.ff_gauss_space_spin.value() == 0.0

    assert widget.ff_flow_weight_spin.value() == 0.5
    assert widget.ff_step_scale_spin.value() == 0.2
    assert widget.ff_max_iter_spin.value() == 100
    assert widget.ff_capture_radius_spin.value() == 3.0

    assert widget.ff_run_btn.text() == "Run"
    assert widget.ff_cancel_btn.text() == "Cancel"
    assert widget.ff_cancel_btn.isEnabled() is False

    widget.deleteLater()
    app.processEvents()


def test_widget_get_set_state_round_trips_flow_following_params(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    state = {
        "flow_following": {
            "median_time": 5, "median_space": 7,
            "gauss_time": 1.5, "gauss_space": 2.0,
            "flow_weight": 0.7, "step_scale": 0.3,
            "max_iter": 200, "capture_radius": 5.0,
        }
    }
    widget.set_state(state)
    got = widget.get_state()

    assert got["flow_following"] == state["flow_following"]

    widget.deleteLater()
    app.processEvents()


def test_widget_input_status_label_shows_check_for_each_required_file(monkeypatch, tmp_path):
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
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif",
                     np.zeros((1, 4, 4), dtype=bool))
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif",
                     np.zeros((1, 4, 4), dtype=np.uint32))

    widget.refresh(pos_dir)

    text = widget.ff_input_lbl.text()
    assert text.count("✓") == 4
    assert "✗" not in text

    widget.deleteLater()
    app.processEvents()


def test_widget_input_status_label_shows_cross_when_files_missing(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget.refresh(pos_dir)

    text = widget.ff_input_lbl.text()
    assert text.count("✗") == 4

    widget.deleteLater()
    app.processEvents()


def test_widget_run_calls_compute_flow_following_movie_and_writes_outputs(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "3_cell").mkdir()

    T, Z, H, W = 2, 2, 6, 6
    prob = np.zeros((T, Z, H, W), dtype=np.float32)
    dp = np.zeros((T, Z, 2, H, W), dtype=np.float32)
    fg = np.ones((T, H, W), dtype=bool)
    nuc = np.zeros((T, H, W), dtype=np.uint32)
    nuc[:, 3, 3] = 5

    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif", dp)
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif", fg)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", nuc)

    fake_filtered = np.full((T, 2, H, W), 0.5, dtype=np.float32)
    fake_labels = np.full((T, H, W), 5, dtype=np.int32)

    captured: dict[str, object] = {}

    def fake_compute(foreground, dp_tcyx, labels, params, progress_cb=None):
        captured["foreground_shape"] = foreground.shape
        captured["dp_shape"] = dp_tcyx.shape
        captured["labels_shape"] = labels.shape
        captured["params"] = params
        if progress_cb is not None:
            progress_cb(T, T)
        return fake_filtered, fake_labels

    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    with patch(
        "cellflow.segmentation.compute_flow_following_movie",
        fake_compute,
    ):
        widget._on_run_flow_following()

    assert captured["foreground_shape"] == (T, H, W)
    assert captured["dp_shape"] == (T, 2, H, W)
    assert captured["labels_shape"] == (T, H, W)
    assert captured["params"].capture_radius == 3.0

    assert (pos_dir / "3_cell" / "filtered_flow_mag.tif").exists()
    assert (pos_dir / "3_cell" / "tracked_labels.tif").exists()
    mag = tifffile.imread(str(pos_dir / "3_cell" / "filtered_flow_mag.tif"))
    assert mag.shape == (T, H, W)
    assert mag.dtype == np.float32
    labels_out = tifffile.imread(str(pos_dir / "3_cell" / "tracked_labels.tif"))
    assert labels_out.dtype == np.uint32

    assert "Filtered Flow Magnitude" in viewer.layers
    assert "Cell Labels" in viewer.layers

    widget.deleteLater()
    app.processEvents()


def test_widget_run_aborts_when_input_file_missing(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget = mod.CellWorkflowWidget(_FakeViewer())
    widget.refresh(pos_dir)

    called = False

    def fake_compute(*args, **kwargs):
        nonlocal called
        called = True
        return None, None

    with patch(
        "cellflow.segmentation.compute_flow_following_movie",
        fake_compute,
    ):
        widget._on_run_flow_following()

    assert called is False
    assert "Missing" in widget.ff_status_lbl.text()

    widget.deleteLater()
    app.processEvents()
```

- [ ] **Step 2: Delete the obsolete preview test file**

```bash
git rm tests/napari/test_cell_workflow_preview.py
```

- [ ] **Step 3: Run the failing widget tests**

Run: `pytest tests/napari/test_cell_workflow_widget.py -q`

Expected: FAIL — most tests will fail at attribute access (`widget.flow_section`, `widget.ff_median_time_spin`, etc.) because the widget still has the old sections.

- [ ] **Step 4: Replace the widget body**

Overwrite `src/cellflow/napari/cell_workflow_widget.py` with:

```python
"""Cell segmentation widget for CellFlow — Flow-Following Segmentation."""
from __future__ import annotations

import logging
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QSpinBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import add_block_button_row, block_grid, sweep_parameter_grid

logger = logging.getLogger(__name__)

_FILTERED_FLOW_LAYER = "Filtered Flow Magnitude"
_CELL_LABELS_LAYER = "Cell Labels"
_FF_SPIN_WIDTH = 80
_FF_SPIN_MIN_WIDTH = int(_FF_SPIN_WIDTH * 0.9)


class CellWorkflowWidget(QWidget):
    """Cell segmentation — Flow-Following Segmentation."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._ff_worker = None
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        self.input_files = PipelineFilesWidget([
            ("Inputs", [
                ("1_cellpose/cell_prob_3dt.tif",   "Cell prob 3D+t"),
                ("1_cellpose/cell_dp_3dt.tif",     "Cell dp 3D+t"),
                ("3_cell/foreground_masks.tif",    "Foreground masks"),
                ("2_nucleus/tracked_labels.tif",   "Nucleus tracked labels"),
            ]),
        ])
        layout.addWidget(self.input_files)

        _ff_inner = QWidget()
        ff_lay = QVBoxLayout(_ff_inner)
        ff_lay.setContentsMargins(4, 4, 4, 4)
        ff_lay.setSpacing(4)
        ff_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        ff_scroll = QScrollArea()
        ff_scroll.setWidgetResizable(True)
        ff_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        ff_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ff_scroll.setFrameShape(QFrame.NoFrame)
        ff_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        ff_params_widget = QWidget()
        ff_params_widget.setMinimumWidth(520)
        ff_params_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        ff_params_lay = QVBoxLayout(ff_params_widget)
        ff_params_lay.setContentsMargins(0, 0, 0, 0)
        ff_params_lay.setSpacing(4)
        ff_params_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        def _dspin(lo, hi, val, step, decimals=1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return s

        def _ispin(lo, hi, val, step=1):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return s

        ff_grid = sweep_parameter_grid(spin_width=_FF_SPIN_WIDTH)

        self.ff_median_time_spin   = _ispin(1, 15, 3)
        self.ff_median_space_spin  = _ispin(1, 15, 5)
        self.ff_gauss_time_spin    = _dspin(0.0, 10.0, 0.0, 0.1)
        self.ff_gauss_space_spin   = _dspin(0.0, 10.0, 0.0, 0.1)
        ff_grid.addWidget(QLabel("Median t kernel:"),  1, 0)
        ff_grid.addWidget(self.ff_median_time_spin,    1, 1)
        ff_grid.addWidget(QLabel("Median xy kernel:"), 2, 0)
        ff_grid.addWidget(self.ff_median_space_spin,   2, 1)
        ff_grid.addWidget(QLabel("Gaussian t σ:"),     3, 0)
        ff_grid.addWidget(self.ff_gauss_time_spin,     3, 1)
        ff_grid.addWidget(QLabel("Gaussian xy σ:"),    4, 0)
        ff_grid.addWidget(self.ff_gauss_space_spin,    4, 1)

        self.ff_flow_weight_spin     = _dspin(0.0, 1.0, 0.5, 0.05, decimals=2)
        self.ff_step_scale_spin      = _dspin(0.05, 1.0, 0.2, 0.05, decimals=2)
        self.ff_max_iter_spin        = _ispin(10, 500, 100, step=10)
        self.ff_capture_radius_spin  = _dspin(0.5, 10.0, 3.0, 0.5)
        ff_grid.addWidget(QLabel("Flow weight:"),       5, 0)
        ff_grid.addWidget(self.ff_flow_weight_spin,     5, 1)
        ff_grid.addWidget(QLabel("Step scale:"),        6, 0)
        ff_grid.addWidget(self.ff_step_scale_spin,      6, 1)
        ff_grid.addWidget(QLabel("Max iterations:"),    7, 0)
        ff_grid.addWidget(self.ff_max_iter_spin,        7, 1)
        ff_grid.addWidget(QLabel("Capture radius:"),    8, 0)
        ff_grid.addWidget(self.ff_capture_radius_spin,  8, 1)
        ff_grid.setColumnStretch(1, 1)

        ff_params_lay.addLayout(ff_grid)

        self.ff_run_btn    = QPushButton("Run")
        self.ff_cancel_btn = QPushButton("Cancel")
        self.ff_cancel_btn.setEnabled(False)
        for btn in (self.ff_run_btn, self.ff_cancel_btn):
            btn.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(btn_row, 0, self.ff_run_btn, self.ff_cancel_btn)
        ff_params_lay.addLayout(btn_row)

        self.ff_input_lbl = QLabel("")
        self.ff_input_lbl.setWordWrap(True)
        ff_params_lay.addWidget(self.ff_input_lbl)

        self.ff_status_lbl = QLabel("")
        self.ff_status_lbl.setWordWrap(True)
        self.ff_status_lbl.setVisible(False)
        ff_params_lay.addWidget(self.ff_status_lbl)

        self.ff_progress_bar = QProgressBar()
        self.ff_progress_bar.setRange(0, 100)
        self.ff_progress_bar.setValue(0)
        self.ff_progress_bar.setVisible(False)
        ff_params_lay.addWidget(self.ff_progress_bar)

        self.ff_files = PipelineFilesWidget([
            ("", [
                ("3_cell/filtered_flow_mag.tif", "Filtered flow magnitude"),
                ("3_cell/tracked_labels.tif",    "Cell labels"),
            ]),
        ])
        ff_params_lay.addWidget(self.ff_files)
        self._update_ff_status_labels()

        ff_scroll.setWidget(ff_params_widget)
        ff_lay.addWidget(ff_scroll)
        self.flow_section = CollapsibleSection(
            "Flow-Following Segmentation", _ff_inner, expanded=True
        )
        layout.addWidget(self.flow_section)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.ff_run_btn.clicked.connect(self._on_run_flow_following)
        self.ff_cancel_btn.clicked.connect(self._on_cancel_flow_following)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif" if self._pos_dir else None

    def _dp_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_dp_3dt.tif" if self._pos_dir else None

    def _foreground_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "foreground_masks.tif" if self._pos_dir else None

    def _nucleus_labels_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    def _flow_mag_out_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "filtered_flow_mag.tif" if self._pos_dir else None

    def _cell_labels_out_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "tracked_labels.tif" if self._pos_dir else None

    # ------------------------------------------------------------------
    # State + status
    # ------------------------------------------------------------------
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self.input_files.refresh(pos_dir)
        self.ff_files.refresh(pos_dir)
        self._update_ff_status_labels()

    def get_state(self) -> dict:
        return {
            "flow_following": {
                "median_time":     self.ff_median_time_spin.value(),
                "median_space":    self.ff_median_space_spin.value(),
                "gauss_time":      self.ff_gauss_time_spin.value(),
                "gauss_space":     self.ff_gauss_space_spin.value(),
                "flow_weight":     self.ff_flow_weight_spin.value(),
                "step_scale":      self.ff_step_scale_spin.value(),
                "max_iter":        self.ff_max_iter_spin.value(),
                "capture_radius":  self.ff_capture_radius_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if "flow_following" in state:
            ff = state["flow_following"]
            if "median_time"    in ff: self.ff_median_time_spin.setValue(ff["median_time"])
            if "median_space"   in ff: self.ff_median_space_spin.setValue(ff["median_space"])
            if "gauss_time"     in ff: self.ff_gauss_time_spin.setValue(ff["gauss_time"])
            if "gauss_space"    in ff: self.ff_gauss_space_spin.setValue(ff["gauss_space"])
            if "flow_weight"    in ff: self.ff_flow_weight_spin.setValue(ff["flow_weight"])
            if "step_scale"     in ff: self.ff_step_scale_spin.setValue(ff["step_scale"])
            if "max_iter"       in ff: self.ff_max_iter_spin.setValue(ff["max_iter"])
            if "capture_radius" in ff: self.ff_capture_radius_spin.setValue(ff["capture_radius"])

    def _update_ff_status_labels(self) -> None:
        if self._pos_dir is None:
            self.ff_input_lbl.setText("Inputs: no project open.")
            return
        check = "✓"
        cross = "✗"
        prob_ok    = (p := self._prob_path()) is not None and p.exists()
        dp_ok      = (p := self._dp_path()) is not None and p.exists()
        fg_ok      = (p := self._foreground_path()) is not None and p.exists()
        nuc_ok     = (p := self._nucleus_labels_path()) is not None and p.exists()
        self.ff_input_lbl.setText(
            f"Inputs: {check if prob_ok else cross} prob  "
            f"{check if dp_ok else cross} dp  "
            f"{check if fg_ok else cross} foreground  "
            f"{check if nuc_ok else cross} nucleus labels"
        )

    def _set_ff_status(self, msg: str) -> None:
        self.ff_status_lbl.setText(msg)
        self.ff_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_ff_buttons_running(self, running: bool) -> None:
        self.ff_run_btn.setEnabled(not running)
        self.ff_cancel_btn.setEnabled(running)
        self.ff_progress_bar.setVisible(running)
        if not running:
            self.ff_progress_bar.setValue(0)

    def _on_ff_progress(self, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            if total > 0:
                self.ff_progress_bar.setRange(0, total)
                self.ff_progress_bar.setValue(done)
            self._set_ff_status(msg)
        else:
            self._set_ff_status(str(data))

    def _on_ff_worker_error(self, exc: Exception) -> None:
        if self._ff_worker is None:
            return
        self._ff_worker = None
        self.ff_progress_bar.setVisible(False)
        self._set_ff_buttons_running(False)
        self._set_ff_status(f"Error: {exc}")
        logger.exception("Flow-following worker error", exc_info=exc)

    # ------------------------------------------------------------------
    # Run / Cancel
    # ------------------------------------------------------------------
    def _on_run_flow_following(self) -> None:
        if self._pos_dir is None:
            self._set_ff_status("No project open.")
            return

        prob_path = self._prob_path()
        dp_path = self._dp_path()
        fg_path = self._foreground_path()
        nuc_path = self._nucleus_labels_path()
        flow_mag_path = self._flow_mag_out_path()
        labels_path = self._cell_labels_out_path()

        for path, name in [
            (prob_path, "cell_prob_3dt.tif"),
            (dp_path,   "cell_dp_3dt.tif"),
            (fg_path,   "foreground_masks.tif"),
            (nuc_path,  "tracked_labels.tif (2_nucleus)"),
        ]:
            if path is None or not path.exists():
                self._set_ff_status(f"Missing: {name}")
                return

        params_snapshot = self._params_from_ui()
        pos_dir = self._pos_dir

        def _on_done(result):
            self._ff_worker = None
            self._set_ff_buttons_running(False)
            filtered_mag, labels = result
            for layer_name, data, kwargs, adder in [
                (_FILTERED_FLOW_LAYER, filtered_mag,
                    {"colormap": "inferno", "blending": "additive"},
                    self.viewer.add_image),
                (_CELL_LABELS_LAYER, labels, {}, self.viewer.add_labels),
            ]:
                if layer_name in self.viewer.layers:
                    try:
                        self.viewer.layers[layer_name].data = data
                    except Exception:
                        self.viewer.layers.remove(self.viewer.layers[layer_name])
                        adder(data, name=layer_name, **kwargs)
                else:
                    adder(data, name=layer_name, **kwargs)
            self.ff_files.refresh(pos_dir)
            self._update_ff_status_labels()
            self._set_ff_status("Flow-following segmentation complete.")

        @thread_worker(connect={
            "yielded":  self._on_ff_progress,
            "returned": _on_done,
            "errored":  self._on_ff_worker_error,
        })
        def _worker():
            from cellflow.database.hypotheses import normalize_seeded_watershed_dp_stack
            from cellflow.segmentation import compute_flow_following_movie

            yield (0, 5, "Loading inputs...")
            prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            if prob.ndim == 3:
                prob = prob[np.newaxis]
            dp_raw = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
            dp_full = normalize_seeded_watershed_dp_stack(dp_raw, prob.shape)  # (T, Z, C, Y, X)
            dp_tcyx = dp_full[:, :, :2].mean(axis=1).astype(np.float32)        # (T, 2, Y, X)

            yield (1, 5, "Loading foreground...")
            foreground = np.asarray(tifffile.imread(str(fg_path)), dtype=bool)

            yield (2, 5, "Loading nucleus labels...")
            nucleus = np.asarray(tifffile.imread(str(nuc_path)), dtype=np.int32)

            yield (3, 5, "Running flow-following segmentation...")
            n_t = dp_tcyx.shape[0]

            def _progress(done: int, total: int) -> None:
                # Worker thread cannot yield from inside a callback; we rely on
                # the kernel finishing per-frame fast enough that the bar updates
                # at the next yield. Persist as text only.
                pass

            filtered_dp, cell_labels = compute_flow_following_movie(
                foreground, dp_tcyx, nucleus, params_snapshot, progress_cb=_progress,
            )

            yield (4, 5, "Saving outputs...")
            filtered_mag = np.sqrt(
                filtered_dp[:, 0] ** 2 + filtered_dp[:, 1] ** 2
            ).astype(np.float32)
            flow_mag_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(flow_mag_path), filtered_mag, compression="zlib")
            tifffile.imwrite(
                str(labels_path),
                cell_labels.astype(np.uint32),
                compression="zlib",
            )
            return filtered_mag, cell_labels.astype(np.uint32)

        self._set_ff_status("Running flow-following segmentation...")
        self._set_ff_buttons_running(True)
        self._ff_worker = _worker()

    def _on_cancel_flow_following(self) -> None:
        if self._ff_worker is not None:
            worker = self._ff_worker
            self._ff_worker = None
            worker.quit()
        self._set_ff_buttons_running(False)
        self._set_ff_status("Flow-following cancelled.")

    def _params_from_ui(self):
        from cellflow.segmentation import FlowFollowingParams
        return FlowFollowingParams(
            median_kernel_time=int(self.ff_median_time_spin.value()),
            median_kernel_space=int(self.ff_median_space_spin.value()),
            gaussian_sigma_time=float(self.ff_gauss_time_spin.value()),
            gaussian_sigma_space=float(self.ff_gauss_space_spin.value()),
            flow_weight=float(self.ff_flow_weight_spin.value()),
            flow_step_scale=float(self.ff_step_scale_spin.value()),
            max_iterations=int(self.ff_max_iter_spin.value()),
            capture_radius=float(self.ff_capture_radius_spin.value()),
        )
```

- [ ] **Step 5: Run the widget tests**

Run: `pytest tests/napari/test_cell_workflow_widget.py -q`

Expected: PASS (6 tests).

- [ ] **Step 6: Run the segmentation tests to confirm no regressions**

Run: `pytest tests/segmentation/test_flow_following.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cellflow/napari/cell_workflow_widget.py tests/napari/test_cell_workflow_widget.py tests/napari/test_cell_workflow_preview.py
git commit -m "feat(napari): replace cell workflow widget with flow-following section"
```

---

### Task 7: Delete Watershed_3d, Build_mean_z, And Their Tests

**Files:**
- Delete: `src/cellflow/segmentation/watershed_3d.py`
- Modify: `src/cellflow/segmentation/__init__.py`
- Modify: `tests/segmentation/test_label_postprocessing.py`

- [ ] **Step 1: Drop the deprecated tests**

In `tests/segmentation/test_label_postprocessing.py`, delete:
- `test_build_mean_z_consensus_boundary_returns_correct_shapes` (lines 48–84)
- `test_build_mean_z_consensus_boundary_single_gamma_default` (lines 86–109)
- `test_build_mean_z_consensus_boundary_invokes_mask_callback` (lines 111–end)

Keep `test_fill_and_close_labels_fills_per_label_bounding_boxes` and any imports it actually uses; delete imports that are now unused (`build_mean_z_consensus_boundary`, anything only the deleted tests required).

- [ ] **Step 2: Drop the watershed_3d export and the build_mean_z function**

In `src/cellflow/segmentation/__init__.py`:

Remove the line:

```python
from cellflow.segmentation.watershed_3d import compute_3d_temporal_watershed
```

Delete the entire `def build_mean_z_consensus_boundary(...)` definition (lines 416–485 in the current file).

- [ ] **Step 3: Delete watershed_3d.py**

```bash
git rm src/cellflow/segmentation/watershed_3d.py
```

- [ ] **Step 4: Run segmentation and label-postprocessing tests**

Run:

```bash
pytest tests/segmentation -q
```

Expected: PASS — the trimmed `test_label_postprocessing.py` keeps one test, and `test_flow_following.py` continues to pass.

- [ ] **Step 5: Sanity-check that nucleus and database tests are unaffected**

Run:

```bash
pytest tests/database tests/napari/test_nucleus_tracking_correction_layout.py tests/tracking -q
```

Expected: PASS — `build_consensus_boundary`, `compute_seeded_watershed`, `compute_cellpose_flow_hypothesis`, and `SeededWatershedParams` still exist, so nothing nucleus-side breaks.

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/segmentation/__init__.py src/cellflow/segmentation/watershed_3d.py tests/segmentation/test_label_postprocessing.py
git commit -m "refactor(segmentation): remove watershed_3d and build_mean_z_consensus_boundary"
```

---

### Task 8: Final Verification

**Files:**
- No new source files unless an earlier task missed something focused.

- [ ] **Step 1: Run focused new-feature tests**

Run:

```bash
pytest tests/segmentation/test_flow_following.py tests/napari/test_cell_workflow_widget.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
pytest -q
```

Expected: PASS. If anything fails, the failure is one of:
- An old test still referencing `tests/napari/test_cell_workflow_preview.py` or `compute_3d_temporal_watershed` — delete the dead reference.
- A nucleus-side import that picked up `compute_3d_temporal_watershed` indirectly — confirm that the conservative deletion list in this plan is intact and only `build_mean_z_consensus_boundary` + `watershed_3d.py` were removed.

- [ ] **Step 3: Commit any verification fixes**

If fixes were needed:

```bash
git add src tests
git commit -m "fix: stabilise flow-following cell segmentation"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review

**Spec coverage:**
- New module `src/cellflow/segmentation/flow_following.py` with `FlowFollowingParams`, `_flow_integrate`, `_fill_foreground`, `compute_flow_following_movie` — Tasks 2–4.
- Median + Gaussian filtering on `(T, 2, Y, X)` flow with channel-axis size 1 — Task 4 (`test_compute_flow_following_movie_applies_median_and_gaussian_filters`).
- EDT-direction-only gravity (no centroid sum, no falloff parameter) — Task 4 orchestration body.
- Per-frame mean-magnitude flow normalisation with the 1e-6 skip — Task 4 orchestration body.
- Numba kernel signature matching spec including bilinear flow / nearest-neighbour gravity — Task 3.
- Voronoi `_fill_foreground` — Task 2.
- Widget restructure: deletes Contour Maps + 3D Temporal Watershed, adds Flow-Following section with the spec's parameters and defaults, reads the four required input files, writes `filtered_flow_mag.tif` + `tracked_labels.tif`, adds inferno + label napari layers — Task 6.
- Edge cases: empty foreground frame → zeros (Task 4); no nuclei → zeros (Task 4); missing input file → status error and worker not started (Task 6); cancel via `worker.quit()` (Task 6).
- Conservative cleanup as confirmed by user: only `watershed_3d.py` + `build_mean_z_consensus_boundary` + their three tests + the obsolete preview-test file go this PR — Task 7.

**Placeholder scan:** no `TBD`, `TODO`, or "implement later" entries; every step that changes code shows the code.

**Type consistency:** `FlowFollowingParams` field names (`median_kernel_time`, `median_kernel_space`, `gaussian_sigma_time`, `gaussian_sigma_space`, `flow_weight`, `flow_step_scale`, `max_iterations`, `capture_radius`) are used identically in the kernel call, the orchestration, and the widget's `_params_from_ui()`. The widget's `get_state()`/`set_state()` use shorter UI-facing keys (`median_time`, etc.) and round-trip through the matching test in Task 6.

**Deviation from spec to surface explicitly:** spec lists four functions for deletion in `segmentation/__init__.py`; per the user's "be conservative" guidance only `build_mean_z_consensus_boundary` is removed in this PR. The other three (`build_consensus_boundary`, `compute_cellpose_flow_hypothesis`, `compute_seeded_watershed`) and `SeededWatershedParams` remain, queued for the proper h5-pipeline cleanup in a follow-up PR.
