# Divergence Maps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Cellpose-mask sweep that builds `2_nucleus/contours.tif` and `2_nucleus/foreground_scores.tif` (plus the `*_prob_zavg.tif` files) with a direct, per-channel computation from `prob_3dt` and `dp_3dt`: `foreground = reduce_z(sigmoid(prob))`, `contours = reduce_z(clip(div(filter(dp)), 0, ∞))`. New outputs live in `1_cellpose/`.

**Architecture:** A Qt-free compute module (`segmentation/divergence_maps.py`) exposes `build_divergence_maps(prob_path, dp_path, contours_out, foreground_out, …)`. A new Qt widget (`napari/divergence_maps_widget.py`) wraps it with the same two-row (Nucleus, Cell) pattern as `CellposeWidget`, runs work in `thread_worker`s with progress/cancel, and is embedded as a top-level section in `main_widget.py` between Cellpose and Nucleus. The threshold-sweep that produces `*_sources.tif` for Ultrack is unchanged but reads the new locations.

**Tech Stack:** Python 3.10+, NumPy, SciPy (`gaussian_filter`, `median_filter`), tifffile, napari `thread_worker`, qtpy.

**Spec:** `notes/divergence_maps_spec.md`

---

## File Structure

### New
- `src/cellflow/segmentation/divergence_maps.py` — pure compute module: `DivergenceMapsReport` dataclass, `sigmoid`, `foreground_from_prob`, `divergence_2d`, `contour_from_dp`, `build_divergence_maps`. No Qt/napari imports.
- `tests/segmentation/test_divergence_maps.py` — unit tests for each helper plus an end-to-end I/O test against tmp TIFFs.
- `src/cellflow/napari/divergence_maps_widget.py` — `DivergenceMapsWidget(QWidget)`, mirrors `CellposeWidget`: per-channel rows (⚙ params / ▶ run-cancel), shared status + progress, embedded `PipelineFilesWidget`. Public API: `refresh(pos_dir)`, `get_state()`, `set_state(state)`, `output_files_tracker`.
- `tests/napari/test_divergence_maps_widget.py` — Qt smoke test with a fake viewer + stubbed `build_divergence_maps`.

### Modified
- `src/cellflow/napari/_paths.py` — `NucleusArtifactPaths`: `contours` → `1_cellpose/nucleus_contours.tif`, rename `foreground_scores` → `foreground` (path `1_cellpose/nucleus_foreground.tif`), add `nucleus_contours`/`nucleus_foreground` aliases, add `cell_contours`/`cell_foreground` (`1_cellpose/cell_*.tif`), drop `nucleus_prob_zavg`/`cell_prob_zavg`.
- `src/cellflow/napari/main_widget.py` — import + instantiate `DivergenceMapsWidget`, insert a section between Cellpose and Nucleus, wire state save/load and refresh.
- `src/cellflow/napari/nucleus_pipeline_widget.py` — drop `_on_build_nucleus_maps`, `_on_preview_contour_maps`, and the `build_nucleus_averaged_maps` branch of `_on_build_segmentation_inputs`. The remaining `_on_build_segmentation_inputs` reads contours/foreground from new locations and only runs `write_ultrack_source_stacks`. Remove `seg_preview_btn` (▷) since there's nothing left to preview from this widget — the new sources sweep is fast enough to just run.
- `src/cellflow/napari/nucleus_segmentation_inputs_widget.py` — drop the cellprob-range / cellprob-step / z-range / z-step controls (and their `RangeThumbProxy` aliases). Source-sweep controls remain.
- `src/cellflow/napari/_thresholds.py` — drop `map_cellprob_thresholds`/`map_z_indices` helpers (no longer referenced).
- `src/cellflow/napari/nucleus_workflow_widget.py` — file-tracker entries updated to new locations; remove the `_on_build_contour_maps`/`_on_preview_contour_maps` assertions/refs in `_connect_signals`.
- `src/cellflow/napari/data_panel_widget.py` — file-tracker entries updated.
- `src/cellflow/napari/cellpose_widget.py` — drop the `*_prob_zavg.tif` rows from `_PIPELINE_FILES`, drop the embedded `CellposeZavgVizWidget`.
- `src/cellflow/napari/cell_workflow_widget.py` — drop `*_prob_zavg.tif` reference rows.
- `src/cellflow/napari/nucleus_correction_widget.py` — replace `_cell_prob_zavg_path`/`_nucleus_prob_zavg_path` (and their callers) with `cell_foreground` / `nucleus_foreground` from `_paths.py`.
- `src/cellflow/napari/cell_correction_widget.py` — same swap: `_cell_prob_zavg_path` / `_nuc_prob_zavg_path` → `cell_foreground` / `nucleus_foreground`.
- `src/cellflow/napari/radial_refinement_widget.py` — `_fg_path` and `_contours_path` use the new `1_cellpose/nucleus_*.tif` locations (via `NucleusArtifactPaths.nucleus_foreground` / `.nucleus_contours`).
- `src/cellflow/tracking_ultrack/reseed.py` — only docstring/comment refers to `contour_maps`; argument names are already generic. Update no-op apart from the docstring sentence.
- `src/cellflow/segmentation/nucleus_segmentation.py` — drop `apply_gamma`, `build_consensus_boundary`, `build_nucleus_averaged_maps`, `NucleusAveragedMapsReport`. Keep `compute_contour_watershed` and its helpers (`_remove_small_labels`, `_remove_low_circularity_labels`, `_fill_and_close_labels`, `ContourWatershedParams`, `_LABEL_DTYPE`, `CancelledError`, `_check_cancel`).
- `src/cellflow/segmentation/__init__.py` — drop re-exports of `build_consensus_boundary`, `build_nucleus_averaged_maps`, `NucleusAveragedMapsReport`. The package-level `apply_gamma` (defined locally in `__init__.py`) stays because `cell_workflow_widget` and `tests/tracking/test_correction.py` still use it. Add `build_divergence_maps` / `DivergenceMapsReport` re-exports.
- `src/cellflow/segmentation/cellpose_runner.py` — `write_outputs` stops writing `{channel}_prob_zavg.tif`.

### Deleted
- `src/cellflow/segmentation/cellpose_probability_zavg.py`
- `src/cellflow/napari/cellpose_zavg_viz_widget.py`
- `scripts/precompute_cellpose_probability_zavgs.py`
- `tests/segmentation/test_cellpose_probability_zavg.py`
- `tests/napari/test_cellpose_zavg_viz_widget.py`
- `tests/segmentation/test_nucleus_averaged_maps.py`
- `tests/segmentation/test_foreground_masks.py` (uses `build_consensus_boundary`)

### Test updates (existing files)
- `tests/napari/test_nucleus_pipeline_widget.py` — drop `_on_build_contour_maps`, `_on_preview_contour_maps`, `build_nucleus_averaged_maps`, `build_consensus_boundary` mocks/assertions; switch source paths to `1_cellpose/nucleus_{contours,foreground}.tif`.
- `tests/napari/test_nucleus_tracking_inputs_widget.py`, `test_nucleus_tracking_correction_layout.py`, `test_nucleus_correction_widget.py`, `test_nucleus_db_browser_widget.py` — drop `apply_gamma`/`build_*` mock entries; replace `*_prob_zavg.tif` setup with `*_foreground.tif`.
- `tests/napari/test_cell_correction_widget.py`, `test_cell_workflow_widget.py` — replace `cell_prob_zavg.tif` / `nucleus_prob_zavg.tif` setup with `cell_foreground.tif` / `nucleus_foreground.tif`.
- `tests/napari/test_cellpose_widget.py`, `test_cellpose_file_contract.py` — drop assertions that `*_prob_zavg.tif` was written / appears in the file tracker.
- `tests/segmentation/test_cellpose_runner.py` — drop the `cell_prob_zavg.tif` / `nucleus_prob_zavg.tif` existence assertions.
- `tests/segmentation/test_2d_t_multilabel_graphcut_experiment.py` — only uses `foreground_scores` as a local variable name; no change needed.

---

## Conventions

- Output dtype: `float32`, compression `"zlib"`, written with `tifffile.imwrite`.
- Stacks: `prob_3dt.tif` is `(T, Z, Y, X)`; `dp_3dt.tif` is `(T, Z, 2, Y, X)` with channels `[dy, dx]`.
- Progress callbacks: `progress_cb(done, total, msg)` (matches `build_nucleus_averaged_maps`).
- Cancel: cooperative `cancel: Callable[[], bool]`, raise `CancelledError` (imported from `cellflow.segmentation.nucleus_segmentation`).
- Reductions: `"mean"` → `np.mean(axis=z)`, `"max"` → `np.max(axis=z)`.
- Sigmoid clamping: clip input logits to `[-88, 88]` before `1/(1+exp(-x))` (matches `cellpose_probability_zavg.sigmoid_z_average`).
- Filter order, fixed: median → gaussian → divergence → `clip(·, 0, ∞)` (per spec).
- `median_radius=0` and `smoothing_sigma=0.0` skip the corresponding filter.

---

## Task 1: Core compute module — `sigmoid` + `foreground_from_prob`

**Files:**
- Create: `src/cellflow/segmentation/divergence_maps.py`
- Test: `tests/segmentation/test_divergence_maps.py`

- [ ] **Step 1: Write the failing test**

Create `tests/segmentation/test_divergence_maps.py`:

```python
"""Unit tests for cellflow.segmentation.divergence_maps."""
from __future__ import annotations

import numpy as np
import pytest


def test_sigmoid_clamps_extreme_logits():
    from cellflow.segmentation.divergence_maps import sigmoid

    x = np.array([-1e6, 0.0, 1e6], dtype=np.float32)
    out = sigmoid(x)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-30)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0, abs=1e-30)


def test_foreground_from_prob_mean_matches_sigmoid_mean():
    from cellflow.segmentation.divergence_maps import foreground_from_prob

    rng = np.random.default_rng(0)
    prob = rng.normal(0, 3, size=(2, 4, 5, 6)).astype(np.float32)
    expected = (1.0 / (1.0 + np.exp(-np.clip(prob, -88, 88)))).mean(axis=1)
    out = foreground_from_prob(prob, reduction="mean")
    assert out.shape == (2, 5, 6)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)


def test_foreground_from_prob_max_matches_sigmoid_max():
    from cellflow.segmentation.divergence_maps import foreground_from_prob

    prob = np.array(
        [[[[-2.0, 2.0]], [[0.0, -3.0]]]],  # T=1, Z=2, Y=1, X=2
        dtype=np.float32,
    )
    out = foreground_from_prob(prob, reduction="max")
    expected = (1.0 / (1.0 + np.exp(-prob))).max(axis=1)
    assert out.shape == (1, 1, 2)
    np.testing.assert_allclose(out, expected)


def test_foreground_from_prob_rejects_unknown_reduction():
    from cellflow.segmentation.divergence_maps import foreground_from_prob

    prob = np.zeros((1, 1, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="reduction"):
        foreground_from_prob(prob, reduction="median")
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/segmentation/test_divergence_maps.py -v`
Expected: `ModuleNotFoundError: No module named 'cellflow.segmentation.divergence_maps'`.

- [ ] **Step 3: Create the module with `sigmoid` + `foreground_from_prob`**

Create `src/cellflow/segmentation/divergence_maps.py`:

```python
"""Divergence-based foreground & contour maps from Cellpose prob/dp outputs.

Replaces the (cellprob × z) mask sweep with a direct computation:

    foreground = reduce_z(sigmoid(prob))
    contours   = reduce_z(clip(div(filter(dp)), 0, ∞))

See ``notes/divergence_maps_spec.md`` for the rationale.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


ZReduction = Literal["mean", "max"]


@dataclass(frozen=True, slots=True)
class DivergenceMapsReport:
    """Summary returned by :func:`build_divergence_maps`."""

    frames: int
    foreground_z_reduction: ZReduction
    contour_z_reduction: ZReduction
    smoothing_sigma: float
    median_radius: int
    contours_path: Path
    foreground_path: Path


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid on float32 logits."""
    x = np.clip(np.asarray(x, dtype=np.float32), -88.0, 88.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def _reduce_z(arr_tzyx: np.ndarray, reduction: ZReduction) -> np.ndarray:
    if reduction == "mean":
        return arr_tzyx.mean(axis=1, dtype=np.float32).astype(np.float32, copy=False)
    if reduction == "max":
        return arr_tzyx.max(axis=1).astype(np.float32, copy=False)
    raise ValueError(f"reduction must be 'mean' or 'max', got {reduction!r}")


def foreground_from_prob(
    prob_tzyx: np.ndarray, *, reduction: ZReduction
) -> np.ndarray:
    """``sigmoid(prob)`` reduced across z. Returns ``(T, Y, X)`` float32 in [0, 1]."""
    p = sigmoid(prob_tzyx)
    return _reduce_z(p, reduction)
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `pytest tests/segmentation/test_divergence_maps.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/segmentation/divergence_maps.py tests/segmentation/test_divergence_maps.py
git commit -m "feat(segmentation): add foreground_from_prob and sigmoid helpers"
```

---

## Task 2: `divergence_2d` (central differences on flow field)

**Files:**
- Modify: `src/cellflow/segmentation/divergence_maps.py`
- Modify: `tests/segmentation/test_divergence_maps.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/segmentation/test_divergence_maps.py`:

```python
def test_divergence_2d_linear_field_returns_constant():
    from cellflow.segmentation.divergence_maps import divergence_2d

    # dy(y, x) = 2y -> ∂dy/∂y = 2; dx(y, x) = 3x -> ∂dx/∂x = 3; sum = 5.
    y, x = np.mgrid[0:8, 0:8].astype(np.float32)
    flow = np.stack([2.0 * y, 3.0 * x], axis=0)
    div = divergence_2d(flow)
    assert div.shape == (8, 8)
    assert div.dtype == np.float32
    # Interior values should equal 5 exactly under central differences.
    np.testing.assert_allclose(div[1:-1, 1:-1], 5.0, atol=1e-5)


def test_divergence_2d_rejects_wrong_shape():
    from cellflow.segmentation.divergence_maps import divergence_2d

    with pytest.raises(ValueError, match=r"\(2, Y, X\)"):
        divergence_2d(np.zeros((3, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\(2, Y, X\)"):
        divergence_2d(np.zeros((4, 5), dtype=np.float32))
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/segmentation/test_divergence_maps.py -v`
Expected: 2 failures with `ImportError: cannot import name 'divergence_2d'`.

- [ ] **Step 3: Implement `divergence_2d`**

Append to `src/cellflow/segmentation/divergence_maps.py`:

```python
def divergence_2d(flow_yx: np.ndarray) -> np.ndarray:
    """Divergence of a ``(2, Y, X)`` flow field with channels ``[dy, dx]``."""
    flow_yx = np.asarray(flow_yx, dtype=np.float32)
    if flow_yx.ndim != 3 or flow_yx.shape[0] != 2:
        raise ValueError(
            f"flow must be (2, Y, X) with channels [dy, dx]; got {flow_yx.shape}"
        )
    d_dy = np.gradient(flow_yx[0], axis=0)
    d_dx = np.gradient(flow_yx[1], axis=1)
    return (d_dy + d_dx).astype(np.float32, copy=False)
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/segmentation/test_divergence_maps.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/segmentation/divergence_maps.py tests/segmentation/test_divergence_maps.py
git commit -m "feat(segmentation): add divergence_2d for Cellpose flow field"
```

---

## Task 3: `contour_from_dp` — filter → divergence → clip → reduce

**Files:**
- Modify: `src/cellflow/segmentation/divergence_maps.py`
- Modify: `tests/segmentation/test_divergence_maps.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/segmentation/test_divergence_maps.py`:

```python
def test_contour_from_dp_skips_filters_when_off(monkeypatch):
    import cellflow.segmentation.divergence_maps as dm

    calls = {"median": 0, "gaussian": 0}

    def _no_median(*a, **kw):
        calls["median"] += 1
        return a[0]

    def _no_gauss(*a, **kw):
        calls["gaussian"] += 1
        return a[0]

    monkeypatch.setattr(dm, "median_filter", _no_median)
    monkeypatch.setattr(dm, "gaussian_filter", _no_gauss)

    # T=1, Z=1, channels=2, Y=4, X=4 — flat field → zero divergence.
    dp = np.zeros((1, 1, 2, 4, 4), dtype=np.float32)
    out = dm.contour_from_dp(
        dp, smoothing_sigma=0.0, median_radius=0, reduction="mean",
    )
    assert calls == {"median": 0, "gaussian": 0}
    assert out.shape == (1, 4, 4)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, 0.0)


def test_contour_from_dp_clips_negative_divergence():
    from cellflow.segmentation.divergence_maps import contour_from_dp

    # Construct a convergent field (div < 0 interior) — all output should be 0.
    y, x = np.mgrid[0:6, 0:6].astype(np.float32)
    dy = -y  # ∂/∂y = -1
    dx = -x  # ∂/∂x = -1
    flow = np.stack([dy, dx], axis=0)
    dp = flow[np.newaxis, np.newaxis]  # (T=1, Z=1, 2, Y, X)
    out = contour_from_dp(
        dp, smoothing_sigma=0.0, median_radius=0, reduction="mean",
    )
    # Interior of negative-divergence field clips to 0.
    np.testing.assert_allclose(out[0, 1:-1, 1:-1], 0.0)


def test_contour_from_dp_invokes_filters_in_order(monkeypatch):
    import cellflow.segmentation.divergence_maps as dm

    order: list[str] = []

    def _median(a, size):
        order.append(f"median:{size}")
        return a

    def _gauss(a, sigma):
        order.append(f"gauss:{sigma}")
        return a

    monkeypatch.setattr(dm, "median_filter", _median)
    monkeypatch.setattr(dm, "gaussian_filter", _gauss)

    dp = np.zeros((1, 1, 2, 4, 4), dtype=np.float32)
    dm.contour_from_dp(dp, smoothing_sigma=1.5, median_radius=2, reduction="max")

    # Each z-slice runs median then gaussian, once per channel (dy, dx).
    assert order == [
        "median:5", "median:5",  # 2 * radius + 1 = 5, applied to dy, dx
        "gauss:1.5", "gauss:1.5",
    ]


def test_contour_from_dp_reduces_max_vs_mean():
    from cellflow.segmentation.divergence_maps import contour_from_dp

    # T=1, Z=2; z=0 has zero div, z=1 has +1 div.
    y, x = np.mgrid[0:6, 0:6].astype(np.float32)
    z0 = np.zeros((2, 6, 6), dtype=np.float32)
    # dy = y/2 -> 0.5; dx = x/2 -> 0.5; sum = 1.0
    z1 = np.stack([y * 0.5, x * 0.5], axis=0).astype(np.float32)
    dp = np.stack([z0, z1], axis=0)[np.newaxis]  # (1, 2, 2, 6, 6)

    mean_out = contour_from_dp(dp, smoothing_sigma=0.0, median_radius=0, reduction="mean")
    max_out = contour_from_dp(dp, smoothing_sigma=0.0, median_radius=0, reduction="max")
    # Interior: mean ≈ (0 + 1)/2 = 0.5; max ≈ 1.0.
    np.testing.assert_allclose(mean_out[0, 1:-1, 1:-1], 0.5, atol=1e-5)
    np.testing.assert_allclose(max_out[0, 1:-1, 1:-1], 1.0, atol=1e-5)
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/segmentation/test_divergence_maps.py -v`
Expected: 4 failures with `ImportError: cannot import name 'contour_from_dp'`.

- [ ] **Step 3: Implement `contour_from_dp`**

Append to `src/cellflow/segmentation/divergence_maps.py`:

```python
from scipy.ndimage import gaussian_filter, median_filter  # noqa: E402


def _filter_flow(
    flow_2yx: np.ndarray, *, smoothing_sigma: float, median_radius: int,
) -> np.ndarray:
    """Apply median → gaussian per channel. Order matters (spec)."""
    out = flow_2yx
    if median_radius > 0:
        size = 2 * int(median_radius) + 1
        out = np.stack(
            [median_filter(out[0], size=size), median_filter(out[1], size=size)],
            axis=0,
        )
    if smoothing_sigma > 0.0:
        sigma = float(smoothing_sigma)
        out = np.stack(
            [gaussian_filter(out[0], sigma=sigma), gaussian_filter(out[1], sigma=sigma)],
            axis=0,
        )
    return out


def contour_from_dp(
    dp_tzcyx: np.ndarray,
    *,
    smoothing_sigma: float,
    median_radius: int,
    reduction: ZReduction,
) -> np.ndarray:
    """Per (t, z): filter → divergence → clip(≥0); then reduce across z.

    ``dp_tzcyx``: shape ``(T, Z, 2, Y, X)`` with channels ``[dy, dx]``.
    Returns ``(T, Y, X)`` float32.
    """
    arr = np.asarray(dp_tzcyx, dtype=np.float32)
    if arr.ndim != 5 or arr.shape[2] != 2:
        raise ValueError(
            f"dp must be (T, Z, 2, Y, X) with channels [dy, dx]; got {arr.shape}"
        )
    n_t, n_z, _, n_y, n_x = arr.shape
    pos = np.empty((n_t, n_z, n_y, n_x), dtype=np.float32)
    for t in range(n_t):
        for z in range(n_z):
            filt = _filter_flow(
                arr[t, z], smoothing_sigma=smoothing_sigma, median_radius=median_radius,
            )
            div = divergence_2d(filt)
            np.clip(div, 0.0, None, out=div)
            pos[t, z] = div
    return _reduce_z(pos, reduction)
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/segmentation/test_divergence_maps.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/segmentation/divergence_maps.py tests/segmentation/test_divergence_maps.py
git commit -m "feat(segmentation): add contour_from_dp with median+gaussian filtering"
```

---

## Task 4: `build_divergence_maps` — end-to-end file I/O

**Files:**
- Modify: `src/cellflow/segmentation/divergence_maps.py`
- Modify: `tests/segmentation/test_divergence_maps.py`

- [ ] **Step 1: Append failing test**

Append to `tests/segmentation/test_divergence_maps.py`:

```python
def test_build_divergence_maps_writes_and_reports(tmp_path):
    import tifffile
    from cellflow.segmentation.divergence_maps import (
        DivergenceMapsReport, build_divergence_maps,
    )

    rng = np.random.default_rng(0)
    prob = rng.normal(0, 2, size=(2, 3, 5, 6)).astype(np.float32)
    dp = rng.normal(0, 1, size=(2, 3, 2, 5, 6)).astype(np.float32)

    prob_path = tmp_path / "prob.tif"
    dp_path = tmp_path / "dp.tif"
    contours_out = tmp_path / "out_contours.tif"
    fg_out = tmp_path / "out_foreground.tif"
    tifffile.imwrite(str(prob_path), prob)
    tifffile.imwrite(str(dp_path), dp)

    calls: list[tuple[int, int, str]] = []
    report = build_divergence_maps(
        prob_path,
        dp_path,
        contours_out,
        fg_out,
        foreground_z_reduction="mean",
        contour_z_reduction="mean",
        smoothing_sigma=0.0,
        median_radius=0,
        progress_cb=lambda d, n, m: calls.append((d, n, m)),
    )

    assert isinstance(report, DivergenceMapsReport)
    assert report.frames == 2
    assert report.contours_path == contours_out
    assert report.foreground_path == fg_out

    fg = tifffile.imread(str(fg_out))
    assert fg.shape == (2, 5, 6)
    assert fg.dtype == np.float32
    assert 0.0 <= fg.min() and fg.max() <= 1.0
    contours = tifffile.imread(str(contours_out))
    assert contours.shape == (2, 5, 6)
    assert contours.dtype == np.float32
    assert contours.min() >= 0.0
    assert len(calls) >= 1
    assert calls[-1][0] == calls[-1][1]  # final report has done==total


def test_build_divergence_maps_respects_cancel(tmp_path):
    import tifffile
    from cellflow.segmentation.divergence_maps import build_divergence_maps
    from cellflow.segmentation import CancelledError

    prob = np.zeros((3, 1, 2, 2), dtype=np.float32)
    dp = np.zeros((3, 1, 2, 2, 2), dtype=np.float32)
    prob_path = tmp_path / "prob.tif"
    dp_path = tmp_path / "dp.tif"
    tifffile.imwrite(str(prob_path), prob)
    tifffile.imwrite(str(dp_path), dp)

    with pytest.raises(CancelledError):
        build_divergence_maps(
            prob_path,
            dp_path,
            tmp_path / "c.tif",
            tmp_path / "f.tif",
            foreground_z_reduction="mean",
            contour_z_reduction="mean",
            smoothing_sigma=0.0,
            median_radius=0,
            cancel=lambda: True,
        )


def test_build_divergence_maps_validates_shapes(tmp_path):
    import tifffile
    from cellflow.segmentation.divergence_maps import build_divergence_maps

    prob = np.zeros((2, 3, 5, 6), dtype=np.float32)
    dp = np.zeros((3, 3, 2, 5, 6), dtype=np.float32)  # T mismatch
    prob_path = tmp_path / "prob.tif"
    dp_path = tmp_path / "dp.tif"
    tifffile.imwrite(str(prob_path), prob)
    tifffile.imwrite(str(dp_path), dp)

    with pytest.raises(ValueError, match="same frame count"):
        build_divergence_maps(
            prob_path, dp_path,
            tmp_path / "c.tif", tmp_path / "f.tif",
            foreground_z_reduction="mean", contour_z_reduction="mean",
            smoothing_sigma=0.0, median_radius=0,
        )
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/segmentation/test_divergence_maps.py -v`
Expected: 3 failures with `ImportError: cannot import name 'build_divergence_maps'`.

- [ ] **Step 3: Implement `build_divergence_maps`**

Append to `src/cellflow/segmentation/divergence_maps.py`:

```python
import tifffile  # noqa: E402

from cellflow.segmentation.nucleus_segmentation import (  # noqa: E402
    CancelledError,
    _check_cancel,
)


def _as_tzyx(stack: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4:
        raise ValueError(f"{name} must be Z×Y×X or T×Z×Y×X.")
    return arr


def _as_tzcyx(stack: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 5 or arr.shape[2] != 2:
        raise ValueError(f"{name} must be Z×2×Y×X or T×Z×2×Y×X.")
    return arr


def build_divergence_maps(
    prob_path: str | Path,
    dp_path: str | Path,
    contours_out: str | Path,
    foreground_out: str | Path,
    *,
    foreground_z_reduction: ZReduction,
    contour_z_reduction: ZReduction,
    smoothing_sigma: float,
    median_radius: int,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> DivergenceMapsReport:
    """Compute and write ``contours`` and ``foreground`` from Cellpose prob/dp.

    Output stacks are ``T × Y × X`` float32. ``progress_cb`` is called per frame.
    """
    prob_stack = _as_tzyx(tifffile.imread(str(prob_path)), "prob")
    dp_stack = _as_tzcyx(tifffile.imread(str(dp_path)), "dp")
    if prob_stack.shape[0] != dp_stack.shape[0]:
        raise ValueError("prob and dp must have the same frame count.")
    if prob_stack.shape[1] != dp_stack.shape[1]:
        raise ValueError("prob and dp must have the same z count.")
    if prob_stack.shape[2:] != dp_stack.shape[3:]:
        raise ValueError("prob and dp must have the same Y×X shape.")

    n_t = int(prob_stack.shape[0])
    contour_frames: list[np.ndarray] = []
    foreground_frames: list[np.ndarray] = []
    for t in range(n_t):
        _check_cancel(cancel)
        if progress_cb is not None:
            progress_cb(t, n_t, f"Divergence maps: frame {t + 1}/{n_t}")
        fg = foreground_from_prob(
            prob_stack[t : t + 1], reduction=foreground_z_reduction,
        )
        contour = contour_from_dp(
            dp_stack[t : t + 1],
            smoothing_sigma=smoothing_sigma,
            median_radius=median_radius,
            reduction=contour_z_reduction,
        )
        foreground_frames.append(fg[0])
        contour_frames.append(contour[0])

    _check_cancel(cancel)
    contours_out = Path(contours_out)
    foreground_out = Path(foreground_out)
    contours_out.parent.mkdir(parents=True, exist_ok=True)
    foreground_out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(contours_out), np.stack(contour_frames).astype(np.float32),
        compression="zlib",
    )
    tifffile.imwrite(
        str(foreground_out), np.stack(foreground_frames).astype(np.float32),
        compression="zlib",
    )
    if progress_cb is not None:
        progress_cb(n_t, n_t, f"Divergence maps: wrote {n_t} frames")
    return DivergenceMapsReport(
        frames=n_t,
        foreground_z_reduction=foreground_z_reduction,
        contour_z_reduction=contour_z_reduction,
        smoothing_sigma=float(smoothing_sigma),
        median_radius=int(median_radius),
        contours_path=contours_out,
        foreground_path=foreground_out,
    )


__all__ = [
    "DivergenceMapsReport",
    "build_divergence_maps",
    "contour_from_dp",
    "divergence_2d",
    "foreground_from_prob",
    "sigmoid",
]
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/segmentation/test_divergence_maps.py -v`
Expected: 13 passed.

- [ ] **Step 5: Re-export from package**

Edit `src/cellflow/segmentation/__init__.py`. Add the import after the existing nucleus_segmentation block:

```python
from cellflow.segmentation.divergence_maps import (
    DivergenceMapsReport,
    build_divergence_maps,
)
```

And add `"DivergenceMapsReport"` and `"build_divergence_maps"` to `__all__`.

- [ ] **Step 6: Verify import**

Run: `python -c "from cellflow.segmentation import build_divergence_maps, DivergenceMapsReport; print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
git add src/cellflow/segmentation/divergence_maps.py src/cellflow/segmentation/__init__.py tests/segmentation/test_divergence_maps.py
git commit -m "feat(segmentation): add build_divergence_maps with file I/O"
```

---

## Task 5: Update `NucleusArtifactPaths` to new file layout

**Files:**
- Modify: `src/cellflow/napari/_paths.py`

This task introduces the new path properties. Old `contours` / `foreground_scores` / `nucleus_prob_zavg` / `cell_prob_zavg` properties are removed in the same change; callers are updated in later tasks (Tasks 8–12). To keep that bisectable, we land the path change first and let later tasks consume the new names.

- [ ] **Step 1: Read the current file**

(Required by the `Edit` tool's prerequisite — open `src/cellflow/napari/_paths.py` in your editor / read it once.)

- [ ] **Step 2: Replace the `NucleusArtifactPaths` body**

Replace lines 22–93 (the entire `NucleusArtifactPaths` class) with:

```python
@dataclass(frozen=True)
class NucleusArtifactPaths:
    """Resolve nucleus-workflow artifact locations under a position directory."""

    pos_dir: Path

    # 0_input
    @property
    def cell_zavg(self) -> Path:
        return self.pos_dir / "0_input" / "cell_zavg.tif"

    @property
    def nucleus_zavg(self) -> Path:
        return self.pos_dir / "0_input" / "nucleus_zavg.tif"

    @property
    def nls_zavg(self) -> Path:
        return self.pos_dir / "0_input" / "NLS_zavg.tif"

    # 1_cellpose — Cellpose outputs
    @property
    def prob(self) -> Path:
        return self.pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif"

    @property
    def cell_prob(self) -> Path:
        return self.pos_dir / "1_cellpose" / "cell_prob_3dt.tif"

    @property
    def dp(self) -> Path:
        return self.pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif"

    @property
    def cell_dp(self) -> Path:
        return self.pos_dir / "1_cellpose" / "cell_dp_3dt.tif"

    # 1_cellpose — Divergence-map outputs (per channel)
    @property
    def nucleus_contours(self) -> Path:
        return self.pos_dir / "1_cellpose" / "nucleus_contours.tif"

    @property
    def nucleus_foreground(self) -> Path:
        return self.pos_dir / "1_cellpose" / "nucleus_foreground.tif"

    @property
    def cell_contours(self) -> Path:
        return self.pos_dir / "1_cellpose" / "cell_contours.tif"

    @property
    def cell_foreground(self) -> Path:
        return self.pos_dir / "1_cellpose" / "cell_foreground.tif"

    # Aliases for the nucleus channel — historical names retained for callers
    # that don't need a channel suffix (e.g. tracking-input writers).
    @property
    def contours(self) -> Path:
        return self.nucleus_contours

    @property
    def foreground(self) -> Path:
        return self.nucleus_foreground

    # 2_nucleus
    @property
    def nucleus_dir(self) -> Path:
        return self.pos_dir / "2_nucleus"

    @property
    def contour_sources(self) -> Path:
        return self.nucleus_dir / "contour_sources.tif"

    @property
    def foreground_sources(self) -> Path:
        return self.nucleus_dir / "foreground_sources.tif"

    @property
    def tracked(self) -> Path:
        return self.nucleus_dir / "tracked_labels.tif"

    @property
    def ultrack_workdir(self) -> Path:
        return self.nucleus_dir / "ultrack_workdir"

    @property
    def ultrack_db(self) -> Path:
        return self.ultrack_workdir / "data.db"
```

- [ ] **Step 3: Update the module docstring**

Replace lines 1–15 (the file docstring) with:

```python
"""On-disk artifact paths for CellFlow position directories.

Canonical file layout under ``<pos_dir>/``:

    0_input/              — cell_zavg.tif, nucleus_zavg.tif, NLS_zavg.tif
    1_cellpose/           — nucleus_prob_3dt.tif, nucleus_dp_3dt.tif,
                            cell_prob_3dt.tif, cell_dp_3dt.tif,
                            nucleus_contours.tif, nucleus_foreground.tif,
                            cell_contours.tif, cell_foreground.tif
    2_nucleus/            — contour_sources.tif, foreground_sources.tif,
                            tracked_labels.tif, ultrack_workdir/data.db
    3_cell/               — filtered_dp.tif, foreground_masks.tif,
                            contours.tif, foreground_scores.tif,
                            tracked_labels.tif
    4_contact_analysis/   — contact_analysis.h5
"""
```

- [ ] **Step 4: Verify import**

Run:

```bash
python -c "from cellflow.napari._paths import NucleusArtifactPaths; \
p = NucleusArtifactPaths(__import__('pathlib').Path('/tmp/x')); \
print(p.nucleus_contours, p.nucleus_foreground, p.cell_contours, p.cell_foreground, p.contours, p.foreground)"
```

Expected: prints the six paths under `/tmp/x/1_cellpose/`.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/_paths.py
git commit -m "refactor(paths): move contour/foreground maps to 1_cellpose, drop zavg"
```

---

## Task 6: `DivergenceMapsWidget` skeleton (UI only, no run logic)

**Files:**
- Create: `src/cellflow/napari/divergence_maps_widget.py`
- Create: `tests/napari/test_divergence_maps_widget.py`

This task lands a runnable widget that builds without instantiating any actual compute. Run logic is wired in Task 7.

- [ ] **Step 1: Write a failing smoke test**

Create `tests/napari/test_divergence_maps_widget.py`:

```python
"""Smoke tests for DivergenceMapsWidget."""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy")


class _FakeViewer:
    def __init__(self):
        from types import SimpleNamespace
        self.layers = {}
        self.dims = SimpleNamespace(current_step=(0,))

    def add_image(self, *a, **kw): pass
    def add_labels(self, *a, **kw): pass


def test_widget_constructs_and_exposes_public_api(qtbot):
    from cellflow.napari.divergence_maps_widget import DivergenceMapsWidget

    viewer = _FakeViewer()
    w = DivergenceMapsWidget(viewer)
    qtbot.addWidget(w)

    # Per-channel rows
    assert w.nucleus_run_btn.isEnabled() in (True, False)
    assert w.cell_run_btn.isEnabled() in (True, False)
    assert w.nucleus_params_btn.isCheckable()
    assert w.cell_params_btn.isCheckable()

    # Per-channel parameter spinners exist with default values from the spec.
    assert w.nuc_smoothing_spin.value() == pytest.approx(1.0)
    assert w.nuc_median_spin.value() == 0
    assert w.nuc_fg_reduction.currentText() == "mean"
    assert w.nuc_contour_reduction.currentText() == "mean"
    assert w.cell_smoothing_spin.value() == pytest.approx(1.0)
    assert w.cell_median_spin.value() == 0

    # Public API used by main_widget.
    assert hasattr(w, "refresh")
    assert hasattr(w, "get_state")
    assert hasattr(w, "set_state")
    assert hasattr(w, "output_files_tracker")


def test_widget_state_roundtrip(qtbot):
    from cellflow.napari.divergence_maps_widget import DivergenceMapsWidget

    w = DivergenceMapsWidget(_FakeViewer())
    qtbot.addWidget(w)
    w.nuc_smoothing_spin.setValue(0.5)
    w.nuc_median_spin.setValue(3)
    w.cell_fg_reduction.setCurrentText("max")
    state = w.get_state()
    assert state["nucleus"]["smoothing_sigma"] == pytest.approx(0.5)
    assert state["nucleus"]["median_radius"] == 3
    assert state["cell"]["foreground_z_reduction"] == "max"

    w2 = DivergenceMapsWidget(_FakeViewer())
    qtbot.addWidget(w2)
    w2.set_state(state)
    assert w2.nuc_smoothing_spin.value() == pytest.approx(0.5)
    assert w2.nuc_median_spin.value() == 3
    assert w2.cell_fg_reduction.currentText() == "max"
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/napari/test_divergence_maps_widget.py -v`
Expected: `ModuleNotFoundError: No module named 'cellflow.napari.divergence_maps_widget'`.

- [ ] **Step 3: Create the widget**

Create `src/cellflow/napari/divergence_maps_widget.py`:

```python
"""Per-channel widget that builds nucleus/cell foreground & contour maps
directly from Cellpose ``prob_3dt`` and ``dp_3dt`` outputs.

Mirrors :class:`CellposeWidget` layout (one row per channel with
⚙ params / ▶ run-cancel and a shared status + progress bar).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import napari
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
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

from cellflow.napari._paths import NucleusArtifactPaths
from cellflow.napari._widget_helpers import tool_btn as _tool_btn
from cellflow.napari.ui_style import stage_header_label, status_label
from cellflow.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    make_pipeline_files_header,
)

logger = logging.getLogger(__name__)


_PIPELINE_FILES = [
    ("Inputs", [
        ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
        ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
        ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
    ]),
    ("Outputs", [
        ("1_cellpose/nucleus_contours.tif", "Nucleus contours"),
        ("1_cellpose/nucleus_foreground.tif", "Nucleus foreground"),
        ("1_cellpose/cell_contours.tif", "Cell contours"),
        ("1_cellpose/cell_foreground.tif", "Cell foreground"),
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


class DivergenceMapsWidget(QWidget):
    """Build per-channel foreground & contour maps from Cellpose prob/dp."""

    _progress_signal = Signal(int, int, str)

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._running_stage: str | None = None
        self._worker = None
        self._cancel_requested = False

        self._setup_ui()
        self._connect_signals()
        self._progress_signal.connect(self._progress)

    # ── UI ───────────────────────────────────────────────────────────
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

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
            self._pipeline_files_section, stage_key="divergence_maps", parent=self,
        )
        root.addWidget(self.pipeline_files_header)
        root.addWidget(self._pipeline_files_section)

        # Nucleus row
        self.nucleus_params_btn = _tool_btn(
            "⚙", "Show parameters for nucleus divergence maps.", checkable=True,
        )
        self.nucleus_run_btn = _tool_btn("▶", "Build nucleus divergence maps.")
        self.nucleus_section = self._build_channel_params_section("nucleus")
        self.nucleus_section.set_header_visible(False)
        self.nucleus_section.collapse()
        self.nucleus_params_btn.toggled.connect(
            lambda checked: self.nucleus_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Nucleus divergence maps"),
            self.nucleus_params_btn,
            self.nucleus_run_btn,
        ))
        root.addWidget(self.nucleus_section)

        # Cell row
        self.cell_params_btn = _tool_btn(
            "⚙", "Show parameters for cell divergence maps.", checkable=True,
        )
        self.cell_run_btn = _tool_btn("▶", "Build cell divergence maps.")
        self.cell_section = self._build_channel_params_section("cell")
        self.cell_section.set_header_visible(False)
        self.cell_section.collapse()
        self.cell_params_btn.toggled.connect(
            lambda checked: self.cell_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Cell divergence maps"),
            self.cell_params_btn,
            self.cell_run_btn,
        ))
        root.addWidget(self.cell_section)

        self.status_lbl = _make_status()
        root.addWidget(self.status_lbl)
        self.progress_bar = _make_progress()
        root.addWidget(self.progress_bar)

    def _build_channel_params_section(
        self, channel: Literal["nucleus", "cell"],
    ) -> CollapsibleSection:
        body = QWidget(self)
        form = QFormLayout(body)
        form.setContentsMargins(8, 4, 4, 4)
        form.setSpacing(4)

        fg_reduction = QComboBox()
        fg_reduction.addItems(["mean", "max"])
        fg_reduction.setCurrentText("mean")
        contour_reduction = QComboBox()
        contour_reduction.addItems(["mean", "max"])
        contour_reduction.setCurrentText("mean")
        smoothing_spin = QDoubleSpinBox()
        smoothing_spin.setRange(0.0, 20.0)
        smoothing_spin.setDecimals(2)
        smoothing_spin.setSingleStep(0.1)
        smoothing_spin.setValue(1.0)
        median_spin = QSpinBox()
        median_spin.setRange(0, 20)
        median_spin.setValue(0)
        form.addRow("Foreground z-reduction", fg_reduction)
        form.addRow("Contour z-reduction", contour_reduction)
        form.addRow("Smoothing sigma", smoothing_spin)
        form.addRow("Median radius", median_spin)

        prefix = "nuc" if channel == "nucleus" else "cell"
        setattr(self, f"{prefix}_fg_reduction", fg_reduction)
        setattr(self, f"{prefix}_contour_reduction", contour_reduction)
        setattr(self, f"{prefix}_smoothing_spin", smoothing_spin)
        setattr(self, f"{prefix}_median_spin", median_spin)
        return CollapsibleSection(
            f"{channel.title()} parameters", body, expanded=False,
        )

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

    # ── Signals ──────────────────────────────────────────────────────
    def _connect_signals(self) -> None:
        self.nucleus_run_btn.clicked.connect(lambda: self._on_run("nucleus"))
        self.cell_run_btn.clicked.connect(lambda: self._on_run("cell"))

    # ── Stubs filled in by Task 7 ───────────────────────────────────
    def _on_run(self, channel: Literal["nucleus", "cell"]) -> None:  # noqa: D401
        """Run/cancel dispatch — implemented in Task 7."""
        raise NotImplementedError("Wired in Task 7")

    def _progress(self, done: int, total: int, msg: str) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(done)
        self.status_lbl.setText(msg)
        self.status_lbl.setVisible(bool(msg))

    # ── Public API ───────────────────────────────────────────────────
    def refresh(self, pos_dir: Path | str | None) -> None:
        self._pos_dir = None if pos_dir is None or str(pos_dir) == "[no project]" else Path(pos_dir)
        self._files_widget.refresh(self._pos_dir)
        self._update_enabled()

    def _update_enabled(self) -> None:
        has_pos = self._pos_dir is not None
        for btn in (self.nucleus_run_btn, self.cell_run_btn,
                    self.nucleus_params_btn, self.cell_params_btn):
            btn.setEnabled(has_pos)

    def get_state(self) -> dict:
        return {
            "nucleus": self._channel_state("nuc"),
            "cell": self._channel_state("cell"),
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if "nucleus" in state:
            self._apply_channel_state("nuc", state["nucleus"])
        if "cell" in state:
            self._apply_channel_state("cell", state["cell"])

    def _channel_state(self, prefix: str) -> dict:
        return {
            "foreground_z_reduction": getattr(self, f"{prefix}_fg_reduction").currentText(),
            "contour_z_reduction": getattr(self, f"{prefix}_contour_reduction").currentText(),
            "smoothing_sigma": float(getattr(self, f"{prefix}_smoothing_spin").value()),
            "median_radius": int(getattr(self, f"{prefix}_median_spin").value()),
        }

    def _apply_channel_state(self, prefix: str, state: dict) -> None:
        if "foreground_z_reduction" in state:
            getattr(self, f"{prefix}_fg_reduction").setCurrentText(state["foreground_z_reduction"])
        if "contour_z_reduction" in state:
            getattr(self, f"{prefix}_contour_reduction").setCurrentText(state["contour_z_reduction"])
        if "smoothing_sigma" in state:
            getattr(self, f"{prefix}_smoothing_spin").setValue(float(state["smoothing_sigma"]))
        if "median_radius" in state:
            getattr(self, f"{prefix}_median_spin").setValue(int(state["median_radius"]))

    # ── Path helpers ────────────────────────────────────────────────
    def _paths(self) -> NucleusArtifactPaths | None:
        return NucleusArtifactPaths(self._pos_dir) if self._pos_dir else None
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/napari/test_divergence_maps_widget.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/divergence_maps_widget.py tests/napari/test_divergence_maps_widget.py
git commit -m "feat(napari): add DivergenceMapsWidget skeleton (UI only)"
```

---

## Task 7: Wire run/cancel handlers in `DivergenceMapsWidget`

**Files:**
- Modify: `src/cellflow/napari/divergence_maps_widget.py`
- Modify: `tests/napari/test_divergence_maps_widget.py`

- [ ] **Step 1: Append failing test**

Append to `tests/napari/test_divergence_maps_widget.py`:

```python
def test_run_invokes_build_divergence_maps(qtbot, tmp_path, monkeypatch):
    from cellflow.napari.divergence_maps_widget import DivergenceMapsWidget
    from cellflow.segmentation.divergence_maps import DivergenceMapsReport
    import cellflow.napari.divergence_maps_widget as widget_mod
    import tifffile, numpy as np

    # Lay out an empty position directory with placeholder inputs.
    pos = tmp_path / "pos00"
    cell = pos / "1_cellpose"
    cell.mkdir(parents=True)
    tifffile.imwrite(cell / "nucleus_prob_3dt.tif", np.zeros((1, 1, 2, 2), dtype=np.float32))
    tifffile.imwrite(cell / "nucleus_dp_3dt.tif", np.zeros((1, 1, 2, 2, 2), dtype=np.float32))

    captured: dict = {}

    def _fake_build(prob_path, dp_path, contours_out, foreground_out,
                    *, foreground_z_reduction, contour_z_reduction,
                    smoothing_sigma, median_radius,
                    progress_cb=None, cancel=None):
        captured.update(dict(
            prob_path=str(prob_path), dp_path=str(dp_path),
            contours_out=str(contours_out), foreground_out=str(foreground_out),
            foreground_z_reduction=foreground_z_reduction,
            contour_z_reduction=contour_z_reduction,
            smoothing_sigma=smoothing_sigma, median_radius=median_radius,
        ))
        return DivergenceMapsReport(
            frames=1,
            foreground_z_reduction=foreground_z_reduction,
            contour_z_reduction=contour_z_reduction,
            smoothing_sigma=smoothing_sigma,
            median_radius=median_radius,
            contours_path=contours_out,
            foreground_path=foreground_out,
        )

    monkeypatch.setattr(widget_mod, "build_divergence_maps", _fake_build)

    w = DivergenceMapsWidget(_FakeViewer())
    qtbot.addWidget(w)
    w.refresh(pos)
    w.nuc_smoothing_spin.setValue(2.0)
    w.nuc_median_spin.setValue(1)
    w._run_blocking("nucleus")  # test-only synchronous entry point

    assert captured["prob_path"].endswith("nucleus_prob_3dt.tif")
    assert captured["dp_path"].endswith("nucleus_dp_3dt.tif")
    assert captured["contours_out"].endswith("nucleus_contours.tif")
    assert captured["foreground_out"].endswith("nucleus_foreground.tif")
    assert captured["smoothing_sigma"] == 2.0
    assert captured["median_radius"] == 1
    assert captured["foreground_z_reduction"] == "mean"
    assert captured["contour_z_reduction"] == "mean"
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/napari/test_divergence_maps_widget.py::test_run_invokes_build_divergence_maps -v`
Expected: `NotImplementedError: Wired in Task 7`.

- [ ] **Step 3: Wire run logic**

In `src/cellflow/napari/divergence_maps_widget.py`:

3a. Add at the top of the file (under existing imports):

```python
import threading
from napari.qt.threading import thread_worker
from cellflow.segmentation import CancelledError
from cellflow.segmentation.divergence_maps import build_divergence_maps
```

3b. Replace the stub `_on_run` with the full implementation. Find the `_on_run` method and replace it (and add the helpers below it):

```python
    def _on_run(self, channel: Literal["nucleus", "cell"]) -> None:
        if self._running_stage is not None:
            self._on_cancel()
            return
        self._start_worker(channel)

    def _start_worker(self, channel: Literal["nucleus", "cell"]) -> None:
        prob_path, dp_path, contours_out, fg_out = self._channel_paths(channel)
        if prob_path is None:
            self._set_status("No project open.")
            return
        for p in (prob_path, dp_path):
            if not p.exists():
                self._set_status(f"Missing: {p}")
                return
        params = self._channel_state("nuc" if channel == "nucleus" else "cell")
        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        self._running_stage = channel
        self._set_button_running(channel)

        def _done(report) -> None:
            self._worker = None
            self._cancel_event = None
            self._running_stage = None
            self._set_button_idle()
            self._progress_bar_hide()
            self._set_status(
                f"{channel.title()} divergence maps built ({report.frames} frames)."
            )
            self._files_widget.refresh(self._pos_dir)

        @thread_worker(
            connect={"yielded": self._on_yield, "returned": _done,
                     "errored": self._on_errored},
        )
        def _worker():
            import queue as _queue
            msg_queue: _queue.SimpleQueue = _queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _progress_cb(done: int, total: int, msg: str) -> None:
                msg_queue.put((done, total, msg))

            def _run() -> None:
                try:
                    result_holder.append(build_divergence_maps(
                        prob_path, dp_path, contours_out, fg_out,
                        foreground_z_reduction=params["foreground_z_reduction"],
                        contour_z_reduction=params["contour_z_reduction"],
                        smoothing_sigma=params["smoothing_sigma"],
                        median_radius=params["median_radius"],
                        progress_cb=_progress_cb,
                        cancel=cancel_event.is_set,
                    ))
                except Exception as e:
                    exc_holder.append(e)

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            yield (0, 1, f"Starting {channel} divergence maps…")
            while t.is_alive() or not msg_queue.empty():
                try:
                    yield msg_queue.get_nowait()
                except _queue.Empty:
                    t.join(timeout=0.05)
            if exc_holder:
                raise exc_holder[0]
            return result_holder[0]

        self._worker = _worker()

    def _run_blocking(self, channel: Literal["nucleus", "cell"]) -> None:
        """Synchronous test helper — runs build_divergence_maps in this thread."""
        prob_path, dp_path, contours_out, fg_out = self._channel_paths(channel)
        params = self._channel_state("nuc" if channel == "nucleus" else "cell")
        contours_out.parent.mkdir(parents=True, exist_ok=True)
        build_divergence_maps(
            prob_path, dp_path, contours_out, fg_out,
            foreground_z_reduction=params["foreground_z_reduction"],
            contour_z_reduction=params["contour_z_reduction"],
            smoothing_sigma=params["smoothing_sigma"],
            median_radius=params["median_radius"],
        )

    def _channel_paths(self, channel: Literal["nucleus", "cell"]):
        paths = self._paths()
        if paths is None:
            return None, None, None, None
        if channel == "nucleus":
            return paths.prob, paths.dp, paths.nucleus_contours, paths.nucleus_foreground
        return paths.cell_prob, paths.cell_dp, paths.cell_contours, paths.cell_foreground

    def _set_status(self, msg: str) -> None:
        self.status_lbl.setText(msg)
        self.status_lbl.setVisible(bool(msg))

    def _progress_bar_hide(self) -> None:
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def _set_button_running(self, channel: Literal["nucleus", "cell"]) -> None:
        run_btn = self.nucleus_run_btn if channel == "nucleus" else self.cell_run_btn
        other_btn = self.cell_run_btn if channel == "nucleus" else self.nucleus_run_btn
        run_btn.setText("✕")
        run_btn.setToolTip("Cancel.")
        other_btn.setEnabled(False)

    def _set_button_idle(self) -> None:
        for btn, tip in (
            (self.nucleus_run_btn, "Build nucleus divergence maps."),
            (self.cell_run_btn, "Build cell divergence maps."),
        ):
            btn.setText("▶")
            btn.setToolTip(tip)
            btn.setEnabled(self._pos_dir is not None)

    def _on_yield(self, payload) -> None:
        if not isinstance(payload, tuple):
            self._set_status(str(payload))
            return
        done, total, msg = payload
        self._progress(done, total, msg)

    def _on_errored(self, exc: Exception) -> None:
        self._worker = None
        self._cancel_event = None
        self._running_stage = None
        self._set_button_idle()
        self._progress_bar_hide()
        if isinstance(exc, CancelledError):
            self._set_status("Cancelled.")
            return
        logger.exception("Divergence-maps worker error", exc_info=exc)
        self._set_status(f"Error: {exc}")

    def _on_cancel(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._worker is not None and hasattr(self._worker, "quit"):
            self._worker.quit()
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/napari/test_divergence_maps_widget.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/divergence_maps_widget.py tests/napari/test_divergence_maps_widget.py
git commit -m "feat(napari): wire run/cancel/progress in DivergenceMapsWidget"
```

---

## Task 8: Embed `DivergenceMapsWidget` in `main_widget.py`

**Files:**
- Modify: `src/cellflow/napari/main_widget.py`

- [ ] **Step 1: Add import**

In the top of `src/cellflow/napari/main_widget.py`, after `from cellflow.napari.cellpose_widget import CellposeWidget`, add:

```python
from cellflow.napari.divergence_maps_widget import DivergenceMapsWidget
```

- [ ] **Step 2: Construct the widget + section**

In `__init__`, after the block that creates `self.cellpose_section` (the `CollapsibleSection("Cellpose", …)`), add:

```python
        self._divergence_maps_widget = DivergenceMapsWidget(self.viewer)
        self.divergence_maps_section = CollapsibleSection(
            "Divergence Maps",
            self._divergence_maps_widget,
            expanded=False,
            accent_color=stage_accent("cellpose"),
        )
```

- [ ] **Step 3: Add the section to the scroll layout**

Find `self.scroll_layout.addWidget(self.cellpose_section)` and add the next line below it:

```python
        self.scroll_layout.addWidget(self.divergence_maps_section)
```

- [ ] **Step 4: Set the section's status**

In the `for section in (...)` block that calls `section.set_status("not_started")`, add `self.divergence_maps_section,` to the tuple. The final tuple should read:

```python
        for section in (
            self.data_section,
            self.cellpose_section,
            self.divergence_maps_section,
            self.nucleus_section,
            self.cell_section,
            self.contact_analysis_section,
        ):
            section.set_status("not_started")
```

- [ ] **Step 5: Wire refresh + get_state/set_state**

Find `self._cellpose_widget.refresh(pos_dir)` in `_refresh_all` and add immediately after:

```python
        self._divergence_maps_widget.refresh(pos_dir)
```

In `get_state` (around the dict containing `"cellpose": …`), add:

```python
            "divergence_maps": self._divergence_maps_widget.get_state(),
```

In `set_state`, after the `cellpose` block, add:

```python
        if "divergence_maps" in state:
            self._divergence_maps_widget.set_state(state["divergence_maps"])
```

Also find the `pipeline_status_from_files` block (~lines 336–347) and add a line for the new section. Inspect what other sections call there; add an equivalent invocation:

```python
        pipeline_status_from_files(
            self.divergence_maps_section,
            self._divergence_maps_widget.output_files_tracker,
            done_group="Outputs",
        )
```

(Place it after the `cellpose_section` call. If `pipeline_status_from_files` has a different signature in this codebase, match the others' invocations exactly.)

- [ ] **Step 6: Smoke-check importability**

Run:

```bash
python -c "from cellflow.napari.main_widget import CellFlowMainWidget; print('ok')"
```

Expected: `ok`.

- [ ] **Step 7: Run main widget tests**

Run: `pytest tests/napari/test_main_widget_cellpose_integration.py -v`
Expected: pass (or update those tests in Task 13 if they assert the section list).

- [ ] **Step 8: Commit**

```bash
git add src/cellflow/napari/main_widget.py
git commit -m "feat(napari): wire DivergenceMapsWidget into CellFlowMainWidget"
```

---

## Task 9: Trim `nucleus_pipeline_widget.py` — remove averaged-maps build step

The remaining `_on_build_segmentation_inputs` should only run `write_ultrack_source_stacks` against the existing `1_cellpose/nucleus_{contours,foreground}.tif`. The cellprob-threshold / z-indices controls go away with this trim.

**Files:**
- Modify: `src/cellflow/napari/nucleus_pipeline_widget.py`
- Modify: `tests/napari/test_nucleus_pipeline_widget.py`

- [ ] **Step 1: Read the current file**

(Open `src/cellflow/napari/nucleus_pipeline_widget.py` so the Edit tool can operate.)

- [ ] **Step 2: Drop the legacy imports**

Replace the import block:

```python
from cellflow.segmentation import (
    CancelledError,
    build_consensus_boundary,
    build_nucleus_averaged_maps,
)
```

with:

```python
from cellflow.segmentation import CancelledError
```

- [ ] **Step 3: Remove the preview button + handler**

Find `self.seg_preview_btn = _tool_btn(...)` (around line 96) and delete the assignment and its tooltip lines. Delete the line `self.preview_contour_btn = self.seg_preview_btn`. Delete the `self.seg_preview_btn.clicked.connect(...)` line. In `build_pipeline_block`, drop `self.seg_preview_btn,` from the `_stage_row` call for "Ultrack Inputs".

- [ ] **Step 4: Delete `_on_preview_contour_maps`, `_on_build_nucleus_maps`, `_on_build_contour_maps`**

Delete these three methods entirely (lines ~527–743 in the original file).

- [ ] **Step 5: Replace `_on_build_segmentation_inputs` with a sources-only version**

Replace the existing `_on_build_segmentation_inputs` method body with:

```python
    def _on_build_segmentation_inputs(self) -> None:
        pos_dir = self._pos_dir
        if pos_dir is None:
            self._status("No project open."); return
        paths = self._paths
        if paths is None:
            self._status("No project open."); return
        contours_path = paths.nucleus_contours
        score_path = paths.nucleus_foreground
        if not contours_path.exists():
            self._status(
                "Missing: nucleus_contours.tif — build divergence maps first."
            ); return
        if not score_path.exists():
            self._status(
                "Missing: nucleus_foreground.tif — build divergence maps first."
            ); return
        contour_sources_path = paths.contour_sources
        foreground_sources_path = paths.foreground_sources

        try:
            contour_thresholds = self._source_contour_thresholds_from_controls()
            foreground_thresholds = self._source_foreground_thresholds_from_controls()
        except ValueError as exc:
            self._status(str(exc)); return

        cancel_event = threading.Event()
        self._contour_cancel = cancel_event

        def _done(result):
            n_sources = result
            self._contour_worker = None
            self._contour_cancel = None
            self._clear_progress()
            self._refresh_files_callback(pos_dir)
            self._status(f"Ultrack source stacks built ({n_sources} sources).")
            self._set_running_stage(None)

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": self._on_contour_worker_error,
        })
        def _worker():
            yield (0, 1, "Building Ultrack source stacks…")
            metadata = write_ultrack_source_stacks(
                contours_path,
                score_path,
                contour_sources_path,
                foreground_sources_path,
                contour_thresholds=contour_thresholds,
                foreground_thresholds=foreground_thresholds,
                cancel=cancel_event.is_set,
            )
            yield (1, 1, "Saved Ultrack source stacks.")
            return len(metadata)

        n_sources = len(contour_thresholds) * len(foreground_thresholds)
        self._status(f"Building Ultrack source stacks ({n_sources} sources)…")
        self._set_running_stage("seg")
        self._contour_worker = _worker()
```

- [ ] **Step 6: Drop the cellprob/z-indices helpers**

Delete the methods `_map_cellprob_thresholds_from_controls` and `_map_z_indices_from_controls`. Also delete the (now-unused) `_nucleus_zavg_path` and `_ensure_nucleus_zavg_layer` methods and the `_NUC_ZAVG_LAYER` constant at the top.

- [ ] **Step 7: Update the contours/foreground accessor methods**

Replace `_contours_path` and `_foreground_scores_path` (now they read from new locations via the renamed `_paths` properties):

```python
    def _contours_path(self) -> Path | None:
        return self._paths.nucleus_contours if self._paths else None

    def _foreground_path(self) -> Path | None:
        return self._paths.nucleus_foreground if self._paths else None
```

Grep within this file for any remaining `_foreground_scores_path` references and rename them to `_foreground_path`. (There are call sites in `_on_run_db_generation` and `_on_run_ultrack`.)

- [ ] **Step 8: Update the existing test mocks**

In `tests/napari/test_nucleus_pipeline_widget.py`:

8a. Remove the mock entries for `apply_gamma`, `build_nucleus_averaged_maps`, `build_consensus_boundary` from the monkeypatch dict (the `cellflow.segmentation` patch block around line 113).

8b. Delete the test `test_build_contour_maps_calls_write_source_stacks` (lines ~386–425) — the build-contour-maps codepath is gone.

8c. Delete the assertions that check `_on_build_contour_maps` and `_on_preview_contour_maps` exist (lines ~230–231).

8d. Update file-setup in tests that write `2_nucleus/foreground_scores.tif` to write `1_cellpose/nucleus_foreground.tif` instead, and `2_nucleus/contours.tif` to `1_cellpose/nucleus_contours.tif`. Look for these lines and update each (search for `2_nucleus/foreground_scores.tif`, `2_nucleus/contours.tif`).

- [ ] **Step 9: Run the test file**

Run: `pytest tests/napari/test_nucleus_pipeline_widget.py -v`
Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/cellflow/napari/nucleus_pipeline_widget.py tests/napari/test_nucleus_pipeline_widget.py
git commit -m "refactor(nucleus): drop averaged-maps build from pipeline widget"
```

---

## Task 10: Trim `nucleus_segmentation_inputs_widget.py` controls

**Files:**
- Modify: `src/cellflow/napari/nucleus_segmentation_inputs_widget.py`
- Modify: `src/cellflow/napari/_thresholds.py`
- Modify: `tests/napari/test_nucleus_tracking_inputs_widget.py` (if it references the removed sliders)

- [ ] **Step 1: Drop the cellprob and z controls**

In `src/cellflow/napari/nucleus_segmentation_inputs_widget.py`, delete:

- the `map_cellprob_range` / `map_cellprob_step_spin` block (lines 38–45)
- the `map_z_range` / `map_z_step_spin` block (lines 65–74)
- the matching `RangeThumbProxy` aliases (`map_cellprob_min_spin`, `map_cellprob_max_spin`, `map_z_start_spin`, `map_z_stop_spin`)
- the `Averaged Map` and `Z Slices` grid sections (the `add_section_header(...)` + `add_section_pair_row(...)` lines that reference those widgets)
- the `set_z_extent` method (no longer used)

- [ ] **Step 2: Drop the unused threshold helpers**

In `src/cellflow/napari/_thresholds.py`, delete the functions `map_cellprob_thresholds` and `map_z_indices`. Run `grep -rn 'map_cellprob_thresholds\|map_z_indices' src tests` to confirm they have no remaining references; if any are surfaced, remove those too.

- [ ] **Step 3: Run tests**

Run: `pytest tests/napari/test_nucleus_tracking_inputs_widget.py tests/napari/test_nucleus_pipeline_widget.py -v`

If a test references `map_cellprob_min_spin`/`map_cellprob_max_spin`/`map_z_start_spin`/`map_z_stop_spin`/`set_z_extent`, delete that assertion. The intent of those tests is the source-sweep controls, which are unaffected.

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/cellflow/napari/nucleus_segmentation_inputs_widget.py src/cellflow/napari/_thresholds.py tests/napari/test_nucleus_tracking_inputs_widget.py
git commit -m "refactor(nucleus): drop cellprob/z-indices controls"
```

---

## Task 11: Update remaining widgets to read from new paths

**Files:**
- Modify: `src/cellflow/napari/radial_refinement_widget.py`
- Modify: `src/cellflow/napari/data_panel_widget.py`
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `src/cellflow/napari/nucleus_correction_widget.py`
- Modify: `src/cellflow/napari/cell_correction_widget.py`
- Modify: `src/cellflow/napari/cell_workflow_widget.py`
- Modify: `src/cellflow/napari/cellpose_widget.py`
- Modify: `src/cellflow/tracking_ultrack/reseed.py`

- [ ] **Step 1: `radial_refinement_widget.py`**

Replace these two methods:

```python
    def _contours_path(self) -> Path | None:
        d = self._pos_dir()
        return d / "1_cellpose" / "nucleus_contours.tif" if d else None

    def _fg_path(self) -> Path | None:
        d = self._pos_dir()
        return d / "1_cellpose" / "nucleus_foreground.tif" if d else None
```

Also grep within the file for the string `foreground_scores` (it appears in a status message) and change to `foreground`.

- [ ] **Step 2: `data_panel_widget.py`**

Replace the `Cellpose` and `Nucleus Workflow` entries in `_TRACKED_FILE_GROUPS` with:

```python
    ("Cellpose", [
        ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
        ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
        ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
    ]),
    ("Divergence Maps", [
        ("1_cellpose/nucleus_contours.tif", "Nucleus contours"),
        ("1_cellpose/nucleus_foreground.tif", "Nucleus foreground"),
        ("1_cellpose/cell_contours.tif", "Cell contours"),
        ("1_cellpose/cell_foreground.tif", "Cell foreground"),
    ]),
    ("Nucleus Workflow", [
        ("2_nucleus/contour_sources.tif", "Contour sources"),
        ("2_nucleus/foreground_sources.tif", "Foreground sources"),
        ("2_nucleus/ultrack_workdir/data.db", "Ultrack DB"),
        ("2_nucleus/tracked_labels.tif", "Tracked labels"),
    ]),
```

- [ ] **Step 3: `nucleus_workflow_widget.py`**

Replace the `Inputs` and `Intermediates` lists in the `PipelineFilesWidget(...)` config (lines ~81–95) with:

```python
                ("Inputs", [
                    ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
                    ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
                    ("1_cellpose/nucleus_contours.tif", "Nucleus contours"),
                    ("1_cellpose/nucleus_foreground.tif", "Nucleus foreground"),
                ]),
                ("Intermediates", [
                    ("2_nucleus/contour_sources.tif", "Contour sources"),
                    ("2_nucleus/foreground_sources.tif", "Foreground sources"),
                    ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
                ]),
```

Also delete the `_nucleus_prob_zavg_path` property (lines ~459–461) and its references, and the `_contour_maps_path`/`_foreground_scores_path` property bodies — update them to return `self.nucleus_pipeline_widget._contours_path()` / `._foreground_path()`. Update the `_connect_signals` block that asserts `_on_build_contour_maps`/`_on_preview_contour_maps` exist (lines 257–258): delete those two lines.

- [ ] **Step 4: `nucleus_correction_widget.py`**

Replace `_cell_prob_zavg_path` and `_nucleus_prob_zavg_path` (lines ~498–502) with:

```python
    def _cell_foreground_path(self):
        return self._paths.cell_foreground if self._paths else None

    def _nucleus_foreground_path(self):
        return self._paths.nucleus_foreground if self._paths else None
```

Grep within the file for `_cell_prob_zavg_path` and `_nucleus_prob_zavg_path` callers (e.g. lines ~680, ~685) and rename to the new methods.

- [ ] **Step 5: `cell_correction_widget.py`**

Replace `_cell_prob_zavg_path` and `_nuc_prob_zavg_path` (lines ~469–473) with:

```python
    def _cell_foreground_path(self) -> Path | None:
        return self._p("1_cellpose", "cell_foreground.tif")

    def _nuc_foreground_path(self) -> Path | None:
        return self._p("1_cellpose", "nucleus_foreground.tif")
```

Update both call sites (`czp, nzp = self._cell_prob_zavg_path(), self._nuc_prob_zavg_path()` on lines ~403 and ~527) to call the new methods.

- [ ] **Step 6: `cell_workflow_widget.py`**

In the `PipelineFilesWidget` config, replace the two `*_prob_zavg.tif` entries (~lines 117, 119) with:

```python
                    ("1_cellpose/cell_foreground.tif", "Cell foreground"),
                    ("1_cellpose/nucleus_foreground.tif", "Nucleus foreground"),
```

- [ ] **Step 7: `cellpose_widget.py`**

In `_PIPELINE_FILES`, delete the two `*_prob_zavg.tif` lines (45 and 48). The `Outputs` group becomes:

```python
    ("Outputs", [
        ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
        ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
        ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
    ]),
```

Delete the import `from cellflow.napari.cellpose_zavg_viz_widget import CellposeZavgVizWidget`, the line `self.zavg_viz_widget = CellposeZavgVizWidget()`, and the `root.addWidget(self.zavg_viz_widget)` line. Find every reference to `self.zavg_viz_widget` (search `zavg_viz_widget` in the file) and delete each.

- [ ] **Step 8: `tracking_ultrack/reseed.py`**

In the `resolve_with_canonical_segment` docstring (line ~520), change:

```
      new: foreground_masks + contour_maps → ultrack.segment → inject → score → link → solve
```

to:

```
      new: foreground_masks + contours → ultrack.segment → inject → score → link → solve
```

(Argument names already say `contour_maps_path` and that's fine — they're internal kwargs accepted by `build_ultrack_database`, not file conventions; no rename required.)

- [ ] **Step 9: Run the test suite**

Run: `pytest tests/napari -v -x`

Some tests will fail because they still set up `*_prob_zavg.tif` or `foreground_scores.tif` on disk. Update them as you encounter:

- `tests/napari/test_nucleus_correction_widget.py`: rename `_cell_prob_zavg_path` / `_nucleus_prob_zavg_path` to `_cell_foreground_path` / `_nucleus_foreground_path` (lines 142–143, 393–469). On-disk setup writes to `1_cellpose/cell_foreground.tif` / `1_cellpose/nucleus_foreground.tif`. Drop the `apply_gamma`/`build_consensus_boundary`/`build_nucleus_averaged_maps` mock entries (lines 100–102).
- `tests/napari/test_cell_correction_widget.py`: rename to `cell_foreground.tif` / `nucleus_foreground.tif` (lines 166–207).
- `tests/napari/test_cell_workflow_widget.py`: same rename (lines 528–553).
- `tests/napari/test_nucleus_db_browser_widget.py`: drop the legacy mocks (lines 99–101); update `foreground_scores.tif` paths to `1_cellpose/nucleus_foreground.tif`.
- `tests/napari/test_nucleus_tracking_inputs_widget.py` and `tests/napari/test_nucleus_tracking_correction_layout.py`: drop the legacy mocks (lines 246–248 / 113–115).
- `tests/napari/test_cell_params_widget.py`: drop the `seg_module.apply_gamma = …` patch line (268).
- `tests/napari/test_cellpose_file_contract.py`: delete the four assertions referencing `cell_prob_zavg.tif` / `nucleus_prob_zavg.tif` (lines 23–24, 34–35).
- `tests/napari/test_cellpose_widget.py`: delete the `*_prob_zavg.tif` existence assertions (lines 255, 272).
- `tests/segmentation/test_cellpose_runner.py`: drop the `nucleus_prob_zavg.tif` / `cell_prob_zavg.tif` existence assertions (lines 347, 364).

Re-run `pytest tests/napari tests/segmentation -v -x` after edits; all should pass.

- [ ] **Step 10: Commit**

```bash
git add src/cellflow/napari/*.py src/cellflow/tracking_ultrack/reseed.py tests/
git commit -m "refactor: point all consumers at 1_cellpose/{channel}_{contours,foreground}.tif"
```

---

## Task 12: Drop dead legacy code

**Files:**
- Modify: `src/cellflow/segmentation/nucleus_segmentation.py`
- Modify: `src/cellflow/segmentation/__init__.py`
- Modify: `src/cellflow/segmentation/cellpose_runner.py`
- Delete: `src/cellflow/segmentation/cellpose_probability_zavg.py`
- Delete: `src/cellflow/napari/cellpose_zavg_viz_widget.py`
- Delete: `scripts/precompute_cellpose_probability_zavgs.py`
- Delete: `tests/segmentation/test_cellpose_probability_zavg.py`
- Delete: `tests/segmentation/test_nucleus_averaged_maps.py`
- Delete: `tests/segmentation/test_foreground_masks.py`
- Delete: `tests/napari/test_cellpose_zavg_viz_widget.py`

- [ ] **Step 1: Trim `nucleus_segmentation.py`**

In `src/cellflow/segmentation/nucleus_segmentation.py`, delete:

- `apply_gamma` (lines 22–28)
- `NucleusAveragedMapsReport` dataclass (lines 50–58)
- `build_consensus_boundary` (lines 120–183)
- `_as_tzyx`, `_as_tzcyx`, `_normalize_cellprob_thresholds`, `_normalize_z_indices` (lines 186–229)
- `build_nucleus_averaged_maps` (lines 232–295)

Keep: `CancelledError`, `_check_cancel`, `_LABEL_DTYPE`, `ContourWatershedParams`, `_remove_small_labels`, `_remove_low_circularity_labels`, `_fill_and_close_labels`, `compute_contour_watershed`.

Update the imports at the top of the file: remove the unused `asdict` import line if it becomes dead (it's still used by `ContourWatershedParams.to_dict`, so keep it).

- [ ] **Step 2: Trim segmentation `__init__.py`**

In `src/cellflow/segmentation/__init__.py`, change the nucleus_segmentation import block to:

```python
from cellflow.segmentation.nucleus_segmentation import (
    CancelledError,
    ContourWatershedParams,
    compute_contour_watershed,
)
```

Drop `NucleusAveragedMapsReport`, `build_consensus_boundary`, `build_nucleus_averaged_maps` from `__all__`.

Keep the local `apply_gamma` function defined at the bottom of `__init__.py` (line 62) — it's still used by `cell_workflow_widget` and `tests/tracking/test_correction.py`.

- [ ] **Step 3: Stop writing zavg in `cellpose_runner.write_outputs`**

In `src/cellflow/segmentation/cellpose_runner.py`, replace lines 247–251 (the zavg write):

```python
    zavg_path = output_dir / f"{channel}_prob_zavg.tif"
    tifffile.imwrite(str(prob_path), prob_3dt.astype(np.float32), compression="zlib")
    tifffile.imwrite(str(dp_path), dp_3dt.astype(np.float32), compression="zlib")
    zavg = prob_3dt.mean(axis=1, dtype=np.float32).astype(np.float32)
    tifffile.imwrite(str(zavg_path), zavg, compression="zlib")
```

with:

```python
    tifffile.imwrite(str(prob_path), prob_3dt.astype(np.float32), compression="zlib")
    tifffile.imwrite(str(dp_path), dp_3dt.astype(np.float32), compression="zlib")
```

Also update the docstring (lines 235–237) to drop the z-avg reference:

```python
    """Write the two canonical TIFFs under output_dir.

    Writes ``{channel}_prob_3dt.tif`` and ``{channel}_dp_3dt.tif``.
    """
```

- [ ] **Step 4: Delete obsolete files**

```bash
rm src/cellflow/segmentation/cellpose_probability_zavg.py
rm src/cellflow/napari/cellpose_zavg_viz_widget.py
rm scripts/precompute_cellpose_probability_zavgs.py
rm tests/segmentation/test_cellpose_probability_zavg.py
rm tests/segmentation/test_nucleus_averaged_maps.py
rm tests/segmentation/test_foreground_masks.py
rm tests/napari/test_cellpose_zavg_viz_widget.py
```

- [ ] **Step 5: Sweep remaining references**

Run:

```bash
grep -rn 'foreground_scores\|contour_maps\|prob_zavg\|build_nucleus_averaged_maps\|build_consensus_boundary\|cellpose_probability_zavg\|precompute_cellpose_probability_zavgs\|CellposeZavgVizWidget' \
    --include='*.py' --include='*.yaml' src tests scripts | grep -v __pycache__
```

Expected output: no matches in `src/` or `tests/`. Matches in `scripts/experiment_*.py` (research scripts that read existing position directories) are acceptable but should be skimmed — if any are still being run, update them to the new file names; otherwise leave them alone since they reference fixed local data paths.

- [ ] **Step 6: Run the full suite**

Run: `pytest -x`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: drop legacy zavg + consensus-boundary code paths"
```

---

## Task 13: Update integration tests + smoke-test in napari

**Files:**
- Modify: `tests/napari/test_main_widget_cellpose_integration.py` (if it references the cellpose zavg viz widget or the section list)
- Modify: any other test surfaced by the full suite

- [ ] **Step 1: Run integration tests**

Run: `pytest tests/napari/test_main_widget_cellpose_integration.py -v`

If a test asserts that `_divergence_maps_widget` does *not* exist, or hard-codes the section count, update it to include the new section. Add a positive assertion that `main.divergence_maps_section` exists and that `main._divergence_maps_widget` has the `refresh` method.

- [ ] **Step 2: Run the full suite**

Run: `pytest -x`
Expected: all green.

- [ ] **Step 3: Manual UI smoke test**

Launch napari with the plugin loaded and open a position directory that already has `1_cellpose/nucleus_prob_3dt.tif` and `1_cellpose/nucleus_dp_3dt.tif`:

```bash
python -m napari --with cellflow
```

Walk through:

1. Project Status panel shows the new `Divergence Maps` group; old `Nucleus prob z-avg` rows are gone.
2. Cellpose section has no z-avg button.
3. New `Divergence Maps` section is collapsed by default. Expanding shows nucleus + cell rows with ⚙ / ▶.
4. ⚙ on the nucleus row reveals four params: foreground z-reduction (default `mean`), contour z-reduction (default `mean`), smoothing sigma (`1.00`), median radius (`0`).
5. Clicking ▶ on the nucleus row writes `1_cellpose/nucleus_contours.tif` and `nucleus_foreground.tif`; status bar reports `(N frames)` on completion. The button toggles to ✕ during the run.
6. Same on the cell row.
7. In Nucleus Segmentation & Tracking, the Ultrack Inputs row has no ▷ preview button; clicking ▶ builds source stacks from the new files.
8. Solve and tracking still complete end-to-end.

Document anything that broke in a follow-up TODO.

- [ ] **Step 4: Commit any test fixes**

```bash
git add tests
git commit -m "test: update integration tests for divergence-maps layout"
```

---

## Self-review checklist

Before declaring the plan complete, verify the spec is fully covered:

- [x] **Compute**: `foreground = reduce_z(sigmoid(prob))` — Task 1.
- [x] **Compute**: filter (median → gaussian) → divergence → clip → reduce — Tasks 2–3.
- [x] **Compute**: `build_divergence_maps` writes `T × Y × X` float32 stacks with progress + cancel — Task 4.
- [x] **Outputs**: `1_cellpose/{nucleus,cell}_{contours,foreground}.tif` — Tasks 4, 5, 7.
- [x] **UI**: nucleus + cell rows, sequential, same progress pattern as Cellpose — Tasks 6–8.
- [x] **UI params**: `foreground_z_reduction`, `contour_z_reduction`, `smoothing_sigma`, `median_radius` per channel with the spec's defaults and ranges — Task 6.
- [x] **Dropped artifacts**: `*_prob_zavg.tif`, `2_nucleus/contours.tif`, `2_nucleus/foreground_scores.tif`, legacy `2_nucleus/contour_maps.tif` — Tasks 5, 11, 12.
- [x] **Dropped code**: `build_nucleus_averaged_maps`, `build_consensus_boundary`, `apply_gamma` (the copy inside `nucleus_segmentation.py`) — Task 12.
- [x] **Dropped widgets/scripts**: `cellpose_zavg_viz_widget`, `cellpose_probability_zavg`, `precompute_cellpose_probability_zavgs.py` — Task 12.
- [x] **Updated**: `_paths.py`, `data_panel_widget`, `radial_refinement_widget`, `reseed.py` — Tasks 5, 11.
- [x] **Grep sweep**: surfaces all `foreground_scores|contour_maps|prob_zavg` call sites — Task 12 step 5.
- [x] **Deferred**: foreground gating; cell-workflow swap. Explicitly out of scope (spec).

Type consistency: `ZReduction` and `DivergenceMapsReport.frames` are consistent across Tasks 1–7; `_paths.NucleusArtifactPaths` exposes `nucleus_contours`, `nucleus_foreground`, `cell_contours`, `cell_foreground` (and the no-prefix aliases `contours`/`foreground`) consumed by Tasks 6–11.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-18-divergence-maps.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review between tasks.
2. **Inline Execution** — execute tasks in this session with batched checkpoints.

Which approach?
