# Cell Segmentation — Contour Maps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the Cell Workflow section into "Cell Segmentation" with a single "Contour Maps" subwidget that implements mean-Z-projection boundary building, deprecating all old seeded-watershed cell logic.

**Architecture:** A new backend function `build_mean_z_consensus_boundary` encapsulates the cell-specific algorithm (mean-project prob and dp across Z first, then sweep threshold×gamma). The widget (`CellWorkflowWidget`) is fully rewritten with only a Contour Maps section — no hypothesis generation, no tracking. The old `SeededWatershedParams` and `compute_seeded_watershed` are deprecated in-place. All old widget tests are replaced.

**Tech Stack:** Python, Qt (qtpy), napari, numpy, tifffile, cellpose, scikit-image (find_boundaries), scipy

---

## File Map

| File | Action | What changes |
|---|---|---|
| `src/cellflow/segmentation/__init__.py` | Modify | Add `build_mean_z_consensus_boundary`; deprecate `SeededWatershedParams` + `compute_seeded_watershed` |
| `src/cellflow/napari/cell_workflow_widget.py` | Rewrite | Replace all content with new Contour Maps–only widget |
| `src/cellflow/napari/main_widget.py` | Modify | Rename section label `"4. Cell Workflow"` → `"4. Cell Segmentation"` |
| `tests/segmentation/test_label_postprocessing.py` | Modify | Add tests for `build_mean_z_consensus_boundary` |
| `tests/napari/test_cell_workflow_preview.py` | Rewrite | Replace old seeded-watershed tests with Contour Maps widget tests |

---

## Task 1: Backend — `build_mean_z_consensus_boundary`

**Files:**
- Modify: `src/cellflow/segmentation/__init__.py`
- Test: `tests/segmentation/test_label_postprocessing.py`

### Algorithm recap

Unlike `build_consensus_boundary` (which runs Cellpose masks per z-slice × threshold), this function:
1. Mean-projects `prob_zyx` across Z per gamma to get `projected_prob (Y,X)`
2. Mean-projects `dp_zcyx` across Z once to get `projected_dp (C,Y,X)`
3. Runs `compute_masks(projected_dp, projected_prob, threshold)` for each gamma×threshold pair
4. Averages `find_boundaries` over all combos
5. Returns `(boundary_yx, foreground_yx)`

- [ ] **Step 1: Write the failing test**

Append to `tests/segmentation/test_label_postprocessing.py`:

```python
def test_build_mean_z_consensus_boundary_returns_correct_shapes(monkeypatch):
    import numpy as np
    from unittest.mock import patch

    rng = np.random.default_rng(0)
    n_z, n_y, n_x = 3, 8, 10
    prob_zyx = rng.standard_normal((n_z, n_y, n_x)).astype(np.float32)
    dp_zcyx = rng.standard_normal((n_z, 2, n_y, n_x)).astype(np.float32)
    thresholds = [-2.0, -1.0, 0.0]
    gammas = [0.8, 1.0]

    call_log = []

    def fake_compute_masks(dp, prob, cellprob_threshold, flow_threshold, niter, do_3D, device):
        call_log.append((dp.shape, prob.shape, cellprob_threshold))
        masks = np.zeros(prob.shape, dtype=np.uint32)
        masks[2:5, 3:8] = 1
        return masks

    with patch("cellpose.dynamics.compute_masks", fake_compute_masks):
        from cellflow.segmentation import build_mean_z_consensus_boundary
        boundary, foreground = build_mean_z_consensus_boundary(
            prob_zyx, dp_zcyx, thresholds, gammas
        )

    assert boundary.shape == (n_y, n_x), f"Expected ({n_y},{n_x}), got {boundary.shape}"
    assert foreground.shape == (n_y, n_x)
    assert boundary.dtype == np.float32
    assert foreground.dtype == np.float32
    assert 0.0 <= float(boundary.min()) <= float(boundary.max()) <= 1.0
    assert 0.0 <= float(foreground.min()) <= float(foreground.max()) <= 1.0
    # 2 gammas × 3 thresholds = 6 calls, all on 2D projected inputs
    assert len(call_log) == 6
    assert all(dp_shape == (2, n_y, n_x) for dp_shape, _, _ in call_log)
    assert all(prob_shape == (n_y, n_x) for _, prob_shape, _ in call_log)
    assert sorted(set(t for _, _, t in call_log)) == thresholds


def test_build_mean_z_consensus_boundary_single_gamma_default(monkeypatch):
    import numpy as np
    from unittest.mock import patch

    n_z, n_y, n_x = 2, 6, 6
    prob_zyx = np.zeros((n_z, n_y, n_x), dtype=np.float32)
    dp_zcyx = np.zeros((n_z, 2, n_y, n_x), dtype=np.float32)

    call_log = []

    def fake_compute_masks(dp, prob, cellprob_threshold, flow_threshold, niter, do_3D, device):
        call_log.append(cellprob_threshold)
        return np.zeros(prob.shape, dtype=np.uint32)

    with patch("cellpose.dynamics.compute_masks", fake_compute_masks):
        from cellflow.segmentation import build_mean_z_consensus_boundary
        boundary, foreground = build_mean_z_consensus_boundary(
            prob_zyx, dp_zcyx, [-1.0, 0.0]
        )

    # Default gammas=(1.0,) → 1 gamma × 2 thresholds = 2 calls
    assert len(call_log) == 2
    assert boundary.shape == (n_y, n_x)


def test_build_mean_z_consensus_boundary_invokes_mask_callback(monkeypatch):
    import numpy as np
    from unittest.mock import patch

    n_z, n_y, n_x = 2, 5, 5
    prob_zyx = np.zeros((n_z, n_y, n_x), dtype=np.float32)
    dp_zcyx = np.zeros((n_z, 2, n_y, n_x), dtype=np.float32)

    def fake_compute_masks(dp, prob, **kwargs):
        return np.zeros(prob.shape, dtype=np.uint32)

    cb_calls = []

    def my_callback(masks, gamma_idx, thresh_idx):
        cb_calls.append((masks.shape, gamma_idx, thresh_idx))

    with patch("cellpose.dynamics.compute_masks", fake_compute_masks):
        from cellflow.segmentation import build_mean_z_consensus_boundary
        build_mean_z_consensus_boundary(
            prob_zyx, dp_zcyx, [-1.0, 0.0], [1.0, 1.2], mask_callback=my_callback
        )

    # 2 gammas × 2 thresholds → 4 callback invocations
    assert len(cb_calls) == 4
    assert all(shape == (n_y, n_x) for shape, _, _ in cb_calls)
    gamma_idx_vals = sorted(set(gi for _, gi, _ in cb_calls))
    thresh_idx_vals = sorted(set(ti for _, _, ti in cb_calls))
    assert gamma_idx_vals == [0, 1]
    assert thresh_idx_vals == [0, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/aruppel/Projects/CellFlow
python -m pytest tests/segmentation/test_label_postprocessing.py::test_build_mean_z_consensus_boundary_returns_correct_shapes tests/segmentation/test_label_postprocessing.py::test_build_mean_z_consensus_boundary_single_gamma_default tests/segmentation/test_label_postprocessing.py::test_build_mean_z_consensus_boundary_invokes_mask_callback -v 2>&1 | tail -20
```

Expected: FAIL with `ImportError: cannot import name 'build_mean_z_consensus_boundary'`

- [ ] **Step 3: Implement `build_mean_z_consensus_boundary`**

Add the import at the top of `src/cellflow/segmentation/__init__.py` (it already has `from collections.abc import Callable`):

```python
import warnings
```

Add after `build_consensus_boundary` (after line ~381):

```python
def build_mean_z_consensus_boundary(
    prob_zyx: np.ndarray,
    dp_zcyx: np.ndarray,
    cellprob_thresholds: list[float],
    gammas: list[float] = (1.0,),
    *,
    mask_callback: Callable[[np.ndarray, int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build boundary map by mean-projecting prob/dp across Z then sweeping threshold×gamma.

    Unlike build_consensus_boundary (threshold×z-slice), this projects Z first so each
    Cellpose call sees a 2-D image that integrates the full z-stack. Appropriate for
    cells that span many z-planes.

    prob_zyx:            (Z, Y, X) Cellpose probability logits
    dp_zcyx:             (Z, C, Y, X) Cellpose flow fields, C ≥ 2
    cellprob_thresholds: list of cellprob_threshold values to sweep
    gammas:              list of gamma correction values; boundary averaged over all combos
    mask_callback:       optional, called as mask_callback(masks_yx, gamma_idx, thresh_idx)
    Returns: (boundary, foreground) both (Y, X) float32
      boundary   — mean find_boundaries density across all gamma×threshold combinations
      foreground — sigmoid of z-mean gamma-corrected prob, averaged over gammas
    """
    try:
        import torch
        from cellpose.dynamics import compute_masks
        from skimage.segmentation import find_boundaries
    except ImportError as exc:
        raise ImportError("cellpose, torch, and scikit-image required") from exc

    prob_zyx = np.asarray(prob_zyx, dtype=np.float32)
    dp_zcyx = np.asarray(dp_zcyx, dtype=np.float32)

    # Mean-project dp across Z once: (Z, C, Y, X) → (C, Y, X)
    projected_dp = dp_zcyx.mean(axis=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boundary_accum = np.zeros(prob_zyx.shape[1:], dtype=np.float32)
    fg_accum = np.zeros(prob_zyx.shape[1:], dtype=np.float32)
    n_total = 0

    for g_idx, gamma in enumerate(gammas):
        # Apply gamma then mean-project across Z: (Z, Y, X) → (Y, X)
        projected_prob = apply_gamma(prob_zyx, gamma).mean(axis=0)
        fg_accum += 1.0 / (1.0 + np.exp(-projected_prob))

        for t_idx, thresh in enumerate(cellprob_thresholds):
            result = compute_masks(
                projected_dp,
                projected_prob,
                cellprob_threshold=float(thresh),
                flow_threshold=0.0,
                niter=200,
                do_3D=False,
                device=device,
            )
            masks = result[0] if isinstance(result, tuple) else result
            masks = np.asarray(masks, dtype=np.uint32)
            boundary_accum += find_boundaries(masks, mode="inner").astype(np.float32)
            n_total += 1
            if mask_callback is not None:
                mask_callback(masks, g_idx, t_idx)

    n_gammas = len(gammas)
    boundary = boundary_accum / n_total if n_total > 0 else boundary_accum
    foreground = fg_accum / n_gammas if n_gammas > 0 else fg_accum
    return boundary.astype(np.float32), foreground.astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/aruppel/Projects/CellFlow
python -m pytest tests/segmentation/test_label_postprocessing.py::test_build_mean_z_consensus_boundary_returns_correct_shapes tests/segmentation/test_label_postprocessing.py::test_build_mean_z_consensus_boundary_single_gamma_default tests/segmentation/test_label_postprocessing.py::test_build_mean_z_consensus_boundary_invokes_mask_callback -v 2>&1 | tail -10
```

Expected: 3 PASSED

- [ ] **Step 5: Deprecate `SeededWatershedParams` and `compute_seeded_watershed`**

At the top of `src/cellflow/segmentation/__init__.py`, add `import warnings` if not already present.

Add `__post_init__` to `SeededWatershedParams` to warn on instantiation:

```python
@dataclass(frozen=True, slots=True)
class SeededWatershedParams:
    """Parameters for nucleus-seeded watershed cell hypothesis generation.

    .. deprecated::
        The seeded-watershed cell approach has been superseded by the Contour Maps
        workflow. This class will be removed in a future version.
    """

    basin: str = "prob"
    foreground_threshold: float = 0.5
    compactness: float = 0.0

    def __post_init__(self) -> None:
        warnings.warn(
            "SeededWatershedParams is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    def to_dict(self) -> dict[str, object]:
        return {"method": "seeded_watershed", **asdict(self)}
```

Add a deprecation warning at the start of `compute_seeded_watershed`:

```python
def compute_seeded_watershed(
    prob_2d: np.ndarray,
    dp_2d: np.ndarray | None,
    seeds_2d: np.ndarray,
    params: SeededWatershedParams,
) -> np.ndarray:
    """Seeded watershed using nucleus labels as markers for one 2D z-slice.

    .. deprecated::
        Use the Contour Maps workflow in CellWorkflowWidget instead.
    """
    warnings.warn(
        "compute_seeded_watershed is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    from scipy.ndimage import binary_fill_holes
    from skimage.segmentation import watershed
    # ... rest of existing body unchanged ...
```

- [ ] **Step 6: Run full segmentation test suite to verify no regressions**

```bash
cd /home/aruppel/Projects/CellFlow
python -m pytest tests/segmentation/ -v 2>&1 | tail -20
```

Expected: All pass (the deprecation warnings are emitted but don't fail tests).

- [ ] **Step 7: Commit**

```bash
cd /home/aruppel/Projects/CellFlow
git add src/cellflow/segmentation/__init__.py tests/segmentation/test_label_postprocessing.py
git commit -m "feat(segmentation): add build_mean_z_consensus_boundary; deprecate seeded watershed"
```

---

## Task 2: Rewrite `cell_workflow_widget.py`

**Files:**
- Rewrite: `src/cellflow/napari/cell_workflow_widget.py`

The entire file is replaced. It contains only the `CellWorkflowWidget` class with a single "1. Contour Maps" collapsible section.

### Layer name constants

```python
_CONTOUR_LAYER = "Contour Map: Cell"
_CELLPROB_LAYER = "Cellprob Map: Cell"
_CONTOUR_SWEEP_WIDTH = 60
_CONTOUR_SWEEP_MIN_WIDTH = int(_CONTOUR_SWEEP_WIDTH * 0.9)
```

### Path helpers

```python
def _prob_path(self) -> Path | None:
    return self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif" if self._pos_dir else None

def _dp_path(self) -> Path | None:
    return self._pos_dir / "1_cellpose" / "cell_dp_3dt.tif" if self._pos_dir else None

def _contour_maps_path(self) -> Path | None:
    return self._pos_dir / "3_cell" / "contour_maps.tif" if self._pos_dir else None
```

### DP normalization helper (private, used in both preview and build workers)

Reuse `normalize_seeded_watershed_dp_stack` from `cellflow.database.hypotheses` to coerce any Cellpose dp layout → `(T, Z, C, Y, X)`, then extract the y/x channels:

```python
# In the worker:
from cellflow.database.hypotheses import normalize_seeded_watershed_dp_stack
dp_stack = normalize_seeded_watershed_dp_stack(dp_raw, prob_stack.shape)
dp_stack = dp_stack[:, :, :2]   # keep only (y-flow, x-flow)
```

- [ ] **Step 1: Write the full new `cell_workflow_widget.py`**

Replace the entire file with:

```python
"""Cell segmentation widget for CellFlow — Contour Maps subwidget."""
from __future__ import annotations

import logging
import shlex
import sys
import tempfile
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import (
    add_block_button_row,
    block_grid,
    compact_spinbox,
    sweep_parameter_grid,
)

logger = logging.getLogger(__name__)

_CONTOUR_LAYER = "Contour Map: Cell"
_CELLPROB_LAYER = "Cellprob Map: Cell"
_CONTOUR_SWEEP_WIDTH = 60
_CONTOUR_SWEEP_MIN_WIDTH = int(_CONTOUR_SWEEP_WIDTH * 0.9)


class CellWorkflowWidget(QWidget):
    """Cell segmentation — Contour Maps."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._build_worker = None
        self._setup_ui()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────────────────────
    # UI setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        # ── Inputs ────────────────────────────────────────────────────────
        self.input_files = PipelineFilesWidget([
            ("Inputs", [
                ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
                ("1_cellpose/cell_dp_3dt.tif",  "Cell dp 3D+t"),
            ]),
        ])
        layout.addWidget(self.input_files)

        # ── 1. Contour Maps ───────────────────────────────────────────────
        _contour_inner = QWidget()
        contour_lay = QVBoxLayout(_contour_inner)
        contour_lay.setContentsMargins(4, 4, 4, 4)
        contour_lay.setSpacing(4)
        contour_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        cp_params_scroll = QScrollArea()
        cp_params_scroll.setWidgetResizable(True)
        cp_params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        cp_params_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cp_params_scroll.setFrameShape(QFrame.NoFrame)
        cp_params_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        cp_params_widget = QWidget()
        cp_params_widget.setMinimumWidth(520)
        cp_params_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        cp_params_lay = QVBoxLayout(cp_params_widget)
        cp_params_lay.setContentsMargins(0, 0, 0, 0)
        cp_params_lay.setSpacing(4)
        cp_params_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        contour_sweep_grid = sweep_parameter_grid(spin_width=_CONTOUR_SWEEP_WIDTH)

        def _sweep_spin(lo, hi, val, step, decimals=1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return s

        self.cp_min_spin  = _sweep_spin(-20.0, 20.0, -8.0, 1.0)
        self.cp_max_spin  = _sweep_spin(-20.0, 20.0,  0.0, 1.0)
        self.cp_step_spin = _sweep_spin(0.1, 10.0,    1.0, 0.5)
        contour_sweep_grid.addWidget(QLabel("Cellprob:"), 1, 0)
        contour_sweep_grid.addWidget(self.cp_min_spin,  1, 1)
        contour_sweep_grid.addWidget(self.cp_max_spin,  1, 2)
        contour_sweep_grid.addWidget(self.cp_step_spin, 1, 3)
        contour_sweep_grid.setColumnStretch(1, 1)
        contour_sweep_grid.setColumnStretch(2, 1)
        contour_sweep_grid.setColumnStretch(3, 1)

        _gamma_tip = (
            "Gamma correction on Cellpose probability logits before boundary building. "
            "<1 boosts dim signals; >1 suppresses them. 1.0 = no correction. "
            "Contour maps are averaged over all gamma values in [min, max]."
        )
        self.cp_gamma_min_spin  = _sweep_spin(0.05, 5.0, 1.0, 0.05, decimals=2)
        self.cp_gamma_max_spin  = _sweep_spin(0.05, 5.0, 1.0, 0.05, decimals=2)
        self.cp_gamma_step_spin = _sweep_spin(0.05, 2.0, 0.25, 0.05, decimals=2)
        for w in (self.cp_gamma_min_spin, self.cp_gamma_max_spin, self.cp_gamma_step_spin):
            w.setToolTip(_gamma_tip)
        contour_sweep_grid.addWidget(QLabel("Gamma:"), 2, 0)
        contour_sweep_grid.addWidget(self.cp_gamma_min_spin,  2, 1)
        contour_sweep_grid.addWidget(self.cp_gamma_max_spin,  2, 2)
        contour_sweep_grid.addWidget(self.cp_gamma_step_spin, 2, 3)
        cp_params_lay.addLayout(contour_sweep_grid)

        self.preview_contour_btn   = QPushButton("Preview")
        self.build_btn             = QPushButton("Build")
        self.contour_terminal_btn  = QPushButton("Run in Terminal")
        self.cancel_build_btn      = QPushButton("Cancel")
        self.cancel_build_btn.setEnabled(False)

        for btn in (
            self.preview_contour_btn,
            self.build_btn,
            self.contour_terminal_btn,
            self.cancel_build_btn,
        ):
            btn.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        contour_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(
            contour_btn_row,
            0,
            self.preview_contour_btn,
            self.build_btn,
            self.contour_terminal_btn,
            self.cancel_build_btn,
        )
        cp_params_lay.addLayout(contour_btn_row)

        self.contour_input_lbl = QLabel("")
        self.contour_input_lbl.setWordWrap(True)
        cp_params_lay.addWidget(self.contour_input_lbl)

        self.contour_output_lbl = QLabel("")
        self.contour_output_lbl.setWordWrap(True)
        cp_params_lay.addWidget(self.contour_output_lbl)

        self.contour_status_lbl = QLabel("")
        self.contour_status_lbl.setWordWrap(True)
        self.contour_status_lbl.setVisible(False)
        cp_params_lay.addWidget(self.contour_status_lbl)

        self.build_progress_bar = QProgressBar()
        self.build_progress_bar.setRange(0, 100)
        self.build_progress_bar.setValue(0)
        self.build_progress_bar.setVisible(False)

        self.contour_files = PipelineFilesWidget([
            ("", [
                ("3_cell/contour_maps.tif", "Contour maps"),
            ]),
        ])
        cp_params_lay.addWidget(self.build_progress_bar)
        cp_params_lay.addWidget(self.contour_files)
        self._update_contour_status_labels()

        cp_params_scroll.setWidget(cp_params_widget)
        contour_lay.addWidget(cp_params_scroll)
        self.contour_section = CollapsibleSection(
            "1. Contour Maps", _contour_inner, expanded=False
        )
        layout.addWidget(self.contour_section)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.build_btn.clicked.connect(self._on_build_contour_maps)
        self.contour_terminal_btn.clicked.connect(self._on_run_contour_terminal)
        self.cancel_build_btn.clicked.connect(self._on_cancel_build)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self.input_files.refresh(pos_dir)
        self.contour_files.refresh(pos_dir)
        self._update_contour_status_labels()

    def get_state(self) -> dict:
        return {
            "cellprob": {
                "min":        self.cp_min_spin.value(),
                "max":        self.cp_max_spin.value(),
                "step":       self.cp_step_spin.value(),
                "gamma_min":  self.cp_gamma_min_spin.value(),
                "gamma_max":  self.cp_gamma_max_spin.value(),
                "gamma_step": self.cp_gamma_step_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if "cellprob" in state:
            cp = state["cellprob"]
            if "min"        in cp: self.cp_min_spin.setValue(cp["min"])
            if "max"        in cp: self.cp_max_spin.setValue(cp["max"])
            if "step"       in cp: self.cp_step_spin.setValue(cp["step"])
            if "gamma_min"  in cp: self.cp_gamma_min_spin.setValue(cp["gamma_min"])
            if "gamma_max"  in cp: self.cp_gamma_max_spin.setValue(cp["gamma_max"])
            if "gamma_step" in cp: self.cp_gamma_step_spin.setValue(cp["gamma_step"])

    # ──────────────────────────────────────────────────────────────────────────
    # Path helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif" if self._pos_dir else None

    def _dp_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_dp_3dt.tif" if self._pos_dir else None

    def _contour_maps_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "contour_maps.tif" if self._pos_dir else None

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _thresholds(self) -> list[float]:
        lo   = self.cp_min_spin.value()
        hi   = self.cp_max_spin.value()
        step = self.cp_step_spin.value()
        return list(np.arange(lo, hi + step / 2, step))

    def _cp_gammas(self) -> list[float]:
        gmin  = self.cp_gamma_min_spin.value()
        gmax  = self.cp_gamma_max_spin.value()
        gstep = self.cp_gamma_step_spin.value()
        return list(np.arange(gmin, gmax + gstep / 2, gstep))

    def _current_t(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) >= 1 else 0

    # ──────────────────────────────────────────────────────────────────────────
    # Status helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _set_contour_status(self, msg: str) -> None:
        self.contour_status_lbl.setText(msg)
        self.contour_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _update_contour_status_labels(self) -> None:
        if self._pos_dir is None:
            self.contour_input_lbl.setText("Inputs: no project open.")
            self.contour_output_lbl.setText("Outputs: no project open.")
            return
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        prob_ok   = prob_path is not None and prob_path.exists()
        dp_ok     = dp_path   is not None and dp_path.exists()
        self.contour_input_lbl.setText(
            f"Inputs: {'✓' if prob_ok else '✗'} prob  {'✓' if dp_ok else '✗'} dp"
        )
        contour_path = self._contour_maps_path()
        contour_ok   = contour_path is not None and contour_path.exists()
        self.contour_output_lbl.setText(
            f"Outputs: {'✓' if contour_ok else '✗'} contour_maps.tif"
        )

    def _set_build_buttons_running(self, running: bool) -> None:
        self.build_btn.setEnabled(not running)
        self.preview_contour_btn.setEnabled(not running)
        self.contour_terminal_btn.setEnabled(not running)
        self.cancel_build_btn.setEnabled(running)
        self.build_progress_bar.setVisible(running)
        if not running:
            self.build_progress_bar.setValue(0)

    def _on_build_progress(self, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            if total > 0:
                self.build_progress_bar.setRange(0, total)
                self.build_progress_bar.setValue(done)
            self._set_contour_status(msg)
        else:
            self._set_contour_status(str(data))

    def _on_contour_worker_error(self, exc: Exception) -> None:
        self._build_worker = None
        self.build_progress_bar.setVisible(False)
        self._set_build_buttons_running(False)
        self._set_contour_status(f"Error: {exc}")
        logger.exception("Contour worker error", exc_info=exc)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Contour map build
    # ──────────────────────────────────────────────────────────────────────────

    def _load_prob_dp(self, prob_path: Path, dp_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load and normalise prob (T,Z,Y,X) and dp (T,Z,2,Y,X) stacks."""
        from cellflow.database.hypotheses import normalize_seeded_watershed_dp_stack

        prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
        if prob.ndim == 3:
            prob = prob[np.newaxis]   # single-frame TIFF without T axis
        dp_raw = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
        dp = normalize_seeded_watershed_dp_stack(dp_raw, prob.shape)
        dp = dp[:, :, :2]            # keep only y/x flow channels
        return prob, dp

    def _on_preview_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return

        t_frame    = self._current_t()
        thresholds = self._thresholds()
        gammas     = self._cp_gammas()

        def _on_done(result):
            self._build_worker = None
            self._set_build_buttons_running(False)
            boundary, cellprob_zavg, t_idx = result
            # Allocate T-stack with zeros; place boundary at current t
            full = np.zeros((cellprob_zavg.shape[0],) + boundary.shape, dtype=np.float32)
            full[t_idx] = boundary
            if _CELLPROB_LAYER in self.viewer.layers:
                self.viewer.layers[_CELLPROB_LAYER].data = cellprob_zavg
            else:
                self.viewer.add_image(
                    cellprob_zavg, name=_CELLPROB_LAYER,
                    colormap="inferno", blending="additive", visible=True,
                )
            if _CONTOUR_LAYER in self.viewer.layers:
                self.viewer.layers[_CONTOUR_LAYER].data = full
            else:
                self.viewer.add_image(full, name=_CONTOUR_LAYER, colormap="magma", visible=True)
            # jump viewer to the previewed frame
            step = list(self.viewer.dims.current_step)
            if step:
                step[0] = t_idx
                self.viewer.dims.current_step = tuple(step)
            self._set_contour_status(
                f"Preview contour map t={t_idx} — "
                f"{len(thresholds)} cellprob thresholds, {len(gammas)} gamma value(s)"
            )

        @thread_worker(connect={
            "returned": _on_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation import build_mean_z_consensus_boundary

            prob, dp = self._load_prob_dp(prob_path, dp_path)
            n_t  = min(prob.shape[0], dp.shape[0])
            t_idx = min(max(t_frame, 0), n_t - 1)
            boundary, _ = build_mean_z_consensus_boundary(
                prob[t_idx], dp[t_idx], thresholds, gammas
            )
            # z-avg sigmoid prob for reference layer
            cellprob_zavg = (1.0 / (1.0 + np.exp(-prob.mean(axis=1)))).astype(np.float32)
            return boundary, cellprob_zavg, t_idx

        self._set_contour_status(f"Previewing contour map for frame t={t_frame}…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_build_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path    = self._prob_path()
        dp_path      = self._dp_path()
        contour_path = self._contour_maps_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return

        thresholds = self._thresholds()
        gammas     = self._cp_gammas()

        def _on_done(pos_dir: Path) -> None:
            self._build_worker = None
            self._set_build_buttons_running(False)
            self.contour_files.refresh(pos_dir)
            self._update_contour_status_labels()
            self._set_contour_status("Cell contour maps built.")

        @thread_worker(connect={
            "yielded":  self._on_build_progress,
            "returned": _on_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation import build_mean_z_consensus_boundary

            prob, dp = self._load_prob_dp(prob_path, dp_path)
            n_t = min(prob.shape[0], dp.shape[0])
            contour_frames: list[np.ndarray] = []

            for t in range(n_t):
                yield (t + 1, n_t, f"Building cell contour maps: frame {t + 1}/{n_t}…")
                boundary, _ = build_mean_z_consensus_boundary(
                    prob[t], dp[t], thresholds, gammas
                )
                contour_frames.append(boundary.astype(np.float32))

            contour_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(contour_path), np.stack(contour_frames), compression="zlib")
            return self._pos_dir

        gamma_desc = (
            f"γ={gammas[0]:.2f}"
            if len(gammas) == 1
            else f"γ={gammas[0]:.2f}–{gammas[-1]:.2f} ({len(gammas)} steps)"
        )
        self._set_contour_status(
            f"Building cell contour maps ({len(thresholds)} cellprob thresholds, {gamma_desc})…"
        )
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_cancel_build(self) -> None:
        if self._build_worker is not None:
            self._build_worker.quit()
            self._build_worker = None
        self._set_build_buttons_running(False)
        self._set_contour_status("Build cancelled.")

    def _on_run_contour_terminal(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path    = self._prob_path()
        dp_path      = self._dp_path()
        contour_path = self._contour_maps_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return

        thresholds = self._thresholds()
        gammas     = self._cp_gammas()

        python_code = (
            "import pathlib\n"
            "import numpy as np\n"
            "import tifffile\n"
            "from cellflow.segmentation import build_mean_z_consensus_boundary\n"
            "from cellflow.database.hypotheses import normalize_seeded_watershed_dp_stack\n"
            f"prob_path = pathlib.Path({str(prob_path)!r})\n"
            f"dp_path = pathlib.Path({str(dp_path)!r})\n"
            f"contour_path = pathlib.Path({str(contour_path)!r})\n"
            f"thresholds = {thresholds!r}\n"
            f"gammas = {gammas!r}\n"
            "prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)\n"
            "if prob.ndim == 3:\n"
            "    prob = prob[np.newaxis]\n"
            "dp_raw = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)\n"
            "dp = normalize_seeded_watershed_dp_stack(dp_raw, prob.shape)\n"
            "dp = dp[:, :, :2]\n"
            "n_t = min(prob.shape[0], dp.shape[0])\n"
            "contour_frames = []\n"
            "for t in range(n_t):\n"
            "    print(f'Building cell contour maps: frame {t + 1}/{n_t}...', flush=True)\n"
            "    boundary, _ = build_mean_z_consensus_boundary(\n"
            "        prob[t], dp[t], thresholds, gammas\n"
            "    )\n"
            "    contour_frames.append(boundary.astype(np.float32))\n"
            "contour_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "print('Writing cell contour maps...', flush=True)\n"
            "tifffile.imwrite(str(contour_path), np.stack(contour_frames), compression='zlib')\n"
            "print('Done.')\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="cellflow_cell_contour_", delete=False
        ) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_contour_status("Cell contour build launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_contour_status(
                "Copied cell contour build command to clipboard (terminal launch unavailable)."
            )
```

- [ ] **Step 2: Verify the module imports cleanly (no Qt display needed)**

```bash
cd /home/aruppel/Projects/CellFlow
QT_QPA_PLATFORM=offscreen python -c "
import sys; sys.path.insert(0, 'src')
# stub heavy imports so module-level code doesn't run cellpose
import types, unittest.mock
sys.modules.setdefault('cellpose', types.ModuleType('cellpose'))
sys.modules.setdefault('cellpose.dynamics', types.ModuleType('cellpose.dynamics'))
from cellflow.napari.cell_workflow_widget import CellWorkflowWidget, _CONTOUR_LAYER
print('OK —', _CONTOUR_LAYER)
" 2>&1 | tail -5
```

Expected: `OK — Contour Map: Cell`

- [ ] **Step 3: Commit**

```bash
cd /home/aruppel/Projects/CellFlow
git add src/cellflow/napari/cell_workflow_widget.py
git commit -m "feat(cell): rewrite CellWorkflowWidget with Contour Maps–only section"
```

---

## Task 3: Update section label in `main_widget.py`

**Files:**
- Modify: `src/cellflow/napari/main_widget.py` line ~82

- [ ] **Step 1: Change the section title**

In `src/cellflow/napari/main_widget.py`, find the line:

```python
        self.cell_section = CollapsibleSection(
            "4. Cell Workflow", self.cell_workflow_widget, expanded=False
        )
```

Replace with:

```python
        self.cell_section = CollapsibleSection(
            "4. Cell Segmentation", self.cell_workflow_widget, expanded=False
        )
```

- [ ] **Step 2: Verify the import still works**

```bash
cd /home/aruppel/Projects/CellFlow
QT_QPA_PLATFORM=offscreen python -c "
import sys; sys.path.insert(0, 'src')
import types
for m in ['napari', 'cellpose', 'cellpose.dynamics', 'tifffile', 'skimage',
          'skimage.segmentation', 'skimage.feature', 'scipy', 'scipy.ndimage']:
    if m not in sys.modules:
        sys.modules[m] = types.ModuleType(m)
# Only check the title string is correct — no Qt display needed
import ast, pathlib
src = pathlib.Path('src/cellflow/napari/main_widget.py').read_text()
assert '\"4. Cell Segmentation\"' in src, 'Label not updated'
print('OK — label is 4. Cell Segmentation')
" 2>&1 | tail -3
```

Expected: `OK — label is 4. Cell Segmentation`

- [ ] **Step 3: Commit**

```bash
cd /home/aruppel/Projects/CellFlow
git add src/cellflow/napari/main_widget.py
git commit -m "refactor(ui): rename Cell Workflow section to Cell Segmentation"
```

---

## Task 4: Replace widget tests

**Files:**
- Rewrite: `tests/napari/test_cell_workflow_preview.py`

The old tests exercise the seeded-watershed preview (`_on_preview`, `_on_load_stack_done`, `basin_combo`) which no longer exist. Replace with tests for the new Contour Maps widget.

- [ ] **Step 1: Write the new test file**

Replace the entire content of `tests/napari/test_cell_workflow_preview.py` with:

```python
"""Tests for the cell Contour Maps widget (CellWorkflowWidget)."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QWidget


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


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.cell_workflow_widget", None)
    mod = importlib.import_module("cellflow.napari.cell_workflow_widget")
    monkeypatch.setitem(sys.modules, "cellflow.napari.cell_workflow_widget", mod)
    return mod


# ── get_state / set_state round-trip ─────────────────────────────────────────

def test_get_set_state_round_trips_cellprob_params(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    state = {
        "cellprob": {
            "min": -5.0, "max": 1.0, "step": 2.0,
            "gamma_min": 0.8, "gamma_max": 1.2, "gamma_step": 0.1,
        }
    }
    widget.set_state(state)
    got = widget.get_state()

    assert got["cellprob"]["min"]        == -5.0
    assert got["cellprob"]["max"]        ==  1.0
    assert got["cellprob"]["step"]       ==  2.0
    assert got["cellprob"]["gamma_min"]  ==  0.8
    assert got["cellprob"]["gamma_max"]  ==  1.2
    assert got["cellprob"]["gamma_step"] ==  0.1

    widget.deleteLater()
    app.processEvents()


# ── status labels ─────────────────────────────────────────────────────────────

def test_refresh_none_shows_no_project_message(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    widget.refresh(None)

    assert "no project" in widget.contour_input_lbl.text().lower()
    assert "no project" in widget.contour_output_lbl.text().lower()

    widget.deleteLater()
    app.processEvents()


def test_refresh_existing_files_shows_checkmarks(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", np.zeros((1, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif",   np.zeros((1, 2, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "contour_maps.tif",      np.zeros((1, 4, 4), dtype=np.float32))

    widget.refresh(pos_dir)

    assert "✓" in widget.contour_input_lbl.text()
    assert "✓" in widget.contour_output_lbl.text()

    widget.deleteLater()
    app.processEvents()


# ── _thresholds / _cp_gammas helpers ─────────────────────────────────────────

def test_thresholds_returns_arange_values(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    widget.cp_min_spin.setValue(-2.0)
    widget.cp_max_spin.setValue(0.0)
    widget.cp_step_spin.setValue(1.0)

    thr = widget._thresholds()

    assert len(thr) == 3
    np.testing.assert_allclose(thr, [-2.0, -1.0, 0.0], atol=1e-6)

    widget.deleteLater()
    app.processEvents()


def test_cp_gammas_single_value_when_min_equals_max(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    widget.cp_gamma_min_spin.setValue(1.0)
    widget.cp_gamma_max_spin.setValue(1.0)
    widget.cp_gamma_step_spin.setValue(0.25)

    gammas = widget._cp_gammas()

    assert gammas == [1.0]

    widget.deleteLater()
    app.processEvents()


# ── preview worker ────────────────────────────────────────────────────────────

def test_on_preview_contour_maps_calls_build_mean_z_consensus_boundary(monkeypatch, tmp_path):
    """Preview should call build_mean_z_consensus_boundary and add napari layers."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)

    n_t, n_z, n_y, n_x = 2, 3, 8, 8
    prob = np.zeros((n_t, n_z, n_y, n_x), dtype=np.float32)
    dp   = np.zeros((n_t, n_z, 2, n_y, n_x), dtype=np.float32)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif",   dp)

    boundary_result = np.full((n_y, n_x), 0.5, dtype=np.float32)
    fg_result       = np.full((n_y, n_x), 0.4, dtype=np.float32)

    viewer = _FakeViewer()
    viewer.dims.current_step = (1,)
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    with patch(
        "cellflow.segmentation.build_mean_z_consensus_boundary",
        return_value=(boundary_result, fg_result),
    ) as mock_fn:
        widget._on_preview_contour_maps()

    mock_fn.assert_called_once()
    call_args = mock_fn.call_args
    # First positional arg is prob[t_idx]: shape (n_z, n_y, n_x)
    assert call_args[0][0].shape == (n_z, n_y, n_x)
    # Second positional arg is dp[t_idx]: shape (n_z, 2, n_y, n_x)
    assert call_args[0][1].shape == (n_z, 2, n_y, n_x)

    # Both napari layers should have been added
    assert mod._CONTOUR_LAYER in viewer.layers
    assert mod._CELLPROB_LAYER in viewer.layers

    # Contour layer is a T-stack; boundary appears at t=1
    contour_data = viewer.layers[mod._CONTOUR_LAYER].data
    assert contour_data.shape == (n_t, n_y, n_x)
    np.testing.assert_array_equal(contour_data[1], boundary_result)
    np.testing.assert_array_equal(contour_data[0], 0.0)

    widget.deleteLater()
    app.processEvents()


# ── build worker writes output file ──────────────────────────────────────────

def test_on_build_contour_maps_writes_tif_file(monkeypatch, tmp_path):
    """Build should stack per-frame boundaries and write contour_maps.tif."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)

    n_t, n_z, n_y, n_x = 3, 2, 6, 6
    prob = np.zeros((n_t, n_z, n_y, n_x), dtype=np.float32)
    dp   = np.zeros((n_t, n_z, 2, n_y, n_x), dtype=np.float32)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif",   dp)

    call_count = 0

    def fake_build(prob_t, dp_t, thresholds, gammas, **kwargs):
        nonlocal call_count
        call_count += 1
        boundary = np.full((n_y, n_x), float(call_count) * 0.1, dtype=np.float32)
        fg       = np.zeros((n_y, n_x), dtype=np.float32)
        return boundary, fg

    widget = mod.CellWorkflowWidget(_FakeViewer())
    widget.refresh(pos_dir)

    with patch("cellflow.segmentation.build_mean_z_consensus_boundary", fake_build):
        widget._on_build_contour_maps()

    contour_path = pos_dir / "3_cell" / "contour_maps.tif"
    assert contour_path.exists(), "contour_maps.tif was not written"
    result = tifffile.imread(str(contour_path))
    assert result.shape == (n_t, n_y, n_x)
    assert call_count == n_t
    # Each frame has the expected fill value
    np.testing.assert_allclose(result[0], 0.1, atol=1e-5)
    np.testing.assert_allclose(result[2], 0.3, atol=1e-5)

    widget.deleteLater()
    app.processEvents()
```

- [ ] **Step 2: Run new tests to verify they pass**

```bash
cd /home/aruppel/Projects/CellFlow
python -m pytest tests/napari/test_cell_workflow_preview.py -v 2>&1 | tail -20
```

Expected: All 8 tests PASS.

- [ ] **Step 3: Run the full test suite to check for regressions**

```bash
cd /home/aruppel/Projects/CellFlow
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: All tests PASS (deprecation warnings from `SeededWatershedParams` appearing in other tests is acceptable; failures are not).

- [ ] **Step 4: Commit**

```bash
cd /home/aruppel/Projects/CellFlow
git add tests/napari/test_cell_workflow_preview.py
git commit -m "test(cell): replace seeded-watershed tests with Contour Maps widget tests"
```

---

## Self-Review

### Spec coverage check

| Requirement | Covered by |
|---|---|
| Deprecate current CellWorkflowWidget subwidgets (remove from UI) | Task 2: entire widget replaced with Contour Maps only |
| Deprecate backend logic not used by nucleus (SeededWatershedParams, compute_seeded_watershed) | Task 1, Step 5 |
| Keep backend logic used by nucleus (build_consensus_boundary, ContourWatershedParams, etc.) | Not modified — correct |
| Rename section to "Cell Segmentation" | Task 3 |
| Contour Maps subwidget: cellprob sweep + gamma sweep | Task 2, _setup_ui |
| Mean-Z projection approach (average Z first, then sweep) | Task 1 `build_mean_z_consensus_boundary` |
| Preview / Build / Run in Terminal / Cancel buttons | Task 2, _setup_ui + _connect_signals |
| Progress bar | Task 2, _on_build_progress |
| Output: 3_cell/contour_maps.tif only (no foreground maps) | Task 2, _contour_maps_path, _on_build_contour_maps |
| get_state / set_state | Task 2, covered and tested |

### Placeholder scan

None found.

### Type consistency

- `build_mean_z_consensus_boundary` signature: `(prob_zyx, dp_zcyx, cellprob_thresholds, gammas=(1.0,), *, mask_callback=None) → tuple[ndarray, ndarray]` — consistent across Task 1 definition, Task 2 widget usage, Task 4 mock patch target.
- `_thresholds()` → `list[float]` — used in `_on_preview_contour_maps` and `_on_build_contour_maps` consistently.
- `_cp_gammas()` → `list[float]` — same.
- `_load_prob_dp()` → `tuple[ndarray, ndarray]` — called in both preview and build workers.
- Layer names `_CONTOUR_LAYER`, `_CELLPROB_LAYER` — defined as module constants, referenced consistently in Task 2 and Task 4 tests.
