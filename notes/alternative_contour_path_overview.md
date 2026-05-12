# Alternative Contours Path — Project Overview & Implementation Context

> **Task**: Create an alternative path to the Contour Maps creation widget in the Cell Segmentation section.
> The alternative uses the same parameters and flow-following mechanism, but instead of Cellpose's
> native `compute_masks`, pixels get captured by the closest nucleus via our custom EDT-gravity method.

---

## Project Tree (relevant files)

```
src/cellflow/
├── napari/
│   ├── __init__.py                          # Plugin entry: exports CellFlowWidget
│   ├── main_widget.py                       # Main UI: ties all sections together
│   ├── cell_workflow_widget.py              # Cell Segmentation parent (embeds boundary widget)
│   ├── cell_boundary_workflow_widget.py     # ★ Contour Maps + Boundary Selection widget
│   ├── cellpose_widget.py                   # Cellpose info panel (read-only)
│   ├── hpc_cellpose_widget.py               # HPC launcher for external Cellpose
│   ├── nucleus_workflow_widget.py           # Nucleus segmentation & tracking
│   ├── correction_widget.py                 # Shared manual correction tools
│   ├── widgets.py                           # Reusable UI (PipelineFilesWidget, etc.)
│   └── ui_style.py                          # Layout helpers (block_grid, etc.)
│
├── segmentation/
│   ├── __init__.py                          # ★ Public API: exports all segmentation functions
│   ├── flow_following.py                    # ★ Flow-following with nucleus capture (numba)
│   ├── contour_filtering.py                 # Contour-map spatial/temporal filtering
│   └── cell_label_icm.py                    # ICM solver for boundary optimization
│
├── core/
│   ├── logging.py
│   ├── paths.py
│   └── data_prep.py
│
└── napari.yaml                              # Plugin manifest
```

---

## Current Pipeline (how it works today)

### Stage A: Contour Maps (`cell_boundary_workflow_widget.py` — section "1. Contour Maps")

1. **Input**: `cell_prob_3dt.tif` (Cellpose probability logits, Z-stack) + `filtered_dp.tif` (filtered flow vectors, 2D per frame)
2. **Processing** (per frame):
   - Apply gamma correction to the probability logits → average over Z → single 2D probability map
   - Call `build_consensus_boundary_2d()` which iterates over cellprob thresholds and calls **Cellpose's `compute_masks()`** (native method) for each threshold
   - `compute_masks` is Cellpose's internal flow-based mask generation: it moves pixels along the flow field and assigns them to masks
   - For each resulting mask, the boundary is extracted via `find_boundaries(mode="inner")`
   - Boundaries and foreground masks are accumulated across all thresholds
3. **Output**: `contour_maps.tif`, `foreground_scores.tif`, `foreground_masks.tif`

### Stage B: Boundary Selection (`cell_boundary_workflow_widget.py` — section "2. Track-Conditioned Boundary Selection")

1. **Initialize**: Uses nucleus tracked labels + contour maps + foreground masks to compute geodesic unary costs and ICM state
2. **Refine**: Runs ICM sweeps to optimize cell boundaries
3. **Commit**: Writes `tracked_labels.tif`

### The Custom Flow-Following Method (already implemented!)

`src/cellflow/segmentation/flow_following.py` already contains `compute_flow_following_movie()` which:

- Takes foreground masks, filtered flow vectors, and **nucleus tracked labels**
- Per frame: computes EDT distance transform from nucleus pixels → gravity direction toward nearest nucleus
- For every foreground pixel: integrates along the flow field, blended with EDT gravity
- Captures the pixel when it enters the `capture_radius` of a nucleus → assigns that nucleus's label
- Uses numba JIT for performance

**This is the "custom method where pixels get captured by the closest nucleus"** — it already exists and works.

---

## The Gap: What Needs to Be Built

The `compute_flow_following_movie()` function exists but is **not wired into the Contour Maps widget**. The Contour Maps widget only calls Cellpose's `compute_masks`. The alternative path should:

1. Add a mode selector in the Contour Maps section (or a new parallel section)
2. When the alternative mode is selected, use `compute_flow_following_movie()` instead of `build_consensus_boundary_2d()`
3. The output should still produce contour maps and foreground masks (or equivalent outputs that the downstream Boundary Selection stage can consume)

**Key difference**: `compute_flow_following_movie` requires nucleus tracked labels as input, which `build_consensus_boundary_2d` does not. This means the alternative path also needs access to `2_nucleus/tracked_labels.tif`.

---

## Relevant Code — Deep Dive

### 1. The Cellpose Native Path (current contour creation)

**File**: `src/cellflow/segmentation/__init__.py` — function `build_consensus_boundary_2d()`

```python
def build_consensus_boundary_2d(
    prob_yx: np.ndarray,        # (Y, X) Cellpose probability logits
    dp_cyx: np.ndarray,         # (2, Y, X) flow vectors
    cellprob_thresholds: list[float],
    flow_threshold: float = 0.0,
    reduction: str = "mean",
    niter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Build consensus boundary from a Z-averaged probability map and 2D flow vectors.
    Returns (boundary, foreground) both (Y, X) float32.
    """
    from cellpose.dynamics import compute_masks
    from skimage.segmentation import find_boundaries

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accum = np.zeros(prob_yx.shape, dtype=np.float32)
    foreground_accum = np.zeros(prob_yx.shape, dtype=np.float32)

    for thresh in cellprob_thresholds:
        result = compute_masks(
            dp_cyx, prob_yx,
            cellprob_threshold=float(thresh),
            flow_threshold=float(flow_threshold),
            niter=int(niter),
            do_3D=False, device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        masks_arr = np.asarray(masks)
        boundary_slice = find_boundaries(masks_arr, mode="inner").astype(np.float32)
        fg_slice = (masks_arr > 0).astype(np.float32)
        # accumulate...
    return boundary, foreground
```

### 2. The Widget That Calls It

**File**: `src/cellflow/napari/cell_boundary_workflow_widget.py` — method `_build_consensus_boundary_averaged()`

```python
def _build_consensus_boundary_averaged(
    self, prob_3d, dp_2d, thresholds, gammas,
    *, flow_threshold, niter,
):
    boundary_accum = foreground_accum = None
    for gamma in gammas:
        prob_2d = apply_gamma(prob_3d, gamma).mean(axis=0)
        b, fg = build_consensus_boundary_2d(
            prob_2d, dp_2d, thresholds,
            flow_threshold=flow_threshold, reduction="mean", niter=niter,
        )
        # accumulate across gamma values...
    return boundary_accum / n, foreground_accum / n
```

This is called from `_on_build_contour_maps()` and `_on_preview_contour_maps()` worker threads.

### 3. The Custom Flow-Following Method (target for alternative path)

**File**: `src/cellflow/segmentation/flow_following.py` — function `compute_flow_following_movie()`

```python
def compute_flow_following_movie(
    foreground_tyx: np.ndarray,    # (T, Y, X) bool — foreground masks
    dp_tcyx: np.ndarray,           # (T, 2, Y, X) float32 — filtered flow vectors
    labels_tyx: np.ndarray,        # (T, Y, X) int32 — nucleus tracked labels
    params: FlowFollowingParams,
    progress_cb: Callable | None = None,
    *, filter_vectors: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame flow-following segmentation.
    Returns (filtered_dp_tcyx, cell_labels_tyx).
    """
    # Per frame:
    #   - Normalize flow vectors by mean magnitude
    #   - Compute EDT from nucleus pixels → gravity direction
    #   - _flow_integrate() → numba kernel that integrates each pixel along
    #     flow + gravity, assigns nucleus label when within capture_radius
    ...
```

The numba kernel `_flow_integrate()` does the actual pixel integration:

```python
@numba.njit(parallel=True, cache=True)
def _flow_integrate(
    nuclear_labels, flow, grav_y, grav_x,
    dist_to_nucleus, nearest_y, nearest_x,
    prob_mask, n_steps, flow_step_scale, flow_weight, capture_radius,
) -> np.ndarray:
    for i in numba.prange(H):
        for j in range(W):
            # skip nucleus pixels and non-foreground pixels
            py, px = float(i), float(j)
            for _ in range(n_steps):
                # bilinear interpolation of flow vector at (py, px)
                flow_y = ...
                flow_x = ...
                # blend flow with EDT gravity
                step_y = flow_weight * flow_y + (1 - flow_weight) * grav_y[...]
                step_x = flow_weight * flow_x + (1 - flow_weight) * grav_x[...]
                py += step_y * flow_step_scale
                px += step_x * flow_step_scale
                # check capture
                if dist_to_nucleus[int(py), int(px)] <= capture_radius:
                    label = nuclear_labels[nearest_y, nearest_x]
                    if label > 0:
                        break
            result[i, j] = label
    return result
```

### 4. FlowFollowingParams

```python
@dataclass(frozen=True, slots=True)
class FlowFollowingParams:
    median_kernel_time: int = 3
    median_kernel_space: int = 5
    gaussian_sigma_time: float = 0.0
    gaussian_sigma_space: float = 0.0
    flow_weight: float = 0.5       # blend between flow (1) and gravity (0)
    flow_step_scale: float = 0.2   # step size multiplier
    max_iterations: int = 100      # max integration steps
    capture_radius: float = 3.0    # distance threshold for nucleus capture
```

---

## Implementation Plan: Files to Modify

### Must Modify

| # | File | What | Why |
|---|------|------|-----|
| 1 | `src/cellflow/napari/cell_boundary_workflow_widget.py` | Add mode selector (Cellpose native / Flow-following) in the Contour Maps section; add new UI controls for FlowFollowingParams; add alternative worker implementation that calls `compute_flow_following_movie` | This is the widget the user interacts with — the alternative path goes here |
| 2 | `src/cellflow/napari/cell_workflow_widget.py` | Wire state get/set for the new mode and flow-following params; possibly pass nucleus labels path to the boundary widget | Parent widget that owns state persistence |

### May Need Modification

| # | File | What | Why |
|---|------|------|-----|
| 3 | `src/cellflow/segmentation/__init__.py` | Export any new helper functions if needed (e.g., a wrapper that produces contour maps from flow-following labels) | Public API for segmentation module |
| 4 | `src/cellflow/segmentation/flow_following.py` | Possibly add a single-frame variant of `compute_flow_following_movie` for the preview path; or add a function to convert flow-following labels to contour maps | The existing `compute_flow_following_movie` works on full stacks — preview needs single-frame |

### Should NOT Modify

| File | Reason |
|------|--------|
| `src/cellflow/segmentation/cell_label_icm.py` | ICM solver is downstream of contour creation; it consumes contour maps regardless of source |
| `src/cellflow/napari/main_widget.py` | No structural changes needed; the boundary widget is already embedded via `CellWorkflowWidget` |
| `src/cellflow/napari/nucleus_workflow_widget.py` | Only change is the boundary widget needs access to nucleus labels path |

---

## Design Considerations

### Input Dependency Chain

```
Current path:
  cell_prob_3dt.tif + filtered_dp.tif
    → build_consensus_boundary_2d (cellpose compute_masks)
    → contour_maps.tif + foreground_scores.tif + foreground_masks.tif
    → ICM boundary selection
    → cell_labels.tif

Alternative path (flow-following):
  cell_prob_3dt.tif + filtered_dp.tif + nucleus_tracked_labels.tif
    → compute_flow_following_movie (EDT gravity capture)
    → cell_labels.tif
    → (derive contour maps from labels if needed for downstream)
```

**Key question**: Does the downstream ICM Boundary Selection stage *require* contour maps, or can it work directly from flow-following cell labels?

Looking at `cell_label_icm.py` → `initialize_icm()`:
- It takes `nuc_tracks`, `fg_mask`, `contours` (contour maps)
- The contour maps are used for: (a) building the geodesic cost field, (b) computing pairwise weights
- If flow-following already produces good cell labels, the ICM refinement stage might be unnecessary or could use a different cost signal

**Options**:
1. **Minimal**: Produce contour maps from flow-following labels (boundary extraction) and feed into existing ICM pipeline
2. **Medium**: Skip contour maps entirely — use flow-following labels directly as the output, bypassing ICM
3. **Full**: Add a new "Flow-Following" section parallel to both Contour Maps and Boundary Selection, producing final cell labels in one step

### Parameter Mapping

The Contour Maps section has these parameters that should carry over:
- `cp_min`, `cp_max`, `cp_step` — cellprob thresholds (map to foreground binarization in flow-following)
- `cp_gamma_min/max/step` — gamma correction (already applied before flow-following)
- `contour_flow_threshold` — maps conceptually to `flow_weight` or could be reused
- `contour_niter` — maps to `max_iterations` in FlowFollowingParams
- `contour_fg_threshold` — maps to foreground mask threshold

Additional params for flow-following mode:
- `flow_weight` — blend between flow direction and EDT gravity (0 = pure gravity, 1 = pure flow)
- `flow_step_scale` — integration step size
- `capture_radius` — distance threshold for nucleus capture
- `median_kernel_time/space`, `gaussian_sigma_time/space` — flow vector filtering (already in `CellWorkflowWidget`)

---

## Summary

The project already has both:

1. **Cellpose native path** (`build_consensus_boundary_2d` → `cellpose.dynamics.compute_masks`): sweeps cellprob thresholds, generates masks via Cellpose's internal flow dynamics, extracts boundaries
2. **Custom flow-following path** (`compute_flow_following_movie` → `_flow_integrate`): integrates pixels along the flow field blended with EDT gravity toward the nearest nucleus, captures when within radius

The task is to wire option #2 into the Contour Maps widget as a selectable alternative, with equivalent parameter controls and output that feeds into the downstream pipeline. The core algorithm already exists and is tested; the work is UI integration and output format adaptation.
