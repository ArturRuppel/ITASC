# Flow-Following Cell Segmentation — Design Spec

Date: 2026-05-05

## Problem

The current cell segmentation pipeline builds a contour map from Cellpose by sweeping `cellprob_threshold` and averaging boundary maps, then runs a 3D temporal watershed (Gaussian + median in (T, Y, X), anisotropic compactness) seeded by tracked nuclear labels. The two-stage process is hard to reason about, introduces artifacts at threshold sweep edges, and the smoothing of contour maps is a weak proxy for what we actually want: a temporally consistent assignment of pixels to cells that respects the underlying flow field.

A direct flow-following approach is conceptually simpler and matches the philosophy of Cellpose: every foreground pixel is advected along the (z-averaged) Cellpose flow field toward a nuclear seed; pixels that never reach a seed fall back to a Voronoi assignment using the same nuclei. Spurious flow vectors at noisy edges are handled by median filtering of the flow stack — far more meaningful than smoothing the contour map.

A v1 implementation (`archive/v1/.../gravity_flow.py`) exists and is the template for this spec, with two key changes: drop the cellpose-prob foreground (we have an external `foreground_masks.tif`), and pre-filter the flow stack in (T, Y, X) so per-frame integration sees a temporally consistent flow field.

---

## Approach

**Per-frame 2D flow-following with pre-integration filtering on the flow stack.**

For each timepoint independently, advect every foreground pixel along the Cellpose flow field with a blended pull toward the nearest tracked nucleus. Capture pixels that come within `capture_radius` of a nucleus; Voronoi-fill any leftovers. Run this independently for each frame; temporal consistency comes entirely from (a) tracked nuclear labels (already temporally consistent via ultrack) and (b) median+Gaussian filtering of the flow stack in (T, Y, X) before per-frame integration.

The 3D temporal watershed and consensus-boundary contour-map machinery are replaced wholesale.

---

## Backend

**New module** `src/cellflow/segmentation/flow_following.py`, exported from `__init__.py`.

### Public API

```python
@dataclass(frozen=True, slots=True)
class FlowFollowingParams:
    median_kernel_time: int = 3       # odd, 1 = off
    median_kernel_space: int = 5      # odd, 1 = off
    gaussian_sigma_time: float = 0.0  # 0 = off
    gaussian_sigma_space: float = 0.0
    flow_weight: float = 0.5          # [0, 1]; 1 = pure flow, 0 = pure gravity
    flow_step_scale: float = 0.2
    max_iterations: int = 100
    capture_radius: float = 3.0


def compute_flow_following_movie(
    foreground_tyx: np.ndarray,   # (T, Y, X) bool
    dp_tcyx: np.ndarray,          # (T, 2, Y, X) float32, z-averaged
    labels_tyx: np.ndarray,       # (T, Y, X) int32, tracked nucleus labels
    params: FlowFollowingParams,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (filtered_dp_tcyx, cell_labels_tyx).

    filtered_dp_tcyx: (T, 2, Y, X) float32 — flow stack after median+Gaussian.
    cell_labels_tyx:  (T, Y, X) int32      — same labelling as input nuclei.
    """
```

### Algorithm

1. **Filter flow stack** — operate on `dp_tcyx` directly (channels treated independently):
   - `median_filter` with `size=(1, median_kernel_time, median_kernel_space, median_kernel_space)` if any kernel > 1
   - `gaussian_filter` with `sigma=(0, gaussian_sigma_time, gaussian_sigma_space, gaussian_sigma_space)` if any σ > 0
2. **Per frame** (loop over T, optionally parallelised at the outer level):
   - Build `prob_mask = foreground_tyx[t]` (already binary; no thresholding).
   - Normalise flow by mean magnitude over `prob_mask`. Skip if foreground empty or mean < 1e-6.
   - Compute EDT to nearest nuclear pixel: `dist, (ny, nx) = distance_transform_edt(labels[t] == 0, return_indices=True)`.
   - Compute **EDT-direction-only gravity field** `(g_y, g_x)`: at each pixel `p`, unit vector toward `(ny[p], nx[p])`. Zero inside nuclei. (No multi-centroid sum, no falloff parameter — we trust the EDT direction.)
   - Run Numba `_flow_integrate` (port of v1):
     - For each foreground pixel not already labelled:
       - Euler step: `s = w · flow_bilinear + (1 − w) · gravity_nn`, position `+= s · flow_step_scale`, clamp to image
       - Capture when `dist[round(pos)] ≤ capture_radius` → assign `labels[t][ny[p], nx[p]]`
   - Voronoi fill any unassigned foreground via the same EDT indices.
3. Return `(filtered_dp_tcyx, cell_labels_tyx)`.

### Numba kernel signature

```python
@numba.njit(parallel=True, cache=True)
def _flow_integrate(
    nuclear_labels: np.ndarray,   # (H, W) int32
    flow: np.ndarray,             # (H, W, 2) float32 — (dy, dx) per pixel
    grav_y: np.ndarray,           # (H, W) float32
    grav_x: np.ndarray,
    dist_to_nucleus: np.ndarray,  # (H, W) float32
    nearest_y: np.ndarray,        # (H, W) int32
    nearest_x: np.ndarray,
    prob_mask: np.ndarray,        # (H, W) bool
    n_steps: int,
    flow_step_scale: float,
    flow_weight: float,
    capture_radius: float,
) -> np.ndarray:                  # (H, W) int32
```

Logic identical to v1 `gravity_flow._flow_integrate`, except:
- Input flow shape is `(H, W, 2)` (caller transposes from `(2, H, W)` Cellpose convention).
- No early-exit branch on `prob_mask` semantics changes (still skips non-foreground).

### Voronoi fill

```python
def _fill_foreground(labels: np.ndarray, prob_mask: np.ndarray) -> np.ndarray:
    missing = prob_mask & (labels == 0)
    if not missing.any():
        return labels
    _, (iy, ix) = distance_transform_edt(labels == 0, return_indices=True)
    out = labels.copy()
    out[missing] = labels[iy[missing], ix[missing]]
    return out
```

### Z-averaging convention

The widget loads `cell_dp_3dt.tif`, normalises with the existing `normalize_seeded_watershed_dp_stack`, and z-averages with `mean(axis=Z)` — same convention as `build_mean_z_consensus_boundary`. The flow-following backend itself takes `dp_tcyx` already z-averaged.

---

## Widget

The "3D Temporal Watershed" collapsible section in `src/cellflow/napari/cell_workflow_widget.py` is **replaced** by a "Flow-Following Segmentation" section. The "Contour Maps" section is **removed**, since flow-following does not consume contour maps.

### Inputs (read from disk, no UI path pickers)

| File | Path |
|------|------|
| Cellpose prob (for shape only)| `1_cellpose/cell_prob_3dt.tif` |
| Cellpose dp (raw) | `1_cellpose/cell_dp_3dt.tif` |
| Foreground masks | `3_cell/foreground_masks.tif` |
| Nucleus tracked labels | `2_nucleus/tracked_labels.tif` |

Input status label shows a checkmark/cross for each file.

### Parameters

**Flow filtering group:**

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| Median temporal kernel | int (odd) | 1–15 | 3 |
| Median spatial kernel | int (odd) | 1–15 | 5 |
| Gaussian temporal σ | float | 0.0–10.0 | 0.0 |
| Gaussian spatial σ | float | 0.0–10.0 | 0.0 |

**Flow-following group:**

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| Flow weight | float | 0.0–1.0 | 0.5 |
| Step scale | float | 0.05–1.0 | 0.2 |
| Max iterations | int | 10–500 | 100 |
| Capture radius | float | 0.5–10.0 | 3.0 |

### Buttons

Run | Cancel

### Outputs

| File | Path | Format |
|------|------|--------|
| Filtered flow magnitude | `3_cell/filtered_flow_mag.tif` | (T, Y, X) float32, zlib |
| Cell labels | `3_cell/tracked_labels.tif` | (T, Y, X) uint32, zlib |

`filtered_flow_mag.tif` is `sqrt(dy² + dx²)` of the filtered flow stack, kept for visual inspection only — debugging aid analogous to the previous "smoothed contours" output.

---

## Run Sequence

`@thread_worker` on Run click:

1. Validate all four input files exist; set error status and return early if not.
2. Load prob → shape only; load `cell_dp_3dt.tif`, normalise via `normalize_seeded_watershed_dp_stack`, z-average to `dp_tcyx` shape `(T, 2, Y, X)`.
3. Load `foreground_masks.tif` → `(T, Y, X) bool`.
4. Load `tracked_labels.tif` → `(T, Y, X) int32`.
5. Call `compute_flow_following_movie(...)`; progress yields per-frame.
6. Save `3_cell/tracked_labels.tif` (uint32, zlib) and `3_cell/filtered_flow_mag.tif` (float32, zlib).
7. Add/update napari layers: "Filtered Flow Magnitude" (inferno) and "Cell Labels" (label colormap).
8. Refresh output files widget.

---

## Edge Cases

| Condition | Behaviour |
|-----------|-----------|
| Median kernel == 1 | Skip median filter |
| Gaussian σ == 0 | Skip Gaussian filter |
| Foreground empty in a frame | Output all zeros for that frame; no error |
| `mean_flow_mag < 1e-6` in a frame | Skip flow normalisation; gravity-only integration |
| No nuclei in a frame | Output all zeros for that frame; warn but do not raise |
| Missing input file | Status label error, worker not started |
| Cancel | `worker.quit()`, Run re-enabled |

---

## Files Changed

| File | Change |
|------|--------|
| `src/cellflow/segmentation/flow_following.py` | New — `FlowFollowingParams`, `compute_flow_following_movie`, `_flow_integrate`, `_fill_foreground` |
| `src/cellflow/segmentation/__init__.py` | Export new symbols |
| `src/cellflow/napari/cell_workflow_widget.py` | Remove Contour Maps + 3D Temporal Watershed sections; add Flow-Following section |
| `tests/segmentation/test_flow_following.py` | New — backend unit tests |
| `tests/test_cell_workflow_widget.py` | Update widget tests for new section |
| `src/cellflow/segmentation/watershed_3d.py` | **Delete** once flow-following is validated |
| `src/cellflow/segmentation/__init__.py` (`build_consensus_boundary`, `build_mean_z_consensus_boundary`, `compute_cellpose_flow_hypothesis`, `compute_seeded_watershed`) | **Delete** once flow-following is validated |

The deletions land in the same PR as the new code: replacement, not coexistence.

---

## Out of scope

- True (T, Y, X) integration through space-time. Considered and rejected: nuclei already exist in every frame, so cross-time pull is unhelpful; temporal consistency comes from the tracked seeds and from filtering the flow stack.
- Multi-centroid gravity (v1's `1/|c−p|^(falloff+1)` sum). EDT-direction-only is faster and equivalent in the regime where capture happens early.
- Vector (geometric) median filter. Componentwise median is standard, fast, and sufficient.
- 3D (Z) integration. Cellpose dp is computed per z-slice and averaged before flow-following; the segmentation lives in (Y, X) per timepoint.
