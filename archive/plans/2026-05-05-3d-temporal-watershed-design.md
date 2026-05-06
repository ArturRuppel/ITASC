# 3D Temporal Watershed — Design Spec

Date: 2026-05-05

## Problem

Cell contour maps produced by Cellpose are temporally inconsistent: boundaries may be sharp in frame t but absent or shifted in t+1. Running per-frame 2D watershed ignores this — it treats each frame independently and cannot suppress spurious one-frame boundaries or reinforce temporally stable ones.

Treating the (T, Y, X) contour stack as a 3D volume and running watershed across it allows temporal neighbours to compete for basin membership alongside spatial neighbours, naturally coupling cell identity across time.

---

## Approach

**Axis-scaling anisotropic watershed.** To support separate compactness values for space and time, scale the T axis by `sqrt(compactness_time / compactness_space)` before running `skimage.segmentation.watershed(..., compactness=compactness_space)`. The Euclidean distance in the scaled space equals `sqrt(cs·(dy²+dx²) + ct·dt²)`, giving exact anisotropic compactness without a custom priority queue.

Optional smoothing (Gaussian and/or median) is applied to the contour stack before watershed, with independent spatial and temporal kernel parameters.

---

## Backend

**New function** in `src/cellflow/segmentation/` (new file `watershed_3d.py`), exported from `__init__.py`:

```python
def compute_3d_temporal_watershed(
    contours_tyx: np.ndarray,      # (T, Y, X) float32
    foreground_tyx: np.ndarray,    # (T, Y, X) bool
    seeds_tyx: np.ndarray,         # (T, Y, X) int32, one label per nucleus per frame
    gaussian_sigma_space: float,
    gaussian_sigma_time: float,
    median_kernel_space: int,
    median_kernel_time: int,
    compactness_space: float,
    compactness_time: float,
) -> tuple[np.ndarray, np.ndarray]:   # (smoothed_contours_tyx, labels_tyx)
```

**Steps:**
1. Gaussian filter with `sigma=(gaussian_sigma_time, gaussian_sigma_space, gaussian_sigma_space)` — skipped if both are 0
2. Median filter with `size=(median_kernel_time, median_kernel_space, median_kernel_space)` — skipped if both are 1
3. Axis scaling: zoom T axis by `sqrt(compactness_time / compactness_space)` on contours and foreground (linear/bilinear interpolation); seeds zoomed with `order=0` (nearest-neighbour) to preserve integer label values; degenerate to no scaling when either compactness is 0
4. `skimage.segmentation.watershed(scaled_contours, markers=scaled_seeds, mask=scaled_foreground, compactness=compactness_space)`
5. Zoom labels back to original T resolution
6. Return `(smoothed_contours_tyx, labels_tyx)` — smoothed contours at original resolution (pre-scaling)

---

## Widget

New collapsible section **"3D Temporal Watershed"** appended to `src/cellflow/napari/cell_workflow_widget.py`, below the existing Contour Maps section. Follows the same structural pattern.

### Inputs (read from disk, no UI path pickers)

| File | Path |
|------|------|
| Contour maps | `3_cell/contour_maps.tif` |
| Foreground masks | `3_cell/foreground_masks.tif` |
| Nucleus tracked labels | `2_nucleus/tracked_labels.tif` |

Input status label shows a checkmark/cross for each file.

### Parameters

**Smoothing group:**

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| Gaussian spatial σ | float | 0.0–10.0 | 1.0 |
| Gaussian temporal σ | float | 0.0–10.0 | 1.0 |
| Median spatial kernel | int (odd) | 1–15 | 1 |
| Median temporal kernel | int (odd) | 1–15 | 1 |

**Watershed group:**

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| Spatial compactness | float | 0.0–100.0 | 0.0 |
| Temporal compactness | float | 0.0–100.0 | 0.0 |

### Buttons

Run | Cancel

### Outputs

| File | Path | Format |
|------|------|--------|
| Smoothed contours | `3_cell/smoothed_contours.tif` | (T, Y, X) float32, zlib |
| Cell labels | `3_cell/tracked_labels.tif` | (T, Y, X) uint32, zlib |

Output files widget shows both paths with existence checkmarks.

---

## Run Sequence

`@thread_worker` on Run click:

1. Validate all three input files exist; set error status and return early if not
2. Load contour maps → (T, Y, X) float32
3. Load foreground masks → (T, Y, X) bool
4. Load nucleus tracked labels → (T, Y, X) int32; call `centroid_markers_from_labels` to get one seed per cell per frame
5. Call `compute_3d_temporal_watershed(...)`
6. Save `3_cell/smoothed_contours.tif` with zlib compression
7. Save `3_cell/tracked_labels.tif` with zlib compression
8. Add/update napari layers: "Cell Smoothed Contours" (inferno), "Cell Labels" (label colormap)
9. Refresh output files widget

Progress yields per-frame (or per major step) during watershed computation.

---

## Edge Cases

| Condition | Behaviour |
|-----------|-----------|
| `compactness_space == 0` or `compactness_time == 0` | Skip axis scaling; run watershed with `compactness=0` |
| Gaussian sigma == 0 | Skip Gaussian step |
| Median kernel == 1 | Skip median step |
| Missing input file | Status label error, worker not started |
| Cancel | `worker.quit()`, Run re-enabled |

---

## Files Changed

| File | Change |
|------|--------|
| `src/cellflow/segmentation/watershed_3d.py` | New — `compute_3d_temporal_watershed` |
| `src/cellflow/segmentation/__init__.py` | Export new function |
| `src/cellflow/napari/cell_workflow_widget.py` | New collapsible section |
| `tests/test_cell_workflow_widget.py` | Tests for new section and backend |
