# cellflow-segmentation

Independent CellFlow piece for **cell segmentation**: a divergence-based
geodesic-Voronoi pipeline that turns Cellpose-derived cell foreground/contour
maps plus tracked nucleus seeds into a `tracked_labels.tif` cell-label stack,
with interactive label correction in napari.

## Install

```bash
pip install cellflow-segmentation   # divergence segmentation + correction
```

The upstream Cellpose foreground/contour maps come from the separate
[`cellflow-cellpose`](../cellflow-cellpose) distribution.

This pulls in `cellflow-core`. Both install into the shared `cellflow.*`
namespace (PEP 420), so `import cellflow.segmentation` works whether or not the
full CellFlow orchestrator is present.

## Use

- **napari plugin:** add the *Cell Segmentation* widget. In standalone mode it
  exposes its own input/output pickers:
  - **Foreground** — cell foreground `.tif` (Cellpose sigmoid map)
  - **Contours** — cell contours `.tif` (Cellpose divergence map)
  - **Nucleus** — tracked nucleus labels `.tif` (segmentation seeds)
  - **Output dir** — receives `3_cell/tracked_labels.tif`

  Tune the cleanup/temporal/segmentation knobs against the live single-frame
  preview, then run the full stack. Correction is the same interactive
  label-editing widget shared with the rest of CellFlow.

- **Headless / scripting:**

  ```python
  import tifffile
  from cellflow.segmentation import (
      CellDivergenceParams,
      segment_cells_divergence,
  )

  contours = tifffile.imread("1_cellpose/cell_contours.tif")
  foreground = tifffile.imread("1_cellpose/cell_foreground.tif")
  nuclei = tifffile.imread("2_nucleus/tracked_labels.tif")
  result = segment_cells_divergence(
      contours, foreground, nuclei, CellDivergenceParams(),
  )
  tifffile.imwrite("3_cell/tracked_labels.tif", result.labels)
  ```

## I/O contract

- **Input:** `cell_foreground` + `cell_contours` (2D+t TIFF, produced upstream by
  the shared Cellpose step), and `nucleus` tracked labels (2D+t TIFF, used as
  segmentation seeds).
- **Output:** `3_cell/tracked_labels.tif` — a 2D+t cell-label stack.
