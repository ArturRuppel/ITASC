# Cell Preview: Temporal Smoothing + On-Demand Labels — Design

**Date:** 2026-06-01
**Widget:** `CellWorkflowWidget` (`src/cellflow/napari/cell_workflow_widget.py`)
**Backend:** `segment_cells_divergence` (`src/cellflow/segmentation/cell_divergence_segmentation.py`)

## Motivation

The live preview deliberately diverges from the full run in two ways that make
tuning misleading:

1. **Temporal smoothing is invisible.** `contour_memory_filter` is a
   bidirectional EMA over the whole movie, so it cannot run on an isolated
   frame; the single-frame preview skips it (`segment_cells_divergence(...,
   frame=t)`). The previewed contours/cost field therefore differ from what the
   full run actually segments whenever `memory_tau > 0`.
2. **No labels in preview.** The geodesic label assignment was removed from the
   preview for performance (`with_labels=False`), so the user never sees the
   actual segmentation while tuning — only the cost field that explains it.

This design restores both **faithfully** (preview matches the full run for the
shown frame) while keeping tuning responsive.

### Non-goal: parallelizing the geodesic across labels

Considered and rejected as structural, not an implementation gap. The frame's
`MCP_Geometric` object is built once and reused across all labels
(`cell_label_icm.py:149-162`); the expensive part is the MCP build, the cheap
part is each seed's Dijkstra. Per-label process parallelism would rebuild the
MCP per worker (multiplying the expensive part), amortize fork/IPC poorly at
single-frame granularity, and serialize a full `(Y, X)` array per label back
through the pipe. The per-frame median normalization
(`cell_label_icm.py:164-174`) also genuinely needs every label's full distance
field, so it cannot collapse to a single Voronoi sweep. The single-frame path
(one MCP, N sequential seeds) is already the optimal shape for one frame.

## Design

### 1. On-demand single-frame labels button

- A new tool button in the segmentation stage row (alongside `◉`), enabled
  **only while preview is active**.
- Clicking it runs the existing single-frame `with_labels=True` path for the
  current frame off the GUI thread and fills `_LABELS_LAYER` (already declared
  at `cell_workflow_widget.py:85`, currently never created).
- It is an explicit action: it does **not** recompute on param edits or time
  scrubs. Tuning stays responsive; the slow geodesic step runs only when asked.
- The labels layer is cleared when preview deactivates (added to
  `_PREVIEW_LAYERS` teardown).

**Exactness:** the per-frame median normalization in `_compute_frame_geodesic`
is already per-frame, so single-frame labels for frame *t* are identical to the
full-run labels for frame *t* given the same contour input. The preview labels
are exact, not an approximation.

### 2. Temporal smoothing in preview (full stack, show current frame)

Only the smoothed cleaned-contour stack needs the whole movie; everything else
in the per-frame pipeline (foreground clean, mask, cost field) stays
single-frame and cheap.

- **`memory_tau > 0`:** the preview computes the full cleaned-and-smoothed
  contour stack off-thread, holds it resident, and slices frame *t* from it for
  the cost field / mask / labels.
- **`memory_tau == 0`:** falls back to today's pure single-frame path and drops
  the cached stack (no smoothing to show).

#### Caching / invalidation (the responsiveness lever)

The smoothed stack is keyed on the params that determine it:
`contour_window`, `contour_strength`, `contour_threshold`, `contour_norm_pct`,
`memory_tau`, `memory_floor`.

| User action | Smoothed stack |
|---|---|
| Scrub time | Reuse cache, slice new frame |
| Tweak `fg_*`, `alpha`, `gamma` | Reuse cache |
| Tweak any contour or temporal knob | Recompute whole stack (off-thread, status: "Temporal smoothing over N frames…") |
| Deactivate preview | Drop cache (free memory) |

The recompute fires only when a knob that actually changes the smoothing is
touched. Both the cost-field preview and the labels button read from this same
cache, so what the user sees and what the labels use is the exact array the full
run uses.

#### Memory

Holding the smoothed stack is ~one `(T, Y, X)` float32 array for the preview
session (e.g. ~1.6 GB for a 100-frame 2048² movie — same order as the full
run's existing `imread`). Accepted; released on deactivate.

### 3. Backend change

`segment_cells_divergence` gains a way to accept a pre-smoothed contour frame so
the single-frame preview path uses the cached array rather than re-cleaning /
re-smoothing. Either:

- an optional `contours_clean_override` argument for the single-frame path, or
- factor the per-frame "rest of pipeline" (mask + cost + optional geodesic) into
  a small helper the widget calls with the cached frame.

The full-run path is unchanged. This keeps the preview using the exact cached
array and the widget thin.

## Affected code

- `src/cellflow/napari/cell_workflow_widget.py` — labels button + handler;
  smoothed-stack cache + invalidation; worker branching (light single-frame vs.
  heavy full-stack smoothing); teardown for the labels layer and cache.
- `src/cellflow/segmentation/cell_divergence_segmentation.py` — pre-smoothed
  contour entry point for the single-frame path.
- `tests/napari/test_cell_workflow_widget.py`,
  `tests/segmentation/test_cell_divergence_segmentation.py` — cover the new
  single-frame-with-override path and cache invalidation.

## Open questions

None outstanding. Memory tradeoff and cache-on-edit behavior confirmed.
