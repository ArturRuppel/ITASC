# Watershed Hierarchy Skips Intermediate Merge Levels — Issue Report

**Date:** 2026-05-06  
**Status:** open — root cause hypothesis confirmed by contour inspection, fix pending

## Problem

The Ultrack watershed hierarchy produces candidate nodes that jump directly from
a full connected-component region to small internal fragments, skipping the
intuitively obvious intermediate subdivisions.

**Expected candidate set:** `full region` → `left half`, `right half` → `fragments`  
**Observed candidate set:**  `full region` → `fragments` (halves missing)

The halves should appear because the contour map is correct: the outer border is
very high, the middle dividing ridge is high, and the internal fragment edges
are low. The expected merge order is:

1. Fragments within each half coalesce first (low internal ridges)
2. The two halves coalesce into the full region (high middle ridge)
3. The full region merges with background (very high outer border)

But step 2 never happens — the full region appears, and the fragments appear,
but the halves do not.

## Evidence Locations

### Contour map (primary evidence)

```
/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/2_nucleus/contour_maps.tif
```

Load in napari. Navigate to a problematic frame. Hover over:
- **Outer border pixels** — should be the brightest (highest value)
- **Middle ridge pixels** — should be intermediate brightness
- **Internal fragment edges** — should be the dimmest (lowest value)

Expected: middle ridge values > internal edge values. If any pixel on the middle
ridge dips to internal-edge levels, that is the leak.

### Foreground masks (companion evidence)

```
.../pos00/2_nucleus/foreground_maps.tif
```

Used as the binary mask for connected-component labeling before watershed.

### Ultrack candidate database

```
.../pos00/2_nucleus/ultrack_workdir/data.db
```

Query to verify missing intermediates:

```sql
SELECT id, t, t_hier_id, area, y, x
FROM nodes
WHERE t = <frame_index>
ORDER BY area DESC;
```

Expected: nodes for full region (area ~N), fragments (area ~N/8 to N/16),
and halves (area ~N/2). Observed: halves-area rows are absent.

### Ultrack source code (installed package)

All paths relative to the `cellflow` conda environment:

```
~/miniconda3/envs/cellflow/lib/python3.10/site-packages/ultrack/core/segmentation/
```

| File | Role |
|------|------|
| `processing.py` | `_process()` — orchestrates per-frame pipeline; calls `create_hierarchies()` |
| `hierarchy.py` | `create_hierarchies()` — labels CCs, removes small objects, oversegments by area, then `Hierarchy(c, **kwargs)` per CC |
| `vendored/hierarchy.py` | `Hierarchy.watershed_hierarchy()` — builds Higra tree; `compute_nodes()` — extracts candidates with area/frontier/height filters |
| `vendored/graph.py` | `mask_to_graph()` — builds the 4-connected (2D) or 6-connected (3D) graph; edge weight = `(image[p] + image[q]) / 2 + 1e-8` |

### CellFlow configuration wiring

| File | Lines | Role |
|------|-------|------|
| `src/cellflow/tracking_ultrack/config.py` | 46–56 | `TrackingConfig` fields: `seg_min_area=300`, `seg_max_area=100_000`, `seg_min_frontier=0.0`, `seg_ws_hierarchy="area"` |
| `src/cellflow/tracking_ultrack/ingest.py` | 107–112 | `_build_ultrack_config()` — maps TrackingConfig to Ultrack `SegmentationConfig` |
| `src/cellflow/tracking_ultrack/db_build.py` | `_run_ultrack_segment()` | Calls `ultrack.segment()` with the config above |

Parameters `max_noise`, `anisotropy_penalization`, and `random_seed` are **not**
mapped to TrackingConfig — they remain at Ultrack defaults:
- `max_noise = 0.0` — no noise added to contour map before watershed
- `anisotropy_penalization = 0.0` — no Z-axis penalty
- `random_seed = "frame"` — per-frame seed from frame index

## Root Cause: Ridge Leak

The watershed hierarchy builds a single binary partition tree per connected
component. A node exists in the tree only at a **merge event**: when two or more
basins merge at a given altitude in the edge-weight hierarchy.

For the halves to appear as a node:

> All fragments within each half must merge together **before** any fragment
> from the left half merges with any fragment from the right half.

This requires the middle ridge to be the highest-altitude edge crossed in the
entire region. If there is any **leak** — a single pixel or short path where the
contour value dips low enough on the middle ridge — the watershed crosses there
early, merging fragments from opposite halves before each half has internally
coalesced. The tree then has no node representing "just the left half" or "just
the right half."

### Why the leak happens

1. **Consensus averaging artifacts.** The contour map is produced by averaging
   `find_boundaries()` output across a threshold sweep. If Cellpose disagrees
   about the exact pixel placement of the ridge at different thresholds, the
   averaged value at that pixel drops, creating a low spot.

2. **Thin ridges (1–2 pixels).** The watershed graph connects 4-connected
   neighbors. A 1-pixel ridge has no width — basins on either side share a
   direct edge. Any dip in a single pixel on the ridge creates a low-weight
   crossing. There is no "ridge width margin" to absorb a single-pixel
   averaging artifact.

3. **No noise perturbation.** With `max_noise=0.0`, the exact same contour map
   is used every time. Small noise would perturb edge weights and can break
   ties that cause premature merging. Combined with a non-deterministic seed,
   multiple runs would produce different trees — some of which might preserve
   the halves.

## Diagnostic Steps

1. **Verify contour values (napari).** Hover over the middle ridge pixels at
   the problematic frame. If any pixel on the ridge has a value near the
   internal fragment edge level, that is the leak. The ridge should be
   uniformly higher than internal edges.

2. **Check area vs. max_area.** Compute pixel area of full region, halves, and
   fragments. If halves exceed `seg_max_area` (100,000) they'd be silently
   pruned — but this is impossible if the full region (strictly larger) survives.

3. **Inspect the Higra tree directly.** Insert debug logging into a patched
   `vendored/hierarchy.py:compute_nodes()` to dump the area of every internal
   node. Check whether nodes at the expected halves area exist in the tree but
   are filtered, or never exist in the tree at all.

4. **Re-run with different `seg_ws_hierarchy`.** Try `"dynamics"` or `"volume"`
   instead of `"area"`. Different merge orders may preserve the halves in at
   least one mode. If `"dynamics"` produces the halves, the issue is
   merge-order-dependent, confirming a leak or tie-breaking problem.

5. **Re-run with lower `seg_min_area`.** Set to 10 to rule out the
   small-object-removal step (`min_area/4 = 75` px default) silently dropping
   the halves.

## Possible Fixes

### Short-term (no code changes)

- Try all three `seg_ws_hierarchy` modes: `"area"`, `"dynamics"`, `"volume"`.
  Different merge orders may preserve the halves in at least one mode.
- Lower `seg_min_area` aggressively (50–100) to keep smaller intermediate nodes.
- Generate the contour map with a narrower threshold sweep or higher gamma to
  sharpen the middle ridge, reducing averaging artifacts.

### Medium-term (small code changes)

- **Expose `max_noise` in TrackingConfig.** Adding small noise (0.01–0.05) to
  the contour map before the watershed perturbs edge weights and can break ties
  that cause premature merging. Combined with per-run `random_seed`, multiple
  runs produce different trees.
- **Run DB generation multiple times with different modes/noise seeds** and
  merge the candidate pools into one `data.db`. The ILP solver handles
  redundant candidates gracefully.

### Longer-term (architecture)

The fundamental limitation is one tree per connected component. A richer
candidate pool could be produced by:

- Running the watershed with multiple noise seeds and merging results.
- Applying multiple thresholds to the contour map before the watershed,
  effectively eroding the ridge at different strengths.
- Bypassing Ultrack's segment for problematic regions and injecting synthetic
  candidate nodes derived from a separate watershed pass with different
  parameters.
- Using a multicut formulation that allows multiple overlapping segmentation
  hypotheses simultaneously, rather than a single hierarchical tree.

## Related

- `notes/2026-05-06-watershed-hierarchy-skips-intermediate-levels.md` — original
  diagnostic scratchpad with more detailed code traces.
- `archive/plans/2026-05-04-cell-segmentation-contour-maps.md` — contour map
  generation design.
- `archive/plans/2026-05-03-canonical-ultrack-nucleus-workflow.md` — workflow
  design that produces the contour maps consumed here.
