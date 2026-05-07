# Watershed Hierarchy Skips Intermediate Merge Levels

Date: 2026-05-06

## Symptom

The Ultrack watershed hierarchy produces candidate nodes that jump directly from
a full connected-component region to small internal fragments, without producing
intermediate nodes that represent the intuitively obvious subdivisions (e.g., a
region that clearly divides into two halves, then each half into smaller
fragments).

Expected candidate set:  `full region` → `left half`, `right half` → `fragments`
Observed candidate set:   `full region` → `fragments`  (halves missing)

The contour map values are correct — the outer border is very high, the middle
dividing ridge is high, and the internal fragment edges are low — yet the
halves never appear as candidates.

## Evidence location

### Contour map

`2_nucleus/contour_maps.tif` — the consensus boundary image produced by the
Contour Maps stage.  Load in napari and inspect the problematic frame.
Expected: middle ridge pixels lighter (higher value) than internal fragment edges.

### Ultrack candidate database

Run DB generation from the UI or via:

```python
from cellflow.tracking_ultrack.db_build import build_ultrack_database
from cellflow.tracking_ultrack.config import TrackingConfig

build_ultrack_database(
    contour_maps_path="2_nucleus/contour_maps.tif",
    foreground_masks_path="2_nucleus/foreground_masks.tif",
    nucleus_prob_zavg_path="1_cellpose/nucleus_prob_zavg.tif",
    working_dir="2_nucleus/ultrack_workdir",
    cfg=TrackingConfig(),
)
```

Then inspect `2_nucleus/ultrack_workdir/data.db`:

```sql
-- Find candidate nodes at the problematic frame
SELECT id, t_node_id, t_hier_id, area, y, x
FROM nodes
WHERE t = <frame_index>
ORDER BY area DESC;
```

Verify: nodes for the full region and fragments exist, but none at the expected
halves area (~half the full region).

### Ultrack source (installed package)

The relevant code is in the `cellflow` conda environment:

```
~/miniconda3/envs/cellflow/lib/python3.10/site-packages/ultrack/core/segmentation/
```

- `processing.py` — `_process()` orchestrates the per-frame pipeline; lines ~58–80
  create the hierarchy from `foreground[time] > threshold` and the contour map.
- `hierarchy.py` — `create_hierarchies()` runs `ndi.label` → `remove_small_objects`
  → `oversegment_components` → per-CC `Hierarchy(c, **kwargs)`.
- `vendored/hierarchy.py` — `watershed_hierarchy()` builds the Higra tree,
  filters by `min_area`/`max_area`/`min_frontier`; `compute_nodes()` extracts
  candidate nodes from the tree.
- `vendored/graph.py` — `mask_to_graph()` builds the graph from contour values;
  edge weight = `(image[p] + image[q]) / 2 + 1e-8`.

### CellFlow wiring

- `src/cellflow/tracking_ultrack/db_build.py` — `_run_ultrack_segment()` calls
  Ultrack's `segment()`.
- `src/cellflow/tracking_ultrack/ingest.py` — `_build_ultrack_config()` maps
  `TrackingConfig` fields to Ultrack's `SegmentationConfig`.  Lines 107–112
  set: `min_area`, `max_area`, `threshold`, `min_frontier`, `ws_hierarchy`,
  `n_workers`.  Parameters `max_noise`, `anisotropy_penalization`, and
  `random_seed` are **not** mapped (left at Ultrack defaults: 0.0, 0.0, "frame").

## Root cause hypothesis

The watershed hierarchy builds a single binary partition tree per connected
component.  A node exists in the tree only at a merge event: when two (or more)
basins merge at a given altitude in the edge-weight hierarchy.  For the halves
to appear as a node, all fragments within each half must merge together *before*
any fragment from the left half merges with any fragment from the right half.

This requires the middle ridge to be the highest-altitude edge crossed in the
entire region.  If there is any "leak" — a single pixel or short path where the
contour value dips low enough on the middle ridge — the watershed crosses there
early, merging fragments from opposite halves before each half has internally
coalesced.  The tree then has no node representing "just the left half" or
"just the right half."

Possible causes of a leak:

1. **Noise or averaging artifacts in the consensus boundary.**  The contour map is
   produced by averaging `find_boundaries()` output over a threshold sweep.  If
   Cellpose disagrees about the exact pixel placement of the ridge at different
   thresholds, the averaged value at that pixel drops, creating a low spot.

2. **The ridge is thin (1–2 pixels).**  The watershed graph connects 4-connected
   (2D) or 6-connected (3D) neighbors.  A 1-pixel ridge has no "width" — the
   basins on either side share a direct edge.  Any dip in a single pixel on the
   ridge creates a low-weight crossing.

3. **`oversegment_components` splits at an unfortunate location.**  If the
   connected component exceeds `max_area` (default 100 000), Ultrack pre-splits
   it using a watershed-by-area cut before building the per-CC hierarchies.
   If this cut happens along the middle ridge, the two halves end up in
   separate hierarchies and the full-region-with-halves tree never forms.
   (This is less likely if the full-region node *does* appear, since that node
   would need both halves in one hierarchy.)

## Diagnostic steps

1. **Verify contour values.**  In napari, hover over the middle ridge pixels
   at the problematic frame.  The values should be *higher* (lighter) than
   the internal fragment edges.  If any pixel on the ridge is low, that's the
   leak.

2. **Check area vs. max_area.**  Compute the pixel area of the full region,
   the expected halves, and the fragments.  Compare to the DB Gen `max_area`
   setting.  If the halves exceed `max_area` they'd be silently pruned — but
   this is mutually exclusive with the full region surviving, since the full
   region is strictly larger.

3. **Inspect the Higra tree directly.**  Insert debug logging into a patched
   `vendored/hierarchy.py:compute_nodes()` to dump the area of every internal
   node in the tree.  Check whether nodes at the expected halves area exist in
   the tree but are filtered, or never exist in the tree at all.

4. **Re-run with different `ws_hierarchy` mode.**  If `area` skips the halves,
   try `dynamics` — the merge order changes.  If `dynamics` produces the halves,
   the issue is merge-order-dependent, consistent with a leak or tie-breaking
   problem.

5. **Re-run with lower `min_area`.**  Set `min_area` very low (e.g. 10) to
   rule out the small-object-removal step (`min_area/4` = 75 px default)
   silently dropping the halves.

## Possible fixes

### Short-term (no code changes)

- Try all three `ws_hierarchy` modes (`area`, `dynamics`, `volume`).  Different
  merge orders may preserve the halves in at least one mode.
- Lower `min_area` aggressively (50–100) to keep smaller intermediate nodes.
- Generate the contour map with a narrower threshold sweep or higher gamma to
  sharpen the middle ridge.

### Medium-term (small code changes)

- Expose `max_noise` in `TrackingConfig` and `_build_ultrack_config`.
  Adding small noise (0.01–0.05) to the contour map before the watershed
  perturbs edge weights and can break ties that cause premature merging.
  Combined with `random_seed=None`, multiple runs produce different trees.
- Run DB generation multiple times with different `ws_hierarchy` modes (or
  different noise seeds) and merge the candidate pools into one `data.db`.
  The ILP solver handles redundant candidates.

### Longer-term (architecture)

- The fundamental limitation is one tree per connected component.  A richer
  candidate pool could be produced by:
  - Running the watershed with multiple noise seeds and merging results.
  - Applying multiple thresholds to the contour map before the watershed
    (effectively eroding the ridge at different strengths).
  - Bypassing Ultrack's segment for problematic regions and injecting
    synthetic candidate nodes derived from a separate watershed pass with
    different parameters.
