# Ultrack Size Filter Findings

The nucleus `ultrack` ingestion path in CellFlow does **not** perform the size
filter itself. CellFlow prepares `foreground.tif` and `contours.tif`, then
hands them to Ultrack via `segment(...)`.

## CellFlow-side handoff

- [`packages/ultrack/src/cellflow/ultrack/stages/tracking.py`](./packages/ultrack/src/cellflow/ultrack/stages/tracking.py)
  - `TrackingConfig.min_area`, `max_area`, `min_frontier`, and `ws_hierarchy`
    are forwarded into Ultrack’s `MainConfig`.
  - `run_segmentation(...)` and `run_hypothesis_ingestion(...)` both call
    `ultrack.core.segmentation.processing.segment(...)`.
- [`packages/ultrack/src/cellflow/ultrack/ingestion.py`](./packages/ultrack/src/cellflow/ultrack/ingestion.py)
  - This module only derives and writes `foreground.tif` / `contours.tif` from
    label hypotheses.

## Actual filtering in Ultrack

Installed Ultrack code in the `cellflow` conda env:

- `/home/aruppel/miniconda3/envs/cellflow/lib/python3.10/site-packages/ultrack/core/segmentation/processing.py`
  - `segment(...)` passes the size thresholds into `create_hierarchies(...)`:
    `max_area=config.max_area`, `min_area=config.min_area`,
    `min_frontier=config.min_frontier`.
- `/home/aruppel/miniconda3/envs/cellflow/lib/python3.10/site-packages/ultrack/core/segmentation/hierarchy.py`
  - `create_hierarchies(...)` removes tiny connected components with:
    `morphology.remove_small_objects(..., min_size=int(kwargs["min_area"] / 4), ...)`
  - This is an early pre-watershed size prune.
- `/home/aruppel/miniconda3/envs/cellflow/lib/python3.10/site-packages/ultrack/core/segmentation/vendored/hierarchy.py`
  - `Hierarchy.watershed_hierarchy()` applies the main candidate filtering:
    - `hg.filter_small_nodes_from_tree(tree, alt, self._min_area)`
    - `hg.simplify_tree(tree, hg.attribute_area(tree) > self._max_area)`
  - `Hierarchy.compute_nodes()` also skips nodes with `area[node_idx] > self._max_area`.

## Conclusion

If you are looking for the exact place where hierarchical-watershed candidates
are filtered by size, it is Ultrack’s
`core/segmentation/vendored/hierarchy.py`, especially
`Hierarchy.watershed_hierarchy()` and `Hierarchy.compute_nodes()`.

CellFlow only configures those thresholds and forwards the images.

## Alternative: Post-Segmentation DB Pruning

If the shape rule does not need to affect the watershed construction itself,
you can apply it after Ultrack segmentation and before linking/solve by
pruning the database in CellFlow.

This is viable because Ultrack’s linker and solver read directly from the SQL
tables:

- `nodes` (`NodeDB`)
- `overlaps` (`OverlapDB`)
- `links` (`LinkDB`)

What this approach would require:

1. Run Ultrack segmentation normally.
2. Query `NodeDB` and compute the shape descriptors from the stored node data.
3. Delete rejected node IDs from `NodeDB`.
4. Delete dependent rows from `OverlapDB`.
5. Clear `LinkDB` and any solve-state tables if linking or solve already ran.
6. Rerun linking and solve.

Why it works:

- `segment(...)` materializes the candidate set in the DB.
- `link(...)` and `solve(...)` consume the DB state directly.
- Existing helpers like `clear_linking_data()` remove links and solution state,
  but they do not delete candidate nodes.

Important caveat:

- Deleting only `NodeDB` is not enough.
- If you prune after linking, you must also rebuild links.
- If you prune after solve, you must clear the solution state too.

Bottom line:

- Upstream Ultrack changes are needed if the filter must participate in
  candidate generation.
- A DB-pruning step in CellFlow is enough if you only need the final candidate
  set cleaned up before linking/solve.
