# TODO

## ~~Resolve-from-validated route loses validated IDs after resolve~~ ✅ FIXED

The active route is `resolve_with_canonical_segment` in
`src/cellflow/tracking_ultrack/reseed.py` (called from the napari widget at
`src/cellflow/napari/nucleus_workflow_widget.py:3677`). It is *not* the older
`resolve_with_validation` — that one already has the ID-preservation logic.

After running "Re-solve from validated" the green validated overlay goes blank
and `validated_cells.json` no longer matches the labelmap, because the IDs in
the exported labelmap are Ultrack track numbers, not the user's original
validated `cell_id`s.

Three compounding bugs:

### 1. Validated IDs are never pasted back into the export
`resolve_with_canonical_segment` ends at `reseed.py:620-655` by returning the
raw output of `export_tracked_labels`. The ID-preservation step
`merge_validated_into_export` that `resolve_with_validation` calls at
`reseed.py:567` is missing here.

`export_tracked_labels` (`export.py:42-49`) uses Ultrack's `to_tracks_layer` /
`tracks_to_zarr`, which rasterizes with Ultrack's own sequential `track_id`s.
A cell validated as ID 42 ends up under whatever number Ultrack picked
(e.g. 7), so `validated_cells.json` keys no longer match anything in the
frame.

This is what makes `_refresh_validated_overlay`
(`nucleus_workflow_widget.py:3270`) show nothing useful after a resolve —
`np.isin(frame, validated_ids)` finds none of the original IDs.

### 2. The solver isn't told to honor the annotations
`reseed.py:649`:
```python
for _step, _total, label in run_solve(working_dir, cfg, overwrite=True):
```
`run_solve` defaults `use_annotations=False` (`solve.py:16`).
`inject_validated_nodes` sets `node_annot=VarAnnotation.REAL` on validated
nodes and `FAKE` on overlapping candidates (`validation_nodes.py:202, 222`),
but the ILP ignores those flags unless `use_annotations=True` is passed.
So validated nodes aren't even guaranteed to appear in the solution — the
ILP can pick a different overlap from the hierarchy.

For comparison, `resolve_with_validation` does pass `use_annotations=True`
(`reseed.py:557`).

### 3. The UI discards the only thing that could salvage it
`nucleus_workflow_widget.py:3608`:
```python
new_labels, _id_map = result
```
`resolve_with_canonical_segment` does compute `id_map` (`reseed.py:660-674`)
by voting which exported ID covers each validated cell's pixels most often.
The widget captures it as `_id_map` and never uses it — it neither relabels
`new_labels` back to original IDs nor updates `validated_cells.json` to
follow the renames.

## Suggested fix order

1. In `resolve_with_canonical_segment`, pass `use_annotations=True` to
   `run_solve` at `reseed.py:649` (one-line fix).
2. Call `merge_validated_into_export` on `new_labels` before returning, so
   original validated IDs are pasted back wholesale (mirrors
   `resolve_with_validation`'s contract).
3. Once (2) is done the `id_map` returned to the UI should normally be empty
   and the widget's `_id_map` discard is fine; the overlay will work because
   validated IDs survive the round-trip.

## Resolve-from-validated: track extension is broken

*See `notes/2026-05-06-resolve-validated-track-extension-debug.md` for full debugging notes.*

Despite recent fixes, track extension remains broken on real data (e.g., Track 93). Validated ranges stay correct, but surrounding frames retain spurious merges/splits.

### Diagnostic Tasks (Track 93)
- Run diagnostic script to reproduce resolve pipeline through linking + boost.
- Check if good candidates exist in segmentation hierarchy at t=7+.
- Check if `min_link_iou=0.1` is blocking links between REAL nodes and good candidates.
- Check if OverlapDB constraints are forcing big merges over small candidates.
- Verify `boost_validated_edges` is actually operating (not returning 0 links).

### Potential Fixes
- If candidates are missing: lower `seg_min_frontier` or inject synthetic extension nodes.
- If links are blocked: bypass `min_link_iou` for REAL nodes.
- If OverlapDB is the issue: cleanup OverlapDB for chains rooted in nodes conflicting with projected paths.

## Meta Analyzer (Long-term)

*See `notes/meta_analyzer_design_sketch.md` for the full design sketch.*

Implement the downstream exploratory layer for CellFlow.

### Phase 1: Source Browser
- Napari UI to discover and load analysis H5s/raw files.
- Overlay contacts/edges from analysis geometry.

### Phase 2: Resolved Tables
- Expose canonical lazy tables: `frames`, `cells`, `contacts`, `tracks`, etc.
- Implement identity annotation import/export and resolver.

### Phase 3: Metric Workbench
- Implement backend metric registry.
- Add Napari dock panel for cohort selection, running metrics, and visualization.
- Export support for tables (CSV) and plots (PNG/SVG).

## ~~Correction generates cell ID conflicts~~ ✅ FIXED

Correction-generated IDs now use a stack-wide fresh ID from the correction widget, with regression coverage for split and draw-new-cell paths.

## UI Polishing

The UI needs to be polished:
- Button placements
- Status labels
- Loading bars with higher resolution
- Etc.

## Local Extend Improvements

The local extend greedy approach could be improved:
- Better scoring functions: IoU, centroid-corrected IoU, shape similarity, etc.
- More power: ability to overwrite, split, or merge if it improves the scoring function.
- Should likely include nearby cells in the evaluation because merges and splits affect them directly.
- Requires careful thought and design.

## Widget Robustness

- The correction widget throws an error when a layer it uses is deleted. It should gracefully deactivate or update instead of crashing.

## Nucleus Data Processing

- Implement Z-averaging (zavg) for nucleus probability and flow, mirroring the implementation for cell probability and flow in the widget. This average image is beneficial for thresholding and will be useful for nuclei as well.

## Foreground Mask Subwidget

Create a subwidget to generate a foreground mask with the following features:
- Two parameters: **threshold** and **gamma correction**.
- Input: uses either **sigmoid-transformed cell/nucleus probability** or **flow DP** (derivative of probability/density).
- Uses **Z-averaging** (similar to the logic in the cell segmentation widget).

## Database Browser Enhancements

Enhance the database browser with the following:
- Control node transparency based on the node probability.
- Add links and control their transparency based on their weights.
- Sliders/controls should have discrete spots corresponding to actual values in the database (i.e., only show available range).

## Database Generation

- Figure out if the granularity of the parameter sweep inside the database generator can be finetuned.
- Investigate why solve and database generation duplicate many parameters. Determine which ones are used and which ones are ignored.

## Parameter Investigation

- Investigate what "seed space (px)" actually does and if it makes any sense.
