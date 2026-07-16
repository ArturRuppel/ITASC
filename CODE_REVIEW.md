# ITASC — Codebase Review

_Correctness · Architecture · Performance — deep multi-agent review, July 2026_

Reviewed at commit `4ede7a2` on branch `claude/codebase-review-f18q9d`.

## How this review was done

The `src/itasc` tree (~26k LOC of non-test Python; the four `packages/*` are
distribution shims over the same source) was split into **14 subsystem slices**.
Each slice was read in full by an independent reviewer hunting correctness bugs,
architecture problems, and performance issues. Every *falsifiable* finding
(correctness + performance — 51 of them) was then handed to a separate **adversarial
verifier** instructed to refute it by reading the actual code paths and callers;
architecture findings (judgment calls) were passed through unverified. A final
synthesis pass identified cross-cutting themes.

**Outcome:** 58 candidate findings → **3 refuted** → **55 kept**
(42 independently confirmed, 6 plausible, 7 unverified-architecture).

| Severity | Count | | Dimension | Count |
|---|---|---|---|---|
| High | 7 | | Correctness | 24 |
| Medium | 19 | | Performance | 24 |
| Low | 29 | | Architecture | 7 |
| **Total** | **55** | | | |

## Executive summary

ITASC is a mature, thoughtfully engineered scientific plugin. Across all twelve reviewed subsystems the numeric cores are careful and well-documented: coordinate/axis conventions (row=pos[-2], col=pos[-1], [dy,dx] flows, TZYX canonicalization), divide-by-zero and empty/single-cell/NaN guards, content-addressed caching, union-find z-stitching, atom merge-tree machinery, and dtype-safe label commits all hold up under how callers actually use them. The reviewers refuted a substantial fraction of candidate bugs, and what survived is concentrated in edge-case handling, scalability, and UI state-sync rather than in the fundamental algorithms. This is the profile of a codebase written by domain experts who understand the science, not a fragile prototype.

The dominant risk theme is scalability on exactly the datasets the plugin advertises as its target: dense, motile monolayers with hundreds-to-thousands of cells across hundreds of frames. Several subsystems that are correct at small scale degrade or crash at production scale — a ~60 GB RAM blowup in geodesic label assignment, unbounded SQLite IN-lists that raise OperationalError on large time-lapses, O(N^2) pure-Python pair loops in collective-motion metrics, and multiple heavy operations (full-stack reads, regionprops, EDTs) run redundantly or synchronously on the GUI thread. These are not exotic edge cases; they are the mainline use of the tool.

The second, more insidious theme is silent scientific corruption: a handful of correctness bugs that produce plausible-looking but wrong output with no error. Validated cell tracks fragment when a solver track id numerically collides with the validated cell id (frequent, since ultrack numbers from 1); a validated mask can hijack a spatially-unrelated candidate at zero IoU; the -1.0 node_prob sentinel leaks into browser probabilities; edge flicker manufactures phantom T1 events that deepen the apparent potential well; and 4-D inputs with fewer frames than z-slices silently transpose T and Z, corrupting every track. Because these are silent, a researcher can publish on them.

The third theme is UI/state fragility in the napari layer: unsaved nucleus-correction edits discarded on deactivate (dirty flag inferred from status strings), a "cancel" that does not cancel and can spawn concurrent writers to the same output file, recompute showing stale overlays, and large setattr aliasing blocks that couple widgets invisibly. The underlying pipeline is sound; the interactive surface around it is where data-loss and trust-eroding behaviors cluster. Overall health is good, with a clear and actionable priority list dominated by scale-hardening and a small set of silent-correctness fixes.

## Top risks (ranked)

The issues most likely to produce **wrong scientific results or silent data loss**:

1. Validated cell tracks silently fragment when the solver's exported track id numerically equals the validated cell id (reseed.py:372). Since ultrack numbers tracks from 1, this collision is frequent for small ids: a validated cell's continuation is scattered to a fresh disconnected id, splitting a hand-verified track in two with no error. Directly corrupts the ground-truth the user just certified.
2. Validated mask hijacks a spatially-unrelated candidate at zero IoU (validation_nodes.py:106). When no free candidate matches, the code overwrites the lowest-id unrelated candidate instead of taking the safe reserved-node fallback, so the validated cell inherits spatially-wrong incoming/outgoing links — corrupting a track precisely through an anchor the user trusts most.
3. 4-D inputs with fewer timepoints than z-slices are silently transposed (shape.py:35). A genuine (3 timepoints × 5 z) acquisition becomes T=5/Z=3, so tracking stitches across timepoints and links across z-slices, corrupting every track while producing plausible-looking output.
4. Geodesic label assignment OOMs on modest real runs (~60 GB for T=50/300-cells/1024^2; cell_label_icm.py:211), and the sibling all-zero-nucleus crash (line 437) and full-stack RAM loads in db_build.py:291 mean the core segmentation/tracking build can abort or be killed on exactly the large monolayers the plugin targets — no output, or a hard failure mid-run.
5. Unsaved nucleus-correction edits are silently discarded on deactivate (nucleus_correction_widget.py:2079). Because the dirty flag is inferred from status strings and hand-draw/erase/extend never set it, toggling correction off skips the save prompt and reloads from disk, destroying in-memory correction work with no warning — direct, silent data loss.
6. 'Cancel' on a cell run does not cancel and can corrupt output (cell_workflow_widget.py:1286). The worker keeps running, writes tracked_labels.tif, and — because the gate resets to idle — a re-run spawns a second concurrent worker; both write the same path and a late-finishing stale worker can overwrite newer results with old labels.
7. Recompute displays stale contact-analysis overlays (contact_analysis_widget.py:853). The .h5 on disk updates but the on-screen edges/tracks/labels keep showing pre-recompute data while the status reports success, so a scientist reads and may act on wrong data that looks authoritative.
8. Edge flicker and greedy/fragmented T1 detection bias the derived effective potential (contacts/build.py:220, 573, 485, signed_contact_length.py:130). Phantom paired forward/reverse events and misattributed re-contacts inject spurious samples near L=0, artificially deepening the well and distorting the deltaE_eff barrier landscape — a silent bias in the scientific conclusion, not a crash.
9. The -1.0 node_prob sentinel leaks into displayed probabilities and summary stats (db_query.py:863), so hover tooltips show p=-1.000, the transparency overlay normalizes against a bogus floor collapsing real contrast, and summary min/mean/median are dragged negative — the browser misrepresents probabilities for the normal DB state, misleading curation decisions.
10. The documented headless aggregate(params={...}) contract crashes for pixel_size/time_interval (records.py:54), and a blank catalog column silently disables physical-unit quantifiers (pipeline.py:302) — batch users either get an aborted run or silently missing shape/dynamics tables with valid config.

## High-severity findings (detail)

### H1. Validated cell's track is fragmented when solver track id equals the validated cell id
`src/itasc/tracking_ultrack/reseed.py:372` — **correctness** · verdict: confirmed

**What:** In merge_validated_into_export, when the dominant solver track overlapping a validated mask has id == cell_id (the 'solver already used the correct ID' branch, lines 370-375), the id is deliberately NOT added to solver_track_remap. Execution then falls through to the reserved-id collision loop (lines 408-432): solver_collision = (exported_labels == cell_id) selects the ENTIRE solver track, only the exact validated-frame mask pixels are cleared (lines 418-425), and every remaining pixel of that same track is relabeled to a single fresh id (line 432). Because the whole-track rename at lines 435-438 only runs for entries in solver_track_remap, the continuation at non-validated frames is never restored. The sibling function apply_post_solve_corrections in corrections.py handles this correctly by skipping the scatter when owning_solver == cell_id (lines 1168-1172); reseed.py lacks that guard.

**Impact:** User validates cell 3 at frames 0-5; the re-solve exports a track with id 3 that correctly covers frames 0-20. dominant_solver_id == 3 == cell_id, so frames 6-20 (all pixels of value 3 outside the validated frames) are scattered to one fresh id (e.g. 5000). Output: cell 3 exists only at frames 0-5 and its continuation becomes a disconnected track 5000. The validated cell's track is split in two. Highly likely for small cell ids since ultrack numbers exported tracks from 1.

**Fix:** Treat dominant_solver_id == cell_id the same as a remap whose source and target coincide: skip the collision-scatter entirely for that cell_id (mirror the owning_solver == cell_id early-continue in corrections.py apply_post_solve_corrections), so the whole solver track that already carries the correct id is preserved and only the validated frames are (harmlessly) re-pasted.

### H2. pixel_size/time_interval satisfy the build gate via `params` but are delivered to compute as `None`, aborting aggregate
`src/itasc/contact_analysis/records.py:54` — **correctness** · verdict: confirmed

**What:** The build-param gate and the actual value source disagree. `_position_frame` (shape_tables.py:195) gates a quantifier with `quantifier.missing_build_params(record_build_params(quantifier, record, params))`, and `record_build_params` (records.py:104-108) merges the shared `params` dict, so a `params`-only `pixel_size_um`/`time_interval_s` satisfies the gate. But the value actually consumed comes from `inputs.pixel_size_um`/`inputs.time_interval_s`, and `position_inputs_from_record` (records.py:54-55) populates those *only* from `record.get(...)` — the `params` mapping is never threaded into `PositionInputs`. The cheap quantifiers read them from `inputs`, not `params`: `CellShapeQuantifier.compute_object_table` (cell_shape.py:29-32) calls `compute_object_shape(pixel_size_um=inputs.pixel_size_um)`, and `CellDynamicsQuantifier`/`NucleusDynamicsQuantifier` (cell_dynamics.py:78-81) call `instantaneous_table(time_interval_s=inputs.time_interval_s)`. (Only `cell_density` correctly reads `fov_area_mm2` from `params`.)

**Impact:** Calling the public API `pipeline.aggregate(catalog, params={"pixel_size_um": 0.5})` — exactly what the aggregate/build_table docstrings say is supported — on a catalog whose records do not already carry `pixel_size_um` passes the gate but reaches `compute_object_shape` with `pixel_size_um=None`, hitting `float(None)` and raising `TypeError`, which aborts the entire aggregate run (no tables written). Same for `time_interval_s=None` in `instantaneous_table` for the dynamics tables. The napari studio happens to avoid this because it derives `params` *from* the records (aggregate_widget.py:207-213), so records always carry the value; the bug is latent on the documented headless path.

**Fix:** Thread `params['pixel_size_um']`/`params['time_interval_s']` into `PositionInputs` in `position_inputs_from_record` (or in `_position_frame`) as a fallback when the record lacks them, so the value that passes the gate is the value compute receives.

### H3. Nucleus correction loses unsaved edits: dirty flag inferred from status strings, and most edit paths never set it
`src/itasc/napari/correction/nucleus_correction_widget.py:2079` — **correctness** · verdict: confirmed

**What:** The nucleus widget's `_correction_dirty` flag is only ever set True as a side effect of `_correction_status()` seeing the substring 'unsaved' (see _correction_layer_lifecycle.py:132). `_on_cells_edited` (line 2076-2079, the callback for every manual mouse edit — draw/erase/split/merge/add/carve made via the embedded CorrectionWidget) emits `labels_edited` but never sets the flag, and those inner edits report through the embedded widget's own hidden status label (`_correction_ui_nucleus.py:216`), never through `_correction_status`. `_on_extend` (line 886-889) and `_on_swap_step` (line 976-979) likewise mutate `layer.data` but produce status text WITHOUT the word 'unsaved'. So a session consisting of manual mask edits, A/D extends, and/or Z/C swaps leaves `_correction_dirty == False`.

**Impact:** User activates nucleus correction, hand-draws/erases cells and/or extends a track with A/D, then clicks the correction toggle off. `_confirm_deactivate_with_unsaved_changes` (line 1486) sees `_correction_dirty == False`, returns True with no save prompt, and deactivate then calls `_refresh_tracked_layer_from_disk()` + `_remove_correction_owned_layers()` (line 1463-1464), discarding all the in-memory edits to '[Correction] Nucleus Labels'. All of that correction work is lost silently. The cell widget does this correctly via `_on_correction_edit` setting `self._correction_dirty = True` (cell_correction_widget.py:649-651); the nucleus widget's `_on_cells_edited` is the analogous hook but omits it.

**Fix:** Set `self._correction_dirty = True` in `_on_cells_edited`, and in every layer-mutating handler (`_on_extend`, `_on_swap_step`/`_apply_swap`, `_paint_extend_assignment`, `_on_retrack`, `_on_reassign_ids_done`, `_on_remove_unvalidated_labels`). Stop relying on status-string substrings to derive dirtiness.

### H4. Cancelling a cell segmentation run does not cancel it — the run still writes output, and re-running spawns a concurrent worker on the same file
`src/itasc/napari/cell_workflow_widget.py:1286` — **correctness** · verdict: confirmed

**What:** `_on_run`'s worker (lines 1340-1356) calls `segment_cells_divergence(contours, foreground, nuc, params, progress_cb=_cb)` with NO cancel callback (the function has no `cancel_cb` parameter — verified in cell_divergence_segmentation.py line 267), and the worker body has no `yield`, so it is a non-generator FunctionWorker. `_on_cancel` (line 1286) calls `self._run_worker.quit()`, sets `_run_worker=None`, `_running=False`, `_set_run_idle()` (which does `gate.set_task('cell_run', False)`), and status 'Cancelled.' — but `quit()` cannot interrupt a FunctionWorker with no cancellation point, and the worker's `returned`→`_done` connection is never disconnected.

**Impact:** User starts a full cell run on a large movie, clicks the ✕ 'Cancel' button, and sees 'Cancelled.' with the run row returned to idle. The geodesic walk keeps running in the background; when it finishes, `_done` fires, `commit_labels` writes `3_cell/tracked_labels.tif` to disk, the layer appears, and the status flips to 'Segmentation complete' — so a 'cancelled' run silently produces output. Worse: because the gate is reset to idle, the user can click Run again while the old worker is still computing; two workers then run concurrently and both call `commit_labels` on the same `output_path`, and the later-finishing stale worker can overwrite the newer result with old labels.

**Fix:** Either make the run genuinely cancellable (thread a `cancel_cb` through `segment_cells_divergence` and poll it, using a GeneratorWorker so `quit()` works), or if it is non-cancellable keep the run button disabled during the run and guard `_done`/`_error` against a superseded worker (ignore the callback unless it came from the current `self._run_worker`).

### H5. Recompute displays stale contact-analysis overlays (fast-path skip not invalidated after rebuild)
`src/itasc/napari/contact_analysis_widget.py:853` — **correctness** · verdict: confirmed

**What:** _show_from_disk has an 'already shown' fast path: if `signature == self._displayed_contact_analysis_signature and self._contact_analysis_layer_names()` it returns without re-adding layers (lines 853-859). The signature is `(contact_analysis_path, *display-option flags)`. After a Recompute (overwrite=True), `_on_compute_done` (line 619) sets `self._cached_contact_analysis = None` (so the fresh .h5 is re-read) and then calls `_show_from_disk()`, but it never clears `_displayed_contact_analysis_signature` and does not call `_clear_contact_analysis_layers` on this path. When the position and the three display-option checkboxes are unchanged (the normal case), the recomputed signature equals the previously displayed one and the old overlay layers are still present, so _show_from_disk takes the skip branch and `add_contact_analysis_layers` is never called with the fresh data.

**Impact:** User edits inputs, clicks 'Run Contact Analysis' (Recompute) to refresh a stale .h5; the build succeeds and the file on disk updates, but the on-screen edges/tracks/labels keep showing the PRE-recompute result. The status even reports success ('showing <name>'), so the wrong data looks authoritative. Only a position switch or a display-option toggle forces the overlays to actually update.

**Fix:** In _on_compute_done (or specifically when `built`/overwrite), set `self._displayed_contact_analysis_signature = None` before calling `_show_from_disk()`, or have the overwrite path route through `_clear_contact_analysis_layers(set_status=False)` first so the fast-path guard cannot short-circuit a freshly rebuilt result.

### H6. Full geodesic label assignment blows up memory: dense per-(frame,label) arrays held in RAM
`src/itasc/segmentation/cell_label_icm.py:211` — **performance** · verdict: confirmed

**What:** _compute_frame_geodesic stores one full-frame (Y,X) float32 geodesic array per alive label (raw[k]=d, then result[k]=nd), and _compute_geodesic_unaries / _compute_geodesic_unaries_parallel accumulate every frame's dict into a single unary dict that is held simultaneously (lines 300-316, 356-381) and returned to initialize_icm. The 'streaming argmin' in _argmin_init_from_dict only avoids allocating ONE additional (T,Y,X,K) volume; the unary dict itself IS that dense volume restricted to alive entries. For the stated domain (dense motile monolayers) nearly every cell is alive every frame, so total resident size is ~T * cells_per_frame * Y * X * 4 bytes. Each cell's cost is stored over the whole frame instead of a local bounding box, even though a nucleus only influences its neighbourhood.

**Impact:** A modest run of T=50 frames, ~300 cells/frame, 1024x1024 produces ~15000 full-frame float32 arrays ≈ 60 GB resident (plus the same data written to the HDF5 cache), OOM-killing the full-stack run driven from cell_workflow_widget._worker (line 1350) that reads the entire contours/foreground/nuc stacks and calls segment_cells_divergence with_labels=True.

**Fix:** Store each label's unary only over its bounding-box crop (with the pixel offset), or compute the running argmin frame-by-frame and discard each frame's dicts immediately instead of accumulating the whole movie; do not hold all (frame,label) full-frame arrays at once.

### H7. Collective metrics use O(N^2) pure-Python pair iteration and recompute the NN matrix twice
`src/itasc/contact_analysis/dynamics/collective.py:138` — **performance** · verdict: confirmed

**What:** _velocity_correlation builds all n*(n-1)/2 pairs via triu_indices and then iterates over every pair in a pure-Python loop (lines 138-140) to bin dots/counts. _median_nn_distance (lines 203-204) materializes a full n*n float64 distance matrix per frame. On top of that, _resolve_bin_width (line 113) computes the per-frame median NN distance for every frame, and the main loop (line 67) computes exactly the same _median_nn_distance again for every frame, so the most expensive O(N^2) step runs twice per frame.

**Impact:** The module docstring and task both describe DENSE motile monolayers (hundreds to thousands of cells per frame). For N=3000 cells and T=500 frames, the Python loop executes ~4.5e6 iterations per frame (2.2e9 total) and each frame transiently allocates a 3000x3000 (~72 MB) distance matrix computed twice; a full run takes minutes-to-hours and can spike memory, where a vectorized version (np.bincount on the bin indices, cKDTree for nearest-neighbour) would be seconds.

**Fix:** Replace the per-pair Python loop with np.bincount(bins, weights=dots) and np.bincount(bins) to accumulate dot-sums/counts. Use scipy.spatial.cKDTree.query(k=2) for nearest-neighbour distances instead of the full N^2 matrix. Compute each frame's median NN distance once and reuse it for both the bin-width default and the nn_distance_um column.

## Cross-cutting architecture themes

### Correct-at-small-scale algorithms lack scale guards for the plugin's own target datasets  
_severity: high_

Multiple subsystems are frame-local or per-track correct but hold whole stacks in RAM or emit unbounded SQL IN-lists: geodesic label assignment materializes ~15000 full-frame float32 arrays (~60 GB) for a modest T=50/300-cells/1024^2 run (cell_label_icm.py:211); build_atom_union_database loads full int32 atom + float32 contour stacks (~16 GB combined) despite being frame-local (db_build.py:291); track_quality and corrections build .in_(node_ids) lists over ~100k nodes that hit SQLite's variable ceiling and raise OperationalError (track_quality.py:81, corrections.py:137). The dense-monolayer use case the code explicitly targets is precisely where these break.

### Silent scientific corruption: wrong results with no error and a success message  
_severity: high_

A recurring failure mode is producing plausible-but-wrong output that reports success. Validated tracks fragment on solver-id/cell-id collision (reseed.py:372); a validated mask hijacks an unrelated candidate at zero IoU (validation_nodes.py:106); the -1.0 node_prob sentinel leaks into displayed probabilities and summary stats (db_query.py:863); edge flicker fabricates paired phantom T1 events that deepen the potential well (contacts/build.py:220); 4-D inputs with T<Z silently transpose axes and corrupt every track (shape.py:35). None raise, so a researcher cannot tell from output that anything is wrong.

### Heavy work left on the Qt/GUI main thread despite an established off-thread pattern  
_severity: medium_

The codebase demonstrably knows how to offload (workers, progress signals, UiGate serialization), yet several hot paths bypass it: full atom extraction runs the residual+watershed pipeline synchronously (nucleus_atom_extraction_widget.py:602), cellpose preview loads the entire TIFF on the GUI thread (cellpose_widget.py:673), and contact-analysis Visualize does large label reads + O(T·H·W) centroid extraction on the main thread (contact_analysis_widget.py:822) — the last even carries a comment acknowledging it freezes the UI. The inconsistency, not ignorance of the pattern, is the architectural smell.

### Dirty-state and cancellation tracking is inferred rather than authoritative  
_severity: high_

Session state is derived from side-channels instead of set at the mutation point. The nucleus correction widget infers 'unsaved changes' from status-message substrings, so real edit paths (hand-draw/erase, extend) never mark the session dirty and are silently discarded on deactivate (nucleus_correction_widget.py:2079), while Commit fails to clear the flag causing spurious prompts (line 1267). The cell run's Cancel resets the gate to idle but does not stop the worker, which still writes output and permits a second concurrent writer to the same file (cell_workflow_widget.py:1286). Authoritative flag-on-edit and real cancellation tokens are the missing discipline.

### Redundant recomputation in hot loops (EDTs, regionprops, centroids, full-stack scans)  
_severity: medium_

Several inner loops recompute invariants: the shape linker re-argwhere-scans each target ~neighbor_count times instead of once (linking.py:255); flow-following recomputes a full-image EDT per shell iteration (flow_following.py:168); clean_stranded_pixels ties expand_labels distance to fragment pixel count, forcing near-full-frame EDTs per fragment (labels.py:956); co-sourced quantifiers re-read TIFFs and re-run regionprops per position (shape_relational.py:30); _per_frame_views does a full boolean table scan per frame, O(F^2) (collective.py:100). Individually low-severity, collectively they turn seconds into minutes at scale.

### Large widgets coupled via setattr aliasing and god-object mixins  
_severity: medium_

Workflow widgets rebind ~40 sub-widget methods/attributes onto self via setattr loops (nucleus_workflow_widget.py:454), so renaming a source method breaks the alias with no import-time error and hides the true owner at the call site. The DB browser mixin (nucleus_db_browser_widget.py:235) fuses DB math, rasterization, and graph layout into the UI class with ~8 undeclared helper dependencies and duplicated alpha-normalization that can drift. This fragile hidden coupling is where the state-sync bugs are easy to introduce and hard to catch.

### Provenance and error records lose information at failure/report boundaries  
_severity: low_

Failure and provenance paths under-record: StageLogger logs only str(exc_val) so a headless overnight failure records 'Stage failed: 0' with no file/line/frame (logging.py:50); DivergenceMapsReport records foreground filter params as 0 even when filtering was applied (divergence_maps.py:361). For a batch/headless scientific pipeline, unreproducible provenance and undiagnosable failures are a real operational cost.

### Documented headless API contract diverges from the interactive path that is actually exercised  
_severity: medium_

The napari studio derives params from records, so records always carry pixel_size/time_interval — masking a latent bug on the documented headless path: pipeline.aggregate(catalog, params={'pixel_size_um': 0.5}) passes the build gate (which consults params) but delivers None to compute_object_shape, raising TypeError and aborting the whole run (records.py:54). A blank pixel-size column can also silently disable physical-unit quantifiers via setdefault (pipeline.py:302). The tested path and the documented path have diverged.

## Medium-severity findings

| # | Dim | Title | Location | Impact (short) |
|---|---|---|---|---|
| M1 | corr | do_3d flow safety guard silently fails when the nucleus volume has exactly 2 z-slices | `src/itasc/cellpose/cellpose_runner.py:382` | A user runs nucleus Cellpose with do_3d=True (exposed via nuc_3d_chk in cellpose_widget.py:531) on a 2-z-slice stack, then runs the divergence maps pipeline. In… |
| M2 | corr | 4-D input axis identity is guessed from axis lengths, silently swapping T and Z for movies with fewer frames than z-slices | `src/itasc/cellpose/shape.py:35` | A legitimate acquisition of 3 timepoints x 5 z-slices (shape (3,5,Y,X)) is silently transposed to T=5, Z=3: track_axiswise then stitches across what are really … |
| M3 | corr | Collective alignment/correlation silently include gap-spanning (multi-frame) velocities as if instantaneous | `src/itasc/contact_analysis/dynamics/collective.py:103` | In gappy tracking (common in dense monolayers), a cell crossing a k-frame gap contributes a chord direction whose random single-step components have averaged ou… |
| M4 | corr | Edge flicker across frames generates spurious and reversed T1 events | `src/itasc/contact_analysis/contacts/build.py:220` | On real dense-monolayer tracking with per-frame boundary noise, the t1_events table is inflated with paired forward/reverse phantom events; each phantom contrib… |
| M5 | corr | Non-touching merge creates disconnected components that clean_stranded_pixels silently deletes; the exclude safeguard is never wired up | `src/itasc/correction/labels.py:426` | User merges two non-adjacent cells belonging to one track (e.g. a fragmented nucleus) → one label with two blobs. Later the user triggers 'Clean fragments' (a g… |
| M6 | corr | Background reassign/annotate workers alias live layer.data while it stays user-editable | `src/itasc/napari/correction/nucleus_correction_widget.py:617` | If the user paints while a reassign/annotate runs: (a) their concurrent edits are silently overwritten when `layer.data = remapped` lands; and (b) if a paint in… |
| M7 | corr | Extend action paints labels without recording undo history | `src/itasc/napari/correction/nucleus_correction_widget.py:871` | User presses D to extend a track one frame forward, sees it went to the wrong cell, and presses Ctrl+Z (advertised in the shortcuts panel as 'Undo'). The extend… |
| M8 | corr | IndexError crash when the nucleus stack has no labels | `src/itasc/segmentation/cell_label_icm.py:437` | Running the segmentation stage before nucleus tracking has produced anything (nuc TIFF all zeros) crashes with 'index 0 is out of bounds for axis 0 with size 0'… |
| M9 | corr | Validated mask hijacks a spatially-unrelated candidate at zero IoU | `src/itasc/tracking_ultrack/validation_nodes.py:106` | A frame with 5 candidates and 1 validated cell located far from all of them (e.g. hierarchy shifted after re-segmentation, or the true node was consumed by a be… |
| M10 | perf | clean_stranded_pixels ties expand_labels distance and crop size to fragment PIXEL COUNT, causing near-full-frame work per fragment | `src/itasc/correction/labels.py:956` | On a 2048x2048 frame, a single stranded fragment of ~1500 px yields distance≈1502 → the crop clamps to essentially the whole frame and expand_labels performs a … |
| M11 | perf | DB browser preview cache grows unbounded and never evicts stale-DB entries | `src/itasc/napari/nucleus_db_browser_widget.py:497` | Browsing a 2048×2048 monolayer across, say, 100 distinct (frame × union-size × merge-group) combinations retains ~1.6 GB of label arrays; regenerating the DB an… |
| M12 | perf | Large label TIFF reads and O(T·H·W) centroid extraction run synchronously on the Qt main thread | `src/itasc/napari/contact_analysis_widget.py:822` | For a large time-lapse (hundreds of frames at multi-megapixel resolution) the first Visualize freezes the entire napari UI for seconds to tens of seconds while … |
| M13 | perf | Every time-slider tick triggers a full, undebounced DB browser refresh | `src/itasc/napari/nucleus_workflow_widget.py:934` | Scrubbing the time slider across N frames enqueues N separate singleShot refreshes; every first-visit frame does a synchronous SQLite query + partition render +… |
| M14 | perf | Full atom extraction runs the heavy residual+watershed pipeline synchronously on the GUI thread | `src/itasc/napari/nucleus_atom_extraction_widget.py:602` | Clicking ▶ Run on the atom extraction section freezes the entire napari window (no repaint, no cancel, spinning cursor) for the whole multi-frame computation — … |
| M15 | perf | Cellpose preview loads the entire input TIFF on the GUI thread before starting the worker | `src/itasc/napari/cellpose_widget.py:673` | Every click of the ▷ preview button synchronously reads the whole raw nucleus/cell stack from disk on the GUI thread. For a large 3D+t input this freezes napari… |
| M16 | perf | Shape linker recomputes target coords/centroid once per candidate source | `src/itasc/tracking_ultrack/linking.py:255` | On a dense monolayer frame with thousands of cells and max_neighbors=10, each target's mask is argwhere-scanned ~20 times instead of once, multiplying the per-f… |
| M17 | perf | Anchor post-solve matching is O(anchors x tracks x T x pixels) via per-track full-stack rescans | `src/itasc/tracking_ultrack/corrections.py:942` | On the stated target (dense motile monolayers: hundreds of cells per frame, hundreds of frames), correcting even a few dozen anchors triggers on the order of an… |
| M18 | perf | build_atom_union_database loads the full atoms and contour stacks into RAM | `src/itasc/tracking_ultrack/db_build.py:291` | For a long time-lapse (e.g. 500 frames x 2048x2048), the int32 atom stack alone is ~8 GB and the float32 contour stack another ~8 GB, both held for the entire b… |
| M19 | arch | curation.apply_curation hard-codes experiment_id/position_id columns that the free-form classification schema does not guarantee | `src/itasc/contact_analysis/curation.py:151` | Joining curation onto a measurement table produced from a catalog whose levels are named, e.g., `condition`/`replicate` (no `experiment_id`) raises `KeyError: '… |

## Low-severity findings

| # | Dim | Title | Location |
|---|---|---|---|
| L1 | corr | DivergenceMapsReport never records the foreground filter parameters that were applied | `src/itasc/cellpose/divergence_maps.py:361` |
| L2 | corr | Cross-time tracking mixes z with y/x in one sqeuclidean metric with no anisotropy scaling | `src/itasc/cellpose/track_laptrack.py:212` |
| L3 | corr | MSD and DAC standard errors treat overlapping-origin samples as independent, understating uncertainty | `src/itasc/contact_analysis/dynamics/msd.py:70` |
| L4 | corr | run() `setdefault` lets a blank pixel_size/time_interval column in the catalog silently disable physical-unit quantifiers | `src/itasc/contact_analysis/pipeline.py:302` |
| L5 | corr | signed_central_junction_lengths attributes every frame a cell pair is ever in contact to the T1 event, including unrelated re-contacts | `src/itasc/contact_analysis/contacts/signed_contact_length.py:130` |
| L6 | corr | Greedy losing<->gaining T1 matching can mispair edges when multiple edges change between adjacent frames | `src/itasc/contact_analysis/contacts/build.py:573` |
| L7 | corr | T1 event location is taken from an arbitrary fragment of a split losing junction | `src/itasc/contact_analysis/contacts/build.py:485` |
| L8 | corr | Commit does not clear the dirty flag, causing a spurious unsaved-changes prompt | `src/itasc/napari/correction/nucleus_correction_widget.py:1267` |
| L9 | corr | Unbounded IN(...) list of FAKE node ids can exceed SQLite's variable limit | `src/itasc/tracking_ultrack/corrections.py:137` |
| L10 | corr | node_prob -1.0 sentinel leaks into preview probabilities and summary stats | `src/itasc/tracking_ultrack/db_query.py:863` |
| L11 | perf | Progressive shell assignment recomputes a full-image EDT on every shell iteration | `src/itasc/cellpose/flow_following.py:168` |
| L12 | perf | _per_frame_views does a full boolean scan of the whole table per frame (O(F^2) on long movies) | `src/itasc/contact_analysis/dynamics/collective.py:100` |
| L13 | perf | Co-sourced quantifiers re-read and re-regionprops the same label stack multiple times per position in one aggregate() | `src/itasc/contact_analysis/quantifiers/shape_relational.py:30` |
| L14 | perf | _order_coordinates re-runs a Python O(n^2) neighbor walk on segments that _coordinate_segments already ordered | `src/itasc/contact_analysis/contacts/build.py:604` |
| L15 | perf | Edge-coordinate accumulation materializes every boundary point as Python floats before conversion | `src/itasc/contact_analysis/contacts/build.py:519` |
| L16 | perf | fill_label_holes / fix_label_semiholes are O(n_background_components x frame_pixels) with per-component full-frame allocation | `src/itasc/correction/labels.py:842` |
| L17 | perf | _load_full_db_stack allocates a full-length label stack sized by max frame index | `src/itasc/napari/nucleus_db_browser_widget.py:528` |
| L18 | perf | Track navigation re-scans the whole label stack on every keystroke | `src/itasc/napari/correction/_correction_navigation.py:54` |
| L19 | perf | Streaming Channel-1 segmentation reassigns the whole (T,Z,Y,X) layer array on every yielded frame | `src/itasc/napari/cellpose_segment_track_widget.py:845` |
| L20 | perf | Cache key hashes the entire input stacks on every call, including cache hits | `src/itasc/segmentation/cell_label_icm.py:462` |
| L21 | perf | Per-frame median normalization of unaries is dead computation | `src/itasc/segmentation/cell_label_icm.py:214` |
| L22 | perf | Unbounded SQL IN-list over all track nodes can exceed SQLite's variable limit | `src/itasc/tracking_ultrack/track_quality.py:81` |
| L23 | perf | Corrected export writes the large label TIFF to disk twice | `src/itasc/tracking_ultrack/export.py:75` |
| L24 | arch | StageLogger discards the traceback on stage failure, logging only str(exc_val) | `src/itasc/core/logging.py:50` |
| L25 | arch | NucleusUltrackDbBrowserMixin is a god-object mixing DB math, rasterization and graph layout into UI | `src/itasc/napari/nucleus_db_browser_widget.py:235` |
| L26 | arch | Dead/duplicated visualization code paths retained alongside the active native-layer path | `src/itasc/napari/contact_visualization.py:629` |
| L27 | arch | Workflow widgets rebind ~40 sub-widget methods/attributes onto self via setattr loops, a fragile hidden coupling | `src/itasc/napari/nucleus_workflow_widget.py:454` |
| L28 | arch | cycle_index and nearest_area_index raise on an empty candidate list | `src/itasc/tracking_ultrack/swap_candidate.py:161` |
| L29 | arch | _reset_annotations creates a SQLAlchemy engine it never disposes | `src/itasc/tracking_ultrack/db_build.py:58` |

## Strengths

- Numeric cores are careful and defensively coded: consistent divide-by-zero, empty-frame, single-cell, gap-aware, and NaN-propagation guards across dynamics, contacts, segmentation, and tracking; unit handling (pixel_size, dt) and coordinate conventions (row=pos[-2], col=pos[-1], [dy,dx] flow order) are consistent between shape and dynamics modules.
- The hardest algorithmic machinery is correct and internally consistent: atom merge-tree/branch-union logic and the node-ID scheme match db_build, geodesic-Voronoi labeling with nucleus anchoring, union-find z-stitching, greedy retrack id-collision avoidance, hierarchy-cut/greedy-coloring in the DB layer, and divergence sign conventions for contours all check out under real caller usage.
- Strong edge-case coverage in the tracking DB and core: engine lifecycle, chunked inserts, empty frames, self-loops, NULL weights, zero-quality guards, NaN distances, and cache invalidation are handled deliberately rather than by accident.
- Disciplined Qt/napari threading where it counts: worker threads marshal progress back to the main thread via QObject signals, a shared UiGate cleanly serializes viewer ownership, and worker bodies mostly avoid touching napari off-thread — a mutual-exclusion model that is genuinely hard to get right.
- Clean architectural separation in the analysis pipeline: compose-only pipeline, self-registering quantifiers, materialized-view pooling, robust TIFF-tag parsing, and 2D/3D stack normalization give a well-factored, extensible batch system.
- Good defensive craftsmanship in the correction/labels module: dtype-ceiling guard in _free_label, half-open bboxes, connected-component reapplication in add_cell, correct border clipping, and even a frozen reference test locking in the clean_stranded_pixels crop optimization.
- Reusable pure helpers are Qt-free and unit-testable (track path, candidates, accordion, utils; CorrectionWidget), and the cell correction widget correctly sets its dirty flag on edit — demonstrating the right pattern exists in-repo and just needs to be applied uniformly.
- High documentation quality throughout: modules explain intent (streaming argmin, do_3d guards, provenance reports), which is what let reviewers refute a large share of candidate bugs and precisely localize the ones that survived.
- The review itself reflects a healthy codebase: the majority of serious findings are scalability and UI state-sync issues with clear fixes, not deep algorithmic errors — the science is sound and the remaining work is hardening.

## Per-subsystem health notes

**core** — Core infrastructure is small, well-documented, and largely correct: lineage segment collapsing, TIFF stack squeeze/round-trip handling, the residual strength==0 fast path, and path/legacy-config resolution all check out under how callers actually use them. The one substantive concern is that the commit-vs-working staleness contract is built entirely on filesystem mtime (via copy2 mtime propagation), which is fragile on coarse-resolution or cross-filesystem storage; a secondary low-severity gap is loss of tracebacks in the persistent stage log.

**cellpose** — Overall the subsystem is well-structured and mostly correct: axis/channel conventions ([dy,dx] flows, TZYX canonicalization, divergence sign for contours, HSV flow visualization, union-find z-stitching, greedy retrack id-collision avoidance) all check out. The defects found are concentrated in edge-case axis handling and metadata/validation rather than in the core numerics. The most serious is a size-collision that silently defeats a safety guard when a do_3d nucleus volume has exactly 2 z-slices; the rest are lower-severity footguns around ambiguous input layouts, provenance reporting, and anisotropy.

**segmentation** — The geodesic-Voronoi label logic, nucleus anchoring, cache content-addressing, and dtype-safe label commit are careful and mostly correct, with good edge handling for empty frames in most places. The two most serious problems are structural rather than numeric: the "streaming argmin" comment promises a memory saving that the per-(frame,label) dense dict does not actually deliver, so a dense monolayer timelapse can consume tens of GB, and an all-zero nucleus stack crashes label assembly. A per-frame median normalization step is provably dead work.

**tracking-db** — The subsystem is largely correct and carefully written: engine lifecycle, chunked inserts, greedy-coloring, and the hierarchy-cut logic all hold up, and edge cases like empty frames, self-loops, and NULL weights are generally handled. The main real defect is an inconsistent treatment of the -1.0 "unset" node_prob sentinel that the module itself documents but two read paths ignore, producing wrong probabilities in the browser. Remaining items are memory/IO efficiency and a small resource leak.

**tracking-corrections** — The subsystem is intricate but mostly sound; node-id and OverlapDB conventions are internally consistent with the DB builders, and the anchor-annotation and extend/swap paths verify cleanly. The one serious defect is in reseed's validated-mask merge, which fragments a validated cell's track whenever the solver's overlapping track id numerically equals the validated cell id (frequent for low ids, since ultrack numbers tracks from 1). A few robustness/performance issues round it out.

**tracking-core** — The subsystem is generally careful and well-documented, with correct handling of most edge cases (empty frames, single cell, NaN distances, zero-quality guards, cache invalidation). The atom merge-tree/branch-union machinery and the node-ID scheme are correct and consistent with db_build. I found one silent-correctness bug in validation-node injection, one hot-loop redundant recomputation in shape linking, and a scalability crash risk from unbounded SQL IN-lists on large time-lapses.

**contacts** — The subsystem is generally well-structured and the coordinate/half-pixel geometry, HDF5 column round-tripping, and empty/single-cell edge cases are handled correctly. The real risk concentrates in the T1-event detection and signed-length reaction coordinate — the scientific core — where greedy pairing, edge flicker, and cell-pair (rather than edge-id) attribution can bias the derived potential landscape. Two hot-loop performance issues would hurt on large dense-monolayer time-lapses.

**contact-dynamics** — The numeric core is careful and well-guarded against divide-by-zero, empty frames, single-cell frames, gaps, and NaN propagation; unit handling (pixel_size, dt, gap-aware velocities) and coordinate conventions are consistent between shape and dynamics. The main weaknesses are in the collective-motion module: an O(N^2) pure-Python pair loop and duplicated pairwise work that will dominate runtime/memory on the dense monolayers this code explicitly targets, plus a leaky reuse of gap-spanning velocities that biases the alignment/correlation metrics. Statistical error bars (MSD/DAC SEM) understate uncertainty because overlapping-origin samples are treated as independent.

**contact-pipeline** — The subsystem is well-structured with clean separation (compose-only pipeline, self-registering quantifiers, materialized-view pooling) and careful edge-case handling in the numeric cores (NaN guards, degenerate-region handling, 2D/3D stack normalization, robust TIFF-tag parsing). The main defect is a real gap between the build-param gate (which consults `params`) and the value delivery path (which reads only the record), so the documented headless `aggregate(..., params={...})` contract crashes for pixel size / frame interval. Secondary issues are footguns around blank catalog columns, curation's hard-coded key coupling, and repeated per-position I/O/regionprops across co-sourced quantifiers.

**correction-labels** — Overall a carefully written, well-commented module with good defensive checks (dtype-ceiling guard in _free_label, half-open bboxes, connected-component reapplication in add_cell, a frozen reference test for the clean_stranded_pixels crop optimization). Coordinate/axis conventions (row=pos[-2], col=pos[-1]) are consistent throughout and border clipping is handled correctly. The main risks are a latent data-loss interaction between the intentional disconnected-component merge and the global fragment cleaner, plus two performance patterns that scale poorly on large dense frames.

**napari-correction** — The subsystem is reasonably well-factored: the reusable CorrectionWidget is clean, pure helpers (track path, candidates, accordion, utils) are Qt-free and unit-testable, and the cell widget tracks its dirty state correctly. The main risk area is the nucleus widget's "unsaved changes" detection, which is inferred from status-message substrings rather than set on edit, so several real mutation paths never mark the session dirty and can silently discard work on deactivate. There are also a background-thread aliasing race and a missing-undo-history gap around the extend action.

**napari-workflow** — The orchestration (main_widget) and its worker/gate wiring are largely sound: progress emitters bridge worker threads to the UI via signals, the shared UiGate cleanly serializes viewer ownership, and worker bodies mostly avoid touching napari from off-thread. The main defects are a deceptive/ineffective cancel path in the cell run that can leave output written and spawn concurrent writers to the same file, and two heavy operations left on the GUI thread (atom full-run, cellpose preview load) that the codebase elsewhere takes care to offload. Architecture leans on large setattr aliasing blocks that are functional but fragile.

**napari-analysis** — The subsystem is largely well-structured for Qt/napari code: worker threads correctly marshal progress back to the main thread via QObject signals, and the mutual-exclusion viewer-owner gating is disciplined. The main risks are one high-impact state-sync bug where "Recompute" silently displays stale overlays, several unbounded/large in-memory caches and stacks that can blow up on long time-lapses, and heavy synchronous disk I/O + pixel iteration on the Qt main thread. The DB browser mixin is also an oversized god-object mixing DB math, rasterization and graph layout into the UI class.

## Project-level observations (outside the per-subsystem sweep)

- **Packaging is sophisticated but fragile-by-design.** The four distributable
  wheels (`itasc-core`, `-cellpose`, `-tracking`, `-aggregate`) are carved out of
  the single `src/itasc` tree via hand-maintained `force-include` file lists plus
  PEP 420 namespace packages. Any drift between those lists and the real import
  graph breaks a standalone wheel at import time. This is **well-mitigated** by a CI
  step that installs each wheel in isolation and imports/exercises it — a genuine
  strength — but the mitigation only covers the specific symbols that step touches.
- **CI is solid:** multi-OS matrix (Linux full Python sweep; Windows/macOS on 3.12),
  offscreen Qt, `ruff` lint, GUI tests gated to Linux with a documented rationale,
  wheel builds, and the isolated-install verification above.
- **Docs build is strict** (`sphinx-build -W`), turning doc drift into a failing check.
