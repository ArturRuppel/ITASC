---
status: complete
date: 2026-04-28
completed: 2026-04-28
predecessor: PLAN_direct_ultrack_ingestion.md (Phases 0–3 complete)
---

# Plan: Cleanup + Validate-and-Resolve Loop

The Ultrack-ILP nucleus tracker is working with decent quality. This plan
covers the next major feature: a *validate-and-resolve* loop where the user
locks in confirmed tracks, then re-solves the rest of the stack — possibly
with tweaked parameters — without losing the validation work.

The core insight: after significant manual correction, the user has built
up real value (the validated tracks). Different parameter sets make
different mistakes, so being able to lock in the good parts and re-solve
to fix the bad parts is more useful than picking one global parameter set.

## High-level design

1. **Validation is per-cell, whole-track.** A "validated cell" means the
   user has confirmed that cell's masks across its full lifecycle. There is
   no partial validation across a track's frames. (May change later;
   storage shape is flexible enough to extend.)
2. **Re-solve = prune + solve + merge.** Remove every hypothesis node that
   intersects any validated mask, run the normal Ultrack pipeline, then
   paste validated cells back in with fresh unique track IDs.
3. **No constraint solver, no node pinning.** Validated cells aren't in
   the solver's graph at all — we sidestep `enforce_nodes_solution_value`,
   batch boundaries, and the spurious-division problem.
4. **Track continuity through validated cells is not preserved.** Every
   validated cell becomes its own track in the merged output. Acceptable
   under the current model (validation = "this segmentation is right",
   not "this lineage is right"). Division lineage is irrelevant in this
   project regardless.

The cell workflow (3_cell) stays on its current frame-level validation +
greedy `propagator.py`. Out of scope.

---

## Step 1 — Cleanup (~half day)

The nucleus widget has dead handlers that reference removed spinboxes
(would crash if called). The anchor-LAP propagator from the v2 experiments
is fully orphaned.

**Delete:**
- `src/cellflow/tracking/propagator_v2.py`
- `tests/tracking/test_propagator_v2.py`
- `scripts/test_anchor_lap_propagator.py`
- `scripts/debug_propagator.py`
- In `src/cellflow/napari/nucleus_workflow_widget.py`:
  - `_on_propagate_next` / `_on_propagate_all`
  - `_on_propagate_next_v2` / `_on_propagate_all_v2`
  - `_on_prop_progress` / `_on_prop_done` if no other caller remains
  - Imports of `propagate_one_frame` and `propagate_one_frame_v2`

**Keep:**
- `src/cellflow/tracking/propagator.py` — used by `cell_workflow_widget.py`
- `src/cellflow/tracking/retracker.py` — used by nucleus widget's
  `_on_retrack_frame` button
- `tests/tracking/test_propagator.py`, `test_retracker.py`

**Acceptance:** `pytest` passes; nucleus widget loads end-to-end.

---

## Step 2 — Validation storage refactor (~half day)

### 2.1 Storage model

`src/cellflow/database/validation.py` currently has two APIs:

- **Frame-level** (`validate_frame`, `read_validated_frames`,
  `validated_frames.json`) — used by `cell_workflow_widget.py`.
  **Stays untouched.** Cell workflow keeps its current behavior.
- **Cell-level** (`validate_cells`, `read_validated_cells` keyed by frame,
  `validated_cells.json` keyed by frame) — used by
  `nucleus_workflow_widget.py`. **Gets reshaped.**

New `validated_cells.json` schema (keyed by **cell ID**, value is list of
frames):

```json
{"47": [10, 11, 12, 13, 14], "82": [3, 4, 5]}
```

Rationale: matches the per-track mental model, makes future track ops
(e.g. merging two tracks → merge two entries) trivial, and works equally
well for the access patterns we need (per-frame lookup is one pass over
the dict).

### 2.2 New nucleus-side API

In `validation.py`, alongside the existing frame-level functions:

- `read_validated_tracks(pos_dir) -> dict[int, set[int]]`
  Returns `{cell_id: {frames}}`. Empty dict if file missing.
- `read_validated_cells_at_frame(pos_dir, t) -> set[int]`
  Derived: returns all cell IDs validated at frame `t`. For overlay rendering.
- `is_track_validated(pos_dir, cell_id) -> bool`
- `validate_track(pos_dir, cell_id, frames: Iterable[int])`
  Mark a whole track validated. Adds the given frames to that cell ID's
  entry (idempotent, accumulates).
- `invalidate_track(pos_dir, cell_id)`
  Remove the entire entry for `cell_id`.

The old per-frame cell API (`validate_cells(pos_dir, t, ids)` etc.) is
removed; nucleus widget call sites get migrated.

`validated_frames.json` and the frame-level "fully validated" cache are
**not** used by the nucleus workflow at all anymore. The frame counter
("4/30 frames complete"), if we still want it, is derived from the new
`validated_cells.json` + the labelmap on the fly.

### 2.3 Tests

Update `tests/database/test_validation.py`:
- Frame-level tests stay (they cover the cell-workflow path).
- Cell-level tests get rewritten against the new track-keyed API.

**Acceptance:** `pytest tests/database/test_validation.py` passes.

---

## Step 3 — Validation viz + UX (~1.5 days)

### 3.1 Overlay layer

In `nucleus_workflow_widget.py`, render a green-tinted overlay for
validated cells of the current frame:

- Separate napari Labels layer (`_VALIDATED_OVERLAY`) with a binary mask =
  `np.isin(tracked[t], list(validated_at_t))`, single green colormap, ~50%
  opacity.
- Updates triggered by:
  - `viewer.dims.events.current_step` (frame change → recompute mask).
  - `tracked_layer.events.data` / `paint` (post-edit → invalidate
    affected tracks, recompute mask).

### 3.2 Validate / invalidate UX

- `V` (with a cell selected): toggle the *whole track* for that cell.
  - Selected cell ID = `c`. Find every frame where `tracked == c`. Either
    `validate_track(pos_dir, c, frames)` or `invalidate_track(pos_dir, c)`
    depending on current state.
- The existing per-frame `Validate Frame` button is dropped (no longer
  meaningful under the per-track model). The keyboard shortcut
  `Ctrl+Shift+V` is also dropped, or repurposed later.

### 3.3 Auto-invalidate on edit

Hook into `correction_widget.py`. After each edit op, the
`_record_history` diff already gives `before/after`; from that we get
`changed_ids = set(before[changed]) | set(after[changed]) - {0}`. For
each changed ID, call `invalidate_track(pos_dir, cell_id)` — the whole
track is invalidated, not just the edited frame.

Requires `correction_widget` to know `pos_dir` (currently doesn't); pass
it in from the nucleus widget at construction.

### 3.4 Status counter

Widget label like `"7 tracks validated, 234 cell-frames covered"`.
Computed by reading the JSON; cheap.

**Acceptance:**
- Validated cells render green, update on frame change and on edit.
- `V` toggles whole-track validation.
- Editing any frame of a validated track removes the green overlay
  everywhere for that track.

---

## Step 4 — Re-solve backend (~1 day)

New module `src/cellflow/tracking_ultrack/reseed.py`:

### 4.1 `prune_validated_overlaps(working_dir, validated_tracks, tracked_labels)`

- `validated_tracks: dict[int, set[int]]` — `{cell_id: {frames}}`.
- `tracked_labels: np.ndarray` — current corrected labelmap, `(T, Y, X)`
  or `(T, Z, Y, X)`.
- For each `(cell_id, t)` pair, extract the validated mask
  `tracked_labels[t] == cell_id` (with bbox).
- For each frame `t` with at least one validated cell, query NodeDB rows
  at that frame, unpickle each `Node`, check bbox-then-pixel intersection
  with any validated mask at `t`. Aggressive criterion: **any pixel
  intersection** counts as conflict.
- Delete matching `NodeDB` rows and all referencing `OverlapDB` rows in
  one transaction.

### 4.2 `merge_validated_into_export(exported_labels, validated_tracks, tracked_labels)`

- After Ultrack export, paste validated masks back onto the exported
  labelmap.
- Generate fresh unique track IDs per validated `cell_id`, starting from
  `exported_labels.max() + 1`. All frames of a given validated cell get
  the same new ID (so the track stays one continuous track in the output).
- Validated masks **overwrite** any pixels in the exported labelmap they
  intersect (the pruning was aggressive but Ultrack may still place
  hypothesis nodes near them; validated cells win).

### 4.3 `resolve_with_validation(working_dir, validated_tracks, tracked_labels, cfg)`

Top-level orchestration:

1. Prune (4.1).
2. Run normal Ultrack `link → solve → export` against the pruned DB.
3. Merge (4.2).
4. Return the merged labelmap.

### 4.4 Tests

`tests/tracking_ultrack/test_reseed.py`:
- Unit test for prune: synthetic NodeDB + a known validated mask, assert
  correct nodes are deleted and OverlapDB rows are cleaned up.
- Unit test for merge: synthetic exported labelmap + validated tracks,
  assert validated pixels are overwritten and IDs are unique.
- Integration test (small synthetic dataset): full
  `resolve_with_validation` round-trip, assert validated cells appear
  unchanged in the output and unvalidated frames have plausible tracks.

**Acceptance:** unit + integration tests pass.

---

## Step 5 — Script-driven end-to-end test (~half day)

Before any UI integration, prove the full flow works on real data.

`scripts/test_validate_and_resolve.py`:

1. Take the curated 10-frame test dataset
   (`2026-04-01_U251.../v2/pos00/2_nucleus/`).
2. Programmatically pick ~5 cells from the GT labelmap, register them as
   validated tracks in `validated_cells.json` (write directly via the API).
3. Run `resolve_with_validation` from the existing hypothesis HDF5.
4. Eyeball the output: validated cells preserved verbatim, surrounding
   tracks reasonable, no crashes.
5. Tweak a tracking parameter (e.g. `division_weight`) and re-run.
   Confirm validated cells are still identical, surrounding tracks change.

**Acceptance:** script runs end-to-end, validated cells round-trip
unchanged, parameter changes affect only the unvalidated regions.

---

## Step 6 — UI integration (~1 day)

Nucleus widget gets a "Re-solve from validated" button:

- Loads current `validated_cells.json` and the current `tracked_labels`.
- Calls `resolve_with_validation` with current `cfg`.
- Replaces the tracked layer's data with the result.
- Status: how many tracks were preserved, how many new tracks were
  generated.

The widget should also surface the relevant tracking parameters as
editable fields so the user can tweak between re-solves. (Some are
already exposed; audit and fill gaps.)

**Acceptance:**
- Validate some tracks → click re-solve → validated cells unchanged in
  output, other tracks updated.
- Tweak parameter → re-solve → validated cells still unchanged.

---

## Out of scope

- Cell workflow (3_cell) validation, Ultrack migration, testing.
- Replacing greedy `propagator.py` for cell pipeline.
- Multi-position batch re-solve.
- Track continuity *through* validated cells (validated cells become
  single tracks; surrounding tracks terminate / restart at validated
  frames).
- Division lineage preservation (irrelevant in this project).
- Partial-track validation (validate frames N..M of a cell but not others).

---

## Estimated total

- Step 1 (cleanup):                    ~0.5 day
- Step 2 (storage refactor):           ~0.5 day
- Step 3 (validation viz + UX):        ~1.5 days
- Step 4 (re-solve backend):           ~1 day
- Step 5 (script test):                ~0.5 day
- Step 6 (UI integration):             ~1 day

**Total: ~5 days.**
