# Ultrack DB/Solver Separation Design

## Goal

Separate Ultrack database construction from solver execution so each UI section only exposes parameters that affect that stage. Move validated-correction influence into database building, where validated nodes, fake conflicts, node probabilities, and boosted links can be inspected before solving. Extend the Ultrack database browser so validated and fake nodes are visible.

## Workflow Boundary

DB Generation owns every parameter and operation that changes `data.db`:

- segmentation hierarchy and foreground thresholds
- candidate node creation
- node probability scoring
- temporal linking
- optional validated-node injection
- optional fake-node conflict marking
- optional seed-affinity node scoring
- optional validated-edge boosting

Solver owns only parameters that select from an existing `data.db`:

- `appear_weight`
- `disappear_weight`
- `division_weight`
- `link_function`
- `power`
- `bias`
- `solution_gap`
- `time_limit`
- `window_size`

The user workflow becomes:

1. Correct and validate cells.
2. In DB Generation, optionally enable validated corrections.
3. Build `data.db`.
4. Inspect candidates, links, REAL nodes, and FAKE conflicts in the DB browser.
5. Solve the existing database.

The current resolve-from-validated action should no longer be a separate solver route. Its validated graph-building behavior belongs in DB generation.

## DB Generation UI

Add a checkbox to the DB Generation section:

```text
Use validated corrections
```

When unchecked, DB generation follows the plain path:

```text
foreground + contours
-> ultrack.segment writes NodeDB/OverlapDB
-> score node_prob from nucleus intensity
-> link writes LinkDB
-> data.db ready
```

When checked, DB generation requires validated tracks and current tracked labels, then follows the validated-aware path:

```text
foreground + contours
-> ultrack.segment writes canonical NodeDB/OverlapDB
-> inject validated masks as NodeDB(node_annot=REAL, node_prob=1.0)
-> mark overlapping candidates as node_annot=FAKE and add OverlapDB conflicts
-> score node_prob using intensity + validated seed affinity
-> link writes LinkDB
-> boost LinkDB weights incident to REAL nodes
-> data.db ready
```

Validated-specific controls are disabled unless the checkbox is active:

- `Seed weight`
- `Seed space sigma`
- `Seed time tau`
- `Seed window`
- `Seed area sigma`
- `Quality exponent`, because it affects validated seed-prior scoring

Progress/status should report validated-specific work explicitly:

- injecting validated nodes
- skipped validated cell-frames
- inserted REAL node count
- FAKE node count
- scoring node probabilities
- linking candidates
- boosted edge count

## Solver UI

The solver should operate on the current `data.db`; it should not rebuild or mutate the database except for normal solve output fields such as selected nodes and parent IDs.

Move `power`, `bias`, and `link_function` to Solver because ultrack applies them during solve-time transformation of node and edge weights. Keep DB-only linking and segmentation controls out of the Solver section.

The solver should auto-detect annotations. If `NodeDB.node_annot` contains REAL or FAKE annotations, call `run_solve(..., use_annotations=True)`; otherwise call it with `use_annotations=False`. This avoids requiring a second user checkbox.

## Export And ID Preservation

The old resolve path used `merge_validated_into_export` after solving to preserve validated cell IDs. After validated-aware DB generation replaces resolve-from-validated, normal solve/export still needs equivalent ID preservation when the database contains validated annotations.

Design rule:

- Plain DB solve exports normally.
- Validated-aware DB solve exports, then applies validated-ID preservation using the current validated tracks and tracked labels.

If validated masks are unavailable at export time, the UI should fail clearly rather than silently losing validated IDs.

## DB Browser Validated Visualization

Extend the existing DB browser rather than adding a separate mode.

For rendered nodes, query and retain `NodeDB.node_annot` metadata:

- normal candidate
- REAL validated node
- FAKE conflicting candidate

Browser behavior:

- Summary includes annotation counts, for example `12038 nodes | 44219 links | REAL 8 | FAKE 31`.
- Selecting a node reports annotation state, for example `Selected node 1023004 [REAL] at t=12`.
- Connected focus works for REAL nodes and normal nodes alike.
- Edge-weight and node-prob transparency continue to work with connected focus.

Visual treatment:

- Keep the existing preview labels layer as the main rendering.
- Add a lightweight annotation overlay layer for outlines or translucent masks.
- REAL nodes render with a strong distinctive outline.
- FAKE nodes are hidden by default. When shown, they render with a muted conflict style.

Add browser controls:

- `Show validated nodes`, default on
- `Show fake nodes`, default off

When `Show fake nodes` is off, FAKE candidates are excluded from the preview and connected-focus display unless the selected node itself is FAKE. If the selected node is hidden, status should explain that it is hidden by annotation filter.

## Implementation Shape

Add a shared DB-build helper so widget workers and terminal scripts do not duplicate pipeline logic in generated Python strings. Suggested shape:

```python
build_ultrack_database(
    contour_maps_path,
    foreground_masks_path,
    nucleus_prob_zavg_path,
    working_dir,
    cfg,
    validated_tracks=None,
    tracked_labels=None,
    use_validated=False,
    progress_cb=None,
)
```

Responsibilities:

- load and normalize inputs
- run canonical ultrack segmentation
- optionally inject validated nodes
- score node probabilities
- link candidates
- optionally boost validated edges
- return a report with counts for UI status

Keep the existing `TrackingConfig` as the backing model initially, but split UI config helpers into DB-build and solve groups so controls match their stage. A later cleanup can introduce separate Pydantic models if the shared model becomes confusing.

Move reusable parts of `resolve_with_canonical_segment` into this DB-build helper. Keep `merge_validated_into_export` as export-time behavior for validated-aware databases.

## Testing

Add focused tests for:

- DB generation checkbox enables and disables validated-specific controls.
- DB generation config excludes solve-only parameters from the DB terminal script.
- Solver config excludes DB-only parameters from the solve terminal script.
- Validated-aware DB build calls segment, inject, score, link, and boost in order.
- Plain DB build does not call inject or boost.
- Solve auto-enables annotations when the DB contains REAL or FAKE annotations.
- Plain solve does not enable annotations when no annotations are present.
- Validated-aware solve/export preserves validated IDs.
- DB browser summary includes REAL and FAKE counts.
- DB browser selected-node status includes annotation state.
- DB browser annotation filters hide/show REAL and FAKE nodes as specified.
- Connected focus works when the selected node is REAL.

Existing plain DB generation, solve, and DB browser tests should remain passing.

## Non-Goals

- Do not introduce a separate validated DB browser mode.
- Do not hand-edit DB rows from the browser.
- Do not add hard no-appearance constraints in this change.
- Do not replace ultrack's solver or linking implementation.
