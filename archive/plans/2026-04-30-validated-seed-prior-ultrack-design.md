# Validated Seed Prior Ultrack Design

Date: 2026-04-30

## Context

The anchor IoU experiment showed that pure IoU linking improves local split and
merge counts near a validated frame, but it hurts foreground coverage and track
continuity. The signal-quality experiments showed that a nucleus intensity
drop metric can help whole-object candidates beat fragments, but only when the
reward is shaped strongly enough. The best current direction is a resolve-only
additive seed prior:

```text
node_prob = drop_frac ^ quality_exponent + seed_weight * best_seed_affinity
```

The current `resolve_with_validation()` path removes hypotheses that overlap
validated masks, solves the remaining database, then pastes validated labels
back into the exported labelmap. The new feature should instead inject
validated masks into the Ultrack database and mark them as true annotations, so
the solver includes them directly.

Ultrack already has a `power` parameter. CellFlow passes it through to
Ultrack, but the napari UI does not surface it yet. Ultrack applies the
configured link function to both link weights and `NodeDB.node_prob`, so the
feature should expose both CellFlow's quality exponent and Ultrack's power with
clear tooltips instead of hiding one behind compensation logic.

## Goals

- Apply the new behavior only when `Resolve from validated` is checked.
- Surface the Ultrack `power` parameter.
- Surface resolve-only controls for segmentation-quality exponent and seed
  proximity scoring.
- Use `validated_cells.json` as the source of validated cell-frame selections.
- Inject validated masks into `NodeDB` as selected-by-annotation candidates.
- Mark overlapping non-validated candidates as false instead of deleting all
  competitors before solving.
- Run the solver with annotations so validated cells are selected by Ultrack.
- Avoid post-export paste-back in the normal success path.

## Non-Goals

- Replacing the normal Ultrack route.
- Applying seed priors when `Resolve from validated` is unchecked.
- Changing the experiment's IoU linker behavior.
- Implementing inverse-transform compensation between `quality_exponent` and
  Ultrack `power`.
- Surfacing `seed_sigma_area` in the first UI version.

## Architecture

The resolve route should replace the current prune-and-paste pipeline with a
solver-owned validation pipeline:

1. Rebuild `NodeDB` and `OverlapDB` from `hypotheses.h5`.
2. Read validated tracks with `read_validated_tracks(pos_dir)`.
3. Use the current tracked labelmap as the source of validated masks.
4. Insert one synthetic `NodeDB` row per validated cell-frame mask.
5. Mark each synthetic validated row as `VarAnnotation.REAL`.
6. Mark intersecting non-validated candidate rows at the same frame as
   `VarAnnotation.FAKE`.
7. Compute resolve-only `node_prob` values for unresolved candidate nodes.
8. Run linking.
9. Run `solve(..., use_annotations=True)`.
10. Export the solver output directly.

Validated masks become first-class Ultrack nodes. They participate in overlap
constraints, selected-node state, linking, parentage, and export.

## Scoring

CellFlow should write this raw score directly to `NodeDB.node_prob`:

```text
node_prob =
    drop_frac ^ quality_exponent
  + seed_weight * best_seed_affinity
```

`drop_frac` is the fraction of one-pixel outer-ring pixels below the node's
inside median intensity.

Seed affinity uses the best validated seed, not a sum over seeds:

```text
best_seed_affinity =
    max over validated seed nodes [
        size_similarity * spatial_decay * temporal_decay
    ]

size_similarity = exp(-abs(log(area_node / area_seed)) / seed_sigma_area)
spatial_decay   = exp(-(centroid_distance / seed_sigma_space)^2)
temporal_decay  = exp(-abs(dt) / seed_tau_time)
```

Only candidate nodes with `abs(dt) <= seed_max_dt` need to be evaluated against
validated seeds. The affinity term is additive, because the additive experiment
preserved coverage and improved the local validated window better than the
multiplicative version.

Injected validated nodes should receive `node_prob=1.0` for debugging
consistency, but their selection must come from `VarAnnotation.REAL`, not from
score magnitude.

Ultrack's `link_function` remains direct. With the default
`link_function="power"`, Ultrack later raises stored node and link weights to
the configured `power`. The UI tooltip must state that the CellFlow quality
exponent shapes the stored node probability, while Ultrack power transforms the
stored weights during solving.

## Validation Injection

Validated injection should use the existing validated-track contract:

```text
dict[int, set[int]]  # {cell_id: validated_frames}
```

For every validated cell-frame:

- If the cell mask is present in the tracked labelmap, build a synthetic
  Ultrack `Node` from that mask.
- Assign IDs after hypothesis ingest by taking the next available `t_node_id`
  in each frame and computing the full ID with the same
  `(t + 1) * max_segments_per_time + t_node_id` convention used by hypothesis
  nodes.
- Abort with a clear configuration error if a frame would exceed
  `max_segments_per_time`, rather than risking an ID collision.
- Store standard node fields: `t`, `t_node_id`, `t_hier_id`, centroid, area,
  and pickled `Node`.
- Use a reserved hierarchy marker such as `t_hier_id=0` to distinguish
  injected validation nodes.
- Set `node_annot = VarAnnotation.REAL`.
- Add or preserve overlap constraints between the injected node and every
  same-frame candidate whose pixels intersect it.
- Set intersecting non-validated candidates to `VarAnnotation.FAKE`.

If a validated cell-frame is listed in JSON but absent from the current tracked
labelmap, skip that cell-frame and include the skipped count in the progress or
error message.

## UI And Config

The new controls belong in the Ultrack Tracking section and are enabled only
when `Resolve from validated` is checked:

- `Ultrack Power`: default `4.0`, maps to existing `TrackingConfig.power`.
- `Quality Exp`: default `8.0`, new `TrackingConfig.quality_exponent`.
- `Seed Weight`: default `0.5`, new `TrackingConfig.seed_weight`.
- `Seed Space (px)`: default `25.0`, new
  `TrackingConfig.seed_sigma_space`.
- `Seed Time`: default `2.0`, new `TrackingConfig.seed_tau_time`.
- `Seed Window`: default `5`, new `TrackingConfig.seed_max_dt`.

Keep `TrackingConfig.seed_sigma_area` hidden/config-only with default `0.5`.

Suggested tooltips:

- `Quality Exp`: "Raises the signal-based segmentation quality before storing
  it as node_prob. Higher values favor high-confidence whole-object candidates
  over fragments."
- `Ultrack Power`: "Ultrack's solver transform for node_prob and link weights.
  With link_function=power, stored weights are raised to this power during
  solving."
- `Seed Weight`: "Additive reward for candidates similar to nearby validated
  cells. Zero disables the seed-local bonus."
- `Seed Space (px)`: "Spatial decay scale for seed proximity. Larger values
  let validated cells influence candidates farther away."
- `Seed Time`: "Temporal decay scale in frames. Larger values let validated
  cells influence more distant frames within the seed window."
- `Seed Window`: "Maximum frame distance from a validated cell used for seed
  affinity."

All surfaced values should persist through widget state save/load. The terminal
resolve command must include the same values so GUI and terminal behavior stay
equivalent.

`Ultrack Power` affects Ultrack solving generally, but the control can live in
the resolve parameter area for this feature because it matters most when
interpreting node-prior experiments.

## Error Handling

- If no validated tracks exist, keep the current UI message and do not run
  resolve.
- If validated injection produces no synthetic nodes, abort before solve with a
  clear message.
- If some validated cell-frames are absent from the tracked labelmap, skip them
  and report how many were skipped.
- If annotations make the ILP infeasible, fail visibly and do not overwrite
  `tracked_labels.tif` or `validated_cells.json`.
- Do not paste validated masks into the export on the normal success path.
- Do not use paste-back fallback for linking, solving, or export failures.
  Fail without overwriting `tracked_labels.tif` or `validated_cells.json`.

## Testing

Add focused tests for:

- `TrackingConfig` exposes the new resolve-only fields and defaults.
- The UI exposes the new controls, disables them when `Resolve from validated`
  is unchecked, enables them when checked, and persists them through
  save/load.
- Validated-node injection inserts synthetic `NodeDB` rows with `REAL`
  annotation and reserved IDs.
- Overlapping non-validated nodes are marked `FAKE`.
- Missing validated cell-frames are skipped and reported.
- Node probability scoring writes
  `drop_frac ^ quality_exponent + seed_weight * best_seed_affinity`.
- `resolve_with_validation()` calls `solve(..., use_annotations=True)`.
- The success path exports solver-selected validated nodes without relying on
  `merge_validated_into_export()`.
- The terminal resolve command includes the new parameters.

## Open Implementation Notes

- The signal source for `drop_frac` should be the same nucleus fluorescence
  image used in the experiment. If that image path is not reliably available in
  the resolve call, the implementation plan should first add explicit plumbing
  for it rather than guessing.
- The injected-node ID scheme must be documented in code and covered by tests,
  because Ultrack IDs are time-scoped through `max_segments_per_time`.
- The first implementation should keep `seed_sigma_area=0.5` hidden. It can be
  surfaced later if datasets show frequent size-scale sensitivity.
