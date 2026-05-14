# Anchors and Validated Tracks — Design

Date: 2026-05-14
Status: proposed

## Goal

Replace the current validate-and-resolve machinery with two per-frame correction primitives that are simpler to reason about and free of the candidate-abduction bug in `inject_validated_nodes`.

## Two primitives, per frame

- **Validated frame**: the user asserts mask and centroid for cell `c` at frame `t`. Solver is excluded from the region — any hierarchy candidate within radius `R` of the validated centroid is marked `FAKE`. After solve, the validated mask is pasted onto the output at frame `t` under cell id `c`. Geometry and identity are locked.
- **Anchor frame**: the user asserts position for cell `c` at frame `t`. The solver still picks geometry from its candidates. Anchor is used as a soft bias — the nearest hierarchy node within `R` gets its `node_prob` boosted; for consecutive anchored frames of the same cell, the LinkDB edge between their nearest nodes is boosted. If no candidate is within `R`, no DB modification happens. After solve, cell id `c` is guaranteed to appear at the anchor position in the output: either by remapping the solver track that passes through, or by stamping a small synthetic disk if no track is there.

A "fully validated track" is a cell where every frame is validated. A "partially validated track" is a cell with one or more anchor frames mixed with solver frames.

## Data model

Single flat record list, persisted alongside the tracked-labels array (which continues to hold validated mask geometry):

```
Correction = (cell_id: int, t: int, kind: {"validated", "anchor"}, y: float, x: float)
```

- For `validated`: geometry comes from the existing labels array at frame `t`, where label == `cell_id`.
- For `anchor`: position is the only payload.

The current `validated_cells.json` (`{cell_id: [frames]}`) is replaced by this list. Anchor list is new.

## Solve-time integration

`build_ultrack_database` gains two passes after segmentation, before linking:

1. **FAKE-mark for validated frames.** For each `validated` correction, query NodeDB at frame `t` for nodes whose centroid is within `cfg.anchor_radius_px` of `(y, x)`. Set `node_annot = 'FAKE'` on each. No new nodes inserted, no OverlapDB rows touched.
2. **Boost for anchors.** For each `anchor` correction, find the nearest NodeDB node at frame `t` within `cfg.anchor_radius_px`, ignoring nodes already marked `FAKE` by pass 1. If found, multiply its `node_prob` by `cfg.anchor_node_boost`. For every consecutive pair of anchors `(c, t)` and `(c, t+1)` where both resolved to a node, multiply the LinkDB weight between them by `cfg.anchor_link_boost` (insert with that weight if no link exists yet).

The FAKE-pass-first ordering is deliberate: if a user places an anchor near someone else's validated frame, the boost silently no-ops rather than fighting the FAKE mask.

That's the full solve-time contract. No `REAL` annotations, no overlap surgery, no node insertion.

## Post-solve integration

After `export_tracked_labels` produces the solver track array:

1. **Anchor remap.** For each `anchor` correction `(c, t, y, x)`: find solver tracks whose centroid at `t` is within `cfg.anchor_radius_px` of `(y, x)`. If at least one match exists, pick the closest; remap that solver track's id to `c` across its full lifetime, breaking ties in favor of the longer track. If two anchors of different `cell_id` map to the same solver track, the one with the closer centroid wins; the loser falls through to the synthetic-stamp branch. If `c` already exists as a solver-track id (collision with an unrelated track), the unrelated track is renumbered to a fresh unused id before the remap.
2. **Anchor stamp.** For each anchor not satisfied by remap: paint a small disk of radius `cfg.anchor_stamp_radius_px` at `(y, x)` on frame `t` with label `cell_id`. Overwrites whatever was at that location.
3. **Validated paste.** For each `validated` correction: paint the validated mask (from the labels array) at frame `t` with label `cell_id`. Overwrites the solver output. Runs last so validated always wins.

Order is anchor remap → anchor stamp → validated paste.

## Configuration

New `TrackingConfig` fields (all optional, sensible defaults):

- `anchor_radius_px: float` — single radius for both FAKE-marking and anchor candidate search. Default sized for typical nucleus diameter.
- `anchor_node_boost: float` — multiplicative boost on node_prob. Default `2.0`.
- `anchor_link_boost: float` — multiplicative boost on LinkDB weight. Default `2.0`.
- `anchor_stamp_radius_px: float` — radius for synthetic stamp when an anchor has no solver coverage. Default same as `anchor_radius_px`.

## Code to delete

- `inject_validated_nodes`, `_best_iou_assignments`, `_raw_iou` in `validation_nodes.py` (file becomes thin or removed).
- `prune_validated_overlaps` and the mask-injection branch of `merge_validated_into_export` in `reseed.py`. `merge_validated_into_export` becomes simple paste-back driven by the validated-correction list.
- The seed-related branches of `write_seed_prior_node_probs` and the validated-edge branch of `boost_validated_edges` in `seed_prior.py`. These are replaced by the new boost pass keyed off the corrections list.
- The validated-aware `database_has_annotations` / `use_annotations=True` plumbing in `solve.py` — no REAL/FAKE-via-Ultrack-annotations is used anymore. (FAKE marking is still applied; whether Ultrack's `fix_annotations` is invoked needs verification — see Open questions.)
- `validated_tracks` round-tripping in `multi_threshold.py` and `nucleus_workflow_widget.py` is replaced by the unified corrections list.

## UI surface (separate work; not implemented in this cut)

- "Validate frame": current "validate track" action restricted to one frame.
- "Anchor here": click on a frame, choose existing `cell_id` or new cell, write an `anchor` record.
- The widget's existing validated-track action becomes "validate every frame in this track" — sugar over per-frame validate.

## Out of scope for this cut

- Free anchors (anchor at a position with no hierarchy candidate, where the user wants to *force* the solver to include that position). Today's plan: those become anchor-stamps in the output and the solver is unaware. If a future workflow needs the solver to actually route through such positions, a separate "free anchor → insert REAL node" mechanism can be added later without breaking this design.
- Inter-frame continuity at validated-track boundaries. The solver sees FAKE-only signal in validated regions, so its tracks may end at the validated boundary. If continuity across the boundary becomes a problem in practice, the fix is to allow the solver one frame of overlap on each side without FAKE marking, but defer that decision until it surfaces.
- Changes to first-solve quality (vector-field registration, motion-aware linking, etc.). Separate threads.

## Open questions to resolve during implementation

- Does Ultrack's solver respect `node_annot = 'FAKE'` when `use_annotations` is not passed, or does FAKE only take effect via `fix_annotations`? If the latter, we keep `use_annotations=True` but only for the FAKE pathway; `REAL` annotations are never written. Needs a small test against Ultrack to confirm.
- Whether `node_prob` is on a scale where a multiplicative `2.0` boost is meaningful given existing values, or whether boost should be additive in logit space. Pick after measuring distribution of existing `node_prob` values.
