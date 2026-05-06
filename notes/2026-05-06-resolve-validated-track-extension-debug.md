# Resolve-from-validated: track extension is broken — debugging notes

**Status:** in progress. Two fixes shipped today did not resolve the user's symptom on real data. Diagnostic on a concrete example (track 93) not yet run.

## Symptom

When the user validates a partial track (a subset of frames where the original tracking is correct), then deletes the spurious cells before/after, then runs "Re-solve from validated", the solver does NOT clean up the spurious merges/splits adjacent to the validated range. The validated frames stay correct (forced by `VarAnnotation.REAL` constraint), but the surrounding bad frames either retain their spurious blobs or get reassigned in incoherent ways.

## Concrete example: track 93 in 2026-04-30 dataset

Path: `/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/2_nucleus/`

Validated: frames 0-6 (`validated_cells.json["93"] = [0..6]`).

Current `tracked_labels.tif` (after a recent resolve run with the new code):
| t | area | centroid (y,x) | comment |
|---|------|-----------------|---------|
| 0 | 237  | (140.1, 165.7) | validated, well-behaved |
| 1 | 248  | (144.6, 166.5) | validated |
| 2 | 258  | (147.6, 168.2) | validated |
| 3 | 285  | (150.3, 165.2) | validated |
| 4 | 289  | (152.2, 164.3) | validated |
| 5 | 321  | (156.1, 164.3) | validated |
| 6 | 333  | (158.7, 164.5) | validated, last good frame |
| 7 | 568  | (96.5, 148.5)  | spurious, 60 px jump, area ~2× |
| 8 | 797  | (84.8, 177.7)  | spurious |
| 9 | 754  | (85.3, 178.3)  | spurious |
| 10 | 992 | (76.1, 192.5) | spurious |
| 11 | 627 | (84.5, 185.4) | spurious |
| 12+ | absent ||

The solver extended track 93 through huge merged blobs at t=7-11 instead of through the small, well-positioned hypothesis it should have. My ID-propagation fix correctly identified the dominant solver track and relabeled it to 93 — so the bug is upstream, in what the solver actually selected.

## User's TrackingConfig

```
power=6.0
appear_weight=-0.10            (10× stronger than default -0.001)
disappear_weight=-0.10
linking_mode="iou"
iou_weight=1.0
max_distance=30
max_neighbors=6
quality_exponent=1.0           (default 8 — user weakened drop_frac dominance)
seed_weight=1.0                (default 0.5)
seed_sigma_space=10.0          (default 25)
seed_tau_time=20.0             (default 2 — user widened time decay)
seed_max_dt=10                 (default 5 — user doubled reach)
```

User has already tuned aggressively to favor extension. Boost should be effective, but isn't.

## What we shipped today

**1. `boost_validated_edges` in `src/cellflow/tracking_ultrack/seed_prior.py`** (lines 108-218)
- For every LinkDB row whose source or target is a `VarAnnotation.REAL` node, compute `_affinity(candidate, seed, cfg)` (size × space × time decay) and add `cfg.seed_weight × affinity` to the existing edge weight.
- Soft additive incentive, no constraint.
- Wired into `resolve_with_canonical_segment` between `run_linking` and `run_solve` (`reseed.py:643-645`).

**2. Track-ID propagation in `merge_validated_into_export`** (`src/cellflow/tracking_ultrack/reseed.py`)
- Find the dominant solver track ID covering each validated mask region.
- Globally relabel that solver track to the validated cell ID across the whole movie.
- Then paste validated masks (geometry wins).
- Tested in `tests/tracking_ultrack/test_reseed_merge.py`.

Both confirmed wired and unit-tested. But the user's real-data run still produces the bad output above.

## Diagnosis prior to today's fixes (still valid)

From haiku investigation of Ultrack solver:
- `VarAnnotation.REAL` adds hard constraint `node_var >= 1` only — does NOT force incoming/outgoing edges. A REAL node can legally remain a single-frame track.
- `appear_weight`/`disappear_weight` enter objective directly (negative values are penalties under maximization).
- `node_prob` is passed through `apply_link_function` (`power=6` here), so `prob^6` collapses sharply for prob<1.
- Edge weights similarly transformed.

Math sanity check (assumed) for a perfect candidate one frame outside the validated range:
- affinity ≈ exp(-0/0.5) × exp(-(0/10)²) × exp(-1/20) = 1.0 × 1.0 × 0.95 = 0.95
- boost adds 1.0 × 0.95 = 0.95 to edge weight
- combined edge weight (assuming IoU base ~0.4) → ~1.35
- after power=6: ~6.0 — a strong reward, dwarfs the 0.10 appear penalty

So *in theory* the boost should make the right candidate win. It doesn't. Something about the assumption is wrong.

## Hypotheses to test (in priority order)

1. **Good candidate hypothesis doesn't exist in the segmentation hierarchy at t=7+.** The `ultrack_segment` watershed-on-contour-map pass might only produce the giant merged blob, with no sub-hypothesis matching the validated cell's area/position. If so, no amount of boost can save it — we'd need to feed better contour maps, or bypass segmentation and inject hypothetical extension nodes.

2. **Good candidate exists but has very low IoU with validated REAL.** If `min_link_iou=0.1` blocks the link entirely (no LinkDB row), the boost has no edge to operate on. Note: validated REAL nodes use the corrected mask; canonical segmentation hypotheses come from a fresh contour-map watershed. They may not align.

3. **OverlapDB constraint forces only the big merge to be selectable.** If the small candidate is in the same `OverlapDB` chain as the merge and the merge happens to win on a different objective term, the small one is excluded. Less likely with strong boost, but possible.

4. **The boost runs but is silently no-op** — e.g., if `LinkDB` rows incident to REAL nodes don't exist (because IoU mode rejects them up front), the function returns `boosted=0`. The notify message would say `boosted 0 link(s)` but the user might not have caught it in the UI.

5. **The exported track from solve is correctly a single-frame at frame 6**, but my ID-propagation re-attributes a *different* solver track (the spurious one at t=7-11) to label 93 because that track happens to overlap a validated-frame pixel region somewhere. — Unlikely because validated frames are 0-6, not 7-11; the dominant overlap must be at 0-6 where the validated cell is well-defined.

## Next step (NOT yet executed)

Write a diagnostic script that reproduces the resolve pipeline up through linking + boost into a non-temp `working_dir`, then inspect SQLite state directly. Concretely:

- Wipe `/tmp/debug_track93_workdir`. Run segment + inject + score + link + boost into it.
- For track 93's validated frames (0-6), find the corresponding REAL `NodeDB` rows by centroid match.
- For REAL@t=6: list all outgoing `LinkDB` rows. For each target, show (t, y, x, area, node_prob, weight). Sort by weight.
- At t=7, list all `NodeDB` rows within 30 px of (158.7, 164.5). Show id, area, node_prob, node_annot. Are any matching cell 93's expected size (~333)?
- Cross-reference: do any of the small/well-positioned candidates have a `LinkDB` row from REAL@t=6 at all? If not, IoU rejected the link at link-time — that's hypothesis 2 confirmed.
- Run boost twice (without committing the second run) — log weight delta to confirm boost is operating.
- Run solve. Report which `selected=True` NodeDB at t=7-11 connects (via parent_id) to REAL@t=6.

Once we know which of (1)/(2)/(3)/(4) is the actual cause, the fix follows:

- If (1): modify segmentation to produce more granular hypotheses (lower `seg_min_frontier`, finer hierarchy), or inject a synthetic continuation node at the validated boundary that matches the validated cell's last-frame shape with a small drift.
- If (2): bypass `min_link_iou` for edges incident to REAL nodes; alternatively force-add a REAL-to-best-candidate link in the linking step.
- If (3): cleanup `OverlapDB` for chains rooted in nodes that conflict with the projected extension path.
- If (4): the boost is the wrong knob; need to tune candidate `node_prob` more, or reduce the power transform on incident edges.

## Files touched today

- `src/cellflow/tracking_ultrack/seed_prior.py` — added `EdgeBoostReport`, `boost_validated_edges`.
- `src/cellflow/tracking_ultrack/reseed.py` — wired boost into `resolve_with_canonical_segment`; rewrote `merge_validated_into_export` to propagate solver track IDs.
- `tests/tracking_ultrack/test_reseed_merge.py` — added `test_merge_propagates_validated_id_along_solver_track`.
- `tests/tracking_ultrack/test_seed_prior_edge_boost.py` — new file, 3 tests.

All new tests pass. Pre-existing test failures (4) are unrelated.

## Where to resume

Run the diagnostic script described under "Next step". Findings determine the fix path.
