# Plan: replace IoU linker with retracker-form scoring

## Motivation

We have three similarity scoring paths that should converge on one formula:

- **Default linker** (`ultrack.core.linking.processing.link`) — `raw_IoU(src, tgt) − distance_weight × distance`. Raw IoU punishes any centroid drift; fast-moving cells score poorly even with identical shape.
- **IoU linker** (`tracking_ultrack/linking.py::_run_iou_linking`) — centroid-corrected IoU + linear distance, convex-blended. Translation-invariant.
- **Greedy retracker** (`tracking_ultrack/extend.py::_extend_score`) — `area_ratio + centroid_corrected_iou + distance_score − overlap_penalty × existing_overlap`. Additive, with area term.

The retracker consistently picks better next-candidates for anchored cells than the ILP solver because the ILP's edge weights (built at link time, from the default linker against the original hypothesis masks) miss the shape signal and the area signal.

The IoU linker is a strict subset of the retracker's scoring (no area-ratio term). Replacing it with the retracker's formula collapses (b) and (c) into one source of truth and lifts the linker's quality to match the retracker's.

## Goal

One scoring function for similarity, used by:
1. The linker (every edge in the graph).
2. The greedy retracker (next-candidate suggestions in the napari widget).
3. (Future) Anchor-incident rescore, if a residual gap remains after (1).

## Scope

In: `tracking_ultrack/linking.py`, `tracking_ultrack/config.py`, `tracking_ultrack/extend.py` (refactor to share scoring).
Out (this plan): `_run_iou_linking`'s **caller path** (`run_linking` dispatch, callers in `reseed.py`, `db_build.py`). The dispatch survives; the mode just gets renamed and its body replaced.
Out (this plan): anchor-incident rescore. Defer until (1) is measured.

## Changes

### 1. New shared scoring module

Create `src/cellflow/tracking_ultrack/scoring.py` containing:

- `centroid_corrected_iou(mask_a, mask_b)` — rasterize-based implementation lifted from `linking._aligned_mask_iou`, since it's 5–10× faster than the set-based version for nucleus-sized masks. Takes raw boolean arrays + origin offsets + centroids (the linker's calling convention) and produces a float in [0, 1].
- `similarity_score(*, area_ratio, centroid_corrected_iou, distance, d_max, area_weight, iou_weight, distance_weight)` — the body of `extend._extend_score`, **without** the `overlap_penalty` term (overlap is global; the ILP handles it via `OverlapDB`).

No circular imports — both `linking.py` and `extend.py` import from `scoring.py`.

### 2. Update `extend.py`

- `_extend_score` becomes a thin wrapper that calls `scoring.similarity_score` and subtracts `overlap_penalty × existing_overlap` itself (retracker still needs the overlap term for local mask-painting suggestions).
- `_centroid_corrected_iou` deletes; calls go through `scoring.centroid_corrected_iou`.

### 3. Rewrite `linking.py::_run_iou_linking`

- Rename mode `"iou"` → `"shape"` (or `"retracker"`; bikeshed). Update `cfg.linking_mode` type literal and any callers / config tests.
- New body:
  - Same KDTree-with-`2 × max_neighbors` pattern over consecutive frame pairs.
  - For each candidate pair, compute `area_ratio` from `node.area` (cheap, scalar).
  - **Area-ratio prefilter:** if `area_ratio < cfg.min_area_ratio` (new config knob, default ~0.3), skip without computing IoU. Cuts most of the cost.
  - Otherwise compute `centroid_corrected_iou` via the shared helper, **caching `np.argwhere(mask)` per node id** to avoid recomputing source coordinates `2 × max_neighbors` times per node.
  - Hard filter on `centroid_corrected_iou < cfg.min_link_iou` (semantics unchanged from current IoU mode — the filter lives in shape space, not combined-score space).
  - Compute distance, call `scoring.similarity_score`, push edge.
- Delete `_aligned_mask_iou`, `_node_origin`, `_node_mask`, `_centroid_tail`, `_blend_score`.

### 4. Config changes (`config.py`)

Remove:
- `cfg.iou_weight` (the [0, 1] convex-blend knob; no analog in the additive form).

Add (defaults matching `extend.py`'s defaults so behavior is consistent across paths):
- `area_weight: float = 1.0`
- `iou_weight: float = 1.0` (different meaning now — additive weight, not blend ratio)
- `distance_weight: float = 0.25`
- `min_area_ratio: float = 0.3`

Keep:
- `max_neighbors`, `max_distance`, `min_link_iou`, `linking_mode` (with new `"shape"` literal).

### 5. Tests

- Existing IoU-mode linker tests under `tests/tracking_ultrack/` — update assertions for the new score formula (score range is no longer [0, 1]; expected weights change).
- One new unit test: two-cell synthetic frame pair with one good-shape-match candidate and one closer-but-wrong-shape candidate. Assert the shape-correct candidate wins after linking.
- One test for the area-ratio prefilter: candidate with `area_ratio < min_area_ratio` produces no edge.

### 6. Sign-convention check

Quick read of the ILP objective to confirm `LinkDB.weight` semantics (higher = more preferred). The current IoU linker writes blended scores in [0, 1] and assumes higher-is-better; the default linker writes `IoU − distance_w × dist` and assumes the same. Our additive form is also higher-is-better. No inversion needed, but verify before merging.

## Performance plan

The substitution does not add cost beyond what IoU mode already pays. Budget for typical nucleus tracking (300 frames × 200 cells × 10 candidates ≈ 600K pairs):

- Naive (no caching): ~30–120 s for the full link step.
- With area-ratio prefilter + argwhere cache: should drop to ~10–30 s.

Approach: implement, time the new "shape" mode on the curated 10-frame dataset (`2026-04-01_U251.../v2/pos00/2_nucleus/`) and on whichever larger stack is handy. If link time dominates the run, profile and tighten. No premature optimization beyond the two cheap wins (prefilter + cache) already in the plan.

## Followup (not in this plan)

After the substitution lands and is timed:

- If anchored cells still get worse next-frame links than the greedy retracker would pick, build the anchor-incident edge rescore (a single function that walks `VarAnnotation.REAL` nodes and overwrites incident `LinkDB.weight` with `scoring.similarity_score` against the user-trusted mask, not the original hypothesis mask).
- Likely small once the linker is already using the same formula — the only remaining asymmetry is "anchor's mask was edited after linking ran, so the precomputed weights are stale for its incident edges."

## Open questions

1. Mode rename: `"shape"` or `"retracker"` or keep `"iou"` for backward compat with any persisted config?
2. Should `boost_validated_edges` and `write_seed_prior_node_probs` (`seed_prior.py`) survive once the linker is doing most of this work? Out of scope for this plan but worth deciding before adding a third path on top.
