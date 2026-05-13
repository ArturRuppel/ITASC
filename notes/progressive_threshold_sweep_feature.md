# Feature Spec: Progressive Multi-Threshold Ultrack DB Builder v2

## Problem Statement

The current `run_ultrack_threshold_sweep_experiment.py` (v1) works functionally but has three fundamental issues:

1. **Browser hierarchy slider is broken** — `t_hier_id` is local to each variant's segmentation. After merge, unrelated nodes from different sources share the same `t_hier_id`. The napari hypotheses browser groups by `(t, t_hier_id)` and paints only the first node per hierarchy ID. The slider cycles through a meaningless mix of hierarchies.

2. **Solver greedily selects too many cells** — All nodes have positive `node_prob` values (~0–1.25). The ILP objective is `MAXIMIZE(sum(node_prob) + edges - 0.001*appear - 0.001*disappear)`. The cost to start/end a track (-0.001) is negligible compared to the reward of selecting a node (+0.5 to +1.0). Additionally, cross-source overlap detection is lossy: the merge reconstructs one labelmap per variant with last-writer-wins semantics, so overlapping nodes within the *same* variant can overwrite each other, causing cross-variant overlaps to be missed.

3. **Input generation is fighting Ultrack's hierarchy** — We pre-binarize the foreground into {0,1} masks and threshold contours externally *before* passing them to Ultrack. This means Ultrack's internal `SegmentationConfig.threshold` has no effect: the foreground already has only one level, so the watershed hierarchy is essentially flat. We then run 6 completely independent segmentations and brute-force merge them, which is both expensive and discards the natural hierarchy.

## Goal

Implement a new pipeline that:
- Passes **continuous** foreground scores to Ultrack so its internal hierarchy mechanism works properly
- Builds contours from **edge detection on masks** (not on probability maps)
- Generates all candidates from a **progressive threshold sweep** that respects hierarchy semantics
- Controls solver greediness via a configurable **bias** parameter
- Produces a merged DB where the **browser hierarchy slider works correctly**

## Detailed Design

### 1. Input Generation (`_generate_inputs_v2`)

Replace: `nucleus_prob_sigmoid_zavg.tif` + 6 binary variants + 2 contour variants
With: two canonical continuous inputs.

**`foreground_scores.tif`**
- Source: `1_cellpose/nucleus_prob_3dt.tif` — cellpose logits, shape `(T, Z, Y, X)`
- `sigmoid = 1.0 / (1.0 + np.exp(-prob))`  (using default k=1, midpoint=0 — logits are negative/positive)
- Average over Z: `foreground_scores = sigmoid.mean(axis=1)` → shape `(T, Y, X)`, float32 in `(0,1)`
- **Do NOT threshold/binarize.** This is a continuous probability map.
- Save as `inputs/foreground_scores.tif`

**`contour_maps.tif`**
- Source: Same cellpose segmentation masks that produced the logits (or re-segment from `fg_3dt.tif`)
- Apply edge detection (`skimage.segmentation.find_boundaries(mode='inner')`) to each frame's mask
- Save as `inputs/contour_maps.tif` — shape `(T, Y, X)`, float32 in `[0, 1]` (1 = boundary, 0 = interior/background)

Rationale: `foreground_scores` and `contour_maps` are derived from the **same underlying segmentation**, ensuring the foreground peaks and the contour boundaries are spatially consistent.

**No gamma sweep.** The probability map is already in `[0,1]` after sigmoid and z-avg.

### 2. Progressive Threshold Sweep

Since `foreground_scores` is now continuous, Ultrack's internal `SegmentationConfig.threshold` **actually works**. When Ultrack segments a continuous foreground, it binarizes internally and builds a multi-resolution watershed hierarchy where:
- Lower thresholds = larger/over-segmented regions (higher in the hierarchy)
- Higher thresholds = smaller/under-segmented regions (leaves)

So **one** Ultrack segmentation run on the continuous foreground already encodes all threshold levels in its hierarchy!

The progressive sweep strategy builds on this:

a. Run Ultrack at **one** initial foreground threshold (e.g. `seg_foreground_threshold = 0.3`) on the continuous `foreground_scores` + `contour_maps`. This produces a full hierarchy stored in `NodeDB` and `OverlapDB`, where all candidate masks already co-exist with meaningful hierarchical relationships.

b. Then, for each additional threshold combination (e.g. lower fg thresholds, or contour modifications), instead of running independent segmentation and merging whole databases, we **augment** the existing database:
- Run Ultrack segment with the new parameters
- Compare the new nodes against existing nodes
- Add only genuinely new candidates (not already present at ~identical mask)
- Link the new candidates into the same hierarchy where possible, or as sibling branches

Open question for the implementer: should this be done by:
- (A) Single segmentation + extract multiple hierarchy levels? (The hierarchy already contains all thresholds)
- (B) Multiple segmentations with progressive addition?
- (C) Multiple segmentations with full merge, but fix `t_hier_id` properly and deduplicate overlapping candidates at identical masks?

**Recommended approach for v2:** Start with **(A)** — single segmentation on continuous foreground — then extract a richer candidate set by configuring Ultrack's `max_segments_per_time` or manually extracting additional hierarchy levels. Use **(C)** only if approach (A) does not produce sufficient candidate diversity.

### 3. New Parameter: `bias`

Add `bias: float = 0.0` to `TrackingConfig` (or pass it directly to `run_solve`).

In the ILP formulation:
```python
solver.set_node_weights(node_probs + bias)
```

A negative bias (e.g. `-0.3` or `-0.5`) dampens weak nodes, forcing the solver to only select high-confidence ones. The default `0.0` preserves backward compatibility.

Why this fixes the "too many cells" problem:
- Current: `+0.5` prob → node is always selected (cost is only `-0.001` appear/disappear)
- With `bias = -0.5`: `+0.5 - 0.5 = 0.0` → neutral, solver only keeps if edge + overlap constraints require it
- Weak nodes with `prob = 0.2` become `0.2 - 0.5 = -0.3` → actively penalized

**TODO:** Confirm how Ultrack's `apply_link_function` transforms the bias (it's applied to both `nodes_prob` and `edge_weights` via the same link function). If `link_function = "power"` and `power = 4`, then `(-0.3)^4 = 0.0081`, which is positive and wrong. The bias should probably be applied **after** the link function, not before.

### 4. Fix Hierarchy Slider (`t_hier_id` Remap)

After any multi-source merge (variants or progressive addition), remap `t_hier_id` globally:

```python
# Pseudocode
hier_counter = 1
hier_remaps = {}  # (source_id, old_t_hier_id) -> new_t_hier_id

for node in all_merged_nodes:
    key = (node.source_db_id, node.t_hier_id)
    if key not in hier_remaps:
        hier_remaps[key] = hier_counter
        hier_counter += 1
    node.t_hier_id = hier_remaps[key]
```

This ensures every original hierarchy from every source gets a globally unique ID. The browser's `seen_hierarchies` set will then correctly treat them as independent paint groups.

Additionally, for nodes with `t_hier_id = 0` (typically unassigned / hierarchy root), ensure they receive a valid unique ID or are handled consistently.

### 5. Files to Create / Modify

**New files:**
- `src/cellflow/tracking_ultrack/progressive_merge.py` — core logic for progressive sweep
- `scripts/run_ultrack_progressive_sweep.py` — new experiment runner replacing v1

**Modify:**
- `src/cellflow/tracking_ultrack/config.py` — add `bias: float = 0.0`
- `src/cellflow/tracking_ultrack/multi_threshold.py` — fix `t_hier_id` global remap in `merge_ultrack_databases`
- `src/cellflow/tracking_ultrack/solve.py` — apply `bias` in solve pipeline
- `src/cellflow/tracking_ultrack/db_build.py` — accept continuous `foreground_scores` input; remove double-binarize warning logic

### 6. Validation Requirements

After implementation, verify:
1. Browser hierarchy slider shows coherent, non-skipped hierarchies
2. Solver with `bias=-0.3` produces fewer tracks than `bias=0.0`
3. Single continuous segmentation produces comparable or better node coverage than 6 independent binary segmentations
4. Report includes: node count, hierarchy count per source, cross-source overlap count, solve runtime, solve track count

## Questions for Consideration

1. Should the progressive sweep use Ultrack's built-in hierarchy extraction (option A), or is there a reason we need independent segmentations at different thresholds (option C)?
2. How should `bias` interact with `link_function = "power"`? Apply before or after power transform?
3. Are contours from `find_boundaries` on masks sufficient, or do we need geodesic/distance-transform contours for better Ultrack hierarchy quality?
4. Should we still keep the old `run_ultrack_threshold_sweep_experiment.py` as a legacy benchmark, or replace it outright?
