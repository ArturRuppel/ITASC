# Atom merge-tree candidate generation with bounded branching

**Date:** 2026-06-10
**Status:** Design — awaiting review
**Area:** `src/cellflow/tracking_ultrack/` (atom-union candidate DB build, stage ②)

## Problem

`build_atom_union_database` (`db_build.py`) builds candidate nodes by calling
`enum_connected_unions` (`atoms.py`), which enumerates **every** connected
subset of atoms up to `atom_union_max_atoms` / `atom_union_max_area`. Two
candidates are marked as overlapping in `OverlapDB` whenever they share an atom,
and ultrack's ILP reads `OverlapDB` as a mutual-exclusion relation.

On dense data this explodes. Measured on
`…/pos03/2_nucleus/atoms.tif` (50 frames, ~100–170 atoms/frame) with the
project config (`max_atoms = 10`, `atom_union_max_area = 8000`):

| frame | overlap pairs (max_atoms=3) | overlap pairs (max_atoms=10) |
|------:|----------------------------:|-----------------------------:|
| t=6   | 3,171                       | 16,490,625 (4.0M after dedup) |
| t=7   | 4,806                       | 145,610,164                  |
| t=15  | 3,371                       | 189,892,803                  |

A single atom ends up shared by 2,000–8,800 candidate unions; overlaps grow with
the **square** of that. The build spends ~99 s on one frame and is OOM-killed on
the densest frames. The DB for pos03 is never produced.

`max_atoms` is the wrong knob: lowering it clips the *deep* end of the lattice
(the maximal connected merge, which we want to keep) while the combinatorial
*middle* still explodes. Enumeration itself is cheap (≤8,740 unions, ≤0.02 s/frame);
the cost is the pairwise overlap construction over a dense non-nested candidate set,
plus materializing millions of ORM rows in memory.

## Goal

Generate a candidate set that:

1. always includes every atom (singletons) and the maximal connected merge per
   component;
2. keeps the *meaningful* intermediate groupings and bounds the combinatorial
   middle;
3. exposes a **single knob** — a per-frame overlap-pair budget — that
   interpolates from the nested hierarchy (budget → 0) to today's full lattice
   (budget → large), denominated in the cost unit that actually matters;
4. degrades gracefully under blow-up — never OOMs, always produces a valid DB;
5. adds no per-candidate scoring cost to the hot path.

## How the original nested approach bounded candidates (reference)

Verified from the installed ultrack source
(`core/segmentation/vendored/hierarchy.py`). Per connected component ultrack
builds **one watershed hierarchy** (a binary merge tree):

- pixels/basins → graph; edge weights = the contour (`edge`) image along each
  boundary (`mask_to_graph`);
- `hg.watershed_hierarchy_by_area` (or `_by_dynamics` / `_by_volume`) merges
  basins one pair at a time in a **single total order** → tree with leaves =
  finest basins, root = whole component, internal nodes = merges
  (~2n−1 candidates for n leaves, not 2ⁿ);
- pruning: `filter_small_nodes_from_tree` (`min_area`); `min_frontier` via
  `attribute_contour_strength` (mean edge weight on the wall separating a node's
  two children — weak walls below threshold collapse the split into the parent);
  `max_area` collapse.

The bound is **structural** (one merge order) + **saliency pruning** (frontier /
area); no per-candidate scoring. Overlaps are ancestor↔descendant only, so the
count stays linear.

## Design

### Overview

Atoms already are watershed basins (atom extraction watershedded the
residual-contour ridge), and the ridge strength on each shared atom border is
exactly `attribute_contour_strength`. So we run the *same* style of hierarchy
ultrack uses, but with **atoms as leaves**, then add bounded branching.

```
atoms.tif ─┐
           ├─► per-frame: atom RAG + per-edge ridge weights
contours ──┘            │
                        ├─ Step 1: backbone merge tree (always)            ─┐
                        └─ Step 2: branch admission, ambiguity-ordered,     ├─► candidates + overlaps ─► DB
                                   capped by the per-frame overlap budget  ─┘
```

### Step 1 — Backbone merge tree (always built, independent of the budget)

Per connected component of the atom adjacency graph:

- nodes = atoms; edges = atom adjacencies; **edge weight = mean residual-contour
  (ridge) value along the shared border** of the two atoms.
- build the watershed hierarchy over this graph (reuse higra, already an ultrack
  dependency: `hg.watershed_hierarchy_by_area` over the atom RAG / its line
  graph, weights = ridge strength).
- apply `min_area`, `max_area`, `min_frontier` exactly as ultrack does.

Every node of the resulting tree is a candidate. Properties:

- leaves = atoms, root = maximal connected merge (both always present);
- overlaps among backbone nodes are ancestor↔descendant only (cheap, linear);
- **at budget → 0 the candidate set is exactly this tree** — one nested
  hierarchy, no alternatives = nested-approach behavior.

The backbone is always built and never charged against the budget — it is the
guaranteed floor. The budget governs only the branch candidates added on top.

Note: this is nested-*equivalent*, not byte-identical to the former pixel-level
higra — leaves are atoms now, not raw watershed basins. Atoms are the agreed
stage-① primitive.

### Step 2 — Branch admission, ambiguity-ordered, capped by the budget

Best-first growth over the same atom graph admits *alternative* connected unions
the tree did not take. At each region the tree merged across the single weakest
wall; an alternative merge across a neighbour wall is "meaningful" in proportion
to how *near-tied* that wall is with the weakest — a near-tie is a coin-flip the
tree may have gotten wrong, a much-stronger wall is a confident separation.

Branch candidates are therefore enqueued in **ascending wall weight** (most
ambiguous first) and admitted in that order until the per-frame **overlap
budget** is reached. This single ordering + single budget *is* the interpolation:

- **budget → 0** → no branches admitted → only the backbone tree → nested
  hierarchy.
- **budget → large** → admission runs until the candidate space is exhausted →
  **all** connected unions ≤ `max_area` = today's full lattice.
- **budget in between** → the tree plus as many near-tie alternatives as fit,
  most-ambiguous first. Non-nested siblings (e.g. {1,2} *and* {2,3}) appear only
  while budget remains, and the cheapest-most-meaningful ones appear first.

Mechanics:

- best-first growth grows connected unions ≤ `max_area`, deduped against the
  backbone and each other via a frozenset `seen` set;
- the budget is charged incrementally (see Blow-up handling) so admission stops
  exactly at the ceiling;
- `max_area` remains a hard physical cap and bounds the search space (growth
  never considers a union exceeding it);
- there is **no `max_atoms` and no separate node cap** — union depth is governed
  by `max_area` + maximal-merge-always, and node count is bounded as a side
  effect of the overlap budget (every admitted branch consumes budget).

### Overlaps

Unchanged semantics: two candidates overlap iff they share ≥1 atom (mutual
exclusion for the ILP). Built from an atom→candidate-id map. Because admission is
capped by the overlap budget (below), overlap count is bounded by construction.
Overlaps are streamed to the DB in chunks (see Blow-up handling), never
materialized as one giant ORM batch.

### Meaningfulness / node scoring

Candidate *selection* is driven entirely by ridge saliency (ambiguity ordering) —
**no node-prob scoring in the candidate-generation hot path**. `node_prob` continues
to be computed later by `apply_annotations_and_score` /
`write_seed_prior_node_probs`, unchanged; it weights nodes for the ILP but no
longer selects candidates. (This supersedes an earlier idea to rank unions by
node-prob during enumeration, which would have added exactly the cost we want to
avoid.)

### Blow-up handling

The build cannot OOM, by construction:

1. **Backbone-first safe floor.** The tree is linear in #atoms and built before
   any branching. Worst case is a fall-back to the nested hierarchy — a valid,
   complete DB.
2. **Budgeted admission, not build-then-die.** Branch candidates are admitted in
   priority order (ascending wall weight — closest near-ties first). The cost
   driver (overlap pairs) is predicted *incrementally*: admitting a candidate
   covering atoms `S` adds `Σ_{a∈S} k_a` overlaps, where `k_a` is how many kept
   candidates already contain atom `a`. A running total is maintained;
   **admission stops the moment the next candidate would cross the budget.** No
   overshoot.
3. **The budget self-adapts per frame.** A sparse frame never reaches the
   ceiling and gets its full (small) candidate space; a pathological dense frame
   throttles toward the tree once the ceiling is hit. Same budget value, same
   run — no per-frame tuning.
4. **Streaming/chunked inserts.** Overlaps (and nodes) are inserted via
   `executemany` in fixed-size chunks, committing per chunk. Memory stays flat
   regardless of count. This removes the mechanical OOM; the budget removes the
   runaway time.
5. **No silent truncation.** Each frame logs whether the budget was hit and how
   many branch candidates were left unadmitted, e.g.
   `frame 7: budget hit — admitted 4,800 branches, ~1.2M candidates skipped`.

### Ridge-weight plumbing

The backbone tree, the `min_frontier` prune, and the branching band all need
per-atom-edge ridge weights, derived from the residual-contour map that stage ①
currently discards.

**Chosen: (a) recompute** the residual-contour from `nucleus_contours.tif` at
DB-gen using the `AtomParams` already embedded in `atoms.tif`
(`read_atoms_params`). Thread the contour path into `build_atom_union_database`
(the foreground path is already available in the widget for scoring; the contour
is a sibling file). No change to the stage-① output contract (`atoms.tif`
remains the sole artifact).

Rejected alternative: (b) persist a per-edge ridge-weight sidecar during atom
extraction — avoids recompute/param-drift but changes the stage-① contract.

Per-edge ridge weight = mean residual-contour value over the shared 4-connected
border pixels between two atoms (the same border traversal already used in
`atom_adjacency` / `_merge_small_atoms`).

## Components / boundaries

- **`atoms.py`**
  - keep `atom_adjacency`; add a ridge-weighted variant returning per-edge mean
    ridge strength (atom RAG with weights).
  - replace the role of `enum_connected_unions` with:
    - `build_atom_merge_tree(adj, weights, areas, *, min_area, max_area, min_frontier)` →
      backbone candidates (frozensets) + nesting/overlap structure;
    - `branch_unions(adj, weights, areas, backbone, *, max_area, overlap_budget)` →
      additional candidates admitted in ascending-wall-weight order until the
      overlap budget is reached, with a report of admitted/skipped counts.
  - `enum_connected_unions` retained (it is the budget → ∞ / full-lattice limit
    and useful for tests), but no longer the default path.
- **`db_build.py`** (`build_atom_union_database`)
  - accept `contour_maps_path` (for ridge recompute);
  - call the merge-tree + branching functions instead of `enum_connected_unions`;
  - build overlaps from the atom→candidate map and stream-insert nodes + overlaps
    in chunks;
  - thread the per-frame branching report into `progress_cb`.
- **`config.py`** (`TrackingConfig`)
  - add `atom_overlap_budget: int = 300_000` (per-frame overlap-pair ceiling —
    the single branching knob);
  - keep `atom_union_max_area` (hard physical cap);
  - **remove `atom_union_max_atoms`** (no longer used anywhere).
- **napari db-gen widget** (`nucleus_pipeline_widget.py`)
  - pass the contour path into the builder;
  - surface `atom_overlap_budget` as the branching control; drop the `max_atoms`
    control;
  - map existing project config `nucleus.db_generation` keys.

## Config / migration

- New db-gen key: `overlap_budget` (→ `atom_overlap_budget`, default `300_000`) —
  the single knob; lower it toward nested, raise it toward full lattice.
- `max_atoms` is **removed** from `TrackingConfig`, the build path, and the
  db-gen widget. A `max_atoms` key left in an existing project
  `cellflow_config.json` is ignored on load (no longer mapped); document this so
  stale keys don't cause confusion.
- Downstream consumers (`swap_candidate.py` containment-lattice walk,
  `reseed.py` overlap pruning, the ILP) are unaffected: `OverlapDB` keeps the
  same share-an-atom semantics.

## Testing

- **Endpoint equivalence:**
  - budget = 0 candidate set == backbone tree (node count linear; overlaps are
    exactly ancestor↔descendant).
  - budget = ∞ (very large) candidate set == `enum_connected_unions` (same
    frozensets, bounded only by `max_area`) on small synthetic graphs.
- **Monotonicity:** candidate count and overlap count are non-decreasing as the
  budget rises.
- **Invariants at every budget:** all singletons present; maximal connected merge
  per component present.
- **Budget cap:** on a synthetic dense clump, overlap count never exceeds
  `atom_overlap_budget`; backbone is always fully present even when the budget is
  hit; the skipped-count is reported.
- **Ambiguity ordering:** when the budget admits only some branches, the admitted
  ones are exactly the lowest-wall-weight (most ambiguous) candidates.
- **Ridge recompute:** residual-contour recomputed from contour + embedded
  `AtomParams` matches the extraction-time map (within float tolerance).
- **Regression on real data:** pos03 `atoms.tif` builds to completion at the
  default budget within a sane time/memory envelope; per-frame overlaps bounded.
- **Memory:** chunked insert keeps peak memory flat as candidate count grows
  (sampled RSS stays within a band).

## Settled default

- **Default `atom_overlap_budget = 300_000`** per frame — comfortably fast, well
  above what a clean frame needs, and a hard ceiling on the densest frames.
  Lower it toward the nested hierarchy for speed; raise it for richer non-nested
  alternatives. Confirm the value at review.
