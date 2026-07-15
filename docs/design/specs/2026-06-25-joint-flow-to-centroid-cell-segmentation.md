# Joint flow-to-centroid cell segmentation (standalone, both channels) — design

> Task #11. **✅ IMPLEMENTED 2026-06-25** by recovering `flow_following`. Standalone
> Cellpose distro only — same hard constraint as P3/P3.5: the integrated app's
> `CellposeWidget`, `cell_label_icm` (geodesic Voronoi), divergence path, and
> `cellpose_runner` stay byte-for-byte unchanged.
>
> **Shipped:** `cellflow/cellpose/flow_following.py` (recovered two-phase
> integrator + bounded orphan drop), `cellflow/cellpose/joint.py`
> (`joint_segment_track` orchestration), and the standalone widget's **Joint** (⧉)
> action (enabled only when both inputs are set). Tests:
> `tests/cellpose/test_flow_following.py`, `tests/cellpose/test_joint.py`,
> and the joint cases in `tests/napari/test_cellpose_segment_track_widget.py`.

## The idea (user's words)

> "If there is only nucleus or only cell, we can just go with standard cellpose
> plus laptrack. But when there are both, we should use our custom,
> cellpose-adjacent algorithm, where we flow pixels along the flow field and then
> assign them to the nucleus where they're closest to the centroid."

So: **one cell per nucleus, cell identity == nucleus identity**, with the cell's
*extent* decided by Cellpose's own flow field rather than by naive distance.

## Why this is the right shape

The integrated app already does the nucleus-anchored thing — `cell_label_icm`
seeds a **geodesic Voronoi** at nucleus centroids over a divergence cost field.
The standalone can't use that path (it's the app's, and it needs the divergence
maps). The user's flow-trace variant is the **cellpose-adjacent** equivalent: it
reuses Cellpose's *raw flow field* (the vectors Cellpose itself integrates to
form masks) and reroutes the final assignment from "cluster by sink" to "assign
to nearest nucleus centroid." It is strictly better than a plain
Euclidean-distance Voronoi because the flow respects learned cell shape: a pixel
spatially closer to a neighbour's nucleus but flowing toward its own ends up
correctly assigned.

## What we already have (no new eval plumbing needed)

This is the simplification vs the earlier assumption that we'd need a
flow-retaining `native_masks` variant. We don't:

| Need | Existing function | Returns |
|---|---|---|
| Cell **flow field** + foreground | `cellpose_runner.run_cell_stack(stack, CellParams)` | `prob (T,Z,Y,X)`, `dp (T,Z,2,Y,X)` |
| Nucleus **seeds** (labels → centroids) | `native_masks.run_nucleus_masks_stack(stack, NucleusParams)` | labels `(T,Z,Y,X)` |
| Time linking | `track_laptrack.track_masks(masks)` | track-consistent labels |

Joint mode **does not run cell native masks at all** — it replaces them. So there
is no redundant `model.eval` and nothing in `native_masks._eval_masks` (which
discards flows) needs to change. We compose existing, app-shared functions
read-only.

## Algorithm (per 2D plane — aligns with P3.5 per-frame direction)

Inputs for plane `(t, z)`: cell foreground `prob`, cell flow `dp = (dy, dx)`,
nucleus labels `nuc` (→ centroids `{label: (cy, cx)}`).

1. **Foreground.** `fg = sigmoid(prob) > 0.5` (cell pixels to assign). Reuse the
   app's foreground convention from `divergence_maps` so the threshold matches.
2. **Flow-trace.** For each `fg` pixel, Euler-integrate its position along the
   normalised flow field for `n_iter` steps (bilinear interpolation of `dp` at the
   running sub-pixel position). Each pixel lands at a convergence point near its
   cell's centre — exactly Cellpose's dynamics, minus the sink-clustering step.
3. **Assign.** Label each `fg` pixel with the **nucleus whose centroid is nearest
   to its landing point** (KD-tree over the plane's nucleus centroids). Result:
   `cell_label == nucleus_label`, one cell per nucleus.
4. **Guards.** A landing point with no nucleus within `max_assign_radius` →
   background (orphan foreground, not force-assigned). Pixels never leave `fg`.

Output: cell labels `(T,Z,Y,X)` keyed by nucleus id, paired 1:1 with the nuclei.

## Tracking comes free

Because cell label == nucleus label *per frame*, we **track the nuclei** (laptrack
on the nucleus centroids — point-like, robust) and **propagate the nucleus track
id to the cell**. No separate, fragile cell tracker; matched nucleus/cell tracks
by construction. This also answers P3.5's "keep cells cohesive across an axis"
for the joint case: cohesion is inherited from the nucleus track.

## New module surface (additive, standalone-only)

`src/cellflow/cellpose/joint_assign.py` (Qt-free):
- `flow_trace_plane(dp_2d, fg_2d, *, n_iter, dt) -> landing (2, n_fg)` — Euler
  integration of foreground pixels along the flow field.
- `assign_to_nearest_nucleus(landing, fg_2d, centroids, *, max_radius) -> labels_2d`.
- `joint_cell_labels(cell_prob, cell_dp, nucleus_labels, *, params) -> (T,Z,Y,X)`
  — orchestrates per plane.
- `joint_segment_track(nuc_stack, cell_stack, nuc_params, cell_params, *, ...)`
  — full path: nucleus masks → cell flows → per-plane assign → track nuclei →
  propagate ids to cells → return `(nucleus_tracked, cell_tracked)`.

App imports nothing from here; `cell_label_icm`/divergence untouched.

## Widget UX (standalone seg&track)

When **both** file pickers are populated, expose an explicit third action —
**Joint (nucleus-anchored cell)** — rather than silently changing what per-channel
**Track** does. Explicit is discoverable and keeps the single-channel path
(native masks + laptrack) exactly as shipped. Greyed out until both inputs exist;
outputs land as `[Nucleus] tracked` + `[Cell] tracked` layers (paired ids).

## Decisions (settled 2026-06-25)

1. **Flow integrator — recover `flow_following.py`** (the proven prior art,
   added 2026-05-05, deleted 2026-06-03 in the publication cleanup `81df367`;
   last live at `39adde3`). It's richer than a plain integrator: per-pixel Euler
   integration of the Cellpose flow field **blended with an EDT "gravity" vector
   toward the nearest tracked nucleus** (`flow_weight`), a direct-capture when a
   pixel lands on a nucleus, then a **progressive shell** assignment for the
   rest. We do NOT use `cellpose.dynamics` and do NOT reinvent. numba is already
   present via the cellpose stack; we lazy-JIT the kernel (pure-Python fallback so
   the module imports and tests run without numba).
2. **Engage mode — explicit Joint button** when both inputs are set. Single-channel
   path (native masks + laptrack) stays exactly as shipped.
3. **Foreground threshold — exposed knob** (`fg_threshold`, default 0.5 on the
   sigmoid), not hard-wired.
4. **Orphans — dropped.** The recovered code's *unlimited* phase-2 fallback is
   replaced by one **bounded by `max_assign_radius`**: a foreground pixel whose
   displaced position has no labelled pixel within the radius stays background.

## Test plan (TDD, all Qt-free except the widget action)

- `flow_trace_plane`: a synthetic radial flow converges foreground to the centre.
- `assign_to_nearest_nucleus`: two nuclei + a flow that routes a spatially-ambiguous
  pixel to the *correct* (flow-target) nucleus, not the nearest-by-distance one.
- `joint_cell_labels`: cell labels equal nucleus labels; one component per nucleus.
- propagation: nucleus track ids carry onto the cell stack (cell tracked == nucleus
  tracked ids).
- orphan guard: foreground with no nucleus in range stays background.
- **app-unchanged**: `cell_label_icm`, `divergence_maps`, `cellpose_runner`
  imports/signatures unchanged; no new caller of app widgets.
