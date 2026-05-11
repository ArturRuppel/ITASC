# Brief — 2D+T multi-label graph-cut cell segmentation experiment

**From:** primary Opus session
**To:** parallel Sonnet session
**Date:** 2026-05-10

You are a parallel session working in the same repo as the primary session. We communicate via files in this folder:

- `BRIEF.md` — this file (read-only for you).
- `STATUS.md` — you write here. Use it to ask clarifying questions and report progress.
- `RESULTS.md` — you write here at the end with metrics and observations.
- The primary will respond in `STATUS.md` (look for lines beginning with `[OPUS]`).

## Why this experiment exists

The repo's "stage 4" cell segmentation widget (`src/cellflow/napari/cell_workflow_widget.py`) takes 3D Cellpose prob+flow + 2D ground-truth nucleus tracks and emits 2D+T cell labels via a per-frame "flow-following gravity walk" (each foreground pixel walks toward the nearest nucleus, blending Cellpose flow with EDT-gradient gravity, and inherits its track ID once it lands). It works decently in cell interiors but produces artifacts and temporal flicker, both concentrated at cell-cell interfaces. Frames are processed independently — no label-level temporal coupling.

Many prior experiments (`scripts/experiment_cell_*.py`) tried fixing this by pre-filtering the contour signal (median, Gaussian, optical-flow blend, agreement, Voronoi fusion) or by per-frame watershed/graph-cut. None added explicit label-level temporal coupling. A prior Ultrack-style hypothesis-graph attempt for cells (`archive/v1/packages/ultrack/...`) was abandoned because Ultrack's discrete watershed-hierarchy candidate pool is structurally a poor fit for fuzzy cell ridges (see `notes/2026-05-06-watershed-hierarchy-skips-intermediate-levels.md`).

We've concluded the right shape of solution is **continuous (no discrete hypothesis pool), joint over (x,y,t), and anchored at nucleus tracks**. This experiment tests one such formulation: a multi-label α-expansion graph cut over the full 2D+T volume.

## What to build

Exactly one new script: `scripts/experiment_cell_2d_t_multilabel_graphcut.py`. Follow the conventions of the other `scripts/experiment_cell_*.py` files (argparse with `--pos-dir`, `_write_json`, timestamped output dir, etc.).

### Inputs (read these from `<pos-dir>`)

Default `--pos-dir`: `/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00`

| Role | Path under `pos-dir` |
|---|---|
| Nucleus tracks (anchors) | `2_nucleus/tracked_labels.tif` (T,Y,X) int |
| Foreground mask | `3_cell/foreground_masks.tif` (T,Y,X) bool/uint8 |
| Contour map | `3_cell/contour_maps.tif` (T,Y,X) float — used both in unary and pairwise costs |
| Cellpose prob (reference, may not be needed in v1) | `1_cellpose/cell_prob_zavg.tif` |
| Cellpose flow (reference, may not be needed in v1) | `3_cell/filtered_dp.tif` |
| **GT cell labels (for evaluation only)** | `3_cell/tracked_labels.tif` (T,Y,X) int — curated benchmark |

If any path doesn't exist or the shape doesn't match, write a `[SONNET]` block in `STATUS.md` and wait — do not improvise.

### Algorithm

Build a multi-label graph cut over the full 2D+T foreground volume.

- **Nodes:** every foreground voxel (x,y,t).
- **Labels:** integer track IDs that are alive at frame t (a track ID is "alive at t" iff the nucleus mask at frame t contains pixels with that ID). Assign a single sentinel label `0` for "background / no cell" — but background pixels (outside foreground) are excluded from the graph entirely; do not give label 0 to foreground pixels unless absolutely necessary.
- **Hard nucleus anchors:** any pixel that is *inside* a nucleus mask at frame t must take that nucleus's track ID. Encode as a very large unary cost (`INF = 1e9`) for any other label at that pixel.
- **Unary cost** for foreground pixel p at frame t and candidate label k (alive at t):
  - Compute geodesic distance from nucleus k's mask at frame t to every foreground pixel at frame t, with cost field `cost_field = 1.0 + alpha_unary * contour_map(t)`. Use `skimage.graph.MCP_Geometric` once per (frame, label).
  - `unary(p, k) = geodesic_distance(p, k) at frame t`. Pixels unreachable from nucleus k get `INF`.
  - Normalize per frame so unaries are comparable across frames (e.g., divide by the median reachable distance per frame).
- **Spatial pairwise** (4-neighbor in 2D, within the same frame): truncated Potts.
  - `pair_spatial(p, q) = lambda_s * exp(-beta_s * 0.5 * (contour(p) + contour(q)))` if labels differ, else 0.
  - Idea: high contour → cheap to switch labels (boundary snaps to ridge); low contour → expensive (interior is smooth).
- **Temporal pairwise** (between (x,y,t) and (x,y,t+1)): constant Potts.
  - `pair_temporal = lambda_t` if labels differ, else 0.
  - This is the new ingredient — it forces label continuity across time.
- **Solver:** pygco α-expansion. Use `pygco.cut_general_graph` (the same package used in `scripts/experiment_cell_2d_graphcut_fast.py`).

### Defaults to start with (v1, no sweep)

```
alpha_unary  = 4.0      # contour weight in the geodesic cost field
lambda_s     = 1.0      # spatial pairwise weight
beta_s       = 5.0      # contour sensitivity in spatial pairwise
lambda_t     = 1.0      # temporal pairwise weight
n_iters      = 3        # α-expansion iterations
INF          = 1e9
```

These are starting points; we'll sweep later. The first run is "does this formulation produce something qualitatively reasonable, and how does it compare to the existing pipeline."

### Memory

The full volume could be tens of millions of voxels. Build the graph sparsely (only foreground voxels are nodes). If pygco runs out of memory, fall back to processing in 50-frame chunks with a 5-frame overlap and stitching by majority vote on the overlap. Document whichever strategy you used.

### Output

Write under `<pos-dir>/4_cell_graphcut/<YYYYMMDD-HHMMSS>/`:

- `cell_labels.tif` — the predicted label volume (T,Y,X) uint16, same shape as inputs.
- `params.json` — every parameter you used.
- `metrics.json` — see below.
- `comparison_frame.png` — a single mid-movie frame showing predicted vs GT side-by-side.

### Evaluation against GT

Compute and report in `metrics.json`:

1. **Per-track temporal-IoU:** for each track ID alive in BOTH pred and GT, compute IoU per frame, average over the frames where both exist, and aggregate (mean / median / 25th-pctile across tracks). Report per-track too.
2. **Coverage:** fraction of foreground voxels that received any non-zero label.
3. **ID consistency:** for each pred track, the fraction of its voxels that fall inside a single GT track (purity), and conversely (completeness). Report both averaged.
4. **Label flicker rate:** for each track, fraction of pixels whose label differs between consecutive frames after rigid alignment by centroid. Compute for both pred and GT (GT should be near-zero, lower is better for pred).
5. **Boundary alignment:** ratio of contour-map intensity at predicted label boundaries vs. interiors (same as `scripts/experiment_cell_2d_optical_flow_watershed.py`).

Be honest about failure modes. If the pred is obviously wrong somewhere, say so in `RESULTS.md` and include a note about which metric caught it.

## Constraints

- **Do not modify any source code under `src/`.** This is an experiment script only.
- **Do not run a parameter sweep in v1** — single run end-to-end with the defaults above. We're checking viability, not tuning.
- **Do not invoke any cellflow napari or correction code.** The experiment should be a self-contained script using only `cellflow.segmentation.centroid_markers_from_labels`-style utilities and standard libs (numpy, scipy, scikit-image, tifffile, pygco).
- **Use existing patterns.** Read `scripts/experiment_cell_3d_geodesic_voronoi.py` for the geodesic + MCP_Geometric pattern, and `scripts/experiment_cell_2d_graphcut_fast.py` for the pygco usage pattern. Follow their CLI / output / JSON conventions.

## Workflow

1. **Read this brief and the orientation files**. Then write a `[SONNET]` block in `STATUS.md` containing:
   - Your understanding of the task in your own words (3-5 sentences).
   - Confirmed paths to the four input files (or report any that don't exist).
   - Any disagreements with the algorithm or default parameters.
   - Any clarifying questions.
2. **Wait** for an `[OPUS]` ACK in `STATUS.md` before implementing. Don't write the script until you see it.
3. **Implement** the script.
4. **Run it end-to-end** on the default `--pos-dir`.
5. **Write `RESULTS.md`** with: parameters used, runtime, all metric numbers, qualitative observations (3-5 bullets — what worked, what didn't, where the formulation succeeded or failed), and recommended next steps.

## Orientation files (read first)

- `scripts/experiment_cell_3d_geodesic_voronoi.py` — geodesic distance + MCP_Geometric pattern.
- `scripts/experiment_cell_2d_graphcut_fast.py` — pygco α-expansion usage.
- `src/cellflow/napari/cell_workflow_widget.py` and `src/cellflow/segmentation/flow_following.py` — current pipeline (so you understand what we're replacing; do NOT modify them).

## Format for STATUS.md

Append-only. Each block is delimited by a header line. Don't rewrite previous blocks.

```
[SONNET 2026-05-10 14:23] understanding-and-questions
... your text ...

[OPUS 2026-05-10 14:30] ack
... my text ...

[SONNET 2026-05-10 15:10] implementing
... your text ...

[SONNET 2026-05-10 16:00] done
... pointer to RESULTS.md ...
```
