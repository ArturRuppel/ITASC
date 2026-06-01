# Simplified Cell Segmentation Widget — Spec

## Motivation

The current `CellWorkflowWidget` runs a four-stage pipeline (flow filtering →
Cellpose-dynamics foreground → threshold×gamma flow-following contour sweep →
6-parameter ICM) with ~13 tunable parameters across the contour stage alone,
many of which produce bad results outside a narrow band. The prototype in
`scripts/experiment_divergence_icm.py` validated a much simpler path that gives
comparable results on pos00/pos01:

- **Foreground and contours come straight from the Cellpose prob/dp** via
  `build_divergence_maps` (no flow filtering, no Cellpose-dynamics masks, no
  threshold×gamma sweep).
- **Segmentation is unary-only** — a contour-aware geodesic Voronoi from the
  nucleus seeds. With the ICM spatial/temporal pairwise weights set to zero the
  optimum is just the per-pixel argmin of the unary, so we call
  `initialize_icm` and take `init_labels` directly (no `refine_icm`). Two energy
  knobs: contour weight (α) and foreground weight (γ).

This spec defines a new widget around that path, adds back the temporal contour
smoothing (proven useful in the old pipeline, now applied to the divergence
contours), and a live whole-pipeline preview like the nucleus atom-extraction
widget.

## Pipeline

All stages operate on the open position directory. Inputs are read-only.

**Decision: this widget consumes the cached divergence maps** produced by
`DivergenceMapsWidget`; it does not compute them. The divergence step
(`build_divergence_maps`) and its params (smoothing σ, median radius,
z-reductions) stay in `DivergenceMapsWidget` as the single source of truth.

| Input | Path | Shape |
|---|---|---|
| Cell contours (divergence) | `1_cellpose/cell_contours.tif` | (T, Y, X) ≥ 0 |
| Cell foreground (sigmoid) | `1_cellpose/cell_foreground.tif` | (T, Y, X) in [0,1] |
| Nucleus seeds (tracked) | `2_nucleus/tracked_labels.tif` | (T, Y, X) |

If the two `1_cellpose/cell_*` maps are missing, the widget shows a "run
Divergence Maps first" status rather than computing them itself.

Stages, in order (the divergence step is **upstream**, in `DivergenceMapsWidget`):

1. **Map cleanup (foreground + contours)** — `atoms.residual`, same scheme as the
   nucleus/atom widget, applied **symmetrically to both maps**: each gets a
   local-mean residual (window, strength) followed by a threshold.
   - `m = clip(map − strength·localmean(map, window), 0)`.
   - `residual(strength=0)` is a no-op (returns the raw map), so foreground
     cleanup degenerates cleanly to "raw sigmoid": `fg_strength=0, fg_threshold=0.1`
     reproduces the approved pos01 result (~69% coverage). Raising `fg_strength`
     subtracts more background.
   - **Foreground** stays on its native sigmoid scale (already [0,1]); the
     threshold (`fg_threshold`) is interpreted there and produces the fill mask
     (stage 3). **Contours** are rescaled to [0,1] by the `contour_norm_pct`
     percentile so `alpha` stays interpretable, then a `contour_threshold` noise
     floor zeros sub-threshold values.
   - Cell caveat: cells tile the field, so high `fg_strength` over-subtracts
     (a 51-px window is mostly cell interior → residual ≈ 0). Default
     `fg_strength` low for cells; that's a default, not a structural limit — the
     knob is the same one the nucleus widget exposes.

2. **Temporal contour smoothing** *(re-added)* — `contour_filtering.contour_memory_filter`.
   - Bidirectional signal-adaptive EMA over time: strong ridges reset memory,
     weak/absent ridges inherit from temporal neighbours. `tau=0` disables.
   - Applied to the **cleaned** contours from stage 1.
   - **Full-run only** — it needs the whole stack, so it cannot run in the
     single-frame live preview (see Preview below).

3. **Foreground mask** — threshold the cleaned foreground from stage 1.
   - `fg_mask = (foreground_clean > fg_threshold) | (nucleus > 0)`.
   - This is the territory the cells are allowed to fill; on pos01,
     `fg_strength=0, fg_threshold=0.1` → ~69% coverage (the target).

4. **Unary-only segmentation** — `initialize_icm` with `lambda_s = lambda_t = 0`.
   - Cost field per frame: `cost = 1 + α·contour + γ·(1 − fg_score)`.
   - This **weighted cost field is itself a previewable intermediate** (see
     Preview) — it is the single image that explains every boundary, so it is
     surfaced as a layer.
   - Geodesic distance (`MCP_Geometric`) from each nucleus seed; assign each
     foreground pixel to its nearest seed (argmin). No iterations.
   - α ≈ 100 makes crossing a full-strength ridge ~100× a clear pixel, which is
     what pulls boundaries onto the contour seams (pos01 finding).
   - Output: `3_cell/tracked_labels.tif`.

## Parameters (complete list)

Grouped by stage. **Primary** = the knobs actually tuned per dataset;
**Advanced** = set-and-forget defaults, collapsed by default. The divergence-map
params (smoothing σ, median radius, z-reductions) are **not** in this widget —
they live in `DivergenceMapsWidget`.

### Map cleanup (same param trio per map, as in the nucleus/atom widget)
| Param | Default | Range | Tier | Meaning |
|---|---|---|---|---|
| `fg_window` | 51 | odd, 3–201 | Advanced | Local-mean window for foreground residual |
| `fg_strength` | 0.0 | 0–1 | **Primary** | 0 = raw sigmoid, 1 = full local-mean subtraction |
| `fg_threshold` | 0.1 | 0–1 | **Primary** | Cleaned-foreground cutoff → fill mask (sigmoid scale) |
| `contour_window` | 51 | odd, 3–201 | Advanced | Local-mean window for contour residual |
| `contour_strength` | 1.0 | 0–1 | Advanced | 0 = raw, 1 = full local-mean subtraction |
| `contour_threshold` | 0.0 | 0–1 | Advanced | Noise floor on normalized contour; below → 0 |
| `contour_norm_pct` | 99.0 | 90–100 | Advanced | Percentile mapped to 1.0 in contour [0,1] normalize |

### Temporal smoothing
| Param | Default | Range | Tier | Meaning |
|---|---|---|---|---|
| `memory_tau` | 0.0 | 0–1 | **Primary** | EMA crossover; ~the contour value you call "weak". 0 = off |
| `memory_floor` | 0.01 | 0.001–0.5 | Advanced | Min per-frame alpha; ghost half-life (~69 frames @0.01) |

### Segmentation
| Param | Default | Range | Tier | Meaning |
|---|---|---|---|---|
| `alpha` (contour weight) | 100.0 | 0–1000 | **Primary** | `cost = 1 + α·contour` — boundary snap strength |
| `gamma` (foreground weight) | 2.0 | 0–100 | **Primary** | `+ γ·(1 − fg_score)` — pull paths toward confident foreground |
| `n_workers` | 4 | 1–cpu | Advanced | Parallel workers for geodesic computation (compute only) |

Primary knobs (what the user actually turns): **`fg_strength`, `fg_threshold`,
`alpha`, `gamma`, `memory_tau`**. Everything else has a default that held across
pos00/pos01.

## UI

A single run path and a single preview path — not the old four run buttons.

- **Pipeline Files** section (collapsible): inputs listed above + the one output
  `3_cell/tracked_labels.tif`. Reuses `PipelineFilesWidget`.
- **Parameters** section (collapsible): one panel, five labelled sub-groups
  matching the stages above. Advanced rows live in a nested collapsed
  "Advanced" block per group so the default view shows only the four primary
  knobs.
- **`◉ Live preview`** toggle (top-level, like the atom widget's preview button).
- **`▶ Run`** button — full pipeline over all frames → writes `tracked_labels.tif`.
- **Correction** section: unchanged, keep delegating to `CellCorrectionWidget`.

### Live preview (single frame, all intermediates)
Mirrors `nucleus_atom_extraction_widget`:
- When `◉` is active, **any param edit or time-slider scrub** recomputes the
  **current frame only**, off the GUI thread (`thread_worker`), with the
  in-flight-worker + `_preview_pending` coalescing so rapid edits collapse into
  one fresh pass.
- Preview produces in-memory napari layers for **every intermediate**, in
  pipeline order:
  1. `foreground (sigmoid)` — raw input map
  2. `foreground cleaned` — after residual+threshold
  3. `contours raw` — raw input map
  4. `contours cleaned` — after residual+normalize+floor
  5. `foreground mask` — the fill territory
  6. **`weighted cost field`** — `1 + α·contour + γ·(1 − fg_score)` over the
     mask, `inf`/blank elsewhere. This is the energy the geodesic walk sees, so
     it is the most diagnostic layer for tuning α/γ — bright ridges should trace
     the seams the labels will land on. Display with a perceptual colormap
     (e.g. `turbo`), background masked.
  7. `cell labels` — the argmin result
  - Layers are full (T, Y, X) stacks with only the current frame filled (so the
    time axis stays aligned), refreshed in place.
- **Temporal smoothing is skipped in preview** (it needs the full stack). The
  preview shows the cleaned, pre-smoothing contour and a status note —
  identical to how the old widget handled `τ` in its contour preview.

### Full run (final output only)
- Read the cached `cell_contours.tif` + `cell_foreground.tif`, run stages 1–4
  over all frames **in memory** (including temporal smoothing), and persist
  **only** `3_cell/tracked_labels.tif`. Intermediates are not written to disk.
- If the cached divergence maps are missing, refuse with a "run Divergence Maps
  first" status (the widget never computes them).
- Runs on a `thread_worker` with a progress bar and a cancel/✕, like the current
  widget's per-stage workers (but one worker for the whole pipeline).

## Reuse / backend changes

Everything the pipeline needs already exists; this is mostly a new widget plus
one thin orchestration helper:

- `residual`, `contour_memory_filter`, `initialize_icm` — used as-is.
  (`build_divergence_maps` stays in `DivergenceMapsWidget`, upstream.)
- New: a `segment_cells_divergence(contours, foreground, nuc, params, *, frame=None, progress_cb)`
  function in `segmentation/` that chains stages 1–4 (or minus-temporal for a
  single `frame`) and returns **all intermediates + labels** — including the
  weighted cost field — as a small result object/dict, so the widget can drop
  each into its preview layer without recomputing. The widget calls this for
  both preview (single frame) and full run (all frames; persists only labels).
  Keeps the widget thin and the pipeline unit-testable without Qt.
- The cost field is already built inside `initialize_icm`
  (`_compute_frame_geodesic`: `cost = 1 + α·contour + γ·(1 − fg_score)`); expose
  it (return it, or factor the cost assembly into a tiny shared helper) so the
  preview shows the *same* array the solver uses rather than a re-derivation.

## Decisions / migration

- **Divergence maps: consume cached outputs.** The widget reads
  `1_cellpose/cell_contours.tif` + `cell_foreground.tif`; divergence params stay
  in `DivergenceMapsWidget`. *(Resolved.)*
- **Replace `CellWorkflowWidget`.** The new widget supersedes it. Remove the
  flow-following contour-sweep UI and the 6-param ICM UI (`CellParamsWidget`).
  Backend modules that only the old path used — `flow_following.py`'s
  consensus-boundary sweep, `cell_foreground.py`, the cellprob×gamma sweep,
  `refine_icm`/pairwise weights in `cell_label_icm.py` — become widget-unused;
  archive or delete after confirming no other consumers (e.g. tests, scripts).
  *(Resolved.)*

## Open questions

1. **Default `fg_strength`.** Structurally kept and fully parametrized (window /
   strength / threshold, same as the nucleus widget). For confluent cells the
   safe default is `fg_strength=0` (raw sigmoid) so the baseline reproduces the
   approved ~70% look; users raise it via the live preview when background needs
   subtracting. Confirm 0 is the right shipped default vs. a small nonzero value.
