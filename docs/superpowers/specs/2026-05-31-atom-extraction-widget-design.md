# Atom Extraction Widget — Design

**Date:** 2026-05-31
**Status:** Draft for review
**Spec sequence:** ① Atom Extraction (this doc) → ② Candidate DB-Gen → ③ Browser rework

## Context

The nucleus tracker currently builds Ultrack candidates with the higra
watershed hierarchy. A validated prototype (`scripts/experiment_atoms_to_db.py`,
tagged `atoms-prototype`) replaces that with an **atom-union** candidate
structure: split the foreground into atoms, then enumerate every connected
union of atoms as a candidate. This fixes higra's missing-permutation weakness
(for touching basins A,B,C it stores A∪B and A∪B∪C but never B∪C).

The atoms come from a **local-mean-subtracted residual** of the cellpose maps:
flattening each map's per-nucleus offset lets a single global threshold catch
faint nuclei without lighting up background. The quality of the whole pipeline
rests on the atoms being right, and "right" is judged visually — so atom
extraction needs an interactive, preview-driven tuning surface.

**Decisions locked in discussion:**
- The atom path **replaces** the higra path entirely (no coexistence).
- Generation is specced **before** the browser; the browser rework is a later
  spec.
- **Single** territory threshold for now (no multi-source threshold sweep).
- Atom extraction and its **live preview live in this (Atom Extraction)
  widget**, not in DB-Gen. All knobs that shape the atoms are tuned here against
  a recoloured-atoms preview.

## Goal

A napari widget that lets the user tune the five knobs that produce atoms and
see the resulting atoms recoloured live on the current frame, plus a shared,
deterministic atom-extraction function that DB-Gen (spec ②) reuses over the full
stack. No DB writing happens here.

## Scope

**In scope**
- A reusable core module: residual + atom extraction over a frame and a stack.
- The `NucleusAtomExtractionWidget` (params + activate + live preview mixin),
  wired into the nucleus workflow widget.
- Config fields for the five atom params; removal of the now-dead higra
  candidate params.
- The handoff contract consumed by DB-Gen.

**Out of scope (later specs)**
- Building NodeDB/OverlapDB from atoms, union enumeration, linking, scoring,
  solve (spec ②).
- The database browser rework (spec ③).
- Multi-source threshold sweeps.

## Stage decomposition

```
cellpose maps ──▶ ① Atom Extraction (this spec)
  fg.tif            knobs : W_fg, fg_cutoff, W_c, contour_floor, atom_min_area
  contour.tif       preview: recoloured ATOMS over raw fg (current frame)
                    output : deterministic extract_atoms_stack(...) + persisted
                             params
                              │  atoms + residual_contour (recomputed on demand)
                              ▼
                  ② Candidate DB-Gen (spec ②) — atoms → union lattice →
                    NodeDB/OverlapDB → link → score → solve
                              │
                              ▼
                  ③ Database Browser rework (spec ③)
```

## Core module

Promote the prototype logic into the package (suggested
`src/cellflow/tracking_ultrack/atoms.py`) so the widget preview and the DB-Gen
full-stack build call the exact same code.

```python
def residual(frame, window):
    """clip(frame - threshold_local(frame, window, method='gaussian'), 0)."""

def extract_atoms_frame(residual_contour, territory, contour_floor, atom_min_area):
    """ridge = residual_contour > contour_floor
       cores = territory & ~ridge
       atoms = watershed(residual_contour, markers=label(cores), mask=territory)
       merge atoms < atom_min_area into neighbours by re-flood (no holes)."""

def extract_atoms_stack(fg, contour, params) -> (atoms, residual_contour):
    """Per frame: residual_fg = residual(fg, W_fg); territory = residual_fg >
       fg_cutoff; residual_contour = residual(contour, W_c); then
       extract_atoms_frame(...). Returns the atom label stack and the float
       residual_contour stack (the watershed elevation DB-Gen also needs)."""
```

`window` is forced odd. The function is pure (no I/O), deterministic, and the
single source of truth for "what an atom is."

## Parameters

| knob | type | range | default | role |
|---|---|---|---|---|
| `W_fg` (foreground residual window) | int (odd) | 3–301 | 51 | local-mean window for the fg map; must exceed a nucleus |
| `fg_cutoff` (territory threshold) | float | 0–1 | 0.01 | residual-fg threshold defining nucleus territory |
| `W_c` (contour residual window) | int (odd) | 3–301 | 51 | local-mean window for the contour map |
| `contour_floor` (ridge noise floor) | float | 0–1 | 0.01 | residual-contour cutoff; keeps watershed off speckle |
| `atom_min_area` | int (px) | 0–5000 | 100 | atoms smaller than this merge into a neighbour |

Defaults are the validated-prototype values. `fg_cutoff` was tuned visually in
the prototype between ~0.002 and 0.01; 0.01 is the verbally-locked default and
users will retune by eye. Windows default equal but are independent.

## Widget

Mirrors the existing section idioms (`CollapsibleSection`, `dslider`/`islider`,
the stage-header `⏻` activate button, the browser's debounced `QTimer`
preview-refresh).

**Layout** — a "Atom Extraction" collapsible section with a header activate
toggle, holding:
- Residual group: `W_fg`, `fg_cutoff`, `W_c`, `contour_floor` (paired rows).
- Atoms group: `atom_min_area`.
- A small status label (atom count for the current frame).

**Preview behavior**
- Activating (`⏻`) hides non-preview layers (as the browser does), loads the raw
  `fg` image if absent, and renders atoms for the current frame.
- Any slider change schedules a debounced (~150 ms) recompute of
  `extract_atoms_frame` for the **current frame only**, then updates a single
  `[Atoms] preview` Labels layer (recoloured).
- Changing the napari frame recomputes for the new frame.
- Optional toggles (off by default) to add `territory` and `residual_contour`
  overlay layers for diagnosing why atoms came out a given way.
- Deactivating removes the preview layers and restores prior layer visibility.

Performance: per-frame residual + watershed is a few hundred ms, matching the
existing one-frame-at-a-time browser rendering — fine for interactive tuning.

## Persistence & config

- Extend `TrackingConfig` (`tracking_ultrack/config.py`) with: `fg_window`,
  `fg_cutoff`, `contour_window`, `contour_floor`, `atom_min_area`.
- Remove the dead higra candidate params now that higra is replaced:
  `seg_min_frontier`, `seg_ws_hierarchy`, `seg_foreground_threshold`, and the
  source-threshold-pairs machinery (`source_*` controls, `threshold_pairs`).
  `seg_min_area`/`seg_max_area` are superseded by `atom_min_area` and (in spec
  ②) `min_area`/`max_area`; remove or repurpose in ②.
- The widget reads/writes these via the same workflow settings/state path the
  current tracking-inputs controls use.

**Source of truth = the params, not files.** The widget persists the five
params; DB-Gen recomputes atoms over the full stack deterministically. This
avoids stale-intermediate-file bugs. (See decision D1.)

## Handoff contract to DB-Gen (spec ②)

DB-Gen calls `extract_atoms_stack(fg, contour, params)` to obtain the atom label
stack and `residual_contour`, then builds the union lattice and DB. As a
**byproduct at generation time**, DB-Gen writes `atoms.tif` and
`residual_contour.tif` next to `data.db` so the browser (spec ③) and manual
inspection can read finished atoms without recomputing. These are caches, not
the source of truth.

## Decisions flagged for review

- **D1 — persistence:** params-as-source-of-truth + recompute (recommended),
  vs. the Atom Extraction widget itself writing `atoms.tif`/`residual_contour.tif`
  on demand. Recommendation keeps one source of truth; the alternative makes the
  artifacts available earlier but risks staleness.
- **D2 — `fg_cutoff` default:** 0.01 (verbally locked) vs. 0.002 (used in the
  50-frame full run, broader coverage). Either is a starting point for visual
  tuning.
- **D3 — diagnostic overlays:** include `territory`/`residual_contour` toggle
  layers in the preview, or keep the preview atoms-only for simplicity.

## Testing

- Unit: `residual` window-odd-forcing and zero-background behavior;
  `extract_atoms_frame` produces no sub-`atom_min_area` atoms and leaves no holes
  in territory; `extract_atoms_stack` shape/determinism.
- Widget: activate/deactivate adds/removes exactly the preview layers; a slider
  change triggers one debounced recompute; frame change re-renders.
- Regression: `extract_atoms_stack` on the 10-frame curated dataset matches the
  prototype's atom counts within tolerance.
