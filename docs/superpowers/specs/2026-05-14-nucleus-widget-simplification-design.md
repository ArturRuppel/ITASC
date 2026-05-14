# Nucleus Widget Simplification Design

## Context

The current nucleus tracking widget exposes implementation details as workflow
steps: contour-map generation, database generation, Ultrack solving, database
inspection, and correction controls all compete for attention. Several controls
also hide important data transformations. In particular, contour maps and
foreground scores are produced by averaging labels across Cellpose probability
thresholds and z-slices, then the database builder applies another threshold
sweep before Ultrack sees the data.

The redesign should make those transformations visible and inspectable, remove
unused controls, and separate normal workflow actions from explicit interactive
modes.

## Goals

- Make segmentation input creation the default expanded, interactive section.
- Make Tracking / Ultrack collapsed by default, because its parameters are
  occasional tuning controls.
- Treat Database Browser and Correction as explicit toggle modes that own their
  napari layers while active and clean them up when deactivated.
- Remove gamma averaging controls and behavior from the contour workflow.
- Remove the Cellpose flow-threshold control from the contour workflow.
- Remove the save-label-images checkbox and source-label output behavior.
- Make the pre-database contour/foreground threshold sweep visible before DB
  building.
- Preserve a clear backend contract for the database builder: it can consume
  multi-source `P x T x Y x X` contour and foreground arrays.

## Non-Goals

- Do not refactor the Ultrack DB browser internals before the workflow shape is
  settled.
- Do not redesign the cell workflow.
- Do not change correction algorithms except where needed for mode activation,
  visibility, and parameter placement.
- Do not remove the DB browser; make it an optional inspection mode.

## Top-Level Layout

The widget should not have a duplicated global action bar. Actions live inside
the section that owns their parameters and outputs.

Default layout:

```text
Nucleus Tracking
────────────────────────────────────────
Artifacts
  nucleus_prob_3dt.tif          present/missing
  nucleus_dp_3dt.tif            present/missing
  contours.tif                  present/missing/stale
  foreground_scores.tif         present/missing/stale
  contour_sources.tif           present/missing/stale
  foreground_sources.tif        present/missing/stale
  data.db                       present/missing/stale
  tracked_labels.tif            present/missing/stale

Segmentation Inputs             expanded by default
Tracking / Ultrack              collapsed by default
Database Browser                inactive, header + Activate button
Correction                      inactive, header + Activate button
```

## Segmentation Inputs Section

This section is expanded by default. It is the main interactive setup area.

It has two logical groups:

```text
Segmentation Inputs
────────────────────────────────────────

A. Averaged map generation
  Cellprob threshold sweep        [ min ][ max ][ step ]
  Z slices                        [ all / range ][ step ]

  [ Preview averaged maps for current frame ]
  [ Build contours + foreground scores ]

  Outputs:
    contours.tif
    foreground_scores.tif

B. Pre-DB threshold sweep
  Contour score threshold sweep   [ min ][ max ][ step ]
  Foreground score threshold sweep[ min ][ max ][ step ]

  Sweep source                    [ slider: 1 / P ]
  View                            ( Averaged contours
                                    Averaged foreground
                                    Thresholded contours
                                    Thresholded foreground
                                    Overlay )

  [ Preview threshold sweep ]
  [ Build thresholded Ultrack inputs ]

  Outputs:
    contour_sources.tif           P x T x Y x X
    foreground_sources.tif        P x T x Y x X
```

Removed from this section:

- gamma min/max/step controls
- flow-threshold control
- save-label-images checkbox
- source-label image output

The section should be collapsible, but visible by default. Collapsing it should
not deactivate any global mode; it only hides the controls.

## Segmentation Data Flow

### Stage A: Cellpose Candidate Averaging

Inputs:

- `1_cellpose/nucleus_prob_3dt.tif`
- `1_cellpose/nucleus_dp_3dt.tif`

Sweeps:

- Cellpose probability threshold
- z-slice

For each frame:

1. Iterate through selected z-slices.
2. Iterate through selected Cellpose probability thresholds.
3. Generate labels for that z/threshold candidate.
4. Accumulate boundary votes from `find_boundaries(labels)`.
5. Accumulate foreground votes from `labels > 0`.
6. Divide both accumulators by the number of generated label candidates.

Outputs:

- `2_nucleus/contours.tif`: continuous averaged boundary score, shape
  `T x Y x X`.
- `2_nucleus/foreground_scores.tif`: continuous averaged occupancy score,
  shape `T x Y x X`.

If the existing artifact name `contour_maps.tif` must be kept for compatibility,
the implementation plan should either preserve that path as an alias or perform
a deliberate migration. The UI should display the simpler label `contours`.

### Stage B: Pre-DB Threshold Sweep

Inputs:

- `2_nucleus/contours.tif`
- `2_nucleus/foreground_scores.tif`

Sweeps:

- contour score threshold
- foreground score threshold

The sweep creates `P` candidate input pairs. `P` is the number of threshold
combinations.

For each source index `p`:

```python
contour_sources[p] = threshold(contours, contour_threshold[p])
foreground_sources[p] = threshold(foreground_scores, foreground_threshold[p])
```

Outputs:

- `2_nucleus/contour_sources.tif`: shape `P x T x Y x X`.
- `2_nucleus/foreground_sources.tif`: shape `P x T x Y x X`.

The pre-DB threshold sweep must be previewable before DB build. The user should
be able to move a source slider and inspect the thresholded contour and
foreground source that will be passed to Ultrack.

## Tracking / Ultrack Section

This section is collapsed by default. The collapsed header shows a compact
status summary:

```text
Tracking / Ultrack
  Inputs: P candidates ready / missing / stale
  DB: missing / built / stale
  Solve: not run / complete / stale
```

When expanded:

```text
Tracking / Ultrack
────────────────────────────────────────
Input contract:
  contour_sources.tif             P x T x Y x X
  foreground_sources.tif          P x T x Y x X

Main action:
  [ Build DB + Solve ]

Visible compact params:
  Use validated corrections       [ ]
  Time limit                      [ 300 ]
  Solution gap                    [ 0.001 ]

Advanced params                  collapsed inside section
  DB/candidate params
  linking params
  node scoring params
  seed prior params
  solver weights
```

Database build and solve should be coupled as one primary action for the normal
workflow. Advanced controls remain available but do not dominate the default
screen.

The backend database builder should accept multi-source contour and foreground
arrays with shape `P x T x Y x X`. Single-source data may be normalized to
`P=1` internally.

## Database Browser Mode

Database Browser is inactive by default.

Inactive state:

```text
Database Browser                  [ Activate ]
```

On activation:

- The section expands.
- DB browser controls become visible.
- DB preview, annotation, and selection layers are added or refreshed in napari.
- Mouse callbacks needed for DB selection are installed.

Active state:

```text
Database Browser                  [ Deactivate ]
  [ Refresh ]
  Source                          [ slider ]
  Hierarchy cut                   [ slider ]
  Show validated nodes            [x]
  Show fake nodes                 [ ]
  Connected focus                 [ ]
  Node prob transparency          [ ]
  Edge weight transparency        [ ]
```

On deactivation:

- The section collapses.
- DB browser layers are removed from napari.
- DB browser callbacks are uninstalled.
- DB-browser-only selection state is cleared or made inactive.

Layers owned by this mode:

- `Ultrack DB Preview`
- `Ultrack DB Annotations`
- `Ultrack DB Selection`

## Correction Mode

Correction is inactive by default and controlled by an `Activate Correction`
action button. The button is the only entry point for loading tracked labels
for manual correction; there is no separate `Load Labels` button and the
embedded correction widget does not expose its own activation button.

Inactive state:

```text
Correction                        [ Activate Correction ]
```

On activation:

- Capture the current viewer state for every existing layer: visibility,
  active layer, and selected layers where feasible.
- Hide all existing non-correction layers.
- Remove any stale correction layers still registered from a previous
  activation.
- Load a fresh correction-owned layer set from disk every time.
- Add every correction-owned layer with `[Correction]` in its display name.
- Register each created layer internally as correction-owned.
- Activate the correction widget against the newly loaded correction labels
  layer.
- Expand the correction controls only after activation succeeds.

Required loaded layers:

- `[Correction] Tracked: Nucleus` from `2_nucleus/tracked_labels.tif`.
- `[Correction] Cell z-avg` from `0_input/cell_zavg.tif`, when present.
- `[Correction] Nucleus z-avg` from `0_input/nucleus_zavg.tif`, when present.
- `[Correction] NLS z-avg` from `0_input/NLS_zavg.tif`, when present.

All z-avg image layers use additive blending. Their contrast limits are derived
from image percentiles, with the 0.05 percentile mapped to the low contrast
limit and the 99.5 percentile mapped to the high contrast limit. If those limits
are not usable because the image is constant or invalid, fall back to data
min/max or napari's default contrast handling. `[Correction] NLS z-avg` uses
the `bop_blue` colormap.

Layer cleanup is driven by the internal registry, not by prefix matching. The
`[Correction]` tag is for user clarity only.

Active state:

```text
Correction                        [ Deactivate Correction ]
  [ Save tracked ]
  [ Extend selected ] [ Retrack selected ]
  [ Reassign ID ] [ Remove unvalidated ]

  Extend before/after             [ before ][ after ]
  Retrack radius/window           [ radius ][ window ]
  Greedy overwrite                [ ]

  CorrectionWidget

  Advanced Correction Params      collapsed
  Shortcuts                       collapsed
```

On deactivation:

- Correction tools and shortcuts are deactivated.
- Only internally registered correction-owned layers are removed.
- The internal correction layer registry is cleared.
- Pre-existing layers that still exist are restored to their previous visibility
  state.
- The previous active and selected layers are restored where feasible.
- The section collapses.
- Persistent output artifacts are not deleted.

Correction parameters that are frequently tuned should be elevated in the main
active section. Less common extend/retrack details stay in an advanced
subsection.

## Artifact Status

The artifact table should distinguish continuous averaged maps from thresholded
Ultrack inputs:

- `contours.tif`: continuous averaged boundary score.
- `foreground_scores.tif`: continuous averaged occupancy score.
- `contour_sources.tif`: thresholded source stack for Ultrack.
- `foreground_sources.tif`: thresholded source stack for Ultrack.
- `data.db`: built from the source stacks.
- `tracked_labels.tif`: solve/export result.

Status should support at least:

- missing
- present
- stale relative to upstream inputs
- failed, when a worker records an error

## Testing Strategy

Backend tests:

- Verify Cellpose averaging sweeps probability thresholds and z-slices.
- Verify averaged contours are boundary-vote means.
- Verify foreground scores are occupancy-vote means.
- Verify gamma and flow-threshold parameters are no longer part of the nucleus
  contour averaging API.
- Verify pre-DB threshold sweep converts continuous `T x Y x X` maps into
  `P x T x Y x X` source stacks.
- Verify the DB builder accepts `P x T x Y x X` arrays and normalizes
  single-source input to `P=1`.

Widget tests:

- Assert Segmentation Inputs is expanded by default and Tracking / Ultrack is
  collapsed by default.
- Assert gamma controls, flow-threshold control, and save-label-images checkbox
  are absent.
- Assert the pre-DB threshold sweep controls and source slider exist.
- Assert Preview threshold sweep updates thresholded preview layers without
  building `data.db`.
- Assert Database Browser activation expands the section and creates owned
  layers.
- Assert Database Browser deactivation collapses the section and removes owned
  layers/callbacks.
- Assert the correction section exposes `Activate Correction` and no longer
  exposes a separate `Load Labels` button.
- Assert Correction activation hides pre-existing layers, loads fresh
  `[Correction]` layers from disk, activates the correction widget against
  `[Correction] Tracked: Nucleus`, and expands the section.
- Assert Correction activation loads `cell_zavg.tif`, `nucleus_zavg.tif`, and
  `NLS_zavg.tif` when present with additive blending and percentile-derived
  contrast limits.
- Assert `[Correction] NLS z-avg` uses the `bop_blue` colormap.
- Assert Correction activation registers the created layers internally.
- Assert Correction deactivation collapses the section, removes only internally
  registered correction-owned layers, clears the registry, deactivates
  correction-only tools/shortcuts, and restores the previous visibility state
  for pre-existing layers.

Compatibility tests:

- If `contour_maps.tif` remains as a compatibility alias, verify old state or
  old projects still resolve the contour artifact.
- Verify old saved widget state that contains removed gamma, flow-threshold, or
  save-label keys is ignored without failing.

## Implementation Order

1. Remove or simplify the obsolete contour controls and backend parameters.
2. Introduce explicit averaged-map and pre-DB threshold-sweep artifacts.
3. Update the database builder contract to consume `P x T x Y x X` source
   arrays.
4. Redesign the widget layout around section ownership and mode toggles.
5. Only after the workflow shape is stable, revisit the Ultrack DB browser
   extraction plan.

This order avoids refactoring around UI and backend boundaries that are about
to change.
