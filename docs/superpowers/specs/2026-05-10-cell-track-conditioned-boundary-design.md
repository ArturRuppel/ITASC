# Cell Track-Conditioned Boundary Selection

## Goal

Replace the current cell flow-following label assignment with a track-conditioned
cell boundary selection pipeline. The nucleus workflow already provides the cell
identities and their temporal tracks. The cell workflow should therefore not
discover tracks; it should choose one coherent cell boundary for each known
nucleus track at each frame.

The design reuses the useful part of the Ultrack workflow: generating a rich
per-frame candidate hierarchy from contour and foreground maps, storing masks,
scores, and overlaps in a database, and solving over that candidate graph.
The solver is cell-specific and conditioned on existing nucleus tracks.

## Problem Statement

The current flow-following cell labels are assigned pixel-by-pixel. With
`flow_weight=1.0`, this respects elongated and protrusive cell shape better than
proximity-to-nucleus gravity, but it has no object-level or temporal coherence
constraint. The same label can appear in disconnected regions, flicker between
frames, or form discontinuous pieces.

Sparse cell foreground and occasional foreground holes are acceptable. The
unacceptable artifacts are disconnected same-label components, frame-to-frame
boundary jumps, and conflicts where two cell identities compete for overlapping
regions.

## Inputs And Outputs

Inputs:

- `1_cellpose/cell_prob_3dt.tif`
- `1_cellpose/cell_dp_3dt.tif`
- `3_cell/foreground_masks.tif`
- `2_nucleus/tracked_labels.tif`
- optionally `0_input/cell_zavg.tif` for candidate scoring and visual QA

New outputs:

- `3_cell/contour_maps.tif`
- `3_cell/ultrack_workdir/data.db` or a compatible sibling database
- `3_cell/tracked_labels.tif`
- optional diagnostics in `3_cell/diagnostics/`

The final `3_cell/tracked_labels.tif` uses the existing nucleus track IDs. It
does not introduce arbitrary cell IDs.

## Architecture

### Candidate Generation

Generate cell contour maps and use the existing cell foreground masks as the
support. The contour maps should describe likely membrane/boundary locations, not
nucleus boundaries. Initial contour candidates can come from Cellpose masks,
Cellpose flow convergence boundaries, probability gradients, or a blend of these
signals.

Run `ultrack.segment()` or equivalent hierarchy construction on:

- foreground: `3_cell/foreground_masks.tif`
- contours: `3_cell/contour_maps.tif`

The result is a hierarchy of connected candidate masks per frame. This is the
main replacement for independent per-pixel assignment.

### Candidate Database

Reuse the database-building strategy from `cellflow.tracking_ultrack.db_build`.
The database stores candidate nodes, masks, per-frame overlaps, and scores. For
cell boundary selection, each node must also be evaluated against known nucleus
tracks in the same frame.

For every candidate node and frame, compute:

- area and bbox;
- foreground support;
- contour/boundary quality;
- Cellpose probability quality;
- overlap with each tracked nucleus label;
- whether the node is eligible for a given nucleus track.

Eligibility is track-conditioned. A candidate is valid for track `k` at frame
`t` only if it passes the hard-anchor rule against nucleus label `k` at `t`.

### Hard Anchor Rule

Hard anchoring is required. A selected cell boundary for track `k` at frame `t`
must contain or overlap the tracked nucleus label `k` at the same frame.

The first implementation should make the threshold explicit and configurable:

- default validity: at least one nucleus pixel from label `k` lies inside the
  candidate;
- optional stricter threshold: minimum fraction of nucleus pixels inside the
  candidate.

Candidates with no overlap to the requested nucleus track are invalid for that
track. Candidates that overlap multiple nucleus labels are not automatically
invalid, because elongated or crowded cells may touch nearby nuclei; they receive
a conflict penalty and can later be forbidden if diagnostics show this is safer.

## Optimization

This is not standard cell tracking. The known tracks define the identities. The
solver chooses boundaries.

### Phase 1: Per-Track Dynamic Programming Prototype

Prototype a per-track dynamic programming solver first. It is easier to debug and
will quickly reveal whether the candidate hierarchy contains good cell masks.

For each nucleus track independently:

- state: a candidate node eligible for that track at time `t`, plus an optional
  missing state;
- unary score: segmentation quality, anchor quality, size plausibility, foreground
  support, and contour quality;
- transition score: area smoothness, centroid displacement, shape overlap, and
  boundary change relative to the previous selected candidate;
- output: one selected candidate per track-frame when possible.

This phase does not fully solve overlap conflicts between tracks. It should emit
diagnostics for conflicting selected masks so the global solver can be calibrated.

### Phase 2: Global ILP

After candidate quality is validated, add a global ILP over all known tracks.

Variables:

- `x[k, t, n] = 1` when track `k` selects node `n` at frame `t`;
- optional missing variable for track `k` at frame `t`.

Constraints:

- each existing nucleus track-frame selects exactly one candidate or the missing
  state;
- candidate `n` can only be selected by track `k` if it satisfies the hard-anchor
  rule for that track;
- two tracks cannot select candidates whose overlap exceeds a configurable
  threshold;
- optional: a candidate can be selected by at most one track.

Objective:

- maximize segmentation and anchor quality;
- maximize temporal coherence for each track;
- penalize missing states;
- penalize or forbid large overlaps between selected candidates;
- penalize implausible area jumps and centroid jumps.

The ILP is the production target, but the DP prototype should come first because
candidate quality is the highest uncertainty.

## UI And Workflow

Extend the cell workflow with a track-conditioned boundary section:

1. Create or preview `3_cell/contour_maps.tif`.
2. Build cell candidate database from cell contour maps and foreground masks.
3. Run track-conditioned boundary selection.
4. Load/export `3_cell/tracked_labels.tif`.

The section should show the existing nucleus labels as required inputs and make
clear that the result keeps nucleus track IDs.

Diagnostics should include:

- selected candidate overlay per track;
- candidate alternatives for a selected track-frame;
- disconnected component count per exported label;
- frame-to-frame area and overlap change;
- overlap conflicts between selected cells;
- missing selected boundary frames.

## Error Handling

Fail early with clear messages when required inputs are missing or shape-mismatched.

Validation rules:

- cell foreground and cell contour maps must have matching `(T, Y, X)` shape;
- nucleus tracked labels must match the same `(T, Y, X)` shape;
- each nucleus track-frame with a nonzero nucleus label is expected to have at
  least one eligible candidate, otherwise the solver uses the missing state and
  reports it.

Database generation should overwrite only the cell candidate workdir, not the
nucleus workdir.

## Testing

Backend tests:

- candidate eligibility from nucleus overlap;
- candidates with no anchor are invalid for that track;
- per-track DP chooses a temporally smoother path over a framewise higher-scoring
  but jumpy path;
- missing state is selected only when no valid candidate exists or all candidates
  are sufficiently poor;
- overlap diagnostics detect conflicting selected candidates;
- export preserves nucleus track IDs.

Integration tests:

- build a tiny synthetic candidate database with two known tracks and verify the
  selected labels are connected and ID-preserving;
- verify shape validation and missing-file messages in the cell workflow.

Manual validation on `pos03`:

- compare disconnected same-label components before and after;
- compare frame-to-frame area jumps;
- inspect several elongated/protrusive cells where proximity-based gravity fails;
- inspect crowded frames for overlap conflict behavior.

## Rollout

1. Add backend utilities for cell contour maps and candidate anchor scoring.
2. Build the cell candidate DB path under `3_cell/ultrack_workdir`.
3. Implement and test the per-track DP prototype.
4. Add diagnostics and manual QA on `pos03`.
5. Add the global ILP once candidate quality and scoring are calibrated.

The existing flow-following workflow should remain available during rollout as a
baseline and fallback until the track-conditioned boundary selector is validated.
