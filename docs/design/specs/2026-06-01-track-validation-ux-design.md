# Easier validation of good tracks — design

Date: 2026-06-01
Status: design approved (pending written-spec review) → next step `writing-plans`.

## Problem

In the nucleus correction widget, validating *good* tracks is slower than it should
be. The act of validating is already one keystroke (left-click a cell, press **V**,
which locks that track's geometry in every frame it appears). The friction is around
it. User-confirmed bottlenecks:

1. **Spotting good tracks** — no way to tell at a glance which tracks are clean
   end-to-end versus which have a break or identity swap somewhere.
2. **Verifying a track across frames** — once a track looks promising, confirming it
   over the whole movie is tedious frame-by-frame scrubbing.
3. **Trusting auto-suggestions** — the user would validate faster if the tool pointed
   at likely-clean tracks.

Explicitly *out of scope*: bulk-blessing many tracks in one action.

## Approach

Two features that share one foundation — a **per-track quality score derived from the
Ultrack solve**:

- **Feature 1 — Quality-ordered track IDs (auto after every solve).** Relabel so
  ID 1 is the highest-quality track, 2 the next, and so on. This collapses "spotting"
  and "auto-suggestions" into one motion: start at cell 1 and cycle downward
  (`Shift+Left/Right` already exists) until tracks stop looking clean.
- **Feature 2 — Whole-track temporal overlay (behind a checkbox).** Paint the selected
  track's mask for every frame onto the current canvas, colored start→finish, with a
  frame number in each mask, so a whole track's life — including any swap or jump — is
  legible in a single view. This is the "verifying across frames" win.

## Shared core: `src/cellflow/tracking_ultrack/track_quality.py` (new)

A pure module (no Qt, no napari) that computes a quality score per exported track from
the solved Ultrack database.

**Score definition (DB weights only — no geometric fallback):**

> `quality(track) = Σ NodeDB.node_prob over the track's nodes
>                 + Σ LinkDB.weight over the track's selected edges`

Higher is better.

**Mapping (resolved against the existing export path).** The exporter
`tracking_ultrack/export.py` builds the tracked stack via
`ultrack.core.export.to_tracks_layer(cfg)` → `tracks_df`, then `tracks_to_zarr`. The
painted label value equals the row's **`track_id`**. `tracks_df` columns:
`id` (= NodeDB node id), `t`, `track_id`, `parent_track_id`. This is the same dataframe
already consumed in `corrections.py` (`tracks_by_node = tracks_df.set_index("id")`).

Therefore, per exported track:
- **Nodes:** all `tracks_df.id` where `track_id == this`. Sum their `NodeDB.node_prob`
  (treat NULL as the existing convention, 1.0, matching `db_query.py`).
- **Edges:** consecutive node pairs along the track (ordered by `t`). For each pair,
  sum the corresponding `LinkDB.weight` (NULL → 1.0, matching `db_query.py`).

**Exclusions (confirm exact rule at implementation time, default as stated):**
- Skip links annotated FAKE (`annotation_name(LinkDB.annotation) == "FAKE"`) — count 0.
- Skip anchor-injected / synthetic nodes if they are distinguishable via
  `node_annot`; otherwise include them (they are part of the chosen solution).

**Public surface (proposed):**
```python
def track_quality_scores(db_path: Path, cfg: TrackingConfig) -> dict[int, float]:
    """Map exported track_id -> quality score (Σ node_prob + Σ edge weight)."""

def quality_order(scores: dict[int, float]) -> list[int]:
    """track_ids sorted by score desc, then by id asc as a stable tiebreak."""
```

These are read-only over the DB and deterministic, so they are unit-testable against a
small fixture DB without napari.

## Feature 1 — Quality-ordered track IDs (auto after solve)

After each solve, before/at export time, relabel the tracked stack so IDs follow
quality order: best track → ID 1, next → 2, etc.

**Mechanism.** Reuse the existing relabeling machinery:
- `reassign_ids_stack()` in `_correction_utils.py` already builds an `old_to_new` LUT
  and applies it to a stack. Generalize it (or add a sibling) to accept an explicit
  ordering of old IDs instead of numeric-sorted order, so we can pass
  `quality_order(scores)`.
- After relabel, call `remap_validated_tracks(pos_dir, old_to_new)` so any existing
  validations/anchors follow their cells (same call the manual reassign path uses).

**Trigger.** Runs automatically as part of the solve→export step (the pipeline widget
path that produces `tracked_labels.tif`). Not a manual button.

**Consequence (accepted).** Track IDs are not stable across re-solves — a re-solve
re-ranks. The user chose auto ordering with this understood.

**Open implementation question (non-blocking):** whether the manual `#` reassign-IDs
button in the correction toolbar (currently numeric compaction) should be repurposed
to quality-order on demand, or left numeric. Default: leave it numeric for now; the
auto-at-solve path is the feature.

## Feature 2 — Whole-track temporal overlay ("comet"), behind a checkbox

A new **"Show track path"** checkbox in correction mode. When ON, for the **currently
selected** track only (one track at a time):

- **Masks for every frame** of the track are painted onto the current canvas, **filled**
  (not outline), colored by a **viridis start→finish** colormap (frame 0 → dark, last
  frame → yellow). Later frames are drawn **on top** where masks overlap (handles the
  "cell barely moves, masks stack" case simply).
- A **frame number** is centered in each mask, reusing the centroid/text-layer approach
  already in `_correction_centroids.py` (it already places per-cell text at centroids).
- The **spotlight is enlarged to the union of all the track's masks** while the checkbox
  is on, so everything outside the *whole trajectory* is dimmed — not just the
  current-frame blob. This replaces the per-frame `_scaled_mask` spotlight behavior
  (`correction_widget.py: _update_spotlight`) for the duration. Implementation: build
  the boolean union of the track's per-frame masks and feed that as the spotlight mask
  (the existing alpha-ramp logic can ramp around the union region).
- Rendered into **dedicated owned layers** (e.g. `[Correction] Track Path` for the
  filled masks + a text/points layer for numbers), registered in
  `_correction_owned_layers` so existing teardown removes them. Rebuilt when the
  **selected track changes**; torn down when the checkbox turns **off**, selection
  clears, or correction mode deactivates.

**Decided design points:**
- Overlap handling: filled fills, newest-frame-on-top. (No outline mode, no skip-Nth.)
- Reveal: a **checkbox** (not always-on, not hold-to-reveal).
- No per-frame flagging of suspicious transitions in this iteration.

**Interaction with existing overlays.** The validated/anchor overlays use
`place_below_spotlight()`; the new track-path layer should slot into the same ordering
discipline so the spotlight dim and the comet compose correctly. Verify z-order at
implementation time.

## Components & boundaries

- `tracking_ultrack/track_quality.py` (new) — pure scoring + ordering. Depends only on
  the DB and `TrackingConfig`. Independently testable.
- `_correction_utils.py` — extend `reassign_ids_stack` (or add `reassign_ids_ordered`)
  to take an explicit old-ID ordering. Pure, testable.
- Solve/export path (`solve.py` / `export.py` / pipeline widget) — call the scorer and
  ordered relabel after solve, then `remap_validated_tracks`. Thin wiring.
- `nucleus_correction_widget.py` — add the "Show track path" checkbox, its signal, and
  build/teardown of the track-path + numbers layers; swap spotlight to union mode while
  active. UI + orchestration only.
- Track-path rendering helper (likely a new `_correction_track_path.py` to keep the
  1600-line widget from growing) — pure-ish function: given the tracked stack + a
  track_id, return (per-frame mask list, viridis colors, centroid+frame-number list).
  Keeps the widget thin and the rendering unit-testable.

## Testing

- `track_quality.py`: unit tests over a tiny fixture DB (a few nodes/links across 2-3
  frames, known node_prob/weight) asserting exact summed scores and ordering, including
  NULL-weight and FAKE-link handling.
- Ordered relabel: unit test that a known score map produces the expected `old_to_new`
  LUT and that `remap_validated_tracks` is called with it.
- Track-path rendering helper: unit test that, for a synthetic 3-frame track, it returns
  one mask per occupied frame, monotonic viridis colors start→finish, and correct
  centroid/frame-number entries; and that a barely-moving track stacks newest-on-top.
- Widget: existing test file `tests/napari/test_nucleus_db_browser_widget.py` is the
  pattern; add coverage that toggling the checkbox creates/removes owned layers and
  switches the spotlight to union mode, mirroring existing correction-widget tests.

## Risks / notes

- Score depends on the FAKE/anchor exclusion rule being right; get it explicit before
  finalizing (it changes rankings near corrected regions).
- Auto-relabel at solve time touches the canonical `tracked_labels.tif` numbering;
  ensure it runs *before* anything downstream caches IDs, and that
  `remap_validated_tracks` keeps prior work attached.
- One-track-at-a-time keeps the comet legible; do not generalize to all-tracks (would
  be visual soup, as explored in the visual-companion mockups).
```
```
