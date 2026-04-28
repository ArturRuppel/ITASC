---
status: draft
date: 2026-04-28
predecessor: PLAN_validation_and_constrained_resolve.md
---

# Plan: Per-Cell Track Extension via Hypothesis Search

After the ILP solve (and any re-solve), the user is in manual-correction
mode. When they spot a track that ends prematurely or is missing a frame,
they should be able to select a single cell and extend its track one
frame forward or backward by pulling the best-matching hypothesis from
`hypotheses.h5` and painting it into the tracked label image with the
same cell ID.

This is intentionally a *per-cell, per-frame, manual* tool — not a
batch propagator. One click extends by one frame.

## Design

1. **Source of candidates: `hypotheses.h5` directly.** The NodeDB
   contains the same masks but requires a DB session and pickle
   round-trip. We don't need overlap/hierarchy info (we're picking one
   candidate, not solving a constraint problem), so reading the h5 is
   simpler and self-contained. All `p` partitions at the target frame
   contribute candidates; each non-zero label = one candidate mask.
   (Hypotheses are stored as `(1, Y, X)` per `(t, p)`; we use the 2D
   slice, matching `ingest.py`.)

2. **Score: distance gate, then area similarity.** Pure IoU rewards
   small fragments that lie entirely inside a displaced source cell's
   footprint over the actual full cell. Centroid distance + area
   similarity sidesteps that:
   - **Gate:** `||centroid_source − centroid_candidate||₂ ≤ d_max`
     (default `40` px; hardcoded for v1, expose later if needed).
   - **Rank:** among gated candidates, pick max
     `min(A_source, A_candidate) / max(A_source, A_candidate)`.
   - **No candidate passes the gate →** don't paint, status reports.

3. **Paint policy (label-image only, no DB writes).**
   1. Wipe any existing instance of `source_id` in the target frame:
      `tracked[t', tracked[t'] == source_id] = 0`.
   2. For every pixel inside the chosen candidate's bbox: if
      `tracked[t', y, x] == 0`, write `source_id`; else leave the
      occupied pixel alone (collisions resolved in favour of whatever
      ID is already there).

4. **No automatic validation.** Validation in this project is
   whole-track only and only after the track is complete. An extension
   is by definition partial work; if the user later runs re-solve
   without first validating the completed track, the painted frames
   are wiped — that's expected and the user's call to make.

5. **Direction.** Forward (`t+1`) and backward (`t-1`) — symmetric.
   No multi-frame stepping in v1.

## Module: `src/cellflow/tracking_ultrack/extend.py`

Pure logic, no Qt. One public function plus a result dataclass.

```python
@dataclass
class ExtendResult:
    target_frame: int
    candidate_label: int        # the (p, label_id) of the chosen hypothesis,
    candidate_partition: int    # mostly for debugging / status text
    mask_2d: np.ndarray         # bool, full-frame (Y, X)
    bbox: tuple[int, int, int, int]  # (y0, x0, y1, x1) of the candidate
    centroid_distance: float
    area_ratio: float           # ∈ [0, 1]; 1.0 = identical area

def extend_track(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,        # (T, Y, X) uint32
    hypotheses_path: Path,
    d_max: float = 40.0,
) -> ExtendResult | None:
    ...
```

**Returns `None` when:**
- Target frame is out of range (`source_frame == 0` going backward,
  or `== T-1` going forward).
- Source cell has no pixels at `source_frame` in `tracked_labels`.
- No hypothesis at the target frame passes the distance gate.

The widget translates each `None` case into a distinct status string.

**Scoring loop:**
- Compute source centroid + area from `tracked_labels[source_frame] ==
  source_id` once.
- For each `p` in `list_hypotheses(...)`: read the labelmap, iterate
  its non-zero labels (e.g. via `regionprops` for centroid + area in
  one pass), apply the gate, track the best area-ratio.
- Tie-break (extremely unlikely with float area): smallest centroid
  distance.

## Widget: `src/cellflow/napari/nucleus_workflow_widget.py`

A new row below the existing Retrack row (see `_setup_ui` around the
`retrack_row` block). The Retrack row already holds three buttons;
extension is a conceptually different operation, so it gets its own
row. We will redesign the correction UI later — keeping these logically
separated now makes that easier.

**New widgets:**
- `self.extend_back_btn = QPushButton("◀ Extend")`
- `self.extend_fwd_btn  = QPushButton("Extend ▶")`

**Handlers:** `_on_extend_backward`, `_on_extend_forward`.

Each handler:
1. Pull selected cell ID from the labels layer
   (`viewer.layers[_TRACKED_LAYER].selected_label`).
2. Pull current `t` from `viewer.dims.current_step[0]`.
3. Resolve `hypotheses_path` via `self._hyp_path()`.
4. Call `extend_track(...)`.
5. On `ExtendResult`: apply paint policy in-place to the labels layer's
   underlying `data` array, then call `layer.refresh()`.
6. Set status via `self._set_status(...)` with one of:
   - `"Extended cell 42 → t=8 (dist=12.3px, area=0.94)"`
   - `"Cell 42 not present at t=7"`
   - `"No hypothesis within 40px at t=8"`
   - `"Already at last frame"` / `"Already at first frame"`

No new keybindings in v1 — buttons only. Easy to add later
(e.g. `Shift+→` / `Shift+←`).

## Tests

`tests/tracking_ultrack/test_extend.py`:
- Synthetic 2-frame tracked array + a tiny synthetic `hypotheses.h5`
  with a few labels at `t=1` (use `write_hypothesis_record` from the
  existing API to build the file).
- Cases:
  - Forward, single perfect-match hypothesis → returned.
  - Forward, fragment + displaced full cell → full cell wins
    (validates the size+distance choice).
  - Forward, no candidate inside `d_max` → returns `None`.
  - Backward at `t=0` → returns `None`.
  - Source cell missing at source frame → returns `None`.
  - Paint policy applied to a target frame that already contains
    `source_id` elsewhere (must be wiped) and another cell's pixels in
    the bbox region (must be preserved).

No widget tests; the widget glue is thin and exercised manually.

## Out of scope for v1

- Centroid-corrected IoU or shape descriptors (Hu moments, etc.). Add
  only if size+distance proves to misrank in practice.
- Multi-frame extension ("extend N frames" or "extend until score
  drops"). User clicks repeatedly for now.
- Tunable `d_max` from the widget. Hardcoded; revisit after dogfooding.
- Undo via napari history. The action mutates the array directly and
  is not undoable through Ctrl+Z; user re-paints or uses re-solve to
  recover. Acceptable for v1.
- Any DB writes. Extension lives entirely in the label image.
