# Swap Candidate Shortcut (Z / C) — Design

Date: 2026-05-15

## Summary

Add two new shortcuts to the nucleus correction mode that mutate the currently
selected cell in place by swapping its mask with a neighbouring hypothesis
fragment from the Ultrack `data.db` `NodeDB`. Pressing `Z` steps to a strictly
smaller fragment, `C` to a strictly larger one. Candidates are restricted to
those whose centroid lies within a user-controlled radius of the originally
selected cell's centroid and are ranked by absolute area.

The feature mirrors the existing extend (`A` / `D`) shortcut workflow in
location and lifecycle, but operates **at the current frame** rather than
stepping across frames.

## Motivation

Manual correction frequently runs into oversegmented or undersegmented cells.
The Ultrack hypothesis tree often already contains a better-sized fragment for
the same physical cell — choosing it is faster than redrawing. The existing
correction tools (merge/split/erase/draw) are pixel-level; this feature lets
the user cycle through the segmenter's own alternatives by size.

## Scope

In scope:

- Nucleus workflow widget only (`2_nucleus`). The cell workflow (`3_cell`) uses
  a different propagator and is out of scope.
- Single-cell, single-frame swap. The mutation replaces only the source cell's
  pixels with the chosen candidate's mask.
- Reuse of existing protection logic (validated tracks + anchors) and history
  recording.

Out of scope:

- Multi-cell swap.
- Cross-frame propagation of the swap.
- Updating Ultrack solver / NodeDB state. The op mutates the labels layer only,
  same as merge/erase/split.

## Open assumptions

- The hypothesis NodeDB at `data.db` contains fragments at the current frame
  with `t == frame`. Verified by `extend.py:337` and `db_query.py:120` usage.
- `read_corrections` and `read_validated_tracks` are already available in
  `nucleus_workflow_widget.py` (used by `_on_extend`).

## Architecture

```
NucleusWorkflowWidget
├── UI: new "Swap radius" QDoubleSpinBox in Extend/Retrack params section
│        (next to extend_max_dist_spin, default 40.0 px, range 0..500)
├── State:  self._swap_cursor: _SwapCursor | None = None
├── Wires:  CorrectionWidget.set_selection_callback to clear cursor on
│           selection change (composed with any existing callback)
├── Wires:  viewer.dims.events.current_step → clear cursor on frame change
├── Shortcuts: Z → self._on_swap_step(direction="smaller")
│              C → self._on_swap_step(direction="larger")
│   Added to _install_correction_shortcuts; enabled/disabled by
│   _on_correction_mode_toggled (same lifecycle as A/D/Q/E/B/S).
└── Slots:  _on_swap_step builds or advances cursor, applies the swap

tracking_ultrack/swap_candidate.py   ── NEW module
├── @dataclass SwapCandidate:
│       node_id: int
│       mask_2d: np.ndarray    # bool, full-frame (Y, X)
│       bbox: tuple[int, int, int, int]
│       centroid: tuple[float, float]
│       area: int
├── @dataclass _SwapCursor:
│       source_id: int
│       frame: int
│       source_centroid: tuple[float, float]
│       source_area: int
│       candidates: tuple[SwapCandidate, ...]   # sorted by area ASC
│       displayed_area: int   # area of mask currently in the layer
│       cursor: int | None    # index in candidates if a swap has happened
│                             # (None right after init: source is shown,
│                             # not yet any candidate)
└── list_swap_candidates(*, db_path, frame, source_centroid,
                          radius_px, frame_shape) -> list[SwapCandidate]
```

`_SwapCursor` lives in `swap_candidate.py` (data) and is owned by the widget.
The mutation step lives on the widget (it needs `validated_tracks`, anchors,
and the layer reference).

## Data flow

### First Z or C press with no cursor

1. Guard: `correction_widget._selected_label` must be non-zero and present at
   the current frame `t`; `_pos_dir`, `data.db` must exist.
2. Compute source mask `tracked[t] == source_id`. Reject if empty.
3. Compute source centroid and area via `regionprops`.
4. Refuse if the source cell is a validated track (whole-track validation in
   nucleus mode): status `Cannot swap a validated cell`.
5. Call `list_swap_candidates(db_path, frame=t,
   source_centroid=source_centroid, radius_px=swap_radius_spin.value(),
   frame_shape=tracked.shape[1:])`. Returns area-ASC list, excluding nodes
   whose paintable area (after subtracting protected pixels) is zero.
6. If list is empty: status `No swap candidates within {r}px`. Do not store
   cursor.
7. Initialise the cursor record: `displayed_area = source_area`,
   `cursor = None` (no candidate has been shown yet).
8. Apply the directional step (see below). Persist `_swap_cursor` only if a
   swap actually occurs.

### Directional step (z / c)

Both first-press and subsequent presses use the same step:

- **z (smaller)**: find the largest `i` with
  `candidates[i].area < displayed_area`. If none exists, status
  `No smaller candidate`; leave cursor untouched; no swap. Else apply the
  swap to `candidates[i]`, set `cursor = i` and
  `displayed_area = candidates[i].area`.
- **c (larger)**: find the smallest `i` with
  `candidates[i].area > displayed_area`. If none, status
  `No larger candidate`. Else swap, set `cursor = i`,
  `displayed_area = candidates[i].area`.

This is equivalent to walking the sorted area list ±1 with the source's
original area acting as a virtual insertion point on the first press. Ties
on area (candidates with identical area) are still reachable: stepping by
strict inequality moves past the tie group rather than cycling inside it,
which is the intended behaviour ("z must produce something strictly
smaller").

### Subsequent Z/C with cursor present

Same step rule. The cached `candidates` list is not rebuilt.

### Cache invalidation

`_swap_cursor` is set to `None` on any of:

- Selection change (`set_selection_callback` fires for any new selection,
  including re-selecting the same cell — this is intentional so the radius
  window and ranking are recomputed against the cell's now-current mask).
- Frame change (`viewer.dims.events.current_step`).
- Correction mode toggled off.
- Tracked layer reloaded.

## Swap application

```python
def _apply_swap(self, layer, t, source_id, candidate, validated_tracks):
    frame = layer.data[t]
    before = frame.copy()

    protected_ids = set()
    for cell_id, frames in validated_tracks.items():
        if t in frames and cell_id != source_id:
            protected_ids.add(cell_id)
    for c in read_corrections(self._pos_dir):
        if c.kind == "anchor" and int(c.t) == t and int(c.cell_id) != source_id:
            protected_ids.add(int(c.cell_id))
    protected_mask = (
        np.isin(frame, list(protected_ids))
        if protected_ids
        else np.zeros_like(frame, dtype=bool)
    )

    frame[frame == source_id] = 0
    paintable = candidate.mask_2d & ~protected_mask
    frame[paintable] = source_id

    self.correction_widget._record_history(layer, t, before)
    layer.refresh()
```

Status message on success:
`Swapped cell {source_id} → candidate {cursor+1}/{len(list)} (area={a} px)`.

## Edge cases

- **No candidates in radius** — status `No swap candidates within {r}px`.
  Cursor not stored.
- **Candidate footprint entirely blocked by protected cells** — filtered out
  during `list_swap_candidates` so the cached list never contains
  zero-paintable entries.
- **Source cell validated** — guard at step 4 above.
- **`data.db` missing** — status `data.db not found — run DB Generation first`
  (same wording as extend).
- **Cursor at edge** — status `No smaller candidate` / `No larger candidate`.
  Cursor is *not* advanced past the edge; pressing Z again at the bottom does
  nothing.

## UI

### Spinbox

In `nucleus_workflow_widget.py:739-754` (the Extend group of the Extend/Retrack
params section), after the existing overlap-penalty spinbox:

```python
self.swap_radius_spin = _dspin(0, 500, 40.0, 1.0, 1)
add_block_row(g, 3, "Swap radius:", compact_spinbox(self.swap_radius_spin))
```

### Shortcut legend

The correction widget's shortcut legend (`correction_widget.py:262-295`) is
**not** modified — that legend covers correction-widget-owned shortcuts only.
A/D/Q/E/B/S aren't listed there either. Z/C are surfaced instead via a
`muted_label` hint placed near the Swap radius spinbox:

> *Z / C — swap selection with smaller / larger hypothesis fragment.*

## Files touched

| File | Change |
| --- | --- |
| `src/cellflow/tracking_ultrack/swap_candidate.py` | new — `SwapCandidate`, `_SwapCursor`, `list_swap_candidates` |
| `src/cellflow/napari/nucleus_workflow_widget.py` | new spinbox, hint label, Z/C in shortcuts list, `_on_swap_step`, `_apply_swap`, cursor invalidation wiring |
| `tests/tracking_ultrack/test_swap_candidate.py` | new — unit tests for the listing function |

## Testing

Pure-function tests in `tests/tracking_ultrack/test_swap_candidate.py`:

- `test_list_swap_candidates_filters_by_radius` — synthetic NodeDB with three
  nodes at known centroids; assert only those within radius are returned.
- `test_list_swap_candidates_sorted_by_area_asc` — assert area-monotonic
  ordering.
- `test_list_swap_candidates_empty_when_db_missing` — nonexistent path returns
  `[]`, no exception.
- `test_directional_step_z_finds_strictly_smaller` — given a sorted area
  list and a `displayed_area` between two entries, z resolves to the largest
  entry strictly less than `displayed_area`.
- `test_directional_step_c_finds_strictly_larger` — symmetric for c.
- `test_directional_step_bounds` — z at the bottom (or c at the top) returns
  no move.

Widget-level behaviour (selection-change invalidation, anchor protection,
history) is verified manually in the napari session, the same way A/D/extend
was.

## Non-goals

- Reusing the extend score function. Ranking here is pure area-ASC; the extend
  score combines distance, IoU, area, and overlap penalty and is not applicable.
- Persisting cursor state to disk. The cursor lives for as long as the cell
  stays selected on its frame.
