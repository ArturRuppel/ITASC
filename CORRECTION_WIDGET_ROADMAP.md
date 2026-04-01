# Correction Widget Roadmap

This document tracks planned improvements to CellFlow's correction widget.
Reference implementation: [Epicure](https://github.com/Image-Analysis-Hub/Epicure)
(cite: https://doi.org/10.64898/2026.03.27.714683)

---

## Known Bugs

### B1. Ctrl+Right-drag split is unreliable ⚠️ partially fixed
**Symptom:** Watershed split via two-seed drag usually does nothing; occasionally splits
but leaves a thin barrier between the resulting cells.

**Root cause (suspected):**
- Seed detection during drag may be picking up incorrect positions (coordinate
  space mismatch, or seeds fall on the boundary rather than cell interior).
- The barrier artifact suggests `split_across()` in `_labels.py` does not remove the
  boundary pixels separating the two seeds before running watershed.

**Reference:** `Epicure/src/epicure/editing.py` — `split_in_crop()` uses a
`disk(radius)` dilation with a retry loop (up to 6 radii) to bridge noisy
boundaries. CellFlow has no retry mechanism and no pre-dilation of seeds.

**Fix plan:**
1. ~~Add debug logging to confirm seed coordinates at the moment of the callback.~~
2. ✅ Adopt the retry-with-increasing-dilation pattern from Epicure.
3. After watershed, use `remove_small_objects` or explicit erosion to eliminate
   sub-pixel barriers before writing back to the labels array.

**Status:** Retry loop with seed dilation (radii 0–6) added to `split_across()`.
Step 3 applied: boundary-zeroing removed entirely — cells now share a direct border
(0-pixel gap). Applied to `split_across`, `_split_in_crop`, and `_move_junction`.
Possible coordinate-space edge cases still to be investigated if split misfires.

---

### B2. Shift+Right-drag split gives an IndexError ⚠️ partially fixed
**Symptom:**
```
IndexError: tuple index out of range
  File ".../napari/_vispy/layers/base.py", line 202, in <listcomp>
    self._world_to_layer_units_scale[x] for x in dims_displayed
```
Triggered when the temporary `CorrectionDraw` shapes layer is added/modified
while napari's camera/dims state is not yet consistent.

**Fix plan:**
- ~~Delay shapes-layer creation until after the viewer has finished initialising
  (connect to `viewer.dims.events.ndisplay` if necessary).~~
- ✅ Create the shapes layer with `ndim=2` explicitly set; `dl.data` updates now
  use only (y, x) coordinates to match.

**Status:** IndexError resolved. Unexpected behaviour during Shift+Right-drag
still observed; further investigation needed.

---

### B3. Swap mode unreliable ⚠️ partially fixed
**Symptom:** The swap-mode key binding appears to register but the mode flag never
flips, so the swap operation is never triggered.

**Fix plan:** ✅ Audit the key-binding registration order in `_widget.py`; napari may
be consuming `Ctrl+W` before the layer callback fires. Consider rebinding to a
key that doesn't conflict with napari/Qt defaults.

**Status:** Partially fixed. Rebound from `w` to `s`. Swap rewritten as a two-click
operation (press `s` → click cell A → click cell B).

**Remaining issue:** Toggle behaviour is unreliable. Should be **hold-to-activate**
(swap fires while `s` is held, returns to normal on key release) rather than a
sticky toggle, to avoid getting stuck in swap mode after a failed click.

**Fix plan:** Bind `s` as a hold key using `napari.layers.Labels.bind_key` with the
`on_release` callback to reset `_swap_mode`. On press: set `_swap_mode = True` and
`_swap_first_pos = None`. On release: reset both.

---

### B4. Shift+Right-drag split by drawn line is unreliable ⚠️ open
**Symptom:** Drawing a line across a cell with Shift+Right-drag frequently fails to
split — `split_draw` returns False or produces no visible change.

**Root cause (suspected):**
- The drawn path positions are collected at irregular intervals during mouse_move;
  thin or fast strokes may not cross enough cell pixels to divide it cleanly.
- `_split_in_crop` requires the line to completely separate the cell into exactly
  2 connected components after dilation; a partial line that doesn't reach both
  sides of the cell will always fail even with the retry loop.
- Coordinate precision: `world_to_data` positions may be sub-pixel; rounding errors
  on small cells could misplace the line.

**Fix plan:**
1. Extend line endpoints to the cell boundary before running `_split_in_crop`, so
   a partial stroke still produces a full cut (similar to Epicure's approach of
   projecting the line to the nearest boundary pixel on each side).
2. Add fallback: if the drawn line doesn't divide the cell, dilate it progressively
   until it does (already done in `_split_in_crop` retry loop — check if max
   dilation radius of 6 is sufficient for thick cells; increase if needed).
3. Collect points more densely (interpolate between consecutive mouse_move events
   that are far apart) to handle fast strokes.

---

### B5. Ctrl+Z undo does not work ✅ works
**Symptom:** Pressing `Ctrl+Z` while the correction widget is active has no effect.

**Status:** napari 0.7.0's `Labels.undo()` tracks in-place numpy writes — confirmed
working. The existing `key_undo` binding (`Control-z` → `_layer.undo()`) is sufficient.

---

### B6. Attribution label unreadable in dark theme ✅ fixed
**Symptom:** The Epicure attribution `QLabel` was dark and hard to read in napari's
default dark theme.

**Fix:** Changed `setStyleSheet` from `color: palette(mid)` to `color: palette(text)`,
which uses the normal foreground colour that adapts to the active Qt palette.

---

### B7. Commands sometimes silently fail / fire inconsistently ✅ fixed
**Symptom:** Correction operations (merge, split, swap, erase) occasionally do nothing
even when the cursor is clearly over the right cell and the correct shortcut is used.
No error appears in the napari notification area.

**Fixes applied:**
1. ✅ **Modifier detection**: replaced fragile `_mod()` string-parser with
   `{m.name for m in event.modifiers}` — uses vispy's stable `.name` attribute.
2. ✅ **Focus stealing**: `_update_highlight()` already reasserts
   `viewer.layers.selection.active = self._layer` at the end; this is sufficient.
3. ✅ **Two-step state frame-guard**: `_ctrl_click_first_t` and `_swap_first_t` store
   the time frame at the first click. If the second click is on a different frame, the
   operation is cancelled with a status message and the first click is reset.
4. **`event.type` guard**: existing guard is correct; no change needed.

---

## Planned Features

### F1. Draw new cells that respect existing boundaries
**Description:** Allow the user to draw a closed or open path on top of an existing
cell (or in an unlabelled gap) and have a new cell label created that follows the
existing segmentation boundaries.

**Reference:** `Epicure/src/epicure/editing.py` — `create_cell_from_line()` and
`drawing_junction_mode()`. Activated with the `j` key; the drawn path is used to
seed a watershed that is constrained by the existing label boundaries.

**Implementation sketch:**
1. Add a `drawing_junction` mode toggle (key `d` or `j`).
2. When active, capture the drawn path via the shapes layer.
3. Dilate path to produce seed regions; run watershed constrained by existing
   label boundaries.
4. Assign new label, update layer.

---

### F2. Cell highlighting ✅ done
**Description:** Left-clicking a cell selects it, overlaying a cyan boundary polygon
on a persistent `CellHighlight` Shapes layer.  The highlight updates automatically
when the time-slider changes.  Clicking the background or starting a new operation
clears the selection.

**Implementation:**
- `_get_highlight_layer()` creates/returns the `CellHighlight` Shapes layer (ndim=2,
  cyan edge, transparent fill).
- `_update_highlight(t, lab)` runs `skimage.measure.find_contours` on the cell mask
  and sets the largest contour as a polygon.
- `_on_dims_change()` is connected to `viewer.dims.events.current_step` so the
  highlight re-renders across time frames.
- State: `_selected_label`, `_ctrl_click_first`, `_ctrl_click_first_label`,
  `_swap_first_pos` in `CorrectionWidget`.

---

### F3. Redesign keyboard shortcuts ✅ done
**Previous issues:** drag-based merge/split were not robust; `s`-toggle swap was
unreliable; right-click-erase conflicted with swap-second-click.

**Current mapping (selection-first model):**

| Action | Shortcut |
|---|---|
| Select / highlight cell | Left-click |
| Erase selected cell | `Delete` |
| Merge two cells | Ctrl+Left-click A → Ctrl+Left-click B (different labels) |
| Split cell (watershed, 2 seeds) | Ctrl+Left-click A → Ctrl+Left-click A again (same label) |
| Swap labels | Ctrl+Right-click A → Right-click B |
| Split (drawn line) | `Shift` + Right-drag (uses selected cell if set) |
| Redraw junction | `Shift` + Left-drag |
| Paint new cell | `N` |
| Fill new cell | `F` |
| Undo | `Ctrl+Z` |

**Add `D` to toggle draw-new-cell mode once F1 is implemented.**

---

### F4. Remove native napari label shortcuts from widget UI ✅ done
**Description:** The help panel now shows only CellFlow custom shortcuts under
the group box "Correction shortcuts" (previously "Label shortcuts"). Undo added
to the help text. Fill changed from `Shift-N` to `F`.

---

### F5. Attribution / citation notice ✅ done
**Description:** Attribution `QLabel` with hyperlinks to Epicure GitHub and DOI
added at the bottom of the correction dock panel.

---

## Implementation Order (suggested)

1. **B2** ✅ — Fix IndexError on Shift+Right-drag
2. **B1** ⚠️ — Fix watershed split (0-px boundary done; coordinate edge cases remain)
3. **B3** ✅ — Swap rewritten as Ctrl+Right-click → Right-click (no more `s` toggle)
4. **F4** ✅ — Cleaned up help panel (CellFlow shortcuts only, undo added)
5. **F3** ✅ — Full shortcut redesign: selection-first model (click to highlight, then act)
6. **F5** ✅ — Attribution added; label colour fixed to `palette(text)`
7. **F2** ✅ — Cell highlighting: left-click selects, cyan contour overlay, dims-aware
8. **B7** ✅ — Fix intermittent silent failures (modifier detection, frame-guard for two-step ops)
9. **B5** ✅ — Ctrl+Z works natively in napari 0.7.0
10. **B4** — Fix split-by-line reliability (line extension + denser sampling)
11. **F1** — Draw new cells from junctions (largest new feature)

---

## Reference Files

| Purpose | CellFlow | Epicure |
|---|---|---|
| Widget / UI | `cellflow/correction/_widget.py` | `src/epicure/editing.py` |
| Label operations | `cellflow/correction/_labels.py` | `src/epicure/editing.py` |
| Utilities | — | `src/epicure/Utils.py` |
| Shortcut config | inline in `_widget.py` | `src/epicure/preferences.py` |
