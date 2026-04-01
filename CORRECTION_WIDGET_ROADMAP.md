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
Unexpected behaviour remains; step 3 and possible coordinate-space issues still to
be investigated.

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

### B3. Ctrl+W swap mode toggle does nothing ⚠️ partially fixed
**Symptom:** The swap-mode key binding appears to register but the mode flag never
flips, so the swap operation is never triggered.

**Fix plan:** ✅ Audit the key-binding registration order in `_widget.py`; napari may
be consuming `Ctrl+W` before the layer callback fires. Consider rebinding to a
key that doesn't conflict with napari/Qt defaults.

**Status:** Rebound from `w` to `s`. Toggle now fires, but unexpected behaviour
in the swap operation itself remains.

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

### F2. Cell highlighting
**Description:** Visually highlight a cell (e.g., on hover or explicit selection)
by overlaying its boundary in a distinct colour, without modifying the labels layer.

**Implementation sketch:**
- Create a persistent `Shapes` layer named "Highlight".
- On mouse-move (or on selection), find the label under the cursor, extract its
  contour with `find_boundaries`, and set it as the shapes-layer data.
- Clear the layer on mode exit.

---

### F3. Redesign keyboard shortcuts
**Current issues:**
- `Shift` bindings cause the IndexError (B2 above).
- `Ctrl+W` conflicts with system/Qt close-window shortcut.
- The overall mapping is not intuitive.

**Proposed new mapping (draft — iterate with user):**

| Action | Shortcut |
|---|---|
| Paint new cell | `N` |
| Fill new cell | `F` |
| Erase cell | Right-click |
| Merge cells | `Ctrl` + left-drag (A→B) |
| Split (watershed, 2 seeds) | `Alt` + left-drag (2 pts on same cell) |
| Split (drawn line) | `Alt` + right-drag |
| Redraw junction | `Ctrl` + right-drag |
| Draw new cell (junction mode) | `D` to toggle, then left-drag |
| Swap labels | `S` |
| Undo | `Ctrl+Z` |

Remove all references to native napari label shortcuts from the widget help text.

---

### F4. Remove native napari label shortcuts from widget UI
**Description:** The help panel currently lists napari's default Labels-layer
shortcuts (e.g., paint, fill, erase keybindings built into napari). These are
confusing alongside CellFlow's custom bindings and should be removed from the
widget text.

---

### F5. Attribution / citation notice
**Description:** Add a visible attribution in the correction widget UI crediting
Epicure, with DOI and GitHub link.

**Text:**
> Correction tools adapted from
> [Epicure](https://github.com/Image-Analysis-Hub/Epicure).
> If you use these tools, please cite:
> https://doi.org/10.64898/2026.03.27.714683

**Placement:** Bottom of the widget dock panel, in a small `QLabel` with a
hyperlink.

---

## Implementation Order (suggested)

1. **B2** — Fix IndexError on Shift+Right-drag (blocks any testing of split-by-line)
2. **B1** — Fix Ctrl+Right-drag watershed split reliability
3. **B3** — Fix Ctrl+W swap mode toggle
4. **F4** — Remove napari native shortcuts from help text (trivial, good cleanup)
5. **F3** — Redesign shortcuts (dependent on B2/B3 being resolved)
6. **F5** — Add attribution (trivial)
7. **F2** — Cell highlighting
8. **F1** — Draw new cells from junctions (largest new feature)

---

## Reference Files

| Purpose | CellFlow | Epicure |
|---|---|---|
| Widget / UI | `cellflow/correction/_widget.py` | `src/epicure/editing.py` |
| Label operations | `cellflow/correction/_labels.py` | `src/epicure/editing.py` |
| Utilities | — | `src/epicure/Utils.py` |
| Shortcut config | inline in `_widget.py` | `src/epicure/preferences.py` |
