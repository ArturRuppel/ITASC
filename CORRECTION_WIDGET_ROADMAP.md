# Correction Widget Roadmap

This document tracks planned improvements to CellFlow's correction widget.
Reference implementation: [Epicure](https://github.com/Image-Analysis-Hub/Epicure)
(cite: https://doi.org/10.64898/2026.03.27.714683)

---

## Known Bugs

### B1. Ctrl+Right-drag split is unreliable ✅ fixed
**Symptom:** Watershed split via two-seed Ctrl+Left-click occasionally splits
but left a thin barrier between the resulting cells, or did nothing.

**Fix applied:** Retry loop with seed dilation (radii 0–6) added to `split_across()`.
Boundary-zeroing removed — cells now share a direct 0-pixel-gap border.
Applied to `split_across`, `_split_in_crop`.

---

### B2. Shift+Right-drag split gives an IndexError ✅ fixed
**Symptom:**
```
IndexError: tuple index out of range
  File ".../napari/_vispy/layers/base.py", line 202, in <listcomp>
    self._world_to_layer_units_scale[x] for x in dims_displayed
```
Triggered when the temporary `CorrectionDraw` shapes layer is added/modified
while napari's camera/dims state is not yet consistent.

**Fix:** Shapes layer created with `ndim=2` explicitly set; `dl.data` updates use
only (y, x) coordinates.

---

### B3. Swap mode unreliable ✅ fixed
**Symptom:** The swap-mode key binding appeared to register but the mode flag never
flipped, so the swap operation was never triggered.

**Fix:** Rebound from `w` to `Ctrl+Right-click`. Swap rewritten as a two-click
operation (Ctrl+Right-click A → Right-click B), eliminating the toggle.

---

### B4. Shift+Right-drag split by drawn line is unreliable ✅ fixed
**Symptom:** Drawing a line across a cell frequently failed to split.

**Fixes applied:**
1. ✅ Endpoint extension: `_extend_endpoints()` prepends/appends points beyond each
   end of the stroke so the line crosses the cell boundary even with a partial drag.
2. ✅ Retry dilation in `_split_in_crop` (radii 0–6) bridges gaps from fast/thin strokes.
3. ✅ Fallback: if the extended line still misses the target cell, infer the target
   from the labels actually under the drawn path.

---

### B5. Ctrl+Z undo does not work ✅ works
**Symptom:** Pressing `Ctrl+Z` while the correction widget is active had no effect.

**Status:** All correction ops call `_record_history()` which pushes a
`(indices, old_values, new_values)` atom onto napari's undo stack.

---

### B6. Attribution label unreadable in dark theme ✅ fixed
**Fix:** `color: palette(text)` instead of `color: palette(mid)`.

---

### B7. Commands sometimes silently fail / fire inconsistently ✅ fixed
**Fixes applied:**
1. ✅ **Modifier detection**: `{m.name for m in event.modifiers}` instead of a string parser.
2. ✅ **Focus stealing**: `_update_highlight()` reasserts `viewer.layers.selection.active`.
3. ✅ **Two-step state frame-guard**: `_ctrl_click_first_t` and `_swap_first_t` cancel
   the pending operation if the time frame changes between clicks.

---

### B8. Selecting a cell with Ctrl held blocks subsequent merge/swap ⚠️ open
**Symptom:** If the user holds Ctrl while left-clicking a cell, `_ctrl_click_first`
split-seed state is set. A subsequent Ctrl+Left-click on a *different* cell cancels
the split state but doesn't trigger a merge because `_selected_label` was never set.

**Fix plan:** When Ctrl+Left-click lands on a different cell than `_ctrl_click_first_label`,
promote the first seed to `_selected_label`/`_selected_pos` and immediately attempt
the merge with the new click, rather than silently dropping state.

---

### B9. Napari tool mode change breaks correction shortcuts ⚠️ open
**Symptom:** Clicking the napari toolbar (paint brush, fill, erase, etc.) or pressing
napari's native label shortcuts switches the Labels layer mode away from `pan_zoom`.
In any mode other than `pan_zoom`, napari intercepts mouse events before the
correction widget's callbacks fire, silently breaking all shortcuts.

**Fix plan:**
1. Connect `layer.events.mode` in `_activate()` and disconnect in `_deactivate()`.
2. When mode changes to anything other than `pan_zoom`, show a visible warning in
   the widget (status label + a "Restore correction mode" button).
3. The button sets `layer.mode = "pan_zoom"` and hides itself.
4. Note: `n` and `f` key bindings have been removed from the correction widget to
   avoid confusing users; the shortcut table no longer lists them.

---

## Planned Features

### F1. Draw cell path ✅ done (merged with old junction-redraw)
**Description:** Shift+Left-drag draws a thickened stroke that overwrites existing
label boundaries.

- **Cell selected:** pixels along the stroke are assigned to the selected cell
  (extends it along the drawn path).
- **No cell selected:** pixels are assigned to a new free label.

**Implementation:** `draw_cell_path(seg, positions, curlabel, radius)` in `_labels.py`.
Stroke is interpolated, endpoints extended by `radius*2`, then dilated with `disk(radius)`
before writing to the segmentation.

---

### F2. Cell highlighting ✅ done
**Description:** Left-clicking a cell selects it, overlaying a cyan boundary polygon
on a persistent `CellHighlight` Shapes layer.  The highlight updates automatically
when the time-slider changes.  Clicking the background or starting a new operation
clears the selection.

---

### F3. Redesign keyboard shortcuts ✅ done
**Current mapping (selection-first model):**

| Action | Shortcut |
|---|---|
| Select / highlight cell | Left-click |
| Erase selected cell | `Delete` |
| Merge two cells | Ctrl+Left-click A → Ctrl+Left-click B (different labels) |
| Split cell (watershed, 2 seeds) | Ctrl+Left-click A → Ctrl+Left-click A again (same label) |
| Swap labels | Ctrl+Right-click A → Right-click B |
| Split (drawn line) | `Shift` + Right-drag (uses selected cell if set) |
| Draw cell path | `Shift` + Left-drag (extends selected cell, or new cell) |
| Undo | `Ctrl+Z` |

---

### F4. Remove native napari label shortcuts from widget UI ✅ done
**Description:** The help panel shows only CellFlow custom shortcuts.

---

### F5. Attribution / citation notice ✅ done
**Description:** Attribution `QLabel` with hyperlinks to Epicure GitHub and DOI
added at the bottom of the correction dock panel.

---

## Implementation Order (suggested)

1. **B2** ✅ — Fix IndexError on Shift+Right-drag
2. **B1** ✅ — Fix watershed split
3. **B3** ✅ — Swap rewritten as Ctrl+Right-click → Right-click
4. **F4** ✅ — Cleaned up help panel
5. **F3** ✅ — Full shortcut redesign: selection-first model
6. **F5** ✅ — Attribution added
7. **F2** ✅ — Cell highlighting
8. **B7** ✅ — Fix intermittent silent failures
9. **B5** ✅ — Ctrl+Z works
10. **B4** ✅ — Fix split-by-line reliability
11. **F1** ✅ — Draw cell path (replaces junction-redraw)
12. **B9** ⚠️ — Napari tool mode warning + reset button
13. **B8** ⚠️ — Ctrl-held merge/swap state fix

---

## Reference Files

| Purpose | CellFlow | Epicure |
|---|---|---|
| Widget / UI | `cellflow/correction/_widget.py` | `src/epicure/editing.py` |
| Label operations | `cellflow/correction/_labels.py` | `src/epicure/editing.py` |
| Utilities | — | `src/epicure/Utils.py` |
| Shortcut config | inline in `_widget.py` | `src/epicure/preferences.py` |
