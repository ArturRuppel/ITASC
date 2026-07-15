# Nucleus Correction Widget Redesign — Design

**Date:** 2026-06-07
**Status:** Approved (design); ready for implementation planning
**Scope:** `NucleusCorrectionWidget` and the view panels it owns. `CellCorrectionWidget` is out of scope.

## Goal

Rework the layout of the nucleus correction widget's **active correction state** (the
right-side workspace dock) so that:

1. The swimlane **track overview** and the per-track **thumbnail film strip** merge into a
   single panel where a track **expands inline into its thumbnails when selected**.
2. The right column is rationalized into a thin left toolbar (action buttons, bigger
   icons), a full-width top bar (title, shortcuts, params, view toggles, status), and the
   unified track panel as the main surface.
3. Zoom is region-dependent: Ctrl+wheel over thumbnails resizes the thumbnails; Ctrl+wheel
   over the track bars changes bar height only; the time axis always fits the panel width.

This is a **layout + unified-panel** change. Correction logic, validation/anchor records,
retrack/extend/swap, layer lifecycle, and the candidate gallery's internals are untouched.
The inactive plugin-dock header (the compact entry point that toggles correction mode on)
is unchanged.

## Active-mode layout

The workspace dock becomes:

```
┌─ top bar (full width) ─────────────────────────────────────────────┐
│ Correction   ⏻  📖 ⚙   👁track ▦gallery 🎨filled        status… │
├─ reveal area (full width, grows to fit; 📖 and ⚙ independent) ──────┤
│ [ ⚙ params: sliders / checkboxes laid out wide ]                   │
│ [ 📖 shortcuts: multi-column grid + disclaimer ]                   │
├─ body (horizontal splitter) ───────────────────────────────────────┤
│ ┌toolbar┐ ┌─gallery─┐ ┌── unified accordion panel (stretch) ──┐    │
│ │  💾   │ │ (toggle)│ │  track bars; selected expands inline   │    │
│ │ ↶ ↷  │ │         │ │  to its wrapped thumbnail band          │    │
│ │ ✓ ⚓  │ │         │ │                                        │    │
│ │  ✎   │ │         │ │                                        │    │
│ │ # 🗑  │ │         │ │                                        │    │
│ └──────┘ └─────────┘ └────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

- **Top bar** spans the full width, over the toolbar, gallery, and accordion.
- **Body** is a horizontal `QSplitter`, ordered left→right: thin toolbar (fixed narrow) ·
  candidate gallery (toggleable, resizable handle) · accordion (stretch = 1, the panel that
  absorbs outer-width changes).

## Component 1 — Unified accordion panel (`TrackAccordionPanel`)

New file `src/cellflow/napari/_correction_track_accordion.py`. One `QGraphicsView` +
`QGraphicsScene`, replacing both `LineageCanvasPanel` and `TrackFilmStripPanel`.

### Per-track rows (laid top-down at a running `y`)

- **Collapsed track:** one thin bar — present-runs as grey rects, validated frames green,
  anchored orange (the existing lane drawing). Height = `lane_h` (Ctrl+wheel-adjustable).
- **Selected/expanded track:** the bar **stays** as a header row, and directly beneath it a
  **wrapped thumbnail band** of that track's per-frame crops, laid row-major and wrapping to
  the panel width. Rows below shift down to make room. **One track expanded at a time**,
  driven by selection.

### Shared geometry

- `cell_w = (viewport_width − left_gutter) / n_frames`, recomputed on every resize, so the
  **whole global time axis always fits the panel width** — no horizontal scroll for bars.
  All tracks keep a single shared time axis.
- One vertical **current-frame guide** line spanning all rows.
- One **selected-row** marker on the open track.
- Thumbnail tiles carry the validated/anchored marker strips and the current-frame border,
  as the film strip does today.

### Interaction

- **Click a bar** → emit `node_activated(frame, cell_id)`: jump there and select the cell,
  which expands that track.
- **Click a thumbnail** → emit `frame_clicked(frame)`: jump to that frame of the selected
  track.
- **Ctrl+wheel** is region-aware via hit-testing the cursor position:
  - over a thumbnail tile / anywhere in the expanded band → resize **tile size** (existing
    `_TILE_PX` clamp);
  - over a bar / elsewhere → change **`lane_h`** (bar height only; `cell_w` is unaffected
    because it is width-derived).
- **Plain wheel** → scroll the panel vertically through the track list.

### Tile rendering reuse

The pure tile builders `TrackFilmStrip` and `build_track_film_strip` stay in
`_correction_track_path.py` and are reused as-is. `rgb_to_qimage` (currently in
`_correction_film_strip.py`) moves into the new accordion module.

## Component 2 — Controller (`LineageCanvasController`)

`src/cellflow/napari/lineage_canvas_controller.py` is simplified to drive the one panel. It
already assembles `LaneView`s and builds the selected track's `TrackFilmStrip`; it now hands
both to a single panel rather than two.

- Public methods keep their names so existing host call sites are unchanged: `refresh`,
  `set_selection`, `set_current_frame`, `center_on_track`, `refresh_status`,
  `refresh_detail`, `teardown`.
- The two embed accessors `overview_panel()` and `film_widget()` collapse into one:
  `panel()`.

## Component 3 — Top bar (`build_correction_header`)

Full-width `QHBoxLayout`:

- `Correction` title label (stage-styled).
- `⏻` activate/deactivate toggle — **moved here** from the plugin header so correction mode
  can be exited while the plugin dock is hidden.
- `📖` shortcuts and `⚙` params toggles.
- View toggles converted from checkboxes to checkable icon tool-buttons: `👁` Track path,
  `▦` Candidate gallery, `🎨` Filled labels. Tooltips carry today's longer descriptions.
- The old **Lineage canvas** checkbox is **removed** — the accordion is the always-on main
  surface.
- `addStretch`, then the **status label** right-aligned at the end of the bar; the
  validation counter rides in/near the status area.

## Component 4 — Thin left toolbar (`build_correction_toolbar`)

The existing toolbar groups, rendered **vertically** in a narrow column with bigger glyphs
(reuse `_enlarge_glyph`, bump scale as needed). Same groups and order:
`💾` save · `↶ ↷` retrack · `✓ ⚓` validate/anchor · `✎` annotate ·
`# 🗑` reassign / remove-unvalidated. Group separators become horizontal rules.

## Component 5 — Reveal area + disclaimer (`build_shortcuts_widget`, params section)

Directly under the top bar, full width. `📖` and `⚙` are **independent** — both can be open
at once; the top area simply grows to fit. Content is reflowed **wide-and-short**:

- **Params (⚙):** scope combo, hole/opening/expand sliders, extend/retrack sliders,
  greedy-overwrite checkbox, and spawn controls, arranged across multiple columns instead of
  a tall stack.
- **Shortcuts (📖):** the shortcut groups arranged as multiple side-by-side columns, with
  the **disclaimer / attribution label** (`_attrib_lbl`) moved to the bottom of this panel.

## Files touched

**New**
- `src/cellflow/napari/_correction_track_accordion.py` — `TrackAccordionPanel` + `LaneView`;
  houses the relocated `rgb_to_qimage`.

**Modified**
- `src/cellflow/napari/nucleus_correction_widget.py` — `_setup_ui` restructure (top bar with
  toggles + status, vertical toolbar, full-width reveal area, body splitter ordered
  toolbar · gallery · accordion); remove `lineage_canvas_check`; move view toggles into the
  top bar; drop the controls-strip container; one `panel()` embed instead of
  `overview_panel()` + `film_widget()`.
- `src/cellflow/napari/lineage_canvas_controller.py` — drive the one panel; replace
  `overview_panel()` / `film_widget()` with `panel()`; keep all other method names.
- `src/cellflow/napari/_correction_ui.py` — `build_correction_header` (full top bar incl.
  toggles + status), `build_correction_toolbar` (vertical, bigger glyphs),
  `build_shortcuts_widget` (wide multi-column + disclaimer at bottom).
- `src/cellflow/napari/_correction_candidate_panel.py` — update the `rgb_to_qimage` import to
  its new home.

**Retired**
- `src/cellflow/napari/_correction_lineage_canvas.py`, `src/cellflow/napari/_correction_film_strip.py`
  — pure bits folded into the new module. `TrackFilmStrip` / `build_track_film_strip` remain
  in `_correction_track_path.py`.

## Testing

- Add `tests/napari/test_correction_track_accordion.py` covering: collapsed-bar geometry;
  accordion expand on selection (bar retained + thumbnail band beneath); width-derived
  `cell_w` fit on resize; region-aware Ctrl+wheel (tile resize vs lane-height); plain-wheel
  scroll; click → `node_activated` / `frame_clicked`.
- Replace `tests/napari/test_lineage_canvas_panel.py` and
  `tests/napari/test_correction_film_strip.py` (their coverage moves into the accordion test).
- Update `tests/napari/test_lineage_canvas_controller.py` for the single-panel API.
- Update `tests/napari/test_correction_relabel_refresh.py` to drop the `lineage_canvas_check`
  fake (that toggle no longer exists).

## Open questions / confirmed assumptions

- **Confirmed:** "horizontally fit the full track" = the entire global time axis is scaled to
  the panel width; the shared time axis and single current-frame guide are preserved.
- **Confirmed:** accordion keeps the selected track's bar as a header above its thumbnails;
  one track expanded at a time.
- **Confirmed:** candidate gallery stays a separate, toggleable panel; placed between the
  toolbar and the accordion.
- **Confirmed:** Ctrl+wheel zooms (region-dependent); plain wheel scrolls vertically.
- **Confirmed:** view toggles live in the top bar; shortcuts/params reveal full-width below
  it and are independently openable.
