# Aggregate Quantification: curation tool (napari)

**Status:** decided direction, pre-implementation (open questions listed)
**Date:** 2026-06-22
**Scope:** the napari UI for authoring the curation exclusion table — browse a
series' positions, scrub frames with the contact visualization as the on-image
overlay, and mark a frame or a whole position excluded with a reason.
**Companions:** the artifact it writes
(`2026-06-22-aggregate-curation-exclusion-table-design.md`); the surrounding napari
refocus (`2026-06-22-aggregate-napari-frontend-refocus-design.md`).

## Statement of need

The exclusion decision is made *by looking at the images* — so it belongs on the
napari canvas, the one place an image-linked judgement is natural. This is the
feature that re-justifies keeping Aggregate Quantification in napari at all (the
depart-napari spec dropped the image canvas precisely because batch stats did not
need it; curation does). The tool turns "scrub through a position, see a bad frame,
exclude it with a note" into rows of the curation table.

## What we already have (reuse)

- **Contact visualization** (`napari/aggregate_quantification/plugins/visualize_contacts.py`
  + `contact_visualization.py`): renders cell labels, cell–cell edges, and T1 events
  as napari layers over the image. This becomes the curation **display** rather than a
  standalone feature.
- **Discovery / catalog**: `build_catalog` / `load_catalog` enumerate the series'
  positions and their label/image paths — the list the curator browses.
- The frame-preview cache + scrubbing patterns already used by other widgets.

## Decisions

### 1. The tool is a curation widget that reads a catalog and writes the exclusion table.

Inputs: a run-config (or its catalog + curation path). It lists the catalog's
positions; selecting one loads its layers and the curator scrubs frames. Output: it
appends/edits rows in the curation CSV (the artifact of the companion spec).

### 2. Canvas = images + contact-viz overlay.

Per position, the canvas shows the raw/label image(s) plus the contact
visualization overlay (labels, edges, T1). The curator judges from this. No plots in
napari (Iris owns plotting).

### 3. Exclusion actions.

- **Exclude this frame** — adds a `(experiment_id, position_id, frame, reason)` row.
- **Exclude this position** — adds a whole-position row (`frame` NA, per the table
  spec).
- A **reason** is required for each action (a text field; the action is disabled
  until non-empty).
- Excluded frames/positions are visibly marked (e.g. a frame-strip tick, a "position
  excluded" badge) and are **reversible** (remove the row) within the session.

### 4. Persistence is the table, not session state.

The widget reads the existing curation CSV on open (so prior decisions show) and
writes back on each change (or on an explicit Save — see Open #2). The table is the
single source of truth; the widget is a view/editor over it.

### 5. Lives in the refocused napari plugin.

It sits alongside discover&add + run in the Aggregate Quantification widget (see the
refocus spec), not as a separate app — "everything in one place."

## Components

- `CurationWidget` — the Qt widget: position list, canvas wiring, frame scrubber,
  exclude-frame / exclude-position buttons, reason field, exclusion list/badges.
- A thin controller that maps widget actions ↔ `curation.py` (`read_curation` /
  append / remove), keeping Qt out of the table logic so the table ops stay unit-
  testable headless.
- Reuses the contact-visualization layer builder as the display backend.

## Open / deferred

1. **Frame-range exclusion UX** — exclude a contiguous run of frames in one gesture
   (select start/end on the scrubber) that expands to individual table rows, vs.
   one-frame-at-a-time. Lean: support a range gesture; table stays per-frame rows.
2. **Auto-save vs explicit Save** — write the CSV on every action (simplest, always
   consistent) vs. batch with a Save button (undo-friendlier). Lean auto-save with
   in-session remove.
3. **What "browse the data" includes** — confirmed image-only (no plots). If a
   per-position summary number is ever wanted as a hint, it would be a label, not a
   plot. Out of scope now.
4. **Multi-channel / 3D display** — which layers load by default and how Z is handled
   is inherited from the existing contact-viz loader; no new decisions here.
