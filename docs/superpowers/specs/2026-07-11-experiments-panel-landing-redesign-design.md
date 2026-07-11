# Experiments-panel landing redesign — folder language, one-step Find, table tagging

Date: 2026-07-11 · Scope: `napari/_experiments_panel.py` + its two hosts
(`main_widget.py`, `contact_analysis_studio.py`) + tests.

## Problem

The landing panel (`ExperimentsPanel`, ported from napariTFM's ExperimentsList) is
the first thing a user meets, and it doesn't say what to do:

- **"Discover" names an action, not an object.** It opens a folder picker, recursively
  scans for subfolders containing the Setup-named images, and stages them. Nobody
  reads "Discover" and thinks "point me at the parent folder of my datasets."
- **The two-step Discover → Add to list is opaque.** Its only real job is baking
  batch **condition-label columns** onto a discovered set (for the aggregate tidy
  table) — but "Add to list" commits *all* staged rows, so the staging gate buys no
  selection, only a preview the committed list already provides.
- **No call to action and no explanation of the data model.** The tool is
  filesystem-centric (your folder *is* the project; results are written back into
  numbered stage subfolders), which is non-obvious and unexplained anywhere in the UI.

## Design

Four changes. Net UI: one fewer button, the Setup batch-column block retired, one new
tagging control, one `?` dialog.

### 1. Folder language

- Section title `Positions` → **Data folders**.
- Button `Discover` → **Find data folders…** (`…` signals a picker).
- Count `N positions` → **N data folders**.
- Docstrings/prose stop saying "position" for the user-facing noun (the internal
  `position_path` payload key and `_discover_positions` can stay — host-internal).

### 2. Collapse Discover → Add to list into one additive **Find**

`discover(root)` stops staging and instead **commits found folders directly**,
append-and-dedupe by key (this is exactly what `_add_entries` already does). Removed:
`commit_discovered`, `discovered`, `_discovered*`, `_preview_rows`, `_on_preview_clicked`,
`_update_staging`, `_staging_label`, the `commit_btn`, and the preview branch in
`delete_selected` / `_rebuild_table` / `keyPressEvent`.

Actions row: `[Find data folders…] [Delete selected]` (was three buttons).

Re-running Find against another root **accumulates** (dedupe by folder path); it never
replaces edited rows. Feedback is a transient status line:
- found: *"Added N data folders."*
- dry: *"No data folders found under `<root>` — check the image filenames in Setup."*
  (`discover()` receives `root`, so it can name it; this teaches the #1 failure mode —
  Setup filenames not matching — which the old silent "0" never did.)

### 3. Replace batch manual columns with table fill-down tagging

Condition labels must still reach the tidy table. Today values arrive two ways:
folder-nesting (auto, `columns_from_levels`) and the Setup batch fields. Cell *values*
are not directly editable (read-only `QLabel`s), and row-click drives selection — so
inline cell editing would fight the interaction model.

Instead, tagging becomes an explicit operation on the current selection:

- Retire the Setup `column / value / + Add column` block (`_build_manual_columns`,
  `add_manual_column`, `manual_columns`, `_manual_fields`, ctor `show_manual_columns`).
- Add a **tagging control** beneath the list:
  `Set column [name: combo of existing names + free text] = [value]  → selected`.
  On apply: `set_column_on_selected(name, value)` — ensure the column exists, write
  `value` into each selected row's `columns[name]`, rebuild, emit `records_changed`.
- Folder-derived columns and editable headers (`rename_column`) are unchanged.

Workflow: Find your folders (nesting auto-tags what it can) → select the WT rows, set
`condition = WT` → select the KO rows, set `condition = KO` → Run. Any subset,
any time, correctable after the fact — strictly more capable than the old batch bake,
and it fills the missing "enter an arbitrary value" gap.

### 4. `?` quickstart dialog

A `?` tool button in the panel heading opens a self-contained `QDialog` (offline —
no browser/hosted-docs dependency; the standalone distros mount this panel too),
distilled from `docs/manual/workflow.md`. Three beats:

1. **What CellFlow works on** — one *data folder* per movie / field of view, each
   holding your raw nucleus + cell images.
2. **How results are stored** — written back into numbered stage folders inside each
   data folder (`0_input → 1_cellpose → 2_nucleus → 3_cell → aggregate_quantification`);
   the folder is the source of truth, nothing hidden.
3. **The flow** — (a) Setup: name your image files; (b) Find data folders: point at
   the parent directory that holds them; (c) tag conditions; (d) Run. Note that the
   condition columns become the grouping columns of the aggregate tidy table.

The empty-list state shows the call to action + a soft nudge: *"Add folders with cell
and nucleus images to start. New here? Click ? for a quickstart."*

## Host impact

Both hosts open a picker then call `panel.discover(root)` — unchanged. Edits: drop
`show_manual_columns=True` (2 call sites), fix the `contact_analysis_studio` docstring
that describes the staging flow. No signal-contract change (`discover_requested` stays).

## Out of scope

- The `[no folder]` header label (separate main-app element; revisit after this lands).
- Collapsing/refactoring the folder-derived-column machinery.
- Directly-editable value cells (fill-down covers the need without fighting row-click).

## Tests

- `test_experiments_panel.py`: replace discover→commit→manual-columns cases with
  additive-Find (dedupe, dry-scan feedback), `set_column_on_selected` fill-down,
  and empty-state CTA visibility. Qt bootstrap via napari `get_qapp()` (bare
  `QApplication([])` aborts under pytest).
- `test_contact_analysis_studio.py`: adjust any staging assertions.
