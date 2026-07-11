# Full-app config vs. project file — autosave params, explicit project catalog

Date: 2026-07-11 · Scope: `napari/main_widget.py` + the three stage widgets it
hosts (`cellpose_widget.py`, `nucleus_workflow_widget.py`, `cell_workflow_widget.py`)
+ a save-side catalog helper + tests. Reuses `contact_analysis/catalog.py` wholesale.

## Problem

The full app's toolbar muddles two different things under one "config" idea, and
omits a third thing entirely:

- **The per-folder config is saved and loaded by hand.** `cellflow_config.json`
  (calibration + Cellpose/nucleus/cell params) belongs to the folder it describes,
  yet the toolbar makes the user click "save config into folder" / "load config
  from folder" to move it. Config-load already happens automatically on folder
  select; config-save does not, so a processed folder can end up with results but
  no record of the parameters that made them.
- **There is no project file.** The Data folders list (folders + classification
  columns) that the landing panel assembles lives only in memory — close napari and
  the catalog and its condition tagging are gone. The aggregate studio already
  serializes exactly this to `catalog.csv`; the full app does not.
- **The `?` quickstart asserts a falsehood-to-be:** *"CellFlow has no project
  file."*

## Design

Two files, separated by what they are. **Config = parameters, autosaved per
folder. Project = the folder catalog, saved explicitly to CSV.**

### 1. Config autosaves into the processed folder

`cellflow_config.json` stops being a manual action:

- `main_widget` connects the existing run buttons' `clicked` signals to one
  `_autosave_config()` slot, touching no child-widget code. The six run entry
  points are already exposed as attributes: `CellposeWidget.nucleus_run_btn` /
  `.cell_run_btn`, `NucleusWorkflowWidget.seg_run_btn` / `.db_run_btn` /
  `.solve_run_btn`, `CellWorkflowWidget.run_btn`. (`main_widget` already reaches
  into these widgets' internals for status, so this coupling is consistent.)
- `_autosave_config()` is
  `if self._pos_dir is not None: self._save_config(self._pos_dir / "cellflow_config.json", quiet=True)`.
  `_save_config` gains a `quiet` flag that suppresses the `show_info` toast, so
  per-run autosave does not spam notifications.
- The cell/nucleus run buttons double as cancel toggles, so a *cancel* click also
  autosaves. Harmless: it rewrites the same config. Accepted, not guarded.
- Autoload on folder select is unchanged (`_on_active_position` /
  `_on_set_position_folder` already load the folder's config if present).

### 2. Config-as-file: keep one Save, one Load

The existing file-based pair (`save_as` / `load_from`) stays, for carrying a tuned
parameter set between experiments. Relabel for clarity:
**Load config from file… / Save config to file…**.

### 3. Project catalog: explicit Save/Load, reusing `catalog.py`

- **Save project…** — map each panel record to a full catalog record (see the
  adapter below), then `save_catalog(path, records)`. Default the save dialog's
  filename to `catalog.csv` so the aggregate studio ingests it unmodified; append
  `.csv` if the user's chosen name lacks it (same guard the studio uses).
- **Load project…** — `load_catalog(path)`, `merge_catalog_records(panel.records(),
  loaded)`, then rebuild panel entries and `panel.set_records(entries)` — the exact
  shape of `contact_analysis_studio._load_csv_from`.

#### The save-side adapter (the integration risk)

The full app's panel records are `{"position_path": Path, "columns": {...}}` — they
carry classification but **no `path` / label-image fields**, because a folder may
not be processed yet. `save_catalog` → `_normalize_catalog_record` would then
default `path` to empty and the aggregate could not locate the `.h5`. So Save
project must stamp each record with the default staged-layout paths before saving:

- `contact_analysis_path` = `position_path / aggregate_quantification/contact_analysis.h5`
  (`catalog.CONTACT_ANALYSIS_RELPATH`, already the fixed default location).
- `cell_tracked_labels_path` = `position_path / 3_cell/tracked_labels.tif`.
- `nucleus_tracked_labels_path` = `position_path / 2_nucleus/tracked_labels.tif`.

These are the same staged-layout paths `main_widget._refresh_all` already hardcodes
when wiring the contact-analysis context; the adapter reuses those constants. The
classification `columns` bag (`condition` / `experiment_id` / `position_id` / any
folder-nesting or manual tag columns) passes through untouched — normalization maps
the recognized keys to their CSV homes and appends the rest as free-form columns.

Put the adapter where the staged-layout knowledge lives (a small
`_catalog_record_for_position(position_path, columns)` in `main_widget`, or a
reused module-level helper if one already holds those path constants). Keep it out
of `catalog.py` — that module is aggregate-side and should not learn the full app's
`3_cell` / `2_nucleus` layout.

**Acceptance bar (verify end-to-end, not just unit tests):** a project CSV saved
from the full app loads into the aggregate studio's catalogue and drives a run with
no hand-editing. Also round-trips within the full app (Save → clear → Load restores
the same rows + columns).

### 4. Toolbar after the change

- **Project group:** `Select folder…` (`new`, kept) · `Load project…` · `Save
  project…` — the latter two act on the catalog CSV.
- **Params group:** `Load config from file…` (`load_from`) · `Save config to
  file…` (`save_as`).
- **Removed:** the two per-folder config buttons (`load` = load-config-from-folder,
  `save` = save-config-into-folder) and their handlers (`_on_load_config`,
  `_on_save_config`). `_on_save_config_as` / `_on_load_config_from` stay.
- `Refresh` unchanged.

Reuse the existing `load` / `save` toolbar glyphs for both Project and Params pairs
(the group caption + arrow direction disambiguate them, as they do today).

### 5. Quickstart correction

Rewrite the `_QUICKSTART_HTML` "no project file" beat in `_experiments_panel.py`:
the folder-on-disk is still the source of truth for *results*, but the folder
catalog + classification now persist to a project CSV you save and reload, and that
CSV is what drives aggregate quantification. Keep it accurate for both hosts (the
standalone aggregate studio already has its catalog CSV).

## Out of scope

- Auto-persisting the project CSV to a scanned root (decided: explicit Save/Load).
- Batch "Run selected" in the full app (`show_run=False` stays; per-stage runs are
  the autosave trigger).
- Unifying the two hosts' discovery payloads or panels beyond the shared catalog
  format.
- The `[no folder]` header label (tracked separately).

## Tests

- `test_main_widget_*`: clicking any of the six wired run buttons writes
  `cellflow_config.json` into `_pos_dir`; autosave no-ops when `_pos_dir is None`;
  `_save_config(..., quiet=True)` emits no toast; the two per-folder config
  buttons/handlers are gone.
- Project Save/Load: the save-side adapter stamps the three default staged paths;
  `save_catalog` output re-`load_catalog`s to the same rows; a full-app-saved CSV
  passes `contact_analysis.catalog` validation (`REQUIRED_CSV_COLUMNS`, unique
  identities) and carries a resolvable `contact_analysis_path`.
- Qt bootstrap via napari `get_qapp()` (bare `QApplication([])` aborts under
  pytest).
- End-to-end (manual/driven): full app Save project → aggregate studio Load →
  run, per the acceptance bar.
