# Contact-Analysis rename + position-status catalog rail — design

**Date:** 2026-07-03
**Status:** design (approved-in-principle; plan next)
**Supersedes naming in:** the `aggregate_quantification` package (rename).
**Builds on:** `2026-06-24-independent-distros-and-batch-mode-todo.md` (P2 commit
contract), `2026-06-29-folder-derived-catalog-metadata-design.md` (the catalog).

## Why

Two things drove this. First, the user wants the napari **catalog UX** to feel as
finished as napariTFM's `ExperimentsList` — specifically a **per-position status
rail** that shows, at a glance, how far each position has progressed through the
(human-in-the-loop, not auto-chained) pipeline. Second, the umbrella package is
still named `aggregate_quantification` while the pipeline **stage name**
(`core/paths.py::STAGE_DIRS`) and the **user-facing label**
(`napari/data_panel_widget.py` header comment: "Contact Analysis →
contact_analysis_widget") already say **contact analysis**. The package name
should catch up to the name the codebase already uses at its edges.

The two are coupled: the status rail's per-stage detection depends on a clean
"this stage is done" signal per position, which is exactly what the P2 commit
contract provides. So this spec sequences three stages:

- **Stage 0 — Rename** `aggregate_quantification` → `contact_analysis` (code-only).
- **Stage 1 — Commit contract** (P2): promote working labels to stable
  base-folder final outputs, giving each stage a crisp done-signal.
- **Stage 2 — Catalog status rail**: rebuild the catalog rows as custom widgets
  carrying a four-dot rail, porting napariTFM's presentation while keeping
  CellFlow's stronger catalog data-model.

Stages 0 and 1 are independently valuable and land first; Stage 2 is the bigger
rewrite and consumes both.

---

## Stage 0 — Rename `aggregate_quantification` → `contact_analysis`

### Scope: code names only. On-disk artifact folders are NOT renamed here.

`STAGE_DIRS` already maps stage `"contact_analysis"` → folder
`"aggregate_quantification"`. We keep that decoupling: the **code** becomes
`contact_analysis`; the **on-disk output folder stays** `aggregate_quantification`
behind the alias, so no existing dataset (COV2D, labbook database) breaks. A
later, optional migration can rename the folder + `.h5` on disk with a
back-compat read path; that is explicitly **out of scope** here.

**Decision knob (default chosen):** code-only rename, folder kept. Flip only if
you want the disk layout to match immediately and accept a data-migration pass.

### Renames

- Package dir `src/cellflow/aggregate_quantification/` → `src/cellflow/contact_analysis/`.
- All `AggregateQuantification*` classes → `ContactAnalysis*`
  (`AggregateQuantificationWidget` → `ContactAnalysisWidget`,
  `AggregateQuantificationStudioWidget` → `ContactAnalysisStudioWidget`).
- Widget files `napari/aggregate_quantification_*.py` →
  `napari/contact_analysis_*.py`; factory
  `make_aggregate_quantification_widget` → `make_contact_analysis_widget`.
- napari command ids: `cellflow.aggregate_quantification_widget` →
  `cellflow.contact_analysis_widget`; menu title stays "Contact Analysis".
- Distro: `packages/cellflow-aggregate/` → `packages/cellflow-contact-analysis/`;
  distribution name `cellflow-aggregate` → `cellflow-contact-analysis`; its
  manifest name + command id `cellflow-aggregate.widget` →
  `cellflow-contact-analysis.widget`. Update the root manifest's mirrored entry
  and the `packages/cellflow-core` comment referencing the distro.
- Module-level re-exports in `__init__.py`, all `from cellflow.aggregate_quantification import …` sites, tests, docs, spec cross-refs.

### Umbrella-vs-member disambiguation (the one real knot)

"Contact analysis" also names a single quantifier (the contacts→`.h5`
computation) via `build_contact_analysis` / `read_position_contact_analysis` /
`build_position_contact_analysis` in `contact_analysis/contacts/`. To stop the
umbrella package and one of its members sharing a name:

- The umbrella package takes `contact_analysis`.
- The contacts quantifier stays under `contacts/` and its functions rename to the
  member's own noun: `build_contact_analysis` → `build_contacts`,
  `build_position_contact_analysis` → `build_position_contacts`,
  `ensure_contact_analysis` → `ensure_contacts`,
  `read_position_contact_analysis` → `read_position_contacts`.
- **The `contact_analysis.h5` filename is left unchanged** (on-disk data compat;
  it is the contacts quantifier's output and a fine name for it). `CONTACT_ANALYSIS_RELPATH` keeps its value; only the code around it moves.

**Decision knob (default chosen):** rename the contacts-quantifier *functions* to
`*_contacts`, keep the `.h5` filename. Flip if you'd rather leave the functions
and accept the umbrella/member name echo.

### Verification

`uv run --frozen pytest` green; `ruff` clean (F401 on the moved re-exports);
`npe2 validate` on all manifests; a smoke `import cellflow.contact_analysis` and a
napari-widget instantiation. Grep for residual `aggregate_quantification`
identifiers (expect only the on-disk folder string in `STAGE_DIRS` and
`CONTACT_ANALYSIS_RELPATH`).

---

## Stage 1 — Commit contract (P2): final-output tier

Lifted from `2026-06-24-independent-distros-and-batch-mode-todo.md` §P2. A
**Commit** action promotes a stage's working labels up to a stable, canonical
base-folder name:

    <pos>/2_nucleus/tracked_labels.tif  --Commit-->  <pos>/nucleus_labels.tif
    <pos>/3_cell/tracked_labels.tif     --Commit-->  <pos>/cell_labels.tif

This is the working-vs-final split the status rail needs: numbered stage dirs hold
re-runnable working artifacts; the base-folder `*_labels.tif` are the committed,
downstream-stable outputs.

### Tasks (TDD)

- `_paths.py`: add `nucleus_labels` → `<pos>/nucleus_labels.tif` and
  `cell_labels` → `<pos>/cell_labels.tif`; document them as the **final outputs**
  in the canonical-layout docstring.
- Qt-free `commit_labels(src, dst)`: validate `src` exists and is a label image,
  copy/overwrite-safely (`shutil.copy2`, or a tiff re-encode if dtype
  normalisation is wanted).
- **Commit** action in `nucleus_workflow_widget` and `cell_workflow_widget`:
  disabled until the working file exists; status feedback; the base file becomes
  the stage's "done" signal.
- **Stale detection**: when a stage is re-run after commit (working file mtime >
  committed mtime), surface a `stale` state; do **not** auto-recommit.
- Contact-analysis discovery already defaults `cell_labels`/`nucleus_labels`
  names — point the defaults at the committed base names, keep the
  explicit-name override.
- Tests: `commit_labels` copies + overwrites; widget Commit writes the base file
  and flips status; discovery finds a committed position by default; stale fires
  on re-run.

**Risk:** low–medium, additive; touches two app widgets + `_paths.py` + discovery
defaults. No pipeline stage changes.

---

## Stage 2 — Catalog status rail (the napariTFM port)

### Principle: keep CellFlow's data-model, adopt napariTFM's presentation

CellFlow's catalog backend is **ahead** of napariTFM's and stays: folder-derived
columns bag, canonical identity (`condition`/`experiment_id`/`position_id`) with a
narrow identity key + collision detection, manual constant columns, CSV
load/save, group-separator rows. What we port from `ExperimentsList` is the
**front-end**: custom-widget rows, a per-position status rail, richer staging,
in-header column-name editing, and the small polish items.

### 2a — Rows become custom widgets (the enabling decision)

The current catalog is a `QTableWidget` (item cells) — it can render a text status
column and nothing richer. A live, hover/click-able status rail cannot live in
item cells. So the catalog table is rebuilt as **stacked custom-widget rows**
(napariTFM `ExperimentRow` model): a header row of editable column-name
`QLineEdit`s over data rows, alignment by shared fixed pixel widths, inside a
bounded `QScrollArea`. This is also the substrate the eventual P4 batch driver
needs — built once, not twice.

Preserve from today: multi-row select (Ctrl/Shift) as analysis **scope** (empty
selection = whole catalogue), `Remove selected` / `Clear` (add a confirm on
Clear), CSV load/save, bold group-separator rows keyed on the columns bag.

### 2b — The four-dot status rail

Per position, four dots, each = "does this artifact exist on disk yet":

| Dot | Stage   | Done artifact                                            | States |
|-----|---------|----------------------------------------------------------|--------|
| 1   | Cellpose| `1_cellpose/{nucleus,cell}_foreground.tif` + `_contours.tif` | 2-state |
| 2   | Ultrack | `nucleus_labels.tif` (committed) ← `2_nucleus/tracked_labels.tif` (working) | 3-state |
| 3   | Cell    | `cell_labels.tif` (committed) ← `3_cell/tracked_labels.tif` (working) | 3-state |
| 4   | Contacts| `aggregate_quantification/contact_analysis.h5`           | 2-state |

**Three-state dots (2 & 3)** consume Stage 1's commit contract and map onto
napariTFM's dot vocabulary:

- empty ring — no working file
- hollow/"ready" — working file exists in the numbered stage dir, not committed
- filled/"done" — committed to the base-folder `*_labels.tif`
- (a `stale` variant — committed but working file is newer — reuses the error/warn dot)

**Two-state dots (1 & 4)** are exists / not. Dot 4 is essentially free: the
catalog already computes `contact_analysis_status` (ready/incomplete) from the
`.h5`.

**Canonical-root guard.** The rail assumes a position's four artifacts share one
`<pos_dir>`. A hand-authored catalog CSV can point `cell_labels`/`nucleus_labels`
at scattered paths with no common root; in that case dots 1 and 4 (the
layout-derived ones) render **grey/"unknown"**, not false-empty. Detection: the
row exposes a canonical `pos_dir` only when its label paths resolve under one
shared position root matching the standard layout.

**Status source (Qt-free, testable):** a pure
`position_stage_status(pos_dir) -> dict[stage, state]` doing header-only file
existence + mtime checks (no pixel decode), mirroring napariTFM's `status_fn` /
`overall_status`. The widget calls it per row on refresh.

### 2c — Ported polish (front-of-app, no pipeline needed)

- **First-class staged preview**: discovered positions become dimmed/italic
  *rows in the same list* (not a separate read-only `QListWidget`),
  individually selectable/removable before commit — kill the current
  all-or-nothing add.
- **In-header column-name editing**: edit column names in the table header,
  directly over their value cells (WYSIWYG), replacing the decoupled per-level
  name editor in the Discover panel. Keep canonical-name routing + `+ Add column`
  manual columns underneath.
- Count/meta line ("N positions"), empty-state collapse, `Delete`/`Ctrl+A`
  keyboard handling, selection lift (accent bar + raised row).

### Explicitly deferred to P4 batch mode (needs the list to *drive* the pipeline)

The full multi-dot **interactive** rail (click-a-dot-to-load-that-stage), the
Run/Cancel toggle, the workers spinbox, and live per-row progress streaming are
batch-execution UX. Stage 2 ships the rail as a **read-only status display**
(dots reflect disk state, refreshed on demand); the interactive/streaming layer
arrives with P4 when the catalog becomes the front-of-app batch driver.

### Verification

Backend `position_stage_status` unit-tested across all state combinations
(missing/working/committed/stale/unknown-root). Widget tests: staged rows
individually removable; header rename carries values table-wide; rail repaints on
refresh; scope selection unchanged; group separators intact. napari smoke: load a
real position tree, confirm the rail matches disk.

---

## Build order

0. Rename (pure refactor, unblocks clean names for everything below).
1. Commit contract (small, independently useful, gives dots 2 & 3 their signal).
2. Catalog rail (custom rows → status rail → ported polish).

Each stage is its own plan + branch → ff-merge → push (per the project's
direct-to-main pattern). `uv run --frozen pytest` throughout (lock is frozen).

## Open decisions (defaults chosen above; flip if wanted)

1. On-disk folder: code-only rename, folder kept behind alias (**default**) vs
   rename folder + `.h5` now with a migration pass.
2. Contacts-quantifier functions: rename to `*_contacts` (**default**) vs leave
   and accept the umbrella/member name echo.
3. Rail scope in Stage 2: read-only status display now (**default**), interactive
   dots + streaming deferred to P4.
