# Aggregate capstone: napari front-end integration (beat 1)

*Design — 2026-07-12*

## Problem

The cross-position **aggregate** step — pool every processed position's
per-position outputs into project-level, label-agnostic tidy tables — currently
has no home in the unified CellFlow napari app. It runs from the standalone
`cellflow-aggregate` CLI / notebooks, and from a legacy interactive **studio**
dock (`contact_analysis_studio.py`) that predates the "aggregate departs napari /
thin front-end" decisions and is registered separately from the main app.

Meanwhile the main app (`napari/main_widget.py`) already has everything the
aggregate needs sitting one layer away: an `ExperimentsPanel` that discovers and
catalogs many position folders (records carrying `contact_analysis_path`, the two
label paths, and folder-derived columns) and per-position workflow sections
(Cellpose, Nucleus, Cell, and a per-position **Contact Analysis** section that
produces each position's `contacts.h5`).

What's missing is the **capstone**: a project-level step that consumes that
catalog and pools everything the user has processed. This design adds it.

## Goal & non-goals

**Goal (beat 1).** Give the aggregate step a home in the main app: a project-level
"Aggregate" section that pools the ready positions into the flat tidy tables and
reports where they landed. The napari widget is a *thin front-end* that authors
the two reproducible artifacts (`catalog.csv`, `config.toml`) and drives the
existing headless engine (`cellflow.contact_analysis.pipeline.run`) — the same
entry point the CLI uses. napari is the primary way users run aggregate; the CLI
remains the identical engine underneath for headless/batch.

**Non-goals (explicitly out of scope for beat 1):**

- **Curation.** Curation is a per-position judgement made while looking at a
  position, so it belongs at the end of the per-position Contact Analysis section,
  *not* on the capstone. Tracked as a separate TODO
  (`TODO.md` → "Fold curation into the per-position Contact Analysis section").
  The capstone merely consumes the resulting `curation` table via the run-config's
  default `curation` key; it does not author it.
- **Plotting.** Plots live in Iris. The capstone produces tidy tables and points
  at the output folder; it renders no figures.
- **Building per-position outputs.** Pool-only (see below): the capstone never
  runs the heavy per-position contact build.

## Decisions (from brainstorming)

1. **napari is the front-end, the CLI/pipeline is the engine.** The widget authors
   `config.toml` and calls `pipeline.run(config)`. Same "thin front-end over a
   headless engine" split already used per-position.
2. **The capstone is a project-level step**, distinct from the per-position
   Contact Analysis section. It operates on the whole catalog, not the selected
   position. It sits at the **bottom** of the section stack as the bookend to the
   `ExperimentsPanel` at the top.
3. **Pool-only.** The capstone pools positions whose `contacts.h5` already exists;
   it does not build missing per-position outputs. Per-position work stays in the
   per-position section.
4. **Not-all-ready → pool the ready subset and report the omissions by name.** No
   hard block, no silent drops. Aggregate incrementally as positions come online.

## How pool-only coexists with `run(config)`

`pipeline.run(config)` internally calls `build_quantities` before `aggregate`.
For the `contacts` producer, `build_quantities` calls `ensure_contacts`, which is
**idempotent**: an existing `contacts.h5` is loaded, not recomputed. So if the
authored `catalog.csv` contains only positions whose `contacts.h5` already exists,
`run(config)` does no heavy per-position work — it is load-and-pool. Pool-only is
therefore enforced *by the catalog the widget authors* (ready subset only), while
still calling the CLI's own single entry point. No new engine code is needed.

## Architecture

```
ExperimentsPanel ──catalog records (all positions)──┐
   (top of app, multi-position)                     │
                                                     ▼
                                         AggregateWidget  (new, bottom of app)
                                           1. readiness: contacts.h5 exists?  →  "9 of 12 ready"
                                           2. Run:
                                                a. records → ready subset
                                                b. save_catalog(project/catalog.csv, ready)
                                                c. write_config(project/config.toml, out_dir=root)
                                                d. thread_worker: pipeline.run(config.toml)
                                                e. report tables + skipped positions + Iris pointer
```

The widget owns no compute. It composes existing headless surfaces:
`catalog.save_catalog`, `config.write_config`, `pipeline.run` — all already the
CLI's building blocks.

### New component: `AggregateWidget`

A single lean widget (target: a few hundred lines), placed as a
`CollapsibleSection` at the bottom of the main-app section stack. Responsibilities:

- **Readiness readout.** For each catalog record, a file-existence check on
  `contact_analysis_path` (`contacts.h5`). Renders a plain line — e.g.
  *"9 of 12 positions analyzed — 3 not yet ready"* — and can name the not-ready
  positions. A one-line subtitle names the scope discontinuity explicitly:
  *"Pools every processed position into project-level tables."* No heavy work
  hides behind the readout.
- **Project folder.** Defaults to the positions' common ancestor (the same folder
  `out_dir` defaults to, where `catalog.csv` + `config.toml` naturally sit
  together so the project is relocatable). Overridable.
- **Run** (on a napari `thread_worker`, matching the per-position section's
  threading pattern):
  1. Build catalog records from the live `ExperimentsPanel`, filtered to the ready
     subset. Reuses the existing `_catalog_record_for_position` stamping in
     `main_widget` (committed output paths + columns).
  2. `save_catalog(project/catalog.csv, ready_records)`.
  3. `write_config(project/config.toml, catalog="catalog.csv", out_dir=...)`.
     `quantities` empty ⇒ all; `curation` default key left as-is.
  4. `pipeline.run(project/config.toml, progress_cb=…)` — build loop is a no-op
     load over present `contacts.h5`, then pool.
  5. Emit results.
- **Done state.** A small files-tracker (like the other sections) listing the
  written table CSVs; a status line reporting the table paths, the **skipped
  positions by name**, and a "plots live in Iris" pointer to the output folder.

### Wiring into `main_widget`

- Instantiate `AggregateWidget`, wrap in a `CollapsibleSection` with a capstone
  accent, append **after** the per-position `contact_analysis_section` in the
  section stack.
- Feed it the `ExperimentsPanel` (as the catalog source) and the existing
  `_catalog_record_for_position` record builder, so it reads exactly the catalog
  "Save project" would write.
- The capstone's readiness readout refreshes on the same signals that already
  repaint stage status when a position's outputs change (a contact-analysis run
  completing already refreshes the files tracker — hook the readout to that).

## Cleanup: retire the legacy studio (verify-first)

The new capstone supersedes the multi-position role of the legacy studio. The
studio is **not** used by `main_widget`; it is returned by the plugin factory in
`contact_analysis_widget.py` (`try: import ContactAnalysisStudioWidget … except
ImportError: standalone ContactAnalysisWidget`). Retiring it entails, in order,
each step verified before the next:

1. Confirm no live path other than that factory reaches the studio.
2. Repoint (or remove) the factory so the full-package plugin no longer serves the
   interactive studio dock (the main app is now the aggregate front-end).
3. Remove `contact_analysis_studio.py`, `contact_analysis_run_area.py`,
   `contact_analysis_params.py`.
4. Remove/adapt `tests/napari/test_run_area.py`, `tests/napari/test_shared_params.py`.

If any step surfaces a live dependency the capstone doesn't yet cover, stop and
fold that coverage into the design rather than deleting. This cleanup is in scope
but strictly gated on the capstone covering the studio's aggregate role.

## Error handling

- **Empty catalog / no ready positions.** Run is a no-op with a clear message
  ("No analyzed positions to pool"); nothing is written.
- **Engine raises** (bad catalog row, missing input, pooling error). The
  `thread_worker` surfaces the exception to a status line via `show_error`; no
  partial "done" state is claimed. Matches the per-position section's handling.
- **Path/relocatability.** `catalog.csv` + `config.toml` are written into the same
  project folder with relative paths (via `write_config`), so the project moves
  together.

## Testing

- **Readiness logic.** Given a mixed catalog (some records with an existing
  `contacts.h5`, some without), the readout reports the correct ready/total split
  and names the not-ready positions. Pure, Qt-free.
- **Run authoring.** Run writes `catalog.csv` (ready subset only) + `config.toml`
  with `out_dir` at the common ancestor; `load_config(written)` round-trips.
- **Engine integration** (small fixture project, ≥2 ready positions): Run produces
  the expected flat tidy-table CSVs at `out_dir` and the reported skipped set
  equals the not-ready positions.
- **Qt-test bootstrap gotcha:** construct the Qt app via napari's `get_qapp()`, not
  a bare `QApplication([])` (the latter aborts under pytest).

## Rollout

1. `AggregateWidget` + unit tests (readiness, authoring) — no `main_widget` change
   yet.
2. Wire into `main_widget`; engine-integration test on a fixture project.
3. Retire the legacy studio (verify-first steps above); adapt/remove its tests.
4. Update `TODO.md` — check off the napari front-end refocus for the aggregate;
   the studio removal closes the "interactive studio" line.
