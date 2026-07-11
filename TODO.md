# TODO

## Lean-aggregate-stage deferred follow-ups

(The catalog-CSV slim and the `4_contact_analysis` rename shipped 2026-07-11 — see
[Lean Aggregate Stage] in memory. In-panel table editing for the project catalog was
evaluated and **dropped** by Artur 2026-07-11: the `ExperimentsPanel` table is a bespoke
row-widget list with row-level selection, so cell-level editing / rectangular paste would
mean either hand-rolling a spreadsheet engine or migrating to `QTableView` + a status-rail
delegate — not worth it; the existing Apply-tag flow stays.)

The lean-aggregate-stage change deliberately scoped itself to the **headless pipeline**:
`run()` persists only `contacts` and pools the cheap quantities in memory. These UI/cleanup
follow-ons were deferred:

The lean-aggregate-stage spec
(`docs/superpowers/specs/2026-07-11-lean-aggregate-stage-design.md`) deliberately
scoped itself to the **headless pipeline**: `run()` persists only `contacts` and pools
the cheap quantities in memory. These UI/cleanup follow-ons were deferred:

- [ ] **Strip per-position pooled-quantity builds from the interactive studio.**
  `BuildArea` (`napari/studio_plugins.py`, live in the standalone `cellflow-aggregate`
  front-end) still shows per-position "built X/N" coverage and can build-and-persist a
  pooled quantity per position — the exact stray file the lean change removes from the
  canonical path. Reframe pooled (non-producer) quantities as pooled-only (no
  per-position build/coverage); only `contacts` keeps a per-position "built" badge.
  Rides with the "aggregate front-end refocus" item above.
- [ ] **Delete the now-dead per-position writers/readers** once the studio no longer
  needs them: the pooled quantifiers' `build` / `read` / `object_table` /
  `default_output_name`, plus `shape/core.py` `build_object_shape` / `write_table_csv`
  / `write_provenance`, `quantifiers/_tidy_table.py`, and the
  `_contacts_derived.persist` / `read_derived_table` re-exports (grep-verify no
  remaining caller; `contacts`' own writer stays). Blocked on the item above.
- [ ] **Run-level provenance for the pooled tables.** Per-position provenance sidecars
  vanish with the per-position files, so the "all quantifiers write provenance JSON"
  item below is superseded *at the per-position grain*; if provenance is still wanted,
  emit it once per aggregate run alongside the pooled tables.

Note: `reduce.py` and `plotting.py` were **removed** (2026-07-11, per Artur) — the
package produces aggregate tidy tables only; reduction and plotting are downstream
(Iris / data repo). Any per-CSV custom grouping/collapse levels are that downstream
layer's concern, keyed off the project CSV's metadata columns.

## Dimensionality Support

- [ ] Check that the nucleus divergence map path works on 2D, 2Dt, 3D, and 3Dt inputs.
- [ ] Check that the cell divergence map path works on 2D, 2Dt, 3D, and 3Dt inputs.

## Aggregate Quantification (quantifier seam)

- [ ] All aggregate quantifiers should write the provenance JSON, not just some of
  them. (Make provenance emission part of the quantifier seam so every registered
  quantifier produces it uniformly, rather than the current ad-hoc subset.)
- [ ] Broaden scope beyond contacts: nucleus track analysis, nucleus-vs-cell centroid
  offset, cell shape analysis, tissue dynamics, etc. (Each a new `quantifiers/*.py`
  module + napari visualizer; the seam is what makes them additive. Cell shape and
  track dynamics already shipped — remaining follow-ons: per-position shape overlay,
  physical units, PRW/Fürth MSD fit, turning-angle/arrest, motion overlays.)

## Aggregate Quantification: napari front-end + curation consolidation

The CLI engine (config-driven `run()`, Iris-only export, analysis subpackage) is now
canonical. napari is refocused from an interactive studio into a thin front-end +
curator for that engine; all plotting moves to Iris. Each item has its own spec.

- [ ] **napari front-end refocus** — remove the in-napari interactive plot panels
  (`plot_panel.py`, `plots/`, dynamics/shape plot UI); keep/refocus discover&add +
  run as the engine driver. Iris owns all plotting.
  → `docs/superpowers/specs/2026-06-22-aggregate-napari-frontend-refocus-design.md`
- [ ] **NLS classification → CLI engine step** — make headless NLS classification an
  optional config-flagged pipeline step like every other step; drop the napari NLS UI.
  → `docs/superpowers/specs/2026-06-22-aggregate-nls-classification-cli-step-design.md`
- [ ] **Curation exclusion table + filter** — new curation tidy table (frame/position
  + reason) left-joined onto the measurement tables to filter; the `.iris` export gets
  the filtered data.
  → `docs/superpowers/specs/2026-06-22-aggregate-curation-exclusion-table-design.md`
- [ ] **Curation tool (napari)** — browse positions, scrub frames with contact-viz as
  the overlay, mark frame/whole-position excluded + reason, writes the exclusion table.
  → `docs/superpowers/specs/2026-06-22-aggregate-curation-tool-napari-design.md`
- [ ] **Fold curation into the per-position Contact Analysis section** — curation is a
  per-position judgement made while looking at that position, so it belongs at the end of
  the per-position pipeline (the Contact Analysis section), not on the all-positions
  aggregate capstone. The capstone just consumes the resulting `curation` table via the
  run-config. (Decided 2026-07-12 while scoping the aggregate front-end integration.)

## TIFF Calibration (pixel size / Z step / frame interval)

Background: the project's TIFF writers (all via `core/tiff.py::imwrite_grayscale`)
never embedded physical calibration. The calibration exists in
`0_input/run_params.json` (`pixel_size_um`, `time_interval_s`) and per-position
`cellflow_config.json`. Chosen format: **OME-TIFF** (`PhysicalSizeX/Y/Z`,
`TimeIncrement`). The migration tool (`scripts/embed_calibration.py`), the reader
patch (`pixel_size_from_tiff`), and pos00 conversion are done; what remains:

- [ ] **Convert remaining positions** pos01–pos13 (same command per folder; each
  reads its own `run_params.json` — verify each `pixel_size_um` before batching).
- [ ] **Wire calibration into the writers** so *fresh* pipeline runs emit calibrated
  TIFFs (extend `imwrite_grayscale` to accept/embed pixel size + Z + frame interval;
  decide how calibration reaches the deep writers — ambient/context vs explicit
  threading). This was the original request; only the migration is done so far.

## Track-dynamics statistics

- [ ] **Investigate MSD / DAC SEM over overlapping-origin samples.** The 2026-07-02
  review flagged that `ensemble_msd` (`dynamics/msd.py`) and `_dac_sem`
  (`dynamics/kinematics.py`) report `SEM = std(ddof=1)/√N` over overlapping-origin lag
  samples, which are strongly autocorrelated — so `N` overcounts the independent-sample
  count and the persisted error bars are systematically too small at large lags (the
  means D/α/persistence are unaffected). The right correction is a methodology choice:
  look at how other tools handle it (e.g. `trackpy` `emsd`/`imsd`, `msdanalyzer`, the
  Michalet & Berglund localization-precision MSD papers, block-averaging /
  non-overlapping windows vs. an autocorrelation-time effective-N) before changing the
  published error bars.

## Maintainability follow-ups (from 2026-07-03 review)

- [ ] **Decide intent on the test-only surface** — for each function in
  `tracking_ultrack` / `validation_state` / `core/lineage` with no production caller:
  export it from an `__init__.py` if it's public scripting API, or delete it with its
  test if it's an accidental leftover. Skip anything that is a test seam (a synchronous
  entry point tests use to drive async code, e.g. `_run_blocking`).
- [ ] **Fold the four correction controllers into `napari/correction/`**
  (`lineage_canvas_controller`, `track_path_controller`, `validated_overlay_controller`,
  `candidate_gallery_controller`) — after deciding where `track_path_controller`'s use
  from `contact_visualization.py` should sit.
- [ ] **Behavior-touching correction refactors** — a shared coordinator base for the
  nucleus/cell correction widgets (they already `# Mirror` each other), plus a
  dead-method sweep of `nucleus_correction_widget.py` (vulture flags several private
  methods; each needs a per-method Qt/signal check first).
- [ ] **Optionally broaden the ruff ruleset** (`B,SIM,PERF,RUF`) so the linter catches
  more automatically; add a CI guard that fails a `def`/`class` whose only
  non-definition references are under `tests/`.
