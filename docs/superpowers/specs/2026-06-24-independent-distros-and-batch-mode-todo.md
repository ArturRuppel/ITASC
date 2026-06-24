# Independent Distros + Batch Mode — Prioritised TODO

> Cross-port of ideas from `Projects/napariTFM` into CellFlow. Source for the
> batch-mode item: napariTFM `docs/superpowers/plans/2026-06-24-experiments-list-at-top.md`
> (the `ExperimentsList`-at-top: one folder list feeding tune / batch / aggregate).

## Hard constraint (read first)

**The integrated CellFlow app and its full pipeline DO NOT MOVE.** Nucleus
tracking (Ultrack), the in-app cell segmentation (`src/cellflow/segmentation/`,
`cell_workflow_widget`), the divergence-map path (cell *and* nucleus), and
aggregate all stay exactly as they are. Every change below lives in the
**independently-shipped distributions** (`packages/*` wheels + their napari
manifests) — except batch mode (P3), which is bolted *on top* of the app, not
*inside* the pipeline.

Corollary: `divergence_maps` keeps feeding the full app. We are **not** removing
it from `src/cellflow/cellpose/`; we are changing what the *standalone* Cellpose
distribution does and ships.

**Exception — P2 (commit/promote) is a deliberate, additive app change.** It adds
a new "final output" tier to the file contract (a Commit button in the app's
nucleus/cell workflow widgets). It does not alter any existing pipeline stage or
output; it only *promotes* an existing working file to a stable canonical name.

Ranked **easiest → hardest** per the request.

---

## P1 — Remove the independent Cell Segmentation distribution (easiest)

**What:** Retire `cellflow-segmentation` as an independently-installable wheel.
It depends on the whole upstream pipeline (it consumes the cellpose
`cell_foreground/contours` maps), so it is useless shipped alone.

**Scope = packaging only. The app keeps `src/cellflow/segmentation/` and the
`Cell Segmentation` stage inside the full app unchanged.**

- [ ] Delete `packages/cellflow-segmentation/` (pyproject, README).
- [ ] Drop the `cellflow-segmentation` napari manifest entry point. Decide
      whether `src/cellflow/segmentation/napari.yaml` (the standalone "Cell
      Segmentation" widget command) is still wanted at all — it only existed to
      back the independent distro. If nothing else references it, remove it too;
      the in-app stage is mounted directly by `main_widget`, not via this
      manifest.
- [ ] Scrub references to the distro: `packages/cellflow-core/pyproject.toml`
      comment (line ~50), `packages/cellflow-cellpose/README.md` (line ~8),
      root `README.md`, any `docs/` install matrix. (`[all]` in the root
      `pyproject.toml` is `cellflow[cellpose,tracking]` — already excludes it,
      so nothing to change there.)
- [ ] Verify nothing in `packages/` build configs force-includes the segmentation
      widgets except the deleted distro: `cell_workflow_widget.py` /
      `cell_correction_widget.py` are app widgets, still needed by the app build.
- [ ] Update any CI / release scripts that build/publish the seg wheel.

**Risk:** low — packaging + docs only, no app code paths touched.

---

## P2 — Commit/promote labels → final-output file contract (easy–medium)

**What:** Add a **Commit** button to the nucleus and cell workflow widgets that
copies the working tracked-labels file up into the **position base folder**,
promoting it to a stable final output:

    <pos_dir>/2_nucleus/tracked_labels.tif  --Commit-->  <pos_dir>/nucleus_labels.tif
    <pos_dir>/3_cell/tracked_labels.tif     --Commit-->  <pos_dir>/cell_labels.tif

This introduces a **working-vs-final** split: the numbered stage dirs hold
re-runnable working artifacts; the base-folder `*_labels.tif` are the committed,
downstream-stable outputs. Aggregate then defaults its discovery to these names.

**Scope:** integrated app (additive). Does not change any pipeline stage; it only
copies an existing file to a canonical name on demand.

### Tasks (TDD)
- [ ] `_paths.py`: add `nucleus_labels` → `<pos_dir>/nucleus_labels.tif` and
      `cell_labels` → `<pos_dir>/cell_labels.tif` properties; document them in the
      canonical-layout docstring as the **final outputs**.
- [ ] Backend: a small Qt-free `commit_labels(src, dst)` helper (copy +
      overwrite-safely, validate src exists & is a label image). Reuse the tiff
      writer for a re-encode rather than a raw byte copy if dtype normalisation is
      wanted; otherwise `shutil.copy2`.
- [ ] `nucleus_workflow_widget`: a **Commit** action that promotes
      `2_nucleus/tracked_labels.tif` → `nucleus_labels.tif`; status feedback;
      disabled until the working file exists. Mirror in the data-status surface
      (the base file becomes the "done" signal).
- [ ] `cell_workflow_widget`: same for `3_cell/tracked_labels.tif` → `cell_labels.tif`.
- [ ] Aggregate: make `nucleus_labels.tif` / `cell_labels.tif` the **default**
      `nucleus_name` / `cell_name` in `discover_catalog_entries` (or the
      catalog-builder caller), so a committed position is auto-discovered. Keep
      the explicit-name override.
- [ ] Tests: `commit_labels` copies + overwrites; widget Commit writes the base
      file and flips status to done; aggregate discovery finds a committed
      position by default.

**Risk:** low–medium — additive, but it touches two app widgets, `_paths.py`, and
aggregate's discovery defaults; needs care that re-running a stage doesn't
silently desync the committed copy (surface "working file newer than committed"
as a `stale` status, don't auto-recommit).

**Cross-links:** the P3 standalone Cellpose distro should be able to commit its
laptrack output to `cell_labels.tif` the same way; P4 batch mode consumes the
committed base files as each experiment's per-position outputs.

---

## P3 — Repurpose the Cellpose distro: native masks + laptrack (medium) — ✅ DONE

**Implemented (Option A — new modules + standalone widget; app untouched):**
- `src/cellflow/cellpose/native_masks.py` — captures Cellpose native masks the
  runner discarded; nucleus (3D or per-z) + cell (per-z) → `(T,Z,Y,X)` labels +
  `write_masks`. Tests: `tests/cellpose/test_native_masks.py`.
- `src/cellflow/cellpose/track_laptrack.py` — centroid build + relabel + lazy
  `laptrack` linker → track-consistent labels. Tests:
  `tests/cellpose/test_track_laptrack.py` (laptrack call behind importorskip).
- `src/cellflow/napari/cellpose_segment_track_widget.py` — standalone two-row
  (Nucleus/Cell) widget, explicit file pickers, Segment + Track per channel,
  **flat** output `{channel}_masks.tif` / `{channel}_tracked.tif` (no
  `0_input`/`1_cellpose`). Tests: `tests/napari/test_cellpose_segment_track_widget.py`.
- Distro repackaged: `napari.yaml` → *Cellpose Segment + Track*; `[laptrack]`
  extra; new-widget force-include (kept `cellpose_widget`/`divergence_maps_widget`
  shipped for the app). App's `CellposeWidget` + divergence path unchanged.

Original plan below (kept for reference).

### (original) P3 — Repurpose the Cellpose distro: native masks + laptrack (medium)

**What:** As an *independent* tool, `cellflow-cellpose` currently only emits the
divergence foreground/contour maps — useless standalone. Make the **standalone
distro** instead: (a) generate **Cellpose native masks** (the runner already
computes them and throws them away — `_, flows, _ = model.eval(...)` in
`cellpose_runner.py`), then (b) link them across time with a **simple laptrack
tracker** → tracked labels. That makes the distro a genuinely useful
"segment + track" product on its own.

**Scope:** additive new modules + a standalone widget/manifest for the distro.
**Must not alter the full app's Cellpose stage or its divergence outputs** — the
app keeps mounting the existing `CellposeWidget` + divergence path. The new
native-mask + laptrack surface is shipped/exposed by the *distro*, not wired
into the integrated pipeline.

### Design decision to settle first (short brainstorm/spec)
The distro and the app share `src/cellflow/cellpose/` and `cellpose_widget.py`.
To keep the app unchanged, decide the seam:
- **Option A (recommended):** new Qt-free backend modules
  (`cellflow/cellpose/native_masks.py`, `cellflow/cellpose/track_laptrack.py`)
  + a **new standalone widget** (e.g. `cellpose_segment_track_widget.py`) that
  the distro's `napari.yaml` points to. App's `CellposeWidget` untouched.
- **Option B:** extend `CellposeWidget` with an optional native-mask+track
  section gated off in the app. Higher risk of disturbing the app surface.

### Tasks (TDD, after the seam is chosen)
- [ ] Backend: `native_masks.py` — capture the labelled masks from
      `model.eval` (currently discarded), TZYX-aware like the existing runner,
      write `cell_masks` (and optionally `nucleus_masks`) `.tif`. Add the
      `cellpose` extra already declared by the distro.
- [ ] Backend: `track_laptrack.py` — link per-frame masks into tracked labels
      via `laptrack` (LAP nearest-neighbour). Add `laptrack` to the distro's
      `[project.optional-dependencies]` (lazy import, like Ultrack's `solve`
      extra in `cellflow-tracking`).
- [ ] Widget: standalone segment→track surface (run native masks, run laptrack,
      load results as napari labels). Reuse `StandalonePathsMixin`.
- [ ] Distro packaging: update `packages/cellflow-cellpose/pyproject.toml`
      force-includes (add the new modules/widget), description, README, and
      `napari.yaml` (point/extend the command). Decide whether the standalone
      distro still ships `divergence_maps_widget` or drops it from its surface.
- [ ] Tests: backend (`tests/cellpose/…` native-mask shape + label continuity
      after tracking) and a napari file-contract test for the distro widget.

**Risk:** medium — new backend + a new standalone widget; the constraint is to
keep the shared `CellposeWidget`/divergence behaviour byte-for-byte the same for
the app.

---

## P3.5 — Drop input-format selection; segment per-frame, track axis-by-axis (medium)

**Insight:** the standalone distro does **not** support Cellpose stitching or true
3D mode — so it never needs to *know* the input layout (2D / 2D+t / 3D / 3D+t).
Segmenting "every frame individually" **is** what stitching reduces to anyway:
treat the input as a flat stack of 2D planes and run native masks on each. That
removes the whole format-declaration surface (the `layout` arg, `do_3d`,
`anisotropy`, `cellpose_runner.to_tzyx(…, "2D+t")`, the `_to_tzyx` layout
guessing) from the standalone — the user just points at a `.tif` and we segment
its planes.

**Tracking generalises to follow:** once every plane is segmented independently,
tracking is "link labels along one axis, then the other." For 2D+t that is just
the existing T-linking. For 3D+t it becomes two passes — e.g. link within Z to
make each timepoint's planes into cohesive 3D objects, then link those objects
across T (or T-first then Z). **The second axis needs care:** naive per-axis LAP
can fragment a cell (a label cohesive in plane *k* drifting from plane *k+1*), so
the cross-axis pass needs an overlap/centroid-consistency rule to keep objects
whole, not just nearest-centroid.

**Scope:** standalone Cellpose distro only — same hard constraint as P3. The app's
`CellposeWidget`, divergence path, and `cellpose_runner.to_tzyx`/`do_3d` 3D mode
stay exactly as they are (the app *does* use real 3D). This only simplifies what
the **standalone** widget + its backend expose.

### Tasks (TDD)
- [ ] Backend: a layout-free `segment_planes(stack)` that flattens any input to a
      stack of 2D planes, runs native masks per plane, and returns labels in the
      same shape. Supersedes the standalone's reliance on `to_tzyx(…, layout)` /
      `do_3d`. App's `native_masks` 3D nucleus path is untouched.
- [ ] Widget: remove the format/layout picker and the `do_3d`/`anisotropy`
      params from the standalone surface; the file pickers alone drive it.
- [ ] Tracking: `track_axis_by_axis(labels)` — link along the fastest axis first,
      then the slower one with an **overlap-consistency** merge so per-plane labels
      stay one object. 2D+t collapses to today's single T-link.
- [ ] Tests: per-plane segmentation on 2D / 3D / 3D+t inputs gives plane-wise
      labels with no layout arg; axis-by-axis tracking keeps a synthetic object
      cohesive across both axes; the app's 3D nucleus path is unchanged.

**Risk:** medium — the per-frame segmentation simplification is easy and a clear
win; the **two-axis tracking cohesion** is the real work (don't ship naive
per-axis LAP). Keep it standalone-only so the app's genuine 3D handling is safe.

**Cross-links:** removes the `layout`/`do_3d` surface that P3's widget still
carries; P2's commit-to-`cell_labels.tif` still applies to the tracked output.

---

## P4 — Flexible batch mode (hardest, bolted on top)

**What:** Port napariTFM's `ExperimentsList`-at-top idea. One shared list of
experiment folders feeding three jobs: **tune** (pick one → operate on it),
**batch** (run all → per-experiment outputs), **aggregate** (fold outputs into
the aggregate tables). The batch leg is "the aggregation tool, but **before**
analysis" — it runs the *pipeline* per experiment, where aggregate today only
runs *after*.

This sits **on top of** the app (a new front-of-app surface), not inside the
pipeline. It builds naturally on the existing aggregate machinery
(`aggregate_quantification/pipeline.py` `run(config.toml)`, `catalog`,
`experiment_id` identity, the DAG-scheduler TODO already noted in project
memory) and the single-position main-widget direction
(`project_main_interface_single_position` memory).

### Tasks (expand into a full TDD plan when its turn comes)
- [ ] Spec/brainstorm: define an "experiment" for CellFlow (a position folder?
      a project root?), the per-stage status vocabulary, and what "batch run"
      executes per experiment (which stages, headless vs in-app). Reuse
      napariTFM's mini-rail/overall-status model as the UI reference only.
- [ ] `ExperimentsList`-equivalent widget above the workflow shell in
      `main_widget`: add/remove folders, single-selection (active experiment),
      per-experiment stage status dots, state persistence.
- [ ] Batch run: iterate the list, run the existing pipeline per experiment,
      stream live status into the rows. Reuse the headless pipeline surfaces; do
      not duplicate compute.
- [ ] Aggregate leg: wire the batch outputs into the existing
      `aggregate_quantification` run (the "after" tool) so list → batch →
      aggregate is one flow.
- [ ] Tests: list add/select/persist, batch dispatches the right folders/stages,
      aggregate receives the produced outputs.

**Risk:** high — net-new front-of-app surface + orchestration across many
experiments. Largest of the three; do last.

---

## Open items to confirm during execution
- P1: is `src/cellflow/segmentation/napari.yaml` still wanted once the distro is
      gone, or removed with it?
- P2: copy vs re-encode on commit; how to surface a `stale` committed file when a
      stage is re-run after a commit; do we also keep a non-default escape hatch
      for callers that still point aggregate at the stage-dir working files?
- P3: Option A vs B for the distro/app seam; whether nucleus also gets native
      masks in the standalone distro (default: cell only — nucleus standalone use
      is served by `cellflow-tracking`).
- P3.5: which axis to link first for 3D+t (Z-then-T vs T-then-Z), and the
      cohesion rule for the second pass (IoU overlap vs centroid-graph) so per-plane
      labels don't fragment into separate tracks.
- P4: experiment granularity (position vs project) and headless-vs-in-app batch
      execution.
