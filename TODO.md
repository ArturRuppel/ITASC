# TODO

## Bugs

- [x] Plotter: plot params should not reset when the input field is changed.
  (Changing the **Value** picker in catalog mode rebuilt the group-by checkboxes
  from scratch, silently clearing the user's grouping. `_rebuild_group_checks`
  now carries over which columns were ticked and re-checks any the newly-selected
  product still offers — plot/level/stat/error/bins were already preserved. See
  `plot_panel.py` + `test_plot_panel.py`.)

- [x] Plotter: the CSV export in the plot field should export a CSV that
  corresponds exactly to the data shown in the plot. There should be only **one**
  export button. (The two CSV buttons (Pooled / Aggregated) collapsed into one
  **Plot data (CSV)** that writes exactly what the current plot draws, via a new
  headless `plotting.plotted_table(df, spec)`: the per-unit samples for the
  distribution family (hist/box/violin/strip/swarm), the per-group aggregate for
  bar, the per-group/per-frame series for line, and the `U(x)` curve for
  potential. Re-plottable elsewhere with no further filtering. See
  `plotting.py`, `plot_panel.py`, `test_plotting.py`, `test_plot_panel.py`.)

- [x] Computing/status labels should live in the Compute/Build section, not the
  catalogue section. (Added a `_build_status_lbl` inside the Compute section;
  build queue/progress/done/error messages route through a new `_set_build_status`
  instead of `_set_catalog_status`, so progress reads next to the controls that
  triggered it. See `aggregate_quantification_studio.py`.)

- [x] The parameters section should be expanded by default. (`expanded=True`.)

- [x] Rename the "Build" section to **Compute**. (Section title + the build
  area's docstring/host renamed.)

- [x] Tools, Compute, and Plot sections should be collapsed by default.
  (`expanded=False` for all three.)

- [x] There should be a "check all" button (toggling to "uncheck all") in the
  Compute section. (`BuildArea` got a **Check all** button that ticks every
  *buildable* metric (disabled/no-input rows untouched) and flips to **Uncheck
  all** once all are ticked; the label also follows manual checkbox edits. See
  `studio_plugins.py` + `test_studio_plugins.py`.)

- [x] The collapsible containers in Aggregate Quantification should shrink to
  fit when everything is collapsed. (Verified the current structure already
  reclaims the height — collapsing every section returns the scroll content from
  its expanded minimum (~1590 px with all plugins open) back to the stacked
  headers (~163 px). Locked in with a regression test that expands then collapses
  all sections and asserts the shrink. See `test_aggregate_quantification_studio.py`.)

- [x] napari `IndexError: tuple index out of range` on canvas draw, from
  `_update_world_units` → `VispyBaseLayer._on_matrix_change`: `dims_displayed`
  is `[1, 2]` but `self._world_to_layer_units_scale` is only a 2-tuple
  `(1.0, 1.0)`, so indexing `[2]` overruns. (Root cause is napari's
  `_recalculate_units_scale` `strict=False` zip yielding a units scale shorter
  than the layer's displayed dims when physical units — e.g. an OME-TIFF's
  `(pixel, pixel)` — reach a layer with more dims than the units tuple (a
  calibrated 2D stack shown in a 3D viewer). Added `patch_vispy_units_scale_guard`
  in `_napari_compat.py` (installed alongside the existing layer-delegate patch,
  so it ships everywhere): it pads `_world_to_layer_units_scale` up to the
  displayed-dim extent with `1.0` — the no-calibration default, so the only
  behaviour change is the draw no longer crashes. Pure padding logic
  (`_padded_units_scale`) is unit-tested in `test_napari_compat.py`.)

- [x] Plotter: seaborn swarmplot overflow ("X% of the points cannot be placed").
  Markers should auto-size / fall back to stripplot so points aren't silently
  dropped. (`_plot_swarm` draws the swarm inside a warning-capture; on the
  "cannot be placed" overflow it shrinks the markers once (5 → 3 px) and, if it
  still overflows, falls back to a stripplot — which jitters but draws **every**
  point. Regression test builds a 600-point swarm in a tiny figure and asserts
  all points are drawn with no overflow warning. See `plotting.py` +
  `test_plotting.py`.)

- [x] Plotter: the click-to-highlight ring matches the data point's y but not its
  x for jittered strip/swarm plots. Fix: look up the actual drawn marker's x from
  the scatter offsets. (`_highlight_point` now calls `_marker_x`, which scans the
  axes' scatter offsets and snaps to the one whose y best matches the picked value
  and whose x is closest to the category column — so the ring lands exactly on the
  jittered point; falls back to the category centre for box/violin/hist. See
  `plot_panel.py` + `test_plot_panel.py`.)

- [x] Interactive data showing feature: the "load in viewer" function isn't
  working — the path is shown but the data doesn't load. (The `ClickToLoad`
  controller was a local in `shape.py::_open_panel` /
  `track_dynamics.py::_open_distribution`, wired via
  `panel.load_requested.connect(controller.load)`; PyQt holds bound-method
  connections weakly and nothing else referenced the controller — the resolver
  closure doesn't capture `self` — so it was GC'd as soon as the panel opened
  and the Load click emitted to nobody. `PlotPanel` now takes a `loader`
  callable it holds strongly (symmetric with `target_resolver`), tying the
  controller's lifetime to the panel; both plugins pass `loader=controller.load`.
  Regression-tested with post-GC load clicks in both plugin test files.)
- [x] Database building: cancelling doesn't work. (The Cancel button set a
  `threading.Event`, but `build_atom_union_database` never observed it — the
  event was only checked *between* whole steps in the worker, so a click during
  the heavy per-frame build did nothing until the entire build finished.
  `build_atom_union_database` now takes a cooperative `cancel: Callable[[], bool]`
  and polls it at the top of each frame and each linking step (via the shared
  `_check_cancel` → `CancelledError` idiom used by the cellpose/divergence
  workers); the pipeline widget passes `cancel=cancel_event.is_set`, so Cancel
  now takes effect within one frame. Already-built frames stay committed
  (`engine.dispose()` per frame). Status text updated to "…after the current
  frame…". Regression tests in `test_db_build.py` cover immediate and mid-build
  cancel.)
- [x] Plotter: every consecutive plot gets smaller and smaller.
- [x] Horizontal space can't be made smaller than way-too-large. When docked for
  the first time, the control dock should not shrink — the napari canvas should
  shrink instead. (Opening a plot used to `splitDockWidget` the controls column,
  keeping window width constant and crushing the control panel. Now, on first
  plot, `PlotDockTabs._grow_window_for_plot` widens the whole window by the
  plot's width and `resizeDocks` pins the controls to their existing width so the
  plot fills the new space. The `PlotPanel` content itself now shrinks rather
  than scrolling: combos/line-edits/spins use an `Ignored` width policy via a
  `_shrinkable` helper, the matplotlib nav toolbar spills overflow tools into a
  `»` menu, the **Load in viewer** button is full-width, and exports sit in a
  tidy equal-width "Export:" grid row. Panel min width dropped to ~327 px (the
  floor is the styling section's labelled checkbox rows). See
  `plugins/_plot_dock.py` + `plot_panel.py`.)

- [x] Plotter: `KeyError: 'Column not found: msd_D_um2_per_s'` when plotting a
  per-track MSD value (`msd_D_um2_per_s` / `msd_alpha`) at *Per-track* view.
  Root cause: a **stale** `*_dynamics.h5` built before the per-track MSD fit
  columns existed (`_merge_track_msd`) — the pooled tracks table had 16 cols
  (12 track + 3 metadata + class_label) and no `msd_*`. The track-dynamics
  plugin still advertised them in `_TRACK_VALUES`, so selecting one fell
  through `reduce_to_units`' groupby on a missing column. Fix: `PlotPanel` now
  filters `value_columns` to those present in the snapshot (symmetric with its
  identity-column filter) so absent quantities aren't offered; and the headless
  `build_figure` renders a "No data in scope" placeholder instead of a KeyError
  when a needed value column is missing. Rebuild (Recompute/overwrite) to get
  MSD back on old data. Regression-tested in `test_plotting.py` +
  `test_plot_panel.py`.)

## Dimensionality Support

- [ ] Check that the nucleus divergence map path works on 2D, 2Dt, 3D, and 3Dt inputs.
- [ ] Check that the cell divergence map path works on 2D, 2Dt, 3D, and 3Dt inputs.

## Contact Analysis Widget

- [x] Remove the "clear labels" feature — it's hella slow and useless anyway. (Removed the "Clear Layers" button + handler; internal clear-before-redraw stays.)
- [x] Fix visualization bug on contact analysis load: weird ghost layers — non-clickable bars in the layer layout that don't show anything. (Skip per-frame Edges/T1 Shapes layers whose cache is empty across the whole movie, instead of adding a permanent blank layer.)
- [x] Fix T1 edges rendering bug: endpoints joined by a straight line. (napari dropped the "path" type to closed "polygon" when an emptied Shapes layer was repopulated via the `data` setter; re-add with explicit `shape_type="path"`.)
- [x] Add a way to clear individual rows in the contact analysis study, not just the whole thing. ("Remove selected" button in the catalogue.)
- [x] CSV files it saves are missing the `.csv` extension.

## NLS Classifier

- [x] Give the NLS classifier batch capabilities. Possible approach: add support for relative paths and for selecting multiple entries. When multiple entries are selected, grey out `measure_classify` and have **Apply to h5** apply the classification to all selected entries. (The NLS image field now resolves **relative to each position** (e.g. `0_input/NLS_zavg.tif`), so one entry works across a batch; absolute paths stay verbatim. Selecting >1 position greys *Measure & classify* + the threshold and turns *Apply to H5* into **Classify & apply to all H5**, which auto-thresholds and writes each selected position via `patch_position_contact_analysis_nls_classes` in a worker, surfacing per-position failures without aborting the batch.)

## Live Previews

- [x] Cell segmentation live previews got a new caching mechanism. Port it to the other live previews. (Extracted the cell preview's per-frame cache into a shared `FramePreviewCache` (`napari/_preview_cache.py`): frame `t` → result, keyed on a params signature, with stale-write safety so a late worker can't poison a re-keyed cache. Wired it into the **divergence maps** and **nucleus atom-extraction** live previews — scrubbing back to a computed frame now repaints instantly (even mid-compute) and any param edit drops the cache; freed on deactivate. Covered by `FramePreviewCache` unit tests + a divergence cache reuse/invalidation integration test.)

## Aggregate Quantification (redesign + rename of "Contact Analysis")

- [x] Rename the "contact analysis" study to **Aggregate Quantification** (`contact_analysis/` → `aggregate_quantification/`). (Full rename incl. classes, manifest, and the standalone dist `cellflow-contact` → `cellflow-aggregate`; the contacts artifact `contact_analysis.h5` + stage stay, as that is the contacts quantifier's own storage.)
- [x] Redesign the interface — the quantifier seam. (Added `Quantifier` registry + `PositionInputs` in `aggregate_quantification/quantifier.py`; contacts is now one registered quantifier (`quantifiers/contacts.py`) that owns its persistence; the studio builds/reads through `available_quantifiers()`, so new quantities plug in without touching it. See `notes/aggregate_quantification_spec.md`.)
- [ ] All aggregate quantifiers should write the provenance JSON, not just some of them. (Make provenance emission part of the quantifier seam so every registered quantifier produces it uniformly, rather than the current ad-hoc subset.)
- [ ] Broaden scope beyond contacts: nucleus track analysis, nucleus-vs-cell centroid offset, cell shape analysis, tissue dynamics, etc. (Each a new `quantifiers/*.py` module + napari visualizer; the seam above is what makes them additive.)
  - [x] **Cell shape** — first real new quantity. `quantifiers/cell_shape.py` over a headless `cell_shape/{build,reader}.py` core (regionprops: area, circularity, aspect ratio, eccentricity, solidity, … per cell per frame → `cell_shape.h5`). Added a generic headless plotting backend (`aggregate_quantification/plotting.py`: pool / aggregate / figure / CSV) and a unified **Cell Shape** group plugin (Compute + Plot, with CSV/figure export). Framework deltas: `Quantifier.default_output` + `object_table` on the base; studio build path now routes through `default_output` (was hardcoded to the contacts artifact); `owns_quantities` lets a group plugin suppress the generic auto-builder. See `notes/2026-06-10-cell-shape-quantifier-and-table-explorer-design.md`. Follow-ons: per-position shape overlay, physical units, contacts/NLS plot sections.
  - [x] **Track dynamics** — motion read off the tracked label stacks. Twin quantifiers `quantifiers/{cell,nucleus}_dynamics.py` over a headless `dynamics/` core (trajectories → instantaneous velocities, per-track summary, ensemble MSD power-law `D`/`α`, directional-autocorrelation persistence time, and collective metrics: order parameter, velocity correlation `C(r)`, 1/e correlation length, NN distance → multi-table `*_dynamics.h5`). Added `frame_interval.py` (resolves `time_interval_s` like `pixel_size.py`) + `PositionInputs.time_interval_s`. A **Track dynamics** group plugin (scope cell/nucleus; Compute + Plot with per-frame / per-track distributions via the generic `PlotPanel` and a bespoke MSD/DAC/`C(r)` curves panel). See `notes/2026-06-11-track-dynamics-quantifier-design.md`. Follow-ons: PRW/Fürth MSD fit, per-track D/α, turning-angle/arrest, motion overlays.

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

## TIFF Calibration (pixel size / Z step / frame interval)

Background: the project's TIFF writers (all via `core/tiff.py::imwrite_grayscale`)
never embedded physical calibration — files had `XResolution` 1/1, `ResolutionUnit`
none. The calibration exists in `0_input/run_params.json` (`pixel_size_um`,
`time_interval_s`) and per-position `cellflow_config.json` (often blank). Chosen
format: **OME-TIFF** (`PhysicalSizeX/Y/Z`, `TimeIncrement`) — works for all dtypes
incl. `uint32`/`int32` labels, unlike ImageJ-TIFF.

- [x] Migration tool for existing files: `scripts/embed_calibration.py` (dry-run
  by default, atomic temp-swap writes). Most stacks → OME-TIFF; `atoms.tif` keeps
  its description verbatim (load-bearing `cellflow_atom_params`) and gets baseline
  pixel-size tags only. Relabels the `QYX` (unlabeled-leading-dim) stacks → `TYX`.
- [x] Patched `aggregate_quantification/pixel_size.py::pixel_size_from_tiff` to read
  OME `PhysicalSizeX` (tried before ImageJ/baseline).
- [x] Converted **pos00** of `2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk`
  (0.65 µm/px, 1.0 µm Z, 900 s/frame) — verified data/labels intact, atoms params
  survived, reader returns 0.65. (Folder is a risk-mitigation copy.)
- [ ] **Convert remaining positions** pos01–pos13 (same command per folder; each
  reads its own `run_params.json` — verify each `pixel_size_um` before batching).
- [ ] **Wire calibration into the writers** so *fresh* pipeline runs emit calibrated
  TIFFs (extend `imwrite_grayscale` to accept/embed pixel size + Z + frame interval;
  decide how calibration reaches the deep writers — ambient/context vs explicit
  threading). This was the original request; only the migration is done so far.

# 

- [x] Pipeline Files loading: `atoms.tif` should load as a **labels** layer,
  not a grayscale image. (`_infer_load_kind` now maps `atoms.tif` → `"labels"`,
  so the load button goes through `add_labels` instead of falling through to a
  gray `add_image`.)
- [x] While here, audit the other load paths for appropriate layer types and
  colormaps. (The direct `add_image`/`add_labels` calls in the other widgets
  were already correct — atom extraction loads atoms via `add_labels`; cellpose
  uses viridis/inferno/gray for prob/flow/reference; divergence uses gray for
  foreground + magma for contours; cell workflow/correction load label stacks
  via `add_labels`. Only `_PipelineFileRow` was wrong. Rewrote `_pick_colormap`
  to be filename-semantic and match those viewers: contour/divergence → magma,
  cellpose prob → viridis, flow/dp → inferno, raw input + foreground → gray.
  Note: divergence/contour maps are *positive* ridge signal, so a sequential
  map (magma, as the divergence widget uses) is right — not a diverging one.)

## Cellpose Segment + Track distro

- [x] UI/UX on arrival — progressive button enabling (extend the existing
  `UiGate` `when`/`reason` predicates) + contextual "next step" status-label
  text, rather than a new onboarding widget. (Channel 1's preview/segment now
  gate on `_channel_present(1)`, Track gates on a `[Channel 1] masks` layer
  existing (with a reason callable distinguishing "bind first" from "segment
  first"), and Channel 2's preview/run gate reasons name whichever input is
  still missing. The status label gets a real idle hint at construction and a
  trailing next-step sentence after binding Channel 1 / after Segment
  completes / after Track completes, via two new pure `_segment_done_status` /
  `_track_done_status` helpers (kept Qt-free so they're unit-testable without
  driving the `thread_worker` path). The embedded corrector's ⏻ activate
  button gets a **local** `_sync_active_btn_enabled` (not routed through the
  parent's `UiGate`, since `CellCorrectionWidget` is reused in disk-mode
  contexts where this precondition doesn't apply), wired to a new
  `viewer.layers.selection.events.active` listener; stays enabled once
  checked so a live session can always be toggled back off even if the active
  layer selection moves away. Spec:
  `docs/superpowers/specs/2026-07-01-cellpose-segment-track-arrival-ux-design.md`,
  plan: `docs/superpowers/plans/2026-07-01-cellpose-segment-track-arrival-ux.md`.
  See `cellpose_segment_track_widget.py` + `cell_correction_widget.py` +
  `test_cellpose_segment_track_widget.py`.)

- [x] **Bug: "segment current frame" clobbers all frames when a full-stack run
  was already run.** It should only mutate the active frame. (`_run_segment`'s
  `_done` callback clears `self._stream` to `None` on completion, but the
  masks/prob/flow layers keep the full-stack result. `_preview` re-inits the
  stream on next use via `_init_stream`, which was zero-filling unconditionally
  — its first `_push_stream_layers()` call then overwrote the real layers with
  zeros a beat before the single-frame result was even computed. `_init_stream`
  now seeds each accumulator from the matching viewer layer's existing data
  (via new `_existing_layer_array`, undoing the display-time Z-squeeze) when
  its shape matches, falling back to zeros only when no layer exists or the
  shape doesn't match (e.g. a different stack loaded). See
  `cellpose_segment_track_widget.py` +
  `test_init_stream_after_a_full_run_reuses_layer_data_not_zeros` /
  `test_init_stream_falls_back_to_zeros_when_layer_shape_differs` in
  `test_cellpose_segment_track_widget.py`.)

- [x] **Preview should mutate the real stack in place** ("segment this frame"),
  not spawn separate throwaway layers. `_preview()`
  (`cellpose_segment_track_widget.py`) now lazily initializes the persistent
  `self._stream` accumulator via a new shared `_init_stream()` helper (also used
  by `_run_segment()`) if no full run has happened yet, computes only the
  current frame (`native_masks.run_nucleus_maps_frame`), writes it straight into
  `self._stream["masks"][t, z]` / `["prob"][t, z]` / `["flow"][t, z]`, and
  refreshes via the existing `_push_stream_layers()` — the same `… masks` /
  `… prob` / `… flow` layers the full **Segment** run fills, one slice mutated,
  no separate `… preview` layers. The now-superseded `preview_channel_maps()`
  module function (full-stack zero-alloc-per-click embedding) was removed along
  with its test. Covered by `test_init_stream_matches_the_real_segment_layers`
  + `test_preview_writes_one_slice_without_clobbering_other_frames` in
  `test_cellpose_segment_track_widget.py`.

- [x] **Tracking is much slower than laptrack itself should be.** Profiled on a
  synthetic dense-monolayer stack (150 frames × ~576 cells/frame): the actual
  bottleneck was **not** our glue code but `laptrack`'s own
  `LapTrack.predict_dataframe()` → `laptrack.data_conversion.tree_to_dataframe`,
  which assigns `tree_id`/`track_id` to every tracked object **one at a time**
  via `DataFrame.loc[(frame, index), col] = value` (the library's own source
  even flags a sibling loop with "XXX there may exist faster impl.") — this
  accounted for >80% of total tracking time, dwarfing laptrack's actual linking
  cost. Fixed in `track_laptrack.py::_run_laptrack` by calling the public,
  lower-level `LapTrack.predict()` (returns the raw tracking graph) instead,
  then extracting track ids ourselves via a single `networkx.connected_components`
  pass + dict lookup (`_track_ids_from_tree`, valid because splitting/merging are
  always disabled here — guarded by an assertion in case that ever changes).
  Also vectorized `build_track_dataframe`'s row construction (minor secondary
  win; `regionprops_table` itself is now the dominant remaining cost there).
  Result: 19.3s → 3.6s on the benchmark (5.3x). `relabel_by_tracks` was
  confirmed negligible (~2% of pipeline time) and left as-is.

- [x] **Port validation / clear-not-validated / track-list navigation into the
  correction tool** (`CellCorrectionWidget`, `cell_correction_widget.py`,
  full-editing/standalone mode only — the integrated app's contour-only
  corrector is unchanged). Adapted rather than rebuilt, per plan:
  - **Validation** is in-memory only (`self._validated_tracks: dict[int,
    set[int]]`) rather than `tracking_ultrack/validation_state.py`'s
    JSON-file store — the standalone tool has no project directory to persist
    into. Same `{cell_id: {frames}}` shape as the disk-backed store, so the
    downstream pieces need no adaptation. New ✓ toolbar button / **V** key
    (`_on_toggle_validation`) toggles validation for the selected cell across
    every frame it appears in (`_frames_with_cell`); reset on each fresh
    layer binding (`_toggle_active_layer_correction`) so stale ids from a
    previous session can't collide.
  - **Clear-not-validated** reuses `napari/_correction_utils.py::
    remove_unvalidated_labels` verbatim (already DB-free — a pure
    `(data, validated_tracks)` function) via a new 🗑 toolbar button
    (`_on_remove_unvalidated_labels`), with per-frame undo history recorded
    through the existing `CorrectionWidget._record_history`.
  - **Track-list navigator**: `napari/_correction_track_accordion.py`
    (`TrackAccordionPanel`) was already fully DB-free and reused unmodified.
    `lineage_canvas_controller.py::LineageCanvasController` gained one new
    optional constructor param, `validated_tracks_provider`, so it can pull
    validated/anchored flags from an in-memory dict instead of
    `validation_state.read_validated_tracks(pos_dir)` (anchors, an
    Ultrack-workflow concept, stay empty here) — fully backward compatible,
    the nucleus workflow widget's own wiring is untouched. Embedded directly
    in `active_content` below the correction toolbar. There is no reliable
    intensity backdrop to crop film-strip thumbnails from in standalone mode
    (the active layer is whatever the user bound), so `intensity_layer_provider`
    is `lambda: None` and the thumbnail band stays empty; the per-track
    presence/validated swimlane bars and click-to-jump (`node_activated` →
    `_navigate_to_cell_from_lineage`, jumps the time slider + selects the
    cell) work fully.
  - **Validated-cell overlay in the viewer** (the "green border" — the
    standalone counterpart of `validated_overlay_controller.py`'s border
    mode): `_refresh_validated_overlay()` adds/updates a new
    `[Correction] Validated: Cell` Labels layer, opaque green
    (`contour=2`, `opacity=1.0`), masked straight from
    `self._validated_tracks` (no `pos_dir`/JSON read — that controller's
    disk coupling didn't fit cleanly, so this is a small dedicated
    rewrite rather than an instantiation of it). Removed again once no
    cell is validated. Refreshed after toggle-validate, remove-unvalidated,
    retrack, and on each fresh layer binding.
  - Covered by 9 new tests in `test_cellpose_segment_track_widget.py`
    (toggle validate/invalidate, validate-requires-presence,
    remove-unvalidated removes/no-ops correctly, navigator gated to
    full-editing only, bar-click navigation, overlay layer appears/matches
    the mask/is removed when the last validated cell is invalidated).

- [ ] **Bug: cancelling a batch segmentation run leaves the UI stuck.** Clicking
  Cancel during a full-stack Segment run in the Cellpose Segment + Track distro
  does stop the run, but the other action buttons stay disabled and the
  progress bar freezes instead of resetting to idle.

- [x] Atom extractor: do the checkbox situation like the cell segmentation widget's
  preview. Put the checkboxes in a row along the top and only compute what is
  checked. (The two stage checkboxes (Foreground, Contour) are now a "Compute:"
  row at the top of the params panel and gate *computation*, not just visibility:
  the live-preview worker runs a level system (`_ATOM_LEVEL_FG` = residual_fg +
  territory; `_ATOM_LEVEL_CONTOUR` adds residual_contour + ridge + atoms) sized to
  the max ticked stage, so an unticked stage is skipped entirely and its layers
  removed. Per-frame cache now stores `(level, slices)` and only reuses a frame
  whose stored level covers the desired one. Contour implies the cheap FG compute
  since the atom watershed runs on the foreground territory. ▶ run still computes
  the full stack and now ticks both boxes so every layer shows. Covered by a new
  compute-gating test in `test_nucleus_workflow_standalone.py`.)

- [ ] **Investigate MSD / DAC SEM over overlapping-origin samples.** The
  2026-07-02 review flagged that `ensemble_msd` (`dynamics/msd.py`) and
  `_dac_sem` (`dynamics/kinematics.py`) report `SEM = std(ddof=1)/√N` over
  overlapping-origin lag samples, which are strongly autocorrelated — so `N`
  overcounts the independent-sample count and the persisted error bars are
  systematically too small at large lags (the means D/α/persistence are
  unaffected). The right correction is a methodology choice: look at how other
  tools handle it (e.g. `trackpy` `emsd`/`imsd`, `msdanalyzer`, the Michalet &
  Berglund localization-precision MSD papers, block-averaging / non-overlapping
  windows vs. an autocorrelation-time effective-N) before changing the
  published error bars. Deferred out of the review-fix pass on purpose.

