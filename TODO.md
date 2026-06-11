# TODO

## Bugs

- [ ] Database building: cancelling doesn't work.
- [ ] Plotter: every consecutive plot gets smaller and smaller.
- [ ] Horizontal space can't be made smaller than way-too-large. When docked for
  the first time, the control dock should not shrink — the napari canvas should
  shrink instead.

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
- [ ] Broaden scope beyond contacts: nucleus track analysis, nucleus-vs-cell centroid offset, cell shape analysis, tissue dynamics, etc. (Each a new `quantifiers/*.py` module + napari visualizer; the seam above is what makes them additive.)
  - [x] **Cell shape** — first real new quantity. `quantifiers/cell_shape.py` over a headless `cell_shape/{build,reader}.py` core (regionprops: area, circularity, aspect ratio, eccentricity, solidity, … per cell per frame → `cell_shape.h5`). Added a generic headless plotting backend (`aggregate_quantification/plotting.py`: pool / aggregate / figure / CSV) and a unified **Cell Shape** group plugin (Compute + Plot, with CSV/figure export). Framework deltas: `Quantifier.default_output` + `object_table` on the base; studio build path now routes through `default_output` (was hardcoded to the contacts artifact); `owns_quantities` lets a group plugin suppress the generic auto-builder. See `notes/2026-06-10-cell-shape-quantifier-and-table-explorer-design.md`. Follow-ons: per-position shape overlay, physical units, contacts/NLS plot sections.
  - [x] **Track dynamics** — motion read off the tracked label stacks. Twin quantifiers `quantifiers/{cell,nucleus}_dynamics.py` over a headless `dynamics/` core (trajectories → instantaneous velocities, per-track summary, ensemble MSD power-law `D`/`α`, directional-autocorrelation persistence time, and collective metrics: order parameter, velocity correlation `C(r)`, 1/e correlation length, NN distance → multi-table `*_dynamics.h5`). Added `frame_interval.py` (resolves `time_interval_s` like `pixel_size.py`) + `PositionInputs.time_interval_s`. A **Track dynamics** group plugin (scope cell/nucleus; Compute + Plot with per-frame / per-track distributions via the generic `PlotPanel` and a bespoke MSD/DAC/`C(r)` curves panel). See `notes/2026-06-11-track-dynamics-quantifier-design.md`. Follow-ons: PRW/Fürth MSD fit, per-track D/α, turning-angle/arrest, motion overlays.

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

