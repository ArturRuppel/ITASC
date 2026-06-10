# TODO

## Bugs

- [ ] Database building: cancelling doesn't work.

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

# 

- [ ] Pipeline Files loading: `atoms.tif` should load as a **labels** layer,
  not a grayscale image. In `napari/widgets.py`, `_infer_load_kind` doesn't
  recognize `atoms.tif` (it isn't `tracked_labels.tif`/`foreground_masks.tif`/
  `*_labels.tif` and has no "labels" in the name), so it falls through to the
  generic `"tiff"` kind and `_load_file_into_viewer` calls
  `viewer.add_image(..., colormap="gray")`. Atoms are integer atom IDs — they
  should go through `add_labels`.
- [ ] While here, audit the other load paths for appropriate layer types and
  colormaps: `_infer_load_kind` + `_pick_colormap` in `napari/widgets.py`,
  plus the direct `add_image`/`add_labels` calls in the other widgets that
  load pipeline outputs (`nucleus_pipeline_widget.py`,
  `nucleus_atom_extraction_widget.py`, `cellpose_widget.py`,
  `cell_workflow_widget.py`, `cell_correction_widget.py`,
  `divergence_maps_widget.py`, `main_widget.py`). Confirm label images use
  `add_labels` and intensity/scalar images use `add_image` with a sensible
  colormap (e.g. divergence maps likely want a diverging colormap, not gray).

- [ ] Atom extractor: do the checkbox situation like the cell segmentation widget's
  preview. Put the checkboxes in a row along the top and only compute what is
  checked (skip computing unchecked items entirely, rather than computing
  everything and just toggling visibility).

