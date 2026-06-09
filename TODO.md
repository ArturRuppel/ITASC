# TODO

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

- [ ] Give the NLS classifier batch capabilities. Possible approach: add support for relative paths and for selecting multiple entries. When multiple entries are selected, grey out `measure_classify` and have **Apply to h5** apply the classification to all selected entries.

## Live Previews

- [ ] Cell segmentation live previews got a new caching mechanism. Port it to the other live previews.

## Aggregate Quantification (redesign + rename of "Contact Analysis")

- [x] Rename the "contact analysis" study to **Aggregate Quantification** (`contact_analysis/` → `aggregate_quantification/`). (Full rename incl. classes, manifest, and the standalone dist `cellflow-contact` → `cellflow-aggregate`; the contacts artifact `contact_analysis.h5` + stage stay, as that is the contacts quantifier's own storage.)
- [x] Redesign the interface — the quantifier seam. (Added `Quantifier` registry + `PositionInputs` in `aggregate_quantification/quantifier.py`; contacts is now one registered quantifier (`quantifiers/contacts.py`) that owns its persistence; the studio builds/reads through `available_quantifiers()`, so new quantities plug in without touching it. See `notes/aggregate_quantification_spec.md`.)
- [ ] Broaden scope beyond contacts: nucleus track analysis, nucleus-vs-cell centroid offset, cell shape analysis, tissue dynamics, etc. (Each a new `quantifiers/*.py` module + napari visualizer; the seam above is what makes them additive.)
