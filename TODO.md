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

## Aggregate Quantification (redesign + rename of "Contact Analysis")

- [ ] Rename the "contact analysis" study to **Aggregate Quantification** (`contact_analysis/` → `aggregate_quantification/`).
- [ ] Redesign the interface: this is the stage where you pool all your data (nucleus labels, cell labels, contacts, tracks) and generate the downstream quantities that get plotted — the convergence point between segmentation/tracking and plotting.
- [ ] Broaden scope beyond contacts: nucleus track analysis, nucleus-vs-cell centroid offset, cell shape analysis, tissue dynamics, etc.
