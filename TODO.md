# TODO

## Dimensionality Support

- [ ] Check that the nucleus divergence map path works on 2D, 2Dt, 3D, and 3Dt inputs.
- [ ] Check that the cell divergence map path works on 2D, 2Dt, 3D, and 3Dt inputs.

## Contact Analysis Widget

- [ ] Remove the "clear labels" feature — it's hella slow and useless anyway.
- [ ] Fix visualization bug on contact analysis load: weird ghost layers — non-clickable bars in the layer layout that don't show anything.
- [ ] Add a way to clear individual rows in the contact analysis study, not just the whole thing.
- [ ] CSV files it saves are missing the `.csv` extension.

## Aggregate Quantification (redesign + rename of "Contact Analysis")

- [ ] Rename the "contact analysis" study to **Aggregate Quantification** (`contact_analysis/` → `aggregate_quantification/`).
- [ ] Redesign the interface: this is the stage where you pool all your data (nucleus labels, cell labels, contacts, tracks) and generate the downstream quantities that get plotted — the convergence point between segmentation/tracking and plotting.
- [ ] Broaden scope beyond contacts: nucleus track analysis, nucleus-vs-cell centroid offset, cell shape analysis, tissue dynamics, etc.
