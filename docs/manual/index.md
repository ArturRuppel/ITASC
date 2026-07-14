# Choosing your install

ITASC is one pipeline, factored into independently installable pieces that
share the `itasc.*` namespace ([PEP 420](https://peps.python.org/pep-0420/)).
The full plugin orchestrates them into one workflow, but each piece installs and
runs on its own. Which one you want is decided by your data and your needs. The table below will help you choose.

| If you have… | Reach for | It gives you |
| --- | --- | --- |
| **Dense, motile cells of varying shape** (a confluent monolayer), from raw stacks to quantified contacts | [Full app](full-app.md) | The unified `ITASC` workflow widget, every stage end to end. |
| **Sparse, well-separated cells** with a cell and/or nucleus marker, and you want to segment and track one or both channels | [itasc-cellpose](cellpose.md) | A local Cellpose-SAM runner for cellpose-powered cell segmentation, then `laptrack` linking across time and tools for manual correction of tracks and masks (adapted from [EpiCure](https://github.com/Image-Analysis-Hub/Epicure)). One channel or two. |
| **Foreground and contour maps already** and you want to skip the cellpose step | [itasc-tracking](tracking.md) | Ultrack candidate database, solving, browsing, and interactive segmentation and tracking correction. |
| **Tracked cell labels already** and you are here for the aggregate quantification | [itasc-aggregate](aggregate.md) | Cell-cell edges, border edges, and T1 events to HDF5, aggregate quantification to `.csv`. |
| **Code to build on** | [itasc-core](core.md) | TIFF/path/label-IO helpers, the lineage model, and napari UI primitives. |

The optional extras (`[cellpose]`, `[laptrack]`, `[solve]`) pull in heavy engines
(Cellpose plus PyTorch, laptrack, the Ultrack solver). They are imported lazily,
so a correction-only session never loads them, and you can drop the extra when
you only need to browse or correct existing results.


[Installation](install.md) for the dependency and extras matrix.
