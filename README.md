# ITASC

Segment, track, correct, and quantify cells in time-lapse microscopy, inside napari.

ITASC (Interactive Tracking And Segmentation of Cells) takes raw time-lapse
stacks to tracked, quantified cells. It segments each frame, links cells across
time, lets you correct the result by hand where the automatics miss, and
measures what the tracked cells do. It is a [napari](https://napari.org) plugin
built for dense, motile monolayers, where segmentation and tracking are often difficult.

<!-- hero screenshot: the ITASC workflow widget docked in napari, its stages
     stacked in run order over a tracked monolayer. Pending capture. -->

## What it does

The full ITASC app moves through four stages:

- **Segment** each frame with [Cellpose-SAM](https://github.com/MouseLand/cellpose).
  For sparse, well-separated cells its masks are the result; for a dense
  monolayer its probability and flow output becomes divergence images that
  separate crowded, variable-shape cells and form the input the tracker runs on.
- **Track** across time: [Ultrack](https://github.com/royerlab/ultrack) for
  dense monolayers, [LapTrack](https://github.com/yfukai/laptrack) for sparse,
  well-separated cells.
- **Correct** tracks and labels interactively, with editing tools adapted from
  [EpiCure](https://github.com/Image-Analysis-Hub/Epicure), where the automatic
  result is wrong.
- **Quantify** what the tracked cells do, in two outputs:
  - cell-cell contacts and edges, identified and tracked through T1 transitions,
    written to a self-describing HDF5 (`.h5`) file.
  - aggregate analysis of tracked nuclei, cell bodies, and cell-cell contacts
    (shape and dynamics over time), exported to `.csv` tables.

ITACS is also distributed as smaller, simplified napari tools for a subset of these steps.
Pick the one that matches the data you have. Each row links to that tool's guide,
which covers how to install it and how to use it.

| If you have… | Reach for | It gives you |
| --- | --- | --- |
| **Dense, motile cells of varying shape** (a confluent monolayer), from raw stacks to quantified contacts | [itasc\[all\]](docs/manual/full-app.md) | The unified `ITASC` workflow widget, every stage end to end. |
| **Sparse, well-separated cells** with a cell and/or nucleus marker, to segment and track one or both channels | [itasc-cellpose](docs/manual/cellpose.md) | A local Cellpose-SAM runner for segmentation, then `laptrack` linking across time, plus manual correction of tracks and masks (adapted from [EpiCure](https://github.com/Image-Analysis-Hub/Epicure)). One channel or two. |
| **Foreground and contour maps already**, to skip the cellpose step | [itasc-tracking](docs/manual/tracking.md) | Ultrack candidate database, solving, browsing, and interactive segmentation and tracking correction. |
| **Tracked cell labels already**, and you want the aggregate quantification | [itasc-aggregate](docs/manual/aggregate.md) | Cell-cell edges, border edges, and T1 events to HDF5, aggregate quantification to `.csv`. |
| **Code to build on** | [itasc-core](docs/manual/core.md) | TIFF/path/label-IO helpers, the lineage model, and napari UI primitives. |

## Built on

ITASC reuses the published methods of four tools. If you use the stage that
depends on one, please cite it:

- **Cellpose-SAM** (segmentation): Pachitariu M, Rariden M, Stringer C.
  *Cellpose-SAM: superhuman generalization for cellular segmentation.* bioRxiv
  (2025). [doi:10.1101/2025.04.28.651001](https://doi.org/10.1101/2025.04.28.651001)
  · [MouseLand/cellpose](https://github.com/MouseLand/cellpose)
- **Ultrack** (dense tracking): Bragantini J, et al. *Ultrack: pushing the
  limits of cell tracking across biological scales.* Nature Methods (2025).
  [doi:10.1038/s41592-025-02778-0](https://doi.org/10.1038/s41592-025-02778-0)
  · [royerlab/ultrack](https://github.com/royerlab/ultrack)
- **LapTrack** (sparse tracking): Fukai YT, Kawaguchi K. *LapTrack: linear
  assignment particle tracking with tunable metrics.* Bioinformatics 39(1),
  btac799 (2023). [doi:10.1093/bioinformatics/btac799](https://doi.org/10.1093/bioinformatics/btac799)
  · [yfukai/laptrack](https://github.com/yfukai/laptrack)
- **EpiCure** (correction tools): Letort G. *EpiCure: a versatile and handy tool
  for curation of epithelial segmentation.* bioRxiv (2026).
  [doi:10.64898/2026.03.27.714683](https://doi.org/10.64898/2026.03.27.714683)
  · [Image-Analysis-Hub/Epicure](https://github.com/Image-Analysis-Hub/Epicure)

## Documentation

- [User guide](docs/index.md): install, the staged workflow, and driving the
  plugin.
- [API reference](docs/api/index.md): the programmatic API, generated
  from the source.

## Status

ITASC is approaching its first public release and JOSS submission. The four
stages and the file-based project layout are settled and in active research use.
Installation and the public API are close to final: expect small changes before
the release and its accompanying manuscript.

## Citing ITASC

Cite the software using the metadata in [`CITATION.cff`](CITATION.cff). A DOI
and manuscript citation will be added with the public release. For
pre-publication citation questions, contact Artur Ruppel at `artur@ruppel.pro`.

## License

AGPL-3.0. See [`LICENSE`](LICENSE).

## AI usage

Generative AI tools (OpenAI GPT and Anthropic Claude) assisted with code
drafting, refactoring, tests, debugging, and documentation. Human authors made
the scientific, architectural, and design decisions, and are fully responsible
for all code and other content in the repository.
