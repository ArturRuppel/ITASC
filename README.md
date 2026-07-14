# ITASC

Segment, track, correct, and quantify cells in time-lapse microscopy, inside napari.

ITASC (Interactive Tracking And Segmentation of Cells) takes raw time-lapse
stacks to tracked, quantified cells. It segments each frame, links cells across
time, lets you correct the result by hand where the automatics miss, and
measures what the tracked cells do. It is a [napari](https://napari.org) plugin
built for dense, motile monolayers, where segmentation and tracking are often difficult.

Dense epithelial monolayers are where automatic segmentation and tracking break
down: cells are crowded, they change shape frame to frame, and they slide past
one another, so a fully automatic pipeline leaves errors that corrupt every
measurement built on top of the tracks. ITASC pairs strong automatic methods
(Cellpose-SAM, Ultrack) with the interactive correction such data demands, and
holds the corrected labels and the quantities derived from them in one project
folder. The effort a monolayer needs is spent once, at the point of correction,
and carried through to the numbers.

<!-- hero-start -->
<p align="center">
  <img src="docs/_static/napari_timelapse_last.png"
       alt="ITASC tracking a monolayer in napari" width="100%">
</p>
<!-- hero-end -->

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

ITASC is also distributed as smaller, simplified napari tools for a subset of these steps.
Pick the one that matches the data you have. Each row links to that tool's guide,
which covers how to install it and how to use it.

| If you have… | Reach for | It gives you |
| --- | --- | --- |
| **Dense, motile cells of varying shape** (a confluent monolayer), from raw stacks to quantified contacts | [itasc\[all\]](https://arturruppel.github.io/ITASC/manual/full-app.html) | The unified `ITASC` workflow widget, every stage end to end. |
| **Sparse, well-separated cells** with a cell and/or nucleus marker, to segment and track one or both channels | [itasc-cellpose](https://arturruppel.github.io/ITASC/manual/cellpose.html) | A local Cellpose-SAM runner for segmentation, then `laptrack` linking across time, plus manual correction of tracks and masks (adapted from [EpiCure](https://github.com/Image-Analysis-Hub/Epicure)). One channel or two. |
| **Foreground and contour maps already**, to skip the cellpose step | [itasc-tracking](https://arturruppel.github.io/ITASC/manual/tracking.html) | Ultrack candidate database, solving, browsing, and interactive segmentation and tracking correction. |
| **Tracked labels for a set of positions** (cell and/or nucleus), to quantify and pool them | [itasc-aggregate](https://arturruppel.github.io/ITASC/manual/aggregate.html) | Contact analysis per position (cell-cell edges, border edges, T1 events to HDF5), then aggregate quantification pooled across the project to `.csv`. Partial data is fine. |
| **Code to build on** | [itasc-core](https://arturruppel.github.io/ITASC/manual/core.html) | TIFF/path/label-IO helpers, the lineage model, and napari UI primitives. |

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

- [User guide](https://arturruppel.github.io/ITASC/): install, the staged workflow, and driving the
  plugin.
- [API reference](https://arturruppel.github.io/ITASC/api/index.html): the programmatic API, generated
  from the source.

## Status

ITASC is approaching its first public release and JOSS submission. The four
stages and the file-based project layout are settled and in active research use.
Installation and the public API are close to final: expect small changes before
the release and its accompanying manuscript.

## Contributing and support

Bug reports, questions, and pull requests are welcome. Open an
[issue](https://github.com/ArturRuppel/ITASC/issues) to report a problem or ask a
usage question (label it `question`), and see
[`CONTRIBUTING.md`](https://github.com/ArturRuppel/ITASC/blob/main/CONTRIBUTING.md)
for how to set up a development environment and send a change. For
pre-publication or scientific questions, contact Artur Ruppel at
`artur@ruppel.pro`.

## Citing ITASC

Cite the software using the metadata in [`CITATION.cff`](https://github.com/ArturRuppel/ITASC/blob/main/CITATION.cff). A DOI
and manuscript citation will be added with the public release. For
pre-publication citation questions, contact Artur Ruppel at `artur@ruppel.pro`.

## License

AGPL-3.0. See [`LICENSE`](https://github.com/ArturRuppel/ITASC/blob/main/LICENSE).

## AI usage

Generative AI tools (OpenAI GPT and Anthropic Claude) assisted with code
drafting, refactoring, tests, debugging, and documentation. Human authors made
the scientific, architectural, and design decisions, and are fully responsible
for all code and other content in the repository.
