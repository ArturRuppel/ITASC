# ITASC

Segment, track, correct, and quantify cells in time-lapse microscopy, inside napari.

<!-- docs-home-start -->
**📖 [Read the full documentation →](https://arturruppel.github.io/ITASC/)**
<!-- docs-home-end -->

ITASC (Interactive Tracking And Segmentation of Cells) is a
[napari](https://napari.org) plugin for **dense, highly motile cell
monolayers**, where automatic segmentation and tracking break down.
It takes a time-lapse and returns, for every cell, a cell mask and a nucleus
mask that share one ID and hold across the whole recording, then quantifies
their shape, dynamics, and topology.

It gets there by deciding outlines and links together rather than frame by
frame: it feeds [Cellpose](https://github.com/MouseLand/cellpose)'s raw
probability and flow maps into [Ultrack](https://github.com/royerlab/ultrack)'s
candidate solver, tracks the compact nuclei first and grows each cell body
outward from its nucleus (so a cell and its nucleus carry one ID), then turns
the result over to you: Ultrack's alternatives become one-click fixes, with
manual redraw tools adapted from
[EpiCure](https://github.com/Image-Analysis-Hub/Epicure) for the rest.

<!-- hero-start -->
<p align="center">
  <img src="docs/_static/napari_timelapse_last.png"
       alt="ITASC tracking a monolayer in napari" width="100%">
</p>
<!-- hero-end -->

## What it does

The full ITASC app moves through four stages. Each writes its result to a project
folder on disk, and the next stage reads it back. The folder is the source of
truth and a run can be inspected or resumed between any two stages.

- **Segment.** [Cellpose-SAM](https://github.com/MouseLand/cellpose) finds the
  cells. On sparse, well-separated cells it outlines them correctly, and ITASC
  takes those outlines as the cell masks. On a dense monolayer it does not, so
  ITASC ignores the outlines and works from Cellpose's two raw outputs instead:
  the **probability** map, how cell-like each pixel looks, and the **flow**
  field, the direction from each pixel toward the center of the cell it sits in.
  From these it builds the two images the Track stage needs: a foreground map
  and a contour map. Every later stage reads those maps, never the raw stack.
- **Track.** [Ultrack](https://github.com/royerlab/ultrack) builds many candidate
  outlines per frame and selects the set that is most consistent in time, solving
  the outlines and the links at once. That is what a dense monolayer needs.
  Sparse cells do not need it, and [LapTrack](https://github.com/yfukai/laptrack)
  links them frame to frame instead.
- **Correct.** No solver is perfect on dense, dividing cells, so a person fixes
  what it missed. The candidates Ultrack already built are offered as selectable
  alternatives, so most fixes are a click rather than a redraw, and manual
  redraw tools adapted from
  [EpiCure](https://github.com/Image-Analysis-Hub/Epicure) cover the rest.
- **Quantify.** Per position: which cells touch, the edges they share, and the T1
  events where two neighbors swap partners, written to one self-describing HDF5
  (`.h5`) file. Across the project: the shape and dynamics of nuclei, cell
  bodies, and contacts over time, pooled into `.csv` tables.

ITASC ships as five separate installs: four napari tools, and the library they
share. You install one, not all of them. Pick the row that matches the data you
have; each links to that tool's guide, which covers installing and using it.

| If you have… | Install | It gives you |
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
for how to set up a development environment and send a change. For questions
about the method itself, or about whether ITASC suits your system, contact Artur
Ruppel at `artur@ruppel.pro`.

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
