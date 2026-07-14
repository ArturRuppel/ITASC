# ITASC

Segment, track, correct, and quantify cells in time-lapse microscopy, inside napari.

ITASC — Interactive Tracking And Segmentation of Cells — takes raw time-lapse
stacks to tracked, quantified cells. It segments each frame, links cells across
time, lets you correct the result by hand where the automatics miss, and
measures what the tracked cells do. It is a [napari](https://napari.org) plugin
built for dense, motile monolayers, where segmentation and tracking are the hard
part.

<!-- hero screenshot: the ITASC workflow widget docked in napari, its stages
     stacked in run order over a tracked monolayer. Pending capture. -->

## What it does

An ITASC run moves through four stages, each usable on its own:

- **Segment** each frame with [Cellpose-SAM](https://github.com/MouseLand/cellpose).
  For sparse, well-separated cells its masks are the result; for a dense
  monolayer its probability and flow output feeds a divergence-based
  segmentation that separates crowded, variable-shape cells.
- **Track** across time: [Ultrack](https://github.com/royerlab/ultrack) for
  dense monolayers, [LapTrack](https://github.com/yfukai/laptrack) for sparse,
  well-separated cells.
- **Correct** tracks and labels interactively, with editing tools adapted from
  [EpiCure](https://github.com/Image-Analysis-Hub/Epicure), where the automatic
  result is wrong.
- **Quantify** cell-cell contacts, T1 transitions, shape, and dynamics, written
  to a self-describing HDF5 file.

## How it is organized

ITASC is one pipeline you can install whole or in parts: pull the full app, or
install only the stage your data needs. The stages hand off through files on
disk: a project is a directory with one subfolder per position, and each stage
reads the `.tif` and HDF5 files the previous stage wrote. You can enter the
pipeline wherever your data already sits.

## Install

```bash
pip install itasc[all]
```

That installs the full interactive app. To install a single stage on its own,
with the engine it needs, see the
[installation guide](docs/reference/install.md).

## Built on

ITASC reuses the published methods of four tools. If you use the stage that
depends on one, cite it:

- **Cellpose-SAM** (segmentation): Pachitariu M, Rariden M, Stringer C.
  *Cellpose-SAM: superhuman generalization for cellular segmentation.* bioRxiv
  (2025). [doi:10.1101/2025.04.28.651001](https://doi.org/10.1101/2025.04.28.651001)
  · [MouseLand/cellpose](https://github.com/MouseLand/cellpose)
- **Ultrack** (dense tracking): Bragantini J, Lange M, Royer L. *Large-scale
  multi-hypotheses cell tracking using ultrametric contour maps.* ECCV (2024).
  [arXiv:2308.04526](https://arxiv.org/abs/2308.04526)
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
- [API reference](docs/reference/api/index.md): the programmatic API, generated
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
the scientific, architectural, and design decisions, and reviewed and validated
all AI-assisted output before it entered the repository.
