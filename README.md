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

- **Segment** each frame with Cellpose-SAM. For sparse, well-separated cells
  its masks are the result; for a dense monolayer its probability and flow
  output feeds a divergence-based segmentation that separates crowded,
  variable-shape cells.
- **Track** across time with the Ultrack solver for dense monolayers, or
  laptrack for sparse, well-separated cells.
- **Correct** tracks and labels interactively, where the automatic result is
  wrong.
- **Quantify** cell-cell contacts, T1 transitions, shape, and dynamics, written
  to a self-describing HDF5 file.

## How it is organized

ITASC is one pipeline, factored into independently installable pieces that
share the `itasc.*` namespace
([PEP 420](https://peps.python.org/pep-0420/)): install the whole app, or
install only the stage your data needs. The stages talk to each other through
files on disk, not Python calls: a project is a directory with one subfolder
per position, and each stage reads the `.tif` and HDF5 files the previous stage
wrote. You can enter the pipeline wherever your data already sits.

Which piece fits your data, and how the stages compose, is laid out in
[Choosing a distribution](docs/explanation/choosing-a-distribution.md) and
[The staged workflow](docs/explanation/staged-workflow.md).

## Install

```bash
pip install itasc[all]
```

That installs the full interactive app. To install a single stage on its own,
with the engine it needs, see the
[installation guide](docs/reference/install.md).

## Documentation

- [User guide](docs/index.md): install, the staged workflow, and driving the
  plugin.
- [API reference](docs/reference/api/index.md): the programmatic API, generated
  from the source.

## Status

ITASC is under active research development. Treat the workflow, installation
details, and public API as provisional until the public release and
accompanying manuscript stabilize.

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
