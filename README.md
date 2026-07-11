# CellFlow

CellFlow segments and tracks cells in time-lapse microscopy, corrects the result
by hand where the automatics miss, and quantifies what the tracked cells do. It
is a [napari](https://napari.org) plugin, and it is factored into independent
pieces that share the `cellflow.*` namespace: install the whole pipeline, or
install only the stage your data actually needs.

The rest of this page starts from your data and points you at the right piece.

## Which piece do you need?

Read down the first column until a row describes your data and your goal. Each
piece installs and runs on its own; the full `cellflow` app orchestrates all of
them into one workflow.

| If you have… | Reach for | It gives you |
| --- | --- | --- |
| **Sparse, well-separated cells** with a cell and/or nucleus marker, and you want to segment and track one or both channels | `cellflow-cellpose` | A local Cellpose-SAM runner for native masks, then [`laptrack`](https://github.com/yfukai/laptrack) linking across time, with correction. One channel or two. Every result lands as a napari layer. |
| **Dense, motile cells of varying shape** (a confluent monolayer) and you want to go from raw stacks all the way to quantified contacts | `cellflow[all]` (the full app) | The unified `CellFlow` workflow widget: Cellpose maps, divergence-based cell segmentation, Ultrack nucleus tracking, interactive correction, and contact analysis, end to end. |
| **Foreground and contour maps already** (from Cellpose or any other source) and you want to skip segmentation | `cellflow-tracking` | Ultrack candidate database, track solving, database browsing, and interactive nucleus correction. Maps in, tracked and corrected labels out. |
| **Tracked cell labels already** and you are here for the analysis | `cellflow-aggregate` | Cell-cell edges, border edges, and T1 events extracted to a self-describing HDF5 file, with napari views of the result. |
| **Code to build on** | `cellflow-core` | The shared substrate: TIFF/path/label-IO helpers, the track lineage model, and reusable napari UI primitives. |

The pieces coexist because they share one namespace ([PEP 420](https://peps.python.org/pep-0420/)):
`import cellflow.contact_analysis` works whether you installed `cellflow-aggregate`
alone or the whole app. The heavy engines (Cellpose plus PyTorch, laptrack, the
Ultrack solver) are optional extras, imported only when you run the stage that
needs them, so correction-only use stays light.

> 📷 **Screenshot:** the CellFlow plugin open in napari, the workflow widget
> docked on the right with its stages stacked in run order.

## Status

CellFlow is under active research development. Treat the workflow,
installation details, and public API as provisional. 

## How the pieces fit

CellFlow processes a project directory with **one subdirectory per position**
(`pos00`, `pos01`, …). Within each position, work flows through staged
subdirectories, each stage reading the previous one's plain `.tif` or HDF5 files
off disk:

```text
pos00/
  0_input/                    raw prepared input stacks
  1_cellpose/                 Cellpose probability and flow outputs + divergence maps
  2_nucleus/                  nucleus segmentation, Ultrack database, tracked labels
  3_cell/                     cell segmentation and tracked labels
  aggregate_quantification/   quantification output (contact_analysis.h5 + tables)
```

The full app drives every stage in order; each standalone piece owns one stage
and reads the files on disk, so you can enter the pipeline wherever your data
already sits. The [staged-workflow guide](docs/manual/workflow.md) walks through
what each stage produces.

## Installation

CellFlow requires **Python 3.10+**. Each piece is a standalone tool: install the
one whose job you have (the table above says which), with its engines switched on.

```bash
pip install "cellflow-cellpose[cellpose,laptrack]"  # sparse cells: segment + track
pip install "cellflow-tracking[solve]"              # maps → tracks + correction
pip install cellflow-aggregate                      # tracked labels → contacts (HDF5)
```

To take the whole interactive workflow at once, install the meta-package:

```bash
pip install cellflow[all]
```

**About the `[...]`.** The bracketed names are optional dependency groups the
package switches on, not separate packages. `cellflow-cellpose` needs `[cellpose]`
to segment (this pulls in Cellpose-SAM and PyTorch) and `[laptrack]` to track;
`cellflow-tracking` needs `[solve]` for the Ultrack solver. They are optional
because PyTorch installs differently for every CUDA/CPU setup: install the `torch`
build that matches your machine first, then add the package and pip leaves your
torch alone. Without its engine a piece imports but cannot do its headline job, so
the commands above turn the engines on by default.

`cellflow-core` is the shared library. The other pieces pull it in automatically;
install it alone (`pip install cellflow-core`) only to build on top of it.

From a local checkout, add `-e` and swap the name for `.`, for example
`python -m pip install -e .[all]`. The [installation guide](docs/manual/install.md)
has the full extras matrix.

## Basic usage in napari

Start napari, then open the plugin from the menu:

```bash
napari
# then: Plugins > CellFlow > CellFlow
```

In the main `CellFlow` widget:

1. Select a project directory.
2. Set or load project metadata: pixel size, time interval, condition,
   position. The widget saves and loads `cellflow_config.json` in the project
   directory.
3. Expand the workflow sections in run order: project status, Cellpose, nucleus
   tracking, cell segmentation, contact analysis.

> 📷 **Screenshot:** the metadata panel filled in for one position, with the
> **Save config** button that writes `cellflow_config.json`.

> 📷 **Screenshot:** the contact-analysis view in napari, cell-cell edges drawn
> over the tracked labels with a T1 event highlighted.

## Programmatic use

CellFlow is published primarily as a napari plugin, but the stages are scriptable.
The top-level `import cellflow` exposes only `__version__` and deliberately avoids
importing napari, Cellpose, or Ultrack, so importing the package is cheap.

Import the stage you need directly: `cellflow.segmentation`,
`cellflow.tracking_ultrack`, `cellflow.correction`, and
`cellflow.contact_analysis`, with generic label-stack IO in
`cellflow.core.label_store`. These APIs are useful for scripting and testing, and
are provisional until the public release stabilizes.

## Testing

Install the development dependencies, then run the suite:

```bash
python -m pip install -e .[dev]
python -m pytest
```

On a headless machine, set Qt to offscreen before the napari/Qt tests:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/napari
```

Individual stages test in isolation:

```bash
python -m pytest tests/segmentation
python -m pytest tests/tracking_ultrack
python -m pytest tests/contact_analysis
```

Some tests and workflow stages need the optional packages (Ultrack, Cellpose, or
napari/Qt) and skip when those are absent.

## AI Usage Disclosure

Generative AI tools were used during CellFlow development and documentation
preparation. The tools included OpenAI GPT-5.5 and Anthropic Claude Opus 4.7,
Claude Opus 4.6, and Claude Sonnet 4.6.

AI assistance was used for software-development support, including code
drafting, refactoring suggestions, test scaffolding, debugging assistance,
documentation drafting, and editorial review. Human authors made the core
scientific, architectural, and design decisions. Human authors reviewed,
modified, and validated AI-assisted outputs before incorporating them into the
repository and remain responsible for the accuracy, originality, licensing, and
ethical/legal compliance of the submitted work.

## Citation

If you use CellFlow, cite the software using the metadata in `CITATION.cff`. A
DOI and manuscript citation will be added when the public release and associated
manuscript are ready. For pre-publication citation questions, contact Artur
Ruppel at `artur@ruppel.pro`.

## License

CellFlow is distributed under the AGPL-3.0 license. See the repository-level
`LICENSE` file for the full terms.
