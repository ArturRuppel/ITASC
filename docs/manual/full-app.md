# Full app (`itasc[all]`)

The full ITASC plugin runs the whole pipeline in one napari widget: segment,
track, correct, and quantify, in order, without leaving the viewer. Reach for it
when you have dense, motile cells of varying shape and want to go from raw stacks
to quantified contacts in one place.

One idea holds the app together: every stage writes its result to disk, and the
next stage reads that result back. The project folder is the source of truth, so
you can inspect it between stages, hand a single stage to a standalone
distribution, or resume from any point.

## Install

```bash
pip install itasc[all]
```

The `[all]` extra pulls the core scientific stack plus the two workflow engines:
Cellpose-SAM (`cellpose`, `torch`, `torchvision`) and the Ultrack solver. From a
local checkout:

```bash
python -m pip install -e .[all]
```

## The project on disk

ITASC expects a project directory with one subdirectory per position (`pos00`,
`pos01`, …). Within a position, each stage reads the `.tif` or HDF5 files the
previous stage wrote and writes its own into the next folder:

```text
pos00/
  0_input/                    raw prepared input stacks
  1_cellpose/                 Cellpose probability and flow outputs + divergence maps
  2_nucleus/                  nucleus segmentation, Ultrack database, tracked labels
  3_cell/                     cell segmentation and tracked labels
  aggregate_quantification/   quantification output (contact_analysis.h5 + tables)
```

The layout is plain files, so you read the state of a run by looking at the
folder.

> 📷 **Screenshot:** a position folder in a file browser, the five stage
> subdirectories populated after a full run.

## The stages in order

The app runs five stages in order. The name in parentheses is the standalone
distribution that owns the same stage, for running it on other data on its own.

1. **Provide input.** Put the raw prepared stacks under `0_input/`.
2. **Cellpose.** Run the nucleus and cell channels to create probability, flow,
   and z-average TIFFs under `1_cellpose/`, then build the divergence-based
   foreground and contour maps that seed tracking and segmentation.
3. **Nucleus tracking** (`itasc-tracking`). Build the Ultrack candidate database
   from the maps, solve the tracks, then correct and validate the nucleus labels
   under `2_nucleus/`.
4. **Cell segmentation.** Turn the cell foreground and contour maps plus the
   tracked nucleus seeds into tracked cell labels under `3_cell/`.
5. **Aggregate quantification** (`itasc-aggregate`). Extract contacts, edges, and
   T1 events to HDF5 under `aggregate_quantification/`, and inspect them in
   napari.

If your data already sits at a later stage, skip ahead: foreground and contour
maps let you start at step 3, tracked cell labels let you start at step 5.
[Choosing your install](index.md) maps each entry point to its standalone
distribution.

## Drive the plugin

Start napari and open the plugin:

```bash
napari
# then: Plugins > ITASC > ITASC
```

In the main `ITASC` widget:

1. Select a project directory.
2. Set or load project metadata: pixel size, time interval, condition, position.
   The widget saves and loads `itasc_config.json` in the project directory, so
   the metadata travels with the data.
3. Expand the workflow sections in run order: project status, Cellpose, nucleus
   tracking, cell segmentation, contact analysis. Each section acts on the
   current position and writes into its stage folder.

> 📷 **Screenshot:** the workflow widget with the Cellpose section expanded, the
> channel selectors and **Run** button visible.

To run one of these stages on its own data, outside the full app, install that
piece on its own: [Installation](install.md) covers the single-stage
distributions and the engine each one needs.
