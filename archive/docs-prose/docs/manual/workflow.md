# The staged workflow

Every CellFlow run is a chain of stages, and the chain is written to disk. Each
stage reads plain `.tif` or HDF5 files the previous stage wrote and writes its
own into the next subdirectory, so the folder on disk is the source of truth: you
can inspect it, hand a single stage to a standalone piece, or resume from any
point without holding the whole pipeline in memory.

CellFlow expects a project directory with **one subdirectory per position**
(`pos00`, `pos01`, …). Within each position, work flows through numbered stage
folders:

```text
pos00/
  0_input/                    raw prepared input stacks
  1_cellpose/                 Cellpose probability and flow outputs + divergence maps
  2_nucleus/                  nucleus segmentation, Ultrack database, tracked labels
  3_cell/                     cell segmentation and tracked labels
  aggregate_quantification/   quantification output (contact_analysis.h5 + tables)
```

## What each stage does

The full `cellflow` app runs these in order. The name in parentheses is the
standalone piece that owns the stage, if you want to run it on its own.

1. **Provide input.** Put the raw prepared stacks under `0_input/`.
2. **Cellpose** (full `cellflow` app). Run the nucleus and cell channels to
   create probability, flow, and z-average TIFFs under `1_cellpose/`, then build
   the divergence-based foreground and contour maps that seed tracking and
   segmentation.
3. **Nucleus tracking** (`cellflow-tracking`). Build the Ultrack candidate
   database from the maps, solve the tracks, then correct and validate the
   nucleus labels under `2_nucleus/`.
4. **Cell segmentation** (full `cellflow` app). Turn the cell foreground and
   contour maps plus the tracked nucleus seeds into tracked cell labels under
   `3_cell/`.
5. **Aggregate quantification** (`cellflow-aggregate`). Extract contacts, edges,
   and T1 events to HDF5 under `aggregate_quantification/`, and inspect them in
   napari.

If your data already sits at a later stage, skip ahead: foreground and contour
maps in hand let you start at step 3, and tracked cell labels let you start at
step 5. The [install guide](install.md) says which piece to install for each
entry point.

> 📷 **Screenshot:** a position folder in a file browser, the five stage
> subdirectories populated after a full run.

## Driving the napari plugin

After a full install, start napari and open the plugin:

```bash
napari
# then: Plugins > CellFlow > CellFlow
```

In the main `CellFlow` widget:

1. Select a project directory.
2. Set or load project metadata: pixel size, time interval, condition, position.
   The widget saves and loads `cellflow_config.json` in the project directory, so
   the metadata travels with the data.
3. Expand the workflow sections in run order: project status, Cellpose, nucleus
   tracking, cell segmentation, contact analysis. Each section acts on the
   current position and writes into its stage folder.

> 📷 **Screenshot:** the workflow widget with the Cellpose section expanded, the
> channel selectors and **Run** button visible.

With the stages understood, the [install guide](install.md) covers the
dependency and extras matrix for the full app and for each standalone piece.
