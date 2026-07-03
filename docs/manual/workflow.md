# The staged workflow

CellFlow expects a project directory containing one directory per position
(`pos00`, `pos01`, …). Within each position, work flows through staged
subdirectories — driven end to end by the full `CellFlow` app, with each stage
consuming the previous one's plain `.tif` / HDF5 files on disk.

```text
pos00/
  0_input/             raw prepared input stacks
  1_cellpose/          Cellpose probability and flow outputs + divergence maps
  2_nucleus/           nucleus segmentation, Ultrack database, tracked labels
  3_cell/              cell segmentation and tracked labels
  aggregate_quantification/  quantification output (contact_analysis.h5 + tables)
```

## Typical run

1. **Provide input** stacks under `0_input/`.
2. **Cellpose** (full `cellflow` app): run the nucleus and cell channels to
   create probability, flow, and z-average TIFFs under `1_cellpose/`, then build
   divergence-based foreground/contour maps.
3. **Nucleus tracking** (`cellflow-tracking`): build the Ultrack candidate
   database, solve tracks, and correct/validate nucleus labels under
   `2_nucleus/`.
4. **Cell segmentation** (full `cellflow` app): turn cell foreground/contour
   maps plus tracked nucleus seeds into tracked cell labels under `3_cell/`.
5. **Aggregate quantification** (`cellflow-contact-analysis`): export contacts, edges, and
   T1 events to HDF5 under `aggregate_quantification/` and inspect them in napari.

## Driving the napari plugin

After a full install, start napari and open the plugin:

```bash
napari
# then: Plugins > CellFlow > CellFlow
```

In the main `CellFlow` widget:

1. Select a project directory.
2. Set or load project metadata (pixel size, time interval, condition,
   position). The widget saves/loads `cellflow_config.json` in the project
   directory.
3. Expand the workflow sections in order: project status, Cellpose, nucleus
   tracking, cell segmentation, contact analysis.
