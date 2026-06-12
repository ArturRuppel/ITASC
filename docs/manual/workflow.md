# The staged workflow

CellFlow expects a project directory containing one directory per position
(`pos00`, `pos01`, …). Within each position, work flows through staged
subdirectories — each owned by one distribution and consumed by the next as
plain `.tif` / HDF5 files on disk.

```text
pos00/
  0_input/             raw prepared input stacks
  1_cellpose/          Cellpose probability and flow outputs + divergence maps
  2_nucleus/           nucleus segmentation, Ultrack database, tracked labels
  3_cell/              cell segmentation and tracked labels
  4_contact_analysis/  contact-analysis HDF5 output
```

## Typical run

1. **Provide input** stacks under `0_input/`.
2. **Cellpose** (`cellflow-cellpose`): run the nucleus and cell channels to
   create probability, flow, and z-average TIFFs under `1_cellpose/`, then build
   divergence-based foreground/contour maps.
3. **Nucleus tracking** (`cellflow-tracking`): build the Ultrack candidate
   database, solve tracks, and correct/validate nucleus labels under
   `2_nucleus/`.
4. **Cell segmentation** (`cellflow-segmentation`): turn cell foreground/contour
   maps plus tracked nucleus seeds into tracked cell labels under `3_cell/`.
5. **Contact analysis** (`cellflow-aggregate`): export contacts, edges, and T1
   events to HDF5 under `4_contact_analysis/` and inspect them in napari.

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
