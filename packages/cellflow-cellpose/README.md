# cellflow-cellpose

Independent CellFlow piece for the **Cellpose stage** — the shared upstream step
that turns raw input stacks into the maps every downstream stage consumes. It
runs a local Cellpose-SAM model to produce per-channel probability/flow (`prob`,
`dp`) stacks, then builds divergence-based **foreground** and **contour** maps
from them. Those foreground/contour `.tif` maps are the inputs to both nucleus
tracking (`cellflow-tracking`) and cell segmentation (`cellflow-segmentation`).

## Install

```bash
pip install cellflow-cellpose             # divergence-map building from prob/dp
pip install "cellflow-cellpose[cellpose]" # + the Cellpose-SAM model runner
```

This pulls in `cellflow-core`. Both install into the shared `cellflow.*`
namespace (PEP 420), so `import cellflow.cellpose` works whether or not the full
CellFlow orchestrator is present. The Cellpose model itself (`cellpose`, `torch`,
`torchvision`) is the optional `[cellpose]` extra — it is imported lazily and is
only needed to *generate* the `prob`/`dp` stacks. Building the foreground/contour
maps from precomputed `prob`/`dp` does not require it.

## Use

- **napari plugin:** add the *Cellpose* widget. In standalone mode it exposes its
  own input/output pickers:
  - **Nucleus channel** — raw nucleus stack `.tif`
  - **Cell channel** — raw cell stack `.tif`
  - **Output dir** — receives the Cellpose `*_prob_3dt.tif` / `*_dp_3dt.tif`
    stacks and the derived `*_foreground.tif` / `*_contours.tif` maps

  Preview/run Cellpose per channel, then build the divergence maps in the same
  panel.

- **Headless / scripting:**

  ```python
  from cellflow.cellpose import build_divergence_maps

  # From precomputed Cellpose prob/dp stacks → foreground + contour maps:
  build_divergence_maps(
      "out/nucleus_prob_3dt.tif",
      "out/nucleus_dp_3dt.tif",
      "out/nucleus_contours.tif",
      "out/nucleus_foreground.tif",
  )
  ```

## I/O contract

- **Input:** raw 2D+t (or 3D+t) `nucleus` / `cell` stacks under `0_input/` (full
  workflow) or any chosen files (standalone).
- **Output (per channel):** `*_prob_3dt.tif`, `*_dp_3dt.tif` (Cellpose), and
  `*_foreground.tif`, `*_contours.tif` (divergence maps). In the full workflow
  these land in `1_cellpose/`; standalone they land in the chosen output dir.
