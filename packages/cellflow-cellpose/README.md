# cellflow-cellpose

Standalone **Cellpose segment + track** tool. It runs a local Cellpose-SAM model
to produce **native masks** per channel, then links those masks across time with
[`laptrack`](https://github.com/yfukai/laptrack) into tracked labels — a
self-contained "segment then track" product that does not need the rest of the
CellFlow pipeline.

> The integrated CellFlow app uses this distribution's Cellpose runner +
> divergence maps for its in-app stage; those modules still ship here and are
> imported directly by the orchestrator. This README covers the **standalone**
> napari surface, which is the segment + track tool.

## Install

```bash
pip install cellflow-cellpose                       # package + Qt-free helpers
pip install "cellflow-cellpose[cellpose]"           # + the Cellpose-SAM model
pip install "cellflow-cellpose[cellpose,laptrack]"  # + the laptrack tracker
```

This pulls in `cellflow-core`. Everything installs into the shared `cellflow.*`
namespace (PEP 420), so `import cellflow.cellpose` works with or without the full
orchestrator. The model (`cellpose`, `torch`, `torchvision`) is the `[cellpose]`
extra and `laptrack`/`pandas` are the `[laptrack]` extra — both imported lazily,
so importing the package does not require them.

## Use

- **napari plugin:** add the *Cellpose Segment + Track* widget. It exposes
  explicit file pickers — point directly at your files, no enforced layout:
  - **Nucleus channel** — raw nucleus stack `.tif`
  - **Cell channel** — raw cell stack `.tif`
  - **Output dir** — receives `{channel}_masks.tif` and `{channel}_tracked.tif`

  Per channel: **Segment** runs Cellpose native masks; **Track** links them over
  time with laptrack (max-distance / frame-gap in *Tracking parameters*).

  After tracking, the embedded **Correction** panel (the basic, ultrack-free cell
  corrector) edits the tracked **cell** labels in place — contour extend/carve,
  fill-holes and stranded-fragment cleanup — and saves back to `cell_tracked.tif`.
  It targets 2D+t (single-Z) cell labels.

- **Headless / scripting:**

  ```python
  import tifffile
  from cellflow.cellpose import cellpose_runner, native_masks, track_laptrack

  stack = cellpose_runner.to_tzyx(tifffile.imread("cell_3dt.tif"), "2D+t")
  params = cellpose_runner.CellParams(diameter=0.0, min_size=0, gamma=1.0)
  masks = native_masks.run_cell_masks_stack(stack, params)       # (T, Z, Y, X)
  tracked = track_laptrack.track_masks(masks, max_distance=15.0)  # tracked labels
  ```

## I/O contract

- **Input:** any multi-dimensional `nucleus` / `cell` `.tif` (2D/2D+t/3D/3D+t),
  picked explicitly — there is no required `0_input/` layout.
- **Output (flat, in the chosen dir):**
  - `{channel}_masks.tif` — Cellpose native masks, `(T, Z, Y, X)` `int32`
  - `{channel}_tracked.tif` — time-linked labels, stable id per track
