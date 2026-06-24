# cellflow-cellpose

Standalone **Cellpose segment + track** tool. It runs a local Cellpose-SAM model
to produce **native masks** per channel, then links those masks across time with
[`laptrack`](https://github.com/yfukai/laptrack) into tracked labels — a
self-contained "segment then track" product that does not need the rest of the
CellFlow pipeline. Every result is added to the napari viewer as a layer; **there
is no output directory** — you save whichever layers you want via napari's own
*Save Selected Layers*.

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
  explicit file pickers — point directly at your files, **no layout to declare**:
  - **Nucleus channel** — raw nucleus stack `.tif`
  - **Cell channel** — raw cell stack `.tif`

  Every plane is segmented individually, so the input shape needs no `2D/2D+t/3D/
  3D+t` selection. Axis identity only matters for tracking, and is inferred: the
  last two axes are `Y, X`; of the remaining leading axes the **shorter is `Z`**
  and the longer is **time** (the preview status shows the inferred `T`/`Z`).

  Per channel: **Preview** (▷) runs native masks on the current frame so you can
  tune diameter/min-size/gamma first; **Segment** (▶) runs Cellpose native masks
  over the whole stack; **Track** (⊳) links them **axis-by-axis** — it stitches
  the `z` axis by overlap (so a cell spanning planes becomes one object), then
  tracks time by motion with laptrack (max-distance / frame-gap in *Tracking
  parameters*). Results land as layers tagged `[Nucleus]` / `[Cell]` (`… masks`,
  `… tracked`, `… preview`); save them via napari.

  **Joint** (⧉) — enabled once *both* a nucleus and a cell input are set — runs a
  nucleus-anchored cell segmentation instead of independent masks: it segments
  and tracks the nuclei, then flows each cell-foreground pixel along Cellpose's
  flow field (blended with a pull toward the nearest nucleus) and assigns it to a
  nucleus. You get **one cell per nucleus, sharing the nucleus' track id** —
  `[Nucleus] tracked` and `[Cell] tracked` are paired by construction (the cell
  is tracked by inheriting the nucleus tracks, not a separate tracker). *Joint
  parameters*: **FG threshold** (cell-foreground cutoff on the sigmoid), **Flow
  weight** (Cellpose flow vs. pull-to-nucleus), **Max assign radius** (foreground
  farther than this from any nucleus is left unassigned). The single-channel
  Segment/Track path above is unchanged.

  The embedded **Correction** panel (the basic, ultrack-free cell corrector)
  edits whatever **Labels** layer is currently active — typically `[Cell] tracked`
  — in place: contour extend/carve, fill-holes and stranded-fragment cleanup. It
  targets 2D+t (single-Z) labels; save the corrected layer via napari.

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

- **Input:** any 2-D..4-D `nucleus` / `cell` `.tif`, picked explicitly — there is
  no required `0_input/` layout and **no layout to declare**. Every plane is
  segmented individually; for tracking the shorter leading axis is read as `Z`,
  the longer as time.
- **Output:** napari **layers**, not files. Masks/tracked/preview are added as
  `int32` Labels layers tagged `[Nucleus]` / `[Cell]` (singleton-Z squeezed to
  `(T, Y, X)` for 2D+t data); the user saves them with napari's *Save Selected
  Layers*. The headless API above still returns plain `(T, Z, Y, X)` arrays.
