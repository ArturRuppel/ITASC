# cellflow-cellpose

Standalone **Cellpose segment + track** tool. It works on **one or two channels**:

- **One channel** → run a local Cellpose-SAM model for **native masks**, then link
  them across time with [`laptrack`](https://github.com/yfukai/laptrack) into
  tracked labels — a self-contained "segment then track" product.
- **Two channels** → **joint** mode (and only joint mode): the first channel is
  the anchor (segmented + tracked); the second channel's foreground is flowed onto
  it, giving one second-channel object per anchor object, sharing its track id.

It does not need the rest of the CellFlow pipeline. Every result is added to the
napari viewer as a layer; **there is no output directory** — you save whichever
layers you want via napari's own *Save Selected Layers*. There is no
nucleus/cell vocabulary; conventionally Channel 1 is the nucleus and Channel 2
the cell, but nothing assumes it.

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

- **napari plugin:** add the *Cellpose Segment + Track* widget. Each channel
  takes its input from **either** a file (`…` browse button) **or** an image
  already open in the viewer (`⧉` load-from-layer button) — **no layout to declare**:
  - **Channel 1** — the anchor stack (typically the nucleus)
  - **Channel 2** — optional second stack (typically the cell)

  Every plane is segmented individually, so the input shape needs no `2D/2D+t/3D/
  3D+t` selection. Axis identity only matters for tracking, and is inferred: the
  last two axes are `Y, X`; of the remaining leading axes the **shorter is `Z`**
  and the longer is **time** (the preview status shows the inferred `T`/`Z`).

  **Channel 1** carries the single-channel pipeline: **Preview** (▷) runs native
  masks on the current frame so you can tune diameter/min-size/gamma first;
  **Segment** (▶) runs Cellpose native masks over the whole stack; **Track** (⊳)
  links them **axis-by-axis** — it stitches the `z` axis by overlap (so an object
  spanning planes becomes one), then tracks time by motion with laptrack
  (max-distance / frame-gap, in the *Channel 2 & tracking parameters* section).
  Results land as layers tagged `[Channel 1]` (`… masks`, `… tracked`,
  `… preview`); save them via napari.

  **Channel 2** is never segmented on its own — it is always run *jointly*. Once a
  second channel is set its two actions mirror Channel 1's: **Preview** (▷) runs
  the joint assignment on the current frame so you can tune the Channel-2 params
  first, and **Run** (▶) commits it over the whole stack. Either way Channel 1 is
  segmented + tracked, then each Channel-2 foreground pixel is flowed along
  Cellpose's flow field (blended with a pull toward the nearest Channel-1 object)
  and assigned to one. You get **one Channel-2 object per Channel-1 object, sharing
  its track id** — `[Channel 1] tracked` and `[Channel 2] tracked` are paired by
  construction (Channel 2 is tracked by inheriting Channel 1's tracks, not a
  separate tracker). *Channel 2 & tracking parameters*: **Diameter/Min size/Gamma**
  shape its Cellpose flow field; **FG threshold** (foreground cutoff on the
  sigmoid), **Flow weight** (Cellpose flow vs. pull-to-anchor) and **Max assign
  radius** (foreground farther than this from any anchor is left unassigned) drive
  the assignment; **Max distance / Max frame gap** tune the Channel-1 tracker that
  both the joint anchor and Channel 1's own Track action run.

  The embedded **Correction** panel (the basic, ultrack-free cell corrector)
  edits whatever **Labels** layer is currently active — typically `[Channel 2]
  tracked` — in place: contour extend/carve, fill-holes and stranded-fragment
  cleanup. It targets 2D+t (single-Z) labels; save the corrected layer via napari.

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

- **Input:** any 2-D..4-D source per channel — a `.tif` (folder button) or an
  image layer already open in the viewer (load-from-layer button). There is no
  required `0_input/` layout and **no layout to declare**. Every plane is
  segmented individually; for tracking the shorter leading axis is read as `Z`,
  the longer as time. Channel 1 is required; Channel 2 is optional (and turns the
  run into joint mode).
- **Output:** napari **layers**, not files. Masks/tracked/preview are added as
  `int32` Labels layers tagged `[Channel 1]` / `[Channel 2]` (singleton-Z squeezed
  to `(T, Y, X)` for 2D+t data); the user saves them with napari's *Save Selected
  Layers*. The headless API above still returns plain `(T, Z, Y, X)` arrays.
