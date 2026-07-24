# Cellpose Segment + Track — folder mode

> Add a second, folder-driven mode to the standalone `itasc-cellpose`
> Segment + Track widget, alongside the current layer-bound mode. Reuses the
> full app's *Data folders* panel and the existing segment/track/correct body.

## What this is (and isn't)

The standalone Segment + Track widget today is **layer-bound**: you bind
Channel 1 / Channel 2 to image layers already in the viewer (`⧉` source pills),
segment, track, correct, and save whatever layers you like by hand. That stays
exactly as it is.

Folder mode adds the full app's project-walking convenience *on top of the same
interactive tool*. You point it at a project root, it discovers the position
folders, and you **walk them one at a time** — select a position, its images
load and bind, you segment and correct interactively, then save the masks into
that folder under a name you specified. A per-position dot shows which positions
you've finished.

This is **not** unattended batch. There is no headless run-all; every position
is segmented and corrected by a human, exactly as in layer mode. The folder list
buys navigation and persistence, nothing more. (Unattended batch across a project
is a separate, larger surface — the app-level P4 "Flexible batch mode" — and is
out of scope here.)

## The toggle

A single control at the top of the widget switches the **I/O header** only; the
entire body below (Channel 1/2 param sections, segment/track/preview actions, the
embedded `CellCorrectionWidget`) is shared and untouched.

- **Source: Layers** — today's behavior. Inputs from the source pills, outputs
  saved by hand.
- **Source: Folders** — the *Data folders* panel below. Inputs from the selected
  position's files (auto-loaded and auto-bound), outputs written to that folder.

Label it by what it binds (Layers vs Folders), not by mode number. Flipping to
Layers and back **keeps the discovered list alive** — the list is project state,
not mode state.

## The Data folders panel (folder mode)

Adapted from the full app's *Data folders* Setup panel, with two changes:

- **Drop** Pixel Size and Frame Length.
- **Four path fields** instead of two image fields — each a relative name under a
  position folder, doing double duty as the discovery pattern *and* the save
  target:
  - `Channel 1 input` (e.g. `0_input/ch1.tif`) — required
  - `Channel 2 input` (e.g. `0_input/ch2.tif`) — optional
  - `Channel 1 output` (e.g. `nucleus_labels.tif`)
  - `Channel 2 output` (e.g. `cell_labels.tif`)

Naming the outputs yourself is what makes the result chain straight into the
aggregate stage, and keeps the channel-agnostic vocabulary — the typing to
`nucleus_labels` / `cell_labels` happens here, in the open, not by a hidden
mapping.

Below the fields: `Find data folders…`, `Delete selected`, an `N data folders`
count, and the discovered list. Persist the four field values via QSettings
(`StandalonePathsMixin`).

## Discovery contract

Reuse `_discover_positions(root, input_names)` (already rglobs each relative name,
groups matches by their position folder). Adaptation:

- **Channel 1 required, Channel 2 optional.** Keep only positions whose matched
  set contains the Channel 1 input name. A folder with only a Channel 2 match is
  not a position.
- A position's Channel 2 is bound only when its Channel 2 file sits in that same
  folder; otherwise the position runs single-channel (matching the existing
  joint-vs-single-channel logic in `joint.py` / `segment_channel`).

## Select → load → bind

Selecting a position row:

1. Loads its Channel 1 (and Channel 2, if present) files as image layers.
2. Binds them via the existing `_set_channel_layer(which, layer)` seam — the same
   entry point the source pills use — so the body below is none the wiser about
   which mode fed it.

This mirrors the app's `refresh(pos_dir)` drive.

## Save contract

A save action writes the tracked masks for the current position to its folder,
under the `Channel 1 output` / `Channel 2 output` names. Writes both channels
when both were produced; Channel 1 only otherwise. On success, flip that row's
dot to **saved**.

## Status dot

One dot per position (segment + track is a single stage, so the app's four-stage
rail collapses to one), rendered by a single-stage `StatusRail`. Three states:

- **missing** — no output file on disk for this position.
- **loaded, unsaved** — this position is the active one and has been worked but
  not yet saved (so a half-corrected position never reads as done).
- **saved** — the output file(s) exist on disk.

## Settled decisions

- Discovery requires Channel 1, Channel 2 optional. **Yes.**
- "Save & advance to next unfinished" convenience button. **No** — out of scope.
- Discovered list persists across the Layers/Folders toggle. **Yes.**

## Reused vs new

Almost entirely assembly of existing primitives:

- `StandalonePathsMixin` (`napari/_standalone_paths.py`) — the four path rows,
  browse handlers, QSettings persistence. Already used by `itasc-tracking` /
  `itasc-segmentation`.
- `_discover_positions` (`napari/main_widget.py`) — discovery, lightly filtered
  for the Ch1-required rule.
- `StatusRail` (`napari/_status_rail.py`) — configured with a single stage.
- `_set_channel_layer` (`cellpose_segment_track_widget.py`) — the load-and-bind
  seam.

New: the toggle + panel wiring in the segment+track widget, the single-stage
status vocabulary, and the save-to-discovered-folder action.

## Out of scope

- Unattended / headless batch over a project (app-level P4).
- Save-and-next navigation.
- Any change to layer mode or to the full app.
