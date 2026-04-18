load corrected nuclear labels always loads position 0 it seems

segmentation xwith flow watershed has unused parameters exposed to the ui

when loading something into the napari viewer that is there already, check it's visibility state and copy that state on the new layer

Run in Terminal should launch kitty and not gnome terminal

Prepare Input Data widget: should have a "pull metadata" button to get px and dt from ndtiff. should discover the available positions and allow to export them all. should always export all timepoints. should overwrite pix size and dt when exporting with metadata. when downsampled, pixelsize needs to be calculated accordingly. export all positions at once should be possible. should also have a run in terminal button, like the other widgets. trying to run a position that doesn't exist should yield an error and not produce an empty folder. metadata of ndtiff files, including positions and so on should be discovered anyways. see ndtiff reader napari plugin for reference

Write a quick user guide to get the user oriented quickly. think about how and where to display it.

expose an overview of each widget somewhere. discrete but obvious. to be designed first

UI improvement: expansion and shrinking of widgets is not great. expanding a collapsed section makes the widget larger, good, but uncollapsing doesn't make it shrink again.

parameter handling: remaining: (1) add "Run in Terminal" to the Foreground Mask section in flow_watershed.py — needs a --mode fg-only CLI flag in cellpose/stages/flow_watershed.py first; (2) wire EdgeAnalysisWidget to save h5 output and ForcesWidget to read from it — agree on h5 schema first. Global config (cellflow_config.json), shared log viewer, overwrite-above-run-buttons, and save/load button consolidation are done.

cell segmentation algorithm flow watershed produces artifacts and weird straight lines. previous stochastic process didn't have this problem... but it was much slower, made less sense and was probabilistic so going back is not really an option. investigate and find solution
