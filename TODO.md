Prepare Input Data widget: should have a "pull metadata" button to get px and dt from ndtiff. should discover the available positions and allow to export them all. should always export all timepoints. should overwrite pix size and dt when exporting with metadata. when downsampled, pixelsize needs to be calculated accordingly. export all positions at once should be possible. should also have a run in terminal button, like the other widgets. trying to run a position that doesn't exist should yield an error and not produce an empty folder. metadata of ndtiff files, including positions and so on should be discovered anyways. see ndtiff reader napari plugin for reference

Write a quick user guide to get the user oriented quickly. think about how and where to display it.

expose an overview of each widget somewhere. discrete but obvious. to be designed first

for resizing the widgets one has to drag an invisible bar. make it visible somehow maybe?

parameter handling should be homogenized, including naming and placement of the buttons. big task. analyze and plan first. remaining: (1) move overwrite checkbox above the run button row in all widgets; (2) add "Run in Terminal" to the Foreground Mask section in flow_watershed.py; (3) wire EdgeAnalysisWidget to save h5 output and ForcesWidget to read from it — agree on h5 schema first; (4) parameter persistence: run trace sidecar JSON + save/load pipeline config — needs schema design (one monolithic JSON vs per-widget, where it lives on disk, which widgets participate).

cell segmentation algorithm flow watershed produces artifacts and weird straight lines. previous stochastic process didn't have this problem... but it was much slower, made less sense and was probabilistic so going back is not really an option. investigate and find solution