Prepare Input Data widget: should have a "pull metadata" button to get px and dt from ndtiff. should discover the available positions and allow to export them all. should always export all timepoints. should overwrite pix size and dt when exporting with metadata. when downsampled, pixelsize needs to be calculated accordingly.

Write a quick user guide to get the user oriented quickly. think about how and where to display it.

expose an overview of each widget somewhere. discrete but obvious. to be designed first

for resizing the widgets one has to drag an invisible bar. make it visible somehow maybe?

parameter handling should be homogenized, including naming and placement of the buttons. big task. analyze and plan first.

finishing a step should default to loading the visualization

flow watershed widget has a bunch of parameters which are not used anymore

foreground.tif should be added to the PIPELINE_LAYOUT.md and also to the Project status widget etc.

Correction Widget doesn't need a target anymore. it should just have a load function which loads by default the nuclear segmentation from the expected path or an alternative "load from layer" button to load from the active layer. The Ultrack widget should work the same way and share the same data field as the correction widget. In fact, they should probably just be one widget and have these two subcomponents

cell segmentation algorithm flow watershed produces artifacts and weird straight lines. previous stochastic process didn't have this problem... but it was much slower, made less sense and was probabilistic so going back is not really an option. investigate and find solution