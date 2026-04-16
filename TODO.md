Prepare Input Data widget: should have a "pull metadata" button to get px and dt from ndtiff. should discover the available positions and allow to export them all. should always export all timepoints. should overwrite pix size and dt when exporting with metadata. when downsampled, pixelsize needs to be calculated accordingly. text next to "overwrite existing files" is too dark, should be white.

Write a quick user guide to get the user oriented quickly. think about how and where to display it.

expose an overview of each widget somewhere. discrete but obvious. to be designed first

for resizing the widgets one has to drag an invisible bar. make it visible somehow maybe?

parameter handling should be homogenized, including naming and placement of the buttons. big task. analyze and plan first.

remove flow threshold from contours widget, both the ui and the parameter itself - it's useless in 3D for mask creation

some widgets, like ultrack or cellpose, are minimal on the first layer, they just contain the subwidget. when expanindg the subwidget, the container is then very small and needs to be expanded manually. it would be nice if the expansion of the collapsed subwidget resized the container widget automatically to fit the whole thing

