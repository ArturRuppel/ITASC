clean function in correction widget should also fill holes in cells.

restore correctin mode warning get's stuck sometimes

still weird bug in cell splitting tool... when line crosses a segment of a cell where the extension of that line crosses the cell again at another spot, the other part gets cut instead of the one where to cut was made

cellpose model dialog looks for "model files" but model files don't have file extensions, so they are not detected. switching to all files and then using the model does work though (e.g. /home/aruppel/projects/cellpose_training/cpsam_U251)
→ FIXED: swapped filter order so "All files (*)" is the default in the dialog


load image doesn't always load the active image layer for some reason
→ FIXED: _on_capture_image now checks the active layer first before falling back to the topmost Image layer

there should be a shortcut for saving a tissue to the current file.
→ FIXED: replaced QShortcut(ApplicationShortcut) with viewer.bind_key("Control-S", overwrite=True) to avoid napari conflict

draw cell path is unreliable, sometimes doesn't cut into neighboring cells properly
→ FIXED: removed the seg==0 background-only restriction for new-cell creation; the drawn closed region always becomes the new cell, overwriting neighbors
