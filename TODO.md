redrawing sometimes leads to stranded pixels. disconnected from their cells but still with the same id. we should clean this up. with a button? or in regular intervalls?
→ DONE: "Cleanup" group added to correction widget with "Clean (this frame)" and "Clean (all frames)" buttons.

draw fails sometimes because a stroke is too short, though it should be valid, why?
→ DONE: draw_cell_path fallback now allows stroke to paint over existing cell pixels when extending, so short border strokes no longer fail.

merging cells doesn't always work. a cell needs to be cleanly selected (no ctrl+click, no splitting or merging action before, no redraw etc.) for it to work.
→ DONE: merge now uses label IDs directly instead of stale click positions, so it works after prior edits.

fix borders should be possible to run on one frame only.
→ DONE: "Fix borders (this frame)" button added alongside "Fix borders (all frames)".

edge analysis param header looks different than the other widgets
→ DONE: "Analyse Tissue" toggle changed from QPushButton to QToolButton with arrow, matching the tracking widget style.

Tracking widget creates a new segmentation layer if input layer was not called Segmentation. which one is ground truth now? 
→ DONE: tracking result is now written back to the source layer by name (from state.tissue.labels_layer), falling back to the segmentation tab's layer, then creating new only as last resort.

it should be possible to toggle between seeing the full label and only the outlines of the labels in the correction widget. epicure has this feature already, could be explored there.
→ DONE: "Show outlines only" toggle button added to correction widget (sets layer.contour = 2 / 0).

there should be a shortcut for saving a tissue to the current file.
→ DONE: Ctrl+S saves to current path without dialog; opens dialog if tissue is unsaved.
