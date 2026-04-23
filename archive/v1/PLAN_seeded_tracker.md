# Seeded Tracker Plan

## Goal

Replace the current Ultrack solve step with a simpler seeded tracker that:

1. builds an initial first frame from the hypothesis stack,
2. lets the user correct that first frame,
3. propagates identities forward by choosing the best next-frame candidate per cell,
4. produces the same labeled outputs the current solve step provides.

This widget will sit above the linking section in the UI.

## High-Level Flow

1. Load `cell_zavg` and `nucleus_zavg`.
2. Display the nucleus layer with additive blending and the `gop orange` LUT.
3. Build the initial frame from the hypothesis database:
   - take the first frame,
   - include all Z slices,
   - include all hypotheses,
   - compute the median image across that stack.
4. Instantiate the whole stack immediately from that consensus first frame.
5. Let the user correct that first frame with the correction widget.
6. Use the corrected first frame as the seed for downstream matching.
7. Provide a compact button like `show best next` that advances one frame and assigns each current cell to the best candidate in the next frame by raw IoU.

## UI Placement

- The widget lives above the linking section.
- It should feel like a bootstrap step, not a replacement for the full tracking UI.
- It should expose the current seeded track state directly to the viewer.

## Data Bootstrap

### Inputs

- `cell_zavg`
- `nucleus_zavg`
- Hypothesis database containing all candidate label stacks

### First-Frame Construction

- Start from the first frame only.
- Pull all Z slices from all hypotheses for that first frame.
- Compute the median across the full stack.
- Use that median result as the initial labels layer.
- Instantiate the full stack immediately so the viewer starts with a complete session, not a partial one.

### Visualization

- Cell layer: use the loaded cell z-averaged stack.
- Nuclear layer: use the loaded nucleus z-averaged stack.
- Nuclear labels should use additive blending.
- Nuclear labels should use the `gop orange` LUT.

## Matching Step

The first matching action is a button with a shorter label such as:

- `show next`
- `next best`
- `best next`

Behavior:

1. Look at the current tracked cells.
2. Query the database for candidate cells in the next frame.
3. Compute raw IoU between each current cell and each candidate.
4. Assign each current cell the best-scoring candidate.
5. Advance the tracker state to that next frame.

For now this is intentionally simple:

- no centroid correction,
- no global optimization,
- no solver,
- no learned model,
- raw IoU only.

## Outputs

The widget should write labeled outputs in the same spirit as the current Gurobi solve path:

- tracked labels stack
- tracks table or equivalent metadata
- the corrected seeded first frame
- any intermediate state needed to resume matching

If the existing pipeline expects `tracked_labels.tif`, this widget should emit an equivalent file.

## Non-Goals For v1

- No ILP solve.
- No Gurobi dependency.
- No long-range global consistency.
- No division reasoning.
- No complex motion model.
- No centroid-aligned IoU.

## Open Questions

- Where exactly the hypothesis database should live in the new flow.
- Whether the corrected first frame should be written back into the database or kept as a separate seed artifact.
- How much of the current Ultrack DB schema can be reused for candidate storage.
- Whether matching should update one frame at a time or allow a short lookahead window.

