# Correction Artifact Cleanup Design

## Goal

Separate the correction widget's mixed `Clean Holes / Islands` behavior into explicit artifact repair operations that are easier to reason about and test. The controls should be available anywhere the shared `CorrectionWidget` is embedded, including the cell workflow and nucleus workflow correction sections.

## Current Behavior

`CorrectionWidget` currently exposes one `Clean Holes / Islands` button and one `Hole radius` control. Pressing the button operates on the current frame only. Internally it first calls `fill_label_holes` when the radius is greater than zero, then always calls `clean_stranded_pixels`.

This mixes three concepts:

- Fully enclosed background holes inside or between cells.
- Disconnected same-label fragments, where only the largest component is kept.
- Tiny enclosed background islands filled by `clean_stranded_pixels`.

It also explicitly preserves border-connected background, so semiholes with a narrow opening to the image border are not repaired.

## User-Facing Design

Replace the single cleanup button with an `Artifact cleanup` block in `CorrectionWidget`.

Controls:

- Scope selector with `Current frame` and `All frames`.
- `Fill Holes` button, using the existing hole radius.
- `Fix Semiholes` button, using a conservative max opening/radius control.
- `Clean Fragments` button, using the existing small-fragment cleanup behavior.

The old label `Clean Holes / Islands` should disappear from the widget. The cell workflow tests should assert the new controls are present in the correction section, and nucleus correction layout tests should be updated because the shared widget changes everywhere.

## Operation Semantics

`Fill Holes` fills fully enclosed zero-valued background gaps by expanding neighboring labels up to the configured hole radius. This is the current `fill_label_holes` behavior.

`Clean Fragments` removes disconnected same-label fragments by keeping the largest connected component for each label and assigning removed pixels to nearby labels. It should no longer fill enclosed background holes as part of this operation. Any tiny background-hole cleanup that remains necessary should live under `Fill Holes`, not under fragment cleanup.

`Fix Semiholes` repairs only conservative border-connected gaps. It should:

- inspect zero-valued connected components that touch the image border;
- estimate the component's border opening/contact;
- reject components whose opening exceeds the configured threshold;
- fill eligible pixels by expanding neighboring labels, preserving unrelated open background.

This must be a separate algorithm from enclosed-hole filling so semihole behavior can be tested and tuned independently.

## Scope Behavior

For `Current frame`, operate only on the viewer's current time point, matching the existing cleanup behavior.

For `All frames`, loop through every time point in the active label layer. Each changed frame should record undo history with the existing per-frame `_record_history` helper so undo remains compatible with napari's label history model.

Both scopes should refresh the layer once after changes and preserve the selected-label highlight when possible.

## Status And Failure Handling

If no active labels layer exists, show an error status as today.

If an operation changes pixels, report the operation, changed frame count, and changed pixel count. Example: `Fixed semiholes in 7 frame(s), 312 px changed. Unsaved.`

If nothing changes, report operation-specific no-op text, such as `No semiholes found`.

Unexpected cleanup exceptions should still surface through napari notifications.

## Implementation Boundaries

Algorithmic functions belong in `cellflow.correction.labels`, not in the Qt widget. The widget should only handle UI state, frame iteration, history, refresh, highlight, and status.

Proposed function split:

- Keep `fill_label_holes(labels, radius)` for enclosed holes.
- Replace or narrow `clean_stranded_pixels(seg, min_size)` so it only handles disconnected label fragments.
- Add `fix_label_semiholes(labels, radius, max_opening)` for conservative border-connected gaps.

All functions should operate on one 2D frame. Stack-wide behavior belongs in the widget.

## Testing

Unit tests in `tests/tracking/test_correction.py` should cover:

- enclosed holes are filled by `fill_label_holes`;
- open background remains unchanged by `fill_label_holes`;
- disconnected label fragments are cleaned without also filling background holes;
- conservative semiholes with narrow border openings are repaired;
- wide border openings are left unchanged;
- zero radius or zero opening thresholds are no-ops where applicable.

Widget tests should cover:

- the shared correction widget exposes the new cleanup controls;
- the old `Clean Holes / Islands` label is absent;
- current-frame scope changes only the selected frame;
- all-frame scope changes multiple frames and records per-frame edits;
- the cell workflow correction section shows the new shared controls.

## Out Of Scope

This design does not add automatic cleanup during cell-label generation. Cleanup remains an explicit correction action so users can inspect and save changes deliberately.

This design does not introduce temporal cleanup that uses neighboring frames to infer corrections. Each cleanup operation remains frame-local.
