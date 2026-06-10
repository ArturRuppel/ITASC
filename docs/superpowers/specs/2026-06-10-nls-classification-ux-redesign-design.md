# NLS Classification plugin — UX redesign

**Date:** 2026-06-10
**Component:** `src/cellflow/napari/aggregate_quantification/plugins/nls_classification.py`

## Problem

The NLS Classification plugin currently exposes two separate buttons —
**Measure & classify** and **Apply to H5** — plus a median-intensity scatter plot
that is always visible. This is poor UX:

- Two buttons for what feels like one task; "Apply to H5" stays greyed until you
  first click "Measure & classify", which is non-obvious.
- The scatter plot occupies panel space even before anything has been measured,
  when it shows nothing useful.
- The plot has a fixed `220px` minimum height and cannot be resized by the user.

## Goals

1. Collapse the two single-position buttons into **one stateful action button**.
2. **Hide the scatter plot until the first position is classified**; show it after.
3. Make the scatter **resizable** via a draggable splitter inside the panel.

Non-goals: changing the measurement/classification/write *core* logic
(`measure_track_nls_intensity`, `auto_threshold`, `classify_by_threshold`,
`write_nls_classification`, `patch_position_contact_analysis_nls_classes`), the
NLS-path resolution, or batch-mode semantics beyond relabeling its single button.

## Design

### Single action button (single-position mode)

One button, `self._action_btn`, driven by a small state machine. It replaces the
former `self._measure_btn` and `self._apply_btn`.

| State | Button label | Enabled when | Click action |
|---|---|---|---|
| **Needs classify** | `Classify` | a single position is selected, its NLS path resolves, and its nucleus labels file exists | run the measure worker → on success reveal the results pane, auto-threshold, classify → transition to *Classified* |
| **Classified** | `Apply to H5` | assignments exist (i.e. a successful classify) | write the classification into the position `.h5`; remain in *Classified* (status line confirms the write; clicking again re-writes) |

Transitions:

- Dragging the threshold line or changing the threshold spinbox → **stays** in
  *Classified*; the button keeps reading `Apply to H5`. (Re-classification is
  live, as today.)
- **Changing the NLS path text**, or **selecting a different position**, →
  reverts to *Needs classify*: hide the results pane, clear measurement state,
  button reads `Classify` again.

`current_threshold()` and the measure → classify → write pipeline are unchanged;
only the button wiring and visibility toggling are new.

### Batch mode (multiple positions selected)

Unchanged in behavior — it is already a single headless action. The same
`self._action_btn` is used; in batch mode it reads **`Classify & apply to all
H5`** and is enabled when every selected position has its NLS image, nucleus
labels, and `contact_analysis.h5` present (`_batch_records()` non-empty). No
scatter is shown in batch mode (the results pane stays hidden).

### Layout: splitter + deferred scatter

Restructure the panel body into a vertical `QSplitter` with two panes:

```
+-- NLS Classification panel ----+
| Position: pos00                |   top pane (always visible)
| NLS image: [______] [Browse]   |
| Positive:[pos] Negative:[neg]  |
| [ Classify  /  Apply to H5 ]   |
| Status: …                      |
|================================|   draggable divider
| Threshold: [ 12.34 ]           |   bottom pane = "results"
|  o    o   o                    |   (hidden until first Classify)
|  --o------o-- (draggable line) |
|  3 positive / 5 negative       |
+--------------------------------+
```

- **Top pane** (always visible): scope label, NLS-image row (+ Browse), the
  positive/negative label row, the single action button, the status label.
- **Bottom "results" pane** (a container widget): the threshold spinbox row, the
  scatter `PlotWidget`, and the counts label — grouped because all three are only
  meaningful after a classification. The whole pane is **hidden on init and after
  any revert**, and shown when a classification first succeeds.
- The splitter handle between the panes is draggable, giving the user control of
  the scatter vs. controls split.
- Remove the fixed `220px` minimum height; instead, when the results pane first
  appears, set sensible initial splitter sizes so the scatter gets usable height.
- **pyqtgraph absent**: the bottom pane shows the existing "Scatter unavailable
  (pyqtgraph not installed)" placeholder in place of the plot. Classification and
  the `Classify → Apply to H5` flow still work; the pane is still hidden until the
  first classify.

### State/visibility helpers

- A method to set the current state (e.g. `_set_state(...)` or extending
  `_update_enabled()`) computes the button label + enabled-ness and shows/hides
  the results pane based on: batch vs single, whether a measurement exists, and
  whether inputs (NLS path / nucleus labels) are valid.
- `_reset_measurement()` additionally hides the results pane and resets the button
  to `Classify`.
- The NLS-path `textChanged` handler reverts to *Needs classify* when the text
  changes after a classification (clears assignments + hides results).

## Testing

Update `tests/napari/test_nls_classification_plugin.py` (and, if affected,
`tests/napari/test_studio_plugins.py`) to the new single-button API:

- Replace references to `_measure_btn` / `_apply_btn` with `_action_btn` and
  assert on its **label** and **enabled** state per transition:
  - fresh single position with valid inputs → button reads `Classify`, enabled;
    results pane hidden.
  - after `_on_measure_done(...)` → button reads `Apply to H5`, enabled; results
    pane visible; `current_threshold()` within expected range.
  - changing the NLS path after classify → reverts to `Classify`, results pane
    hidden.
  - >1 record in context → button reads `Classify & apply to all H5`, gated on
    `_batch_records()`.
- Keep existing coverage of the measurement/threshold/write core (it is
  unchanged).

Note: `test_single_position_measures_and_auto_thresholds` is currently failing on
`main` for an unrelated reason (auto-threshold returns `0.0` on its synthetic
fixture); this redesign does not attempt to fix that, but the test will be
re-pointed to `_action_btn` as part of the API update.

## Risks / trade-offs

- The state machine adds a little control-flow complexity to an already-large
  widget; mitigated by funneling all label/enable/visibility decisions through one
  place.
- Re-clicking `Apply to H5` re-writes the `.h5`; this is intentional (idempotent,
  forgiving) and matches the approved behavior.
