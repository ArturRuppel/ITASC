# Cellpose Segment + Track: arrival & progressive-enable UX

## Problem

Arriving at the "Cellpose Segment + Track" napari widget
(`cellpose_segment_track_widget.py`), it is unclear what to do first. The
widget presents two dense rows of icon-only buttons (⧉ ⚙ ▷ ▶ ⊳ for Channel 1,
similarly for Channel 2) plus an embedded correction tool (⏻), with no
onboarding text and — critically — every action button is clickable
regardless of whether its precondition is met. Clicking Preview/Segment/Track
before the right input exists just produces a status-line error after the
fact (e.g. "Missing Channel 1 input.", "No '…masks' layer — segment first.").
The status label is blank at idle, so there is no passive guidance either.

This is not unique to this widget — icon-only buttons + tooltips + a shared
status label is the deliberate house style across every `cellflow.napari`
widget (`docs/superpowers/specs/2026-05-18-cellpose-widget-design.md`,
`2026-05-17-nucleus-workflow-stage-rows-design.md`) — but this is the
JOSS-facing standalone tool, so it's the one where a stranger's first five
minutes matter most.

## Goal

Make the arrival state self-explanatory through **affordance, not
exposition**: at any point, only the buttons whose preconditions are met are
enabled; the rest are visibly greyed out with a tooltip explaining what's
missing. A status label always names the next step (or steps) available,
instead of sitting blank until something goes wrong.

## Non-goals

- No new widget/component (no stepper, no wizard, no empty-state banner).
  Rejected in discussion: a full onboarding panel doesn't match the
  established house style and a one-time wizard fails the "returning after
  six weeks, forgot the order" case, which matters as much as the true
  first-time case.
- No changes to any other `cellflow.napari` widget. The same gap exists
  elsewhere but is out of scope here.
- No change to Channel 2's optionality — it stays optional, just mentioned in
  the idle hint.

## Design

### 1. Progressive button enabling via the existing `UiGate`

`UiGate` (`ui_gate.py`) is already the single source of truth for control
enablement in this widget; buttons register a `when()` predicate and a
`reason` (shown as the disabled tooltip). Today's registrations
(`_register_gate_controls`, `cellpose_segment_track_widget.py:1091`) only gate
on busy-state:

- `ch1_preview_btn`, `ch1_seg_btn`: extend `when` to also require
  `self._ch1_layer is not None`. `reason`: "Bind Channel 1 first — click ⧉."
- `ch1_track_btn`: extend `when` to also require a Channel-1 masks layer in
  the viewer (`_layer_name(_CH1_LABEL, "masks") in self.viewer.layers`).
  `reason`: a callable that returns "Bind Channel 1 first — click ⧉." if
  unbound, else "Segment Channel 1 first." if bound but no masks layer yet.
- `ch2_preview_btn`, `ch2_run_btn`: already gated on `_both_inputs()`; add a
  `reason` callable — "Bind Channel 1 first." / "Bind Channel 2 for joint
  segmentation." depending on which is missing.
- Correction tool's activate button (⏻, `cell_correction_widget.py`): **not**
  routed through the parent's `UiGate` — `CellCorrectionWidget` is reused in
  other (disk-mode) contexts where this check doesn't apply. Instead, add a
  local `setEnabled(self._correction_data_available())` scoped to
  `self._active_layer_mode()`, recomputed whenever the readiness could have
  changed. This needs new wiring: `cell_correction_widget.py` does not
  currently listen to `viewer.layers.selection.events.active` at all — add
  that connection (active-layer mode only) and call the same recompute from
  it. Reuse the existing status text ("Select a Labels layer to correct — the
  active layer is not one.", `cell_correction_widget.py:454-456`) as the
  disabled tooltip.

All the new `when` predicates rely on state changes that already trigger
`gate.recompute()` (`_set_channel_layer` at bind time, `_on_layers_changed` on
any layer insert/remove — which fires when Segment/Track add their output
layers) — no new event plumbing needed for the Channel 1/2 side.

### 2. Contextual "next step" status label

Rather than a new persistent stepper, enrich the *existing* `_status(...)`
call sites plus give the true idle state a real message instead of blank:

- Idle, nothing bound (widget construction / both channels unbound): "Bind
  Channel 1 to begin. (Optional: also bind Channel 2 for joint segmentation.)"
- After binding Channel 1 (`_set_channel_layer`, currently "Channel 1 ← layer
  'x'."): append " Preview a frame or Segment the full stack."
- After Segment completes (`_run_segment`'s `_done`, currently "Channel 1 →
  'masks', 'prob', 'flow'. Save from the layers."): append " Track to link
  masks across time."
- After Track completes (`_run_track`'s `_done`, currently "Channel 1 tracked
  → 'tracked'. Save from the layer."): append " Select it below to correct."

No new state machine: each message is emitted at the same call site that
already exists, just with an added trailing sentence naming the next action.

## Files touched (implementation, in the follow-up plan — not this spec-write)

- `src/cellflow/napari/cellpose_segment_track_widget.py` — gate registrations,
  status message text.
- `src/cellflow/napari/cell_correction_widget.py` — activate-button local
  gating + new selection-changed listener (active-layer mode only).
- `tests/napari/test_cellpose_segment_track_widget.py` (and the correction
  widget's test file) — assert enabled/disabled state and tooltip reason per
  precondition state, and status label content at each transition.
