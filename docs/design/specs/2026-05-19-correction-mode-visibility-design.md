# Correction Mode Visibility Design

## Goal

Make correction-mode activation obvious and easy to exit in the nucleus and cell workflow widgets, while keeping the workflows permissive. Users should be able to run other widget actions while correction mode is active, but they should never lose track of that active state or have to hunt for a tiny control to turn it off.

## Current Behavior

Both correction widgets use a compact checkable power icon in the correction header as the master on/off control. Activating correction mode hides existing viewer layers, loads correction-owned layers, activates the embedded `CorrectionWidget`, and expands the correction panel. Deactivation restores the previous viewer state, removes correction-owned layers, refreshes tracked layers from disk, and collapses correction-specific content.

This behavior is mostly correct, but the activation state is visually too quiet. When users continue working elsewhere in the widget, correction mode can remain active without a strong reminder. Turning it off requires finding the same compact icon again.

## Requirements

- Keep correction mode permissive. Do not disable pipeline, DB browser, file panel, or other workflow controls while correction is active.
- Make correction state visible from the always-visible correction header.
- Provide a large, obvious exit affordance inside the active correction panel.
- Reuse the existing activation/deactivation code paths so viewer-state capture, restoration, layer cleanup, shortcut state, and tracked-layer refresh remain centralized.
- Apply the same interaction pattern to nucleus and cell correction widgets.
- Preserve the existing params and shortcuts controls.

## UI Design

The correction header will expose a prominent checkable text control:

- Inactive label: `Correction Mode`
- Active label: `Correction Active`

When active, the control should be visually louder than the existing icon button: bold text, active accent styling, and enough width to be findable at a glance. This prominent toggle replaces the existing small power icon as the visible correction-mode control, while keeping the same activation/deactivation role in code. Params and shortcuts stay as compact controls.

The expanded correction content will include an active-state banner at the top:

- Status label: `Correction mode active`
- Button: `Exit Correction`

The banner appears only while correction mode is active. Clicking `Exit Correction` toggles the same active button off rather than calling cleanup logic directly.

## Behavior

Activating correction mode should continue to:

- Capture the current viewer layer visibility and selection.
- Hide regular layers.
- Load correction-owned layers.
- Activate the embedded correction widget.
- Expand the correction section.

Deactivating correction mode should continue to:

- Deactivate the embedded correction widget.
- Disable correction shortcuts where applicable.
- Refresh the main tracked layer from disk.
- Remove correction-owned layers.
- Restore the previous viewer layer visibility and selection.
- Hide the active-state banner.

No other workflow controls are disabled by this feature.

## Implementation Scope

Primary files:

- `src/cellflow/napari/nucleus_correction_widget.py`
- `src/cellflow/napari/cell_correction_widget.py`

Likely supporting file:

- `src/cellflow/napari/ui_style.py`

Tests:

- `tests/napari/test_nucleus_correction_widget.py`
- `tests/napari/test_cell_correction_widget.py`
- `tests/napari/test_nucleus_tracking_correction_layout.py` if workflow-level aliases need coverage

## Test Plan

Add focused widget tests that verify:

- Each correction widget exposes the prominent mode toggle and exit button.
- The toggle text changes between inactive and active labels.
- The active banner is hidden while inactive and visible while active.
- Clicking `Exit Correction` deactivates correction mode through the same checked state.
- Existing params and shortcuts controls still toggle their sections.

Existing correction activation and cleanup tests should continue to cover layer and viewer-state behavior.

## Non-Goals

- Do not make correction mode modal.
- Do not block other workflow actions while correction is active.
- Do not redesign the correction operation toolbar.
- Do not change correction algorithms, save formats, or layer names.
