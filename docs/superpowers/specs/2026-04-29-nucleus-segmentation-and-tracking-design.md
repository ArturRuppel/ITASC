# Nucleus Segmentation & Tracking Design

**Date:** 2026-04-29

## Goal

Redesign the nucleus workflow UI so tracking and correction are presented as one coherent workflow stage named `Nucleus Segmentation & Tracking`, while leaving the contour-map, hypothesis-generation, and database-browser sections functionally unchanged.

## Scope

This design is limited to the napari nucleus workflow UI in [src/cellflow/napari/nucleus_workflow_widget.py](/home/aruppel/Projects/CellFlow/src/cellflow/napari/nucleus_workflow_widget.py) and the embedded correction UI in [src/cellflow/napari/correction_widget.py](/home/aruppel/Projects/CellFlow/src/cellflow/napari/correction_widget.py). It is primarily a layout and control-surface refactor. Existing tracking, resolving, correction, save/load, extend, retrack, and reassign backend behavior should be reused wherever possible.

Out of scope:
- Changes to contour-map generation
- Changes to hypothesis generation or sweep behavior
- Changes to the database browser
- Rewriting ultrack, correction, extend, retrack, or validation backend logic beyond what is required to expose existing parameters in the UI

## Current-State Summary

The current nucleus widget has separate top-level sections for:
- `4. Tracking`
- `5. Manual Correction`

The current `Tracking` section contains:
- ultrack parameters in a single-column form
- a run button and terminal button
- a progress bar
- tracked-label save/load buttons
- a `Reassign IDs` button

The current `Manual Correction` section contains:
- extend forward/backward buttons
- retrack forward/backward buttons
- `Re-solve from validated`
- terminal resolve button
- the embedded `CorrectionWidget`

The current correction-related hardcoded parameters that should become editable are:
- retrack uses `max_dist_px=20.0` in both forward and backward retrack handlers
- extend uses the current `extend_track(...)` selection behavior without any UI-exposed tuning controls

## Top-Level Layout

Rename the workflow to:
- `Nucleus Segmentation & Tracking`

Keep these top-level sections unchanged in purpose and overall layout:
- `1. Contour Maps`
- `2. Hypothesis Generation`
- `3. Database Browser`

Replace the current separate tracking and manual-correction sections with one top-level section:
- `4. Tracking & Correction`

This top-level section should use the same `CollapsibleSection` visual language as the rest of the plugin.

## Internal Structure of `4. Tracking & Correction`

Inside `4. Tracking & Correction`, use two collapsible subsections:
- `Ultrack Tracking`
- `Correction`

No tabs should be used anywhere in this area. The redesign should stay consistent with the existing collapsible-section pattern already used elsewhere in the plugin.

## `Ultrack Tracking` Subsection

### Purpose

This subsection is the single home for the ultrack pipeline. Normal ultrack tracking and resolve-from-validated are treated as the same pipeline with different input routes.

### Layout

The subsection should contain:
- one shared ultrack parameter area
- one shared local status label
- one shared local progress bar
- one shared action area

The ultrack parameter area should be arranged in **two columns**, not the current single-column form layout.

Buttons in this subsection should shrink and reflow consistently with the rest of the plugin. The behavior should match the compact/stretchable button rows used in the other widgets rather than forcing rigid widths.

### Parameter Model

The subsection should continue to expose the full ultrack parameter set already used by the widget, including:
- min area
- max partitions per frame
- first N frames
- linking mode
- max distance
- IoU weight
- appear penalty
- disappear penalty
- division penalty
- max neighbors
- solver display

These parameters are shared across both pipeline routes.

### Route Selection

This subsection should not expose separate top-level action groups for “normal ultrack” and “resolve from validated.” Instead, it should use:
- one main run button
- one terminal run button
- one mode selector or checkbox that switches the input route between:
  - normal ultrack run
  - resolve from validated

The exact control may be a checkbox or another minimal selector, but it should be a single route modifier rather than multiple competing action rows.

### Status and Progress

The subsection should have its own local:
- status label
- progress/loading bar

These should communicate ultrack/resolve activity without relying exclusively on the widget-wide status label.

### Controls Removed from This Subsection

The following controls should be removed from `Ultrack Tracking`:
- `Save Tracked Labels`
- `Load Tracked Labels`
- `Reassign IDs`

These controls move to `Correction`.

## `Correction` Subsection

### Purpose

This subsection is the home for interactive editing of tracked labels and related manual refinement actions.

### Contents

`Correction` should contain:
- the embedded correction subwidget
- tracked-label persistence controls
- reassign-ID control
- extend and retrack action buttons
- nested collapsibles for shortcuts and advanced parameters

### Always-Visible Action Controls

The main correction area should keep the following actions visible even when advanced parameter groups are collapsed:
- extend backward
- extend forward
- retrack backward
- retrack forward
- `Load Tracked Labels`
- `Save Tracked Labels`
- `Reassign IDs`

The extend and retrack buttons should remain directly accessible in the main correction UI rather than being buried inside advanced collapsibles.

### Nested `Correction Shortcuts`

The correction shortcut reference should move into a nested collapsible inside `Correction`:
- title: `Correction Shortcuts`
- default state: expanded

This keeps the shortcuts available without requiring them to permanently occupy space when the user collapses them later.

### Nested `Extend Parameters`

The extend action should gain a nested collapsible parameter section inside `Correction`:
- title: `Extend Parameters`
- default state: collapsed

This section exists to expose extend-related behavior that is currently effectively hardcoded behind the `extend_track(...)` call. The implementation plan must define exactly which extend parameters are surfaced based on what the backend already supports or can support with minimal changes.

The corresponding extend action buttons remain visible in the main correction area regardless of the collapsible state.

### Nested `Retrack Parameters`

The retrack action should gain a nested collapsible parameter section inside `Correction`:
- title: `Retrack Parameters`
- default state: collapsed

At minimum, this section should expose the retrack distance parameter that is currently hardcoded as:
- `max_dist_px=20.0`

If both forward and backward retrack use the same parameter, the UI should expose one shared retrack-distance control unless implementation constraints require separate fields.

The corresponding retrack action buttons remain visible in the main correction area regardless of the collapsible state.

## Responsiveness and Sizing

The redesigned tracking/correction UI must behave well in the constrained width of a napari dock widget.

Requirements:
- button rows must shrink with the widget width
- controls should prefer the existing compact button/spinbox behavior already used elsewhere in the codebase
- the two-column ultrack parameter layout should remain legible at typical dock widths
- collapsible subsections should keep the default view compact, with advanced controls hidden until needed

## Behavior and State Rules

The redesign is primarily a UI reorganization. Existing backend handlers should be reused where possible.

Behavioral expectations:
- contour maps, hypothesis generation, and database browser should behave as before
- ultrack and resolve-from-validated should share one parameter surface
- resolve-from-validated should be presented as an alternate input route within ultrack, not as a separate top-level workflow stage
- correction remains the place where tracked labels are loaded, saved, reassigned, and manually edited
- extend and retrack should gain explicit parameter controls instead of relying entirely on hidden hardcoded values

## Implementation Notes

Likely files affected:
- [src/cellflow/napari/nucleus_workflow_widget.py](/home/aruppel/Projects/CellFlow/src/cellflow/napari/nucleus_workflow_widget.py)
- [src/cellflow/napari/correction_widget.py](/home/aruppel/Projects/CellFlow/src/cellflow/napari/correction_widget.py)
- [src/cellflow/napari/widgets.py](/home/aruppel/Projects/CellFlow/src/cellflow/napari/widgets.py) if a small shared layout helper is useful

Preferred implementation direction:
- keep most backend method names and signal wiring intact
- move controls rather than rewrite logic
- add only the minimum new state needed to support route selection and parameter exposure
- preserve current keyboard shortcuts and correction activation behavior

## Acceptance Criteria

The redesign is complete when:
- the workflow is labeled `Nucleus Segmentation & Tracking`
- `Contour Maps`, `Hypothesis Generation`, and `Database Browser` remain untouched in behavior
- tracking and correction are merged into one top-level section
- `Ultrack Tracking` and `Correction` are collapsible subsections inside that section
- ultrack parameters are shown in two columns
- ultrack has its own local status label and progress bar
- ultrack uses one main run path plus a route selector/modifier for resolve-from-validated
- tracked-label load/save and `Reassign IDs` are moved to `Correction`
- `Correction Shortcuts` is nested inside `Correction` and open by default
- `Extend Parameters` and `Retrack Parameters` exist as nested collapsibles inside `Correction`, both closed by default
- extend and retrack action buttons remain always visible
- the redesigned buttons shrink appropriately with dock width
