# Stage Header Button Cluster Design

## Problem

CellFlow workflow stage rows currently split the visual header into a left-side
stage title and right-side controls. This creates an unclear interaction model:
the title area reads like the section identity, while the controls that expose or
run that section can appear detached from it. Passive right-side labels or status
text should not become click targets because they do not have an obvious toggle
or command meaning.

## Goal

Make stage-row actions feel attached to the stage they affect without turning
passive labels into ambiguous clickable regions.

## Design

Stage rows should group their compact action buttons immediately to the right of
the stage title:

```text
[ Stage title  gear  run ]                         [optional passive status]
```

The title remains a stage-colored pill. The adjacent action buttons use a related
pill treatment so the left cluster reads as one coherent stage control group, but
each action remains an independent button with its own tooltip, checked state,
keyboard focus, and click behavior.

The right side of the row remains available for passive status or metadata. Those
labels are not clickable.

## Components

- Add a shared style helper for compact stage-header action buttons.
- Reuse existing `QToolButton` controls for parameter toggles, run buttons,
  preview buttons, and activation buttons.
- Update the cell and nucleus workflow stage-row builders so buttons are added
  directly after the stage label, followed by stretch and any passive trailing
  widgets.

## Behavior

- Parameter buttons still expand or collapse their associated inline
  `CollapsibleSection`.
- Run, preview, active, and file-browser buttons keep their current semantics.
- Passive labels and status summaries do not toggle sections.
- Existing keyboard navigation remains intact because the controls remain normal
  Qt buttons.

## Testing

- Add focused Qt tests that inspect representative cell and nucleus stage rows:
  action buttons should appear immediately after the stage label in layout order.
- Add a style test for the new stage-header action button helper, including
  checked-state affordance for parameter toggles.
- Preserve existing tests that validate hidden `CollapsibleSection` headers,
  section expansion state, and compact stage header styling.

## Out of Scope

- Do not make whole header rows clickable.
- Do not make passive sublabels or status text clickable.
- Do not replace gear icons with chevrons in this change.
- Do not change the underlying stage execution or parameter state logic.
