# Stage Header Button Cluster Design

## Problem

CellFlow workflow stage rows currently split the visual header into a left-side
stage title, detached right-side controls, and occasional right-side status
indicators. This creates an unclear interaction model:
the title area reads like the section identity, while the controls that expose or
run that section can appear detached from it. Passive right-side labels also
compete with the actual active-mode/status surfaces.

## Goal

Make stage-row actions feel attached to the stage they affect without turning
passive status indicators into a second header surface.

## Design

Stage rows should group their compact action buttons immediately to the right of
the stage title:

```text
[ Stage title  gear  run ]
```

The title remains a stage-colored pill. The adjacent action buttons use a related
pill treatment so the left cluster reads as one coherent stage control group, but
each action remains an independent button with its own tooltip, checked state,
keyboard focus, and click behavior.

Right-side status indicators should be removed from stage headers. State should
instead appear in the shared status/progress area, active-state banners, button
checked state, and disabled-reason tooltips.

## Components

- Add a shared style helper for compact stage-header action buttons.
- Reuse existing `QToolButton` controls for parameter toggles, run buttons,
  preview buttons, and activation buttons.
- Update the cell and nucleus workflow stage-row builders so buttons are added
  directly after the stage label, followed by stretch only when layout spacing
  needs it.

## Behavior

- Parameter buttons still expand or collapse their associated inline
  `CollapsibleSection`.
- Run, preview, active, and file-browser buttons keep their current semantics.
- Stage headers do not show passive right-side status indicators.
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
- Do not add passive sublabels or status text to the right side of stage headers.
- Do not replace gear icons with chevrons in this change.
- Do not change the underlying stage execution or parameter state logic.
