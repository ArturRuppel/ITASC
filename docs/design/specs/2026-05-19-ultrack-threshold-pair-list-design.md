# Ultrack Threshold Pair List Design

Date: 2026-05-19

## Goal

Replace the current Ultrack threshold sweep interface with an explicit threshold-pair workflow. Users should preview one `(contour_threshold, foreground_threshold)` pair at a time, add chosen pairs to a list, and use exactly that list for Ultrack database generation.

## Current Behavior

The nucleus workflow currently exposes threshold ranges under the separate "Ultrack Inputs" stage. Database generation reads contour and foreground threshold arrays from those controls and expands them into a Cartesian sweep. The threshold-expanded source stacks are calculated in memory, which should remain true.

## Proposed UI

Remove the separate "Ultrack Inputs" threshold sweep as a standalone stage. Threshold configuration moves into `Database Generation Parameters` as a "Source Thresholds" block.

The block contains:

- `Contour threshold`: numeric control in `[0, 1]`.
- `Foreground threshold`: numeric control in `[0, 1]`.
- `Preview`: builds and displays preview layers for the current pair only.
- `Add`: appends the current pair to the explicit threshold-pair list.
- Threshold-pair list: compact table/list showing contour and foreground values in order.
- Remove and clear controls for managing the list.

The threshold-pair list starts empty. No default pair is added automatically.

## User Workflow

1. User adjusts contour and foreground threshold controls.
2. User clicks `Preview`.
3. Napari preview layers update for only that current pair.
4. User clicks `Add` if the pair is acceptable.
5. User repeats until the list contains the desired candidate sources.
6. User runs Ultrack database generation.

Database generation uses the list exactly as shown, in order. Previewing does not add a pair and has no effect on database generation until the user clicks `Add`.

## Backend Behavior

Introduce a threshold-pair representation as an ordered sequence of mappings containing:

- `contour_threshold: float`
- `foreground_threshold: float`

The source-stack builder should support explicit pairs directly. For each pair, it should:

- preserve contour-map values at or above `contour_threshold`, zeroing lower values;
- binarize foreground scores using `foreground_threshold`;
- preserve the user-defined pair order in metadata and source order.

The existing in-memory source-stack behavior stays in place. The database build should not write threshold-expanded source-stack TIFFs.

Exact duplicate pairs should not be added twice from the UI. If the user tries to add a pair already in the list, leave the list unchanged and show a short status message.

## Empty List Handling

If the threshold-pair list is empty, database generation must not start. The UI should show a clear status message:

`Add at least one threshold pair before DB generation.`

## State Handling

Workflow state should persist the explicit threshold-pair list, not the old min/max/step sweep values. Loading older saved state must initialize the list as empty rather than reconstructing the old Cartesian sweep.

## Testing

Add focused tests for:

- the pair-list UI starts empty;
- adding and removing pairs updates the list;
- preview uses only the current pair and does not mutate the list;
- DB generation passes exactly the explicit pairs in order;
- DB generation refuses to start when the list is empty;
- source-stack generation from explicit pairs preserves order and does not write source-stack TIFFs.

Existing tests that assert range-sweep behavior should be updated or removed where the old workflow no longer exists.
