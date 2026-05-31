# Atom Extraction Widget — UX Cleanup Design

**Date:** 2026-05-31
**Status:** Approved (design)
**Predecessor:** `2026-05-31-atom-extraction-widget-design.md` (stage ① — shipped on `main`)

## Goal

Refine the already-shipped napari **Atom Extraction** widget so it matches the
established pipeline-stage UX (stage row with a `⚙` params toggle and a `▶`
run icon), moves to the top of the workflow, and visualizes its outputs and
intermediates directly — while persisting only `atoms.tif`. Purely a
UX/structure refinement of stage ①; the atom-extraction math is unchanged.
Still additive with respect to the higra DB-generation machinery (removed later
in spec ②).

## Scope

In scope: widget layout/structure, the run action's visualization behavior, one
additive core helper, the file contract, and tests. Out of scope: the exclusive
viewer-activity guard (atom preview still does not participate in
`_reject_conflicting_viewer_activity` — unchanged, tracked separately), worker
threads/cancellation for the run, and any spec-② consumer work.

## Design

### 1. Placement

Atom Extraction becomes the **first workflow section**, directly below Pipeline
Files. In `NucleusWorkflowWidget._setup_ui`, call
`self._build_atom_extraction_section(root)` immediately after the pipeline-files
section is added (and after the viewer-activity banner), before
`_build_segmentation_inputs_section` / `_build_tracking_ultrack_section`.

### 2. Stage-row header

The current widget renders a redundant double header (the `CollapsibleSection`
title *and* the widget's own internal header). Collapse them into one stage row
mirroring the "Ultrack database" / "Ultrack solve" rows:

```
Atom Extraction      [⚙] [👁] [▶]
   └ collapsible params (hidden header, driven by ⚙)
        FG window / FG cutoff
        Contour window / Contour floor
        Min area
   status: "Frame 7: 118 atoms"  /  "Wrote 50 frames → atoms.tif"
```

- `⚙` `params_btn` — checkable `tool_btn`, styled `stage_header_action_button(_, "nucleus")`.
  Wired so `params_btn.toggled → section._toggle.setChecked(checked)`. The params
  `CollapsibleSection` is created with its own header hidden
  (`set_header_visible(False)`) and `collapse()`-d by default; `params_btn`
  starts unchecked.
- `👁` `active_btn` — checkable `tool_btn` (repurposes today's `⏻`). Drives the
  live single-frame preview; tooltip "Live atom preview (tune against the
  current frame)."
- `▶` `run_btn` — non-checkable `tool_btn`, replaces the
  `Compute atoms (full stack)` `QPushButton`. Tooltip "Compute atoms for all
  frames, show them, and write atoms.tif."
- Stage label styled with `stage_header_label(_, "nucleus")`.

The widget exposes `self.header` (the stage row) and `self.section` (the params
`CollapsibleSection` wrapping the slider body + status label). The widget
self-wires `params_btn ↔ section`. `_build_atom_extraction_section` adds
`w.header` then `w.section` to `root` and calls `_alias_atom_extraction_controls()`.

### 3. Visualization behavior

Three layers, always (the `territory_overlay_check` / `residual_overlay_check`
checkboxes are **removed**):

- `[Atoms] preview` — Labels (atoms)
- `[Atoms] territory` — Labels
- `[Atoms] residual_contour` — Image (magma, additive)

**Live preview** (`👁` active): on activate and on debounced param/frame change,
recompute the **current frame** and update all three layers.

**Run** (`▶`): compute the **full stack**, update all three layers with the
full `(T,Y,X)` stacks, and write **only** `atoms.tif` (via the existing
`write_atoms_tif`, params + fingerprint embedded). Intermediates are
display-only — never written. The run is synchronous with a status message
(extraction is lightweight; no worker/cancel added).

### 4. Visibility-preserving layer updates (required)

The layer-update helpers must preserve user-set visibility. The rule:

- If the layer exists and its data ndim is unchanged, replace `.data` in place
  and **do not touch `.visible`**.
- If the layer must be removed and re-added (it isn't the expected type, or the
  data ndim changed — e.g. switching between a 2D live-preview frame and a 3D
  full-stack run), **capture the old layer's `.visible` before removal and
  reapply it after re-adding**.
- A brand-new layer (none existed) is added visible by default.

Consequence: if the user hides `[Atoms] residual_contour` (or any of the three)
in napari's layer list, it stays hidden across all later refreshes and runs,
including across the 2D↔3D transition. This replaces the old
`_toggle_atom_overlay` logic, which forced `visible = True` on update.

Deactivating the live preview (`👁` off) still removes the three layers, as
today.

### 5. Core helper (`atoms.py`, additive)

Add:

```python
def extract_atoms_stack_with_maps(
    fg: np.ndarray, contour: np.ndarray, params: AtomParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(atoms, territory, residual_contour) stacks, each (T, Y, X)."""
```

Refactor `extract_atoms_stack` to delegate to it and return only `atoms` — its
public signature and behavior are unchanged, so existing tests stay green.
`territory` is returned as `uint8` (0/1), `residual_contour` as `float32`,
`atoms` as `int32`. The run uses this helper; only `atoms` is persisted.

### 6. File contract & status

- `_NUCLEUS_PIPELINE_FILE_GROUPS` (in `nucleus_workflow_widget.py`): add
  `("2_nucleus/atoms.tif", "Atoms")` under the **Intermediates** group.
- `_paths.py` module docstring: add `atoms.tif` to the `2_nucleus/` line of the
  canonical layout.
- Update the `project_atom_extraction_stage1` memory to describe the new UX.

### 7. State

No change. Only the five spin params are persisted under `"atom_extraction"`;
the removed overlay checkboxes were never persisted.

## Testing

- `test_atoms.py`: add a test for `extract_atoms_stack_with_maps` — returns three
  `(T,Y,X)` stacks with correct dtypes, and its `atoms` equals
  `extract_atoms_stack(...)` for the same inputs.
- `test_nucleus_atom_extraction_widget.py`:
  - Controls: `params_btn` and `run_btn` exist; `active_btn` is checkable and
    unchecked; the overlay checkboxes are gone; spin defaults unchanged.
  - `⚙` wiring: toggling `params_btn` expands/collapses `section`.
  - Live preview: activate adds the three layers; a refresh after the user sets
    a layer's `visible = False` keeps it hidden (visibility-preserving rule).
  - Run: `_run_atom_extraction` (renamed from `_compute_atoms_full_stack`) writes
    `atoms.tif` with the right fingerprint and adds the three full-stack layers.
  - Workflow wiring + state round-trip tests updated for the new control names.

## Risks

- Renaming `_compute_atoms_full_stack` / removing the overlay checkboxes breaks
  the existing widget tests; they are updated in lockstep.
- Live-preview (2D, single frame) and run (3D, full stack) share layer names;
  updating a 2D layer's data to 3D (and vice-versa) must be handled by removing
  and re-adding when the ndim changes, otherwise napari raises on the in-place
  data swap.
