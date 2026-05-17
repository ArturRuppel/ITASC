# Nucleus Workflow Widget — Per-Stage Rows

**Date:** 2026-05-17
**Scope:** `src/cellflow/napari/nucleus_workflow_widget.py` and `nucleus_pipeline_widget.py` (plus light edits to the two inputs widgets and affected tests).

## Problem

The current nucleus workflow widget shows three separate collapsible parameter
sections (Segmentation Input Parameters, Database Generation Parameters,
Ultrack Solver Parameters) and then, *below* them, a stage block with three
checkboxes (one per stage) and a single Run button that chains the checked
stages. The parameters for a stage live far from its run control, and the
checkbox-chain implies a "select stages, then run" workflow that's heavier
than what the user actually wants. The user wants the stage to be the unit:
name, params, preview, run — all together on one row.

## Goals

- Each pipeline stage is one row, anchored under a green (nucleus accent)
  label.
- Each row carries 2–3 trailing icons: **parameters**, **preview** (Segmentation
  only), **run**.
- Params open *inline* below the stage row (collapsed by default).
- Only one stage runs at a time; no chained multi-stage run, no checkboxes.
- Shared status + progress bar live below all three rows, as today.

## Non-goals

- No change to the Pipeline Files panel, Ultrack DB Browser, or Correction
  sections.
- No change to underlying worker handlers, threshold/config plumbing, or
  artifact paths.
- No change to themes / palette / accent system.

## Final layout

```
Pipeline Files                                    [unchanged]
─────────────────────────────────────────────────
Segmentation inputs            ⚙   ▷   ▶
  ┊ (params block, collapsed by default)
Ultrack database               ⚙       ▶
  ┊ (params block, collapsed by default)
Ultrack solve                  ⚙       ▶
  ┊ (params block, collapsed by default)
[████████░░░░] Building averaged maps…           [shared progress + status]
─────────────────────────────────────────────────
Activate DB Browser            [unchanged]
Activate Correction Mode       [unchanged]
```

## Stage rows

Each row is a horizontal layout:

| Element              | Source                                                                 |
|----------------------|------------------------------------------------------------------------|
| Stage label          | New `QLabel`, bold, color = `stage_accent("nucleus")`, font ~11pt      |
| Spacer (stretch)     | —                                                                      |
| Parameters icon ⚙   | New `QToolButton`, checkable; toggles the inline params block          |
| Preview icon ▷       | Existing `preview_contour_btn` — moved into Segmentation row (Seg only)|
| Run icon ▶ / Cancel ✕| New `QToolButton`; swaps glyph + handler depending on run state        |

Tooltips:
- ⚙ → "Show parameters for this stage."
- ▷ → "Preview the current frame's segmentation input source sweep without writing artifacts." (existing tooltip text)
- ▶ → e.g. "Run segmentation inputs." / "Run Ultrack database build." / "Run Ultrack solve."
- ✕ (while running) → "Cancel."

### Inline params block

The existing `CollapsibleSection` instances built inside the two inputs widgets
(`nucleus_segmentation_inputs_widget.section`, `nucleus_tracking_inputs_widget.db_section`,
`nucleus_tracking_inputs_widget.solve_section`) are reused, but their own
toggle headers are hidden and they default to collapsed. The stage's ⚙ icon
becomes the toggle — `setChecked(bool)` on the icon expands/collapses the
section's inner content.

Implementation note: keep `CollapsibleSection` as the container (so the green
left-stripe styling and accent inheritance keep working) and add a small
`set_header_visible(bool)` helper that hides/shows its own toggle button.
The stage row's ⚙ button is checkable; when toggled it calls
`section._toggle.setChecked(checked)` so the existing `_on_toggled` path
expands/collapses the content.

## Stage → handler mapping

| Stage label           | Params widget                                   | Preview          | Run handler                |
|-----------------------|-------------------------------------------------|------------------|----------------------------|
| Segmentation inputs   | `nucleus_segmentation_inputs_widget.section`    | `_on_preview_contour_maps` | `_on_build_segmentation_inputs` |
| Ultrack database      | `nucleus_tracking_inputs_widget.db_section`     | —                | `_on_run_db_generation`    |
| Ultrack solve         | `nucleus_tracking_inputs_widget.solve_section`  | —                | `_on_run_ultrack`          |

All three handlers already live on `NucleusPipelineWidget` and are called by
the existing chain dispatcher; they remain unchanged.

## Run state & cancel

While a stage runs:

- That row's ▶ icon swaps to ✕; clicking it calls `_on_cancel`.
- The other two stages' ⚙ and ▶ icons are disabled. The currently-running
  row's ⚙ stays enabled (so the user can still peek at params).
- When the worker finishes (success, error, or cancel), all rows return to
  their idle ▶ state and re-enable.

This is handled by replacing `_set_pipeline_buttons_enabled(bool)` with
`_set_running_stage(stage_key | None)`:
- `None` → idle, all ▶ enabled, no ✕.
- `"seg" | "db" | "ultrack"` → that row shows ✕, others disabled.

`_on_contour_worker_error`, `_on_db_gen_worker_error`,
`_on_ultrack_worker_error`, `_on_db_gen_done`, `_on_run_ultrack_done`,
`_on_cancel`, and `_done` (in `_on_build_segmentation_inputs`) all call
`_set_running_stage(None)` instead of `_set_pipeline_buttons_enabled(True)`.

## Code to remove

In `nucleus_pipeline_widget.py`:
- `stage_seg_check`, `stage_db_check`, `stage_ultrack_check` (the QCheckBoxes)
- `_chain_remaining`, `_on_run_chain`, `_STAGE_DISPATCH`,
  `_run_next_chain_step`, `_chain_continue_or_finish`, `_abort_chain`
- `run_btn` (the chained Run QPushButton) and the shared `cancel_btn` —
  replaced by per-row run/cancel toggles
- `build_pipeline_block()` is rewritten to produce the three stage rows

In `nucleus_workflow_widget.py`:
- The `_alias_pipeline_controls` entries for the removed attributes
  (`stage_seg_check`, `stage_db_check`, `stage_ultrack_check`, `run_btn`,
  `cancel_btn`, `_on_run_chain`).
- The chain-step _done methods stop calling `_chain_continue_or_finish` and
  just call `_set_running_stage(None)`.

## Code to keep

- All worker bodies (`_on_build_segmentation_inputs`, `_on_run_db_generation`,
  `_on_run_ultrack`, `_on_preview_contour_maps`), their progress/yield
  contracts, and their error handlers.
- All params widgets and their `*_config` / `*_thresholds_from_controls`
  helpers.
- The shared `pipeline_status_lbl` and `pipeline_progress_bar` widgets and
  helper methods (`_status`, `_progress`, `_on_progress`, `_clear_progress`).
- DB browser and Correction sections.

## Files changed

1. `src/cellflow/napari/nucleus_pipeline_widget.py` — rewrite
   `build_pipeline_block()`; remove chain machinery; introduce
   `_set_running_stage`; create three per-stage rows with their own
   run/preview/params buttons.
2. `src/cellflow/napari/nucleus_workflow_widget.py` — replace the three
   `root.addWidget(self.tracking_db_section)` / `tracking_solve_section` and
   the seg section additions with whatever `build_pipeline_block()` now
   returns; drop removed aliases.
3. `src/cellflow/napari/nucleus_segmentation_inputs_widget.py` and
   `nucleus_tracking_inputs_widget.py` — no changes; the workflow widget
   calls `section.set_header_visible(False)` on the three existing sections
   after construction.
4. `src/cellflow/napari/widgets.py` — add a `set_header_visible(bool)` method
   on `CollapsibleSection` for the workflow widget to call.
5. `tests/napari/test_nucleus_tracking_inputs_widget.py`,
   `tests/napari/test_nucleus_tracking_correction_layout.py`,
   `tests/napari/test_nucleus_pipeline_widget.py` — adjust assertions that
   reference removed widgets (`stage_seg_check`, `run_btn`, etc.) to the new
   per-stage attributes.

## Naming for the new widgets (on `NucleusPipelineWidget`)

```
self.seg_params_btn       # ⚙ on segmentation row
self.seg_preview_btn      # ▷ on segmentation row (formerly preview_contour_btn)
self.seg_run_btn          # ▶/✕ on segmentation row

self.db_params_btn        # ⚙ on db row
self.db_run_btn           # ▶/✕ on db row

self.solve_params_btn     # ⚙ on solve row
self.solve_run_btn        # ▶/✕ on solve row

self._running_stage: str | None  # "seg" | "db" | "ultrack" | None
```

`preview_contour_btn` is kept as an alias of `seg_preview_btn` so existing
tests/state references still resolve while the new name is preferred for new
code.

## Testing

- **New tests:** clicking each ⚙ toggles the corresponding section's expanded
  state; clicking each ▶ invokes the right handler; while a worker is in
  flight, the running row shows ✕ and the others are disabled.
- **Updated tests:** anything that today asserts on the three checkboxes or
  the shared Run button is rewritten in terms of the new per-row buttons.
- **Manual:** in napari, open a position, click each ⚙ and verify the params
  block opens/closes inline; run each stage; cancel a run mid-flight; verify
  status text + progress still update.

## Open questions

None at design time. Glyphs (⚙ / ▷ / ▶ / ✕) and the exact tooltip wording can
be tuned during implementation.
