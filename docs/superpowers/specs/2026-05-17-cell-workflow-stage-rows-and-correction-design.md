# Cell Workflow Widget - Stage Rows and Correction Header

**Date:** 2026-05-17
**Scope:** `src/cellflow/napari/cell_workflow_widget.py`,
`cell_params_widget.py`, `cell_correction_widget.py`, and affected napari tests.

## Problem

The cell workflow still uses the older flat action-button grid: five large
pipeline buttons sit above one shared `Parameters` collapsible section, while
the correction widget is appended as a separate child widget. The recently
overhauled nucleus workflow is more direct: each stage has its own row, the
stage's parameters open inline under that row, status/progress are shared, and
correction has a stage-style header with icon controls that expand a shared
correction content section.

The cell workflow should follow that same interaction pattern without changing
the underlying pipeline operations or correction behavior.

## Goals

- Replace the cell pipeline button grid with per-stage rows.
- Put each stage's relevant parameters inline under that stage row, collapsed
  by default and controlled by a gear button.
- Keep the existing pipeline semantics: no new chained run behavior and no
  implicit merging of stages.
- Add the same per-row run/cancel behavior used by the nucleus workflow.
- Restructure cell correction to match the nucleus correction presentation:
  a stage-style correction header with shortcuts, params, and activation icons.
- Preserve existing state keys, control aliases, worker handlers, artifact
  paths, and correction operations.

## Non-goals

- No change to segmentation algorithms, correction algorithms, or file paths.
- No conversion of correction actions into pipeline stages.
- No change to saved state schema.
- No rewrite of the shared `CorrectionWidget`.
- No visual redesign beyond applying the established nucleus row/header pattern
  with the cell accent color.

## Final Layout

```text
Pipeline Files                                  [unchanged external header]
──────────────────────────────────────────────
Flow filtering                    ⚙       ▶
  ┊ flow filtering params, collapsed by default
Foreground masks                  ⚙       ▶
  ┊ foreground params, collapsed by default
Contours                          ⚙   ▷   ▶
  ┊ contour params, collapsed by default
Segmentation                      ⚙       ▶
  ┊ segmentation params, collapsed by default
[██████░░░░] Building contours...             [shared progress + status]
──────────────────────────────────────────────
Correction                        📖   ⚙   ⏻
  ┊ correction params, shortcuts, and/or active correction controls
```

The correction area mirrors nucleus: `CellCorrectionWidget` remains the owner
of correction behavior, but the workflow widget hides the owner widget and
adds its visible header/content pieces to the main layout.

## Pipeline Stage Rows

Each pipeline row is a horizontal layout:

| Element | Source |
| --- | --- |
| Stage label | `QLabel` styled by `stage_header_label(label, "cell")` |
| Parameters icon | Checkable `QToolButton` with `⚙`; toggles inline section |
| Preview icon | `QToolButton` with `▷`; contours row only |
| Run/cancel icon | `QToolButton` with `▶` when idle and `✕` while running |

Stage mapping:

| Row label | Params section | Preview | Run handler |
| --- | --- | --- | --- |
| Flow filtering | `cell_params_widget.flow_filter_section` | - | `_on_filter_flow` |
| Foreground masks | `cell_params_widget.foreground_section` | - | `_on_build_foreground` |
| Contours | `cell_params_widget.contour_section` | `_on_preview_contours` | `_on_build_contours` |
| Segmentation | `cell_params_widget.segmentation_section` | - | `_on_segment` |

The old handler methods stay in place. New click dispatchers only decide
whether to start the stage or cancel the current run, then delegate to those
existing handlers.

## Cell Parameter Sections

`CellParamsWidget` will keep owning all cell pipeline parameter controls, but
it will expose four `CollapsibleSection` instances instead of one combined
`section`:

- `flow_filter_section`: median and Gaussian flow filtering controls.
- `foreground_section`: cellprob threshold.
- `contour_section`: cellprob sweep, flow-following, gamma averaging, and
  temporal memory controls.
- `segmentation_section`: ICM segmentation controls.

Compatibility aliases on `CellWorkflowWidget` remain unchanged, so callers and
tests can still access controls such as `ff_median_time_spin`,
`fg_cellprob_threshold_spin`, `cp_min_spin`, and `alpha_unary_spin` directly.
`get_state()` and `set_state()` keep their existing top-level keys:
`flow_filtering`, `foreground`, `contour`, `segmentation`, and `correction`.

The old `cell_params_widget.section` remains as a compatibility alias that
contains the four stage-specific sections. The workflow layout uses the four
stage-specific sections directly.

## Run State and Cancel

Add `CellWorkflowWidget._running_stage: str | None` with stage keys:
`"flow"`, `"foreground"`, `"contour"`, and `"segmentation"`.

`_set_running_stage(stage_key)` follows the nucleus behavior:

- `None`: all stage gear and run buttons are enabled and show `▶`.
- Running stage: that row's run button shows `✕` and remains enabled; its gear
  stays enabled; other stage gear/run buttons are disabled.
- Preview stays disabled while any pipeline stage is running.

Cancellation calls a new `_on_cancel()` helper that quits the currently running
worker if one exists. Existing worker completion and error callbacks reset the
running stage to `None`, clear progress, and refresh files as they do today.

## Correction Header and Content

`CellCorrectionWidget` will adopt the nucleus correction presentation:

- `header`: stage-style row with `Correction` label using the cell accent.
- `shortcuts_btn`: `📖`, checkable; shows correction shortcuts.
- `params_btn`: `⚙`, checkable; shows correction parameters.
- `active_btn`: `⏻`, checkable; activates correction mode.
- `section`: hidden-header `CollapsibleSection` that expands when params,
  shortcuts, or active correction content is visible.

The workflow widget will alias these controls similarly to nucleus:

- `correction_header`, `correction_header_lbl`
- `correction_shortcuts_btn`, `correction_params_btn`, `correction_active_btn`
- `correction_mode_section`
- existing correction operation aliases such as `load_labels_btn`,
  `save_labels_btn`, `fill_holes_btn`, `fix_semiholes_btn`, `cleanup_btn`, and
  `expand_cell_btn`

Correction actions remain correction-scoped and are not disabled by pipeline
run state unless an existing correction handler already does so.

## Testing

Update or add focused tests for:

- Cell workflow exposes `QToolButton` stage row controls rather than the old
  `QPushButton` grid.
- Each stage gear toggles only its matching params section.
- Contours row exposes both preview and run buttons.
- `_set_running_stage(...)` swaps the active run button to `✕` and disables
  other rows.
- Cell correction exposes the nucleus-style header controls and expands the
  shared correction section when params, shortcuts, or activation are toggled.
- Existing state round trips and correction aliases continue to work.

Manual verification in napari:

- Open a position and expand each pipeline gear.
- Run each pipeline stage and cancel one in flight.
- Use contour preview.
- Toggle correction shortcuts, correction params, and correction activation.
- Confirm correction load/save/fill/fix/cleanup/expand actions still operate.

## Implementation Notes

- Reuse `_widget_helpers.tool_btn`, `_widget_helpers.make_status`,
  `_widget_helpers.make_progress`, `stage_header_label`, and
  `CollapsibleSection.set_header_visible(False)`.
- Keep edits scoped to the cell workflow and cell correction widgets plus their
  tests.
- Preserve old private handler names to avoid unnecessary churn.
