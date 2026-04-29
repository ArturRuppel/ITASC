# UI Style Foundation Design

**Date:** 2026-04-29

## Goal

Add a small shared UI helper layer for the napari plugin so common spacing, button sizing, muted text, status text, and danger styling are applied consistently without a broad redesign.

## Scope

This pass is intentionally conservative. It introduces reusable style/layout helpers and applies them to the highest-friction inconsistencies:

- shared file/status rows in `widgets.py`
- project header controls in `main_widget.py`
- data preparation metadata/status labels in `data_prep_widget.py`
- Cellpose descriptive text in `cellpose_widget.py`
- correction activation/status/danger affordances in `correction_widget.py`
- danger and action buttons in nucleus/cell workflow database panels

Out of scope:

- redesigning the workflow hierarchy
- rewriting `PipelineFilesWidget` layout or behavior
- removing duplicated correction surfaces
- changing backend behavior
- replacing napari/Qt theme styling globally

## Design

Create `src/cellflow/napari/ui_style.py` as a small, dependency-light module containing semantic helpers:

- compact margins and spacing constants
- `compact_spinbox(...)`
- `action_button(...)`
- `tiny_button(...)`
- `icon_button(...)`
- `muted_label(...)`
- `status_label(...)`
- `danger_button(...)`
- `checked_success_button(...)`

Helpers should return the widget they modify so callers can use them inline. Styles should prefer Qt palette colors where possible. Explicit colors are acceptable only for semantic states such as success and danger.

Existing widgets should keep their current layout and behavior. This change is mostly about replacing ad hoc inline styles and one-off size policies with named helpers.

## Testing

Add focused tests for the helper module and migrate existing napari layout tests without changing their expected behavior. Verification should include:

- helper functions set the expected size policies and styles
- existing napari layout tests still pass
- Python compilation succeeds for modified UI modules

