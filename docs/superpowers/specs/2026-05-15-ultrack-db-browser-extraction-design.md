# Ultrack DB Browser Extraction Design

## Goal

Extract the Ultrack database browser slice out of `src/cellflow/napari/nucleus_workflow_widget.py` into a focused section widget without changing user-visible behavior.

## Architecture

Create `src/cellflow/napari/nucleus_db_browser_widget.py` with `NucleusUltrackDbBrowserWidget`. The new widget owns the DB-browser controls, preview caches, selection state, layer rendering helpers, and refresh/activation behavior. `NucleusWorkflowWidget` remains the workflow orchestrator and provides only the context the browser cannot own: current position directory, current viewer frame, viewer-frame setter, and viewer layers.

## Compatibility

The first extraction keeps existing public attribute names on `NucleusWorkflowWidget`, including `ultrack_db_browser_section`, `ultrack_db_info_lbl`, `ultrack_db_active_btn`, `ultrack_db_hierarchy_slider`, and related controls. It also keeps parent forwarding methods for private names already used by tests, so the broader test file can keep passing while DB-browser tests are moved in a later cleanup.

## Scope

In scope:
- Move DB-browser UI creation from `_build_db_browser_section`.
- Move DB-browser state from `NucleusWorkflowWidget.__init__`.
- Move behavior methods from `_set_ultrack_db_status` through `_node_mask_and_bbox`.
- Add a focused extraction test that fails until the child widget exists and owns the section.

Out of scope:
- Rewriting all existing DB-browser tests in this pass.
- Changing DB rendering behavior, layer names, cache keys, annotation filters, or hierarchy slider semantics.
- Splitting correction or DB-generation controls.

## Testing

Use TDD for the extraction seam: add a test asserting that `NucleusWorkflowWidget` exposes `ultrack_db_browser_widget` as a `NucleusUltrackDbBrowserWidget`, while the stable legacy attributes still point at the child-owned controls. Then run the focused DB-browser/layout tests and the full `tests/napari/test_nucleus_tracking_correction_layout.py` file.

