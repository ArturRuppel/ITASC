# Ultrack DB Hierarchy Browser — Design Spec
**Date:** 2026-05-03

## Overview

Simplify the Ultrack Database Browser section from six render modes down to two: **Summary only** and **Hierarchy cut**. The hierarchy cut mode paints a non-overlapping 2D label image for the current frame at a user-chosen position in the watershed dendrogram, live-updating when the frame or slider changes.

## Background — ultrack hierarchy

Each frame's foreground is segmented into a watershed hierarchy (via `higra`). Each hierarchy node is stored in `NodeDB` with:

- `height` — persistence value (altitude span of the node in the dendrogram). Lower threshold → more, smaller segments; higher threshold → fewer, larger segments.
- `hier_parent_id` — parent in the watershed tree (`-1` = root).
- `pickle` — auto-unpickled by `MaybePickleType` to an object with `.bbox` and `.mask`.

A **horizontal cut at threshold H** gives a non-overlapping set of segments: keep node N iff `N.height <= H AND (N.hier_parent_id == -1 OR parent.height > H)`.

## UI Changes

**Remove:**
- Modes: "Selected solution", "Single node", "Link neighborhood", "Overlap neighborhood"
- Controls: `ultrack_db_threshold_spin`, `ultrack_db_node_id_spin`, `ultrack_db_limit_spin`

**Keep:**
- Mode combo: `["Summary only", "Hierarchy cut"]`
- Summary info label
- Refresh button
- Status label

**Add:**
- `QSlider` (integer 0–100, maps to 0.0–1.0) — visible only in "Hierarchy cut" mode
- Read-only `QLabel` beside slider showing mapped actual height value, e.g. `→ 42.3`

Layout: summary info label on top row; slider + height label on second row (hidden in "Summary only" mode).

## Data Flow

### Height range (cached)

On first "Hierarchy cut" render, query:
```sql
SELECT MIN(height), MAX(height) FROM nodes
```
Cache result keyed by `(db_path, mtime_ns)`. Used to map slider int value `v` (0–100) to:
```
t = v / 100.0
h_actual = h_min + t * (h_max - h_min)
```
The height label updates immediately on slider drag (no DB hit).

### Dendrogram-cut query

```python
from sqlalchemy.orm import aliased
P = aliased(NodeDB)
nodes = (
    session.query(NodeDB)
    .outerjoin(P, NodeDB.hier_parent_id == P.id)
    .where(NodeDB.t == frame)
    .where(NodeDB.height <= h_actual)
    .where((NodeDB.hier_parent_id == NO_PARENT) | (P.height > h_actual))
    .all()
)
```

`NO_PARENT = -1`. Result is a non-overlapping set of watershed segments for the frame.

### Painting

Reuse `_node_mask_and_bbox` (already fixed to use `MaybePickleType`-unpickled objects) and `_paint_ultrack_db_nodes`.

### Caching

Cache key: `(str(db_path), mtime_ns, frame, slider_int_value)`. Clear on any param change (existing `_on_ultrack_db_browser_param_changed`).

### Live updates

- **Slider drag:** 150ms debounce via `QTimer.singleShot`. Height label updates immediately (no debounce).
- **Frame step:** `_on_dims_step_changed` checks if browser section is expanded and mode is "Hierarchy cut"; if so, calls `_refresh_ultrack_db_browser` immediately (bypasses debounce).

## Removed Helpers

These methods are deleted entirely (no longer needed):

- `_node_frame_filter`
- `_node_score`
- `_selected_node_value`
- `_link_source_column`
- `_link_target_column`
- `_overlap_columns`
- The old `_render_ultrack_db_preview` (replaced by a new `_render_hierarchy_cut`)

## Error Handling

- DB missing: existing "data.db not found" message, no render attempt.
- Frame has no nodes at the cut level: status message "No segments at this threshold for frame N.", return empty array.
- `h_min == h_max` (degenerate DB): treat entire slider range as `h_actual = h_min`, paint all nodes for the frame.

## Testing

Update existing tests that reference removed widgets/modes. New tests:

1. **Height label mapping** — given `h_min=10, h_max=110`, slider at 50 → label shows `→ 60.0`.
2. **Dendrogram cut non-overlap** — fabricate NodeDB rows with parent/child heights; verify query returns only cut-level nodes.
3. **Frame step triggers repaint** — when section is expanded and mode is "Hierarchy cut", `_on_dims_step_changed` calls `_refresh_ultrack_db_browser`.
4. **DB missing** — graceful status message, no exception.
