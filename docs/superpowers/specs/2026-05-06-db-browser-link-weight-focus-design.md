# Database Browser Link Weight Focus Design

## Goal

Enhance the Ultrack database browser so a user can inspect temporal link evidence around one selected database node. The browser should keep the current napari viewer workflow, update its existing preview layer in place, and make link weights visible without drawing a dense global edge graph.

## User Experience

The database browser gains a connected-node focus workflow:

- In normal mode, the browser keeps its current behavior: it renders all hierarchy-cut nodes for the current frame.
- With connected focus enabled and no node selected, the preview remains usable and the status prompts the user to click a DB preview node.
- Clicking a rendered DB preview node selects its underlying `NodeDB.id`, not its temporary display label.
- The selected node is highlighted with the same cyan contour style used by the correction widget.
- When the viewer remains on the selected node's frame, the preview shows the selected node.
- When the viewer moves to the previous frame, the preview shows only predecessor nodes connected to the selected node.
- When the viewer moves to the following frame, the preview shows only successor nodes connected to the selected node.
- Other frames show an empty connected preview with status text explaining that connected focus is anchored at the selected frame.

## Controls

Add three database-browser checkboxes:

- `Connected focus`: filters the DB preview to the selected node and its temporal neighbors.
- `Edge weight transparency`: modulates connected-node opacity by the product of edge weights between the selected node and that connected node.
- `Node prob transparency`: modulates node opacity by normalized `NodeDB.node_prob`.

The two transparency controls can be combined. If both are disabled, connected nodes render at full opacity. The selected node is always opacity `1.0` in edge-weight mode because it is the anchor, not evidence reached through an edge.

## Data Model

Hierarchy rendering should return metadata in addition to the rendered label image:

- `display_label -> NodeDB.id`
- `display_label -> node_prob`
- `NodeDB.id -> display_label`

Selection stores:

- selected `NodeDB.id`
- selected frame `t`
- selected display label for the current render, when available

The display label is render-local and may change when the hierarchy threshold changes. Persistent selection must therefore use `NodeDB.id`.

## Link Query Semantics

For selected node `S` at frame `t`:

- predecessors are rows where `LinkDB.target_id == S`
- successors are rows where `LinkDB.source_id == S`

The focused preview uses predecessors when the viewer frame is `t - 1`, successors when the viewer frame is `t + 1`, and the selected node itself when the viewer frame is `t`.

If multiple links connect the selected node and the same neighbor, their weights are multiplied for edge transparency. Final alpha is clamped to a readable range so weak but present connections remain visible.

## Rendering

The existing `Ultrack DB Preview` layer remains the main preview layer. When either transparency checkbox is enabled, the preview should use an RGBA image layer. Otherwise it can use a labels layer as today.

Per-node alpha:

```text
selected node:
  alpha = 1.0

connected node:
  alpha = 1.0
  if edge weight transparency:
      alpha *= product(edge weights between selected node and connected node)
  if node prob transparency:
      alpha *= normalized node_prob
  alpha = clamp(alpha, min_visible_alpha, 1.0)
```

The selected-node contour should be a separate shapes layer, using the correction widget pattern: compute the selected mask contour with `skimage.measure.find_contours`, draw the largest contour, and keep it cyan with transparent fill.

## Refresh Behavior

Connected focus updates in place on:

- browser refresh
- hierarchy slider changes
- transparency checkbox changes
- connected-focus checkbox changes
- viewer frame changes
- successful DB preview node selection

If the selected node is no longer present at the current hierarchy threshold, keep the selected `NodeDB.id` but clear the contour and show status text explaining that the selected node is hidden at the current threshold.

## Status Text

Status should include the selected node id, anchor frame, current relation, connected-node count, and edge-weight range when relevant. Example:

```text
Selected node 12873 at t=42 | t+1: 3 connected nodes | edge product range 0.21-0.86
```

## Testing

Add focused widget tests for:

- the browser exposes the new connected-focus and edge-weight controls
- hierarchy rendering produces `display_label -> NodeDB.id` metadata
- clicking a DB preview label stores the selected `NodeDB.id`
- connected focus renders only predecessors, the selected node, or successors depending on the current frame
- edge-weight transparency makes stronger connected nodes more opaque while keeping the selected node at alpha `1.0`
- edge-weight and node-prob transparency combine multiplicatively
- disabling connected focus preserves the existing all-nodes hierarchy-cut behavior

