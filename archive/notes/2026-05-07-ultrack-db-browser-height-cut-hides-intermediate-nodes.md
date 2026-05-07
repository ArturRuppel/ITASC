# Ultrack DB Browser Height Cut Hides Intermediate Nodes

**Date:** 2026-05-07  
**Status:** diagnosis confirmed; fix pending

## Summary

The apparent Ultrack hierarchy problem is not caused by missing intermediate
watershed candidates in the tested case. The intermediate candidates are present
in the hierarchy and in `data.db`. The issue is in the napari Ultrack Database
Browser's hierarchy-cut rendering logic.

The browser uses `NodeDB.height` as if it were a unique hierarchy altitude/cut
coordinate. In Ultrack, `height` is Higra `attribute_height`, and parent/child
nodes can validly share the same height. When that happens, the browser's SQL
cut query hides equal-height intermediate children and jumps visually from
smaller fragments directly to the full merged component.

## Confirmed Example

Dataset:

```text
/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos02/2_nucleus/
```

Frame and region inspected:

```text
frame = 25
ROI = y=140:200, x=170:230
```

The foreground connected component intersecting the two cells is CC 13:

```text
CC area: 1199
tracked label 139 area: 433
tracked label 160 area: 766
union(label 139, label 160) area: 1199
IoU(CC, union): 1.0
```

Directly building the Ultrack hierarchy for this component with
`min_frontier=0.0` emits the expected intermediate candidates:

```text
area 1199  full merged component
area 771   IoU 0.994 with tracked label 160
area 428   IoU 0.988 with tracked label 139
area 402   subfragment of tracked label 160
area 369   subfragment of tracked label 160
```

The same nodes are present in the correct database:

```sql
SELECT id, t_node_id, t_hier_id, area, frontier, height, hier_parent_id, y, x
FROM nodes
WHERE t = 25
  AND area IN (1199, 771, 428, 402, 369)
ORDER BY area DESC;
```

Observed rows:

```text
area 1199  height 392  parent -1
area 771   height 392  parent 1199
area 428   height 392  parent 1199
area 402   height 38   parent 771
area 369   height 38   parent 771
```

So the real candidate structure is:

```text
402 / 369 fragments -> 771 / 428 cell-level candidates -> 1199 full merge
```

## Browser Bug

The browser queries distinct `NodeDB.height` values for its slider:

```python
session.query(NodeDB.height).distinct().order_by(NodeDB.height)
```

Then `_render_hierarchy_cut()` selects displayed nodes with:

```python
.where(NodeDB.height <= h_actual)
.where((NodeDB.hier_parent_id == NO_PARENT) | (P.height > h_actual))
```

Source location:

```text
src/cellflow/napari/nucleus_workflow_widget.py
```

This fails for equal-height parent/child plateaus. In the confirmed example:

```text
root 1199 height = 392
child 771 height = 392
child 428 height = 392
```

At `h_actual = 392`, the children satisfy `NodeDB.height <= h_actual`, but their
parent does **not** satisfy `P.height > h_actual` because `392 > 392` is false.
Therefore the browser hides the intermediate `771` and `428` nodes and displays
only the root. At the lower slider value it displays the smaller fragments. The
UI therefore appears to jump:

```text
fragments -> full merged component
```

even though the database contains the intermediate candidates.

## Why The Previous Diagnosis Was Too Narrow

The earlier "ridge leak" hypothesis claimed that the watershed hierarchy skipped
the intermediate merge level because the contour graph crossed the middle ridge
too early. That can happen in principle, but it is not the explanation for the
confirmed frame/region above.

Synthetic Higra probes showed:

- A clean contour-threshold split is preserved by `watershed_hierarchy_by_area`.
- Low or tied cross-edges can remove the intended split.
- However, in the inspected real ROI the intermediate split is present.

Therefore, for this concrete case, the issue is a visualization/query problem in
the database browser, not candidate generation.

## Implications

Do not use the current DB Browser hierarchy-cut view as proof that intermediate
nodes are missing from `data.db`.

Before diagnosing segmentation, inspect the actual nodes or build the hierarchy
directly. For this case, both show the expected intermediate nodes.

## Likely Fix Directions

The browser needs a rendering model that does not treat `NodeDB.height` as a
strict unique cut altitude. Possible approaches:

- Add an "all candidates" mode that paints all nodes or selected hierarchy groups
  without attempting a height cut.
- Use a tree-depth / hierarchy-level traversal instead of `height`.
- Reconstruct per-hierarchy parent/child relationships from `hier_parent_id` and
  choose nodes with explicit equal-height plateau handling.
- If a cut slider is kept, use a tie-aware rule or a stable secondary ordering so
  parent and child nodes with equal `height` do not collapse incorrectly.

Any fix should include a regression test with a parent and child sharing the same
`height`, where the child must still be displayable.

