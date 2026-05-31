# Spec: Database Browser — local node-link panel

Add a pyqtgraph node-link panel to the Ultrack Database Browser that renders
the local graph structure (predecessors / successors) and per-link edge
weights for the currently selected node. It augments the existing in-canvas
preview; it does not replace any current behavior.

## Background

The browser (`src/cellflow/napari/nucleus_db_browser_widget.py`) already lets
the user click a candidate in the `[Database] Ultrack DB Preview` labels layer
to select a `NodeDB.id`. On select, `_select_ultrack_db_preview_label`
(browser:753) sets `_ultrack_db_selected_node_id` and writes a one-line status:
node id, annotation, `p=`, and aggregate link annotation counts.

The local graph is already one query away. `query_connected_nodes`
(`src/cellflow/tracking_ultrack/db_query.py:77`) returns
`predecessors {neighbor_id: weight}` and `successors {neighbor_id: weight}`
from `LinkDB`. The "Connected focus" mode (`_render_ultrack_db_connected_focus`,
browser:870) paints neighbor masks frame-by-frame and reduces the weights to a
single `edge range min–max` string (browser:921-927). That is the only place
edge weights surface today, and it is lossy.

Two limitations motivate this spec:

- **Weights are not legible.** Only a min–max range is shown, never the
  per-edge value, annotation, or which neighbor it belongs to.
- **napari slicing fights a flat graph.** Neighbors live in adjacent frames
  (t−1 / t+1), so an in-canvas node-link diagram is never fully visible at one
  time point. A panel decoupled from the time slider sidesteps this.

`query_connected_nodes` also **multiplies** weights when a neighbor pair has
more than one link (db_query.py:100-103). That product is fine for the focus
overlay but wrong for a per-edge readout — this spec needs raw per-link rows.

## What

A read-only pyqtgraph panel inside the Database Browser section that, whenever a
node is selected, draws:

- the selected node centered, its predecessors (t−1) in a left column, its
  successors (t+1) in a right column;
- one edge per link, width and color encoding the **raw** link weight, with the
  numeric weight drawn at the edge midpoint;
- node fill encoding annotation (REAL / FAKE / UNKNOWN), node size encoding
  `node_prob`.

Clicking a node in the panel selects it in the canvas (and vice versa). The
panel is empty with a "Click a preview node" hint when nothing is selected.

Decided in design discussion: primary goal is **reading edge weights
precisely**; renderer is **pyqtgraph** (over matplotlib / napari-only layers).

## New query helper

Add to `db_query.py`, alongside `query_connected_nodes`:

```
def query_node_edges(db_path, node_id) -> NodeEdges
```

Returns a small frozen dataclass:

```
@dataclass(frozen=True)
class NodeEdge:
    neighbor_id: int
    weight: float          # raw LinkDB.weight (NOT a product)
    annotation: str        # annotation_name(LinkDB.annotation): REAL/FAKE/UNKNOWN
    direction: str         # "pred" (neighbor -> selected) or "succ" (selected -> neighbor)
    neighbor_t: int        # NodeDB.t of the neighbor
    neighbor_prob: float   # NodeDB.node_prob (1.0 if NULL)
    neighbor_annot: str    # annotation_name(NodeDB.node_annot)

@dataclass(frozen=True)
class NodeEdges:
    selected_id: int
    selected_t: int
    selected_prob: float
    selected_annot: str
    edges: tuple[NodeEdge, ...]
```

Implementation notes:

- One `LinkDB` query filtered by `(source_id == node_id) | (target_id ==
  node_id)`; **do not collapse** duplicate neighbor pairs — each link is its own
  `NodeEdge`. `source_id == node_id` → `direction="succ"`; `target_id ==
  node_id` → `direction="pred"`.
- Join / second query `NodeDB` for `t`, `node_prob`, `node_annot` of the
  selected node and every neighbor.
- Reuse `annotation_name` (db_query.py:725) for all annotation mapping.
- Engine lifecycle and `try/finally engine.dispose()` follow the existing
  helpers exactly. Pure read; no schema writes.

`query_connected_nodes` stays as-is (the focus overlay still uses it).

## Widget

In `NucleusUltrackDbBrowserWidget.__init__`, after the checkbox block, add a
collapsible row containing a `pyqtgraph.GraphicsLayoutWidget` with a single
`ViewBox` (locked aspect, mouse-pan/zoom enabled, no axes). Inside it:

- one `pyqtgraph.GraphItem` for all nodes + edges;
- a list of `pyqtgraph.TextItem` for edge weight labels (GraphItem cannot label
  edges itself) and node id labels;
- fixed minimum height (~180 px) so it does not collapse the controls.

Expose the panel via the existing alias pattern (`_alias_ultrack_db_browser_*`)
and gate it with the browser active state in `_set_ultrack_db_controls_enabled`.

### Layout (deterministic, not force-directed)

Layered, no networkx needed:

- selected node at `x = 0, y = 0`;
- predecessors at `x = -1`, stacked at `y = 0, ±1, ±2, …` in stable id order;
- successors at `x = +1`, stacked the same way;
- vertically center each column.

Force-directed layout is rejected — it is unstable across reselections and
makes t−1/t/t+1 position meaningless.

### Encoding

- **Edge width** ∝ weight, linearly mapped from the panel's current weight range
  to e.g. `[1, 6] px`. Single-edge case maps to mid width.
- **Edge color** by a perceptual colormap over the same weight range; if you
  prefer annotation over magnitude, color by `NodeEdge.annotation` instead —
  pick one and keep node color free for the other. Default: color edges by
  weight, nodes by annotation.
- **Edge label**: `f"{weight:.2f}"` `TextItem` at the edge midpoint.
- **Node fill**: REAL / FAKE / UNKNOWN, reusing `_ULTRACK_DB_ANNOTATION_COLORS`
  so the panel matches the annotation overlay. Selected node gets a distinct
  outline (cyan, matching the `[Database] Ultrack DB Selection` highlight).
- **Node size** ∝ `node_prob` over `[min, max]` of the visible nodes, clamped to
  a sane radius range.
- **Node label**: node id (small, below the node).

## Interaction

- **Canvas → panel**: in `_select_ultrack_db_preview_label` (browser:753) and
  `_deselect_ultrack_db_node` (browser:780), call a new
  `_refresh_node_graph_panel()`. It runs `query_node_edges`, rebuilds the
  GraphItem, and updates the TextItems. Deselect clears to the hint state.
- **Panel → canvas**: connect node-click (GraphItem scatter `sigClicked`) to a
  handler that maps the clicked node id back to a display label via
  `_ultrack_db_node_id_to_label` and calls `_select_ultrack_db_preview_label`,
  so clicking a neighbor in the panel selects it in the canvas. If that neighbor
  is in a different frame (t−1 / t+1), advance the viewer frame first via
  `_set_viewer_frame` before selecting.
- Refresh is cheap, but debounce through the existing
  `_ultrack_db_refresh_timer` pattern if needed. No new DB writes ever.

## Dependency

`pyqtgraph` is **not** currently installed and is not in `uv.lock`. Add it to
the napari extra in `pyproject.toml` and re-lock. It is pure-Python, works on
the existing PyQt5 / qtpy stack, and has no heavy transitive deps. Given the
known lock drift (`uv sync --frozen`), the lockfile update must land in the same
change. Import `pyqtgraph` lazily inside the widget module (module-level import
guarded the way other optional napari deps are) so a missing dep degrades to
"panel unavailable" rather than breaking the whole widget.

## Edge cases

- **No selection / deselect** → hint text, no nodes.
- **Selected node has no links** → draw the single selected node, label "no
  links".
- **Neighbor hidden** at the current union-size / merge-group or annotation
  filter → still draw it in the panel (the panel is graph-truth, independent of
  what the canvas paints); panel→canvas click on a hidden neighbor falls back to
  the existing "is hidden" status message instead of selecting.
- **NULL weight** → treat as `1.0` (matches `query_connected_nodes`,
  db_query.py:97) and note it (e.g. dashed edge).
- **Many neighbors** (>~20 per side) → cap the drawn count, sort by weight
  descending, and label the panel with "showing top N of M". No silent
  truncation.
- **Self / degenerate links** (`source_id == target_id`) → skip.

## Testing

- `tests/.../test_db_query.py`: `query_node_edges` against a small fixture DB —
  asserts per-link rows are not collapsed, directions are correct, NULL weight →
  1.0, annotations map via `annotation_name`. Pure, no Qt.
- `tests/napari/test_nucleus_db_browser_widget.py`: extend with a headless
  (`pytest-qt`) test that selecting a node populates the GraphItem node/edge
  counts, deselect clears it, and a panel node-click routes to
  `_select_ultrack_db_preview_label`. Skip if `pyqtgraph` is unavailable.

## Out of scope

- t−2 / t+2 or deeper neighborhoods (only immediate predecessors/successors).
- Editing links or annotations from the panel (read-only).
- Replacing the Connected-focus overlay or the `edge range` status line — both
  stay.
- Real spatial centroids / 3D Tracks overlay (Options B/C from the design
  discussion) — deferred.
