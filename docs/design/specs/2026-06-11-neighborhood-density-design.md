# Neighborhood & Density Analysis — Design Spec

Date: 2026-06-11

> **Status: design.** Aligns the *exact* calculations before building. Follows the
> same seams as the Potential Landscape increment
> (`2026-06-11-potential-landscape-contacts-design.md`): a headless derivation
> module on `PositionContactAnalysis`, the quantity-agnostic plotting backend
> (`aggregate_quantification/plotting.py`), and the pool-a-snapshot /
> launch-a-panel thin-plugin pattern of Potential Landscape
> (`napari/aggregate_quantification/plugins/contact_energetics.py`) and Track
> Dynamics.

The next Aggregate Quantification analysis: **neighborhood and density** of the
cell–cell contact network. It answers three questions, pooled across every
in-scope position and groupable by cell-type label:

1. **How many neighbors does each cell have?** (adjacency degree)
2. **How many cells per unit area?** (density, over a user-defined field of view)
3. **Do cell types sort or mix?** All cell-type contact permutations (aa, ab, bb)
   plotted as raw composition *and* normalized to what abundance alone predicts —
   both a per-cell **enrichment ratio** and a label-shuffle **z-score**.

The contact graph already exists: the `edges` table of `contact_analysis.h5`
carries one row per cell–cell (and per cell–border) boundary segment with
`cell_a`, `cell_b`, `kind`, `frame`. The `cell_cell` edges *are* the adjacency
graph; degree is the count of incident `cell_cell` edges. Cell-type labels come
from the existing NLS sidecar CSV (`cell_id → class_label`, the same source the
Cell Shape and NLS plugins join on).

## Decisions (locked with the user)

1. **Cell-type source: the NLS `class_label` sidecar CSV, exactly two types.**
   Reuse `contacts/nls_classification.py`'s
   `nls_classification_csv_path` / `read_nls_classification_csv`
   (`cell_id → label`). Two labels in practice (e.g. `positive`/`negative`); a
   position with >2 distinct labels is rejected for the typed views with a clear
   message. Cells with no label are **`unclassified`**: still counted as
   neighbors and as cells (degree, total density), but **excluded** from the
   enrichment and z-score maths (their edges are dropped; abundances are computed
   over labeled cells only).

2. **Normalization: both per-cell enrichment and permutation z-score.**
   - **Per-cell neighbor enrichment** — a per-cell ratio that flows through the
     existing per-cell plotting/grouping machinery as a *distribution*.
   - **Permutation z-score** — a per-position summary against a label-shuffle
     null; the rigorous "is this more than chance" statistic.

3. **Density is area-normalized with a user-entered field-of-view size.**
   `density = n_cells / fov_area_mm2`. Cells are often confined to a subregion,
   so the FOV area is a **plugin input (mm²)**, defaulting to the full image area
   (`H·W·pixel_size_um² / 1e6`). One global value applies to all pooled positions
   this increment (per-position entry is deferred).

4. **No new persisted artifact, no new plot mode.** Like Potential Landscape, the
   plugin reads the existing `contact_analysis.h5` (the contacts quantifier
   already builds it) and derives every table in memory at pool time. All four
   views land on the **existing** hist/box/violin/strip/swarm/bar machinery via
   preset `value` / `group_by` — no `plotting.py` plot-type additions.

5. **Reuse the generic `PlotPanel`.** No bespoke neighborhood panel / enrichment
   heatmap this increment. A "View" selector in the plugin pools the relevant
   table and opens the generic panel with sensible defaults, exactly as Track
   Dynamics opens distribution views.

## Layer 1 — headless derivations (`contacts/neighborhood.py`)

Qt-free, operates on an already-read `PositionContactAnalysis` plus an optional
`labels: Mapping[int, str]` (the NLS map) and a `fov_area_mm2: float | None`.
Returns tidy column-major tables (`dict[str, np.ndarray]`) shaped for
`PositionSource` / `pool_object_tables`. Each function is independently
unit-tested.

### Adjacency helper (shared, private)

```python
def _frame_adjacency(analysis) -> dict[int, dict[int, set[int]]]:
    """frame -> {cell_id -> set(neighbor cell_ids)} from cell_cell edges.

    Only ``kind == "cell_cell"`` rows; border edges are excluded. Each edge
    (cell_a, cell_b) adds each endpoint to the other's neighbor set, so a
    boundary fragmented into several edge rows still counts as one neighbor
    relation (set union dedupes). Degree = len(neighbors[cell_id]).
    """
```

### 1. Neighbor count (degree) — `cell_neighbor_counts`

```python
def cell_neighbor_counts(analysis) -> dict[str, np.ndarray]:
    """Per (frame, cell_id): n_neighbors (adjacency degree)."""
```

- Columns: `frame`, `cell_id`, `n_neighbors` (int).
- One row per cell present in `cells` for that frame (degree 0 for an isolated
  cell). No labels needed — `class_label` is attached later by the standard
  per-position join in `pool_object_tables`.
- This is the headline "how many neighbors each cell has," plottable as a
  per-cell distribution grouped by `class_label`, and the numerator for density.

### 2. Per-cell neighbor enrichment — `neighbor_enrichment`

```python
def neighbor_enrichment(analysis, labels) -> dict[str, np.ndarray]:
    """Long per (frame, cell_id, neighbor_label) enrichment table.

    For a focal cell of type i with degree d, in a frame with N labeled cells
    and N_j cells of type j:

        expected_j = d * f_j'      where
            f_j' = N_j / (N - 1)            for j != i
            f_i' = (N_i - 1) / (N - 1)      for j == i   (self-excluded)
        observed_j = # of the cell's neighbors that are type j
        enrichment = observed_j / expected_j      (NaN when expected_j == 0)
    """
```

- Columns: `frame`, `cell_id`, `focal_label`, `neighbor_label`, `observed`
  (int), `expected` (float), `enrichment` (float).
- Emitted **per labeled focal cell × per label j present in the frame**, so a
  two-type frame yields up to four `(focal,neighbor)` combinations: `aa`, `ab`,
  `ba`, `bb`. `ab` and `ba` cover the same physical edges seen from each
  endpoint; kept separate because their per-cell denominators differ, and they
  read as the three undirected types when symmetric.
- Edges to `unclassified` neighbors are dropped from `observed`; abundances
  `N`, `N_j` count labeled cells only. A focal `unclassified` cell contributes no
  rows.
- `enrichment > 1` ⇒ that neighbor type is over-represented around this cell
  (sorting); `< 1` ⇒ avoided (mixing).
- Plotted with `value="enrichment"`, `group_by=("focal_label","neighbor_label")`
  → an aa/ab/ba/bb distribution (box/violin/strip). The labels ride in the table,
  so no CSV join is needed for grouping.

### 3. Contact-type z-score — `contact_type_zscores`

```python
def contact_type_zscores(
    analysis, labels, *, n_shuffles: int = 1000, seed: int = 0
) -> dict[str, np.ndarray]:
    """Per (frame, contact_type) observed vs label-shuffle null.

    Each cell_cell edge between two labeled cells is typed by its sorted label
    pair (aa | ab | bb). For each frame: count each type, then n_shuffles times
    permute the labels across that frame's labeled cells and recount, giving a
    null mean/sd per type. z = (observed - mean_null) / sd_null.
    """
```

- Columns: `frame`, `contact_type` (`"aa"|"ab"|"bb"`, using the actual label
  strings, e.g. `"positive·positive"`), `observed_count` (int), `mean_null`
  (float), `sd_null` (float), `z_score` (float), `observed_fraction`,
  `expected_fraction` (= analytic `f_i·f_j` homotypic, `2·f_i·f_j` heterotypic;
  carried for reference alongside the empirical null).
- **Vectorized shuffle.** The frame's edge list is fixed; a shuffle only relabels
  nodes. Build endpoint-label index arrays once; per shuffle, permute the
  label vector and index both endpoints, then `np.add.at`-tally the
  (sorted-pair) types — `n_shuffles` columns at once. Cheap per frame.
- `z_score = NaN` when `sd_null == 0` (a degenerate frame: one label only, or
  too few cells). Edges touching `unclassified` cells are excluded throughout.
- Plotted with `value="z_score"`, `group_by=("contact_type","condition")` as a
  bar (position is the aggregation unit; frames collapse per position). This is
  the undirected aa/ab/bb "more than chance?" headline.

### 4. Density — `cell_density`

```python
def cell_density(analysis, labels, *, fov_area_mm2) -> dict[str, np.ndarray]:
    """Per (frame, label) cells and density = n_cells / fov_area_mm2.

    Emits one row per label present plus a label="all" total row. density is
    cells/mm². fov_area_mm2 is the user field-of-view area; labels may be empty
    (only the "all" row is produced then).
    """
```

- Columns: `frame`, `label`, `n_cells` (int), `density` (float, cells/mm²).
- `label="all"` counts every cell (including `unclassified`); per-label rows
  count that label's cells. The plugin supplies `fov_area_mm2`.
- Plotted with `value="density"`, `group_by=("label","condition")` as a bar.

## Layer 2 — the plugin (`plugins/neighborhood.py`, "Neighborhood & Density")

A thin `AnalysisPlugin` modeled on `contact_energetics.py` — **Plot section only**
(no Compute; reads the existing `contact_analysis.h5`), matplotlib-Qt probe +
disabled-button fallback, `PlotDockTabs` for the output panels.

**Controls**

- **View** dropdown → one of the four tables:
  - *Neighbor count* → `value="n_neighbors"`,
    `group_by=("condition","date","position_id","class_label")`, default plot
    `box`. Joins the NLS CSV for `class_label` (per-position, on `cell_id`).
  - *Neighbor enrichment* → `value="enrichment"`,
    `group_by=("condition","focal_label","neighbor_label")`, default `box`.
  - *Contact-type z-score* → `value="z_score"`,
    `group_by=("contact_type","condition")`, default `bar`.
  - *Density* → `value="density"`, `group_by=("label","condition")`, default
    `bar`.
- **Field-of-view area (mm²)** input, used by the *Density* view. Placeholder
  `auto`; blank ⇒ full image area `H·W·pixel_size_um² / 1e6` resolved per
  position (so a missing pixel size leaves density unavailable with a clear
  status, never a crash).
- **Shuffles** input (z-score view only), default `1000`.
- **Plot…** button.

**Pooling (off-thread `thread_worker`)** — for each built record: read the
analysis, load its NLS map (`read_nls_classification_csv` at
`nls_classification_csv_path`; missing ⇒ empty map), derive the selected table,
wrap as `PositionSource(metadata={condition,date,position_id}, table=...)`, and
`pool_object_tables(sources)`. The *Neighbor count* view additionally passes the
NLS map as the `join_table`/`join_columns=("class_label",)` so `class_label`
attaches through the standard join; the other three carry their labels in-table.

A position missing its NLS CSV still contributes the **un-typed** views (neighbor
count → all `unclassified`; density → `label="all"` only); it is skipped for
enrichment/z-score with a per-position note in the status line.

`PlotPanel` opens with the preset `value_columns`/`group_columns`/`default_plot`
above. No `PlotPanel` changes are required (the existing plot types cover every
view).

## Files

| File | Change |
|---|---|
| `aggregate_quantification/contacts/neighborhood.py` | **new** — `_frame_adjacency`, `cell_neighbor_counts`, `neighbor_enrichment`, `contact_type_zscores`, `cell_density`. |
| `aggregate_quantification/contacts/__init__.py` | export the four public functions. |
| `napari/aggregate_quantification/plugins/neighborhood.py` | **new** — "Neighborhood & Density" plugin (view selector, FOV-area + shuffles inputs, pool + launch generic `PlotPanel`). |
| `tests/aggregate_quantification/test_neighborhood.py` | **new** — degree, enrichment math, z-score null, density, unclassified handling. |
| `tests/napari/test_neighborhood_plugin.py` | **new** — pool a fake multi-position context per view + headless launch; disabled state without a Qt backend. |

## Testing

- **`_frame_adjacency` / `cell_neighbor_counts`**: hand-built `edges` fixture →
  expected degree per cell; border edges excluded; a boundary split into several
  `cell_cell` rows counts as one neighbor; an isolated cell has degree 0.
- **`neighbor_enrichment`**: a frame with known `N_a`, `N_b` and a hand-set
  neighborhood → `expected_j` matches `d·N_j/(N-1)` with self-exclusion;
  `enrichment = observed/expected`; a fully-sorted layout gives homotypic
  enrichment > 1 and heterotypic < 1; `expected==0` ⇒ NaN; edges to
  `unclassified` dropped and excluded from abundances; `unclassified` focal cell
  emits no rows.
- **`contact_type_zscores`**: a perfectly mixed checkerboard → near-zero / negative
  homotypic z; a fully segregated layout → large positive homotypic z; seeded
  shuffle is deterministic; `sd_null==0` ⇒ NaN; vectorized recount matches a
  naive per-shuffle loop on a small fixture; `expected_fraction` equals the
  analytic `f_i f_j` / `2 f_i f_j`.
- **`cell_density`**: `density == n_cells / fov_area_mm2`; `label="all"` totals
  every cell including `unclassified`; per-label rows sum (over labels) to the
  labeled-cell count; empty `labels` ⇒ only the `all` row.
- **`neighborhood` plugin**: pools a fake two-position context for each view and
  launches the panel headlessly (the matplotlib-Qt-guarded pattern); a position
  without an NLS CSV contributes the un-typed views and is noted for the typed
  ones; disabled button when the backend is unavailable.

## Out of scope (deferred)

- Per-position FOV-area entry (one global value this increment).
- A bespoke enrichment-matrix heatmap / dedicated neighborhood panel.
- More than two cell types (all unordered label pairs) — the maths generalize,
  but the UI and validation assume two this increment.
- Higher-order neighborhoods (k-ring), Delaunay/radius graphs — the analysis uses
  the existing shared-boundary adjacency only.
