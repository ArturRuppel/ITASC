# Cell Shape: Quantifier + Plotting Backend + Group Plugin — Design Spec

Date: 2026-06-10

> **Status: implemented.** All four steps below are built and tested
> (`tests/aggregate_quantification/test_cell_shape_build.py`, `test_plotting.py`,
> `test_quantifier.py`; `tests/napari/test_cell_shape_plugin.py`,
> `test_studio_plugins.py`). The open sub-decision was resolved as **(a)**: the
> group plugin owns the build via `owns_quantities` and the studio suppresses the
> generic auto-builder for `cell_shape`. One refinement vs. the draft: the studio
> build path resolves the output via `output_for_record` (a documented contacts
> fallback to the catalogue's explicit/nested `contact_analysis_path`) rather
> than `default_output` unconditionally, so contacts' custom nested paths don't
> regress; `pool_object_tables` consumes already-read tables (kept the backend
> free of the napari path resolver).

The **first real new quantity** for Aggregate Quantification, beyond the seed
contacts quantifier. It establishes a pattern, not just a feature:

- **In the backend, compute and plotting are two separate headless layers.**
  Compute = the `Quantifier` registry (cell_shape is the new member). Plotting =
  a generic, quantity-agnostic aggregation/figure module that operates on tidy
  tables. Both are scriptable without napari (standalone wheel, notebooks).
- **In the UI, a plugin unifies compute + plotting per logical group.** There is
  no monolithic "plot everything" plugin. Each logical group (cell shape now;
  contacts and NLS later) is one plugin that surfaces *its* build action and
  *its* plots, both delegating to the shared backend layers. (User decision —
  refines the earlier "separate generic plot plugin" idea: keep the genericity in
  the backend, organize the UI by domain.)

So this unit ships three things: the **CellShapeQuantifier** (compute), a generic
**plotting backend**, and a unified **Cell Shape group plugin** that surfaces
both. It is the follow-on named in `notes/aggregate_quantification_spec.md`
("Out of scope → cell shape") and `TODO.md` ("Broaden scope … cell shape
analysis").

## Decisions (locked)

1. **Backend: compute ⟂ plotting, both headless.** The compute layer
   (quantifiers) writes per-position artifacts. A *separate* plotting layer
   (`aggregate_quantification/plotting.py`) pools tidy tables, aggregates, and
   builds matplotlib figures — no Qt/napari, so it runs in scripts and the
   standalone wheel. The two never import each other.
2. **UI: one plugin per logical group, unifying build + plot.** The Cell Shape
   plugin has a *Compute* section (build the quantity for in-scope positions) and
   a *Plot* section (configure + render + export), both calling the backend
   layers. No global Table Explorer. Existing groups (contacts, NLS) surface
   plotting the same way as a follow-on.
3. **v1 joins subpopulation class.** "Group by label" includes splitting shape
   distributions by subpopulation (NLS positive/negative). The plotting layer
   left-joins the classification columns from the contacts `.h5` `cells/table`
   onto the pooled table by `(frame, cell_id)`. Missing contacts artifact /
   unclassified cells → blank class, never an error.
4. **Shape owns its persistence: per-position `cell_shape.h5`.** A single
   `shape/table` group of 1-D arrays, mirroring the contacts `cells/table`
   schema, plus a `provenance` group. No new dependency; `read()` returns a
   dict-of-arrays → `pd.DataFrame(...)`. Follows the locked "each quantifier owns
   its persistence" rule.
5. **Descriptors in pixel units in v1.** Raw `regionprops`; the dimensionless
   ratios are scale-free. Physical-unit scaling is a later add via the `params`
   dict `build()` already accepts (`{"pixel_size_um": …}`).
6. **Plot backend = matplotlib (Agg).** Already a dependency; renders headless
   (figures returned as objects), best for grouped statistical charts and vector
   export. pyqtgraph stays reserved for the live-interactive NLS scatter.
7. **Descriptive only.** Means, SD/SEM, n, per-position spread — no p-values
   (matches the meta-analyzer sketch's "robust descriptive signals first").
8. **h5 is the canonical store; CSV is a first-class export.** `cell_shape.h5`
   is the internal artifact (typed, compact, the build output). CSV is *not* a
   second store — it is an on-demand export of a tidy table, available both as a
   scriptable backend function and as plugin buttons. Three export shapes:
   per-position raw, pooled raw (all in-scope positions + metadata), and the
   aggregated summary. CSV export never round-trips back into compute.

## Architecture

```
BACKEND (headless, no Qt)
  compute layer ── Quantifier registry
    contacts ─→ contact_analysis.h5   (frame, cell_id, area, …, class_label, nls_status)
    cell_shape ─→ cell_shape.h5       (frame, cell_id, area, circularity, aspect_ratio, …)   ← NEW
        │  object_table(path) -> {col: array}   (tidy-table contract, both implement it)
        ▼
  plotting layer ── aggregate_quantification/plotting.py   ← NEW, quantity-agnostic
    pool_object_tables(records, quantifier, join=…) -> DataFrame   (+ condition/date/position_id)
    aggregate(df, spec) -> DataFrame                               (per-cell pooled | per-position)
    build_figure(df, spec) -> matplotlib.Figure                    (hist | box/violin | bar | line)

UI (napari)
  one plugin per logical group, unifying compute + plot:
    Cell Shape plugin
      ▸ Compute:  Build cell_shape for in-scope positions   → studio build callback
      ▸ Plot:     value · group-by · level · type · split-by-subpop · export
                  → plotting layer → FigureCanvas
    (contacts / NLS groups adopt the same Compute+Plot shape later)
```

User flow: catalogue positions → open **Cell Shape** plugin → *Compute* builds
`cell_shape.h5` for in-scope rows → *Plot* pools them and renders; export CSV /
figure.

## 1. Cell Shape quantifier (compute layer)

`src/cellflow/aggregate_quantification/quantifiers/cell_shape.py` — a self-
registering `Quantifier` (backend, no Qt). Mirrors `ContactsQuantifier`. The
compute core is a small headless module `aggregate_quantification/cell_shape/
{build,reader}.py` (`build_cell_shape`, `read_cell_shape`), parallel to
`contacts/build.py` + `contacts/reader.py`, unit-testable without the registry.
`regionprops` runs per frame over the label stack (same loop as contacts'
`_extract_cell_columns`).

```python
class CellShapeQuantifier(Quantifier):
    quantity_id = "cell_shape"
    display_name = "Cell shape"
    requires = ("cell_labels_path",)          # cell labels only
    default_output_name = "cell_shape.h5"

    def default_output(self, inputs): return inputs.position_dir / self.default_output_name
    def build(self, inputs, output_path, *, params=None, progress_cb=None) -> Path: ...
    def read(self, output_path) -> dict[str, np.ndarray]: ...
    def object_table(self, output_path) -> Mapping[str, np.ndarray]: ...   # tidy-table contract
```

### Descriptors (per cell, per frame)

Keys `frame`, `cell_id`. From `regionprops`:

| column | source / formula | units |
|---|---|---|
| `area` | `prop.area` | px² |
| `perimeter` | `prop.perimeter` | px |
| `equivalent_diameter` | `prop.equivalent_diameter` | px |
| `major_axis_length` | `prop.major_axis_length` | px |
| `minor_axis_length` | `prop.minor_axis_length` | px |
| `aspect_ratio` | `major / minor` (guard `minor>0`, else NaN) | — |
| `circularity` | `4π·area / perimeter²` (guard `perimeter>0`; clamp ≤ 1) | — |
| `eccentricity` | `prop.eccentricity` | — |
| `solidity` | `prop.solidity` | — |
| `extent` | `prop.extent` | — |
| `orientation` | `prop.orientation` | rad |
| `centroid_y`, `centroid_x` | `prop.centroid` | px (drilldown/overlay) |

A deliberate, owned superset of the geometry contacts already stores — shape is
its own quantity with its own descriptor set; no attempt to dedupe.

### `cell_shape.h5` layout

```
/provenance   .attrs: source_position_path, cell_tracked_labels_path,
                     params_json, created_at, cellflow_version
/shape/table  1-D datasets: frame, cell_id, area, perimeter, …, centroid_x
```

`read()` returns `{name: array}` for every dataset under `shape/table` — the
contacts reader's `_read_table` pattern.

## 2. Plotting backend (plotting layer)

`src/cellflow/aggregate_quantification/plotting.py` — generic, quantity-agnostic,
**no Qt/napari**. Three functions + a small `PlotSpec` dataclass, all scriptable:

```python
@dataclass(frozen=True)
class PlotSpec:
    value: str                       # numeric column to plot, e.g. "circularity"
    group_by: tuple[str, ...] = ()   # condition | date | position_id | class_label | frame
    level: str = "cell"              # "cell" (pooled) | "position" (per-position summary)
    plot: str = "hist"               # hist | box | violin | bar | line
    stat: str = "mean"               # for bar/line: mean | median | count
    error: str = "sd"                # sd | sem | none

def pool_object_tables(records, quantifier, *, join=None) -> pd.DataFrame:
    """Read each in-scope position's object_table, prepend condition/date/
    position_id, concat. `join` (optional) attaches another quantity's columns
    (e.g. contacts class_label/nls_status) by left-join on (frame, cell_id)."""

def aggregate(df: pd.DataFrame, spec: PlotSpec) -> pd.DataFrame:
    """Tidy aggregated frame. level='position' first reduces within each position
    (mean/median/count) before grouping, so cross-position comparisons aggregate
    tissues, not cells (pseudoreplication guard)."""

def build_figure(df: pd.DataFrame, spec: PlotSpec) -> matplotlib.figure.Figure:
    """Render hist/box/violin/bar/line per spec; return a Figure (Agg)."""

def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Export a tidy DataFrame to CSV (raw pooled, per-position slice, or an
    aggregated summary — all are just DataFrames). Ensures a `.csv` suffix."""
```

CSV export is a thin, scriptable wrapper over these — no separate export engine.
The three export shapes are all `write_csv(df, …)` of: a single position's
`pd.DataFrame(quantifier.object_table(path))`; `pool_object_tables(...)`; or
`aggregate(pool_object_tables(...), spec)`. A notebook can produce any of them
without napari; the plugin wires the same three to buttons.

- `pool_object_tables` reads through the quantifier's `object_table` (below), so
  it works for *any* quantity, not just shape.
- `level="position"` + `stat="count"` is exactly "average cell number per tissue,
  grouped by date/condition/subpopulation".
- Returns figures/DataFrames — no widgets — so a notebook can `build_figure(...)`
  and `fig.savefig(...)`, and the plugin embeds the same figure in a canvas.

### Tidy-table contract (what the plotting layer pools over)

One optional method on the `Quantifier` base lets the plotting layer pull a tidy
per-object table from any quantity:

```python
def object_table(self, output_path: Path) -> Mapping[str, np.ndarray] | None:
    """A tidy, column-major per-object table with at least `frame` + `cell_id`,
    or None if this quantifier has no per-object table. Default: None."""
    return None
```

- **cell_shape** → its `shape/table` dict.
- **contacts** → its `cells/table` dict (already has `frame`, `cell_id`,
  `class_label`, `nls_status`, …) — this is what makes the subpopulation join
  work without a bespoke reader.

Numpy-keyed (not pandas) so the headless compute core stays pandas-free; the
plotting layer wraps it in a DataFrame.

## 3. Cell Shape group plugin (UI, unifies compute + plot)

`src/cellflow/napari/aggregate_quantification/plugins/cell_shape.py` — an
`AnalysisPlugin` (cross-position; `requires=("cell_labels_path",)`). Body = two
collapsible sections:

**Compute** — "Build cell shape for N in-scope positions" + a *Recompute*
checkbox. Delegates to the studio's build callback (the same path the generic
builder uses), so building stays centralized (threaded, status-refreshed). This
is what "unify compute + plot per group" means concretely: the build button
lives *in the group*, not in a separate builder checkbox.

**Plot** — controls → `PlotSpec` → `plotting.build_figure(...)` → `FigureCanvas`:
- **Value**: any numeric shape descriptor.
- **Level**: *Per cell (pooled)* | *Per position*.
- **Group / facet by**: `condition`, `date`, `position_id`, `class_label` (when
  joined), `frame` — multi-select.
- **Plot type**: histogram/KDE · box/violin · grouped bar (mean±SD/SEM or count)
  · line over `frame`.
- **Split by subpopulation** (checkbox): sets the `join` so the pooled frame
  gains `class_label`/`nls_status` from each position's contacts `.h5`.
- **Pseudoreplication guard**: when a group axis is set, default bar/box to
  *Per position*; a "pool all cells" override carries a one-line hint.
- **Export** (all via `plotting.write_csv` / `fig.savefig`): *pooled CSV* (full
  annotated per-cell frame across in-scope positions) · *per-position CSV* (the
  focused row's raw table) · *aggregated CSV* (the grouped summary shown) ·
  *figure PNG/SVG*. CSV save reuses the studio's `.csv`-suffix safeguard.

Pooling many HDF5s runs in a `thread_worker` (matching the other widgets); the
redraw on the returned frame stays on the GUI thread. A small read cache avoids
re-opening the same artifact across re-plots (spirit of the studio's
`_analysis_cache`).

## Framework deltas (small, named)

1. **Generalize the studio build path.** `_run_quantity_build`
   (`aggregate_quantification_studio.py:421`) sets the output to
   `record.get("contact_analysis_path")` — the *contacts* artifact name — for
   every quantifier, so a second quantity would write `cell_shape` data into
   `contact_analysis.h5`. Fix: compute the output via
   `quantifier.default_output(position_inputs_from_record(record))`. Add
   `default_output(self, inputs) -> Path` to the `Quantifier` base (contacts
   already has it; shape adds it).
2. **Add `object_table` to the `Quantifier` base** (default `None`), implemented
   by contacts and cell_shape.
3. **Group plugins own their quantity's builder.** Because compute now lives
   *inside* a group plugin, the studio should not also render the generic
   auto-builder checkbox for a quantity that a group plugin claims. Proposal: a
   group plugin may declare `owns_quantities = ("cell_shape",)`;
   `available_studio_plugins` skips the auto-`BuilderPlugin` for any claimed
   `quantity_id`. Quantities *without* a dedicated plugin keep the free
   auto-builder, so the no-code "build button for every new quantifier" seam
   survives. **← the one open sub-decision; see below.**

No change to the catalogue, discovery, or the checkbox registry mechanics.

> **Open sub-decision (delta #3):** do we (a) let the group plugin own the build
> and suppress the auto-builder for that quantity [recommended — true
> per-group unification, no duplicate checkbox], or (b) keep the auto-builder
> *and* the group plugin's plot-only section side by side [less code, but two
> checkboxes per group, which is the split this refinement is trying to remove]?
> The spec assumes (a).

## Out of scope / follow-ons

- **Contacts & NLS gain a Plot section** via the same backend — surfacing
  plotting in the groups we already have. (Direct payoff of putting plotting in a
  reusable backend layer.)
- **Per-position shape overlay** — recolour the cell-labels layer by a chosen
  descriptor; a separate napari view later.
- **Physical units** — `pixel_size_um` in `params`.
- **Generic cross-quantity join UI** — v1 hardwires the contacts class columns;
  generalize "attach columns from quantity X" when a third joinable quantity
  appears.
- **Track-level shape** (per-`track_id` summaries) — needs a tracks input; defer
  with the kinematics quantifier.
- **Formal statistics** — descriptive only for now.

## Test plan

Compute layer (headless):
- `build_cell_shape` on a synthetic 2D+t label stack → `cell_shape.h5` with one
  `shape/table` row per (frame, label); spot-check `area`, disk `circularity`≈1,
  ellipse `aspect_ratio`, NaN guards on degenerate cells.
- `read_cell_shape` round-trips; `CellShapeQuantifier.read`/`object_table` agree;
  `default_output` ends in `cell_shape.h5`.
- Registry includes `cell_shape`; `can_build` honours `requires`.

Plotting layer (headless — no Qt needed):
- `pool_object_tables` over two fake positions → DataFrame with metadata cols +
  one row per cell-frame; `join` left-joins contacts `class_label`, blank where
  no contacts `.h5`.
- `aggregate` with `level="position", stat="count"` reproduces cell-number-
  per-tissue; grouped means match a hand computation.
- `build_figure` returns a `Figure` for each plot type without a display.
- `write_csv` emits a `.csv` for each export shape (per-position, pooled,
  aggregated); re-reading reproduces the columns/rows.

Studio + plugin (`QT_QPA_PLATFORM=offscreen`):
- Build path writes a second quantifier to *its own* `default_output`, not
  `contact_analysis.h5` (guards delta #1).
- Auto-builder suppressed for a quantity claimed by a group plugin; still present
  for unclaimed quantities (guards delta #3).
- Cell Shape plugin: Compute triggers a build; Plot renders; pooled/aggregated
  CSV export round-trips and lands with `.csv`.

Run: `QT_QPA_PLATFORM=offscreen uv run --quiet pytest tests/`.

## Implementation order

1. Compute: `aggregate_quantification/cell_shape/{build,reader}.py` +
   `quantifiers/cell_shape.py` adapter; tests on synthetic stacks + registry.
2. Framework deltas #1/#2: `Quantifier.default_output` + `object_table` on the
   base; route the studio build path through `default_output`; implement
   `object_table` on contacts + cell_shape. Build-path test.
3. Plotting backend: `aggregate_quantification/plotting.py`
   (`PlotSpec`, `pool_object_tables`, `aggregate`, `build_figure`); headless tests.
4. Delta #3 + UI: builder-ownership in `available_studio_plugins`;
   `plugins/cell_shape.py` group plugin (Compute + Plot sections, FigureCanvas,
   exports); plugin tests.
5. Update `TODO.md` (start checking off "broaden scope … cell shape").
```
