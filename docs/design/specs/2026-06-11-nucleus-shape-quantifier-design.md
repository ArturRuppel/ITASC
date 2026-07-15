# Nucleus shape + relational shape quantifiers — design

**Date:** 2026-06-11
**Status:** approved, pre-implementation

## Goal

Extend the Aggregate Quantification "Cell shape" feature so the same per-object
morphology measurement can run on the **nucleus** as well as the cell, and so a
**both** scope can pair each nucleus with its cell and emit relational quantities
(nuclear:cell area ratio, centroid offset, …). The current single-purpose "Cell
shape" plugin becomes a scope-aware "Shape" plugin with a **cell / nucleus /
both** dropdown.

Temporal/population quantities ("cell number over time") are explicitly **out of
scope** for this spec — see [§9](#9-out-of-scope-deferred).

## Background — the seam we build on

The existing architecture already carries everything we need below the UI:

- `Quantifier` (`aggregate_quantification/quantifier.py`) is the extension seam: a
  subclass with a non-empty `quantity_id` auto-registers; the studio discovers it.
  Each quantifier *owns its persistence* — `build()` writes whatever format suits
  it, `read()`/`object_table()` parse it back. The framework imposes no schema.
- `PositionInputs` already exposes `cell_labels_path`, `nucleus_labels_path`,
  `pixel_size_um`, `position_dir`.
- The shape core (`cell_shape/build.py`) is **label-agnostic**: it just runs
  `skimage.measure.regionprops` over a label stack. The only cell-specific bits
  are the `cell_id` key-column name, the output filename, and which input field
  it reads.
- The plotting backend (`plotting.py`) pools `object_table()` results — dicts of
  numpy arrays keyed on `("frame", "cell_id")` — and is format-agnostic. It
  already supports a `line` plot over `frame` with `stat=count`.
- The studio (`aggregate_quantification_studio.py`) hosts "group plugins" that own
  a quantity's build (delegated to a centralized threaded build path via
  `set_build_callback`) and launch detached `PlotPanel` docks.

**Shared label id is assumed:** a nucleus carries the same label id as its cell
(cells are nucleus-seeded). Pairing for the relational scope is therefore a direct
`(frame, id)` join — no geometry. Consequently the object-key column stays
`cell_id` for every shape table (it is really the shared track id), and nothing in
the pooling/plotting layer changes.

## Chosen approach

**Three quantifiers + one scope-aware "Shape" plugin.** Generalize the shape core,
register `cell_shape` (exists), `nucleus_shape`, and `shape_relational`. The
dropdown is pure UI over which registered quantifier the one plugin builds and
plots.

Approaches considered and rejected:

- **One `shape` quantifier with a `scope` build param + one combined file.** Fights
  the framework: `object_table()` and `requires` are static/param-less, and
  `can_build` can't express "cell OR nucleus." More special-casing.
- **Two quantifiers; relational derived at pool time, not persisted.** Lighter, but
  the relational quantity isn't cached or exportable as an artifact and the plot
  panel grows derivation logic.

## 1. Module layout

Rename the `cell_shape/` package to `shape/` (safe — virtually no analyses have
been persisted yet, so no migration is needed):

- `aggregate_quantification/shape/core.py`
  - `build_object_shape(label_path, output_path, *, pixel_size_um, object_key,
    source_path=None, params=None, progress_cb=None) -> Path` — `regionprops` per
    frame, writes a **CSV** plus a provenance sidecar (see [§3](#3-persistence)).
  - `read_shape_table(path) -> dict[str, np.ndarray]`.
  - `DESCRIPTOR_COLUMNS` — the existing per-object descriptor set, unchanged.
- `aggregate_quantification/shape/relational.py`
  - `build_relational(cell_labels_path, nucleus_labels_path, output_path, *,
    pixel_size_um, source_path=None, params=None, progress_cb=None) -> Path` —
    reads both stacks, `regionprops` each per frame, inner-joins on `(frame, id)`,
    emits the relational columns. Writes CSV + sidecar.
  - `read_relational_table(path) -> dict[str, np.ndarray]`, `RELATIONAL_COLUMNS`.

The object-key column is `cell_id` for all three tables.

## 2. Three quantifiers

Under `aggregate_quantification/quantifiers/`:

| quantity_id      | reads                                            | `requires`                                            | default output                          |
|------------------|--------------------------------------------------|-------------------------------------------------------|-----------------------------------------|
| `cell_shape`     | cell labels                                      | `cell_labels_path`, `pixel_size_um`                   | `aggregate_analysis/cell_shape.csv`     |
| `nucleus_shape`  | nucleus labels                                   | `nucleus_labels_path`, `pixel_size_um`                | `aggregate_analysis/nucleus_shape.csv`  |
| `shape_relational` | both label stacks                              | `cell_labels_path`, `nucleus_labels_path`, `pixel_size_um` | `aggregate_analysis/shape_relational.csv` |

`cell_shape` / `nucleus_shape` are thin wrappers over `build_object_shape`,
differing only by which input field they read and their output filename (both use
`object_key="cell_id"`). `default_output_name` is
nested under `aggregate_analysis/`; `default_output` resolves to
`position_dir / default_output_name`, and the builder `mkdir`s the parent. Contacts
keeps its explicit legacy `contact_analysis.h5` path.

## 3. Persistence

Each artifact is a flat tidy **CSV** plus a sidecar `<name>.provenance.json`
(source path, `pixel_size_um`, params, ISO timestamp, `quantity_id`, column list).
Rationale: the shape tables are small (~10⁴–10⁵ rows of floats) and flat, so h5's
hierarchy / compression / partial-read wins do not apply; CSV is git-readable,
Excel/R/pandas-openable, and is already what users export from the plot panel. The
`Quantifier` seam hides the format from everything downstream (`object_table()`
returns numpy arrays), so nothing in pooling/plotting changes.

- `read()` / `object_table()` read the CSV only.
- `is_built()` checks the CSV exists.
- Contacts stays on h5 — it is genuinely hierarchical (cells, edges, T1 events,
  NLS columns patched in place).

## 4. Relational descriptor set

One row per `(frame, cell_id)` present in **both** label stacks:

- `nc_area_ratio` = nucleus_area / cell_area
- `centroid_offset_um` = Euclidean distance between nucleus & cell centroids
- `centroid_offset_norm` = `centroid_offset_um` / cell equivalent radius
  (`sqrt(cell_area/π)`) — dimensionless, comparable across cell sizes
- `orientation_delta` = ellipse-orientation difference, folded to `[0, π/2]`
- `nc_perimeter_ratio` = nucleus_perimeter / cell_perimeter
- `nc_major_axis_ratio` = nucleus_major_axis / cell_major_axis
- carried for convenience: `cell_area_um2`, `nucleus_area_um2`

`RELATIONAL_COLUMNS` lists all of the above (the value axis a plot/export chooses
from). Ids present in only one source are dropped from the join; the dropped count
is reported via `progress_cb` / the build status line so silent loss is visible.

## 5. The "Shape" plugin (renames `CellShapePlugin`)

`aggregate_quantification/plugins/cell_shape.py` → `shape.py`, class `ShapePlugin`
(`plugin_id = "shape"`, `display_name = "Shape"`).

- `requires` relaxed: a position is in scope if it has cell **or** nucleus labels.
- `owns_quantities = ("cell_shape", "nucleus_shape", "shape_relational")` so the
  studio suppresses generic auto-builders for all three.
- A **scope dropdown** — `cell` / `nucleus` / `both` — selects which quantifier the
  Compute and Plot sections drive.
- **Default scope rule:** `both` when ≥1 in-scope position has both labels; else
  whichever single source is present across the scope. Per-position buildability is
  still gated by each quantifier's `can_build`; the status line reports positions
  skipped for missing inputs or pixel size.
- Compute and pixel-size handling are unchanged from the current plugin, only
  parameterized by the selected quantifier.

## 6. Plot honors scope + separate column

- `cell` / `nucleus` → pool that quantity's `object_table`, feed
  `PlotPanel(value_columns=DESCRIPTOR_COLUMNS, …)`.
- `both` → pool `shape_relational`, feed `value_columns=RELATIONAL_COLUMNS`. The
  plot offers **strictly the relational columns** (not the raw per-source
  descriptors).
- The contacts `class_label` left-join (on `(frame, cell_id)`) still applies to all
  three scopes, exactly as today.
- `PlotPanel` opens via `add_dock_widget(area="right")` — docked as the right-hand
  column beside the catalogue/plugins, not floating.

## 7. Catalogue / discovery

No changes — discovery and the catalogue already track
`nucleus_tracked_labels_path`, and `position_inputs_from_record` already populates
`nucleus_labels_path`.

## 8. Testing

Mirror the existing `test_cell_shape_*` backend tests and the napari plugin tests:

- **core:** build over synthetic 2-frame label stacks → expected columns/values,
  physical (µm) scaling, CSV round-trip via `read_shape_table`.
- **nucleus quantifier:** correct `requires`, reads nucleus labels, output lands at
  `aggregate_analysis/nucleus_shape.csv`.
- **relational:** synthetic cell+nucleus stacks with shared ids → ratios/offset
  correct; ids present in only one source are dropped and the drop count reported;
  `requires` includes both label paths + pixel size.
- **output location:** every shape artifact resolves under `aggregate_analysis/`.
- **plugin:** scope-default rule (both/cell/nucleus per available inputs), build
  delegates the scope-selected quantifier, plot uses scope-correct `value_columns`.

## 9. Out of scope (deferred)

Temporal / population quantities. "Cell number over time" is already expressible
via the existing `line` plot with `stat=count` over `frame` on any shape table. A
dedicated per-frame population quantifier (count, density, mean size over time) is
a separate future spec.
