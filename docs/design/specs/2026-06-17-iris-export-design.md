# Iris export: `.iris` bundles with premade SuperPlot analyses

**Status:** approved design, pre-implementation
**Date:** 2026-06-17
**Scope:** backend-only feature in `cellflow-aggregate`

## Statement of need

CellFlow produces tidy per-position quantity tables (`cells_by_frame`,
`edges_by_frame`, …). Iris is a separate desktop tool (React/Tauri + a Python
engine) that does the inferential statistics and publication-figure work CellFlow
deliberately does *not* do. Today a user must leave CellFlow, open Iris, import a
CSV, and rebuild every comparison by hand.

This feature lets CellFlow **export each tidy table as a ready-to-analyze `.iris`
bundle** carrying a comprehensive set of **premade SuperPlot analyses**. The user
opens the bundle in Iris and immediately has the figures + recommended tests, with
the pseudoreplication-correct statistical unit already wired in.

This is the additive half of a larger decision (see "Relationship to CellFlow's
own plotting" below): CellFlow keeps its existing plotting/reduction unchanged for
now; this feature is layered on top and removes nothing.

## Hard constraints (non-negotiable)

1. **No `import iris_engine`.** CellFlow must write the `.iris` ZIP by hand using
   stdlib `zipfile` + `pyarrow` + `pandas` + `json`. CellFlow is the JOSS
   submission; it must not depend on the (unpublished) Iris repo. The two are
   coupled *only* through the `.iris` file format.
2. **Backend-only.** No Qt / napari imports. The module lives where the
   standalone `cellflow-aggregate` wheel and headless/batch runs can use it.
3. **The `.iris` format is a frozen, versioned contract.** It is documented here
   and guarded by a structural test. CellFlow targets `format_version` 1.0.

## The `.iris` format (contract, v1.0)

A `.iris` file is a ZIP containing:

| Entry | Content |
|---|---|
| `manifest.json` | `{format_version: "1.0", modified: <iso8601>, engine_snapshot: {...}}` |
| `data/table.parquet` | the tidy table, Parquet + zstd, written `ZIP_STORED` (already compressed) |
| `data/schema.json` | `{schema_version: "1.0", columns: [{name, type, label, unit?, levels?}]}` |
| `analyses/NN-<id>.json` | one analysis spec per file, `NN` = 01, 02, … (a *list* of analyses is native) |
| `provenance.json` | free-form provenance dict |

`type ∈ {numeric, categorical, identifier}`. Iris re-infers nothing from the
table — the schema is authoritative. Iris computes statistics and renders figures
from the analysis specs **on load**; the specs carry figure *intent*, not baked
results.

### Analysis spec (Iris grammar 2.0)

```jsonc
{
  "spec_version": "2.0",
  "id": "superplot__class_label__cell_shape.area_um2",
  "title": "Cell shape · Area (µm²) — by class",
  "encodings": {"x": {"column": "class_label"}, "y": {"column": "cell_shape.area_um2"},
                "color": {"column": "date"}, "shape": {"column": "date"}, "size": null},
  "facet": {"row": null, "col": null, "share_x": true, "share_y": true},
  "hierarchy": {"spine": ["date", "position_id", "cell_id", "frame"], "fn": {}},
  "layers": [
    {"geom": "violin", "level": "cell_id", "params": {}},
    {"geom": "dot",    "level": "date",    "params": {"layout": "swarm"}}
  ],
  "stats": {}
}
```

Each **encoding is an object** `{"column": name}` (not a bare string); an unmapped
channel is `null`. The test is **unpinned** (`stats: {}`): Iris's recommender
chooses it on load. The coarsest layer level (`date`) is the grain the test reads,
so the inferential unit is the date — Iris's pseudoreplication guard. (Verified
end-to-end against the Iris engine: `inferential_level == "date"`, a recommended
test is run, and the figure serializes to valid gid-tagged SVG.) For `class_label`
(home level = cell), Iris reports the comparison as *unpaired* — a cell carries
only one class, so no single cell crosses both levels; it is a nested design, not
a pairing.

**Single-level axis ⇒ describe-only.** An unpinned test on a one-group axis raises
422 in Iris ("needs at least 2 groups"); it does **not** silently degrade. So the
exporter inspects each axis's level count and emits `stats:
{"chosen_by": "describe_only"}` when an axis has `< 2` levels (this dataset's
`condition`), and the unpinned `stats: {}` only when it has `≥ 2`. Either way the
bundle opens; describe-only renders the SuperPlot with descriptive stats and no
test.

## The SuperPlot template

Per **kept numeric descriptor `D`** × per **axis `A`** present in the table
(`A ∈ {condition, class_label}`), emit one analysis:

- `x = A`, `y = D`, `color = date`, `shape = date`
- `hierarchy.spine` = the table's spine (see below)
- layers:
  1. `violin` @ finest object level — the per-object distribution, one violin per
     group (group colour).
  2. `dot` @ `date` level, `layout: swarm` — the per-date replicate means as a
     beeswarm of distinct-shaped markers (coloured + shaped by date).

**No per-object swarm, no box, no notch.** With tens of thousands of cells a swarm
is unreadable and a median-CI notch is wildly overconfident (the notch collapses
to nothing at large n). The violin carries the distribution shape; the only points
drawn are the few per-date means.

### Why this renders as intended (verified against Iris)

- **`date` must be typed `identifier`, not `categorical`.** Iris
  (`compiler.py:655-667`) dodges a categorical colour into sub-columns, which would
  split the violin per date. An *identifier* colour with a per-point layer present
  (the date dots) colours each dot in place and leaves the violin as one-per-group
  — the SuperPlot idiom. The exporter's schema typing is what unlocks this.
- The inferential test reads the coarsest layer level = `date`; verified
  `inferential_level == "date"`, a recommended test runs for a ≥2-level axis, and
  the figure serializes to valid gid-tagged SVG.

### Pruning

Descriptors that are reference-frame coordinates, vector components, or raw angles
get **no** premade analysis — a cross-condition comparison of them is not
meaningful. Pruned: leaf names `x_um, y_um, centroid_x_um, centroid_y_um,
vx_um_per_s, vy_um_per_s, orientation`; plus the redundant
`shape_relational.{cell,nucleus}_area_um2` (re-exports of the shape-family areas).
The columns stay in the table for manual use; only their premade analyses are
dropped. (On `cells_by_frame` this takes 94 analyses → 62.)

### Grouping & naming

Iris's analysis sidebar is a **flat list with no section UI**. So analyses are
**ordered by quantity family** (`cell_shape, cell_dynamics, nucleus_shape,
nucleus_dynamics, shape_relational, neighbor_count`) and each carries a
family-prefixed `title` (e.g. `"Cell shape · Area (µm²) — by condition"`, shown as
the editable analysis name). They therefore cluster into readable family groups.
True collapsible sections would need an Iris frontend change (separate repo) and
are out of scope.

### Single-level axis ⇒ describe-only

An unpinned test on a one-group axis raises 422 in Iris ("needs at least 2
groups"); it does **not** silently degrade. The exporter inspects each axis's level
count and emits `stats: {"chosen_by": "describe_only"}` when an axis has `< 2`
levels (this dataset's `condition`), and unpinned `stats: {}` only when it has
`≥ 2`. Either way the bundle opens; describe-only renders the SuperPlot with
descriptive stats and no test, and the same template becomes a real comparison on
multi-condition data.

## Per-table configuration

The spine is uniformly `date → position_id → <object_key> → frame` (coarse →
fine), so picking the object level averages `frame` away, and every layer `level`
the template references (`<object_key>` and `date`) is a spine column — required
by Iris's `hierarchy.materialize_levels`.

**Scope (now): `cells_by_frame` only.** `export_dir` writes just `cells_by_frame`
while the template is tuned (`TABLES_TO_EXPORT`); the other tables keep their
object-key mapping (`TABLE_OBJECT_KEYS`) and remain exportable via `export_table`
directly, deferred until their object-grain plots are worth shipping.

| Table | Object key `<object_key>` | Axes (≥1 level) | Status |
|---|---|---|---|
| `cells_by_frame` | cell_id | condition, class_label | **exported** |
| `cell_neighbors_by_frame` | cell_id | condition, class_label | deferred |
| `contact_types_by_frame` | contact_type | condition | deferred |
| `density_by_frame` | label | condition | deferred |
| `edges_by_frame` | t1_event_id | condition | deferred |

**Caveat — non-cell tables (when re-enabled).** For `contact_types`, `density`,
and `edges` the object key is itself category-like (`contact_type`) or event-like
(`t1_event_id`), and the most natural comparison is often *across that key* rather
than across `condition`. The premade `condition` SuperPlot is a reasonable default,
but the user may want to re-pivot these in Iris (e.g. `x = contact_type`). The
exporter does not guess that; it emits the agreed axes only.

## Components

All under `src/cellflow/aggregate_quantification/iris_export/` (backend-only):

1. **`document.py`** — `write_iris(table_df, schema, analyses, provenance) -> bytes`.
   Builds the v1.0 ZIP. The single place that knows the byte format. Approach A
   (reimplement ~25 lines); the format contract above is the spec it implements.
2. **`schema.py`** — `infer_schema(df) -> dict`. Column typing:
   - `identifier`: `date, position_id, cell_id, frame, t1_event_id, label,
     focal_label, neighbor_label, focal_id, partner_id`
   - `categorical`: `condition, class_label, contact_type, role`
   - `numeric`: everything else
   - `label` = leaf after last `.`; `unit` parsed from `_um2 → µm²`, `_um → µm`;
     `levels` listed for categorical columns. `numeric_descriptors(schema)` returns
     the numeric value columns.
3. **`analyses.py`** — `build_analyses(df, schema, object_key) -> list[dict]`.
   Emits the violin + date-swarm SuperPlot per kept descriptor × axis, applying the
   prune list, family ordering/titles, and the 1-level-axis (describe-only) rule.
4. **`export.py`** — `export_table(csv, out_dir)` / `export_dir(data_dir, out_dir)`
   + `__main__.py` CLI. `export_dir` writes the `TABLES_TO_EXPORT` set
   (`cells_by_frame` for now), adding `id`/`excluded` columns, to `<table>.iris`.

### Defaults

- **Output location:** `<data_dir>/iris/<table>.iris`.
- **Writer:** approach A (reimplement). Approach B (vendor Iris's `save_document`)
  is the fallback if format drift is ever observed.

## Testing

- **Structural contract test:** a written bundle is a valid ZIP with the five entry
  kinds; schema types `date`/`position_id`/`cell_id` as `identifier`; every
  analysis has the violin@object + dot@date layers, a family-prefixed `title`, and
  `color=date`/`shape=date` encodings; pruned columns are absent; `analyses/NN-*.json`
  numbering is contiguous.
- **Fixture round-trip:** a small synthetic tidy table → export → re-open the ZIP
  and assert table values + schema survive Parquet round-trip.
- **Optional live integration:** when an Iris engine checkout is discoverable
  (`IRIS_ENGINE` env), load each bundle through it and assert it produces a figure
  + a stats read-out without error. Skipped otherwise so the suite stays
  self-contained.

## Run target

```
python -m cellflow.aggregate_quantification.iris_export \
    /home/aruppel/Data/aggregate_quantification
# → /home/aruppel/Data/aggregate_quantification/iris/{cells_by_frame, ...}.iris
```

## Relationship to CellFlow's own plotting

Out of scope here. CellFlow's existing reduction + descriptive/domain plotting is
**unchanged**. The remove/simplify decision is deferred to after JOSS and revisited
once it is known whether the `.iris` path is preferred over the in-napari plots.
To keep that door open, the exporter's nesting semantics
(`date → position_id → object → frame`, equal-weighted-per-date) must match the
semantics CellFlow's own plotting uses, so a CellFlow figure and the Iris figure of
the same data agree by construction.

## Non-goals

- Removing or changing CellFlow's plotting/reduction.
- Any UI (napari/Qt) entry point. Backend + CLI only for now.
- Pre-baking statistical results into the bundle (Iris computes on load).
- A Python dependency on Iris in either direction.
