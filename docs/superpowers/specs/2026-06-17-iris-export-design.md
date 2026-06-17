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
  "encodings": {"x": "class_label", "y": "cell_shape.area_um2",
                "color": "date", "shape": "date", "size": null},
  "facet": {"row": null, "col": null, "share_x": true, "share_y": true},
  "hierarchy": {"spine": ["date", "position_id", "cell_id", "frame"], "fn": {}},
  "style": {"overrides": {"notch": true}},
  "layers": [
    {"geom": "dot", "level": "cell_id", "params": {"layout": "swarm"}},
    {"geom": "box", "level": "cell_id", "params": {}},
    {"geom": "dot", "level": "date",    "params": {"jitter": 0.0}}
  ],
  "stats": {}
}
```

The test is **unpinned** (`stats: {}`): Iris's recommender chooses it on load. The
coarsest layer level (`date`) is the grain the test reads, so the inferential unit
is the date — Iris's pseudoreplication guard. For a categorical comparison whose
home level is finer than `date` (e.g. `class_label`, which lives at the cell
level), Iris's pairing detection sees the same date holding both levels and treats
the test as paired across dates.

## The SuperPlot template

Per **table** × per **numeric descriptor `D`** × per **axis `A`** present in the
table (`A ∈ {condition, class_label}`), emit one analysis:

- `x = A`, `y = D`, `color = date`, `shape = date`
- `hierarchy.spine` = the table's spine (see below)
- `style.overrides.notch = true`
- layers:
  1. `dot` @ finest object level, `layout: swarm` — every object, color + shape by
     date. **Emitted only when object count ≤ 3000** (Iris `POINT_CAP`).
  2. `box` @ finest object level — notched, **one box per group** (not split by
     date).
  3. `dot` @ `date` level, `jitter: 0` — the per-date replicate means as
     distinct-shaped markers, overlaid.

### Why this renders as intended (verified against Iris)

- **`date` must be typed `identifier`, not `categorical`.** Iris
  (`compiler.py:655-667`) dodges a categorical color into sub-columns, which would
  split the box per date. An *identifier* color with a per-point layer present
  colors each dot in place and leaves the box as one-per-group — the SuperPlot
  idiom. The exporter's schema typing is what unlocks this.
- **Notch** is a style flag (`style.overrides.notch`), gated on `len(ys) > 5`
  (`compiler.py:775`). The box sits at the finest grain (hundreds+ objects), so it
  renders.
- **Swarm cap:** a per-row geom above 3000 points is blocked by Iris. For tables
  whose finest grain exceeds the cap, layer 1 is dropped and the analysis is box +
  date-overlay only. A 10k-point swarm would be unreadable regardless.

### Graceful degradation for a 1-level axis

When an axis has a single level in the data (this dataset: `condition` = 1 level),
the SuperPlot becomes a **single-column descriptive** plot — one swarm/box +
per-date dots, no test. Still informative (distribution + replicate structure),
and the *same template* becomes a real comparison on a multi-condition dataset.
This means every table yields content even when only `condition` is present.

## Per-table configuration

The spine is uniformly `date → position_id → <object_key> → frame` (coarse →
fine), so picking the object level averages `frame` away, and every layer `level`
the template references (`<object_key>` and `date`) is a spine column — required
by Iris's `hierarchy.materialize_levels`.

| Table | Object key `<object_key>` | Axes (≥1 level) | Swarm? (objects vs 3000) |
|---|---|---|---|
| `cells_by_frame` | cell_id | condition, class_label | yes (1726) |
| `cell_neighbors_by_frame` | cell_id | condition, class_label | no (3452) |
| `contact_types_by_frame` | contact_type | condition | yes (81) |
| `density_by_frame` | label | condition | yes (81) |
| `edges_by_frame` | t1_event_id | condition | no (9610) |

**Caveat — non-cell tables.** For `contact_types`, `density`, and `edges` the
object key is itself category-like (`contact_type`) or event-like
(`t1_event_id`), and the most natural comparison is often *across that key* rather
than across `condition`. The premade `condition`/`class_label` SuperPlot is a
reasonable, honest default (and a real comparison on multi-condition data), but
the user may want to re-pivot these in Iris (e.g. `x = contact_type`). The
exporter does not try to guess that; it emits the agreed axes only.

Numeric descriptors = every column that is not a key/metadata column. Per the
explicit decision, **all** numeric descriptors are included (including
coordinate/reference columns like `centroid_x_um`, `orientation`); pruning is left
to the user in Iris.

## Components

All under `src/cellflow/aggregate_quantification/iris_export/` (backend-only):

1. **`document.py`** — `write_iris(table_df, schema, analyses, provenance) -> bytes`.
   Builds the v1.0 ZIP. The single place that knows the byte format. Approach A
   (reimplement ~25 lines); the format contract above is the spec it implements.
2. **`schema.py`** — `infer_schema(df, table_name) -> dict`. Column typing:
   - `identifier`: `date, position_id, cell_id, frame, t1_event_id, label,
     focal_label, neighbor_label`
   - `categorical`: `condition, class_label, contact_type, role`
   - `numeric`: everything else
   - `label` = leaf after last `.`; `unit` parsed from `_um2 → µm²`, `_um → µm`,
     `_s → s` suffix; `levels` listed for categorical columns.
3. **`analyses.py`** — `build_analyses(df, schema, table_name) -> list[dict]`.
   Emits the SuperPlot template per descriptor × axis, applying the swarm-cap and
   1-level-axis rules.
4. **`export.py`** — `export_dir(data_dir, out_dir)` + a thin CLI. Discovers the
   known CSVs in `data_dir`, builds schema/analyses/bundle per table, writes
   `<table>.iris`.

### Defaults

- **Output location:** `<data_dir>/iris/<table>.iris`.
- **Writer:** approach A (reimplement). Approach B (vendor Iris's `save_document`)
  is the fallback if format drift is ever observed.

## Testing

- **Structural contract test:** a written bundle is a valid ZIP with the five entry
  kinds; schema types `date`/`position_id`/`cell_id` as `identifier`; every
  analysis has the 3-layer (or 2-layer, capped) shape, `notch` true, and
  `color=date`/`shape=date` encodings; `analyses/NN-*.json` numbering is contiguous.
- **Fixture round-trip:** a small synthetic tidy table → export → re-open the ZIP
  and assert table values + schema survive Parquet round-trip.
- **Optional live integration:** when an Iris engine checkout is discoverable
  (`IRIS_ENGINE` env), load each bundle through it and assert it produces a figure
  + a stats read-out without error. Skipped otherwise so the suite stays
  self-contained.

## Run target

```
python -m cellflow.aggregate_quantification.iris_export.export \
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
