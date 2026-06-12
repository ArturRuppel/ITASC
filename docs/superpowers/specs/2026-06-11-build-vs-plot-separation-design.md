# Aggregate Quantification — Build / Plot Separation

**Date:** 2026-06-11
**Status:** Implemented (2026-06-11)

## Implementation notes (as built)

- **Plot consumer registry** — `napari/aggregate_quantification/plots/__init__.py`:
  headless `Plot` base (`plot_id` / `display_name` / `family` / `consumes`,
  `is_available` / `missing`, `prepare` + `create_panel`), mirroring `Quantifier`.
- **Availability** — `studio_plugins.built_quantity_ids(records)` is the product
  set the Plot area gates on.
- **16 plot consumers**: Shape (3), Dynamics (8: per-frame/track/tissue + curves ×
  cell/nucleus), Contacts (5: potential landscape + 4 neighborhood/density). Generic
  ones subclass `PoolPlot` (pool one product → `PlotPanel`); bespoke ones (dynamics
  curves) build their own widget. `_pooling.pool_quantity` is the shared pooling.
- **Plot area** — `aggregate_quantification_plot_area.PlotAreaWidget`: family-grouped,
  availability-gated launcher; disabled plots name the missing product; launches
  off-thread (`prepare`) then docks via the shared `PlotDockTabs`.
- **Build area** — every quantifier now gets a generic `BuilderPlugin`
  (`owns_quantities` suppression removed). The old vertical plot plugins (shape,
  track_dynamics, neighborhood, contact_energetics) were deleted; NLS classification,
  contacts visualizer, and catalogue summary survive as tools.
- **Shared plot parameters** — the old per-plugin plot-time knobs are unified into
  **one** `PlotParams` (pixel size, FOV area, z-score shuffles) surfaced as a single
  "Plot parameters" section in the Plot area and threaded into every
  `Plot.prepare(records, params)`. Each plot reads only the fields it needs; blank =
  auto (per-position resolution). No per-plot duplication.
- NLS interactive tuner unchanged (instrumental, stays in build).


## Problem

The Aggregate Quantification studio is organized around **vertical plugins**
(`AnalysisPlugin` subclasses: `shape`, `track_dynamics`, `neighborhood`,
`contact_energetics`, `nls_classification`, …). Each plugin staples together two
concerns that don't actually belong together:

1. **Build** — run a per-position calculation over label/contact data and write
   the result to a file.
2. **Plot** — render some of those results as a figure.

That staple is the wrong mental model. In practice **every** plugin does the same
build step (compute → write file), and plotting is a *cross-cutting* concern over
whatever results happen to exist — not a property of any single analysis. Some
results can be plotted, some can't; some plots need only a subset of outputs; some
plots combine outputs from two different analyses. The current structure forces
each plugin to re-invent the build→plot wiring and hardcode its own answer to
"what can I plot right now," which is opaque to the user and duplicated in code.

The backend already proves the split is natural: `Quantifier`
(`aggregate_quantification/quantifier.py`) is a clean **producer** abstraction
(`quantity_id`, `requires` input fields, `is_built`, emits an `object_table`), and
`PlotPanel` (`napari/aggregate_quantification/plot_panel.py`) is a napari-free,
quantity-agnostic **consumer** (`df, value_columns, group_columns` → figure).
What's missing is the explicit *product* vocabulary that connects them, plus the
availability logic that currently lives, scattered and imperative, inside each
plugin.

## Core idea: Producer → Product → Consumer

Separate the studio into two areas joined by a single shared vocabulary — the
**Product** (a named, on-disk analysis output):

```
ANALYSES (build area)              PRODUCTS (the contract)        PLOTS (plot area)
  produce ──────────────▶   e.g. cell_shape.object_table   ◀────────── consume
  Quantifier.requires =            contacts.energetics              Plot.consumes =
    {label/contact inputs}         track.msd                          {product ids}
                                   neighborhood.density
```

- An **Analysis (producer)** declares the input data it needs (`requires`, exists
  today) and the **products it emits** (`produces` — made explicit, today implicit
  in `object_table`).
- A **Plot (consumer)** declares only the **products it consumes**. It never knows
  which analysis produced them.
- **"Is this plottable?" is no longer a property of the analysis.** It is simply:
  *does any registered Plot consume a product this analysis emits?* The "some but
  not all can be plotted" behavior falls out for free.

The `AnalysisPlugin` vertical-slice layer **almost entirely dissolves** into two
thin registries that mirror the existing `Quantifier` registry: producers and
presentation consumers.

## The decisive distinction: presentation vs instrumental

Not every "visualization" belongs in the plot area. The dividing rule:

> A visualization goes in the **plot area** iff it consumes a product that already
> exists on disk. If it exists only to help *create* the product, it stays in the
> **build area**, coupled to its analysis.

- **Presentation plot** — reads a *finished* product for browsing/styling/export.
  Read-only, downstream, no feedback into the data. → **plot area**, as a consumer.
  (Shape, dynamics, energetics landscape, neighborhood presets.)
- **Instrumental visualization** — a render that lives *inside the build loop* so a
  parameter can be tuned against live feedback before the artifact is committed.
  The viz is an input to *producing* the output, not a view of it. → **build area**,
  coupled to its analysis.

`nls_classification` is the *only* instrumental case: its napari overlay is the
feedback signal for picking a threshold, not a product anyone revisits in a
plotting surface. It is therefore **not** split — there is nothing to split.

## The build area is near-trivial

Building is automatic for essentially every analysis (NLS included — its
auto-threshold path already works well). So the build area is **not** a surface of
per-analysis widgets. It collapses to a single generic control driven entirely by
the `Quantifier` registry:

> select scope → see which products are **built / stale / buildable** (derived from
> `requires` + `is_built` over in-scope positions) → tick what to (re)build → run.

Adding a quantifier adds a row; nothing else. NLS uses this same generic builder
with auto-threshold as its normal path. Its interactive tuner becomes an **optional
escape hatch** — a per-position "this looks off → adjust" action launched only when
the automatic result is wrong. It hangs off the generic builder; it is not a
co-equal UI and does not distort the build surface.

This repoints the design weight: the build side is "wire the registry to one
widget"; **all the real design surface is on the plot side.**

## The plot area (where the design weight is)

A registry of **Plot** consumers, each declaring `consumes = (product_id, …)`.
Three mechanisms make the analysis↔plot relationship *clear* to the user:

### 1. Availability is derived, never declared per-plugin

A plot is **live** iff every product it consumes is present in the intersection of:

1. products the plot **consumes** (declared on the plot),
2. products **registered** at all (declared across analyses),
3. products actually **built for the current selection** (`is_built` over in-scope
   positions).

Otherwise the plot is shown but **greyed**, with the affordance that makes it
clear *why*:

> *Box plot — needs `cell_shape` (not built for any selected position). → [Build it]*

This single rule replaces all per-plugin `requires` / `owns_quantities` / gating
logic. The user never wonders "why can't I plot this" — the list names the missing
product and offers a jump to building it.

### 2. Grouping by product family ("type of input data")

The plot list is grouped by **product family / source**, not by plugin:
contacts-derived plots in one section, label/shape-derived in another,
track-derived in another. This is the "group plots by type of input data" the user
asked for. A plot that consumes products from two families lands in a "combined"
group (or under its primary input — see open items).

### 3. Two consumer flavors, one registration contract

Both register identically (same `consumes` contract); they differ only in how they
render:

- **Generic statistical plots** — distribution/bar/line over tidy columns.
  `PlotPanel` already *is* this; it is fed the in-scope built products' tables
  instead of one plugin's table.
- **Bespoke plots** — energetics landscape, neighborhood presets — own their figure
  and bring their own render widget instead of routing through `PlotPanel`. They
  still declare `consumes`, so availability and grouping stay uniform.

## How current plugins map onto the model

| Today (vertical plugin) | Build side (producer)                       | Plot side (consumer)                  |
|-------------------------|---------------------------------------------|---------------------------------------|
| Shape                   | cell/nucleus/relational shape quantifiers   | generic `PlotPanel` (shape products)  |
| Track dynamics          | cell/nucleus dynamics quantifiers           | generic `PlotPanel` + per-tissue view |
| Neighborhood            | density quantifier                          | bespoke preset plots                  |
| Contact energetics      | energetics quantifier                       | bespoke landscape plot                |
| NLS classification      | classifier (auto-threshold) **+ optional tuner** | — (instrumental, *not* in plot area) |
| Catalog / Visualize     | — (not producers)                           | viewer-only, unchanged                |

## Net architecture

- **Build area** — one generic builder over the `Quantifier` registry. NLS = same
  builder + an optional per-position tuner (escape hatch).
- **Plot area** — pure presentation consumers, grouped by product family, gated by
  derived availability; generic `PlotPanel` or bespoke render.
- **`AnalysisPlugin`** — dissolves into two registries (producers + presentation
  consumers) plus the one optional NLS tuner.

## Decided: product granularity is coarse (per `quantity_id`)

A **product is a whole `object_table`, identified by its `quantity_id`** — not a
column or column-set. This is the natural unit because **a quantifier writes its
entire table atomically**: every column appears together or not at all, so
column-level granularity gives the same on/off availability answer as table-level
with more bookkeeping. The "this plot uses only `area`, that one uses
`eccentricity`" concern is a *different* job, already handled by `value_columns`
selection **inside** `PlotPanel`; the dependency graph only needs "is the table
built." Keeping these separate is the point.

The one case that would justify finer products: two *different* analyses each
independently supplying the same logical column (e.g. centroids from either the
shape or the dynamics quantifier), so a plot wants "centroids, from whoever has
them." That is rare; handle it *then* by naming that single shared product
explicitly — do **not** make everything column-granular to pre-empt it. No current
plot needs this, so it is out of scope until one does.

## Open items to confirm during implementation

- **Multi-input plots** — where does a shape×contact plot group, and is its
  availability the AND of both products? (AND: yes. Grouping placement: open.)
- **Bespoke plots via the same registry** — confirm they declare `consumes` and
  bring their own widget rather than living as special cases outside the plot area.
- **NLS tuner surface** — confirm the tuner is a per-position on-demand action off
  the generic builder, not a persistent widget.

## Explicitly out of scope (YAGNI)

- Rewriting the headless `Quantifier` backend — it already is the producer model.
- A plugin marketplace / external plugin loading — the registry stays in-process
  (`__init_subclass__` auto-registration), as today.
- Live-linked plotting — the plot panel keeps its snapshot semantics (see the
  detached plot-panel design).
