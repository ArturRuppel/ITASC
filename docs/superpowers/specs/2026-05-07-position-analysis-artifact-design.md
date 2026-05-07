# Per-Position Analysis Artifact Design

Date: 2026-05-07

## Purpose

CellFlow needs one canonical analysis artifact per position before the meta analyzer can aggregate across positions, experiments, and conditions. This artifact describes tracked cells, their matching nuclei, canonical edge trajectories, T1 transition markers, and later object-level measurements.

The artifact is stored as HDF5. It is intentionally lightweight: tracked label images remain external TIFF files, and the HDF5 stores paths plus derived tables and edge geometry.

## Inputs

The first generator consumes:

- `cell/tracked_labels.tif`
- `nucleus/tracked_labels.tif`

The integer label value is the persistent identity. For every object, the following invariant holds:

```text
cell_id == nucleus_id
```

There is no separate `track_id` in the first artifact schema.

## Storage Principles

- Store paths to tracked label TIFFs, not embedded image arrays.
- Do not store cell or nucleus contours. They can be reconstructed from the referenced tracked label images using `cell_id`.
- Store edges in the HDF5 as canonical first-class objects, including border edges.
- Store edge coordinates in the HDF5 because later force and topology analyses attach values to those exact edge objects.
- Store only factual extracted values. Derived conventions such as signed edge length are produced by the meta analyzer.
- Later scalar measurements are stored as named aligned datasets under cell or edge measurement groups.

## HDF5 Layout

```text
/provenance
  attrs:
    source_position_path
    cell_tracked_labels_path
    nucleus_tracked_labels_path
    edge_extraction_params_json
    created_at
    cellflow_version

/cells/table
  columns:
    frame
    cell_id
    class_label
    area
    centroid_y
    centroid_x
    perimeter
    bbox_min_y
    bbox_min_x
    bbox_max_y
    bbox_max_x

/cells/measurements/<name>
  values
  attrs:
    unit
    source
    created_at
    run_id

/edges/table
  columns:
    frame
    edge_id
    cell_a
    cell_b
    kind
    edge_label
    is_t1_frame
    t1_event_id
    length
    midpoint_y
    midpoint_x
    coord_offset
    coord_count

/edges/coordinates
  y
  x

/edges/measurements/<name>
  values
  attrs:
    unit
    source
    created_at
    run_id

/t1_events/table
  columns:
    t1_event_id
    frame
    edge_id
    losing_cell_a
    losing_cell_b
    gaining_cell_a
    gaining_cell_b
    location_y
    location_x
```

String columns may be implemented as UTF-8 HDF5 string datasets. Nullable integer fields such as `t1_event_id` should use a sentinel value, initially `-1`, with the sentinel documented in dataset attributes.

## Cells

`/cells/table` has one row per cell per frame. `cell_id` is the persistent label value from `cell/tracked_labels.tif` and also identifies the matching nucleus in `nucleus/tracked_labels.tif`.

`class_label` is frame-scoped and initially empty for every row. Later annotation steps may populate it with labels such as `KO`, `Ctrl`, or other project-specific classes. A track-level or majority class can be derived later by the meta analyzer.

Cell and nucleus contours are intentionally omitted. Loaders can reconstruct them on demand from the referenced label images.

## Edges

`/edges/table` has one row per edge per frame. `edge_id` is persistent through time and through T1 transitions. This means an edge trajectory can refer to one cell pair before a T1 and another cell pair after the T1 while retaining the same `edge_id`.

Border edges are first-class edges. They use:

```text
cell_b = 0
kind = "border"
edge_label = "border"
```

Cell-cell edges use:

```text
kind = "cell_cell"
edge_label = ""
```

Each edge row stores its own measured length, midpoint, and coordinate span. Edge coordinates are stored as ragged arrays in `/edges/coordinates/y` and `/edges/coordinates/x`. `coord_offset` and `coord_count` point from each edge row into those arrays.

## T1 Events

T1 transition frames are represented in two places:

- `/edges/table/is_t1_frame` flags the edge row at the transition frame.
- `/edges/table/t1_event_id` links the edge row to `/t1_events/table` when applicable.

`/t1_events/table` records the event frame, persistent `edge_id`, losing pair, gaining pair, and event location.

The artifact does not store signed edge lengths. The meta analyzer can derive signed lengths from edge lengths, T1 flags, and the T1 event table.

## Measurements

Later analysis steps attach scalar object measurements as aligned datasets:

- cell pressure: `/cells/measurements/pressure`
- edge tension: `/edges/measurements/tension`
- other scalar values as additional named measurement datasets

Each measurement dataset has one `values` array aligned row-for-row with the corresponding base table. Metadata attributes record unit, source, creation time, and run ID.

Non-scalar or complex analysis outputs should be stored under a future analysis-run namespace rather than forced into the base object schema.

## Relationship To The Meta Analyzer

This per-position H5 is the canonical source artifact for one position. The meta analyzer later discovers many such artifacts, indexes their provenance, lazily loads their tables and referenced label images, resolves annotations, derives higher-level quantities, and builds cross-position tables.

The meta analyzer owns derived views such as signed lengths, cohorts, condition-level summaries, and aggregate metrics. The per-position artifact stays close to extracted facts.

## Implementation Notes

The archived `TissueGraphTimeSeries`, `CellData`, `JunctionData`, `T1Event`, and `EdgeTrajectory` concepts are useful implementation references, but the new persisted artifact should be table-oriented and reference-friendly. Existing graph extraction and T1 logic can be adapted if it preserves the schema invariants above.

The first generator should be deterministic for fixed inputs and edge extraction parameters. Re-running it on unchanged tracked labels and parameters should produce the same cell rows, edge rows, edge IDs, T1 flags, and coordinate spans.
