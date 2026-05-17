# Position Analysis HDF5 Project Description

Analysis root: `/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis`

This project explores CellFlow `contact_analysis.h5` files. Each file is a canonical per-position artifact built from tracked cell and nucleus label images. The artifact stores tabular cell measurements, cell-cell edge/contact measurements, T1 transition records, edge coordinate arrays, and provenance attributes.

The notebook discovers files with this pattern:

```text
pos*/4_contact_analysis/contact_analysis.h5
```

It intentionally excludes other `.h5` files such as graphcut or ICM unary caches.

## Current Discovery

Discovered position artifacts: 9

Positions: pos00, pos01, pos02, pos03, pos04, pos05, pos06, pos07, pos08

Total cell rows: 20,284

Total edge rows: 163,438

Total T1 events: 780

## Tables

### cells/table

Rows: 20,284

Columns:
- `frame`
- `cell_id`
- `area`
- `centroid_y`
- `centroid_x`
- `perimeter`
- `bbox_min_y`
- `bbox_min_x`
- `bbox_max_y`
- `bbox_max_x`
- `class_label`
- `nls_status`
- `nls_track_median_intensity`
- `nls_track_pixel_count`
- `nls_track_frame_count`
- `shape_index`


### edges/table

Rows: 163,438

Columns:
- `frame`
- `edge_id`
- `cell_a`
- `cell_b`
- `kind`
- `edge_label`
- `is_t1_frame`
- `t1_event_id`
- `length`
- `midpoint_y`
- `midpoint_x`
- `coord_offset`
- `coord_count`


### t1_events/table

Rows: 780

Columns:
- `t1_event_id`
- `frame`
- `edge_id`
- `losing_cell_a`
- `losing_cell_b`
- `gaining_cell_a`
- `gaining_cell_b`
- `location_y`
- `location_x`


## Provenance

Each file has a `provenance` group with attributes such as source position path, tracked cell labels path, tracked nucleus labels path, edge extraction parameters, creation timestamp, and CellFlow version.

## DataFrames Created By The Notebook

- `catalog`: one row per discovered HDF5 file
- `cells`: pooled `cells/table` rows across positions
- `edges`: pooled `edges/table` rows across positions
- `t1_events`: pooled `t1_events/table` rows across positions
- `inventory`: one row per HDF5 dataset path per position
- `schema_summary`: compact schema inventory across positions
- `cells_per_frame`: cell counts and total cell area by position/frame
- `edges_per_frame`: edge counts and mean edge length by position/frame
