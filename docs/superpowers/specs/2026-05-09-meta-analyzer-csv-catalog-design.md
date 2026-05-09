# Meta Analyzer CSV Catalog Design

Date: 2026-05-09

## Purpose

The meta analyzer needs a source discovery system that does not depend on a rigid `condition/experiment/position` folder layout. Users should be able to build a catalog of position analysis H5 files from single-file selections and folder autodiscovery, then attach the metadata needed for cross-position analysis.

The catalog must be easy to inspect, edit, version, and script against. CSV is the storage format for this first implementation.

## CSV Contract

The meta catalog is a CSV file with one row per H5 source artifact. The required columns are:

```text
path,date,condition,id,labels
```

- `path`: path to the source H5 file. When saving a catalog, paths should be written relative to the CSV file when possible and absolute otherwise. When loading, relative paths resolve against the CSV file parent.
- `date`: experiment or acquisition date. The first implementation treats it as text so labs can use their current naming convention.
- `condition`: experimental condition or cohort.
- `id`: source identifier, typically a position, tissue, sample, or field-of-view ID.
- `labels`: optional free-text tags. Multiple labels may be stored as comma-separated text inside the CSV cell.

The loader should tolerate extra columns and preserve them in record dictionaries where practical, but the UI only needs to expose the five required columns at first.

## Backend Design

Extend `cellflow.meta.catalog` with CSV-oriented helpers while keeping the existing folder discovery function available for compatibility:

- `load_meta_catalog(csv_path)`: read CSV rows, resolve each `path`, normalize record keys, and return records sorted by `condition`, `date`, and `id`.
- `save_meta_catalog(csv_path, records)`: write the required columns, preferring relative paths from the CSV location.
- `discover_h5_files(folder, recursive=True)`: return sorted H5 paths from a selected folder. The default should include nested folders because CellFlow outputs are usually inside stage subdirectories.
- `records_from_h5_paths(paths, defaults=None)`: create catalog rows from selected or discovered H5 files, filling metadata with conservative defaults.
- `merge_catalog_records(existing, incoming)`: add new records while avoiding duplicate resolved H5 paths.

For compatibility with the existing meta source browser, each loaded CSV row should also expose the record keys currently used by the widget:

- `condition_id` from `condition`
- `experiment_id` from `date`
- `position_id` from `id`
- `artifact_path` from resolved `path`
- `analysis_status` as `ready` when the H5 path exists, otherwise `incomplete`

The existing label-image readiness checks from the folder-contract discovery should not block CSV records. A CSV row represents an explicit user-selected analysis H5, so loading should depend on the H5 path being present.

## Napari Widget Design

Update `MetaSourceBrowserWidget` to operate on a CSV catalog in addition to the current in-memory record list.

Initial controls:

- Open catalog: choose an existing CSV and load it.
- Save catalog: write the current records to the active CSV path.
- Add H5: choose one H5 file and append it to the catalog if it is not already present.
- Autodiscover folder: choose a folder, find H5 files below it, and append new entries.
- Existing condition/date/id selectors continue to drive source selection.
- Load Source continues to read the selected `artifact_path` and add artifact layers to the viewer.

Metadata editing can start with conservative defaults in the generated rows:

- `date`: `unknown_date`, unless a simple parent-folder date convention is obvious.
- `condition`: `unknown_condition`.
- `id`: H5 stem or nearest position-like parent folder, made unique when needed.
- `labels`: empty string.

The first implementation should keep inference minimal. Users can edit the CSV externally, reload it, and a future UI iteration can add an in-widget table editor.

## Data Flow

Single file add:

1. User clicks Add H5.
2. File browser returns one H5 path.
3. Backend converts it to a catalog record.
4. Widget merges it into the current record list.
5. Selectors refresh and Load Source becomes available when the file exists.

Folder autodiscovery:

1. User clicks Autodiscover folder.
2. Folder browser returns one directory.
3. Backend recursively finds H5 files.
4. Backend converts paths to records and deduplicates against existing records.
5. Widget refreshes selectors.

Catalog persistence:

1. Open catalog loads CSV records and stores the active CSV path.
2. Save catalog writes current records to the active CSV path.
3. If no active CSV path exists, Save catalog asks for a target path.

## Error Handling

- Missing or unreadable CSV files should leave the current catalog unchanged and surface a concise widget status message.
- Missing H5 paths load as `analysis_status="incomplete"` and disable Load Source for that row.
- Duplicate H5 paths should be skipped during merge.
- Empty autodiscovery results should not clear existing records.
- CSV rows missing required columns should raise a clear validation error naming the missing columns.

## Testing

Add focused backend tests for:

- CSV load/save round trip.
- Relative path resolution from the CSV parent.
- Duplicate avoidance by resolved H5 path.
- Recursive folder discovery of H5 files.
- Missing H5 path status.
- Extra column tolerance.

Add widget tests for:

- Catalog action buttons exist.
- Loading a CSV populates selectors.
- Adding an H5 appends a ready record.
- Autodiscovery appends multiple H5 records and skips duplicates.
- Save writes the current catalog through the backend helper.
- Load Source still uses the selected record artifact path.

## Out Of Scope

- Metric computation, cohort analysis, and plotting.
- A full in-widget spreadsheet editor.
- Automatic parsing of arbitrary lab naming conventions.
- Copying, moving, or modifying source H5 files.
- Requiring nucleus or cell label TIFF files for CSV-selected records.
