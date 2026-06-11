# NLS classification as a sidecar CSV + unified `aggregate_quantification/` output folder

**Date:** 2026-06-11
**Status:** Implemented

## Problem

The NLS subpopulation classifier currently patches its result directly into the
contact-analysis `.h5` (`cells/table`): a `class_label` column, several
companion columns (`nls_status`, `nls_track_intensity`, `nls_track_pixel_count`,
`nls_track_frame_count`), and an audit metadata group under
`cells/measurements/nls_classification`.

This is fragile and wrongly coupled:

- The contacts build **owns and rewrites** the `.h5`. Re-running the build
  silently wipes any classification patched into it.
- Burying an iterative, hand-tuned classification inside a binary build artifact
  makes it hard to inspect, version, hand-edit, or share.
- Writing into `cells/table` couples the classifier to the H5 internal schema.

Separately, the Aggregate Quantification outputs are inconsistently located: the
shape quantifiers write into an `aggregate_analysis/` subfolder while contacts
writes a loose `contact_analysis.h5` at the position root.

## Goals

1. The NLS classifier writes its own **sidecar CSV** — exactly two columns,
   `id,label` — and never touches the `.h5`.
2. The `.h5` becomes a pure, regenerable build artifact (no `class_label`).
3. The CSV is the single source of truth for the subpopulation label; consumers
   join it by `cell_id` where they need it.
4. **All** Aggregate Quantification outputs (contacts, shape family, NLS) live in
   one per-position `aggregate_quantification/` folder.

## Non-goals

- No migration of pre-existing loose `contact_analysis.h5` files. This is the
  mid-redesign branch; changing the default output path simply orphans old loose
  files, which is acceptable.
- No persistence of the audit trail (threshold, method, percentile, per-track
  intensity/pixel/frame counts). The interactive plugin still *displays*
  intensities live; they are not written anywhere. (Decided: strictly two
  columns.)

## Design

### 1. Output-folder convention (all quantifiers + NLS)

Introduce a shared module constant `OUTPUT_SUBDIR = "aggregate_quantification"`.

`Quantifier.default_output` becomes:

```
position_dir / OUTPUT_SUBDIR / default_output_name
```

so each quantifier sets just a bare filename:

| quantity          | `default_output_name`   | resolves to                                            |
|-------------------|-------------------------|--------------------------------------------------------|
| contacts          | `contact_analysis.h5`   | `aggregate_quantification/contact_analysis.h5`         |
| cell_shape        | `cell_shape.csv`        | `aggregate_quantification/cell_shape.csv`              |
| nucleus_shape     | `nucleus_shape.csv`     | `aggregate_quantification/nucleus_shape.csv`           |
| shape_relational  | `shape_relational.csv`  | `aggregate_quantification/shape_relational.csv`        |
| NLS (not a quantifier) | `nls_classification.csv` | `aggregate_quantification/nls_classification.csv` |

This renames the shape outputs' folder from `aggregate_analysis/` →
`aggregate_quantification/` and moves the contacts `.h5` into the same folder.

Catalog's `contact_name` default changes to
`aggregate_quantification/contact_analysis.h5`.

### 2. NLS headless core (`contacts/nls_classification.py`)

- **Remove** `write_nls_classification` (the H5 patcher), the H5 plumbing in the
  per-position helper, the `cells/table` overlap check, and `read_position_cell_ids`.
  The classifier needs only the NLS image + nucleus labels — never the `.h5`.
- **Add** `write_nls_classification_csv(path, assignments, *, positive_label,
  negative_label)` that writes exactly two columns, `id,label`: one row per
  classified track, the label being the user's positive/negative string.
- **Add** `read_nls_classification_csv(path) -> dict[int, str]` (cell_id → label)
  for consumers.
- Rename `patch_position_contact_analysis_nls_classes` → a CSV-writing batch
  helper that takes the NLS image + labels + output CSV path (no H5 input). It
  measures, auto-thresholds, classifies, and writes the CSV. `NLSClassificationSummary`
  drops `h5_path` in favor of the CSV path.
- Keep all measurement / thresholding / classification logic unchanged
  (`measure_track_nls_intensity`, `auto_threshold`, `classify_by_threshold`,
  the two-cluster / Otsu splitters).

### 3. The label join (`plotting.py`)

The CSV is keyed by `cell_id` only (one row per track, no `frame`). Generalize
`_join_position` so the join key is `KEY_COLUMNS ∩ join_table.columns` and must
include `cell_id`:

- join table with only `cell_id` → join on `cell_id` alone, broadcasting the
  label across every frame of that cell;
- join table with `(frame, cell_id)` → unchanged behavior.

The downstream grouping column stays named `class_label`. The CSV file uses
`id,label`; the `label` column is mapped to `class_label` when a consumer builds
its join table.

### 4. Consumers

- **`contacts/build.py`** — remove the empty `class_label` column from
  `_extract_cell_columns`, its entry in the column-name list, and its membership
  in the string-column set. The H5 no longer carries `class_label`.
- **shape plugin `_pool_records`** — instead of joining `class_label` from
  `contacts.object_table`, build the join table from the NLS CSV
  (`{cell_id, class_label}`) when the CSV exists for the record. Positions
  without a CSV fall to `unclassified` exactly as today.
- **`contact_visualization.py`** — `build_cell_centroid_points` and
  `build_edge_shapes` read the NLS CSV and map `cell_id → class_label` for the
  `color_by_label` feature, instead of reading `class_label` from the H5. Absent
  CSV → all cells `unclassified`.
- **NLS plugin** (`napari/.../plugins/nls_classification.py`) — `_on_apply` and
  the batch path write the CSV (path derived from each record's position folder +
  subdir) rather than patching the H5. The `contact_analysis_path` and
  cell-id-overlap logic is removed. Button labels change from "Apply to H5" /
  "Classify & apply to all H5" to CSV-oriented wording (e.g. "Save CSV" /
  "Classify & save all CSVs").

### 5. CSV-path discovery

Add a small helper that derives the NLS CSV path from a catalogue record:

```
record["position_path"] / OUTPUT_SUBDIR / "nls_classification.csv"
```

used by the NLS plugin (write), the shape plugin (join), and contact_visualization
(color). Optionally surface `nls_classification_path` + a status on catalogue
records for consistency with `contact_analysis_path`, but consumers can derive it
directly from `position_path`.

## Data flow (after)

```
NLS image + nucleus labels ──> measure ──> threshold ──> classify
                                                            │
                                                            ▼
                              aggregate_quantification/nls_classification.csv  (id,label)
                                                            │
        ┌───────────────────────────────────────────────────┼─────────────────────────┐
        ▼                                                     ▼                         ▼
  shape plugin join (cell_id)                     contact_visualization            (future consumers)
  → class_label on pooled df                      color_by_label
```

The contacts `.h5` is built and read with no `class_label` anywhere in it.

## Testing

- **Core:** `write_nls_classification_csv` / `read_nls_classification_csv`
  round-trip (two columns only); classification → CSV content; batch helper
  writes one CSV per position with no H5 dependency.
- **Plotting:** cell_id-only join broadcasts the label across frames;
  `(frame, cell_id)` join still works; missing CSV / missing column → `unclassified`.
- **Build:** the H5 `cells/table` no longer has a `class_label` dataset.
- **Plugins:** NLS write targets `aggregate_quantification/nls_classification.csv`;
  shape pooling joins `class_label` from the CSV; contact_visualization colors
  from the CSV.
- **Update existing tests** that assert H5-patching: `test_nls_classification.py`,
  `test_nls_classification_plugin.py`, `test_public_private_boundary.py`, plus any
  catalog/output-path tests that hard-code `contact_analysis.h5` at the position
  root or `aggregate_analysis/`.
