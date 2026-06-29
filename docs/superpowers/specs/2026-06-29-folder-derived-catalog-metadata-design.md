# Folder-Derived Catalog Metadata — Discover-and-Add Port from napariTFM

**Date:** 2026-06-29
**Status:** Approved (design); implementing

## Problem

CellFlow's Aggregate Quantification "Discover & add" types fixed metadata
(`condition` / `date` / `notes`) once per discovered batch. napariTFM's newer
discover-and-add uses **free-form columns** — arbitrary `(name, value)` pairs
copied onto each row, with rows grouped by their column caption. We want that
flexibility here, but in CellFlow `condition` / `experiment_id` / `position_id`
are not cosmetic: they are **identity + pooling axes** stamped onto every output
table row (`shape_tables._position_metadata`, `METADATA_COLUMNS`), used for dedup
(`_identity_key`), and feed the deterministic row-id hash (`_ROW_ID_IDENTITY`).
A naive free-form model would let an identity axis go missing or vary.

## Approach: the folder tree is the metadata table

Discovery already finds *position folders* (folders holding ≥1 input). The path
from the scan root to each position folder is a sequence of segments. **Each
nesting level becomes a named column; its value is that folder's name**, inherited
per position from its own path.

```
root/
  WT/              level 1  → named "condition"      → value "WT"
    2024-01-15/    level 2  → named "experiment_id"  → value "2024-01-15"
      pos3/        level 3  → named "position_id"     → value "pos3"
        cell_labels.tif        (discovery input)
```

- **Anchor at root** (level 1 = first folder under root). Depth must be **uniform**
  across discovered positions; differing depths warn/refuse.
- **Level names editable before and after discovery.** Renaming a level
  re-derives that column on every row live.
- Levels named `condition` / `experiment_id` / `position_id` (and `date`,
  `notes`) are the **recognized** identity/pooling axes; any other level name is
  an **extra descriptor column** that rides through to the output tables.
- A **manual `+ Add column`** unit (napariTFM-style) supplies batch-wide constant
  tags not encoded in the folder tree, copied onto every row of the batch.

## Data model

A record gains a canonical metadata bag `columns: dict[str, str]` (the union of
folder-derived levels + manual constant columns). Flat aliases
(`condition` / `experiment_id` / `id` / `date` / `notes`) are kept for backward
compatibility with every existing reader; they are derived from `columns` with
today's defaults/fallbacks.

- `catalog.py`
  - `save_catalog`: write the fixed core columns (`position_path`, `path`,
    `cell_labels`, `nucleus_labels`, `id`) **plus the union of all `columns` keys**;
    `condition` / `date` / `experiment_id` / `notes` always emitted (backward
    compatible). Extra columns appended.
  - `load_catalog`: already keeps extra columns; additionally collect non-core
    columns into `record["columns"]`.
  - Identity stays narrow: `_identity_key` / `_check_unique_identity` keep using
    `(experiment_id, condition, id)` only. Extra columns never join dedup.
  - New helpers: `relative_levels(root, position_path)` → segment tuple;
    `columns_from_levels(level_names, segments)` → dict; a uniform-depth check.
- `shape_tables.py`
  - `_position_metadata` returns the recognized axes **plus every `columns` key**,
    so each extra column is stamped onto output rows as a constant descriptor.
    `_ROW_ID_IDENTITY` unchanged (row-id hash stays stable).
- `reduce.py`
  - Add `experiment_id` to `IDENTITY_COLUMNS`. Free-form **string** descriptors
    already survive `Collapse` (kept when constant within a group, never averaged
    because non-numeric). Numeric-looking descriptors are out of scope (stamped as
    strings; document the edge).

## UI — `aggregate_quantification_studio._build_discover_section`

- Replace the three fixed `Condition / Date / Notes` fields with:
  - a **levels editor**: one name field per detected nesting level (pre-seeded:
    innermost = `position_id`, then guesses outward), editable before and after
    discovery; editing re-derives columns on staged/added rows.
  - the manual **`+ Add column`** repeatable `(name, value)` unit.
- **Discover → stage → commit polish**: live staging label
  ("N positions discovered — Add to catalogue"), enabled/disabled commit button,
  and a **dynamic Discover tooltip** naming the filled input files.
- **Bounded scroll** for the discovered-staging list and the catalogue table.
- **Rich rows + group separators**: keep the `QTableWidget` (so CellFlow's
  multi-select + Remove selected + Clear survive), add non-selectable
  **group-separator rows** captioning each batch by its column values.

## Keep / drop

- **Keep** (CellFlow): multi-select rows + Remove selected + Clear; Load/Save CSV;
  Catalogue/Parameters/Tools/Run layout; discover semantics (≥1 input).
- **Drop / not porting**: progressive disclosure (G0/G1/G2) and the Project
  save/load/dirty-flag shell.

## Tests

- `catalog.py`: round-trip a catalog with extra free-form columns through
  save/load; identity stays narrow; legacy fixed-column CSVs still load; level
  helpers derive columns from segments; uniform-depth check.
- `shape_tables.py` / `reduce.py`: an extra column appears on output rows and
  survives a `Collapse` as a constant descriptor.
- Studio (headless where possible): discover → stage → commit copies columns onto
  rows; renaming a level re-derives the column; group separators caption batches;
  multi-select remove still works.

## Risk

Backward compatibility of `catalog.csv` for the external data-repo (cov2d)
code — mitigated by always emitting the recognized columns and only *appending*
extra ones.
