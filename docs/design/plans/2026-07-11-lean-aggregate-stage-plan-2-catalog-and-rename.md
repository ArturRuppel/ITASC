# Lean Aggregate Stage — Plan 2: Catalog slim + `4_contact_analysis` rename

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the contact-analysis output folder to `4_contact_analysis/`, drop `date` from the pooled tables, slim the project catalog CSV to identity + canonical artifact paths, and fix the full-app adapter to stamp the *committed* label paths.

**Architecture:** `OUTPUT_SUBDIR` becomes `4_contact_analysis` (one constant; `CONTACT_ANALYSIS_RELPATH` derives from it), with three stray literals repointed at the constant. `shape_tables` stops stamping a `date` descriptor. `catalog.py` drops the `path` / `date` / `notes` columns (the contacts path moves to a canonical `contact_analysis_path` column) and requires `position_path`. `main_widget._catalog_record_for_position` stamps `cell_labels.tif` / `nucleus_labels.tif` (the finalize-button outputs) instead of the pre-commit `3_cell` / `2_nucleus` working paths. No data migration.

**Tech Stack:** Python, csv, pandas, pytest, PyQt (napari), tifffile.

**Spec:** `docs/superpowers/specs/2026-07-11-lean-aggregate-stage-design.md` (Workstreams B + C).

**Prerequisite:** Plan 1 (quantifier seam) is merged. In particular the Plan 1 Task 9 test asserts the `aggregate_quantification/contact_analysis.h5` path; Task 1 below updates it to `4_contact_analysis/`.

---

## File Structure

- `src/cellflow/contact_analysis/quantifier.py:30` — `OUTPUT_SUBDIR = "4_contact_analysis"`.
- `src/cellflow/napari/main_widget.py:63-64, 602-604, 717` — repoint literals at the constant; adapter fix (committed label paths).
- `src/cellflow/napari/contact_analysis_widget.py:156`, `src/cellflow/napari/data_panel_widget.py:45` — repoint literals.
- `src/cellflow/contact_analysis/shape_tables.py:58, 188-204` — drop `date` from pooled metadata.
- `src/cellflow/contact_analysis/catalog.py:36, 41-51, 201-226` — slim CSV columns.
- Tests: many reference `aggregate_quantification` or the old columns; each task updates the ones it breaks.

---

## Task 1: Rename `OUTPUT_SUBDIR` to `4_contact_analysis`

**Files:**
- Modify: `src/cellflow/contact_analysis/quantifier.py:26-30`
- Modify: `src/cellflow/napari/main_widget.py:717`, `src/cellflow/napari/contact_analysis_widget.py:156`, `src/cellflow/napari/data_panel_widget.py:45`
- Tests to update: `tests/contact_analysis/test_dynamics_quantifier.py`, `tests/contact_analysis/test_pipeline.py`, `tests/contact_analysis/test_shape_tables.py`, `tests/contact_analysis/test_catalog.py`, `tests/contact_analysis/test_quantifier.py`, `tests/napari/test_contact_analysis_widget.py`, `tests/napari/test_studio_plugins.py`, `tests/napari/test_stage_status.py`, `tests/napari/test_contact_analysis_studio.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/contact_analysis/test_quantifier.py`:

```python
def test_output_subdir_is_stage_numbered():
    from cellflow.contact_analysis.quantifier import OUTPUT_SUBDIR
    from cellflow.contact_analysis.catalog import CONTACT_ANALYSIS_RELPATH
    assert OUTPUT_SUBDIR == "4_contact_analysis"
    assert CONTACT_ANALYSIS_RELPATH == "4_contact_analysis/contact_analysis.h5"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/contact_analysis/test_quantifier.py::test_output_subdir_is_stage_numbered -v`
Expected: FAIL (`OUTPUT_SUBDIR == "aggregate_quantification"`).

- [ ] **Step 3: Rename the constant + repoint literals**

In `src/cellflow/contact_analysis/quantifier.py`, change line 30 and its docstring (lines 26-30):

```python
#: Per-position subfolder holding the contact-analysis output, numbered to mirror
#: the staged layout (``0_input`` … ``3_cell``). After the lean-aggregate change it
#: holds only ``contact_analysis.h5`` (the one persisted per-position artifact); the
#: cheap quantities are pooled in memory, not written here.
OUTPUT_SUBDIR = "4_contact_analysis"
```

Repoint the three hardcoded literals at the constant:

`src/cellflow/napari/main_widget.py:717` — replace
`out_path=pos_dir / "aggregate_quantification" / "contact_analysis.h5",` with
`out_path=pos_dir / CONTACT_ANALYSIS_RELPATH,` (it is already imported at line 23).

`src/cellflow/napari/contact_analysis_widget.py:156` — replace
`("aggregate_quantification/contact_analysis.h5", "Contact analysis"),` with a use of
the imported constant: add `from cellflow.contact_analysis.catalog import CONTACT_ANALYSIS_RELPATH`
and use `(CONTACT_ANALYSIS_RELPATH, "Contact analysis"),`.

`src/cellflow/napari/data_panel_widget.py:45` — same change (import the constant, use it).

- [ ] **Step 4: Update the broken tests**

In each listed test file, replace `"aggregate_quantification"` with `"4_contact_analysis"` (path assertions such as `tmp_path / "aggregate_quantification" / "contact_analysis.h5"` and the dynamics `default_output` assertions). Update the Plan 1 Task 9 test in `tests/contact_analysis/test_pipeline.py` (`test_run_persists_only_contacts_per_position`) to assert `pos / "4_contact_analysis" / "contact_analysis.h5"` and the negative `pos / "4_contact_analysis" / "cell_shape.csv"`.

- [ ] **Step 5: Run to verify green + fix docstrings**

Run: `pytest tests/contact_analysis/ tests/napari/ -q`
Expected: PASS. Then update the prose that names the old folder (no behaviour): `src/cellflow/napari/_paths.py:17`, `src/cellflow/napari/main_widget.py:493` docstring, `src/cellflow/contact_analysis/catalog.py:10`, the quantifier module docstrings (`cell_shape.py:5`, `cell_dynamics.py:5`, `nucleus_shape.py:5`, `nucleus_dynamics.py:5`, `shape_relational.py:7`, `cell_density.py` if present), and the `_experiments_panel.py:118` ASCII tree (`aggregate_quantification` → `4_contact_analysis`).

- [ ] **Step 6: Commit**

```bash
git add src/cellflow tests
git commit -m "refactor(contact-analysis): rename output dir to 4_contact_analysis"
```

---

## Task 2: Drop `date` from the pooled tables

**Files:**
- Modify: `src/cellflow/contact_analysis/shape_tables.py:57-58` (`METADATA_COLUMNS`), `188-204` (`_position_metadata`)
- Test: `tests/contact_analysis/test_shape_tables.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/contact_analysis/test_shape_tables.py` (uses the Plan 1 `_stub_compute` helper):

```python
def test_pooled_table_has_no_date_column(tmp_path, monkeypatch):
    rec = _record(tmp_path, "a", date="2026-05-09")
    _stub_compute(monkeypatch, CellShapeQuantifier, {"a": _cell_shape_table([1, 2], [0])})
    df = build_table("cell_shape", [rec])
    assert "date" not in df.columns
    assert {"condition", "experiment_id", "position_id"} <= set(df.columns)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/contact_analysis/test_shape_tables.py::test_pooled_table_has_no_date_column -v`
Expected: FAIL (`date` is still stamped).

- [ ] **Step 3: Remove `date` from the metadata**

In `src/cellflow/contact_analysis/shape_tables.py`, change `METADATA_COLUMNS` (line 58):

```python
METADATA_COLUMNS = ("condition", "experiment_id", "position_id")
```

Rewrite `_position_metadata` (lines 188-204) so it neither stamps `date` in the base dict nor lets a stray `date` bag key re-enter:

```python
def _position_metadata(record: dict) -> dict[str, str]:
    """The per-position descriptor columns stamped onto every pooled row.

    The recognized identity axes (:data:`METADATA_COLUMNS`) come first, followed by
    any extra free-form columns carried in ``record["columns"]``. ``date`` is no
    longer a descriptor axis (removed with the catalog's date column) and is skipped
    even if a legacy bag still carries it."""
    meta = {
        "condition": str(record.get("condition", "")),
        "experiment_id": str(record.get("experiment_id", "")),
        "position_id": str(record.get("id", "")),
    }
    for key, value in (record.get("columns") or {}).items():
        if key not in meta and key != "date":
            meta[key] = str(value)
    return meta
```

- [ ] **Step 4: Update the other date-asserting tests**

In `tests/contact_analysis/test_shape_tables.py`:
- `test_experiment_id_broadcast_onto_pooled_rows`: delete the line `assert (df["date"] == "2026-05-09").all()`.
- `test_two_positions_pool_into_one_table`: change the column-subset assertion to `{"condition", "position_id", "frame", "cell_id"} <= set(df.columns)` (drop `"date"`).

- [ ] **Step 5: Run to verify green**

Run: `pytest tests/contact_analysis/test_shape_tables.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/contact_analysis/shape_tables.py tests/contact_analysis/test_shape_tables.py
git commit -m "refactor(aggregate): drop date descriptor from pooled tables"
```

---

## Task 3: Slim the catalog CSV columns

**Files:**
- Modify: `src/cellflow/contact_analysis/catalog.py:33-51` (`REQUIRED_CSV_COLUMNS`, `CSV_COLUMNS`), `201-226` (`save_catalog` row dict)
- Test: `tests/contact_analysis/test_catalog.py`

New columns: `position_path`, `contact_analysis_path`, `condition`, `experiment_id`, `id`, `cell_labels`, `nucleus_labels` (+ free-form extras). Dropped: `path`, `date`, `notes`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/contact_analysis/test_catalog.py` (match the file's existing import/fixture style):

```python
def test_saved_catalog_has_slim_columns(tmp_path):
    import csv
    from cellflow.contact_analysis.catalog import save_catalog

    pos = tmp_path / "ctrl" / "p1"
    pos.mkdir(parents=True)
    record = {
        "position_path": pos,
        "contact_analysis_path": pos / "4_contact_analysis" / "contact_analysis.h5",
        "cell_tracked_labels_path": pos / "cell_labels.tif",
        "nucleus_tracked_labels_path": pos / "nucleus_labels.tif",
        "condition": "ctrl",
        "experiment_id": "E1",
        "id": "p1",
    }
    out = tmp_path / "catalog.csv"
    save_catalog(out, [record])

    with out.open(newline="") as fh:
        header = next(csv.reader(fh))
    assert "date" not in header and "notes" not in header and "path" not in header
    assert header[:7] == [
        "position_path", "contact_analysis_path", "condition",
        "experiment_id", "id", "cell_labels", "nucleus_labels",
    ]


def test_catalog_roundtrip_slim(tmp_path):
    from cellflow.contact_analysis.catalog import save_catalog, load_catalog

    pos = tmp_path / "ctrl" / "p1"
    pos.mkdir(parents=True)
    record = {
        "position_path": pos,
        "contact_analysis_path": pos / "4_contact_analysis" / "contact_analysis.h5",
        "cell_tracked_labels_path": pos / "cell_labels.tif",
        "nucleus_tracked_labels_path": pos / "nucleus_labels.tif",
        "condition": "ctrl", "experiment_id": "E1", "id": "p1",
    }
    out = tmp_path / "catalog.csv"
    save_catalog(out, [record])
    loaded = load_catalog(out)

    assert len(loaded) == 1
    r = loaded[0]
    assert r["condition"] == "ctrl" and r["id"] == "p1" and r["experiment_id"] == "E1"
    assert r["contact_analysis_path"].name == "contact_analysis.h5"
    assert r["cell_tracked_labels_path"].name == "cell_labels.tif"
    assert r["nucleus_tracked_labels_path"].name == "nucleus_labels.tif"


def test_load_requires_position_path(tmp_path):
    import pytest
    from cellflow.contact_analysis.catalog import load_catalog

    out = tmp_path / "bad.csv"
    out.write_text("condition,id\nctrl,p1\n")
    with pytest.raises(ValueError, match="position_path"):
        load_catalog(out)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/contact_analysis/test_catalog.py -v -k "slim or roundtrip_slim or requires_position_path"`
Expected: FAIL (header still has `path`/`date`/`notes`; `position_path` not required).

- [ ] **Step 3: Slim the schema**

In `src/cellflow/contact_analysis/catalog.py`, change the column constants (lines 36-51):

```python
# Identity columns validated on load. ``position_path`` is the anchor every
# canonical artifact path is resolved against, so it is required.
REQUIRED_CSV_COLUMNS = ("position_path", "condition", "id")
# Full column order written on save: the absolute position folder, the canonical
# contact-analysis h5 (relative to it), identity, and the two committed label
# images (relative). The cheap derived quantities are pooled-only (no per-position
# file), so no ``date`` / ``notes`` / free-form ``path`` columns.
CSV_COLUMNS = (
    "position_path",
    "contact_analysis_path",
    "condition",
    "experiment_id",
    "id",
    "cell_labels",
    "nucleus_labels",
)
```

Rewrite the `save_catalog` row dict (lines 210-224) to the slim shape (drop `path`/`date`/`notes`; write the canonical h5 under `contact_analysis_path`):

```python
            row = {
                "position_path": str(position_path) if position_path is not None else "",
                "contact_analysis_path": _path_for_csv(
                    normalized["contact_analysis_path"], file_base
                ),
                "condition": normalized["condition"],
                "experiment_id": normalized["experiment_id"],
                "id": normalized["id"],
                "cell_labels": _optional_path_for_csv(
                    normalized.get("cell_tracked_labels_path"), file_base
                ),
                "nucleus_labels": _optional_path_for_csv(
                    normalized.get("nucleus_tracked_labels_path"), file_base
                ),
            }
            row.update({key: columns[key] for key in extras if key in columns})
            writer.writerow(row)
```

Leave `_normalize_catalog_record`, `_BAG_TO_CSV`, and `_columns_bag` unchanged: `_normalize` already reads `contact_analysis_path` (its fallback `normalized.get("path", normalized.get("contact_analysis_path", ""))` covers both old and new headers), `date` stays in `_BAG_TO_CSV` so a legacy bag `date` is suppressed from the free-form extras rather than re-emitted, and `notes` is already skipped by `_columns_bag`. A legacy fat CSV that still has `position_path` therefore loads and re-saves slim; its `path`/`date`/`notes` columns are simply not written back.

- [ ] **Step 4: Run to verify green + update broken catalog tests**

Run: `pytest tests/contact_analysis/test_catalog.py -q`
Expected: PASS. Fix any pre-existing test in `test_catalog.py` that asserts a `path` / `date` / `notes` column in the saved header or a required-column message that named `path`/`date` (update to `position_path`). Also check `tests/contact_analysis/test_author_config.py`, `tests/contact_analysis/test_reduce.py`, `tests/napari/test_contact_analysis_studio.py`, and `tests/napari/test_main_widget_config_project.py`: update any that assert the old header columns or write a hand-made CSV without `position_path`.

Run: `pytest tests/contact_analysis/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/contact_analysis/catalog.py tests
git commit -m "refactor(catalog): slim CSV to identity + canonical paths (drop path/date/notes)"
```

---

## Task 4: Full-app adapter stamps the committed label paths

**Files:**
- Modify: `src/cellflow/napari/main_widget.py:63-64` (constants), `591-606` (`_catalog_record_for_position`)
- Test: `tests/napari/test_main_widget_config_project.py`

The finalize/"commit" button writes `NucleusArtifactPaths.cell_labels` (`<pos>/cell_labels.tif`) and `.nucleus_labels` (`<pos>/nucleus_labels.tif`). The adapter must stamp those, not the pre-commit `3_cell` / `2_nucleus` working paths.

- [ ] **Step 1: Write the failing test**

Add to `tests/napari/test_main_widget_config_project.py` (reuse the file's existing widget/`_fake_viewer` construction; the test only needs the adapter method):

```python
def test_catalog_record_stamps_committed_label_paths(tmp_path):
    widget = _make_main_widget(tmp_path)  # existing helper in this file
    pos = tmp_path / "pos00"
    rec = widget._catalog_record_for_position(pos, {"condition": "ctrl", "position_id": "pos00"})

    assert rec["cell_tracked_labels_path"] == pos / "cell_labels.tif"
    assert rec["nucleus_tracked_labels_path"] == pos / "nucleus_labels.tif"
    assert rec["contact_analysis_path"] == pos / "4_contact_analysis" / "contact_analysis.h5"
```

(If the file constructs the widget differently, follow that; the assertion is on `_catalog_record_for_position` only.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/napari/test_main_widget_config_project.py::test_catalog_record_stamps_committed_label_paths -v`
Expected: FAIL (`cell_tracked_labels_path` is currently `pos / "3_cell/tracked_labels.tif"`).

- [ ] **Step 3: Point the constants at the committed outputs**

In `src/cellflow/napari/main_widget.py`, change the constants (lines 63-64):

```python
_CELL_LABELS_RELPATH = "cell_labels.tif"
_NUCLEUS_LABELS_RELPATH = "nucleus_labels.tif"
```

The `_catalog_record_for_position` body (lines 599-606) already builds
`pos / _CELL_LABELS_RELPATH` and `pos / _NUCLEUS_LABELS_RELPATH` and
`pos / CONTACT_ANALYSIS_RELPATH`, so it now yields the committed label paths and the
`4_contact_analysis` h5 automatically. Update the method docstring to say it stamps the
committed label outputs (`cell_labels.tif` / `nucleus_labels.tif`) and the canonical
contact-analysis h5. Optionally reference `NucleusArtifactPaths` in a comment so the
single source of the layout is clear:

```python
        # The committed label outputs (the finalize button writes
        # NucleusArtifactPaths.cell_labels / .nucleus_labels at the position root)
        # and the canonical 4_contact_analysis h5 — all fixed relative to the folder.
```

- [ ] **Step 4: Run to verify green**

Run: `pytest tests/napari/test_main_widget_config_project.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/main_widget.py tests/napari/test_main_widget_config_project.py
git commit -m "fix(main-widget): catalog adapter stamps committed label paths, not working paths"
```

---

## Task 5: End-to-end verification (no new code)

- [ ] **Step 1: Full suite**

Run: `pytest tests/contact_analysis/ tests/napari/ -q`
Expected: PASS.

- [ ] **Step 2: Lint**

Run: `ruff check src/cellflow/`
Expected: clean.

- [ ] **Step 3: Acceptance — a full-app-shaped catalog drives an aggregate run**

Write a throwaway check (in the scratchpad, not committed): build a slim catalog with two positions whose folders contain committed `cell_labels.tif` + a built `4_contact_analysis/contact_analysis.h5`, `save_catalog` it, `load_catalog` it, and confirm `run()` on an `author_config` over those records writes pooled tables and no per-position `cell_shape.csv`. Confirms Workstreams B + C compose with Plan 1.

Run: `python /path/to/scratchpad/acceptance_lean_aggregate.py`
Expected: prints the pooled table paths and "no per-position cheap files" OK.

No commit (verification only).

---

## Notes for the executor

- **No migration:** pre-existing on-disk `aggregate_quantification/` folders are orphaned by design. Do not write a migration or a fallback that reads the old folder name.
- **`_normalize_catalog_record` stays lenient:** it still computes `date` internally and reads `cell_labels` / `nucleus_labels` / `path` fallbacks, so a recent legacy CSV (one that carries `position_path`) still loads. Do not tighten it beyond adding `position_path` to `REQUIRED_CSV_COLUMNS`.
- The interactive studio (`BuildArea`) is out of scope (see Plan 1 and TODO.md). It keeps working because the writers are retained; its per-position "built" badges for pooled quantities are addressed in the deferred front-end-refocus work.
