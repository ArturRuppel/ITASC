# Curation Exclusion Table + Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore curation as a separate, git-versioned tidy table of exclusion flags (frame- and position-level, with a required reason) that is left-joined onto the aggregated measurement tables and filtered out at export time, so `.iris` bundles see only kept data while the on-disk measurement CSVs stay pure.

**Architecture:** A small pure module (`curation.py`) holds the table I/O and the join/filter logic, keyed on the natural keys the measurement tables already carry (`experiment_id`, `position_id`, `frame`). `RunConfig` gains an optional `curation` path (default `curation.csv` beside the config). The export step (`iris_export.export.export_dir`, forwarded by `pipeline.export` and `pipeline.run`) reads each tidy CSV, marks rows via `apply_curation`, drops the excluded ones via `filter_excluded`, and writes the filtered frame to `.iris` with provenance recording the curation file and the number of rows dropped. Nothing mutates the source CSVs.

**Tech Stack:** Python 3.11+, pandas, pytest, TOML (`tomllib`). All backend / headless — no Qt or napari in this plan.

---

## Background the engineer needs

- **This was here before and was removed.** Commit `95df159` deleted `curation.py` when the `.iris` export became a pure function. The old module is recoverable at rev `39d0df2` (`git show 39d0df2:src/cellflow/aggregate_quantification/curation.py`). The old version keyed the join on a deterministic row `id`. **We are NOT restoring that key** — we key on the natural keys instead (see below). Read the old file for the docstring style and the "filter, don't delete" philosophy, but expect the join logic to change.

- **The measurement tables already carry the keys we join on.** `shape_tables.aggregate` writes tidy CSVs whose metadata columns are `condition, experiment_id, date, position_id` (`shape_tables.py:60`) plus a `frame` column for frame-grained tables. `position_id` is the catalog record's `id`; `experiment_id` is the record's `experiment_id`. All are written to CSV and read back as strings / ints. We compare `experiment_id` and `position_id` as strings and `frame` as an integer.

- **The export path today.** `pipeline.run` → `pipeline.export(tables_dir, export_dir)` → `iris_export.export.export_dir(tables_dir, out_dir/"iris")`. `export_dir` loops `TABLES_TO_EXPORT = ("cell_shape", "nucleus_shape", "cell_dynamics", "nucleus_dynamics")`, and for each present `<stem>.csv` calls `export_table(csv_path, out_dir)` which reads the CSV and calls `export_table_frame(df, stem, out_dir, object_key=..., source=...)`. `export_table_frame` is the in-memory seam we hook: it takes an already-built frame, so filtering happens before it.

- **Provenance.** `export_table_frame(..., source=<dict>)` merges `source` into the bundle's `provenance.json` under `{"exporter": ..., **source}`. We pass curation info there.

- **`config.py` resolution pattern.** `load_config` resolves each path with `_resolve(base, raw)` (relative to the config file's directory). The `curation` key follows `catalog` exactly, but with a default filename when the key is absent.

- **Tests live in** `tests/aggregate_quantification/`. The curation unit tests go in a new `test_curation.py`; config tests extend `test_config.py`; export/pipeline wiring tests extend `test_pipeline.py` and `test_iris_export.py`. Run a single test with `uv run pytest <path>::<name> -v`. Run the whole suite with `uv run pytest tests/aggregate_quantification -q`.

---

## File Structure

- **Create** `src/cellflow/aggregate_quantification/curation.py` — pure table I/O + join/filter. No pandas-external deps, no Qt. Public API: `CURATION_COLUMNS`, `read_curation`, `apply_curation`, `filter_excluded`.
- **Create** `tests/aggregate_quantification/test_curation.py` — unit tests for the above.
- **Modify** `src/cellflow/aggregate_quantification/config.py` — add `curation: Path` field to `RunConfig`; resolve it in `load_config` (default `curation.csv` beside the config).
- **Modify** `tests/aggregate_quantification/test_config.py` — cover the new key + default.
- **Modify** `src/cellflow/aggregate_quantification/iris_export/export.py` — `export_dir` gains `curation` + `curation_path` keyword args; filters per table when curation present; records provenance.
- **Modify** `tests/aggregate_quantification/test_iris_export.py` — cover `export_dir` filtering.
- **Modify** `src/cellflow/aggregate_quantification/pipeline.py` — `export` forwards `curation`/`curation_path`; `run` loads the curation table via `read_curation` and threads it in.
- **Modify** `tests/aggregate_quantification/test_pipeline.py` — end-to-end run with curation that drops a frame and a whole position.

---

## Task 1: The `curation.py` module — read, mark, filter

**Files:**
- Create: `src/cellflow/aggregate_quantification/curation.py`
- Test: `tests/aggregate_quantification/test_curation.py`

The schema (one row per exclusion): `experiment_id, position_id, frame, excluded, exclusion_reason`. A row with `frame` empty/NA = the whole position. `apply_curation` left-joins by either `(experiment_id, position_id, frame)` (frame present) or `(experiment_id, position_id)` (frame NA), marking matched measurement rows `excluded=True` and copying the `exclusion_reason`; unmatched rows default to kept. `filter_excluded` drops the marked rows and removes the two marker columns so the kept frame matches the source schema.

- [ ] **Step 1: Write the failing tests**

Create `tests/aggregate_quantification/test_curation.py`:

```python
"""The curation artifact: human QC exclusions joined onto the measurement tables.

A separate, git-versioned tidy table — ``experiment_id, position_id, frame,
excluded, exclusion_reason`` — authored by hand and kept apart from the disposable
measurement tables. At export it is left-joined by the *natural* keys the tables
already carry: a frame-level exclusion matches ``(experiment_id, position_id,
frame)``; a whole-position exclusion is the ``frame``-is-NA row and matches
``(experiment_id, position_id)``. Rows with no entry default to kept; filter,
don't delete.
"""
from __future__ import annotations

import pandas as pd

from cellflow.aggregate_quantification.curation import (
    apply_curation,
    filter_excluded,
    read_curation,
)


def _table() -> pd.DataFrame:
    # Two positions of one experiment, three frames each.
    rows = []
    for position_id in ("p1", "p2"):
        for frame in (0, 1, 2):
            rows.append({
                "experiment_id": "EXP1",
                "position_id": position_id,
                "frame": frame,
                "cell_shape.area_um2": 10.0 * frame + 1.0,
            })
    return pd.DataFrame(rows)


def test_apply_curation_marks_frame_level_exclusion():
    cur = pd.DataFrame({
        "experiment_id": ["EXP1"],
        "position_id": ["p1"],
        "frame": [1],
        "excluded": [True],
        "exclusion_reason": ["out of focus"],
    })

    out = apply_curation(_table(), cur)

    # Only (EXP1, p1, frame 1) is excluded.
    marked = out[out["excluded"]]
    assert list(zip(marked["position_id"], marked["frame"])) == [("p1", 1)]
    assert list(out[out["excluded"]]["exclusion_reason"]) == ["out of focus"]
    # Everything else kept, no reason.
    assert (out.loc[~out["excluded"], "exclusion_reason"] == "").all()


def test_apply_curation_position_level_excludes_every_frame():
    # frame NA => whole position p2.
    cur = pd.DataFrame({
        "experiment_id": ["EXP1"],
        "position_id": ["p2"],
        "frame": [pd.NA],
        "excluded": [True],
        "exclusion_reason": ["debris"],
    })

    out = apply_curation(_table(), cur)

    excluded = out[out["excluded"]]
    assert set(excluded["position_id"]) == {"p2"}
    assert sorted(excluded["frame"]) == [0, 1, 2]  # all three frames
    assert (excluded["exclusion_reason"] == "debris").all()


def test_apply_curation_none_keeps_everything():
    out = apply_curation(_table(), None)
    assert not out["excluded"].any()
    assert (out["exclusion_reason"] == "").all()


def test_apply_curation_does_not_mutate_input():
    table = _table()
    apply_curation(table, pd.DataFrame({
        "experiment_id": ["EXP1"], "position_id": ["p1"],
        "frame": [0], "excluded": [True], "exclusion_reason": ["x"],
    }))
    assert "excluded" not in table.columns


def test_apply_curation_keys_compared_as_strings():
    # CSV round-trips ids as strings; a numeric-looking id must still match.
    # Single position so frame 0 is unique (matches exactly one row).
    table = pd.DataFrame({
        "experiment_id": ["EXP1", "EXP1", "EXP1"],
        "position_id": ["10", "10", "10"],
        "frame": [0, 1, 2],
        "cell_shape.area_um2": [1.0, 2.0, 3.0],
    })
    cur = pd.DataFrame({
        "experiment_id": ["EXP1"], "position_id": [10],  # int in curation
        "frame": [0], "excluded": [True], "exclusion_reason": ["x"],
    })
    out = apply_curation(table, cur)
    assert out["excluded"].sum() == 1


def test_filter_excluded_drops_marked_rows_and_marker_columns():
    table = _table()
    cur = pd.DataFrame({
        "experiment_id": ["EXP1"], "position_id": ["p1"],
        "frame": [pd.NA], "excluded": [True], "exclusion_reason": ["debris"],
    })
    marked = apply_curation(table, cur)

    kept, dropped = filter_excluded(marked)

    assert dropped == 3  # all of p1
    assert set(kept["position_id"]) == {"p2"}
    assert "excluded" not in kept.columns
    assert "exclusion_reason" not in kept.columns
    # Index is reset so the kept frame is clean.
    assert list(kept.index) == list(range(len(kept)))


def test_filter_excluded_no_marker_column_is_noop():
    table = _table()
    kept, dropped = filter_excluded(table)
    assert dropped == 0
    assert len(kept) == len(table)


def test_read_curation_missing_or_none_is_none(tmp_path):
    assert read_curation(None) is None
    assert read_curation(tmp_path / "nope.csv") is None


def test_read_curation_reads_csv(tmp_path):
    path = tmp_path / "curation.csv"
    pd.DataFrame({
        "experiment_id": ["EXP1"], "position_id": ["p1"],
        "frame": [pd.NA], "excluded": [True], "exclusion_reason": ["debris"],
    }).to_csv(path, index=False)

    cur = read_curation(path)

    assert cur is not None
    assert list(cur["position_id"]) == ["p1"]
    # The empty frame round-trips as NA, not the string "".
    assert cur["frame"].isna().all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/aggregate_quantification/test_curation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'cellflow.aggregate_quantification.curation'`.

- [ ] **Step 3: Write the implementation**

Create `src/cellflow/aggregate_quantification/curation.py`:

```python
"""The curation artifact — hand QC exclusions joined onto the measurement tables.

A separate, git-versioned tidy table kept apart from the disposable measurement
tables (the dividing line is who made the decisions in it). Schema, one row per
exclusion::

    experiment_id, position_id, frame, excluded, exclusion_reason

``frame`` empty/NA means *the whole position* (every frame). At export the table
is **left-joined** onto a measurement table by the natural keys the table already
carries — a frame-level exclusion matches ``(experiment_id, position_id, frame)``;
a position-level exclusion (``frame`` NA) matches ``(experiment_id, position_id)``
— marking matched rows ``excluded = True`` and copying the reason. Rows with no
entry default to kept; filter, don't delete. The measurement source is never
mutated.

This restores the curation that commit 95df159 removed, re-keyed on the natural
keys (the old version, at rev 39d0df2, joined on a deterministic row ``id``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["CURATION_COLUMNS", "read_curation", "apply_curation", "filter_excluded"]

#: The columns a curation CSV carries.
CURATION_COLUMNS = (
    "experiment_id",
    "position_id",
    "frame",
    "excluded",
    "exclusion_reason",
)


def read_curation(path: Path | str | None) -> pd.DataFrame | None:
    """Read the curation CSV at *path*, or ``None`` when *path* is unset/absent.

    A missing file is not an error: an uncurated series simply keeps every row.
    """
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    return pd.read_csv(path)


def apply_curation(
    table: pd.DataFrame, curation: pd.DataFrame | None
) -> pd.DataFrame:
    """Return *table* with ``excluded`` / ``exclusion_reason`` columns marked.

    A measurement row is marked excluded iff a curation entry matches it by either
    ``(experiment_id, position_id, frame)`` (frame-level) or
    ``(experiment_id, position_id)`` with the curation ``frame`` NA
    (position-level). Keys are compared as strings (CSV round-trips ids as
    strings); ``frame`` as an integer. Unmatched rows default to kept with an empty
    reason. The input frame is not mutated.
    """
    out = table.copy()
    out["excluded"] = False
    out["exclusion_reason"] = ""
    if curation is None or len(curation) == 0:
        return out

    exp = out["experiment_id"].astype(str)
    pos = out["position_id"].astype(str)

    for _, entry in curation.iterrows():
        if not bool(entry.get("excluded", True)):
            continue  # a future un-exclude override; ignored for now
        key = (exp == str(entry["experiment_id"])) & (pos == str(entry["position_id"]))
        frame = entry.get("frame")
        if pd.notna(frame):
            if "frame" not in out.columns:
                continue  # a frame-level entry cannot match a frameless table
            key &= out["frame"].astype("int64") == int(frame)
        out.loc[key, "excluded"] = True
        out.loc[key, "exclusion_reason"] = str(entry.get("exclusion_reason", ""))

    return out


def filter_excluded(marked: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop the rows :func:`apply_curation` marked excluded.

    Returns ``(kept, n_dropped)`` where *kept* has the two marker columns removed
    (so it matches the source schema) and a reset index. A frame with no
    ``excluded`` column is returned unchanged with ``n_dropped == 0``.
    """
    if "excluded" not in marked.columns:
        return marked, 0
    excluded = marked["excluded"].astype(bool)
    n_dropped = int(excluded.sum())
    drop_cols = [c for c in ("excluded", "exclusion_reason") if c in marked.columns]
    kept = marked.loc[~excluded].drop(columns=drop_cols).reset_index(drop=True)
    return kept, n_dropped
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/aggregate_quantification/test_curation.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/aggregate_quantification/curation.py tests/aggregate_quantification/test_curation.py
git commit -m "feat(aggregate): restore curation table join/filter on natural keys

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N3vuJPVYYuYWh6EQLh8vrv"
```

---

## Task 2: `RunConfig` gains the optional `curation` path

**Files:**
- Modify: `src/cellflow/aggregate_quantification/config.py`
- Test: `tests/aggregate_quantification/test_config.py`

`RunConfig` gains `curation: Path`. It resolves like `catalog` (relative to the config dir), but defaults to `curation.csv` beside the config when the key is absent. The default path need not exist — `read_curation` returns `None` for a missing file, so an uncurated project just keeps every row.

- [ ] **Step 1: Write the failing tests**

Append to `tests/aggregate_quantification/test_config.py`:

```python
def test_curation_defaults_beside_config(tmp_path):
    """Absent ``curation`` key defaults to ``curation.csv`` beside the config."""
    cfg_path = _write(tmp_path, 'catalog = "catalog.csv"\n')

    cfg = load_config(cfg_path)

    assert cfg.curation == (tmp_path / "curation.csv").resolve()


def test_curation_explicit_path_resolved(tmp_path):
    cfg_path = _write(
        tmp_path,
        'catalog = "catalog.csv"\ncuration = "qc/exclusions.csv"\n',
    )

    cfg = load_config(cfg_path)

    assert cfg.curation == (tmp_path / "qc" / "exclusions.csv").resolve()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/aggregate_quantification/test_config.py -k curation -v`
Expected: FAIL — `AttributeError: 'RunConfig' object has no attribute 'curation'`.

- [ ] **Step 3: Add the field and resolution**

In `src/cellflow/aggregate_quantification/config.py`:

Add the default constant beside `_DEFAULT_EXPORT_DIR` (after line 30):

```python
#: Default for the optional curation-table key, relative to the config dir.
_DEFAULT_CURATION = "curation.csv"
```

Add the field to `RunConfig` (after the `export_dir: Path` line, line 49):

```python
    catalog: Path
    export_dir: Path
    curation: Path = field(default=Path(_DEFAULT_CURATION))
    quantities: tuple[str, ...] = ()
```

> Note: `field(default=...)` keeps `RunConfig` constructible without `curation`; `load_config` always passes a resolved absolute path, so the dataclass default is only a placeholder for direct construction in tests.

Thread it through `load_config` — change the returned `RunConfig(...)` (lines 82-89) to add the `curation` argument:

```python
    return RunConfig(
        catalog=_resolve(base, data["catalog"]),
        export_dir=_resolve(base, data.get("export_dir", _DEFAULT_EXPORT_DIR)),
        curation=_resolve(base, data.get("curation", _DEFAULT_CURATION)),
        quantities=quantities,
        params=dict(data.get("params", {})),
        render_plots=render_plots,
        plot_formats=plot_formats,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/aggregate_quantification/test_config.py -q`
Expected: PASS (all config tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/aggregate_quantification/config.py tests/aggregate_quantification/test_config.py
git commit -m "feat(aggregate): add optional curation path to RunConfig

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N3vuJPVYYuYWh6EQLh8vrv"
```

---

## Task 3: Filter at export — `export_dir` applies curation

**Files:**
- Modify: `src/cellflow/aggregate_quantification/iris_export/export.py`
- Test: `tests/aggregate_quantification/test_iris_export.py`

`export_dir` gains keyword args `curation: pd.DataFrame | None = None` and `curation_path: str | None = None`. When `curation` is given, each table is read into a frame, marked (`apply_curation`), filtered (`filter_excluded`), and written via `export_table_frame` with provenance recording the curation file and rows dropped. When `curation` is `None`, the existing `export_table` path is used unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/aggregate_quantification/test_iris_export.py` (it already imports `zipfile`, `io`, `pd`, `export_dir`; confirm `export_dir` is in the `from ...iris_export import (...)` block — it is used at line 248):

```python
def test_export_dir_filters_curated_rows(tmp_path):
    # A small cell_shape table: one experiment, two positions, two frames.
    rows = []
    for position_id in ("p1", "p2"):
        for frame in (0, 1):
            for cell_id in (1, 2):
                rows.append({
                    "condition": "ctrl",
                    "experiment_id": "EXP1",
                    "date": "d1",
                    "position_id": position_id,
                    "frame": frame,
                    "cell_id": cell_id,
                    "cell_shape.area_um2": 100.0,
                })
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "cell_shape.csv", index=False)

    curation = pd.DataFrame({
        "experiment_id": ["EXP1", "EXP1"],
        "position_id": ["p1", "p2"],
        "frame": [1, pd.NA],          # p1 frame 1, and all of p2
        "excluded": [True, True],
        "exclusion_reason": ["blur", "debris"],
    })

    written = export_dir(
        tmp_path, curation=curation, curation_path="qc/exclusions.csv"
    )

    assert [p.name for p in written] == ["cell_shape.iris"]
    # Read the parquet back out of the bundle and check the kept rows.
    with zipfile.ZipFile(written[0]) as zf:
        back = pd.read_parquet(io.BytesIO(zf.read("data/table.parquet")))
        provenance = json.loads(zf.read("provenance.json"))
    # Only p1 frame 0 survives (2 cells). p1 frame 1 dropped, all of p2 dropped.
    assert set(zip(back["position_id"], back["frame"])) == {("p1", 0)}
    assert len(back) == 2
    # Marker columns are not exported.
    assert "excluded" not in back.columns
    assert "exclusion_reason" not in back.columns
    # Provenance records the filter.
    assert provenance["curation"]["rows_dropped"] == 6
    assert provenance["curation"]["file"] == "qc/exclusions.csv"


def test_export_dir_without_curation_keeps_all_rows(tmp_path):
    _cells_table().to_csv(tmp_path / "cell_shape.csv", index=False)

    written = export_dir(tmp_path)  # no curation

    with zipfile.ZipFile(written[0]) as zf:
        back = pd.read_parquet(io.BytesIO(zf.read("data/table.parquet")))
        provenance = json.loads(zf.read("provenance.json"))
    assert len(back) == len(_cells_table())
    assert "curation" not in provenance
```

> The `cell_shape` table's object key (`cell_id`) is resolved automatically by `export_table_frame` via `_object_key_for("cell_shape")`, so no `object_key` need be passed.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/aggregate_quantification/test_iris_export.py -k "export_dir_filters or export_dir_without" -v`
Expected: FAIL — `export_dir() got an unexpected keyword argument 'curation'`.

- [ ] **Step 3: Implement the filtering in `export_dir`**

In `src/cellflow/aggregate_quantification/iris_export/export.py`:

Add the import near the top (after the `from .schema import infer_schema` line, line 19):

```python
from ..curation import apply_curation, filter_excluded
```

Replace `export_dir` (lines 108-121) with:

```python
def export_dir(
    data_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    curation: "pd.DataFrame | None" = None,
    curation_path: str | None = None,
) -> list[Path]:
    """Export every known table found in *data_dir*.

    Defaults the output to ``<data_dir>/iris/``. Returns the written paths in
    table order; tables not present are skipped.

    When *curation* is given (the parsed exclusion table), each table is read into
    a frame, marked by :func:`~cellflow.aggregate_quantification.curation.apply_curation`,
    and its excluded rows dropped by
    :func:`~cellflow.aggregate_quantification.curation.filter_excluded` before the
    ``.iris`` is written — so the bundle sees only kept data. The on-disk tidy
    CSVs are never touched; the bundle's provenance records *curation_path* and how
    many rows were dropped. With no *curation*, the unfiltered table is exported.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir is not None else data_dir / "iris"
    written: list[Path] = []
    for stem in TABLES_TO_EXPORT:
        csv_path = data_dir / f"{stem}.csv"
        if not csv_path.is_file():
            continue
        if curation is None:
            written.append(export_table(csv_path, out_dir))
            continue
        kept, n_dropped = filter_excluded(apply_curation(pd.read_csv(csv_path), curation))
        source = {
            "source_csv": str(csv_path.resolve()),
            "curation": {"file": curation_path, "rows_dropped": n_dropped},
        }
        written.append(export_table_frame(kept, stem, out_dir, source=source))
    return written
```

> `pd` is already imported at module top (line 15). The string annotation `"pd.DataFrame | None"` avoids any import-order concern; a plain `pd.DataFrame | None` is also fine since `pd` is imported.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/aggregate_quantification/test_iris_export.py -k "export_dir" -v`
Expected: PASS (the two new tests plus the existing `test_export_dir_discovers_known_tables`).

- [ ] **Step 5: Run the full iris-export test file to catch regressions**

Run: `uv run pytest tests/aggregate_quantification/test_iris_export.py -q`
Expected: PASS (live-engine integration tests may be skipped — that is fine).

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/aggregate_quantification/iris_export/export.py tests/aggregate_quantification/test_iris_export.py
git commit -m "feat(aggregate): filter curated rows at .iris export with provenance

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N3vuJPVYYuYWh6EQLh8vrv"
```

---

## Task 4: Wire curation through `pipeline.export` and `pipeline.run`

**Files:**
- Modify: `src/cellflow/aggregate_quantification/pipeline.py`
- Test: `tests/aggregate_quantification/test_pipeline.py`

`pipeline.export` gains `curation`/`curation_path` keyword args forwarded to `_export_iris` (`iris_export.export.export_dir`). `pipeline.run` loads the curation table with `read_curation(cfg.curation)` and threads it (plus the path string for provenance) into `export`. When the curation file is absent, `read_curation` returns `None` and behaviour is unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/aggregate_quantification/test_pipeline.py` (it already imports `np`, `pd`, `tifffile`, `pipeline`, and uses `save_catalog`; the `import io, json, zipfile` may be needed — add them at the top of the test file if absent):

```python
def test_run_applies_curation_filter(tmp_path):
    import io
    import json
    import zipfile

    from cellflow.aggregate_quantification.catalog import save_catalog

    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    study = tmp_path / "study"
    recs = []
    for pid in ("a", "b"):
        pdir = study / pid
        pdir.mkdir(parents=True)
        cells = pdir / "cells.tif"
        # three frames so a frame-level exclusion is visible
        tifffile.imwrite(cells, np.stack([frame, frame, frame]))
        recs.append({
            "id": pid,
            "condition": "ctrl",
            "date": "d1",
            "experiment_id": f"EXP-{pid}",
            "position_path": pdir,
            "cell_tracked_labels_path": cells,
        })
    save_catalog(tmp_path / "catalog.csv", recs)

    # Exclude position b entirely, and frame 0 of position a.
    pd.DataFrame({
        "experiment_id": ["EXP-b", "EXP-a"],
        "position_id": ["b", "a"],
        "frame": [pd.NA, 0],
        "excluded": [True, True],
        "exclusion_reason": ["debris", "blur"],
    }).to_csv(tmp_path / "curation.csv", index=False)

    config = tmp_path / "config.toml"
    config.write_text(
        'catalog = "catalog.csv"\n'
        'quantities = ["cell_shape"]\n'
        'export_dir = "export"\n'
        "[params]\npixel_size_um = 0.25\n"
    )

    written = pipeline.run(config)

    # The on-disk measurement CSV stays pure: both positions, all frames.
    from cellflow.aggregate_quantification.quantifier import OUTPUT_SUBDIR

    measured = pd.read_csv(study / OUTPUT_SUBDIR / "cell_shape.csv")
    assert set(measured["position_id"]) == {"a", "b"}

    # The .iris bundle is filtered: no position b, and no frame 0 of a.
    with zipfile.ZipFile(written[0]) as zf:
        back = pd.read_parquet(io.BytesIO(zf.read("data/table.parquet")))
        provenance = json.loads(zf.read("provenance.json"))
    assert set(back["position_id"]) == {"a"}
    assert 0 not in set(back["frame"])
    assert provenance["curation"]["rows_dropped"] > 0


def test_run_without_curation_file_exports_everything(tmp_path):
    import io
    import zipfile

    from cellflow.aggregate_quantification.catalog import save_catalog

    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    study = tmp_path / "study"
    pdir = study / "a"
    pdir.mkdir(parents=True)
    cells = pdir / "cells.tif"
    tifffile.imwrite(cells, np.stack([frame, frame]))
    save_catalog(tmp_path / "catalog.csv", [{
        "id": "a", "condition": "ctrl", "date": "d1", "experiment_id": "EXP-a",
        "position_path": pdir, "cell_tracked_labels_path": cells,
    }])

    config = tmp_path / "config.toml"
    config.write_text(
        'catalog = "catalog.csv"\nquantities = ["cell_shape"]\n'
        'export_dir = "export"\n[params]\npixel_size_um = 0.25\n'
    )

    # No curation.csv written → read_curation returns None → unfiltered.
    written = pipeline.run(config)

    with zipfile.ZipFile(written[0]) as zf:
        back = pd.read_parquet(io.BytesIO(zf.read("data/table.parquet")))
    assert set(back["position_id"]) == {"a"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/aggregate_quantification/test_pipeline.py -k "curation" -v`
Expected: FAIL — `test_run_applies_curation_filter` fails because `run` does not yet load/apply curation (the bundle still contains position b).

- [ ] **Step 3: Forward curation through `pipeline.export`**

In `src/cellflow/aggregate_quantification/pipeline.py`, replace the `export` function signature + body's final line (lines 246-266). Change the signature and the `return`:

```python
def export(
    tables_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    curation: "object | None" = None,
    curation_path: str | None = None,
) -> list[Path]:
    """Write ``.iris`` bundles from the aggregated tables.

    *tables_dir* is the directory holding the aggregated tidy CSVs (what
    :func:`aggregate` wrote into — the ``aggregate_quantification`` folder). The
    export is **Iris-only**: for each table selected for premade SuperPlots
    (``iris_export.TABLES_TO_EXPORT``) found there, one ``.iris`` document is
    written under ``<out_dir>/iris/``. *out_dir* defaults to *tables_dir*.

    When *curation* (a parsed exclusion table) is given, its excluded frames /
    positions are filtered out of each table before the ``.iris`` is written, and
    *curation_path* is recorded in the bundle provenance. The on-disk tidy CSVs in
    *tables_dir* stay pure (all rows); only the bundle sees the filtered view, so
    each ``.iris`` is a pure function of ``(table, curation)``.

    Returns the written ``.iris`` paths.
    """
    tables_dir = Path(tables_dir)
    out_dir = Path(out_dir) if out_dir is not None else tables_dir
    return _export_iris(
        tables_dir,
        out_dir=out_dir / "iris",
        curation=curation,
        curation_path=curation_path,
    )
```

> The `"object | None"` annotation avoids importing pandas into `pipeline.py` just for a type hint; the value is passed straight through to `export_dir`.

- [ ] **Step 4: Load + thread curation in `run`**

In `run` (lines 269-315), add the import and load the table, then pass it to `export`. Add to the imports at the top of the module (after `from .config import RunConfig, load_config`, line 31):

```python
from .curation import read_curation
```

Change the export call inside `run` (line 303) from:

```python
    written = export(tables_dir, cfg.export_dir)
```

to:

```python
    curation = read_curation(cfg.curation)
    written = export(
        tables_dir,
        cfg.export_dir,
        curation=curation,
        curation_path=str(cfg.curation) if curation is not None else None,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/aggregate_quantification/test_pipeline.py -k "curation" -v`
Expected: PASS (both new tests).

- [ ] **Step 6: Run the whole aggregate suite for regressions**

Run: `uv run pytest tests/aggregate_quantification -q`
Expected: PASS (no regressions; live-engine integration tests may be skipped).

- [ ] **Step 7: Commit**

```bash
git add src/cellflow/aggregate_quantification/pipeline.py tests/aggregate_quantification/test_pipeline.py
git commit -m "feat(aggregate): thread curation table through run() into .iris export

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N3vuJPVYYuYWh6EQLh8vrv"
```

---

## Final verification

- [ ] **Run the full aggregate test suite once more**

Run: `uv run pytest tests/aggregate_quantification -q`
Expected: all pass (skips allowed for the opt-in live Iris-engine tests).

- [ ] **Sanity-check the public surface**

Run: `uv run python -c "from cellflow.aggregate_quantification.curation import read_curation, apply_curation, filter_excluded, CURATION_COLUMNS; from cellflow.aggregate_quantification.config import RunConfig; print(RunConfig.__dataclass_fields__.keys())"`
Expected: prints the dataclass fields including `curation`; no import error.

---

## Notes on scope (deferred, per the spec's Open section — do NOT implement here)

- **Position-level representation** stays `frame`-is-NA (spec Decision 3); no explicit `scope` column.
- **Frame ranges** are many individual rows authored by the *tool* (next plan); the table has no range syntax.
- **Un-exclude / override rows** (`excluded = False`): `apply_curation` already skips entries whose `excluded` is falsy, but no precedence rule between overlapping rows is defined — leave as is.
- **The curation *tool*** (napari widget that authors this CSV) is a separate spec/plan — not in scope here.
```
