# NLS Classification CLI Step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make headless NLS subpopulation classification an optional, config-gated pipeline step (`classify`) that runs before `aggregate`, and remove the interactive napari NLS UI.

**Architecture:** The classification *engine* (`contacts/nls_classification.py`) is already fully headless and is left untouched. We add a thin orchestration step `classify(catalog, *, config)` to `pipeline.py` that, per position, resolves the marker image + nucleus labels, picks a threshold by the configured method, and calls the existing `classify_position_nls_to_csv` to write the per-position sidecar CSV. The downstream join (`shape_tables._join_class`) already picks up the sidecar automatically at aggregate time, so nothing else changes downstream. A new `[nls]` table in the run-config gates the step. The napari NLS plugin (a hand-operated trigger for the same engine) is deleted.

**Tech Stack:** Python 3.11, pandas, numpy, tifffile, pytest, dataclasses, tomllib.

---

## Background for the implementer (read once)

You have **zero prior context** on this codebase. Key facts you need:

- **The engine is done.** `src/cellflow/aggregate_quantification/contacts/nls_classification.py` exposes:
  - `measure_track_nls_intensity(nls_zavg, nucleus_labels) -> dict[int, TrackNLSMeasurement]` (each measurement has `.intensity`).
  - `auto_threshold(intensities: Mapping[int, float]) -> float` (two-cluster, falls back to Otsu).
  - `split_tracks_otsu(intensities) -> (threshold, assignments)` and `split_tracks_two_clusters(intensities) -> (threshold, assignments)`.
  - `classify_position_nls_to_csv(csv_path, nls_zavg_path, nucleus_labels_path, *, threshold=None, positive_label="positive", negative_label="negative") -> NLSClassificationSummary`. With `threshold=None` it uses `auto_threshold`; pass a float to pin it. It reads only the two images and writes a two-column `id,label` sidecar CSV.
  - `nls_classification_csv_path(position_path) -> Path` — the sidecar location (`<position_path>/aggregate_quantification/nls_classification.csv`).
  - `_read_image_stack(path) -> np.ndarray` — reads a TIFF into a `(T, Y, X)` array (private but used by the napari plugin already).
  - `NLSClassificationError` — raised on degenerate inputs.

  **Do not modify this module.**

- **The downstream join already works.** `shape_tables._join_class` left-joins `{cell_id: class_label}` from the sidecar onto every `cell_id`-keyed table *whenever the sidecar exists*. So once `classify` writes sidecars, aggregation picks them up with no further wiring. You do **not** touch `shape_tables.py`.

- **Catalogue records** are plain dicts. Relevant keys: `position_path` (the position folder), `nucleus_tracked_labels_path` (the tracked-nucleus label TIFF), `id` (the position id), `experiment_id`. See `records.py` / `shape_tables._position_metadata`.

- **`RunConfig`** (`config.py`) is a frozen dataclass parsed from TOML by `load_config`. Relative paths resolve against the config file's directory via `_resolve`.

- **The pipeline** (`pipeline.py`) `run(config_path)` does: `load_config` → `load_catalog` → `build_quantities` → `aggregate` → `export`. We insert `classify` **between build and aggregate**.

- **Always run tests with `--frozen`** — the lockfile is intentionally stale and `uv sync` (implicit on a bare `uv run`) is unsatisfiable. Every test command below uses `uv run --frozen pytest`.

- **The napari NLS plugin self-registers** via `__init_subclass__` + a `pkgutil` walk of the `plugins/` package (`plugins/__init__.py`). There is **no manifest entry or explicit import to edit** — deleting the module file removes it from discovery cleanly.

---

## File Structure

- **Modify** `src/cellflow/aggregate_quantification/config.py` — add `NlsConfig` dataclass; add `nls` field to `RunConfig`; parse `[nls]` in `load_config` with method validation.
- **Modify** `src/cellflow/aggregate_quantification/pipeline.py` — add `classify()` step + private helpers; export `classify` in `__all__`; call it from `run()` before `aggregate`.
- **Create** `tests/aggregate_quantification/test_classify.py` — unit tests for `classify` and helpers using synthetic TIFFs.
- **Modify** `tests/aggregate_quantification/test_config.py` — tests for `[nls]` parsing.
- **Modify** `tests/aggregate_quantification/test_pipeline.py` — integration test: a `run()` with `[nls]` writes sidecars and the aggregated `cell_id`-keyed table carries `class_label`.
- **Delete** `src/cellflow/napari/aggregate_quantification/plugins/nls_classification.py` — the interactive UI.
- **Delete** `tests/napari/test_nls_classification_plugin.py` — its test.

---

## Task 1: `NlsConfig` + `[nls]` config parsing

**Files:**
- Modify: `src/cellflow/aggregate_quantification/config.py`
- Test: `tests/aggregate_quantification/test_config.py`

The `[nls]` table is optional. When absent, `RunConfig.nls` is `None` (step skipped). When present, it parses into a frozen `NlsConfig`. `method` is validated against `{"auto", "otsu", "two_cluster", "fixed"}` so a typo fails loudly. `image` is kept as a **string** (it is resolved per-position at classify time, not against the config dir — one relative entry must resolve across every position).

- [ ] **Step 1: Write the failing tests**

Add to `tests/aggregate_quantification/test_config.py` (match the file's existing imports/style; it already imports `load_config` and writes temp TOML files):

```python
def test_load_config_without_nls_table_has_none(tmp_path):
    (tmp_path / "catalog.csv").write_text("id\n")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('catalog = "catalog.csv"\n')
    cfg = load_config(cfg_path)
    assert cfg.nls is None


def test_load_config_parses_nls_table(tmp_path):
    (tmp_path / "catalog.csv").write_text("id\n")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        'catalog = "catalog.csv"\n'
        "[nls]\n"
        "enabled = true\n"
        'image = "0_input/NLS_zavg.tif"\n'
        'method = "fixed"\n'
        "threshold = 12.5\n"
    )
    cfg = load_config(cfg_path)
    assert cfg.nls is not None
    assert cfg.nls.enabled is True
    # image stays a (relative) string — resolved per-position, not against the config dir.
    assert cfg.nls.image == "0_input/NLS_zavg.tif"
    assert cfg.nls.method == "fixed"
    assert cfg.nls.threshold == 12.5


def test_load_config_nls_defaults(tmp_path):
    (tmp_path / "catalog.csv").write_text("id\n")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        'catalog = "catalog.csv"\n'
        "[nls]\n"
        "enabled = true\n"
    )
    cfg = load_config(cfg_path)
    assert cfg.nls.method == "auto"
    assert cfg.nls.image == "0_input/NLS_zavg.tif"
    assert cfg.nls.threshold == 0.0


def test_load_config_rejects_unknown_nls_method(tmp_path):
    import pytest

    (tmp_path / "catalog.csv").write_text("id\n")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        'catalog = "catalog.csv"\n'
        "[nls]\n"
        "enabled = true\n"
        'method = "bogus"\n'
    )
    with pytest.raises(ValueError, match="bogus"):
        load_config(cfg_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --frozen pytest tests/aggregate_quantification/test_config.py -k nls -v`
Expected: FAIL (`AttributeError: ... has no attribute 'nls'` / parse errors).

- [ ] **Step 3: Implement `NlsConfig`, the `RunConfig` field, and parsing**

In `config.py`, add the constant near the other defaults (after `_DEFAULT_CURATION`):

```python
#: Default per-position-relative marker image for the optional [nls] step.
_DEFAULT_NLS_IMAGE = "0_input/NLS_zavg.tif"

#: The NLS thresholding methods the classify step understands.
_NLS_METHODS = ("auto", "otsu", "two_cluster", "fixed")
```

Add the `NlsConfig` dataclass immediately above `RunConfig`:

```python
@dataclass(frozen=True)
class NlsConfig:
    """Parsed ``[nls]`` table — the optional NLS classification step's knobs.

    *image* is the marker image **relative to each position directory** (e.g.
    ``0_input/NLS_zavg.tif``) so one entry resolves across a batch; an absolute
    path is used verbatim. *method* picks the thresholding: ``auto`` (per-position
    two-cluster, Otsu fallback — the default), ``otsu``, ``two_cluster``, or
    ``fixed`` (pins *threshold* across the series).
    """

    enabled: bool = False
    image: str = _DEFAULT_NLS_IMAGE
    method: str = "auto"
    threshold: float = 0.0
```

Add the field to `RunConfig` (after `curation`):

```python
    nls: "NlsConfig | None" = None
```

In `load_config`, before the `return RunConfig(...)`, add:

```python
    nls = _parse_nls(data.get("nls"))
```

and add `nls=nls,` to the `RunConfig(...)` call.

Add the parser helper near `_check_known_quantities`:

```python
def _parse_nls(table: dict | None) -> "NlsConfig | None":
    """Parse the optional ``[nls]`` table into an :class:`NlsConfig` (or ``None``).

    A missing table means the step is off. *method* is validated so a typo fails
    loudly rather than silently classifying with the wrong splitter.
    """
    if table is None:
        return None
    method = str(table.get("method", "auto"))
    if method not in _NLS_METHODS:
        listed = ", ".join(_NLS_METHODS)
        raise ValueError(
            f"Run-config [nls] selects unknown method {method!r}. Available: {listed}."
        )
    return NlsConfig(
        enabled=bool(table.get("enabled", False)),
        image=str(table.get("image", _DEFAULT_NLS_IMAGE)),
        method=method,
        threshold=float(table.get("threshold", 0.0)),
    )
```

Add `"NlsConfig"` to `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --frozen pytest tests/aggregate_quantification/test_config.py -v`
Expected: PASS (all config tests, old and new).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/aggregate_quantification/config.py tests/aggregate_quantification/test_config.py
git commit -m "feat(aggregate): parse optional [nls] run-config table"
```

---

## Task 2: The `classify` pipeline step

**Files:**
- Modify: `src/cellflow/aggregate_quantification/pipeline.py`
- Test: `tests/aggregate_quantification/test_classify.py` (create)

`classify(catalog, *, config, progress_cb=None)` iterates positions, resolves the marker image (per-position relative or absolute) and the nucleus labels, picks a threshold per `config.method`, and calls `classify_position_nls_to_csv` to write each sidecar. It returns the list of written sidecar paths. Positions missing the image, the labels, or a `position_path` are skipped (not fatal). When `config` is `None` or disabled, it is a no-op returning `[]`.

For `method="auto"` the threshold is left `None` (the engine's `auto_threshold`). For `method="fixed"` the configured `threshold` is used. For `otsu`/`two_cluster` the step measures intensities once and computes the splitter threshold, then passes it through (keeping the engine untouched).

- [ ] **Step 1: Write the failing tests**

Create `tests/aggregate_quantification/test_classify.py`:

```python
"""The optional NLS classification pipeline step.

``classify`` is the headless, config-gated step that writes each position's NLS
sidecar CSV before aggregation. The classification *engine* is tested separately
(test_nls_classification.py); these cover the orchestration: per-position path
resolution, method→threshold selection, graceful skipping, and the disabled
no-op.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from cellflow.aggregate_quantification import pipeline
from cellflow.aggregate_quantification.config import NlsConfig
from cellflow.aggregate_quantification.contacts.nls_classification import (
    nls_classification_csv_path,
)


def _write_position(root: Path, *, with_nls: bool = True) -> dict:
    """A position dir with a 2-track nucleus-label stack and (optionally) a marker
    image where track 1 is bright and track 2 is dim — so any splitter separates
    them. Returns the catalogue record."""
    position = root / "pos1"
    inputs = position / "0_input"
    inputs.mkdir(parents=True)
    # Two frames, two tracks. Track 1 occupies the top rows, track 2 the bottom.
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[:, :2, :] = 1
    labels[:, 2:, :] = 2
    labels_path = inputs / "nucleus_tracked_labels.tif"
    tifffile.imwrite(labels_path, labels)

    record = {
        "id": "pos1",
        "experiment_id": "EXP1",
        "position_path": str(position),
        "nucleus_tracked_labels_path": str(labels_path),
    }
    if with_nls:
        nls = np.zeros((2, 4, 4), dtype=np.float32)
        nls[:, :2, :] = 100.0  # bright track 1
        nls[:, 2:, :] = 1.0    # dim track 2
        tifffile.imwrite(inputs / "NLS_zavg.tif", nls)
    return record


def test_classify_disabled_is_noop(tmp_path):
    record = _write_position(tmp_path)
    assert pipeline.classify([record], config=None) == []
    assert pipeline.classify([record], config=NlsConfig(enabled=False)) == []
    assert not nls_classification_csv_path(record["position_path"]).exists()


def test_classify_auto_writes_sidecar(tmp_path):
    record = _write_position(tmp_path)
    written = pipeline.classify(
        [record], config=NlsConfig(enabled=True, image="0_input/NLS_zavg.tif", method="auto")
    )
    csv_path = nls_classification_csv_path(record["position_path"])
    assert written == [csv_path]
    assert csv_path.is_file()
    table = pd.read_csv(csv_path)
    assert set(table.columns) == {"id", "label"}
    by_id = dict(zip(table["id"], table["label"]))
    assert by_id[1] == "positive"  # bright track
    assert by_id[2] == "negative"  # dim track


def test_classify_fixed_threshold(tmp_path):
    record = _write_position(tmp_path)
    # A threshold between the dim (1.0) and bright (100.0) tracks.
    written = pipeline.classify(
        [record],
        config=NlsConfig(enabled=True, method="fixed", threshold=50.0),
    )
    table = pd.read_csv(written[0])
    by_id = dict(zip(table["id"], table["label"]))
    assert by_id[1] == "positive"
    assert by_id[2] == "negative"


def test_classify_otsu_writes_sidecar(tmp_path):
    record = _write_position(tmp_path)
    written = pipeline.classify(
        [record], config=NlsConfig(enabled=True, method="otsu")
    )
    assert written and written[0].is_file()


def test_classify_skips_position_without_marker_image(tmp_path):
    record = _write_position(tmp_path, with_nls=False)
    written = pipeline.classify([record], config=NlsConfig(enabled=True))
    assert written == []
    assert not nls_classification_csv_path(record["position_path"]).exists()


def test_classify_absolute_image_path(tmp_path):
    record = _write_position(tmp_path)
    abs_image = str(Path(record["position_path"]) / "0_input" / "NLS_zavg.tif")
    written = pipeline.classify(
        [record], config=NlsConfig(enabled=True, image=abs_image, method="auto")
    )
    assert written and written[0].is_file()


def test_classify_reports_progress(tmp_path):
    record = _write_position(tmp_path)
    calls: list[tuple[int, int, str]] = []
    pipeline.classify(
        [record],
        config=NlsConfig(enabled=True),
        progress_cb=lambda done, total, name: calls.append((done, total, name)),
    )
    assert calls == [(1, 1, "pos1")]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --frozen pytest tests/aggregate_quantification/test_classify.py -v`
Expected: FAIL (`AttributeError: module 'pipeline' has no attribute 'classify'`).

- [ ] **Step 3: Implement `classify` + helpers**

In `pipeline.py`, add to the imports near the top (after the existing `from .curation import read_curation` line):

```python
from .config import NlsConfig
```

(`RunConfig` / `load_config` are already imported from `.config`; extend that import or add this line — keep it a single clean import from `.config`.)

Add `"classify"` to `__all__` (place it after `"build_quantities"`).

Add the step and helpers (put `classify` after `build_quantities`, before `_dependency_order`):

```python
def classify(
    catalog: Iterable[dict],
    *,
    config: "NlsConfig | None",
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    """Write each position's NLS subpopulation sidecar CSV (the optional step).

    Gated by the run-config ``[nls]`` table: when *config* is ``None`` or
    ``config.enabled`` is false this is a no-op returning ``[]`` (positions that
    already carry a sidecar still get their ``class_label`` joined at aggregate
    time — unchanged behaviour). Otherwise, for every position that has the marker
    image, the nucleus tracked-labels, and a folder to write into, it resolves a
    threshold per ``config.method`` and calls the headless engine's
    :func:`classify_position_nls_to_csv`. Positions missing any input are skipped
    (not fatal). Returns the written sidecar paths.

    Runs **before** :func:`aggregate` so the existing ``cell_id`` join
    (:func:`shape_tables._join_class`) finds fresh sidecars.
    """
    if config is None or not config.enabled:
        return []

    jobs: list[tuple[dict, Path, Path]] = []
    for record in catalog:
        image = _resolve_nls_image(record, config.image)
        labels = record.get("nucleus_tracked_labels_path")
        labels_path = Path(labels) if labels else None
        if (
            image is not None and image.is_file()
            and labels_path is not None and labels_path.is_file()
            and record.get("position_path")
        ):
            jobs.append((record, image, labels_path))

    from .contacts.nls_classification import (
        classify_position_nls_to_csv,
        nls_classification_csv_path,
    )

    written: list[Path] = []
    total = len(jobs)
    for index, (record, image, labels_path) in enumerate(jobs, start=1):
        if progress_cb is not None:
            progress_cb(index, total, str(record.get("id", "?")))
        threshold = _nls_threshold(config, image, labels_path)
        csv_path = nls_classification_csv_path(record["position_path"])
        classify_position_nls_to_csv(csv_path, image, labels_path, threshold=threshold)
        written.append(csv_path)
    return written


def _resolve_nls_image(record: dict, image: str) -> Path | None:
    """Resolve the marker-image spec against *record*: absolute as-is, else joined
    onto the record's ``position_path`` (so one relative entry resolves per
    position). ``None`` when the record has no folder to resolve a relative path."""
    path = Path(image)
    if path.is_absolute():
        return path
    position = record.get("position_path")
    return Path(position) / path if position else None


def _nls_threshold(config: "NlsConfig", image: Path, labels_path: Path) -> float | None:
    """The threshold to classify with, per ``config.method``.

    ``auto`` ⇒ ``None`` (the engine's :func:`auto_threshold`). ``fixed`` ⇒ the
    configured value. ``otsu`` / ``two_cluster`` measure the per-track intensities
    once and compute the splitter boundary, keeping the engine module untouched.
    """
    if config.method == "auto":
        return None
    if config.method == "fixed":
        return float(config.threshold)

    from .contacts.nls_classification import (
        _read_image_stack,
        measure_track_nls_intensity,
        split_tracks_otsu,
        split_tracks_two_clusters,
    )

    nls = _read_image_stack(image)
    labels = _read_image_stack(labels_path)
    intensities = {
        track_id: item.intensity
        for track_id, item in measure_track_nls_intensity(nls, labels).items()
    }
    splitter = split_tracks_otsu if config.method == "otsu" else split_tracks_two_clusters
    threshold, _ = splitter(intensities)
    return float(threshold)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --frozen pytest tests/aggregate_quantification/test_classify.py -v`
Expected: PASS (all 7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/aggregate_quantification/pipeline.py tests/aggregate_quantification/test_classify.py
git commit -m "feat(aggregate): add config-gated classify() pipeline step"
```

---

## Task 3: Wire `classify` into `run()`

**Files:**
- Modify: `src/cellflow/aggregate_quantification/pipeline.py:run`
- Test: `tests/aggregate_quantification/test_pipeline.py`

`run()` calls `classify(catalog, config=cfg.nls)` after `build_quantities` and before `aggregate`, so the `cell_id` join finds fresh sidecars. When `cfg.nls` is `None`/disabled the call is a no-op (existing behaviour preserved).

- [ ] **Step 1: Write the failing test**

Add to `tests/aggregate_quantification/test_pipeline.py` (it already imports `pipeline`, `numpy`, `pandas`, `tifffile`, `Path`). This test drives the full `run()` with a real `cell_shape` build plus a marker image, and asserts the aggregated `cell_shape` table carries `class_label` (proving classify ran before aggregate and the join picked the sidecar up):

```python
def test_run_with_nls_classifies_before_aggregate(tmp_path):
    """A [nls]-enabled run writes sidecars before aggregate, so the aggregated
    cell-keyed table carries the joined class_label."""
    import tomli_w  # write the TOML run-config

    position = tmp_path / "pos1"
    inputs = position / "0_input"
    inputs.mkdir(parents=True)

    # Two cell tracks (top/bottom halves), two frames — drives the real
    # CellShapeQuantifier so the aggregated table is cell_id-keyed.
    cells = np.zeros((2, 8, 8), dtype=np.int32)
    cells[:, :4, :] = 1
    cells[:, 4:, :] = 2
    cell_path = inputs / "cell_tracked_labels.tif"
    tifffile.imwrite(cell_path, cells)

    # Nucleus labels co-located with the cells; marker bright on track 1.
    nucleus_path = inputs / "nucleus_tracked_labels.tif"
    tifffile.imwrite(nucleus_path, cells)
    nls = np.where(cells == 1, 100.0, 1.0).astype(np.float32)
    tifffile.imwrite(inputs / "NLS_zavg.tif", nls)

    catalog_csv = tmp_path / "catalog.csv"
    pd.DataFrame([
        {
            "id": "pos1",
            "experiment_id": "EXP1",
            "condition": "ctrl",
            "date": "2026-06-22",
            "position_path": str(position),
            "cell_tracked_labels_path": str(cell_path),
            "nucleus_tracked_labels_path": str(nucleus_path),
        }
    ]).to_csv(catalog_csv, index=False)

    config = {
        "catalog": "catalog.csv",
        "export_dir": "export",
        "quantities": ["cell_shape"],
        "nls": {"enabled": True, "image": "0_input/NLS_zavg.tif", "method": "auto"},
    }
    config_path = tmp_path / "config.toml"
    with config_path.open("wb") as handle:
        tomli_w.dump(config, handle)

    pipeline.run(config_path)

    # The sidecar was written by the classify step.
    sidecar = position / "aggregate_quantification" / "nls_classification.csv"
    assert sidecar.is_file()

    # The aggregated cell_shape table carries the joined class_label.
    table = pd.read_csv(tmp_path / "aggregate_quantification" / "cell_shape.csv")
    assert "class_label" in table.columns
    assert set(table["class_label"].dropna().unique()) <= {"positive", "negative"}
```

> **Note:** if `tomli_w` is not available in the environment, mirror however the *existing* `test_pipeline.py` / `test_config.py` tests author TOML (e.g. writing the text directly). Check the imports already present in those files before adding a dependency — prefer the established pattern. If they write TOML as text, write this config as a text TOML string instead of using `tomli_w`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --frozen pytest tests/aggregate_quantification/test_pipeline.py::test_run_with_nls_classifies_before_aggregate -v`
Expected: FAIL (no `class_label` column — classify is not yet wired into `run`).

- [ ] **Step 3: Wire `classify` into `run`**

In `pipeline.py`, in `run()`, after the `build_quantities(...)` call and before `tables = aggregate(catalog)`, insert:

```python
    # Optional, config-gated: write per-position NLS sidecars before aggregate so
    # the cell_id join picks up fresh class_labels (no-op when [nls] is absent).
    classify(catalog, config=cfg.nls)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --frozen pytest tests/aggregate_quantification/test_pipeline.py -v`
Expected: PASS (the new test and all existing pipeline tests).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/aggregate_quantification/pipeline.py tests/aggregate_quantification/test_pipeline.py
git commit -m "feat(aggregate): run NLS classify step before aggregate"
```

---

## Task 4: Remove the interactive napari NLS UI

**Files:**
- Delete: `src/cellflow/napari/aggregate_quantification/plugins/nls_classification.py`
- Delete: `tests/napari/test_nls_classification_plugin.py`

The plugin is a hand-operated trigger for the now-config-driven engine. It self-registers via the `plugins/` package walk, so deleting the module removes it from discovery — no manifest or `__init__` edit is needed. The **engine** module (`contacts/nls_classification.py`) and its test (`tests/aggregate_quantification/test_nls_classification.py`) stay; the package `__init__` re-exports (`measure_track_nls_intensity`, `auto_threshold`, `classify_position_nls_to_csv`, …) stay — they are used by `shape_tables`, `plots/_pooling.py`, and the widget.

- [ ] **Step 1: Confirm nothing imports the plugin module/class**

Run:
```bash
grep -rn "plugins.nls_classification\|NLSClassificationPlugin" src/ tests/
```
Expected: matches **only** inside the two files being deleted. (References to `contacts.nls_classification` and the engine functions are fine — those stay.)

- [ ] **Step 2: Delete the plugin and its test**

```bash
git rm src/cellflow/napari/aggregate_quantification/plugins/nls_classification.py
git rm tests/napari/test_nls_classification_plugin.py
```

- [ ] **Step 3: Verify plugin discovery + napari suites still pass**

Run:
```bash
uv run --frozen pytest tests/napari/ -v
```
Expected: PASS, with no import errors from plugin discovery. If `tests/napari/test_public_private_boundary.py` or `tests/napari/test_aggregate_quantification_widget.py` assert the NLS plugin is present/registered, update those assertions to reflect its removal (the NLS plugin should no longer appear among `available_analysis_plugins()`); do **not** weaken unrelated assertions.

- [ ] **Step 4: Run the full aggregate + napari suites**

Run:
```bash
uv run --frozen pytest tests/aggregate_quantification/ tests/napari/ -q
```
Expected: PASS (no references to the deleted plugin remain).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(napari): remove interactive NLS classification UI (now a CLI step)"
```

---

## Final review

After all four tasks, dispatch a final code-review over the whole branch diff and confirm:
- The engine module `contacts/nls_classification.py` is unmodified.
- `classify` is a pure no-op when `[nls]` is absent/disabled (no behaviour change for existing configs).
- `run()` calls `classify` strictly between `build_quantities` and `aggregate`.
- No dangling imports/registrations of the deleted napari plugin.
- The full aggregate suite is green: `uv run --frozen pytest tests/aggregate_quantification/ -q`.

## Notes on scope (deferred — do NOT implement)

These are explicitly out of scope per the design spec; do not add them:
- **Global (series-wide) auto threshold** — only per-position thresholding here.
- **Sidecar relocation** — keep `nls_classification_csv_path` (per-position) as is.
- **Multi-class / >2 labels** — engine stays binary.
- **Provenance stamping** of method/threshold/image into the sidecar — deferred to the quantifier provenance-JSON work.
