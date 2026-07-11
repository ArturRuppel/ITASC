# Aggregate Capstone napari Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the cross-position aggregate step a home in the main CellFlow napari app as a project-level "Aggregate" capstone that pools every processed position into the flat tidy tables.

**Architecture:** A lean new `AggregateWidget` sits at the bottom of the main app's section stack, reads the same catalog records the `ExperimentsPanel` builds, and drives the existing headless engine (`author_config` then `pipeline.run`). Pool-only is enforced by authoring a `catalog.csv` that contains only positions whose `contacts.h5` already exists, so `run` does load-and-pool with no per-position recompute. Plotting stays in Iris; curation is out of scope (folded into the per-position section in a separate TODO).

**Tech Stack:** Python, qtpy/napari (`thread_worker`), pytest. Reuses `cellflow.contact_analysis` (`author_config`, `run`, `catalogue_root`, `save_catalog`, `write_config`).

**Spec:** `docs/superpowers/specs/2026-07-12-aggregate-capstone-napari-integration-design.md`

---

## File Structure

**Create:**
- `src/cellflow/napari/aggregate_widget.py` — the capstone. Two module-level pure functions (`partition_ready`, `pool_positions`) plus the Qt `AggregateWidget` (readiness readout, Run button, results list, `thread_worker` driver). One clear responsibility: pool ready positions and report.
- `tests/napari/test_aggregate_widget.py` — unit tests for the pure functions and the widget.

**Modify:**
- `src/cellflow/contact_analysis/__init__.py` — re-export `author_config` (already public in `pipeline`, not yet on the package surface).
- `src/cellflow/napari/ui_style.py` — add an `"aggregate"` accent to `STAGE_ACCENTS`.
- `src/cellflow/napari/main_widget.py` — instantiate `AggregateWidget`, add its section after the per-position "Results" section, feed it catalog records on refresh, refactor the catalog-record building into a reused helper, and set its section status.
- `src/cellflow/napari/contact_analysis_widget.py` — simplify `make_contact_analysis_widget` to drop the retired studio (Task 5).

**Delete (Task 5, verify-first):**
- `src/cellflow/napari/contact_analysis_studio.py`
- `src/cellflow/napari/contact_analysis_run_area.py`
- `src/cellflow/napari/contact_analysis_params.py`
- `tests/napari/test_contact_analysis_studio.py`
- `tests/napari/test_run_area.py`
- `tests/napari/test_shared_params.py`

---

## Task 1: Expose `author_config` on the package surface

The widget composes `author_config` (write `catalog.csv` + `config.toml`) then `run`. `author_config` is public in `cellflow.contact_analysis.pipeline` but not re-exported from the package `__init__`, unlike `run` / `aggregate`. Add it so the widget imports from the stable surface.

**Files:**
- Modify: `src/cellflow/contact_analysis/__init__.py`
- Test: `tests/contact_analysis/test_author_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/contact_analysis/test_author_config.py`:

```python
def test_author_config_is_on_package_surface():
    import cellflow.contact_analysis as ca

    assert hasattr(ca, "author_config")
    assert ca.author_config is __import__(
        "cellflow.contact_analysis.pipeline", fromlist=["author_config"]
    ).author_config
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/contact_analysis/test_author_config.py::test_author_config_is_on_package_surface -v`
Expected: FAIL with `AttributeError: module 'cellflow.contact_analysis' has no attribute 'author_config'`.

- [ ] **Step 3: Add the re-export**

In `src/cellflow/contact_analysis/__init__.py`, add `author_config` to the `from .pipeline import (...)` block and to `__all__`:

```python
from .pipeline import (
    aggregate,
    author_config,
    build_catalog,
    build_quantities,
    run,
    select_quantifiers,
)
```

And in `__all__`, add `"author_config",` next to `"run"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/contact_analysis/test_author_config.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/contact_analysis/__init__.py tests/contact_analysis/test_author_config.py
git commit -m "feat(contact-analysis): re-export author_config from the package surface"
```

---

## Task 2: `partition_ready` pure function + module

Create the widget module with the readiness partition first (pure, Qt-free logic), so it is testable in isolation. A record is *ready* when its `contact_analysis_path` (the per-position `contacts.h5`) exists on disk. The `_record` test helper here is the full main_widget-shaped record (identity in a `columns` bag) so later tasks reuse it unchanged.

**Files:**
- Create: `src/cellflow/napari/aggregate_widget.py`
- Test: `tests/napari/test_aggregate_widget.py`

- [ ] **Step 1: Write the failing test**

Create `tests/napari/test_aggregate_widget.py`:

```python
"""Aggregate capstone: readiness partition + engine drive."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cellflow.napari.aggregate_widget import partition_ready


def _record(pos_dir: Path, *, ready: bool) -> dict:
    """A main_widget-shaped catalog record (see ``_catalog_record_for_position``).

    Identity lives in the ``columns`` bag under the seed level names
    (``condition`` / ``experiment_id`` / ``position_id``), which is what
    ``save_catalog`` reads via its ``_BAG_TO_CSV`` mapping. ``ready`` controls
    whether the per-position ``contacts.h5`` exists on disk.
    """
    h5 = pos_dir / "4_contact_analysis" / "contact_analysis.h5"
    if ready:
        h5.parent.mkdir(parents=True, exist_ok=True)
        h5.write_bytes(b"")
    return {
        "position_path": pos_dir,
        "contact_analysis_path": h5,
        "cell_tracked_labels_path": pos_dir / "cell_labels.tif",
        "nucleus_tracked_labels_path": pos_dir / "nucleus_labels.tif",
        "columns": {
            "condition": "ctrl",
            "experiment_id": "exp1",
            "position_id": pos_dir.name,
        },
    }


def test_partition_ready_splits_by_h5_presence(tmp_path):
    a = _record(tmp_path / "posA", ready=True)
    b = _record(tmp_path / "posB", ready=False)
    ready, not_ready = partition_ready([a, b])
    assert ready == [a]
    assert not_ready == [b]


def test_partition_ready_empty():
    assert partition_ready([]) == ([], [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/napari/test_aggregate_widget.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cellflow.napari.aggregate_widget'`.

- [ ] **Step 3: Create the module with the readiness partition**

Create `src/cellflow/napari/aggregate_widget.py`:

```python
"""Aggregate capstone: pool every processed position into project-level tables.

The main app's project-level bookend to the per-position sections. Reads the same
catalog records the ``ExperimentsPanel`` builds, and drives the headless engine
(``author_config`` then ``pipeline.run``). Pool-only: it aggregates positions
whose ``contacts.h5`` already exists and never builds missing ones, so ``run`` is
load-and-pool with no per-position recompute. Plots live in Iris.
"""
from __future__ import annotations

from pathlib import Path


def partition_ready(records):
    """Split catalog *records* into ``(ready, not_ready)`` by ``contacts.h5``.

    A record is *ready* when its ``contact_analysis_path`` exists on disk.
    """
    ready, not_ready = [], []
    for rec in records:
        path = rec.get("contact_analysis_path")
        if path is not None and Path(path).exists():
            ready.append(rec)
        else:
            not_ready.append(rec)
    return ready, not_ready


def _position_name(record) -> str:
    path = record.get("position_path")
    return Path(path).name if path else "(unknown)"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/napari/test_aggregate_widget.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_widget.py tests/napari/test_aggregate_widget.py
git commit -m "feat(napari): add aggregate capstone readiness partition"
```

---

## Task 3: `pool_positions` engine drive (ready subset only)

Add the function that authors the project artifacts for the ready positions and runs the engine over them, reporting the skipped names. Monkeypatch `run` in the test so it does not need real `contacts.h5` content (the engine's pooling is covered by `tests/contact_analysis/test_pipeline.py`); let `author_config` really write so the authored catalog can be asserted.

**Files:**
- Modify: `src/cellflow/napari/aggregate_widget.py`
- Test: `tests/napari/test_aggregate_widget.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/napari/test_aggregate_widget.py`:

```python
from cellflow.contact_analysis import load_catalog
from cellflow.napari import aggregate_widget as aw


def test_pool_positions_authors_ready_subset_and_runs(tmp_path, monkeypatch):
    ready = [
        _record(tmp_path / "study" / "posA", ready=True),
        _record(tmp_path / "study" / "posB", ready=True),
    ]
    seen = {}

    def _fake_run(config_path):
        seen["config_path"] = Path(config_path)
        return {"object_table": Path(config_path).parent / "object_table.csv"}

    monkeypatch.setattr(aw, "run", _fake_run)

    result = aw.pool_positions(ready, skipped_names=["posC"])

    # The engine was driven with the authored config.
    config_path = seen["config_path"]
    assert config_path.name == "config.toml"
    # The authored catalog contains exactly the two ready positions.
    catalog = load_catalog(config_path.parent / "catalog.csv")
    names = sorted(Path(rec["position_path"]).name for rec in catalog)
    assert names == ["posA", "posB"]
    # The result carries the skipped names and table map for the UI.
    assert result["skipped"] == ["posC"]
    assert "object_table" in result["tables"]
    # Tables land in the ready positions' common ancestor.
    assert result["project_dir"] == (tmp_path / "study")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/napari/test_aggregate_widget.py::test_pool_positions_authors_ready_subset_and_runs -v`
Expected: FAIL with `AttributeError: module 'cellflow.napari.aggregate_widget' has no attribute 'pool_positions'` (or `ImportError`/`monkeypatch` setattr on a missing name). This proves `pool_positions` does not exist yet.

- [ ] **Step 3: Add `pool_positions` and its engine imports**

In `src/cellflow/napari/aggregate_widget.py`, add the engine imports directly below `from pathlib import Path`:

```python
from cellflow.contact_analysis import author_config, run
from cellflow.contact_analysis.shape_tables import catalogue_root
```

Then append `pool_positions` after `_position_name`:

```python
def pool_positions(ready_records, skipped_names):
    """Author the project artifacts for *ready_records* and run the engine.

    Writes ``catalog.csv`` + ``config.toml`` into the ready positions' common
    ancestor (:func:`catalogue_root`), then ``run``s the pipeline over them.
    Returns a result dict for the UI: the ``name -> path`` table map, the
    ``skipped`` position names, and the ``project_dir`` the tables landed under.
    """
    project_dir = catalogue_root(ready_records)
    config_path = author_config(project_dir, ready_records, quantities=())
    tables = run(config_path)
    return {
        "tables": tables,
        "skipped": list(skipped_names),
        "project_dir": project_dir,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/napari/test_aggregate_widget.py -v`
Expected: PASS (all tests in the file). The `partition_ready` tests are unaffected: they read only `position_path` / `contact_analysis_path`.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_widget.py tests/napari/test_aggregate_widget.py
git commit -m "feat(napari): pool_positions authors the ready subset and drives the engine"
```

---

## Task 4: The `AggregateWidget` Qt class

Add the Qt widget: a scope subtitle, a readiness readout, a "Pool ready positions" button, a progress bar, a results list, and a status line. Run happens on a `thread_worker` calling `pool_positions`. The widget exposes `set_records(records)` (fed by `main_widget`) and `section_status()` (for the section dot).

**Files:**
- Modify: `src/cellflow/napari/aggregate_widget.py`
- Test: `tests/napari/test_aggregate_widget.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/napari/test_aggregate_widget.py`:

```python
from napari.qt import get_qapp


def test_widget_readout_reports_ready_split_and_names_not_ready(tmp_path):
    get_qapp()
    from cellflow.napari.aggregate_widget import AggregateWidget

    w = AggregateWidget()
    w.set_records([
        _record(tmp_path / "posA", ready=True),
        _record(tmp_path / "posB", ready=False),
    ])
    assert "1 of 2" in w.readout.text()
    assert "posB" in w.readout.text()
    assert w.run_btn.isEnabled() is True
    assert w.section_status() == "in_progress"


def test_widget_run_button_disabled_when_nothing_ready(tmp_path):
    get_qapp()
    from cellflow.napari.aggregate_widget import AggregateWidget

    w = AggregateWidget()
    w.set_records([_record(tmp_path / "posA", ready=False)])
    assert w.run_btn.isEnabled() is False
    assert w.section_status() == "not_started"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/napari/test_aggregate_widget.py::test_widget_readout_reports_ready_split_and_names_not_ready -v`
Expected: FAIL with `ImportError: cannot import name 'AggregateWidget'`.

- [ ] **Step 3: Implement the widget**

Append to `src/cellflow/napari/aggregate_widget.py` (add the imports at the top of the file alongside the existing ones):

```python
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import (
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
```

```python
class AggregateWidget(QWidget):
    """Project-level capstone: pool every ready position into tidy tables.

    Fed catalog records via :meth:`set_records` (the same records the app's
    ``ExperimentsPanel`` builds). Pool-only: Run aggregates positions whose
    ``contacts.h5`` exists and reports the ones it skipped by name.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: list[dict] = []
        self._worker = None
        self._has_run = False

        layout = QVBoxLayout(self)
        self.subtitle = QLabel(
            "Pools every processed position into project-level tables."
        )
        self.subtitle.setWordWrap(True)
        self.readout = QLabel("No data folders yet.")
        self.readout.setWordWrap(True)
        self.run_btn = QPushButton("Pool ready positions")
        self.run_btn.clicked.connect(self._on_run)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.results = QListWidget()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        for widget in (
            self.subtitle,
            self.readout,
            self.run_btn,
            self.progress,
            self.results,
            self.status,
        ):
            layout.addWidget(widget)
        self._refresh_readout()

    # ------------------------------------------------------------------ inputs
    def set_records(self, records) -> None:
        """Replace the catalog records the readiness readout reflects."""
        self._records = list(records or [])
        self._refresh_readout()

    def section_status(self) -> str:
        """Status for the enclosing section dot: not_started / in_progress / done."""
        ready, _ = partition_ready(self._records)
        if not ready:
            return "not_started"
        return "done" if self._has_run else "in_progress"

    # --------------------------------------------------------------- rendering
    def _refresh_readout(self) -> None:
        ready, not_ready = partition_ready(self._records)
        total = len(self._records)
        if total == 0:
            self.readout.setText("No data folders yet.")
        else:
            message = f"{len(ready)} of {total} positions analyzed"
            if not_ready:
                names = ", ".join(_position_name(r) for r in not_ready)
                message += f" — not yet ready: {names}"
            self.readout.setText(message)
        self.run_btn.setEnabled(bool(ready) and self._worker is None)

    # --------------------------------------------------------------------- run
    def _on_run(self) -> None:
        ready, not_ready = partition_ready(self._records)
        if not ready:
            show_info("No analyzed positions to pool.")
            return
        skipped = [_position_name(r) for r in not_ready]
        self.results.clear()
        self.status.setText("Pooling…")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.run_btn.setEnabled(False)

        @thread_worker(
            connect={"returned": self._on_done, "errored": self._on_error}
        )
        def _work():
            return pool_positions(ready, skipped)

        self._worker = _work()

    def _on_done(self, result: dict) -> None:
        self._worker = None
        self._has_run = True
        self.progress.setVisible(False)
        for name, path in sorted(result["tables"].items()):
            self.results.addItem(f"{name}: {path}")
        message = f"Pooled into {result['project_dir']}. Plots live in Iris."
        if result["skipped"]:
            message += f" Skipped (not analyzed): {', '.join(result['skipped'])}."
        self.status.setText(message)
        show_info(message)
        self._refresh_readout()

    def _on_error(self, exc: Exception) -> None:
        self._worker = None
        self.progress.setVisible(False)
        self.status.setText(f"Aggregate failed: {exc}")
        show_error(f"Aggregate failed: {exc}")
        self._refresh_readout()
```

Note: `—` is the em-dash character used in the readout string (UI copy, not prose). Keep it as the escape or a literal em-dash; either renders identically.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/napari/test_aggregate_widget.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_widget.py tests/napari/test_aggregate_widget.py
git commit -m "feat(napari): AggregateWidget capstone (readiness readout, Run, results)"
```

---

## Task 5: Add the `aggregate` stage accent

The section needs an accent color like the other stage sections. Add an `"aggregate"` key to `STAGE_ACCENTS` mapped to the palette's `teal` (a valid color name in every theme; distinct from the per-position `contact_analysis` lavender).

**Files:**
- Modify: `src/cellflow/napari/ui_style.py`
- Test: `tests/napari/test_aggregate_widget.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/napari/test_aggregate_widget.py`:

```python
def test_aggregate_stage_accent_resolves():
    from cellflow.napari.ui_style import stage_accent

    accent = stage_accent("aggregate")
    assert isinstance(accent, str) and accent.startswith("#")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/napari/test_aggregate_widget.py::test_aggregate_stage_accent_resolves -v`
Expected: FAIL with `KeyError: 'aggregate'`.

- [ ] **Step 3: Add the accent**

In `src/cellflow/napari/ui_style.py`, add to the `STAGE_ACCENTS` dict:

```python
STAGE_ACCENTS = {
    "project_status":   "sapphire",
    "cellpose":         "sapphire",
    "nucleus":          "peach",
    "cell":             "green",
    "contact_analysis": "lavender",
    "aggregate":        "teal",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/napari/test_aggregate_widget.py::test_aggregate_stage_accent_resolves -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/ui_style.py tests/napari/test_aggregate_widget.py
git commit -m "feat(napari): add the aggregate stage accent"
```

---

## Task 6: Wire the capstone into `main_widget`

Instantiate `AggregateWidget`, wrap it in a `CollapsibleSection` titled "Aggregate", append it after the per-position "Results" section, feed it catalog records whenever the app refreshes, and set its section dot. Refactor the catalog-record building (currently inline in `_on_save_project`) into a reused helper so the aggregate feed and Save-project share one code path (DRY).

**Files:**
- Modify: `src/cellflow/napari/main_widget.py`
- Test: `tests/napari/test_main_widget_aggregate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/napari/test_main_widget_aggregate.py`:

```python
"""The aggregate capstone is wired into the full CellFlow app."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari.qt import get_qapp
from cellflow.napari.main_widget import CellFlowMainWidget


def _fake_viewer():
    class _Sel:
        active = None

    class _Layers(dict):
        selection = _Sel()
        events = SimpleNamespace(removed=SimpleNamespace(connect=lambda cb: None))

        def remove(self, layer):
            self.pop(layer.name, None)

    viewer = SimpleNamespace()
    viewer.layers = _Layers()
    viewer.dims = SimpleNamespace(
        current_step=(0, 0),
        events=SimpleNamespace(current_step=SimpleNamespace(connect=lambda cb: None)),
    )
    viewer.add_image = MagicMock()
    viewer.add_labels = MagicMock()
    viewer.add_shapes = MagicMock()
    viewer.bind_key = MagicMock()
    return viewer


def test_app_has_aggregate_section_after_results():
    get_qapp()
    w = CellFlowMainWidget(_fake_viewer())
    assert hasattr(w, "aggregate_widget")
    assert hasattr(w, "aggregate_section")
    # The capstone is the last section in the stage stack.
    layout = w.scroll_layout
    order = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert order.index(w.aggregate_section) > order.index(w.contact_analysis_section)


def test_catalog_records_helper_feeds_aggregate(tmp_path):
    get_qapp()
    w = CellFlowMainWidget(_fake_viewer())
    records = w._catalog_records_for_panel([
        {"position_path": tmp_path / "posA", "columns": {"condition": "ctrl", "id": "posA"}},
    ])
    assert records[0]["contact_analysis_path"].name == "contact_analysis.h5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/napari/test_main_widget_aggregate.py -v`
Expected: FAIL with `AttributeError: 'CellFlowMainWidget' object has no attribute 'aggregate_widget'`.

- [ ] **Step 3: Add the import and refactor the record helper**

In `src/cellflow/napari/main_widget.py`, add the import near the other napari widget imports:

```python
from cellflow.napari.aggregate_widget import AggregateWidget
```

Add a reusable helper method on `CellFlowMainWidget` (place it next to `_catalog_record_for_position`):

```python
def _catalog_records_for_panel(self, panel_records) -> list[dict]:
    """Catalog records (with committed output paths) for a list of panel rows."""
    return [
        self._catalog_record_for_position(
            rec["position_path"], rec.get("columns", {})
        )
        for rec in panel_records
    ]
```

Then change `_on_save_project` to use it (replace the inline list comprehension that builds `records`):

```python
records = self._catalog_records_for_panel(self._positions_panel.records())
```

- [ ] **Step 4: Add the section (project-level, not a per-position detail pane)**

In `CellFlowMainWidget.__init__`, after the `self.contact_analysis_section = CollapsibleSection(...)` block, add the widget and section:

```python
self.aggregate_widget = AggregateWidget()
self.aggregate_section = CollapsibleSection(
    "Aggregate",
    self.aggregate_widget,
    expanded=False,
    accent_color=stage_accent("aggregate"),
)
```

Add it to the scroll layout right after the contact-analysis section:

```python
self.scroll_layout.addWidget(self.contact_analysis_section)
self.scroll_layout.addWidget(self.aggregate_section)
```

Do NOT add it to `self._stage_sections`. That tuple is the *selected-position detail pane*, whose members `_update_disclosure` hides whenever no position is active (line ~308). The aggregate capstone is project-level: it pools all positions and must not be gated on a single selection. Instead, after the `_stage_sections` block and its `set_status("not_started")` loop, seed the capstone's own initial state (hidden until positions exist):

```python
self.aggregate_section.setVisible(False)
self.aggregate_section.set_status(self.aggregate_widget.section_status())
```

- [ ] **Step 5: Feed records and drive visibility from the catalog, not selection**

Add a method on `CellFlowMainWidget` (place it next to `_catalog_records_for_panel`):

```python
def _refresh_aggregate(self) -> None:
    """Feed the project-level catalog to the capstone; show it once positions exist."""
    records = self._catalog_records_for_panel(self._positions_panel.records())
    self.aggregate_widget.set_records(records)
    self.aggregate_section.setVisible(bool(records))
    self.aggregate_section.set_status(self.aggregate_widget.section_status())
```

Wire it to the panel's `records_changed` signal (so discovering or clearing positions updates the capstone even with no row selected). Add this next to the other `self._positions_panel.<signal>.connect(...)` lines in `__init__`:

```python
self._positions_panel.records_changed.connect(self._refresh_aggregate)
```

Also call it from `_refresh_all`, after `self._positions_panel.refresh_statuses()`, so a completed per-position contact run repaints the readiness readout:

```python
self._refresh_aggregate()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/napari/test_main_widget_aggregate.py -v`
Expected: PASS (2 tests).

Also run the existing main-widget tests to confirm no regression:

Run: `pytest tests/napari/test_main_widget_config_project.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cellflow/napari/main_widget.py tests/napari/test_main_widget_aggregate.py
git commit -m "feat(napari): wire the aggregate capstone into the full app"
```

---

## Task 7: Retire the legacy studio (verify-first)

The new capstone supersedes the studio's multi-position role. The studio is reached only through the plugin factory `make_contact_analysis_widget`; simplify it to always return the bare per-position `ContactAnalysisWidget` (the standalone dock), then delete the studio modules and their tests. Do each step only after the prior one is verified.

**Files:**
- Modify: `src/cellflow/napari/contact_analysis_widget.py`
- Delete: `src/cellflow/napari/contact_analysis_studio.py`, `contact_analysis_run_area.py`, `contact_analysis_params.py`
- Delete: `tests/napari/test_contact_analysis_studio.py`, `test_run_area.py`, `test_shared_params.py`

- [ ] **Step 1: Verify the studio has no live callers beyond the factory + its own tests**

Run:
```bash
grep -rn "contact_analysis_studio\|ContactAnalysisStudioWidget\|contact_analysis_run_area\|RunArea\|contact_analysis_params\|SharedParamsWidget" src tests
```
Expected: hits only in `contact_analysis_widget.py:make_contact_analysis_widget`, the three studio modules themselves, and the three studio test files. If any *other* live module imports them, STOP: that dependency is not yet covered by the capstone. Report it rather than deleting.

- [ ] **Step 2: Simplify the factory**

In `src/cellflow/napari/contact_analysis_widget.py`, replace the `try/except ImportError` that returns `ContactAnalysisStudioWidget` with a direct return of the standalone widget, and update the docstring:

```python
def make_contact_analysis_widget(napari_viewer=None):
    """napari plugin entry point: the per-position Contact Analysis dock widget.

    Returns the bare per-position :class:`ContactAnalysisWidget` in standalone
    mode (own file pickers + config). The cross-position aggregate role now lives
    in the full app's Aggregate capstone (``cellflow.napari.aggregate_widget``),
    so there is no separate interactive studio to serve here. Runs the napari
    layer-delegate patch (normally done by the orchestrator package).
    """
    try:
        from cellflow.napari._napari_compat import patch_napari_layer_delegate

        patch_napari_layer_delegate()
    except Exception:  # pragma: no cover - patch is best-effort
        pass
    if napari_viewer is None:
        napari_viewer = napari.current_viewer()
    return ContactAnalysisWidget(viewer=napari_viewer, standalone=True)
```

- [ ] **Step 3: Verify the factory still imports and returns a widget**

Run:
```bash
python -c "import os; os.environ.setdefault('QT_QPA_PLATFORM','offscreen'); from napari.qt import get_qapp; get_qapp(); from cellflow.napari.contact_analysis_widget import make_contact_analysis_widget as f; w=f(); print(type(w).__name__)"
```
Expected: prints `ContactAnalysisWidget`.

- [ ] **Step 4: Delete the studio modules and their tests**

```bash
git rm src/cellflow/napari/contact_analysis_studio.py \
       src/cellflow/napari/contact_analysis_run_area.py \
       src/cellflow/napari/contact_analysis_params.py \
       tests/napari/test_contact_analysis_studio.py \
       tests/napari/test_run_area.py \
       tests/napari/test_shared_params.py
```

- [ ] **Step 5: Verify nothing dangling references the deleted modules**

Run:
```bash
grep -rn "contact_analysis_studio\|ContactAnalysisStudioWidget\|contact_analysis_run_area\|RunArea\|contact_analysis_params\|SharedParamsWidget" src tests
```
Expected: no output.

- [ ] **Step 6: Run the napari test suite**

Run: `pytest tests/napari -q`
Expected: PASS (no import errors from the deletions).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(napari): retire the legacy contact-analysis studio (superseded by the aggregate capstone)"
```

---

## Task 8: Update TODO.md

Check off the aggregate napari front-end work and note the studio retirement.

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Edit the aggregate section**

In `TODO.md`, under "## Aggregate Quantification: napari front-end + curation consolidation", mark the "napari front-end refocus" item done and add a note that the capstone shipped and the legacy studio was retired:

```markdown
- [x] **napari front-end refocus** — the cross-position aggregate now lives in the
  full app as the project-level "Aggregate" capstone (`napari/aggregate_widget.py`):
  a thin front-end that authors `catalog.csv` + `config.toml` and drives
  `pipeline.run` over the ready positions (pool-only). The legacy interactive studio
  (`contact_analysis_studio` + `_run_area` + `_params`) is retired. Iris owns all
  plotting. (Shipped 2026-07-12.)
```

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "docs: mark the aggregate napari front-end shipped; studio retired"
```

---

## Final verification

- [ ] **Run the full test suite**

Run: `pytest tests/napari tests/contact_analysis -q`
Expected: PASS.

- [ ] **Manual smoke (optional, if a napari display is available)**

Launch the app, discover a study root with a mix of analyzed and un-analyzed positions, confirm the "Aggregate" section shows the correct "N of M" readout, press "Pool ready positions", and confirm the results list shows the written table CSVs and the status line names the skipped positions and points at Iris.
