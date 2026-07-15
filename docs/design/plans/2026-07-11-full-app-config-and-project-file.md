# Full-app config autosave + project catalog CSV — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In the full CellFlow app, autosave the per-folder parameter config on every run, and give the Data folders panel explicit Save/Load of a project catalog CSV that the aggregate studio can run from unmodified.

**Architecture:** All changes are in `napari/main_widget.py` plus its tests; no stage-widget code changes (autosave wires the existing run buttons' `clicked` signals). The project CSV reuses `contact_analysis/catalog.py` (`save_catalog` / `load_catalog` / `merge_catalog_records`), with a small save-side adapter in `main_widget` that stamps each panel row with the default staged-layout paths (`aggregate_quantification/contact_analysis.h5`, `3_cell/tracked_labels.tif`, `2_nucleus/tracked_labels.tif`).

**Tech Stack:** Python 3.10+, Qt (qtpy), napari, pytest. Qt tests bootstrap via napari `get_qapp()`.

**Reference spec:** `docs/superpowers/specs/2026-07-11-full-app-config-and-project-file-design.md`

---

## Preflight (read before Task 1)

Run the existing suite to confirm a green baseline:

```bash
cd /home/aruppel/Projects/CellFlow
python -m pytest tests/napari/test_main_widget_cellpose_integration.py -q
```

Key facts the tasks rely on (verified against the code, do not re-derive):

- Toolbar buttons in `main_widget._setup_project_ui`: `project_btn` (icon `new`),
  `load_btn` (icon `load`), `save_btn` (icon `save`), `load_from_btn` (icon `load`),
  `save_as_btn` (icon `save`), `refresh_btn` (icon `reset`).
- Handlers wired in `__init__` (~lines 262-268):
  `project_btn→_on_set_position_folder`, `save_btn→_on_save_config`,
  `save_as_btn→_on_save_config_as`, `load_btn→_on_load_config`,
  `load_from_btn→_on_load_config_from`, `refresh_btn→_refresh_all`.
- `_save_config(self, path)` (~line 583) writes `get_state()` as JSON and calls
  `show_info`. `_load_config` (~593) reads + `set_state` + `show_info`.
- `_gate.register(...)` in `_register_gate_controls` lists `project_btn`,
  `load_btn`, `load_from_btn` as `CONTEXT_CHANGING`.
- Panel is `self._positions_panel` (an `ExperimentsPanel`); it exposes
  `records()`, `set_records(entries)`, `keys()`. Each entry is
  `{"key": str, "columns": {...}, "payload": {"position_path": Path}}`.
- `records()` returns `{**payload, "columns": {...}}`, i.e.
  `{"position_path": Path, "columns": {...}}` per row.
- Run buttons to wire for autosave:
  `self._cellpose_widget.nucleus_run_btn`, `self._cellpose_widget.cell_run_btn`,
  `self.nucleus_workflow_widget.seg_run_btn`, `self.nucleus_workflow_widget.db_run_btn`,
  `self.nucleus_workflow_widget.solve_run_btn`, `self.cell_workflow_widget.run_btn`.
- `catalog.py` exports: `CONTACT_ANALYSIS_RELPATH` (`"aggregate_quantification/contact_analysis.h5"`),
  `save_catalog(csv_path, records)`, `load_catalog(csv_path)`,
  `merge_catalog_records(existing, incoming)`, `REQUIRED_CSV_COLUMNS`.
- The staged label paths `main_widget._refresh_all` already uses:
  `pos_dir/"3_cell"/"tracked_labels.tif"`, `pos_dir/"2_nucleus"/"tracked_labels.tif"`.

---

## Task 1: `_save_config` gains a quiet flag

**Files:**
- Modify: `src/cellflow/napari/main_widget.py` (`_save_config`, ~line 583)
- Test: `tests/napari/test_main_widget_config_project.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/napari/test_main_widget_config_project.py`:

```python
"""Config autosave + project-catalog CSV behavior in the full CellFlow app."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from napari._qt.qt_main_window import get_qapp
from cellflow.napari.main_widget import CellFlowMainWidget


@pytest.fixture
def widget(make_napari_viewer):
    get_qapp()
    viewer = make_napari_viewer()
    return CellFlowMainWidget(viewer)


def test_save_config_quiet_suppresses_toast(widget, tmp_path):
    cfg = tmp_path / "cellflow_config.json"
    with patch("cellflow.napari.main_widget.show_info") as info:
        widget._save_config(str(cfg), quiet=True)
    assert cfg.exists()
    info.assert_not_called()
    # And the loud path still notifies.
    with patch("cellflow.napari.main_widget.show_info") as info:
        widget._save_config(str(cfg))
    info.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/napari/test_main_widget_config_project.py::test_save_config_quiet_suppresses_toast -q`
Expected: FAIL (`_save_config() got an unexpected keyword argument 'quiet'`).

- [ ] **Step 3: Implement the quiet flag**

In `main_widget.py`, change `_save_config`:

```python
    def _save_config(self, path: str, *, quiet: bool = False) -> None:
        """Save state to a JSON file. ``quiet`` suppresses the success toast."""
        state = self.get_state()
        try:
            with open(path, "w") as f:
                json.dump(state, f, indent=4)
            if not quiet:
                show_info(f"Config saved to {path}")
        except Exception as e:
            show_error(f"Error saving config: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/napari/test_main_widget_config_project.py::test_save_config_quiet_suppresses_toast -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/main_widget.py tests/napari/test_main_widget_config_project.py
git commit -m "feat(main-widget): quiet flag for _save_config"
```

---

## Task 2: Autosave config on every run

**Files:**
- Modify: `src/cellflow/napari/main_widget.py` (`__init__` signal wiring; add `_autosave_config`)
- Test: `tests/napari/test_main_widget_config_project.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/napari/test_main_widget_config_project.py`:

Do NOT emit the run buttons' `clicked` in tests: that also fires each widget's real
run handler (which may spawn a worker or raise without real data). Two guarantees
cover the wiring instead: the six run-button attributes must exist (a rename crashes
`__init__`, since the wiring loop calls `getattr` at construction), and
`_autosave_config` must behave correctly.

```python
RUN_BUTTONS = [
    ("_cellpose_widget", "nucleus_run_btn"),
    ("_cellpose_widget", "cell_run_btn"),
    ("nucleus_workflow_widget", "seg_run_btn"),
    ("nucleus_workflow_widget", "db_run_btn"),
    ("nucleus_workflow_widget", "solve_run_btn"),
    ("cell_workflow_widget", "run_btn"),
]


@pytest.mark.parametrize("widget_attr,btn_attr", RUN_BUTTONS)
def test_run_button_attr_exists(widget, widget_attr, btn_attr):
    # The __init__ wiring loop getattr's each of these; a rename would already have
    # crashed the fixture. This pins the exact names the autosave loop depends on.
    assert hasattr(getattr(widget, widget_attr), btn_attr)


def test_autosave_writes_config_into_pos_dir(widget, tmp_path):
    widget._pos_dir = tmp_path
    widget._autosave_config()
    assert (tmp_path / "cellflow_config.json").exists()


def test_autosave_noop_without_pos_dir(widget):
    widget._pos_dir = None
    widget._autosave_config()  # must not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/napari/test_main_widget_config_project.py -q -k "autosave or attr_exists"`
Expected: FAIL (`_autosave_config` attribute error / not writing).

- [ ] **Step 3: Add the slot and wire the buttons**

In `main_widget.py` `__init__`, in the "Connect signals" block (after the existing
`self.refresh_btn.clicked.connect(...)`), add:

```python
        # Config travels with the folder: snapshot params into the folder on every
        # run. The stage widgets' run buttons are the trigger (cancel re-clicks
        # re-save the same config, which is harmless).
        for widget_obj, attr in (
            (self._cellpose_widget, "nucleus_run_btn"),
            (self._cellpose_widget, "cell_run_btn"),
            (self.nucleus_workflow_widget, "seg_run_btn"),
            (self.nucleus_workflow_widget, "db_run_btn"),
            (self.nucleus_workflow_widget, "solve_run_btn"),
            (self.cell_workflow_widget, "run_btn"),
        ):
            getattr(widget_obj, attr).clicked.connect(self._autosave_config)
```

Add the method (near `_on_save_config`):

```python
    def _autosave_config(self) -> None:
        """Write the current config into the active folder (quiet) on each run."""
        if self._pos_dir is None:
            return
        self._save_config(str(self._pos_dir / "cellflow_config.json"), quiet=True)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/napari/test_main_widget_config_project.py -q -k "autosave or attr_exists"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/main_widget.py tests/napari/test_main_widget_config_project.py
git commit -m "feat(main-widget): autosave config into the folder on every run"
```

---

## Task 3: Save-side catalog adapter

**Files:**
- Modify: `src/cellflow/napari/main_widget.py` (add `_catalog_record_for_position`, module constants)
- Test: `tests/napari/test_main_widget_config_project.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from cellflow.contact_analysis.catalog import (
    CONTACT_ANALYSIS_RELPATH,
    REQUIRED_CSV_COLUMNS,
    load_catalog,
)


def test_catalog_record_stamps_default_paths(widget, tmp_path):
    pos = tmp_path / "WT" / "pos01"
    pos.mkdir(parents=True)
    rec = widget._catalog_record_for_position(
        pos, {"condition": "WT", "position_id": "pos01"}
    )
    assert Path(rec["position_path"]) == pos
    assert rec["contact_analysis_path"] == pos / CONTACT_ANALYSIS_RELPATH
    assert rec["cell_tracked_labels_path"] == pos / "3_cell" / "tracked_labels.tif"
    assert rec["nucleus_tracked_labels_path"] == pos / "2_nucleus" / "tracked_labels.tif"
    assert rec["columns"]["condition"] == "WT"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/napari/test_main_widget_config_project.py::test_catalog_record_stamps_default_paths -q`
Expected: FAIL (`_catalog_record_for_position` missing).

- [ ] **Step 3: Implement adapter + constants**

Near the top of `main_widget.py` (after imports), add the staged-layout constants
(single source of truth; `_refresh_all` may be updated to reuse them later, but do
not change its behavior in this task):

```python
from cellflow.contact_analysis.catalog import CONTACT_ANALYSIS_RELPATH

#: Staged-layout paths relative to a position folder, stamped onto catalog rows so
#: a project CSV saved from the full app carries everything the aggregate needs.
_CELL_LABELS_RELPATH = "3_cell/tracked_labels.tif"
_NUCLEUS_LABELS_RELPATH = "2_nucleus/tracked_labels.tif"
```

Add the method to `CellFlowMainWidget` (near the config helpers):

```python
    def _catalog_record_for_position(self, position_path: Path, columns: dict) -> dict:
        """A catalog record for one data folder, stamped with default stage paths.

        Panel rows carry only ``position_path`` + classification ``columns``; the
        aggregate catalog also needs the contact-analysis ``.h5`` and the two label
        images. Those sit at fixed staged-layout locations, so fill their defaults
        here (whether or not the folder has been processed yet).
        """
        pos = Path(position_path)
        return {
            "position_path": pos,
            "contact_analysis_path": pos / CONTACT_ANALYSIS_RELPATH,
            "cell_tracked_labels_path": pos / _CELL_LABELS_RELPATH,
            "nucleus_tracked_labels_path": pos / _NUCLEUS_LABELS_RELPATH,
            "columns": dict(columns or {}),
        }
```

Note: `main_widget.py` already imports from `cellflow.contact_analysis.catalog`
(`columns_from_levels`, `relative_levels`); add `CONTACT_ANALYSIS_RELPATH` to that
existing import line rather than adding a second import statement.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/napari/test_main_widget_config_project.py::test_catalog_record_stamps_default_paths -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/main_widget.py tests/napari/test_main_widget_config_project.py
git commit -m "feat(main-widget): catalog record adapter stamping default stage paths"
```

---

## Task 4: Save / Load project handlers

**Files:**
- Modify: `src/cellflow/napari/main_widget.py` (add `_on_save_project`, `_on_load_project`; import `save_catalog`/`load_catalog`/`merge_catalog_records`)
- Test: `tests/napari/test_main_widget_config_project.py`

- [ ] **Step 1: Write the failing tests**

Append (drives the handlers via patched file dialogs, exactly how the studio tests
its CSV load/save):

```python
def _seed_two_rows(widget, tmp_path):
    a = tmp_path / "WT" / "pos01"
    b = tmp_path / "KO" / "pos01"
    for p in (a, b):
        p.mkdir(parents=True)
    widget._positions_panel.set_records([
        {"key": str(a), "columns": {"condition": "WT", "position_id": "pos01"},
         "payload": {"position_path": a}},
        {"key": str(b), "columns": {"condition": "KO", "position_id": "pos01"},
         "payload": {"position_path": b}},
    ])
    return a, b


def test_save_project_writes_loadable_catalog(widget, tmp_path):
    _seed_two_rows(widget, tmp_path)
    csv_path = tmp_path / "catalog.csv"
    with patch(
        "cellflow.napari.main_widget.QFileDialog.getSaveFileName",
        return_value=(str(csv_path), ""),
    ):
        widget._on_save_project()
    assert csv_path.exists()
    loaded = load_catalog(csv_path)
    assert len(loaded) == 2
    for col in REQUIRED_CSV_COLUMNS:
        assert col in loaded[0]
    # The stamped default h5 path survives the round-trip.
    assert loaded[0]["contact_analysis_path"].name == "contact_analysis.h5"


def test_save_project_appends_csv_suffix(widget, tmp_path):
    _seed_two_rows(widget, tmp_path)
    chosen = tmp_path / "myproject"  # no extension
    with patch(
        "cellflow.napari.main_widget.QFileDialog.getSaveFileName",
        return_value=(str(chosen), ""),
    ):
        widget._on_save_project()
    assert (tmp_path / "myproject.csv").exists()


def test_load_project_repopulates_panel(widget, tmp_path):
    a, b = _seed_two_rows(widget, tmp_path)
    csv_path = tmp_path / "catalog.csv"
    with patch(
        "cellflow.napari.main_widget.QFileDialog.getSaveFileName",
        return_value=(str(csv_path), ""),
    ):
        widget._on_save_project()
    widget._positions_panel.set_records([])
    assert widget._positions_panel.keys() == []
    with patch(
        "cellflow.napari.main_widget.QFileDialog.getOpenFileName",
        return_value=(str(csv_path), ""),
    ):
        widget._on_load_project()
    assert len(widget._positions_panel.keys()) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/napari/test_main_widget_config_project.py -q -k project`
Expected: FAIL (`_on_save_project` / `_on_load_project` missing).

- [ ] **Step 3: Implement the handlers**

Extend the existing catalog import line in `main_widget.py` to add the three
functions:

```python
from cellflow.contact_analysis.catalog import (
    CONTACT_ANALYSIS_RELPATH,
    columns_from_levels,
    load_catalog,
    merge_catalog_records,
    relative_levels,
    save_catalog,
)
```

Add the handlers (near the config-file handlers):

```python
    def _on_save_project(self) -> None:
        """Write the Data folders catalog to a CSV the aggregate studio can run."""
        if not self._positions_panel.keys():
            show_warning("No data folders to save: the catalog is empty.")
            return
        path = QFileDialog.getSaveFileName(
            self, "Save project catalog", "catalog.csv", filter="CSV (*.csv)"
        )[0]
        if not path:
            return
        # getSaveFileName does not always append the filter suffix on Linux/Qt.
        if not path.lower().endswith(".csv"):
            path = f"{path}.csv"
        records = [
            self._catalog_record_for_position(
                rec["position_path"], rec.get("columns", {})
            )
            for rec in self._positions_panel.records()
        ]
        try:
            save_catalog(Path(path), records)
            show_info(f"Project saved to {path}")
        except Exception as e:  # noqa: BLE001 - surface any write/validation error
            show_error(f"Error saving project: {e}")

    def _on_load_project(self) -> None:
        """Load a project catalog CSV, merging its rows into the Data folders list."""
        path = QFileDialog.getOpenFileName(
            self, "Load project catalog", filter="CSV (*.csv)"
        )[0]
        if not path:
            return

        def action() -> None:
            try:
                loaded = load_catalog(Path(path))
            except Exception as e:  # noqa: BLE001
                show_error(f"Error loading project: {e}")
                return
            merged = merge_catalog_records(self._positions_panel.records(), loaded)
            entries = [
                {
                    "key": str(rec.get("position_path") or rec["id"]),
                    "columns": dict(rec.get("columns") or {}),
                    "payload": {"position_path": Path(rec["position_path"])}
                    if rec.get("position_path")
                    else {"position_path": None},
                }
                for rec in merged
            ]
            self._positions_panel.set_records(entries)
            show_info(f"Project loaded: {len(entries)} data folder(s).")

        self._change_context(action)
```

Note the `noqa: BLE001` comments only if the project's ruff config flags broad
excepts; otherwise omit them. Match the surrounding style.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/napari/test_main_widget_config_project.py -q -k project`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/main_widget.py tests/napari/test_main_widget_config_project.py
git commit -m "feat(main-widget): Save/Load project catalog CSV"
```

---

## Task 5: Rewire the toolbar

**Files:**
- Modify: `src/cellflow/napari/main_widget.py` (`_setup_project_ui`, `__init__` signal wiring, `_register_gate_controls`; remove `_on_save_config`, `_on_load_config`)
- Test: `tests/napari/test_main_widget_config_project.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_toolbar_has_project_and_params_pairs(widget):
    # Project group: select folder + load/save PROJECT (catalog CSV).
    assert hasattr(widget, "load_project_btn")
    assert hasattr(widget, "save_project_btn")
    # Params group: load/save CONFIG file.
    assert hasattr(widget, "load_from_btn")
    assert hasattr(widget, "save_as_btn")
    # The per-folder config buttons/handlers are gone.
    assert not hasattr(widget, "save_btn")
    assert not hasattr(widget, "load_btn")
    assert not hasattr(widget, "_on_save_config")
    assert not hasattr(widget, "_on_load_config")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/napari/test_main_widget_config_project.py::test_toolbar_has_project_and_params_pairs -q`
Expected: FAIL (`save_btn` / `_on_save_config` still present).

- [ ] **Step 3: Rewire `_setup_project_ui`**

Replace the Project + Params button block in `_setup_project_ui` (the section that
builds `project_btn`, `load_btn`, `save_btn`, then `load_from_btn`, `save_as_btn`)
with:

```python
        row.addWidget(self._toolbar_group_label("Project"))
        # ``new`` = open a single position folder (fallback front door). The
        # load/save pair here act on the PROJECT catalog CSV (the Data folders
        # list + classification columns), not the per-folder config.
        self.project_btn = self._toolbar_icon_btn("new", "Select a position folder…")
        self.load_project_btn = self._toolbar_icon_btn(
            "load", "Load a project catalog (CSV) into the Data folders list"
        )
        self.save_project_btn = self._toolbar_icon_btn(
            "save", "Save the Data folders list to a project catalog (CSV)"
        )
        for btn in (self.project_btn, self.load_project_btn, self.save_project_btn):
            row.addWidget(btn)

        row.addWidget(self._toolbar_group_label("Params"))
        # Per-folder config now autosaves on run; these move a tuned param SET
        # between experiments as a standalone file.
        self.load_from_btn = self._toolbar_icon_btn("load", "Load config from a file…")
        self.save_as_btn = self._toolbar_icon_btn("save", "Save config to a file…")
        for btn in (self.load_from_btn, self.save_as_btn):
            row.addWidget(btn)
```

- [ ] **Step 4: Update signal wiring in `__init__`**

In the "Connect signals" block, replace the config-button connections:

```python
        self.project_btn.clicked.connect(lambda: self._on_set_position_folder())
        self.save_as_btn.clicked.connect(lambda: self._on_save_config_as())
        self.load_from_btn.clicked.connect(lambda: self._on_load_config_from())
        self.save_project_btn.clicked.connect(lambda: self._on_save_project())
        self.load_project_btn.clicked.connect(lambda: self._on_load_project())
```

(Remove the old `self.save_btn.clicked...` and `self.load_btn.clicked...` lines.)

- [ ] **Step 5: Update `_register_gate_controls`**

Swap `self.load_btn` for `self.load_project_btn` in the `CONTEXT_CHANGING` tuple
(project load swaps the folder set, so it stays context-changing; config-file load
via `load_from_btn` already routes through `_change_context` in its handler):

```python
        for control in (
            self.project_btn,
            self.load_project_btn,
            self.load_from_btn,
        ):
            self.gate.register(control, ControlClass.CONTEXT_CHANGING)
```

- [ ] **Step 6: Delete the dead handlers**

Remove `_on_save_config` (~552) and `_on_load_config` (~566) entirely. Keep
`_on_save_config_as`, `_on_load_config_from`, `_save_config`, `_load_config`.

- [ ] **Step 7: Run the test + full file suite**

Run: `python -m pytest tests/napari/test_main_widget_config_project.py -q`
Expected: PASS (all tests in the file).

Run: `python -m pytest tests/napari/test_main_widget_cellpose_integration.py -q`
Expected: PASS (no regression; if it referenced `save_btn`/`load_btn`, update it to
the new names).

- [ ] **Step 8: Commit**

```bash
git add src/cellflow/napari/main_widget.py tests/napari/
git commit -m "feat(main-widget): toolbar = Project(catalog) + Params(config file); drop per-folder config buttons"
```

---

## Task 6: Fix the quickstart's "no project file" claim

**Files:**
- Modify: `src/cellflow/napari/_experiments_panel.py` (`_QUICKSTART_HTML`, ~line 90)
- Test: `tests/napari/test_experiments_panel.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/napari/test_experiments_panel.py` (import/fixtures as that file
already uses; it bootstraps Qt via `get_qapp()`):

```python
def test_quickstart_mentions_project_file():
    from cellflow.napari._experiments_panel import _QUICKSTART_HTML

    text = _QUICKSTART_HTML.lower()
    assert "no project file" not in text
    assert "project" in text and "catalog" in text
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/napari/test_experiments_panel.py::test_quickstart_mentions_project_file -q`
Expected: FAIL (`"no project file"` still present).

- [ ] **Step 3: Rewrite the first beat of `_QUICKSTART_HTML`**

Replace the opening `<h3>One folder per movie</h3>` paragraph (the one asserting
"CellFlow has no project file") with:

```python
_QUICKSTART_HTML = """
<h3>One folder per movie</h3>
<p>Each movie or field of view lives in its own <i>data folder</i> holding the raw
nucleus and cell images, and every stage writes its results back into that same
folder: the folder on disk is the source of truth for results. The list of data
folders and how you classify them (conditions, replicates) is your <i>project</i>,
which you save to and reload from a <b>project catalog</b> (a CSV). That catalog is
also what drives aggregate quantification across the whole set.</p>
```

Leave the "A worked example" and "Where results go" beats unchanged.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/napari/test_experiments_panel.py::test_quickstart_mentions_project_file -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/_experiments_panel.py tests/napari/test_experiments_panel.py
git commit -m "docs(quickstart): describe the project catalog instead of denying it exists"
```

---

## Task 7: Full-suite regression + end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Run the napari test suite**

Run: `python -m pytest tests/napari -q`
Expected: PASS. Fix any test that referenced the removed `save_btn`/`load_btn` or
`_on_save_config`/`_on_load_config`.

- [ ] **Step 2: Run ruff**

Run: `python -m ruff check src/cellflow/napari/main_widget.py src/cellflow/napari/_experiments_panel.py`
Expected: clean.

- [ ] **Step 3: End-to-end acceptance (the integration bar)**

This is the spec's acceptance criterion and cannot be proven by unit tests alone.
Write and run a headless driver script under the scratchpad that:

1. Builds two fake data folders with `columns` (condition WT/KO).
2. Instantiates `CellFlowMainWidget`, seeds the panel via `set_records`, and calls
   `_on_save_project` (patched dialog) to write `catalog.csv`.
3. Feeds that CSV to `cellflow.contact_analysis.catalog.load_catalog` AND to the
   aggregate studio's `_load_csv_from` path, asserting the rows load and each has a
   resolvable `contact_analysis_path` and unique identity (no `ValueError`).

Run: `python -m pytest tests/napari/test_main_widget_config_project.py tests/napari/test_contact_analysis_studio.py -q`

Expected: PASS. If the studio's loader rejects the full-app-saved CSV, that is a
real defect: fix the adapter (Task 3) so normalization produces a studio-loadable
record, do not weaken the assertion.

- [ ] **Step 4: Report**

Summarize: files changed, tests added/passing, and confirm the end-to-end
round-trip holds. Do not claim completion without the green suite output.
