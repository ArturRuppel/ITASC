# Curation Tool (napari) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A napari tool to author the curation exclusion table by eye — browse a series' positions over the contact-visualization display, and mark a frame or a whole position excluded with a reason — writing rows of the curation CSV the export pipeline already consumes.

**Architecture:** Three layers, most logic pushed below Qt so it stays unit-testable. (1) **Authoring ops** added to the existing pure `curation.py` (`write_curation` / `append_exclusion` / `remove_exclusion` / `empty_curation`). (2) A Qt-free **`CurationController`** that wraps a curation-CSV path, loads on construction, mutates via the authoring ops, and auto-saves after every change — the single seam the widget talks to. (3) A thin **`CurationWidget`** (an `AnalysisPlugin`) that embeds the existing contact-visualization view for the on-image display, reads the current frame from the napari viewer, and drives the controller from a reason field + two exclude buttons + an exclusions list.

**Tech Stack:** Python 3.11, pandas, pytest, qtpy/Qt (offscreen for tests), napari plugin framework.

---

## Background for the implementer (read once)

You have **zero prior context**. Key facts:

- **The curation backend already exists** at `src/cellflow/aggregate_quantification/curation.py`. It has `CURATION_COLUMNS = ("experiment_id", "position_id", "frame", "excluded", "exclusion_reason")` and the *read/apply/filter* functions `read_curation(path) -> DataFrame | None`, `apply_curation(table, curation)`, `filter_excluded(marked)`. It has **no writer / no authoring ops** — Task 1 adds them. The export pipeline already reads this CSV and filters excluded rows at `.iris` time, so anything this tool writes is automatically honoured by the next `run()`. **Do not change the read/apply/filter functions.**

- **Schema semantics:** one row per exclusion. `frame` empty/NA = the **whole position** (all frames); a numeric `frame` = that one frame. `excluded` is a bool, `exclusion_reason` a string. Keys are compared as **strings** for `experiment_id`/`position_id` and as **int** for `frame` (CSV round-trips ids as strings). `position_id` corresponds to a catalogue record's `id`; `experiment_id` to its `experiment_id`.

- **The napari analysis-plugin framework** (`src/cellflow/napari/aggregate_quantification/plugins/__init__.py`): subclass `AnalysisPlugin(QWidget)`, set class attrs `plugin_id` + `display_name`, override `set_context(self, ctx: AnalysisContext)`. Defining the subclass auto-registers it (via `__init_subclass__`); `available_analysis_plugins()` discovers it by importing every non-underscore module in the `plugins/` package. `AnalysisContext` carries `records: list[dict]` (normalized catalogue rows) and `viewer`.

- **The contact-visualization display already exists** as `VisualizeContactsPlugin` (`plugins/visualize_contacts.py`). It embeds `AggregateQuantificationWidget(viewer=..., standalone=False)` and drives it via `self._view.set_context(cell_labels=..., nucleus_labels=..., out_path=..., status_root=...)` for a single in-scope record. **Reuse this exact embedding pattern** for the curation display.

- **Current frame** is read from the napari viewer as `int(self.viewer.dims.current_step[0])` (see `nucleus_correction_widget.py:1134`). Guard for a missing/stub viewer in tests.

- **`catalogue_root(records)`** in `shape_tables.py` returns the common-ancestor directory of the in-scope positions — the natural default home for `curation.csv` (the aggregate tables land there too).

- **UI style helpers** in `cellflow.napari.ui_style`: `action_button(button, expand=False)` and `status_label(label, muted=False)` (used by the other plugins).

- **Test conventions for Qt widgets** (model file: `tests/napari/test_visualize_contacts_plugin.py`): set `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` at top, get an app via `QApplication.instance() or QApplication([])`, construct the plugin, `monkeypatch` collaborators, call `set_context(AnalysisContext(records=[...]))`, assert, then `plugin.deleteLater(); app.processEvents()`.

- **Always run tests with `--frozen`**: `uv run --frozen pytest ...` (the lockfile is intentionally stale; a bare `uv run` is unsatisfiable).

---

## File Structure

- **Modify** `src/cellflow/aggregate_quantification/curation.py` — add `empty_curation`, `write_curation`, `append_exclusion`, `remove_exclusion`; extend `__all__`.
- **Create** `src/cellflow/napari/aggregate_quantification/curation_controller.py` — `CurationController` (Qt-free).
- **Create** `src/cellflow/napari/aggregate_quantification/plugins/curation.py` — `CurationWidget(AnalysisPlugin)`.
- **Create** `tests/aggregate_quantification/test_curation_authoring.py` — Task 1 tests.
- **Create** `tests/napari/test_curation_controller.py` — Task 2 tests (Qt-free).
- **Create** `tests/napari/test_curation_plugin.py` — Task 3 tests (offscreen Qt).

---

## Task 1: Curation authoring ops (pure)

**Files:**
- Modify: `src/cellflow/aggregate_quantification/curation.py`
- Test: `tests/aggregate_quantification/test_curation_authoring.py` (create)

Add the write/append/remove operations the editor needs. All are pure (return new frames; never mutate the input); `frame=None` means whole-position (stored as NA). `append_exclusion` is **idempotent on the key** — it first removes any existing row with the same `(experiment_id, position_id, frame)` so re-excluding only updates the reason rather than duplicating.

- [ ] **Step 1: Write the failing tests**

Create `tests/aggregate_quantification/test_curation_authoring.py`:

```python
"""Authoring ops for the curation table — the write side the curation tool needs.

The read/apply/filter side is covered in test_curation.py; these cover building
and editing the tidy exclusion table: append (frame-level and whole-position),
idempotent re-exclude, remove, and the CSV round-trip.
"""
from __future__ import annotations

import pandas as pd

from cellflow.aggregate_quantification.curation import (
    CURATION_COLUMNS,
    append_exclusion,
    empty_curation,
    read_curation,
    remove_exclusion,
    write_curation,
)


def test_empty_curation_has_schema_and_no_rows():
    cur = empty_curation()
    assert tuple(cur.columns) == CURATION_COLUMNS
    assert len(cur) == 0


def test_append_frame_level_exclusion():
    cur = append_exclusion(
        empty_curation(),
        experiment_id="EXP1", position_id="p1", frame=3, reason="out of focus",
    )
    assert len(cur) == 1
    row = cur.iloc[0]
    assert row["experiment_id"] == "EXP1"
    assert row["position_id"] == "p1"
    assert int(row["frame"]) == 3
    assert bool(row["excluded"]) is True
    assert row["exclusion_reason"] == "out of focus"


def test_append_position_level_exclusion_stores_frame_na():
    cur = append_exclusion(
        empty_curation(),
        experiment_id="EXP1", position_id="p2", frame=None, reason="debris",
    )
    assert len(cur) == 1
    assert pd.isna(cur.iloc[0]["frame"])
    assert cur.iloc[0]["exclusion_reason"] == "debris"


def test_append_is_idempotent_on_key_updating_reason():
    cur = append_exclusion(empty_curation(),
                           experiment_id="EXP1", position_id="p1", frame=3, reason="first")
    cur = append_exclusion(cur,
                           experiment_id="EXP1", position_id="p1", frame=3, reason="second")
    assert len(cur) == 1  # same key -> replaced, not duplicated
    assert cur.iloc[0]["exclusion_reason"] == "second"


def test_append_frame_and_position_level_coexist():
    cur = append_exclusion(empty_curation(),
                           experiment_id="EXP1", position_id="p1", frame=3, reason="a")
    cur = append_exclusion(cur,
                           experiment_id="EXP1", position_id="p1", frame=None, reason="b")
    assert len(cur) == 2  # (p1, frame 3) and (p1, whole position) are distinct


def test_append_does_not_mutate_input():
    base = empty_curation()
    append_exclusion(base, experiment_id="EXP1", position_id="p1", frame=1, reason="x")
    assert len(base) == 0


def test_remove_frame_level_exclusion():
    cur = append_exclusion(empty_curation(),
                           experiment_id="EXP1", position_id="p1", frame=3, reason="a")
    cur = append_exclusion(cur,
                           experiment_id="EXP1", position_id="p1", frame=4, reason="b")
    out = remove_exclusion(cur, experiment_id="EXP1", position_id="p1", frame=3)
    assert len(out) == 1
    assert int(out.iloc[0]["frame"]) == 4


def test_remove_position_level_exclusion():
    cur = append_exclusion(empty_curation(),
                           experiment_id="EXP1", position_id="p1", frame=None, reason="a")
    cur = append_exclusion(cur,
                           experiment_id="EXP1", position_id="p1", frame=2, reason="b")
    out = remove_exclusion(cur, experiment_id="EXP1", position_id="p1", frame=None)
    # Only the whole-position row is removed; the frame-2 row remains.
    assert len(out) == 1
    assert int(out.iloc[0]["frame"]) == 2


def test_remove_missing_key_is_noop():
    cur = append_exclusion(empty_curation(),
                           experiment_id="EXP1", position_id="p1", frame=3, reason="a")
    out = remove_exclusion(cur, experiment_id="EXP1", position_id="zzz", frame=3)
    assert len(out) == 1


def test_write_then_read_round_trips(tmp_path):
    cur = append_exclusion(empty_curation(),
                           experiment_id="EXP1", position_id="p1", frame=None, reason="debris")
    cur = append_exclusion(cur,
                           experiment_id="EXP1", position_id="p2", frame=5, reason="blurry")
    path = tmp_path / "curation.csv"
    write_curation(path, cur)
    back = read_curation(path)
    assert back is not None
    assert set(back["position_id"].astype(str)) == {"p1", "p2"}
    # The whole-position row's frame is NA after the round-trip, not the string "".
    p1 = back[back["position_id"].astype(str) == "p1"].iloc[0]
    assert pd.isna(p1["frame"])


def test_write_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "curation.csv"
    write_curation(path, empty_curation())
    assert path.is_file()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --frozen pytest tests/aggregate_quantification/test_curation_authoring.py -v`
Expected: FAIL with ImportError (the new names don't exist yet).

- [ ] **Step 3: Implement the authoring ops**

In `curation.py`, extend `__all__`:

```python
__all__ = [
    "CURATION_COLUMNS",
    "empty_curation",
    "read_curation",
    "write_curation",
    "append_exclusion",
    "remove_exclusion",
    "apply_curation",
    "filter_excluded",
]
```

Add these functions (place them after `read_curation`, before `apply_curation`):

```python
def empty_curation() -> pd.DataFrame:
    """An empty curation table with the canonical columns (object dtype).

    Object columns keep an empty ``frame`` as ``NA`` (whole-position) rather than
    coercing the column to a float that would render NaN on write.
    """
    return pd.DataFrame({col: pd.Series(dtype="object") for col in CURATION_COLUMNS})


def write_curation(path: Path | str, curation: pd.DataFrame) -> None:
    """Write *curation* to *path* as CSV (creating the parent dir), index-free."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    curation.to_csv(path, index=False)


def append_exclusion(
    curation: pd.DataFrame | None,
    *,
    experiment_id: str,
    position_id: str,
    frame: int | None,
    reason: str,
) -> pd.DataFrame:
    """Return *curation* with one exclusion row added.

    ``frame=None`` records a **whole-position** exclusion (stored as NA); a numeric
    *frame* records a single-frame exclusion. Idempotent on the
    ``(experiment_id, position_id, frame)`` key: an existing row with the same key
    is replaced (its reason updated) rather than duplicated. The input is not
    mutated.
    """
    base = empty_curation() if curation is None else curation
    out = remove_exclusion(
        base, experiment_id=experiment_id, position_id=position_id, frame=frame
    )
    new_row = {
        "experiment_id": str(experiment_id),
        "position_id": str(position_id),
        "frame": pd.NA if frame is None else int(frame),
        "excluded": True,
        "exclusion_reason": str(reason),
    }
    return pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)


def remove_exclusion(
    curation: pd.DataFrame | None,
    *,
    experiment_id: str,
    position_id: str,
    frame: int | None,
) -> pd.DataFrame:
    """Return *curation* without the row(s) matching the given key.

    ``frame=None`` removes the **whole-position** row (``frame`` NA); a numeric
    *frame* removes that single-frame row. A non-matching key is a no-op. The
    input is not mutated.
    """
    if curation is None or len(curation) == 0:
        return empty_curation()
    out = curation.copy()
    key = (out["experiment_id"].astype(str) == str(experiment_id)) & (
        out["position_id"].astype(str) == str(position_id)
    )
    if frame is None:
        key &= out["frame"].isna()
    else:
        key &= out["frame"].notna() & (out["frame"].astype("float") == float(frame))
    return out.loc[~key].reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --frozen pytest tests/aggregate_quantification/test_curation_authoring.py tests/aggregate_quantification/test_curation.py -v`
Expected: PASS (the new authoring tests **and** the pre-existing read/apply/filter tests).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/aggregate_quantification/curation.py tests/aggregate_quantification/test_curation_authoring.py
git commit -m "feat(aggregate): add curation table authoring ops (write/append/remove)"
```

---

## Task 2: `CurationController` (Qt-free)

**Files:**
- Create: `src/cellflow/napari/aggregate_quantification/curation_controller.py`
- Test: `tests/napari/test_curation_controller.py` (create)

The controller is the single seam the widget talks to: it owns a curation-CSV path and an in-memory frame, loads on construction (empty if the file is absent), mutates through the Task 1 ops, and **auto-saves** after every change. It exposes per-position queries so the widget can render badges/lists without touching pandas. No Qt imports.

- [ ] **Step 1: Write the failing tests**

Create `tests/napari/test_curation_controller.py` (no QApplication needed — it is Qt-free):

```python
"""The Qt-free curation editing controller: load, mutate-and-autosave, query."""
from __future__ import annotations

import pandas as pd

from cellflow.napari.aggregate_quantification.curation_controller import (
    CurationController,
)
from cellflow.aggregate_quantification.curation import read_curation


def test_loads_empty_when_file_absent(tmp_path):
    ctrl = CurationController(tmp_path / "curation.csv")
    assert len(ctrl.curation) == 0


def test_loads_existing_file(tmp_path):
    path = tmp_path / "curation.csv"
    pd.DataFrame({
        "experiment_id": ["EXP1"], "position_id": ["p1"], "frame": [pd.NA],
        "excluded": [True], "exclusion_reason": ["debris"],
    }).to_csv(path, index=False)
    ctrl = CurationController(path)
    assert len(ctrl.curation) == 1


def test_exclude_frame_autosaves(tmp_path):
    path = tmp_path / "curation.csv"
    ctrl = CurationController(path)
    ctrl.exclude_frame(experiment_id="EXP1", position_id="p1", frame=3, reason="blurry")
    # Persisted immediately.
    back = read_curation(path)
    assert back is not None and len(back) == 1
    assert int(back.iloc[0]["frame"]) == 3


def test_exclude_position_autosaves_with_na_frame(tmp_path):
    path = tmp_path / "curation.csv"
    ctrl = CurationController(path)
    ctrl.exclude_position(experiment_id="EXP1", position_id="p1", reason="all bad")
    back = read_curation(path)
    assert pd.isna(back.iloc[0]["frame"])


def test_remove_autosaves(tmp_path):
    path = tmp_path / "curation.csv"
    ctrl = CurationController(path)
    ctrl.exclude_frame(experiment_id="EXP1", position_id="p1", frame=3, reason="x")
    ctrl.remove(experiment_id="EXP1", position_id="p1", frame=3)
    assert len(read_curation(path) or []) == 0


def test_exclusions_for_position(tmp_path):
    ctrl = CurationController(tmp_path / "curation.csv")
    ctrl.exclude_frame(experiment_id="EXP1", position_id="p1", frame=3, reason="x")
    ctrl.exclude_frame(experiment_id="EXP1", position_id="p2", frame=4, reason="y")
    got = ctrl.exclusions_for(experiment_id="EXP1", position_id="p1")
    assert list(got["position_id"].astype(str)) == ["p1"]


def test_is_frame_excluded(tmp_path):
    ctrl = CurationController(tmp_path / "curation.csv")
    ctrl.exclude_frame(experiment_id="EXP1", position_id="p1", frame=3, reason="x")
    assert ctrl.is_frame_excluded(experiment_id="EXP1", position_id="p1", frame=3)
    assert not ctrl.is_frame_excluded(experiment_id="EXP1", position_id="p1", frame=2)


def test_is_position_excluded(tmp_path):
    ctrl = CurationController(tmp_path / "curation.csv")
    ctrl.exclude_position(experiment_id="EXP1", position_id="p1", reason="x")
    assert ctrl.is_position_excluded(experiment_id="EXP1", position_id="p1")
    assert not ctrl.is_position_excluded(experiment_id="EXP1", position_id="p2")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --frozen pytest tests/napari/test_curation_controller.py -v`
Expected: FAIL (module/class does not exist).

- [ ] **Step 3: Implement the controller**

Create `src/cellflow/napari/aggregate_quantification/curation_controller.py`:

```python
"""The Qt-free editing controller behind the napari curation tool.

It is the single seam the :class:`CurationWidget` talks to: it owns the curation
CSV path, loads it on construction (empty when absent), mutates it through the
pure authoring ops in :mod:`cellflow.aggregate_quantification.curation`, and
**auto-saves** after every change so the table is always the source of truth. It
exposes per-position queries so the widget renders badges/lists without touching
pandas. Keeping it Qt-free keeps the table logic unit-testable headless.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellflow.aggregate_quantification.curation import (
    append_exclusion,
    empty_curation,
    read_curation,
    remove_exclusion,
    write_curation,
)


class CurationController:
    """Load → mutate-and-autosave → query over one curation CSV."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        loaded = read_curation(self.path)
        self.curation = empty_curation() if loaded is None else loaded

    # --------------------------------------------------------------- mutations
    def exclude_frame(
        self, *, experiment_id: str, position_id: str, frame: int, reason: str
    ) -> None:
        self.curation = append_exclusion(
            self.curation,
            experiment_id=experiment_id,
            position_id=position_id,
            frame=int(frame),
            reason=reason,
        )
        self._save()

    def exclude_position(
        self, *, experiment_id: str, position_id: str, reason: str
    ) -> None:
        self.curation = append_exclusion(
            self.curation,
            experiment_id=experiment_id,
            position_id=position_id,
            frame=None,
            reason=reason,
        )
        self._save()

    def remove(
        self, *, experiment_id: str, position_id: str, frame: int | None
    ) -> None:
        self.curation = remove_exclusion(
            self.curation,
            experiment_id=experiment_id,
            position_id=position_id,
            frame=None if frame is None else int(frame),
        )
        self._save()

    # ----------------------------------------------------------------- queries
    def exclusions_for(self, *, experiment_id: str, position_id: str) -> pd.DataFrame:
        """The exclusion rows for one position (reset index), for the widget list."""
        cur = self.curation
        key = (cur["experiment_id"].astype(str) == str(experiment_id)) & (
            cur["position_id"].astype(str) == str(position_id)
        )
        return cur.loc[key].reset_index(drop=True)

    def is_position_excluded(self, *, experiment_id: str, position_id: str) -> bool:
        rows = self.exclusions_for(experiment_id=experiment_id, position_id=position_id)
        return bool(rows["frame"].isna().any())

    def is_frame_excluded(
        self, *, experiment_id: str, position_id: str, frame: int
    ) -> bool:
        rows = self.exclusions_for(experiment_id=experiment_id, position_id=position_id)
        if rows.empty:
            return False
        # A whole-position exclusion covers every frame, too.
        if rows["frame"].isna().any():
            return True
        return bool((rows["frame"].dropna().astype("float") == float(frame)).any())

    # -------------------------------------------------------------------- save
    def _save(self) -> None:
        write_curation(self.path, self.curation)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --frozen pytest tests/napari/test_curation_controller.py -v`
Expected: PASS (all 8).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_quantification/curation_controller.py tests/napari/test_curation_controller.py
git commit -m "feat(napari): add Qt-free CurationController (load/mutate/autosave)"
```

---

## Task 3: `CurationWidget` plugin

**Files:**
- Create: `src/cellflow/napari/aggregate_quantification/plugins/curation.py`
- Test: `tests/napari/test_curation_plugin.py` (create)

The plugin is the Qt glue: embed the contact-viz display, expose a curation-path field (defaulting to `<catalogue_root>/curation.csv`), a reason field, two exclude buttons (enabled only for a single in-scope position with a non-empty reason), and a per-position exclusions list with a remove button. It reads the current frame from the napari viewer. All table logic lives in the controller — the widget only collects inputs and calls it.

Design contract the tests pin down:
- `plugin_id = "curation"`, `display_name = "Curation"`; registered/discoverable.
- `set_context` with **one** record: store `experiment_id` (`record["experiment_id"]`) + `position_id` (`record["id"]`), (re)build the controller pointed at `<catalogue_root>/curation.csv` (or the path field if the user set one), drive the embedded display, and refresh the exclusions list + button enablement. With **zero or many** records: disable the exclude buttons and clear the display (ambiguous for one viewer).
- Exclude buttons are disabled when the reason field is empty; enabled when a single position is in scope and the reason is non-empty.
- "Exclude this frame" calls `controller.exclude_frame(experiment_id, position_id, frame=<viewer current frame>, reason=<reason text>)`.
- "Exclude this position" calls `controller.exclude_position(experiment_id, position_id, reason=<reason text>)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/napari/test_curation_plugin.py`:

```python
"""The napari curation tool widget — Qt glue over the CurationController."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from cellflow.napari.aggregate_quantification.plugins import (
    AnalysisContext,
    available_analysis_plugins,
)
from cellflow.napari.aggregate_quantification.plugins.curation import CurationWidget


def _app():
    return QApplication.instance() or QApplication([])


class _FakeDims:
    def __init__(self, frame):
        self.current_step = (frame, 0, 0)


class _FakeViewer:
    def __init__(self, frame=0):
        self.dims = _FakeDims(frame)


def _record(tmp_path, pid="p1"):
    pdir = tmp_path / "study" / pid
    pdir.mkdir(parents=True, exist_ok=True)
    return {
        "id": pid,
        "experiment_id": "EXP1",
        "position_path": pdir,
        "cell_tracked_labels_path": None,
        "nucleus_tracked_labels_path": None,
        "contact_analysis_path": None,
    }


def test_curation_plugin_is_registered():
    assert CurationWidget in available_analysis_plugins()
    assert "curation" in {cls.plugin_id for cls in available_analysis_plugins()}


def test_single_position_enables_actions_only_with_reason(tmp_path, monkeypatch):
    app = _app()
    plugin = CurationWidget(viewer=_FakeViewer(frame=2))
    # Don't exercise the embedded display in this unit test.
    monkeypatch.setattr(plugin, "_update_display", lambda record: None)

    plugin.set_context(AnalysisContext(records=[_record(tmp_path)]))
    # No reason yet -> disabled.
    assert not plugin._exclude_frame_btn.isEnabled()
    assert not plugin._exclude_position_btn.isEnabled()
    # Reason present -> enabled.
    plugin._reason_edit.setText("out of focus")
    assert plugin._exclude_frame_btn.isEnabled()
    assert plugin._exclude_position_btn.isEnabled()

    plugin.deleteLater()
    app.processEvents()


def test_zero_or_many_positions_disables_actions(tmp_path, monkeypatch):
    app = _app()
    plugin = CurationWidget(viewer=_FakeViewer())
    monkeypatch.setattr(plugin, "_update_display", lambda record: None)
    plugin._reason_edit.setText("x")

    plugin.set_context(AnalysisContext(records=[]))
    assert not plugin._exclude_frame_btn.isEnabled()

    plugin.set_context(AnalysisContext(records=[_record(tmp_path, "p1"),
                                               _record(tmp_path, "p2")]))
    assert not plugin._exclude_frame_btn.isEnabled()

    plugin.deleteLater()
    app.processEvents()


def test_exclude_frame_calls_controller_with_current_frame(tmp_path, monkeypatch):
    app = _app()
    plugin = CurationWidget(viewer=_FakeViewer(frame=5))
    monkeypatch.setattr(plugin, "_update_display", lambda record: None)
    plugin.set_context(AnalysisContext(records=[_record(tmp_path)]))
    plugin._reason_edit.setText("blurry")

    calls = []
    monkeypatch.setattr(plugin._controller, "exclude_frame",
                        lambda **kw: calls.append(kw))
    plugin._exclude_frame_btn.click()

    assert calls == [{"experiment_id": "EXP1", "position_id": "p1",
                      "frame": 5, "reason": "blurry"}]

    plugin.deleteLater()
    app.processEvents()


def test_exclude_position_calls_controller(tmp_path, monkeypatch):
    app = _app()
    plugin = CurationWidget(viewer=_FakeViewer())
    monkeypatch.setattr(plugin, "_update_display", lambda record: None)
    plugin.set_context(AnalysisContext(records=[_record(tmp_path)]))
    plugin._reason_edit.setText("all bad")

    calls = []
    monkeypatch.setattr(plugin._controller, "exclude_position",
                        lambda **kw: calls.append(kw))
    plugin._exclude_position_btn.click()

    assert calls == [{"experiment_id": "EXP1", "position_id": "p1",
                      "reason": "all bad"}]

    plugin.deleteLater()
    app.processEvents()


def test_exclude_frame_writes_through_to_csv(tmp_path, monkeypatch):
    """End-to-end through the real controller: the action lands in the CSV."""
    app = _app()
    plugin = CurationWidget(viewer=_FakeViewer(frame=7))
    monkeypatch.setattr(plugin, "_update_display", lambda record: None)
    plugin.set_context(AnalysisContext(records=[_record(tmp_path)]))
    plugin._reason_edit.setText("blurry")
    plugin._exclude_frame_btn.click()

    from cellflow.aggregate_quantification.curation import read_curation
    csv_path = tmp_path / "study" / "curation.csv"
    back = read_curation(csv_path)
    assert back is not None and len(back) == 1
    assert int(back.iloc[0]["frame"]) == 7

    plugin.deleteLater()
    app.processEvents()
```

> The end-to-end test relies on the controller defaulting its path to `<catalogue_root>/curation.csv`. With a single position at `tmp_path/study/p1`, `catalogue_root` resolves to `tmp_path/study`. If `catalogue_root` resolves elsewhere for a single position, read the path the widget actually used (`plugin._controller.path`) instead of hard-coding — but prefer making the default land at `<catalogue_root>/curation.csv`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --frozen pytest tests/napari/test_curation_plugin.py -v`
Expected: FAIL (module/class does not exist).

- [ ] **Step 3: Implement the widget**

Create `src/cellflow/napari/aggregate_quantification/plugins/curation.py`:

```python
"""Curation tool: author the exclusion table by eye, over the contact display.

The image-linked judgement ("scrub through a position, see a bad frame, exclude
it with a note") is what re-earns napari for Aggregate Quantification. This
plugin embeds the contact-visualization display for one selected position and
turns exclude actions into rows of the curation CSV — via the Qt-free
:class:`~cellflow.napari.aggregate_quantification.curation_controller.CurationController`,
which auto-saves so the table is always the source of truth. No plots live here
(Iris owns plotting).
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.shape_tables import catalogue_root
from cellflow.napari.aggregate_quantification.curation_controller import (
    CurationController,
)
from cellflow.napari.aggregate_quantification.plugins import (
    AnalysisContext,
    AnalysisPlugin,
)
from cellflow.napari.ui_style import action_button, status_label


class CurationWidget(AnalysisPlugin):
    """Mark a frame / a whole position excluded with a reason, over the image."""

    plugin_id = "curation"
    display_name = "Curation"
    # No ``requires``: a row may carry only a loose .h5 (show an existing result)
    # or labels (compute-on-demand); the embedded display self-handles missing
    # inputs, mirroring VisualizeContactsPlugin.

    def __init__(self, viewer=None, parent: QWidget | None = None) -> None:
        super().__init__(viewer=viewer, parent=parent)
        # Lazy import keeps the plugin module import cheap and avoids an import
        # cycle with the widget module.
        from cellflow.napari.aggregate_quantification_widget import (
            AggregateQuantificationWidget,
        )

        self._experiment_id: str | None = None
        self._position_id: str | None = None
        self._controller: CurationController | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self._scope_lbl = QLabel("Select a single position to curate.")
        self._scope_lbl.setWordWrap(True)
        layout.addWidget(self._scope_lbl)

        # Curation file (defaults to <catalogue root>/curation.csv on scope).
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("curation.csv (defaults beside the tables)…")
        self._path_edit.editingFinished.connect(self._on_path_edited)
        layout.addLayout(self._labelled_row("Curation file:", self._path_edit))

        # The embedded contact-visualization display (the curation canvas).
        self._view = AggregateQuantificationWidget(viewer=viewer, standalone=False)
        self._view.pipeline_files_header.setVisible(False)
        self._view._pipeline_files_section.setVisible(False)
        layout.addWidget(self._view, 1)

        # Reason + actions.
        self._reason_edit = QLineEdit()
        self._reason_edit.setPlaceholderText("Reason (required)…")
        self._reason_edit.textChanged.connect(self._update_enabled)
        layout.addLayout(self._labelled_row("Reason:", self._reason_edit))

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)
        self._exclude_frame_btn = QPushButton("Exclude this frame")
        action_button(self._exclude_frame_btn, expand=True)
        self._exclude_frame_btn.clicked.connect(self._on_exclude_frame)
        self._exclude_position_btn = QPushButton("Exclude this position")
        action_button(self._exclude_position_btn, expand=True)
        self._exclude_position_btn.clicked.connect(self._on_exclude_position)
        actions.addWidget(self._exclude_frame_btn)
        actions.addWidget(self._exclude_position_btn)
        layout.addLayout(actions)

        # Current exclusions for the in-scope position + a remove button.
        layout.addWidget(QLabel("Exclusions for this position:"))
        self._exclusions_list = QListWidget()
        self._exclusions_list.setMaximumHeight(120)
        layout.addWidget(self._exclusions_list)
        self._remove_btn = QPushButton("Remove selected")
        action_button(self._remove_btn)
        self._remove_btn.clicked.connect(self._on_remove_selected)
        layout.addWidget(self._remove_btn)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        status_label(self._status_lbl)
        layout.addWidget(self._status_lbl)

        self._update_enabled()

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _labelled_row(label: str, edit: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        lbl = QLabel(label)
        lbl.setFixedWidth(90)
        row.addWidget(lbl)
        row.addWidget(edit, 1)
        return row

    def _current_frame(self) -> int:
        dims = getattr(self.viewer, "dims", None)
        step = getattr(dims, "current_step", (0,))
        return int(step[0]) if step else 0

    def _has_single_position(self) -> bool:
        return self._position_id is not None and self._controller is not None

    def _reason(self) -> str:
        return self._reason_edit.text().strip()

    # -------------------------------------------------------------- plugin API
    def set_context(self, ctx: AnalysisContext) -> None:
        if ctx.viewer is not None:
            self.viewer = ctx.viewer
        records = list(ctx.records)
        if len(records) == 1:
            record = records[0]
            self._experiment_id = str(record.get("experiment_id", ""))
            self._position_id = str(record.get("id", ""))
            self._ensure_controller(records)
            self._update_display(record)
            self._scope_lbl.setText(f"Curating position: {self._position_id}")
        else:
            self._experiment_id = None
            self._position_id = None
            self._controller = None
            self._clear_display()
            self._scope_lbl.setText(
                "Select a single position to curate."
                if not records
                else f"{len(records)} positions selected — pick exactly one to curate."
            )
        self._refresh_exclusions()
        self._update_enabled()

    def _ensure_controller(self, records: list[dict]) -> None:
        """(Re)build the controller for the curation path (field, else default)."""
        text = self._path_edit.text().strip()
        if text:
            path = Path(text)
        else:
            path = catalogue_root(records) / "curation.csv"
            self._path_edit.setText(str(path))
        self._controller = CurationController(path)

    def _on_path_edited(self) -> None:
        text = self._path_edit.text().strip()
        if text:
            self._controller = CurationController(Path(text))
            self._refresh_exclusions()
            self._update_enabled()

    def _update_display(self, record: dict) -> None:
        self._view.set_context(
            cell_labels=record.get("cell_tracked_labels_path"),
            nucleus_labels=record.get("nucleus_tracked_labels_path"),
            out_path=record.get("contact_analysis_path"),
            status_root=record.get("position_path"),
        )

    def _clear_display(self) -> None:
        self._view.set_context(
            cell_labels=None, nucleus_labels=None, out_path=None, status_root=None
        )

    # -------------------------------------------------------------- enablement
    def _update_enabled(self) -> None:
        can_act = self._has_single_position() and bool(self._reason())
        self._exclude_frame_btn.setEnabled(can_act)
        self._exclude_position_btn.setEnabled(can_act)
        self._remove_btn.setEnabled(self._has_single_position())

    # ----------------------------------------------------------------- actions
    def _on_exclude_frame(self) -> None:
        if not self._has_single_position() or not self._reason():
            return
        frame = self._current_frame()
        self._controller.exclude_frame(
            experiment_id=self._experiment_id,
            position_id=self._position_id,
            frame=frame,
            reason=self._reason(),
        )
        self._status_lbl.setText(f"Status: excluded frame {frame}.")
        self._refresh_exclusions()

    def _on_exclude_position(self) -> None:
        if not self._has_single_position() or not self._reason():
            return
        self._controller.exclude_position(
            experiment_id=self._experiment_id,
            position_id=self._position_id,
            reason=self._reason(),
        )
        self._status_lbl.setText("Status: excluded whole position.")
        self._refresh_exclusions()

    def _on_remove_selected(self) -> None:
        if not self._has_single_position():
            return
        item = self._exclusions_list.currentItem()
        if item is None:
            return
        frame = item.data(256)  # Qt.UserRole; None for a whole-position row
        self._controller.remove(
            experiment_id=self._experiment_id,
            position_id=self._position_id,
            frame=frame,
        )
        self._status_lbl.setText("Status: removed exclusion.")
        self._refresh_exclusions()

    def _refresh_exclusions(self) -> None:
        self._exclusions_list.clear()
        if not self._has_single_position():
            return
        rows = self._controller.exclusions_for(
            experiment_id=self._experiment_id, position_id=self._position_id
        )
        from qtpy.QtWidgets import QListWidgetItem

        for _, row in rows.iterrows():
            frame = row["frame"]
            if _is_na(frame):
                text = f"whole position — {row['exclusion_reason']}"
                payload = None
            else:
                text = f"frame {int(frame)} — {row['exclusion_reason']}"
                payload = int(frame)
            item = QListWidgetItem(text)
            item.setData(256, payload)  # Qt.UserRole
            self._exclusions_list.addItem(item)


def _is_na(value) -> bool:
    import pandas as pd

    return pd.isna(value)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --frozen pytest tests/napari/test_curation_plugin.py -v`
Expected: PASS (all 6).

If a test fails because `catalogue_root` resolves the single-position default somewhere other than `tmp_path/study`, debug by printing `plugin._controller.path`; adjust the widget so the default is `<catalogue_root>/curation.csv` (do not weaken the test). If `action_button`/`status_label` signatures differ from what's used here, check `cellflow/napari/ui_style.py` and the other plugins (e.g. the old `visualize_contacts.py`) for the exact call form.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_quantification/plugins/curation.py tests/napari/test_curation_plugin.py
git commit -m "feat(napari): add Curation tool widget (author exclusions over contact display)"
```

---

## Final review

After all three tasks:
- Run `uv run --frozen pytest tests/aggregate_quantification/ tests/napari/ -q`. Expected: green except the **3 pre-existing, unrelated** failures (`test_aggregate_area.py::test_status_reflects_written_table`, `test_plots_contacts.py::test_typed_views_empty_for_unclassified_positions`, `test_plots_contacts.py::test_signed_contact_length_pool_handles_no_t1` — a pandas `df.insert` multi-column issue that predates this branch). Confirm no *new* failures.
- Confirm `CurationWidget` appears in `available_analysis_plugins()` alongside the others.
- Confirm the read/apply/filter functions in `curation.py` are unchanged (only additions).
- Confirm the controller has **no Qt imports** (the table logic stays headless-testable).

## Notes on scope (deferred — do NOT implement)

Per the design spec's "Open / deferred", these are intentionally out of this slice:
- **Frame-range exclusion gesture** (select start/end on the scrubber → many per-frame rows). This slice does single-frame + whole-position only; the per-frame row schema already supports a range being added later without change.
- **Explicit Save / undo stack** — auto-save is the chosen behaviour (in-session remove is the undo).
- **Per-position summary numbers / any plots** — image-only display (Iris owns plotting).
- **Multi-channel / 3D display choices** — inherited from the embedded contact-viz loader; no new decisions.
- **Surrounding-widget slimming** (discover&add + run colocated with curation) belongs to the separate napari front-end refocus plan, which must run *after* this tool is proven.
