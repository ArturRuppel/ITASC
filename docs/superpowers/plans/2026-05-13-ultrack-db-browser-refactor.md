# Ultrack DB Browser Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the Ultrack database browser logic out of `NucleusWorkflowWidget` into testable non-Qt modules while preserving current user-visible behavior, auditing stale tests, and fixing the source-slider/cache bug with explicit coverage.

**Architecture:** Keep `NucleusWorkflowWidget` as the UI coordinator for controls, napari layers, and status text. Move Ultrack SQLAlchemy queries, hierarchy-cut state calculation, source filtering, annotation filtering, preview metadata, and connected-focus computations into focused modules under `cellflow.tracking_ultrack`. Introduce characterization tests before extraction, then replace widget internals incrementally.

**Tech Stack:** Python, NumPy, SQLAlchemy, Ultrack `NodeDB`/`LinkDB`, pytest, qtpy/napari tests where UI wiring is unavoidable.

---

## File Structure

- Create `src/cellflow/tracking_ultrack/db_browser.py`
  - Own SQLAlchemy read helpers for Ultrack `data.db`.
  - Own hierarchy/source/summary query functions.
  - Return plain dataclasses and built-in Python containers.
- Create `src/cellflow/tracking_ultrack/db_browser_preview.py`
  - Own pure preview computations: hierarchy cut states, annotation filtering, label/node metadata, connected-focus filtering, alpha calculations.
  - Avoid Qt and napari imports.
- Modify `src/cellflow/napari/nucleus_workflow_widget.py`
  - Replace inline Ultrack DB query and preview computation methods with calls into the new modules.
  - Keep layer updates, widget controls, status labels, and mouse callbacks here.
- Add `tests/tracking_ultrack/test_db_browser.py`
  - Cover SQL/query-facing helpers with monkeypatched sessions or small sqlite fixtures where practical.
- Add `tests/tracking_ultrack/test_db_browser_preview.py`
  - Cover pure hierarchy, filtering, metadata, and alpha behavior without Qt.
- Modify `tests/napari/test_nucleus_tracking_correction_layout.py`
  - Keep only thin widget wiring tests for controls, refresh calls, and layer integration.
- Audit existing relevant tests:
  - `tests/database/test_validation.py`
  - `tests/tracking_ultrack/test_db_build.py`
  - `tests/tracking_ultrack/test_solve.py`
  - DB-browser-related tests in `tests/napari/test_nucleus_tracking_correction_layout.py`

## Agent Strategy

- Use cheap/dumb agents for read-only reconnaissance and test classification.
- Use worker agents only after this plan is split into exact file ownership.
- Do not let multiple agents edit `src/cellflow/napari/nucleus_workflow_widget.py` concurrently.
- Good parallel scopes:
  - Agent A: classify relevant tests as keep/rewrite/delete.
  - Agent B: inventory current DB-browser widget methods and state fields.
  - Agent C: draft pure preview tests for hierarchy/source/annotation behavior.
  - Agent D: inspect docs/specs for intended source-slider and hierarchy-browser behavior.

## Task 1: Test Audit And Behavior Inventory

**Files:**
- Create: `docs/superpowers/plans/2026-05-13-ultrack-db-browser-test-audit.md`
- Read only: `src/cellflow/napari/nucleus_workflow_widget.py`
- Read only: `tests/database/test_validation.py`
- Read only: `tests/tracking_ultrack/test_db_build.py`
- Read only: `tests/tracking_ultrack/test_solve.py`
- Read only: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Inventory DB-browser methods and state**

Record every widget method and state field whose name starts with `_ultrack_db`, plus DB-browser controls named `ultrack_db_*`.

Use this command:

```bash
rg -n "(_ultrack_db|ultrack_db_)" src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
```

Expected: a method/control inventory covering refresh, source slider, hierarchy slider, preview cache, annotation filters, connected focus, metadata, and layer updates.

- [ ] **Step 2: Classify relevant tests**

Create `docs/superpowers/plans/2026-05-13-ultrack-db-browser-test-audit.md` with this exact structure:

```markdown
# Ultrack DB Browser Test Audit

## Keep

- `tests/database/test_validation.py`: keep. It documents the JSON schema and read/write behavior for validation metadata.
- `tests/tracking_ultrack/test_db_build.py`: keep. It documents DB build sequencing and validated-node integration contracts.
- `tests/tracking_ultrack/test_solve.py`: keep. It documents annotation-aware solve behavior.

## Rewrite

- `tests/napari/test_nucleus_tracking_correction_layout.py`: rewrite DB-browser tests that assert complex computation through the widget. Move pure computation expectations to `tests/tracking_ultrack/test_db_browser_preview.py`, leaving only UI wiring in napari tests.

## Delete Candidates

- None by default. Delete only tests for features confirmed removed from the current widget or tests that assert behavior contradicted by the new design and have replacement coverage in the new pure test files.

## Characterization Targets

- Source-slider selection must be part of hierarchy-state cache identity.
- Refresh must not reset a user's selected source unless the selected source no longer exists.
- Annotation filters must hide `REAL` and `FAKE` nodes according to the existing checkbox semantics.
- Connected focus must preserve selected `NodeDB.id` across render-local display-label changes.
```

- [ ] **Step 3: Run the existing focused test set**

Run:

```bash
pytest tests/database/test_validation.py tests/tracking_ultrack/test_db_build.py tests/tracking_ultrack/test_solve.py tests/napari/test_nucleus_tracking_correction_layout.py -q
```

Expected: either PASS, or failures documented in the audit file under a new `## Current Failures` section with the exact failing test names. Do not delete failing tests in this task.

- [ ] **Step 4: Commit the audit only**

```bash
git add docs/superpowers/plans/2026-05-13-ultrack-db-browser-test-audit.md
git commit -m "docs: audit ultrack db browser tests"
```

## Task 2: Characterization Tests For Current Widget Behavior

**Files:**
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Add a source-cache characterization test**

Add a test showing that hierarchy states differ by selected source and that the cache key must include source selection. If the current implementation fails this test, mark it as the bug to fix during extraction.

```python
def test_ultrack_db_hierarchy_cache_distinguishes_source_selection(monkeypatch, tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "data.db"
    db_path.write_bytes(b"placeholder")
    mtime_ns = 123
    frame = 5

    calls = []

    class FakeSlider:
        def __init__(self):
            self._value = 0
            self._maximum = 2
        def value(self):
            return self._value
        def maximum(self):
            return self._maximum

    widget.ultrack_db_source_slider = FakeSlider()

    def fake_query_for_source(source):
        calls.append(source)
        return (source,)

    monkeypatch.setattr(
        widget,
        "_query_hierarchy_cut_states_uncached_for_test",
        fake_query_for_source,
        raising=False,
    )

    # This test should be adapted to the current implementation if needed:
    # first call source 0, second call source 1, both must compute independently.
    widget.ultrack_db_source_slider._value = 0
    first = widget._query_hierarchy_cut_states(db_path, mtime_ns, frame)
    widget.ultrack_db_source_slider._value = 1
    second = widget._query_hierarchy_cut_states(db_path, mtime_ns, frame)

    assert first != second
    assert calls == [0, 1]

    widget.deleteLater()
    viewer.close()
```

If the widget cannot be tested cleanly with this hook, replace the hook with a smaller test after extracting `HierarchyQueryKey` in Task 3. Keep the behavioral assertion: source selection belongs in the cache key.

- [ ] **Step 2: Add a source-slider preservation characterization test**

Add a test proving refresh preserves a valid user-selected source:

```python
def test_ultrack_db_source_slider_preserves_valid_selection(monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.ultrack_db_source_slider.setRange(0, 3)
    widget.ultrack_db_source_slider.setValue(2)
    monkeypatch.setattr(widget, "_query_available_sources", lambda *_args: (0, 1, 2, 3))

    widget._configure_ultrack_db_source_slider(Path("/tmp/data.db"), 1)

    assert widget.ultrack_db_source_slider.value() == 2
    assert widget.ultrack_db_source_lbl.text() == "2/3"

    widget.deleteLater()
    viewer.close()
```

Expected today: likely FAIL if `_configure_ultrack_db_source_slider` resets to zero.

- [ ] **Step 3: Run the two characterization tests**

Run:

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_source_slider_preserves_valid_selection -q
```

Expected: FAIL before the bug fix if the current widget resets the slider.

- [ ] **Step 4: Commit only the characterization tests**

```bash
git add tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "test: characterize ultrack db browser source handling"
```

## Task 3: Extract Pure Preview Types And Hierarchy Cut State

**Files:**
- Create: `src/cellflow/tracking_ultrack/db_browser_preview.py`
- Create: `tests/tracking_ultrack/test_db_browser_preview.py`

- [ ] **Step 1: Write pure tests for hierarchy state calculation**

Create `tests/tracking_ultrack/test_db_browser_preview.py` with:

```python
from __future__ import annotations

from cellflow.tracking_ultrack.db_browser_preview import (
    HierarchyNodeRow,
    compute_hierarchy_cut_states,
)


def test_compute_hierarchy_cut_states_promotes_children_to_parent():
    rows = [
        HierarchyNodeRow(node_id=1, parent_id=-1, height=2.0),
        HierarchyNodeRow(node_id=2, parent_id=1, height=1.0),
        HierarchyNodeRow(node_id=3, parent_id=1, height=1.0),
    ]

    states = compute_hierarchy_cut_states(rows, no_parent=-1)

    assert [state.node_ids for state in states] == [(2, 3), (1,)]
    assert [state.height for state in states] == [1.0, 2.0]


def test_compute_hierarchy_cut_states_handles_empty_rows():
    assert compute_hierarchy_cut_states([], no_parent=-1) == ()
```

- [ ] **Step 2: Run tests and verify red**

```bash
pytest tests/tracking_ultrack/test_db_browser_preview.py -q
```

Expected: FAIL because `db_browser_preview.py` does not exist.

- [ ] **Step 3: Implement dataclasses and pure hierarchy function**

Create `src/cellflow/tracking_ultrack/db_browser_preview.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class HierarchyNodeRow:
    node_id: int
    parent_id: int
    height: float


@dataclass(frozen=True)
class HierarchyCutState:
    node_ids: tuple[int, ...]
    height: float | None


def compute_hierarchy_cut_states(
    rows: Iterable[HierarchyNodeRow],
    *,
    no_parent: int,
) -> tuple[HierarchyCutState, ...]:
    materialized = tuple(rows)
    if not materialized:
        return ()

    node_ids = {row.node_id for row in materialized}
    heights_by_id = {row.node_id: row.height for row in materialized}
    parent_by_id = {
        row.node_id: row.parent_id
        for row in materialized
        if row.parent_id != no_parent and row.parent_id in node_ids
    }
    children: dict[int, set[int]] = {}
    for child_id, parent_id in parent_by_id.items():
        children.setdefault(parent_id, set()).add(child_id)

    active = {row.node_id for row in materialized if row.node_id not in children}
    if not active:
        active = set(node_ids)

    states: list[HierarchyCutState] = []
    seen: set[tuple[int, ...]] = set()

    def append_state() -> None:
        ordered = tuple(sorted(active, key=lambda node_id: (heights_by_id[node_id], node_id)))
        if ordered in seen:
            return
        seen.add(ordered)
        height = max((heights_by_id[node_id] for node_id in ordered), default=None)
        states.append(HierarchyCutState(ordered, height))

    append_state()
    while True:
        promotable = [
            parent_id
            for parent_id, child_ids in children.items()
            if parent_id not in active and child_ids and child_ids.issubset(active)
        ]
        if not promotable:
            break
        min_height = min(heights_by_id[parent_id] for parent_id in promotable)
        for parent_id in sorted(
            candidate for candidate in promotable if heights_by_id[candidate] == min_height
        ):
            active.difference_update(children[parent_id])
            active.add(parent_id)
        append_state()

    return tuple(states)
```

- [ ] **Step 4: Run tests and verify green**

```bash
pytest tests/tracking_ultrack/test_db_browser_preview.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/db_browser_preview.py tests/tracking_ultrack/test_db_browser_preview.py
git commit -m "refactor: extract ultrack hierarchy cut preview logic"
```

## Task 4: Extract Query Layer And Cache Keys

**Files:**
- Create: `src/cellflow/tracking_ultrack/db_browser.py`
- Modify: `src/cellflow/tracking_ultrack/db_browser_preview.py`
- Create: `tests/tracking_ultrack/test_db_browser.py`

- [ ] **Step 1: Write tests for cache-key data and source filtering**

Create `tests/tracking_ultrack/test_db_browser.py`:

```python
from __future__ import annotations

from pathlib import Path

from cellflow.tracking_ultrack.db_browser import HierarchyQueryKey


def test_hierarchy_query_key_includes_source_index():
    db_path = Path("/tmp/data.db")

    first = HierarchyQueryKey.from_path(db_path, mtime_ns=10, frame=3, source_index=0)
    second = HierarchyQueryKey.from_path(db_path, mtime_ns=10, frame=3, source_index=1)

    assert first != second
    assert first.source_index == 0
    assert second.source_index == 1
```

- [ ] **Step 2: Run tests and verify red**

```bash
pytest tests/tracking_ultrack/test_db_browser.py -q
```

Expected: FAIL because `db_browser.py` does not exist.

- [ ] **Step 3: Implement query-key dataclass**

Create `src/cellflow/tracking_ultrack/db_browser.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HierarchyQueryKey:
    db_path: str
    mtime_ns: int
    frame: int
    source_index: int | None

    @classmethod
    def from_path(
        cls,
        db_path: str | Path,
        *,
        mtime_ns: int,
        frame: int,
        source_index: int | None,
    ) -> "HierarchyQueryKey":
        return cls(
            db_path=str(Path(db_path).resolve()),
            mtime_ns=int(mtime_ns),
            frame=int(frame),
            source_index=None if source_index is None else int(source_index),
        )
```

- [ ] **Step 4: Add SQL query helpers**

Extend `db_browser.py` with:

```python
def sqlite_url(db_path: str | Path) -> str:
    return f"sqlite:///{Path(db_path)}"
```

Then move these responsibilities from the widget into functions:

- `query_available_sources(db_path) -> tuple[int, ...]`
- `query_distinct_heights(db_path) -> tuple[float, ...]`
- `query_hierarchy_rows(db_path, frame, source_index) -> tuple[HierarchyNodeRow, ...]`
- `query_connected_nodes(db_path, selected_node_id) -> tuple[dict[int, float], dict[int, float]]`
- `query_summary_text(db_path, frame) -> str`
- `query_middle_frame(db_path) -> int | None`

Each function should create and dispose its own engine at first, matching current behavior. Do not introduce long-lived DB connections in this refactor.

- [ ] **Step 5: Add monkeypatched unit tests for non-SQL helpers**

Add tests for `sqlite_url()` and `HierarchyQueryKey`. For direct SQL helpers, defer full coverage until Task 7 unless a lightweight fixture is already available.

- [ ] **Step 6: Run tests**

```bash
pytest tests/tracking_ultrack/test_db_browser.py tests/tracking_ultrack/test_db_browser_preview.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cellflow/tracking_ultrack/db_browser.py src/cellflow/tracking_ultrack/db_browser_preview.py tests/tracking_ultrack/test_db_browser.py
git commit -m "refactor: extract ultrack db browser query helpers"
```

## Task 5: Move Annotation, Metadata, And Alpha Computation

**Files:**
- Modify: `src/cellflow/tracking_ultrack/db_browser_preview.py`
- Modify: `tests/tracking_ultrack/test_db_browser_preview.py`

- [ ] **Step 1: Add tests for annotation normalization**

Add:

```python
from cellflow.tracking_ultrack.db_browser_preview import annotation_name


def test_annotation_name_normalizes_real_fake_and_unknown():
    assert annotation_name(None) == "UNKNOWN"
    assert annotation_name("REAL") == "REAL"
    assert annotation_name("VarAnnotation.FAKE") == "FAKE"
    assert annotation_name("other") == "UNKNOWN"
```

- [ ] **Step 2: Add tests for alpha composition**

Add:

```python
from cellflow.tracking_ultrack.db_browser_preview import connected_alpha


def test_connected_alpha_uses_edge_weight_and_probability():
    alpha = connected_alpha(
        label_id=2,
        edge_weight=0.5,
        probabilities={1: 0.0, 2: 0.5, 3: 1.0},
        use_edge_weight=True,
        use_probability=True,
    )

    assert 0.05 <= alpha <= 1.0
    assert alpha == 0.5 * (0.15 + 0.85 * 0.5)


def test_connected_alpha_clamps_to_visible_minimum():
    assert connected_alpha(
        label_id=2,
        edge_weight=0.001,
        probabilities={2: 0.0},
        use_edge_weight=True,
        use_probability=True,
    ) == 0.05
```

- [ ] **Step 3: Run tests and verify red**

```bash
pytest tests/tracking_ultrack/test_db_browser_preview.py -q
```

Expected: FAIL because these helpers do not exist.

- [ ] **Step 4: Implement helpers**

Add to `db_browser_preview.py`:

```python
def annotation_name(value: object) -> str:
    if value is None:
        return "UNKNOWN"
    raw = getattr(value, "value", value)
    if raw is None:
        return "UNKNOWN"
    name = str(raw).split(".")[-1].upper()
    return name if name in {"REAL", "FAKE"} else "UNKNOWN"


def connected_alpha(
    *,
    label_id: int,
    edge_weight: float,
    probabilities: dict[int, float],
    use_edge_weight: bool,
    use_probability: bool,
) -> float:
    import numpy as np

    alpha = 1.0
    if use_edge_weight:
        alpha *= float(edge_weight)
    if use_probability and probabilities:
        values = [float(value) for value in probabilities.values()]
        min_prob, max_prob = min(values), max(values)
        denom = max(max_prob - min_prob, 1e-9)
        prob = float(probabilities.get(int(label_id), 1.0))
        alpha *= 0.15 + 0.85 * (prob - min_prob) / denom
    return float(np.clip(alpha, 0.05, 1.0))
```

- [ ] **Step 5: Run tests and commit**

```bash
pytest tests/tracking_ultrack/test_db_browser_preview.py -q
git add src/cellflow/tracking_ultrack/db_browser_preview.py tests/tracking_ultrack/test_db_browser_preview.py
git commit -m "refactor: extract ultrack db browser preview helpers"
```

Expected: tests PASS, then commit succeeds.

## Task 6: Wire Extracted Helpers Into The Widget

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `tests/napari/test_nucleus_tracking_correction_layout.py`

- [ ] **Step 1: Replace hierarchy-state computation**

In `nucleus_workflow_widget.py`, replace the body of `_query_hierarchy_cut_states` with:

```python
from cellflow.tracking_ultrack.db_browser import HierarchyQueryKey, query_hierarchy_rows
from cellflow.tracking_ultrack.db_browser_preview import compute_hierarchy_cut_states
from ultrack.utils.constants import NO_PARENT

source_idx = self._selected_ultrack_db_source()
key = HierarchyQueryKey.from_path(
    db_path,
    mtime_ns=mtime_ns,
    frame=frame,
    source_index=source_idx,
)
cached = self._ultrack_db_cut_state_cache.get(key)
if cached is not None:
    return cached
rows = query_hierarchy_rows(db_path, frame=frame, source_index=source_idx)
states = compute_hierarchy_cut_states(rows, no_parent=NO_PARENT)
self._ultrack_db_cut_state_cache[key] = states
return states
```

Add `_selected_ultrack_db_source()`:

```python
def _selected_ultrack_db_source(self) -> int | None:
    max_source = int(self.ultrack_db_source_slider.maximum())
    if max_source <= 0:
        return None
    return int(self.ultrack_db_source_slider.value())
```

- [ ] **Step 2: Fix source-slider preservation**

Change `_configure_ultrack_db_source_slider` so it preserves the current value when valid:

```python
sources = self._query_available_sources(db_path, mtime_ns)
if not sources:
    self.ultrack_db_source_slider.setRange(0, 0)
    self.ultrack_db_source_slider.setValue(0)
    self.ultrack_db_source_lbl.setText("all")
    return False

max_source = max(sources)
current = int(self.ultrack_db_source_slider.value())
value = current if current in sources else min(sources)
self.ultrack_db_source_slider.setRange(0, max_source)
self.ultrack_db_source_slider.setValue(value)
self.ultrack_db_source_lbl.setText(f"{value}/{max_source}")
return len(sources) > 1
```

- [ ] **Step 3: Replace simple query wrappers**

Convert widget methods into thin wrappers:

```python
def _query_available_sources(self, db_path, mtime_ns):
    from cellflow.tracking_ultrack.db_browser import query_available_sources

    key = (str(db_path.resolve()), mtime_ns, "sources")
    cached = self._ultrack_db_sources_cache.get(key)
    if cached is not None:
        return cached
    result = query_available_sources(db_path)
    self._ultrack_db_sources_cache[key] = result
    return result
```

Apply the same pattern to:

- `_query_distinct_heights`
- `_query_ultrack_db_connected_nodes`
- `_ultrack_db_summary_text`
- `_ultrack_db_middle_frame`

- [ ] **Step 4: Run characterization tests**

```bash
pytest tests/napari/test_nucleus_tracking_correction_layout.py::test_ultrack_db_source_slider_preserves_valid_selection -q
```

Expected: PASS.

- [ ] **Step 5: Run broader focused tests**

```bash
pytest tests/tracking_ultrack/test_db_browser.py tests/tracking_ultrack/test_db_browser_preview.py tests/napari/test_nucleus_tracking_correction_layout.py -q
```

Expected: PASS or only known unrelated failures recorded in the test audit.

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_tracking_correction_layout.py
git commit -m "refactor: route ultrack db browser through extracted helpers"
```

## Task 7: Move Painting-Independent Preview Metadata

**Files:**
- Modify: `src/cellflow/tracking_ultrack/db_browser_preview.py`
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py`
- Modify: `tests/tracking_ultrack/test_db_browser_preview.py`

- [ ] **Step 1: Add tests for metadata extraction from node-like objects**

Add:

```python
from types import SimpleNamespace

from cellflow.tracking_ultrack.db_browser_preview import node_preview_metadata


def test_node_preview_metadata_maps_display_labels_to_node_ids_and_probs():
    nodes = [
        SimpleNamespace(id=101, node_prob=0.25),
        SimpleNamespace(id=202, node_prob=None),
    ]

    probs, label_to_node, node_to_label = node_preview_metadata(nodes)

    assert probs == {1: 0.25, 2: 1.0}
    assert label_to_node == {1: 101, 2: 202}
    assert node_to_label == {101: 1, 202: 2}
```

- [ ] **Step 2: Implement `node_preview_metadata`**

Move logic from widget `_ultrack_db_node_preview_metadata` into:

```python
def node_preview_metadata(nodes: Iterable[object]) -> tuple[dict[int, float], dict[int, int], dict[int, int]]:
    prob_by_label: dict[int, float] = {}
    label_to_node: dict[int, int] = {}
    node_to_label: dict[int, int] = {}
    for label, node in enumerate(nodes, start=1):
        try:
            prob = float(node.node_prob if node.node_prob is not None else 1.0)
        except (AttributeError, TypeError, ValueError):
            prob = 1.0
        prob_by_label[label] = prob
        try:
            node_id = int(node.id)
        except (AttributeError, TypeError, ValueError):
            continue
        label_to_node[label] = node_id
        node_to_label[node_id] = label
    return prob_by_label, label_to_node, node_to_label
```

- [ ] **Step 3: Replace widget metadata helpers**

In `_finalize_hierarchy_nodes`, import and call:

```python
from cellflow.tracking_ultrack.db_browser_preview import (
    annotation_name,
    node_preview_metadata,
)
```

Keep `_paint_ultrack_db_nodes` in the widget for now because it depends on `_node_mask_and_bbox` and viewer plane shape.

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/tracking_ultrack/test_db_browser_preview.py tests/napari/test_nucleus_tracking_correction_layout.py -q
git add src/cellflow/tracking_ultrack/db_browser_preview.py src/cellflow/napari/nucleus_workflow_widget.py tests/tracking_ultrack/test_db_browser_preview.py
git commit -m "refactor: move ultrack preview metadata helpers"
```

Expected: PASS or only known unrelated failures recorded in the audit.

## Task 8: Prune Or Rewrite Tests With Evidence

**Files:**
- Modify: `docs/superpowers/plans/2026-05-13-ultrack-db-browser-test-audit.md`
- Modify tests only when justified by the audit.

- [ ] **Step 1: Re-run relevant tests**

```bash
pytest tests/database/test_validation.py tests/tracking_ultrack/test_db_build.py tests/tracking_ultrack/test_solve.py tests/tracking_ultrack/test_db_browser.py tests/tracking_ultrack/test_db_browser_preview.py tests/napari/test_nucleus_tracking_correction_layout.py -q
```

Expected: PASS, except for tests explicitly classified as stale in the audit.

- [ ] **Step 2: Delete only confirmed stale tests**

For each deleted test, add one audit entry:

```markdown
- Deleted `path::test_name`: covered removed behavior `<short reason>`. Replacement coverage: `new_path::new_test_name`.
```

Do not delete whole files unless every test in the file is classified with a replacement or removed feature.

- [ ] **Step 3: Rewrite brittle widget tests**

Move assertions about pure computation from napari tests into `tests/tracking_ultrack/test_db_browser_preview.py`. Keep napari tests limited to:

- controls exist and are wired
- refresh calls query/preview helpers
- layers are created/updated with expected data shape
- status labels are updated for key user-visible states

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/database/test_validation.py tests/tracking_ultrack/test_db_build.py tests/tracking_ultrack/test_solve.py tests/tracking_ultrack/test_db_browser.py tests/tracking_ultrack/test_db_browser_preview.py tests/napari/test_nucleus_tracking_correction_layout.py -q
git add docs/superpowers/plans/2026-05-13-ultrack-db-browser-test-audit.md tests
git commit -m "test: prune stale ultrack db browser coverage"
```

Expected: PASS.

## Task 9: Final Verification And Manual Smoke Check

**Files:**
- No code changes unless verification exposes a bug.

- [ ] **Step 1: Run focused verification**

```bash
pytest tests/database/test_validation.py tests/tracking_ultrack/test_db_build.py tests/tracking_ultrack/test_solve.py tests/tracking_ultrack/test_db_browser.py tests/tracking_ultrack/test_db_browser_preview.py tests/napari/test_nucleus_tracking_correction_layout.py -q
```

Expected: PASS.

- [ ] **Step 2: Run broader tracking verification**

```bash
pytest tests/tracking_ultrack tests/database -q
```

Expected: PASS.

- [ ] **Step 3: Inspect diff for accidental broad edits**

```bash
git status --short
git diff --stat
git diff -- src/cellflow/napari/nucleus_workflow_widget.py src/cellflow/tracking_ultrack/db_browser.py src/cellflow/tracking_ultrack/db_browser_preview.py
```

Expected: diffs are limited to the planned extraction, wrappers, tests, and audit docs.

- [ ] **Step 4: Optional interactive smoke check**

Open the napari widget in the project environment and verify:

- DB browser opens for an existing `data.db`.
- source slider retains a valid selected source across refresh.
- hierarchy slider still changes rendered segments.
- REAL/FAKE filters still hide/show annotations.
- connected focus still selects by `NodeDB.id` and updates across `t - 1`, `t`, and `t + 1`.

Record any manual findings in the final implementation summary.

## Completion Criteria

- `NucleusWorkflowWidget` no longer owns SQLAlchemy query bodies for the Ultrack DB browser.
- Hierarchy state calculation is covered by pure tests.
- Source selection is part of hierarchy cache identity.
- Valid source-slider selections are preserved across refresh.
- Relevant stale tests are either kept, rewritten, or deleted with evidence in the audit.
- Focused database, tracking-ultrack, and napari tests pass in the project environment.
