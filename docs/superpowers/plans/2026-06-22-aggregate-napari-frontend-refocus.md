# Napari Front-End Refocus (plot-layer teardown) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the in-napari plotting layer from Aggregate Quantification so visualization is Iris-only, keeping the studio's discover&add + Compute + Aggregate + the analysis plugins (curation, visualize-contacts) fully working.

**Architecture:** Pure teardown with surgical rewiring. The studio (`AggregateQuantificationStudioWidget`) drops its "Plots" collapsible; `SharedParamsWidget` sheds its dependency on the plot layer (keeping the build knobs that double as build params); then the orphaned plot modules and their tests are deleted. Tasks are ordered so the import tree stays valid and the test suite stays green after **every** commit (rewire consumers first, delete the now-orphaned modules last).

**Tech Stack:** Python 3.11, qtpy/Qt (offscreen for tests), napari plugin framework, pytest.

**Scope note (decided with the user):** This slice is **deletion only**. The separate "refocus Run to drive the full `run()` and author a `config.toml` from the UI" capability (spec decision #2 / open #1) is **deferred to a follow-up slice** — do NOT build it here. When it is built, the config will be authored from UI fields (the user's chosen direction).

---

## Background for the implementer (read once)

You have **zero prior context**. Key facts:

- **Why this is safe now:** in-napari plotting has been superseded by Iris (the `.iris` export + the static figure render + the analysis subpackage). The three prior consolidation slices (curation table, NLS CLI step, curation tool) are merged. This is the "cut gate": the plot layer is deleted only after its replacements are proven.

- **The studio** is `src/cellflow/napari/aggregate_quantification_studio.py` → class `AggregateQuantificationStudioWidget`. Its `__init__` builds sections in order: Catalogue, Parameters, Tools, Compute (build), Aggregate, **Plots**. We remove only the **Plots** section. Compute, Aggregate, Tools (the plugin list, which now includes `curation` + `visualize_contacts`), and Catalogue all survive untouched.

- **The bare widget** `AggregateQuantificationWidget` (`aggregate_quantification_widget.py`) is reused as the embedded display by the `visualize_contacts` and `curation` plugins. It does **not** import any plot module (verified) — leave it alone; it must keep working.

- **`SharedParamsWidget`** (`aggregate_quantification_params.py`) has FOV + shuffles + pixel-size + frame-interval fields. Its `plot_params()` method (returning `PlotParams`) is used **only** by the deleted plot area. Its `build_params()` and `stamp()` survive — and `build_params()` legitimately reads FOV + shuffles (they are real build knobs: the contact-type z-score's `shuffles` and the density's `fov_area_mm2`). The only coupling to delete is the `PlotParams` import + the `plot_params()` method; `PlotParams().shuffles` (default `1000`) is inlined as a local constant.

- **Files that will be deleted (the plot layer):**
  - `src/cellflow/napari/aggregate_quantification_plot_area.py` (`PlotAreaWidget`)
  - `src/cellflow/napari/aggregate_quantification/plot_panel.py`
  - `src/cellflow/napari/aggregate_quantification/dynamics_curves_panel.py`
  - `src/cellflow/napari/aggregate_quantification/shape_editor.py`
  - `src/cellflow/napari/aggregate_quantification/_mpl_toolbar.py` (imported only by the two panels above)
  - `src/cellflow/napari/aggregate_quantification/plugins/_plot_dock.py`
  - `src/cellflow/napari/aggregate_quantification/plugins/_click_to_load.py`
  - `src/cellflow/napari/aggregate_quantification/plots/` (whole package: `__init__.py`, `shape.py`, `dynamics.py`, `contacts.py`, `_pool_plot.py`, `_pooling.py`)

  The `_plot_dock.py` / `_click_to_load.py` modules are underscore-prefixed, so the plugin-discovery walk (`plugins/__init__.py`, which skips `_`-prefixed modules) never registered them — deleting them does not affect plugin discovery.

- **Test files that will be deleted (cover only the deleted modules):** `tests/napari/test_shape_editor.py`, `test_plots_registry.py`, `test_plot_area.py`, `test_plots_contacts.py`, `test_plots_aggregate_source.py`, `test_click_to_load.py`, `test_plots_shape.py`, `test_plots_dynamics.py`, `test_plot_panel.py`. **`tests/napari/test_shared_params.py` is updated, not deleted** (the params widget survives).

- **The backend `iris_export` owns real plotting** and is *not* touched. The `plots/` package being deleted is the **napari** in-app plotting, not the backend export/figures.

- **Pre-existing unrelated failures:** before this work, 3 napari tests fail on a pandas `df.insert` multi-column issue: `test_aggregate_area.py::test_status_reflects_written_table`, `test_plots_contacts.py::test_typed_views_empty_for_unclassified_positions`, `test_plots_contacts.py::test_signed_contact_length_pool_handles_no_t1`. **Two of these live in `test_plots_contacts.py`, which this plan deletes** — so after this work only the `test_aggregate_area.py` one remains (it is unrelated to plotting and survives the refocus). Do not try to "fix" it here; just confirm it is the *only* remaining failure.

- **Always run tests with `--frozen`:** `uv run --frozen pytest ...` (the lockfile is intentionally stale; a bare `uv run` is unsatisfiable).

- **Qt test convention:** `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")`, app via `QApplication.instance() or QApplication([])`, `deleteLater()` + `processEvents()` at the end.

---

## File Structure

- **Modify** `src/cellflow/napari/aggregate_quantification_studio.py` — drop the Plots section (import, builder call, builder method, `_push_context` block).
- **Modify** `tests/napari/test_aggregate_quantification_studio.py` — drop the "Plots" assertion; assert it is gone.
- **Modify** `src/cellflow/napari/aggregate_quantification_params.py` — remove the `PlotParams` import + `plot_params()`; inline the shuffles default.
- **Modify** `tests/napari/test_shared_params.py` — drop the `plot_params` test/import; add a `build_params` test.
- **Delete** the 8 plot-layer source paths + the 9 plot-layer test files (Task 3).

---

## Task 1: Drop the studio's Plots section

**Files:**
- Modify: `src/cellflow/napari/aggregate_quantification_studio.py`
- Test: `tests/napari/test_aggregate_quantification_studio.py`

Remove the Plots collapsible and every reference to it. After this commit nothing in the studio imports the plot area; the orphaned `aggregate_quantification_plot_area.py` still exists (deleted in Task 3) and its own tests still pass, so the suite stays green.

- [ ] **Step 1: Update the studio test first (it pins the new shape)**

In `tests/napari/test_aggregate_quantification_studio.py`, find `test_section_defaults_and_compute_rename` (around line 368). Replace its body's comment + assertions so it no longer expects a "Plots" section and asserts the section is gone:

```python
def test_section_defaults_and_compute_rename(monkeypatch):
    # Bugs 16/18/20: Parameters expanded; Build renamed to Compute; Tools /
    # Compute collapsed by default. The in-napari Plots section is removed
    # (visualization is Iris-only).
    from cellflow.napari.widgets import CollapsibleSection

    app = _app()
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [])
    widget = mod.AggregateQuantificationStudioWidget()
    by_title = {
        s.title: s for s in widget.findChildren(CollapsibleSection)
    }
    assert "Build" not in by_title and "Compute" in by_title
    assert "Plots" not in by_title
    assert by_title["Parameters"].is_expanded is True
    assert by_title["Tools"].is_expanded is False
    assert by_title["Compute"].is_expanded is False
    widget.deleteLater()
    app.processEvents()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --frozen pytest "tests/napari/test_aggregate_quantification_studio.py::test_section_defaults_and_compute_rename" -v`
Expected: FAIL — the "Plots" section still exists, so `assert "Plots" not in by_title` fails.

- [ ] **Step 3: Remove the Plots section from the studio**

In `src/cellflow/napari/aggregate_quantification_studio.py` make exactly these four edits:

1. Delete the import line (around line 57):
```python
from cellflow.napari.aggregate_quantification_plot_area import PlotAreaWidget
```

2. In `__init__`, delete the `_build_plot_section` call (around line 174). The block currently reads:
```python
        # Tools on top, then the pure Build area, the Aggregate area, then Plot.
        self._build_tools_section(layout)
        self._build_build_section(layout)
        self._build_aggregate_section(layout)
        self._build_plot_section(layout)
```
Change it to:
```python
        # Tools on top, then the pure Build area, then the Aggregate area.
        # (Plotting is Iris-only — no in-napari Plot area.)
        self._build_tools_section(layout)
        self._build_build_section(layout)
        self._build_aggregate_section(layout)
```

3. Delete the entire `_build_plot_section` method (around lines 366–373):
```python
    def _build_plot_section(self, layout) -> None:
        """The Plot area: every registered plot, grouped by family and gated by
        product availability. Separate from building — it plots whatever the
        in-scope positions have built."""
        self._plot_area = PlotAreaWidget(
            self.viewer, params_provider=self._shared_params.plot_params
        )
        layout.addWidget(CollapsibleSection("Plots", self._plot_area, expanded=False))
```

4. In `_push_context`, delete the `_plot_area` block (around lines 747–749):
```python
        plot_area = getattr(self, "_plot_area", None)
        if plot_area is not None:
            self._push_context_to(plot_area)
```

Leave the `build_area` and `aggregate_area` blocks in `_push_context` intact.

- [ ] **Step 4: Run the studio + widget suites to verify green**

Run: `uv run --frozen pytest tests/napari/test_aggregate_quantification_studio.py tests/napari/test_aggregate_quantification_widget.py -v`
Expected: PASS. Then `uv run --frozen ruff check src/cellflow/napari/aggregate_quantification_studio.py` — clean (no unused `PlotAreaWidget`/`CollapsibleSection` import; note `CollapsibleSection` is still used by other sections, so it stays).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_quantification_studio.py tests/napari/test_aggregate_quantification_studio.py
git commit -m "$(cat <<'EOF'
refactor(napari): remove the in-studio Plots section (Iris owns plotting)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N3vuJPVYYuYWh6EQLh8vrv
EOF
)"
```

---

## Task 2: Decouple `SharedParamsWidget` from the plot layer

**Files:**
- Modify: `src/cellflow/napari/aggregate_quantification_params.py`
- Test: `tests/napari/test_shared_params.py`

Drop the `PlotParams` import and the `plot_params()` method (only the deleted plot area used it). Keep the FOV/shuffles fields and `build_params()`/`stamp()` (FOV + shuffles are build knobs). Inline the shuffles default as a local constant so params no longer imports the `plots` package. After this commit, nothing outside the plot layer imports the plot layer.

- [ ] **Step 1: Update the test first**

In `tests/napari/test_shared_params.py`:

1. Remove the import line:
```python
from cellflow.napari.aggregate_quantification.plots import PlotParams
```

2. Delete `test_plot_params_reads_fields_with_auto_defaults` (the whole function, lines ~18–35).

3. Add this test in its place (locks the surviving build-knob behaviour, including the shuffles default):
```python
def test_build_params_reads_fov_and_shuffles_with_default():
    app = _app()
    try:
        w = SharedParamsWidget()
        # Blank shuffles -> the default (1000); blank fov -> None (unset, for gating).
        params = w.build_params()
        assert params["shuffles"] == 1000
        assert params["fov_area_mm2"] is None

        w._fov_edit.setText("1.5")
        w._shuffles_edit.setText("200")
        params = w.build_params()
        assert params["fov_area_mm2"] == 1.5
        assert params["shuffles"] == 200
    finally:
        w.deleteLater()
        app.processEvents()
```

> Match the existing file's fixtures: it already defines `_app()` and imports `SharedParamsWidget`. If the existing `plot_params` test used a different app/teardown idiom, mirror whatever the *kept* `test_stamp_*` tests in this file do.

- [ ] **Step 2: Run it to characterize current behaviour (this is a refactor, not new behaviour)**

This task is a refactor: the new `build_params` test characterizes behaviour that already holds (the current code already returns `shuffles == 1000` by default via `PlotParams().shuffles`), so it should **PASS** against the current source. Run it to confirm the characterization is correct before changing the source:

Run: `uv run --frozen pytest "tests/napari/test_shared_params.py::test_build_params_reads_fov_and_shuffles_with_default" -v`
Expected: PASS. (The whole file may not collect cleanly if you have already removed the `PlotParams` import while the old `plot_params` test still references it — run just this one test by name for the characterization check, then do Step 3, which removes the source dependency so the whole file is consistent again.)

- [ ] **Step 3: Slim the params widget**

In `src/cellflow/napari/aggregate_quantification_params.py`:

1. Replace the import:
```python
from cellflow.napari.aggregate_quantification.plots import PlotParams
```
with a local constant near the top of the module (after the imports):
```python
#: Default permutation count for the contact-type z-score null (was PlotParams's
#: default; the in-napari plot layer that defined PlotParams has been removed).
_DEFAULT_SHUFFLES = 1000
```

2. Delete the `plot_params()` method entirely (the block under the `# ---- plot side` comment):
```python
    # ----------------------------------------------------------------- plot side
    def plot_params(self) -> PlotParams:
        """Package the plot-time fields (blank/invalid → auto / default)."""
        shuffles = _parse_int(self._shuffles_edit.text())
        return PlotParams(
            pixel_size_um=_parse_positive(self._pixel_size_edit.text()),
            fov_area_mm2=_parse_positive(self._fov_edit.text()),
            shuffles=shuffles if shuffles and shuffles > 0 else PlotParams().shuffles,
        )
```

3. In `build_params()`, replace the `PlotParams().shuffles` default with the constant:
```python
        shuffles = _parse_int(self._shuffles_edit.text())
        return {
            "shuffles": shuffles if shuffles and shuffles > 0 else _DEFAULT_SHUFFLES,
            "pixel_size_um": _parse_positive(self._pixel_size_edit.text()),
            "time_interval_s": _parse_positive(self._frame_interval_edit.text()),
            "fov_area_mm2": _parse_positive(self._fov_edit.text()),
        }
```

4. Update the module docstring (lines ~14–15) to drop the `plot_params` / `PlotParams` reference (it now only stamps build params + supplies `build_params`). Keep it accurate and brief.

- [ ] **Step 4: Run the params + studio suites to verify green**

Run: `uv run --frozen pytest tests/napari/test_shared_params.py tests/napari/test_aggregate_quantification_studio.py -v`
Expected: PASS. Then `uv run --frozen ruff check src/cellflow/napari/aggregate_quantification_params.py` — clean (no unused `PlotParams`).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_quantification_params.py tests/napari/test_shared_params.py
git commit -m "$(cat <<'EOF'
refactor(napari): decouple SharedParamsWidget from the plot layer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N3vuJPVYYuYWh6EQLh8vrv
EOF
)"
```

---

## Task 3: Delete the orphaned plot layer + its tests

**Files:** (all deletions)
- Delete src: `aggregate_quantification_plot_area.py`, `aggregate_quantification/plot_panel.py`, `aggregate_quantification/dynamics_curves_panel.py`, `aggregate_quantification/shape_editor.py`, `aggregate_quantification/_mpl_toolbar.py`, `aggregate_quantification/plugins/_plot_dock.py`, `aggregate_quantification/plugins/_click_to_load.py`, `aggregate_quantification/plots/` (whole dir)
- Delete tests: `test_shape_editor.py`, `test_plots_registry.py`, `test_plot_area.py`, `test_plots_contacts.py`, `test_plots_aggregate_source.py`, `test_click_to_load.py`, `test_plots_shape.py`, `test_plots_dynamics.py`, `test_plot_panel.py`

After Tasks 1–2, every one of these is orphaned (nothing in the surviving tree imports it). Verify, then delete.

- [ ] **Step 1: Confirm the modules are orphaned**

Run:
```bash
grep -rn "plot_panel\|dynamics_curves_panel\|shape_editor\|_plot_dock\|_click_to_load\|_mpl_toolbar\|aggregate_quantification.plots\|from .plots\|aggregate_quantification_plot_area\|PlotAreaWidget\|PlotParams\|plot_params" src/cellflow
```
Expected: matches **only** inside the files about to be deleted (the plot-layer modules referencing each other). There must be **no** match in `aggregate_quantification_studio.py`, `aggregate_quantification_params.py`, `aggregate_quantification_widget.py`, `main_widget.py`, `studio_plugins.py`, or anywhere else in the surviving tree. If a surviving file still references any of these, STOP and report — Tasks 1–2 missed a reference.

- [ ] **Step 2: Delete the source modules**

```bash
git rm src/cellflow/napari/aggregate_quantification_plot_area.py \
       src/cellflow/napari/aggregate_quantification/plot_panel.py \
       src/cellflow/napari/aggregate_quantification/dynamics_curves_panel.py \
       src/cellflow/napari/aggregate_quantification/shape_editor.py \
       src/cellflow/napari/aggregate_quantification/_mpl_toolbar.py \
       src/cellflow/napari/aggregate_quantification/plugins/_plot_dock.py \
       src/cellflow/napari/aggregate_quantification/plugins/_click_to_load.py
git rm -r src/cellflow/napari/aggregate_quantification/plots
```

- [ ] **Step 3: Delete the plot-layer tests**

```bash
git rm tests/napari/test_shape_editor.py \
       tests/napari/test_plots_registry.py \
       tests/napari/test_plot_area.py \
       tests/napari/test_plots_contacts.py \
       tests/napari/test_plots_aggregate_source.py \
       tests/napari/test_click_to_load.py \
       tests/napari/test_plots_shape.py \
       tests/napari/test_plots_dynamics.py \
       tests/napari/test_plot_panel.py
```

- [ ] **Step 4: Verify plugin discovery + full suites**

Run:
```bash
uv run --frozen pytest tests/napari/ tests/aggregate_quantification/ -q
```
Expected: PASS except the **single** pre-existing unrelated failure `tests/napari/test_aggregate_area.py::test_status_reflects_written_table` (the pandas `df.insert` issue). Confirm there are **no import errors** (a dangling import would surface as a collection error) and **no new failures**. If a collection/import error appears, it points at a missed reference — fix by removing that reference (do not resurrect a deleted module).

Also confirm the plugins still register:
```bash
uv run --frozen python -c "from cellflow.napari.aggregate_quantification.plugins import available_analysis_plugins; print(sorted(c.plugin_id for c in available_analysis_plugins()))"
```
Expected: includes `curation` and `visualize_contacts`; does NOT include any plot/NLS plugin.

- [ ] **Step 5: Commit**

```bash
git add -A
git status   # confirm only the intended deletions are staged (no stray files, no planning doc)
git commit -m "$(cat <<'EOF'
refactor(napari): delete the in-napari plot layer (Iris owns visualization)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N3vuJPVYYuYWh6EQLh8vrv
EOF
)"
```

---

## Final review

After all three tasks:
- `uv run --frozen pytest tests/napari/ tests/aggregate_quantification/ -q` → green except the one known `test_aggregate_area` pre-existing failure; no new failures, no import/collection errors.
- The studio opens with sections Catalogue / Parameters / Tools / Compute / Aggregate — **no Plots**.
- `curation` + `visualize_contacts` plugins still register and the embedded `AggregateQuantificationWidget` display still works.
- `grep -rn "PlotAreaWidget\|aggregate_quantification.plots\|plot_panel\|plot_params" src/cellflow` returns nothing.
- The backend `iris_export` (the real plotting/figures path) is untouched.

## Notes on scope (deferred — do NOT implement)

- **Run front-end driving full `run()` + authoring `config.toml`** (spec decision #2 / open #1) — deferred to a follow-up slice; when built, author config from UI fields. The studio keeps its existing Compute + Aggregate areas for now.
- **`cellflow-aggregate` manifest re-confirmation** (spec open #2) — the manifest points at `make_aggregate_quantification_widget`, which is unaffected by this teardown; no manifest change is needed here.
- **Slimming `SharedParamsWidget`'s FOV/shuffles UI** — kept as-is because both feed `build_params()`; do not remove the fields.
