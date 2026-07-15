# Interactive Plot → Click-to-Load Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user click a data point in a Shape / Track-dynamics distribution plot to select it, then press a button to replace the napari layers with that point's *input* labels, jump to its frame, and highlight its cell — plus add adjustable x/y axis ranges.

**Architecture:** A new headless `pickable_points()` in `plotting.py` reports which plotted points map back to which DataFrame rows (all rows for strip/swarm; only Tukey outliers for box). `PlotPanel` connects a matplotlib click handler that resolves the nearest point to an identity dict, shows its input-data path, and (on a separate **Load** button) emits a `load_requested` signal carrying a napari-free `LoadTarget`. A shared `_click_to_load.py` helper turns that target into viewer mutations (replace layer, jump frame, select + center cell). The Shape and Track-dynamics plugins wire the resolver + loader; all other plugins are unaffected (resolver defaults to `None`).

**Tech Stack:** Python, pandas, seaborn/matplotlib (Agg + QtAgg backends), qtpy, napari, tifffile, pytest (offscreen Qt).

---

## Context

The plotting surface (`PlotPanel`) was designed with this feature in mind: it already carries `selection_changed = Signal(object)` documented as a "dormant seam for the future napari highlight," and already threads identity columns `("position_id", "frame", "cell_id")` into every snapshot (`plot_panel.py:54,73`). What's missing is (1) recovering which DataFrame row a clicked point came from, (2) a per-quantity resolver that knows the *input* data behind a plot (Shape/Dynamics are derived from the cell/nucleus **tracked-labels TIFF**), and (3) the viewer-side load. The user also wants adjustable axis ranges folded into the same work.

**Decisions locked in with the user:**
- **Scope:** only the two label-based plugins — **Shape** and **Track dynamics**. NLS / contacts / catalog-summary are out of scope (their `PlotPanel`s simply pass no resolver and behave exactly as today). The MSD/DAC/C(r) **curves** panel has no per-object points and is untouched.
- **Two-step interaction:** click selects (updates a path label); a separate **"Load in viewer"** button performs the load.
- **Replace, don't stack:** loading removes the layer this feature previously added and shows only the one input-data layer.
- **Input data per point:** the scope's tracked-labels TIFF — `cell_tracked_labels_path` for cell scope, `nucleus_tracked_labels_path` for nucleus scope (the plugins already expose this via `_label_field()`).
- **On load:** replace layer → jump to frame → set the cell as the active label (`show_selected_label`) → center the camera on its centroid.
- **Frame for aggregate points:** per-track points have no `frame`; jump to their `frame_start` (already a column in the per-track table). Tissue/position-level points keep the whole stack with no jump.
- **Box plots:** only the **outlier (flier)** points are clickable.

### Key facts established during exploration
- **Distribution plots ignore `level`** — strip/swarm/box always plot raw **per-cell rows**, so every pickable point carries a full `{position_id, frame|frame_start, cell_id}` identity (`plotting.py:_plot_distribution`). `level` only affects `bar`/`line` (not pickable).
- Distribution plots set the **x-axis to the hue category**: `sns.boxplot/stripplot/swarmplot(x=hue, y=spec.value, ...)` where `hue` is the `" · "`-joined group label from `_group_label_column()` (`plotting.py`). The y-value is the real, un-jittered measurement.
- **Per-track table** (`dynamics/kinematics.py` `TRACK_COLUMNS`) has `frame_start`, `frame_end`, `cell_id` but **no `frame`**. **Per-frame** tables (shape `core.py`, dynamics instantaneous) have `frame`, `cell_id`. Pooling prepends `position_id = str(record["id"])` (`shape.py:_metadata`, `track_dynamics.py:_metadata`).
- Plugins already hold `self.viewer`, `self._records`, and `_label_field()` (cell vs nucleus per scope). PlotPanel is built at `shape.py:354-365` (`_open_panel`) and `track_dynamics.py:367-379` (`_open_distribution`).
- Layer loading pattern in the repo: `tifffile.imread(path)` → `viewer.add_labels(arr, name=...)`; frame via `viewer.dims.set_current_step(0, f)`; `layer.selected_label` / `layer.show_selected_label`; `viewer.camera.center`.
- Underscored modules under `plugins/` are excluded from plugin auto-discovery (`plugins/__init__.py:_import_plugin_modules`), so `_click_to_load.py` is a safe home for shared helpers (same convention as `_plot_dock.py`).
- Tests run headless with `QT_QPA_PLATFORM=offscreen` (`tests/napari/test_plot_panel.py`); plotting-core tests live in `tests/aggregate_quantification/test_plotting.py`.

---

## File Structure

- **Modify** `src/cellflow/aggregate_quantification/plotting.py` — add axis-limit fields to `StyleSpec`, apply them in `_apply_style`; add `PickPoint` dataclass + `pickable_points()`.
- **Modify** `src/cellflow/napari/aggregate_quantification/plot_panel.py` — `LoadTarget` dataclass, `target_resolver` ctor param, `load_requested` signal, x/y-range controls, pick handler + hit-testing, path label + Load button.
- **Create** `src/cellflow/napari/aggregate_quantification/plugins/_click_to_load.py` — shared `ClickToLoad` (resolver factory + napari loader). Underscore keeps it out of plugin discovery.
- **Modify** `src/cellflow/napari/aggregate_quantification/plugins/shape.py` — build a resolver, pass it to `PlotPanel`, connect `load_requested`.
- **Modify** `src/cellflow/napari/aggregate_quantification/plugins/track_dynamics.py` — same wiring in `_open_distribution` (frame + track views; curves untouched).
- **Tests:** extend `tests/aggregate_quantification/test_plotting.py`, `tests/napari/test_plot_panel.py`, `tests/napari/test_shape_plugin.py`, `tests/napari/test_track_dynamics_plugin.py`; create `tests/napari/test_click_to_load.py`.

---

## Task 1: Adjustable x/y axis range (headless)

**Files:**
- Modify: `src/cellflow/aggregate_quantification/plotting.py` (`StyleSpec`, `_apply_style`)
- Test: `tests/aggregate_quantification/test_plotting.py`

- [ ] **Step 1: Write the failing test**

```python
def test_axis_limits_applied_when_set():
    from cellflow.aggregate_quantification.plotting import (
        PlotSpec, StyleSpec, build_figure,
    )
    import pandas as pd
    df = pd.DataFrame({"condition": ["A", "A", "B"], "frame": [0, 1, 0],
                       "cell_id": [1, 2, 1], "position_id": ["p", "p", "p"],
                       "area": [10.0, 20.0, 30.0]})
    spec = PlotSpec(value="area", group_by=("condition",), level="cell", plot="strip")
    style = StyleSpec(xmin=None, xmax=None, ymin=5.0, ymax=40.0)
    ax = build_figure(df, spec, style).axes[0]
    assert ax.get_ylim() == (5.0, 40.0)

def test_axis_limits_default_to_auto():
    from cellflow.aggregate_quantification.plotting import StyleSpec
    s = StyleSpec()
    assert s.xmin is None and s.xmax is None and s.ymin is None and s.ymax is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/aggregate_quantification/test_plotting.py -k axis_limits -v`
Expected: FAIL (`StyleSpec` has no `xmin`).

- [ ] **Step 3: Implement**

In `StyleSpec` (after `font_size`), add:

```python
    #: Optional axis limits; ``None`` keeps matplotlib's autoscaled bound. Each
    #: side is independent, so e.g. only ``ymin`` may be pinned.
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
```

At the end of `_apply_style(ax, spec, style_spec)` (before it returns / after the grid line), add:

```python
    if style_spec.xmin is not None or style_spec.xmax is not None:
        ax.set_xlim(left=style_spec.xmin, right=style_spec.xmax)
    if style_spec.ymin is not None or style_spec.ymax is not None:
        ax.set_ylim(bottom=style_spec.ymin, top=style_spec.ymax)
```

Note: `_apply_style` has an early `return` in the no-legend branch — place the axis-limit block **before** the legend handling so it always runs.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/aggregate_quantification/test_plotting.py -k axis_limits -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/aggregate_quantification/plotting.py tests/aggregate_quantification/test_plotting.py
git commit -m "feat(plotting): optional x/y axis limits in StyleSpec"
```

---

## Task 2: x/y range controls in PlotPanel

**Files:**
- Modify: `src/cellflow/napari/aggregate_quantification/plot_panel.py` (`_build_styling`, `current_style`)
- Test: `tests/napari/test_plot_panel.py`

- [ ] **Step 1: Write the failing test**

```python
def test_axis_range_fields_feed_style_and_render():
    app = _app()
    panel = _panel()
    panel._ymin_edit.setText("0")
    panel._ymax_edit.setText("100")
    style = panel.current_style()
    assert style.ymin == 0.0 and style.ymax == 100.0
    panel._render()
    assert panel._canvas.figure.axes[0].get_ylim() == (0.0, 100.0)
    panel.deleteLater(); app.processEvents()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/napari/test_plot_panel.py -k axis_range -v`
Expected: FAIL (`panel` has no `_ymin_edit`).

- [ ] **Step 3: Implement**

In `_build_styling`, after the box-plot row, add a range row (reuse the existing `QLineEdit` + "auto" placeholder idiom used for title/labels):

```python
        self._xmin_edit = _range_edit()
        self._xmax_edit = _range_edit()
        self._ymin_edit = _range_edit()
        self._ymax_edit = _range_edit()
        xr = QHBoxLayout(); xr.setContentsMargins(0, 0, 0, 0)
        xr.addWidget(QLabel("X range:")); xr.addWidget(self._xmin_edit, 1); xr.addWidget(self._xmax_edit, 1)
        col.addLayout(xr)
        yr = QHBoxLayout(); yr.setContentsMargins(0, 0, 0, 0)
        yr.addWidget(QLabel("Y range:")); yr.addWidget(self._ymin_edit, 1); yr.addWidget(self._ymax_edit, 1)
        col.addLayout(yr)
        for edit in (self._xmin_edit, self._xmax_edit, self._ymin_edit, self._ymax_edit):
            edit.editingFinished.connect(self._render)
```

Add a factory near the other UI factories at module bottom:

```python
def _range_edit() -> QLineEdit:
    edit = QLineEdit()
    edit.setPlaceholderText("auto")
    return edit
```

Add a parse helper (module level):

```python
def _parse_float(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
```

In `current_style()`, pass the four limits:

```python
            xmin=_parse_float(self._xmin_edit.text()),
            xmax=_parse_float(self._xmax_edit.text()),
            ymin=_parse_float(self._ymin_edit.text()),
            ymax=_parse_float(self._ymax_edit.text()),
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/napari/test_plot_panel.py -k axis_range -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_quantification/plot_panel.py tests/napari/test_plot_panel.py
git commit -m "feat(plot-panel): x/y range controls"
```

---

## Task 3: `pickable_points()` — map plotted points back to rows (headless)

**Files:**
- Modify: `src/cellflow/aggregate_quantification/plotting.py` (new `PickPoint`, `pickable_points`)
- Test: `tests/aggregate_quantification/test_plotting.py`

Behavior: for `strip`/`swarm`, every finite-value row is a pickable point. For `box`, only rows outside the Tukey whiskers (`Q1 - whis·IQR`, `Q3 + whis·IQR`, per category, `whis = style.box_whis`) — matching matplotlib's flier rule. All other plot types return `[]`. Each point reports the hue category string (`""` when no group-by) so the panel can scope a click to the right x-category, plus the value and the DataFrame row index.

- [ ] **Step 1: Write the failing test**

```python
def test_pickable_points_strip_is_one_per_finite_row():
    from cellflow.aggregate_quantification.plotting import (
        PlotSpec, StyleSpec, pickable_points,
    )
    import numpy as np, pandas as pd
    df = pd.DataFrame({"condition": ["A", "A", "B"], "area": [1.0, np.nan, 3.0]})
    spec = PlotSpec(value="area", group_by=("condition",), level="cell", plot="strip")
    pts = pickable_points(df, spec, StyleSpec())
    assert {p.row_index for p in pts} == {0, 2}            # NaN row dropped
    assert {p.category for p in pts} == {"A", "B"}
    assert next(p for p in pts if p.row_index == 0).value == 1.0

def test_pickable_points_box_is_outliers_only():
    from cellflow.aggregate_quantification.plotting import (
        PlotSpec, StyleSpec, pickable_points,
    )
    import pandas as pd
    vals = [10, 11, 12, 13, 12, 11, 10, 12, 11, 200]      # 200 is the flier
    df = pd.DataFrame({"condition": ["A"] * len(vals), "area": [float(v) for v in vals]})
    spec = PlotSpec(value="area", group_by=("condition",), level="cell", plot="box")
    pts = pickable_points(df, spec, StyleSpec(box_whis=1.5))
    assert [p.row_index for p in pts] == [9]

def test_pickable_points_hist_is_empty():
    from cellflow.aggregate_quantification.plotting import (
        PlotSpec, StyleSpec, pickable_points,
    )
    import pandas as pd
    df = pd.DataFrame({"area": [1.0, 2.0]})
    spec = PlotSpec(value="area", group_by=(), level="cell", plot="hist")
    assert pickable_points(df, spec, StyleSpec()) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/aggregate_quantification/test_plotting.py -k pickable -v`
Expected: FAIL (`pickable_points` undefined).

- [ ] **Step 3: Implement**

Add to `plotting.py` (export both in `__all__`):

```python
@dataclass(frozen=True)
class PickPoint:
    """One clickable plotted point mapped back to its source row.

    ``category`` is the hue label the point sits under ("" when there is no
    group-by); ``value`` is the plotted measurement; ``row_index`` is the index
    into the DataFrame ``build_figure`` was given.
    """
    category: str
    value: float
    row_index: int


def pickable_points(df, spec: PlotSpec, style_spec: StyleSpec) -> list[PickPoint]:
    """Plotted points that map 1:1 to a source row, for click-to-select.

    ``strip``/``swarm`` expose every finite-value row; ``box`` exposes only the
    Tukey outliers (``whis`` from *style_spec*); all other plots expose none. The
    category string matches the x-axis tick label seaborn draws (see
    :func:`_group_label_column`), so the UI can scope a click to one category.

    ``row_index`` is positional (``.iloc``): the pooled tables from
    ``pool_object_tables`` carry a default ``RangeIndex``.
    """
    if spec.plot not in ("strip", "swarm", "box") or df.empty:
        return []
    data, hue = _group_label_column(df, list(spec.group_by))
    values = pd.to_numeric(data[spec.value], errors="coerce")
    cats = data[hue].astype(str) if hue is not None else pd.Series([""] * len(data), index=data.index)
    finite = values.notna()
    if spec.plot in ("strip", "swarm"):
        idx = values.index[finite]
        return [PickPoint(str(cats[i]), float(values[i]), int(i)) for i in idx]
    # box: outliers only, per category
    out: list[PickPoint] = []
    for cat, group in values[finite].groupby(cats[finite]):
        q1, q3 = group.quantile(0.25), group.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - style_spec.box_whis * iqr, q3 + style_spec.box_whis * iqr
        for i, v in group.items():
            if v < lo or v > hi:
                out.append(PickPoint(str(cat), float(v), int(i)))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/aggregate_quantification/test_plotting.py -k pickable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/aggregate_quantification/plotting.py tests/aggregate_quantification/test_plotting.py
git commit -m "feat(plotting): pickable_points maps strip/swarm/box-flier points to rows"
```

---

## Task 4: `LoadTarget`, pick handling, path label + Load button in PlotPanel

**Files:**
- Modify: `src/cellflow/napari/aggregate_quantification/plot_panel.py`
- Test: `tests/napari/test_plot_panel.py`

The panel stays napari-free. It gains a `target_resolver` callback (identity dict → `LoadTarget | None`), a `load_requested` signal, a path label, and a Load button. Hit-testing lives in a pure method `_nearest_row_index(xdata, ydata)` so it is unit-testable without synthesizing matplotlib events.

- [ ] **Step 1: Write the failing test**

```python
def test_pick_resolves_identity_and_enables_load():
    from pathlib import Path
    from cellflow.napari.aggregate_quantification.plot_panel import LoadTarget, PlotPanel
    app = _app()
    seen = {}
    def resolver(identity):
        seen.update(identity)
        return LoadTarget(path=Path("/tmp/labels.tif"), kind="labels",
                          frame=identity.get("frame"), cell_id=identity.get("cell_id"),
                          identity=identity)
    panel = PlotPanel(_df(), value_columns=("area",),
                      group_columns=("condition", "date", "position_id", "class_label", "frame"),
                      target_resolver=resolver)
    panel._plot_combo.setCurrentText("strip")
    panel._render()
    # Simulate clicking near the first plotted point (its category + value).
    from cellflow.aggregate_quantification.plotting import pickable_points
    pts = pickable_points(panel._df, panel.current_spec(), panel.current_style())
    p0 = pts[0]
    cat_x = panel._category_x().get(p0.category, 0)
    row = panel._nearest_row_index(cat_x, p0.value)
    assert row == p0.row_index
    panel._select_row(row)
    assert panel._load_btn.isEnabled()
    assert "/tmp/labels.tif" in panel._path_label.text()
    assert seen["cell_id"] == int(panel._df.iloc[row]["cell_id"])
    emitted = []
    panel.load_requested.connect(emitted.append)
    panel._load_btn.click()
    assert emitted and emitted[0].path == Path("/tmp/labels.tif")
    panel.deleteLater(); app.processEvents()

def test_no_resolver_means_no_load_ui():
    app = _app()
    panel = _panel()                       # no target_resolver
    assert not panel._load_btn.isEnabled()
    panel.deleteLater(); app.processEvents()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/napari/test_plot_panel.py -k "pick_resolves or no_resolver" -v`
Expected: FAIL (`LoadTarget` / `_nearest_row_index` undefined).

- [ ] **Step 3: Implement**

Add the imports and dataclass near the top of `plot_panel.py`:

```python
from collections.abc import Callable
from dataclasses import dataclass
from cellflow.aggregate_quantification.plotting import (
    PlotSpec, StyleSpec, aggregate, build_figure, pickable_points, write_csv,
)


@dataclass(frozen=True)
class LoadTarget:
    """A picked point's input data + where to look in it (napari-free).

    Produced by a plugin-supplied resolver, carried out of the panel via
    ``load_requested``; the plugin's loader turns it into viewer layers.
    """
    path: "Path"
    kind: str                 # "labels"
    frame: int | None
    cell_id: int | None
    identity: dict
```

Extend `_IDENTITY_COLUMNS` so per-track points carry their start frame:

```python
_IDENTITY_COLUMNS = ("position_id", "frame", "frame_start", "cell_id")
```

`__init__` signature gains the resolver and a `load_requested` signal:

```python
    load_requested = Signal(object)   # emits a LoadTarget on the Load button

    def __init__(self, dataframe, value_columns, group_columns,
                 target_resolver: "Callable[[dict], LoadTarget | None] | None" = None,
                 parent=None):
        ...
        self._target_resolver = target_resolver
        self._selected_target: LoadTarget | None = None
```

Add the selection UI (build it **before** the `self._render()` call at the end of `__init__`, so `_clear_selection` can touch the widgets), between the canvas holder and the exports:

```python
        self._path_label = QLabel("Click a point to select its input data.")
        status_label(self._path_label, muted=True)
        self._path_label.setWordWrap(True)
        layout.addWidget(self._path_label)
        self._load_btn = QPushButton("Load in viewer")
        action_button(self._load_btn)
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._on_load_clicked)
        layout.addWidget(self._load_btn)
```

In `_render`, after creating the new `canvas`, connect the click and reset selection:

```python
        canvas.mpl_connect("button_press_event", self._on_pick)
        self._clear_selection()
```

Add the hit-testing + selection methods:

```python
    def _category_x(self) -> dict[str, float]:
        """Map each drawn x-axis category label to its x position."""
        if self._canvas is None:
            return {}
        ax = self._canvas.figure.axes[0]
        ticks = ax.get_xticks()
        labels = [t.get_text() for t in ax.get_xticklabels()]
        return {lab: float(x) for x, lab in zip(ticks, labels) if lab}

    def _nearest_row_index(self, xdata: float, ydata: float) -> int | None:
        """Row whose plotted point is nearest the click: snap to the x-category,
        then the closest value within it. Returns None when nothing is pickable."""
        pts = pickable_points(self._df, self.current_spec(), self.current_style())
        if not pts:
            return None
        cat_x = self._category_x()
        if cat_x:
            cat = min(cat_x, key=lambda c: abs(cat_x[c] - xdata))   # snap to category
            candidates = [p for p in pts if p.category == cat]
        else:
            candidates = list(pts)        # single, ungrouped bucket
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(p.value - ydata)).row_index

    def _on_pick(self, event) -> None:
        if self._target_resolver is None or event.inaxes is None or event.ydata is None:
            return
        row = self._nearest_row_index(event.xdata, event.ydata)
        if row is not None:
            self._select_row(row)

    def _select_row(self, row: int) -> None:
        record = self._df.iloc[row]
        identity = {c: _py(record[c]) for c in self._identity_columns}
        self.selection_changed.emit(identity)        # existing dormant seam
        target = self._target_resolver(identity) if self._target_resolver else None
        self._selected_target = target
        if target is None:
            self._path_label.setText("No input data found for this point.")
            self._load_btn.setEnabled(False)
        else:
            self._path_label.setText(str(target.path))
            self._load_btn.setEnabled(True)

    def _clear_selection(self) -> None:
        self._selected_target = None
        if hasattr(self, "_load_btn"):
            self._load_btn.setEnabled(False)
            self._path_label.setText("Click a point to select its input data.")

    def _on_load_clicked(self) -> None:
        if self._selected_target is not None:
            self.load_requested.emit(self._selected_target)
```

Add a small coercion helper (numpy scalars → plain Python) at module bottom:

```python
def _py(value):
    try:
        return value.item()       # numpy scalar -> Python scalar
    except AttributeError:
        return value
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/napari/test_plot_panel.py -k "pick_resolves or no_resolver" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_quantification/plot_panel.py tests/napari/test_plot_panel.py
git commit -m "feat(plot-panel): click-to-select with input-path label and Load button"
```

---

## Task 5: Shared `ClickToLoad` helper (resolver + napari loader)

**Files:**
- Create: `src/cellflow/napari/aggregate_quantification/plugins/_click_to_load.py`
- Test: `tests/napari/test_click_to_load.py`

`ClickToLoad` is constructed per plugin with the viewer. `resolver(records, label_field)` returns a closure mapping an identity dict → `LoadTarget`. `load(target)` performs the replace/jump/select/center. The loader computes the cell centroid directly from the label array (no dependence on per-table centroid columns, so it works for shape, dynamics-frame, and dynamics-track alike).

- [ ] **Step 1: Write the failing test**

```python
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import numpy as np
import tifffile
from cellflow.napari.aggregate_quantification.plot_panel import LoadTarget
from cellflow.napari.aggregate_quantification.plugins._click_to_load import ClickToLoad


class _FakeLayer:
    def __init__(self, data, name):
        self.data, self.name = data, name
        self.selected_label = None
        self.show_selected_label = False


class _FakeDims:
    def __init__(self): self.steps = []
    def set_current_step(self, axis, value): self.steps.append((axis, value))


class _FakeCamera:
    def __init__(self): self.center = None


class _FakeViewer:
    def __init__(self):
        self.layers = []
        self.dims = _FakeDims()
        self.camera = _FakeCamera()
    def add_labels(self, data, name=None):
        layer = _FakeLayer(data, name)
        self.layers.append(layer)
        return layer


def test_resolver_maps_identity_to_input_path():
    rec = {"id": "p1", "cell_tracked_labels_path": "/data/p1/cells.tif"}
    ctl = ClickToLoad(_FakeViewer())
    resolve = ctl.resolver([rec], "cell_tracked_labels_path")
    target = resolve({"position_id": "p1", "frame": 4, "cell_id": 7})
    assert target.path == Path("/data/p1/cells.tif")
    assert target.frame == 4 and target.cell_id == 7

def test_resolver_uses_frame_start_when_no_frame():
    rec = {"id": "p1", "cell_tracked_labels_path": "/data/p1/cells.tif"}
    resolve = ClickToLoad(_FakeViewer()).resolver([rec], "cell_tracked_labels_path")
    target = resolve({"position_id": "p1", "frame_start": 9, "cell_id": 3})
    assert target.frame == 9

def test_resolver_none_when_position_missing_or_no_labels():
    ctl = ClickToLoad(_FakeViewer())
    assert ctl.resolver([], "cell_tracked_labels_path")({"position_id": "x", "cell_id": 1}) is None
    rec = {"id": "p1", "cell_tracked_labels_path": None}
    assert ctl.resolver([rec], "cell_tracked_labels_path")({"position_id": "p1", "cell_id": 1}) is None

def test_load_replaces_jumps_selects_and_centers(tmp_path):
    # 3-frame stack; cell 7 is a block in frame 2.
    stack = np.zeros((3, 10, 10), dtype=np.uint16)
    stack[2, 4:6, 6:8] = 7
    path = tmp_path / "cells.tif"
    tifffile.imwrite(path, stack)
    viewer = _FakeViewer()
    ctl = ClickToLoad(viewer)
    target = LoadTarget(path=path, kind="labels", frame=2, cell_id=7,
                        identity={"position_id": "p1", "frame": 2, "cell_id": 7})
    ctl.load(target)
    assert len(viewer.layers) == 1
    assert viewer.dims.steps[-1] == (0, 2)
    assert viewer.layers[0].selected_label == 7
    assert viewer.layers[0].show_selected_label is True
    assert viewer.camera.center[-2:] == (4.5, 6.5)   # centroid of the 4:6 x 6:8 block
    ctl.load(target)                                  # second load replaces the first
    assert len(viewer.layers) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/napari/test_click_to_load.py -v`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement** `_click_to_load.py`

```python
"""Shared click-to-load: a picked plot point's identity -> input labels in the
viewer. Used by the Shape and Track-dynamics plugins; the underscore keeps it out
of plugin auto-discovery (see ``plugins/__init__.py``)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile

from cellflow.napari.aggregate_quantification.plot_panel import LoadTarget


class ClickToLoad:
    """Resolves picked points to input labels and loads them, replacing the
    previously loaded layer each time (one position shown at a time)."""

    def __init__(self, viewer) -> None:
        self._viewer = viewer
        self._layer = None

    def resolver(self, records: list[dict], label_field: str) -> Callable[[dict], LoadTarget | None]:
        """Closure: identity dict -> LoadTarget for *label_field*'s TIFF, or None
        when the position is unknown or has no labels of that scope."""
        by_id = {str(r.get("id")): r for r in records}

        def resolve(identity: dict) -> LoadTarget | None:
            record = by_id.get(str(identity.get("position_id")))
            if record is None:
                return None
            path = record.get(label_field)
            if not path:
                return None
            frame = identity.get("frame")
            if frame is None:
                frame = identity.get("frame_start")
            cell_id = identity.get("cell_id")
            return LoadTarget(
                path=Path(path), kind="labels",
                frame=None if frame is None else int(frame),
                cell_id=None if cell_id is None else int(cell_id),
                identity=identity,
            )

        return resolve

    def load(self, target: LoadTarget) -> None:
        """Replace the loaded layer with *target*'s labels, jump to its frame, and
        select + center its cell."""
        labels = np.asarray(tifffile.imread(target.path))
        if self._layer is not None and self._layer in list(self._viewer.layers):
            self._viewer.layers.remove(self._layer)
        self._layer = self._viewer.add_labels(labels, name=f"input · {target.path.parent.name}")

        if target.frame is not None and labels.ndim >= 3:
            self._viewer.dims.set_current_step(0, int(target.frame))
        if target.cell_id is not None:
            self._layer.selected_label = int(target.cell_id)
            self._layer.show_selected_label = True
            self._center_on_cell(labels, target.frame, int(target.cell_id))

    def _center_on_cell(self, labels: np.ndarray, frame: int | None, cell_id: int) -> None:
        plane = labels[frame] if (frame is not None and labels.ndim >= 3) else labels
        ys, xs = np.nonzero(plane == cell_id)
        if ys.size:
            try:
                self._viewer.camera.center = (float(ys.mean()), float(xs.mean()))
            except Exception:           # camera centering is best-effort across napari versions
                pass
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/napari/test_click_to_load.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_quantification/plugins/_click_to_load.py tests/napari/test_click_to_load.py
git commit -m "feat(aq): shared ClickToLoad resolver + viewer loader"
```

---

## Task 6: Wire ClickToLoad into the Shape plugin

**Files:**
- Modify: `src/cellflow/napari/aggregate_quantification/plugins/shape.py` (`_open_panel`)
- Test: `tests/napari/test_shape_plugin.py`

- [ ] **Step 1: Write the failing test**

```python
def test_shape_panel_gets_resolver_and_load_is_wired():
    # Build a plugin with a fake viewer + one in-scope built record, open a panel,
    # drain the pooling thread-worker as the existing tests do, then:
    ...
    panel = plugin._panel
    assert panel._target_resolver is not None
```

Implementation note: have `_open_panel` store `self._panel = panel` so tests can assert wiring; follow whatever pattern `test_shape_plugin.py` already uses to await the worker.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/napari/test_shape_plugin.py -k resolver -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `_open_panel(self, pooled)` (currently `shape.py:354-365`), build a fresh controller targeting the current viewer, pass its resolver to the panel, and connect Load:

```python
    def _open_panel(self, pooled: pd.DataFrame) -> None:
        if pooled.empty:
            self._plot_status.setText("No built tables in scope.")
            return
        from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel
        from cellflow.napari.aggregate_quantification.plugins._click_to_load import ClickToLoad

        controller = ClickToLoad(self.viewer)
        panel = PlotPanel(
            pooled, value_columns=self._value_columns(), group_columns=_GROUP_COLUMNS,
            target_resolver=controller.resolver(self._records, self._label_field()),
        )
        panel.load_requested.connect(controller.load)
        self._panel = panel
        name = self._dock_name()
        self._add_dock(panel, name)
        self._plot_status.setText(...)   # keep existing status text
```

A fresh `ClickToLoad` per plot always targets the current viewer; the "replace previous" guarantee holds within a panel's lifetime because the same controller handles every Load from that panel.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/napari/test_shape_plugin.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_quantification/plugins/shape.py tests/napari/test_shape_plugin.py
git commit -m "feat(shape): wire click-to-load into the Shape plot panel"
```

---

## Task 7: Wire ClickToLoad into the Track-dynamics plugin

**Files:**
- Modify: `src/cellflow/napari/aggregate_quantification/plugins/track_dynamics.py` (`_open_distribution`)
- Test: `tests/napari/test_track_dynamics_plugin.py`

Apply the same wiring as Task 6 in `_open_distribution` (`track_dynamics.py:367-379`) — it covers both the per-frame and per-track views. `_open_curves` is **not** wired (curves have no per-object points).

- [ ] **Step 1: Write the failing test**

```python
def test_dynamics_distribution_panel_gets_resolver():
    # Mirror test_track_dynamics_plugin.py setup; for the per-track view assert the
    # resolver turns a {position_id, frame_start, cell_id} identity into a target
    # whose frame == frame_start.
    ...
    assert panel._target_resolver is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/napari/test_track_dynamics_plugin.py -k resolver -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `_open_distribution(self, view, pooled)`:

```python
        from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel
        from cellflow.napari.aggregate_quantification.plugins._click_to_load import ClickToLoad

        values, groups = (
            (_FRAME_VALUES, _FRAME_GROUPS) if view == "frame" else (_TRACK_VALUES, _TRACK_GROUPS)
        )
        controller = ClickToLoad(self.viewer)
        panel = PlotPanel(pooled, value_columns=values, group_columns=groups,
                          target_resolver=controller.resolver(self._records, self._label_field()))
        panel.load_requested.connect(controller.load)
        self._panel = panel
        name = self._dock_name()
        self._add_dock(panel, name)
        self._plot_status.setText(f"Opened {name} ({len(pooled)} rows).")
```

Note: `_TRACK_GROUPS` has no `"frame"` axis, so per-track points naturally carry `frame_start` (added to `_IDENTITY_COLUMNS` in Task 4) and the resolver falls back to it.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/napari/test_track_dynamics_plugin.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_quantification/plugins/track_dynamics.py tests/napari/test_track_dynamics_plugin.py
git commit -m "feat(track-dynamics): wire click-to-load into distribution panels"
```

---

## Verification (end-to-end)

1. **Unit tests:**
   ```bash
   pytest tests/aggregate_quantification/test_plotting.py tests/napari/test_plot_panel.py \
          tests/napari/test_click_to_load.py tests/napari/test_shape_plugin.py \
          tests/napari/test_track_dynamics_plugin.py -v
   ```
   All green; the new picking/axis/resolver/loader tests pass and the existing panel/plugin tests still pass (back-compat: `target_resolver` defaults to `None`).

2. **Manual in napari** (host, with a built position):
   - Open the Aggregate Quantification studio, select positions with cell (or nucleus) tracked labels.
   - **Shape:** Build, then Plot. Choose `strip` or `swarm`. Click a point → the path label shows the position's `cell_tracked_labels_path`; press **Load in viewer** → the viewer shows only that labels TIFF, jumps to the point's frame, and the clicked cell is highlighted/centered. Click a point from a *different* position and Load → the previous layer is replaced, not stacked.
   - Choose `box` → confirm only **outlier** markers respond to clicks (clicking the box body selects nothing).
   - **Track dynamics:** Plot → Per-track → `strip`; click a point → Load jumps to that track's `frame_start`. Per-frame view jumps to the exact `frame`.
   - **Styling:** set X/Y range fields → axes clamp; clear them → autoscale returns.
   - **Curves** view and the NLS/contacts/catalog plugins: confirm no Load button behavior changed (no resolver wired).

## Risks / notes
- **Pick precision:** strip jitter / swarm spread means the drawn x differs slightly from the model category center; the handler snaps to the nearest *category* then nearest *value*, so the value axis (the real measurement) disambiguates. Two points with near-identical values in the same category may be indistinguishable — acceptable (they share an image anyway, modulo cell_id).
- **Row index contract:** `pickable_points` returns positional row indices, valid because `pool_object_tables` yields a default `RangeIndex`. If a caller ever passes a non-range-indexed frame, the panel must `reset_index(drop=True)` first — the pooled path never does.
- **Camera centering** is best-effort (wrapped in `try/except`) to tolerate napari version differences in `camera.center` arity.
