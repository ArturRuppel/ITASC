# UI-authored config + single Run front-end — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the studio's piecemeal Compute/Aggregate sections with one **Run** section that authors `catalog.csv` + `config.toml` from the UI and drives the whole `pipeline.run()`.

**Architecture:** Three layers — (1) a dependency-free TOML writer `write_config` beside `load_config`; (2) a headless `author_config` composing `save_catalog` + `write_config`, plus a `progress_cb` pass-through on `run()`; (3) a thin `RunArea` Qt widget the studio wires to `author_config`/`run` on a worker thread.

**Tech Stack:** Python 3.10+, `tomllib`/`tomli` (read only — writer is hand-rolled), napari/qtpy, pandas, pytest. Always run tests with `uv run --frozen pytest` (the lock is intentionally stale; bare `uv run` is unsatisfiable).

**Spec:** `docs/superpowers/specs/2026-06-22-aggregate-run-config-frontend-design.md`

---

## Task 1: TOML writer + `write_config` (headless)

**Files:**
- Modify: `src/cellflow/aggregate_quantification/config.py`
- Test: `tests/aggregate_quantification/test_config_writer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/aggregate_quantification/test_config_writer.py`:

```python
"""write_config — the inverse of load_config (round-trips through TOML)."""
from __future__ import annotations

from cellflow.aggregate_quantification.config import (
    NlsConfig,
    load_config,
    write_config,
)


def test_round_trip_minimal(tmp_path):
    path = tmp_path / "config.toml"
    write_config(path, catalog="catalog.csv", quantities=["contacts"])
    cfg = load_config(path)
    assert cfg.catalog == (tmp_path / "catalog.csv").resolve()
    assert cfg.export_dir == (tmp_path / "export").resolve()
    assert cfg.curation == (tmp_path / "curation.csv").resolve()
    assert cfg.quantities == ("contacts",)
    assert cfg.nls is None
    assert cfg.render_plots is False


def test_params_drop_unset_keys(tmp_path):
    path = tmp_path / "config.toml"
    write_config(
        path,
        catalog="catalog.csv",
        params={"pixel_size_um": 0.25, "time_interval_s": None,
                "fov_area_mm2": None, "shuffles": 1000},
    )
    cfg = load_config(path)
    assert cfg.params == {"pixel_size_um": 0.25, "shuffles": 1000}


def test_nls_table_round_trips(tmp_path):
    path = tmp_path / "config.toml"
    write_config(
        path,
        catalog="catalog.csv",
        nls=NlsConfig(enabled=True, image="0_input/NLS_zavg.tif",
                      method="fixed", threshold=12.5),
    )
    cfg = load_config(path)
    assert cfg.nls == NlsConfig(enabled=True, image="0_input/NLS_zavg.tif",
                                method="fixed", threshold=12.5)


def test_plots_table_round_trips(tmp_path):
    path = tmp_path / "config.toml"
    write_config(path, catalog="catalog.csv", render_plots=True,
                 plot_formats=["png", "pdf"])
    cfg = load_config(path)
    assert cfg.render_plots is True
    assert cfg.plot_formats == ("png", "pdf")


def test_quantities_empty_means_all(tmp_path):
    """No quantities key -> load_config reads () -> 'every quantifier'."""
    path = tmp_path / "config.toml"
    write_config(path, catalog="catalog.csv", quantities=[])
    assert "quantities" not in path.read_text()
    assert load_config(path).quantities == ()


def test_string_escaping(tmp_path):
    """Backslashes / quotes in a path survive the round trip."""
    path = tmp_path / "config.toml"
    write_config(path, catalog=r'weird"\name.csv')
    assert load_config(path).catalog.name == r'weird"\name.csv'


def test_returns_written_path(tmp_path):
    path = tmp_path / "config.toml"
    assert write_config(path, catalog="catalog.csv") == path
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen pytest tests/aggregate_quantification/test_config_writer.py -q`
Expected: FAIL with `ImportError: cannot import name 'write_config'`.

- [ ] **Step 3: Implement `write_config` + the closed-schema TOML writer**

In `src/cellflow/aggregate_quantification/config.py`:

Add to the imports block (top, after the existing `from pathlib import Path`):

```python
from collections.abc import Mapping, Sequence
```

Extend `__all__`:

```python
__all__ = ["NlsConfig", "RunConfig", "load_config", "write_config"]
```

Append at the end of the module:

```python
def write_config(
    path: Path | str,
    *,
    catalog: str = "catalog.csv",
    export_dir: str = _DEFAULT_EXPORT_DIR,
    curation: str = _DEFAULT_CURATION,
    quantities: Sequence[str] = (),
    params: Mapping[str, object] | None = None,
    nls: NlsConfig | None = None,
    render_plots: bool = False,
    plot_formats: Sequence[str] = ("png", "svg"),
) -> Path:
    """Author a TOML run-config at *path* — the inverse of :func:`load_config`.

    Paths are written **relative** (verbatim), so the project folder stays
    relocatable. ``quantities`` is emitted only when non-empty (empty round-trips
    to ``()`` = "all"). ``params`` keys that are ``None`` are dropped (an unset
    pixel size etc.). The ``[nls]`` table is written only when *nls* is given;
    ``[plots]`` is always written. ``load_config(write_config(path, ...))``
    reproduces the inputs (paths resolved against ``path.parent``). Returns *path*.
    """
    path = Path(path)
    lines: list[str] = [
        f"catalog = {_toml_str(catalog)}",
        f"export_dir = {_toml_str(export_dir)}",
        f"curation = {_toml_str(curation)}",
    ]
    if quantities:
        lines.append(f"quantities = {_toml_array(quantities)}")

    if params:
        kept = {k: v for k, v in params.items() if v is not None}
        if kept:
            lines.append("")
            lines.append("[params]")
            lines.extend(f"{k} = {_toml_value(v)}" for k, v in kept.items())

    if nls is not None:
        lines += [
            "",
            "[nls]",
            f"enabled = {_toml_value(nls.enabled)}",
            f"image = {_toml_str(nls.image)}",
            f"method = {_toml_str(nls.method)}",
            f"threshold = {_toml_value(float(nls.threshold))}",
        ]

    lines += [
        "",
        "[plots]",
        f"render = {_toml_value(render_plots)}",
        f"formats = {_toml_array(plot_formats)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _toml_str(value: object) -> str:
    """A TOML basic string: backslash and double-quote escaped."""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _toml_array(values: Sequence[object]) -> str:
    return "[" + ", ".join(_toml_str(v) for v in values) + "]"


def _toml_value(value: object) -> str:
    """Serialize a scalar for our closed schema (bool / int / float / str)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return _toml_str(value)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --frozen pytest tests/aggregate_quantification/test_config_writer.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/aggregate_quantification/config.py tests/aggregate_quantification/test_config_writer.py
git commit -m "feat(aggregate): add write_config — author run-config TOML from values

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N3vuJPVYYuYWh6EQLh8vrv"
```

---

## Task 2: `author_config` + `run()` progress (headless)

**Files:**
- Modify: `src/cellflow/aggregate_quantification/pipeline.py`
- Test: `tests/aggregate_quantification/test_author_config.py`, `tests/aggregate_quantification/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/aggregate_quantification/test_author_config.py`:

```python
"""author_config — write catalog.csv + config.toml, ready for run()."""
from __future__ import annotations

from cellflow.aggregate_quantification.config import NlsConfig, load_config
from cellflow.aggregate_quantification.pipeline import author_config


def _record(tmp_path, pid="p1"):
    pdir = tmp_path / "study" / pid
    pdir.mkdir(parents=True, exist_ok=True)
    return {"id": pid, "condition": "ctrl", "date": "2026-06-22",
            "position_path": pdir, "notes": ""}


def test_writes_both_files_into_out_dir(tmp_path):
    out = tmp_path / "study"
    records = [_record(tmp_path)]
    config_path = author_config(out, records, quantities=["contacts"])
    assert config_path == out / "config.toml"
    assert (out / "catalog.csv").is_file()
    cfg = load_config(config_path)
    assert cfg.catalog == (out / "catalog.csv").resolve()
    assert cfg.quantities == ("contacts",)


def test_creates_missing_out_dir(tmp_path):
    out = tmp_path / "fresh"
    author_config(out, [_record(tmp_path)], quantities=["contacts"])
    assert (out / "config.toml").is_file()


def test_threads_knobs_into_config(tmp_path):
    out = tmp_path / "study"
    config_path = author_config(
        out, [_record(tmp_path)],
        quantities=["contacts"],
        params={"pixel_size_um": 0.25, "shuffles": 1000},
        nls=NlsConfig(enabled=True, method="auto"),
        render_plots=True, plot_formats=["png"],
    )
    cfg = load_config(config_path)
    assert cfg.params == {"pixel_size_um": 0.25, "shuffles": 1000}
    assert cfg.nls is not None and cfg.nls.enabled is True
    assert cfg.render_plots is True and cfg.plot_formats == ("png",)
```

Add to `tests/aggregate_quantification/test_pipeline.py` (a new test; keep existing ones):

```python
def test_run_forwards_progress_cb(tmp_path, monkeypatch):
    """run(progress_cb=...) threads the callback into build_quantities."""
    from cellflow.aggregate_quantification import pipeline

    seen = {}

    def fake_build(catalog, *, quantifiers=None, params=None, progress_cb=None):
        seen["build"] = progress_cb

    monkeypatch.setattr(pipeline, "build_quantities", fake_build)
    monkeypatch.setattr(pipeline, "classify", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "aggregate", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "load_catalog", lambda p: [])

    cfg = tmp_path / "config.toml"
    cfg.write_text('catalog = "catalog.csv"\n', encoding="utf-8")

    def cb(done, total, label):  # pragma: no cover - identity check only
        pass

    pipeline.run(cfg, progress_cb=cb)
    assert seen["build"] is cb
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen pytest tests/aggregate_quantification/test_author_config.py tests/aggregate_quantification/test_pipeline.py::test_run_forwards_progress_cb -q`
Expected: FAIL (`author_config` import error; `run()` rejects `progress_cb`).

- [ ] **Step 3: Implement**

In `src/cellflow/aggregate_quantification/pipeline.py`:

Update imports — add `write_config` and `save_catalog` is already imported; confirm the config import line reads:

```python
from .config import NlsConfig, RunConfig, load_config, write_config
```

Add `"author_config"` to `__all__` (after `"build_catalog"` is fine; keep list readable).

Append `author_config` near `run()` (before it):

```python
def author_config(
    out_dir: Path | str,
    records: Sequence[dict],
    *,
    quantities: Sequence[str] = (),
    params: Mapping[str, object] | None = None,
    nls: NlsConfig | None = None,
    render_plots: bool = False,
    plot_formats: Sequence[str] = ("png", "svg"),
    catalog_name: str = "catalog.csv",
    config_name: str = "config.toml",
) -> Path:
    """Write ``catalog.csv`` + ``config.toml`` into *out_dir*; return the config path.

    The composition point behind the studio's "Save config…" / "Run": persist the
    in-memory *records* to a catalog CSV, then author a run-config beside it that
    points at that CSV (a relative ``catalog`` key, so the folder stays
    relocatable). ``run(author_config(...))`` reproduces the UI's run headlessly.
    Creates *out_dir* if missing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_catalog(out_dir / catalog_name, records)
    return write_config(
        out_dir / config_name,
        catalog=catalog_name,
        quantities=quantities,
        params=params,
        nls=nls,
        render_plots=render_plots,
        plot_formats=plot_formats,
    )
```

Update `run()`'s signature and the two forwarding calls:

```python
def run(config_path: Path | str, *, progress_cb=None) -> list[Path]:
```

Then in the body, `build_quantities(... )` call gains `progress_cb=progress_cb` and `classify(catalog, config=cfg.nls)` becomes `classify(catalog, config=cfg.nls, progress_cb=progress_cb)`. (Update the docstring's first line to note the optional `progress_cb` is forwarded to the build + classify stages.)

`save_catalog` is already imported at the top (`from .catalog import discover_catalog_entries, load_catalog, save_catalog`). Confirm; if not, add it.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --frozen pytest tests/aggregate_quantification/test_author_config.py tests/aggregate_quantification/test_pipeline.py -q`
Expected: PASS (all author_config tests + the full pipeline suite green).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/aggregate_quantification/pipeline.py tests/aggregate_quantification/test_author_config.py tests/aggregate_quantification/test_pipeline.py
git commit -m "feat(aggregate): author_config + run() progress_cb pass-through

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N3vuJPVYYuYWh6EQLh8vrv"
```

---

## Task 3: `RunArea` widget (napari)

**Files:**
- Create: `src/cellflow/napari/aggregate_quantification_run_area.py`
- Test: `tests/napari/test_run_area.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/napari/test_run_area.py`:

```python
"""RunArea — the studio's single Run section (controls + RunChoices)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.config import NlsConfig
from cellflow.napari.aggregate_quantification_run_area import RunArea, RunChoices


def _app():
    return QApplication.instance() or QApplication([])


class _Ctx:
    def __init__(self, records):
        self.records = records


def _area():
    return RunArea(save_callback=lambda c: None, run_callback=lambda c: None)


def test_quantities_default_all_checked():
    app = _app()
    area = _area()
    choices = area.choices()
    assert isinstance(choices, RunChoices)
    assert len(choices.quantities) >= 1  # every registered quantifier, checked
    area.deleteLater()
    app.processEvents()


def test_nls_off_by_default_gives_none():
    app = _app()
    area = _area()
    assert area.choices().nls is None
    area.deleteLater()
    app.processEvents()


def test_nls_enabled_populates_config():
    app = _app()
    area = _area()
    area._nls_enabled.setChecked(True)
    area._nls_image.setText("0_input/NLS_zavg.tif")
    area._nls_method.setCurrentText("otsu")
    nls = area.choices().nls
    assert nls == NlsConfig(enabled=True, image="0_input/NLS_zavg.tif",
                            method="otsu", threshold=0.0)
    area.deleteLater()
    app.processEvents()


def test_plots_choices():
    app = _app()
    area = _area()
    area._render_plots.setChecked(True)
    area._formats.setText("png, pdf")
    choices = area.choices()
    assert choices.render_plots is True
    assert choices.plot_formats == ("png", "pdf")
    area.deleteLater()
    app.processEvents()


def test_buttons_disabled_with_no_records():
    app = _app()
    area = _area()
    area.set_context(_Ctx([]))
    assert not area._run_btn.isEnabled()
    assert not area._save_btn.isEnabled()
    area.set_context(_Ctx([{"id": "p1"}]))
    assert area._run_btn.isEnabled()
    assert area._save_btn.isEnabled()
    area.deleteLater()
    app.processEvents()


def test_zero_quantities_disables_actions():
    app = _app()
    area = _area()
    area.set_context(_Ctx([{"id": "p1"}]))
    for cb in area._quantity_checks.values():
        cb.setChecked(False)
    assert not area._run_btn.isEnabled()
    area.deleteLater()
    app.processEvents()


def test_run_button_invokes_callback_with_choices():
    app = _app()
    seen = []
    area = RunArea(save_callback=lambda c: None,
                   run_callback=lambda c: seen.append(c))
    area.set_context(_Ctx([{"id": "p1"}]))
    area._run_btn.click()
    assert len(seen) == 1 and isinstance(seen[0], RunChoices)
    area.deleteLater()
    app.processEvents()


def test_save_button_invokes_callback_with_choices():
    app = _app()
    seen = []
    area = RunArea(save_callback=lambda c: seen.append(c),
                   run_callback=lambda c: None)
    area.set_context(_Ctx([{"id": "p1"}]))
    area._save_btn.click()
    assert len(seen) == 1 and isinstance(seen[0], RunChoices)
    area.deleteLater()
    app.processEvents()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen pytest tests/napari/test_run_area.py -q`
Expected: FAIL with `ModuleNotFoundError` for `aggregate_quantification_run_area`.

- [ ] **Step 3: Implement `RunArea` + `RunChoices`**

Create `src/cellflow/napari/aggregate_quantification_run_area.py`:

```python
"""The Aggregate Quantification studio's single Run section.

One section replaces the old piecemeal Compute + Aggregate areas: it gathers the
run-level choices (which quantities, the optional NLS step, plot rendering) and
hands them — as a :class:`RunChoices` — to the studio, which authors
``catalog.csv`` + ``config.toml`` and drives :func:`pipeline.run`. The shared
**Parameters** bar supplies ``[params]``; this widget owns only the run-level
knobs and the Save/Run controls. Reading state into a plain value keeps the
authoring + threading testable without Qt.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.config import NlsConfig
from cellflow.aggregate_quantification.quantifier import available_quantifiers
from cellflow.napari.ui_style import action_button, parameter_heading, status_label

#: The NLS thresholding methods (mirrors config._NLS_METHODS, in UI order).
_NLS_METHODS = ("auto", "otsu", "two_cluster", "fixed")
_DEFAULT_NLS_IMAGE = "0_input/NLS_zavg.tif"


@dataclass
class RunChoices:
    """The run-level selections the studio threads into ``author_config``."""

    quantities: tuple[str, ...]
    nls: NlsConfig | None
    render_plots: bool
    plot_formats: tuple[str, ...]


class RunArea(QWidget):
    """Quantity / NLS / plots controls + Save config… and Run buttons."""

    def __init__(
        self,
        save_callback: Callable[[RunChoices], None],
        run_callback: Callable[[RunChoices], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._save_callback = save_callback
        self._run_callback = run_callback
        self._records: list[dict] = []
        self._quantity_checks: dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        intro = QLabel(
            "Author catalog.csv + config.toml from the whole catalogue and run the "
            "pipeline (build → aggregate → export). Save config… writes the files "
            "without running."
        )
        intro.setWordWrap(True)
        status_label(intro, muted=True)
        layout.addWidget(intro)

        self._build_quantities(layout)
        self._build_nls(layout)
        self._build_plots(layout)
        self._build_buttons(layout)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        status_label(self._status, muted=True)
        layout.addWidget(self._status)

        self._refresh_enabled()

    # ----------------------------------------------------------------- sections
    def _build_quantities(self, layout) -> None:
        heading = QLabel("QUANTITIES")
        parameter_heading(heading)
        layout.addWidget(heading)
        for q_cls in available_quantifiers():
            cb = QCheckBox(q_cls.display_name or q_cls.quantity_id)
            cb.setChecked(True)
            cb.toggled.connect(lambda *_: self._refresh_enabled())
            layout.addWidget(cb)
            self._quantity_checks[q_cls.quantity_id] = cb

    def _build_nls(self, layout) -> None:
        heading = QLabel("NLS CLASSIFICATION")
        parameter_heading(heading)
        layout.addWidget(heading)
        self._nls_enabled = QCheckBox("Classify NLS subpopulations")
        layout.addWidget(self._nls_enabled)

        grid = QGridLayout()
        grid.setContentsMargins(12, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.addWidget(QLabel("method"), 0, 0)
        self._nls_method = QComboBox()
        self._nls_method.addItems(_NLS_METHODS)
        grid.addWidget(self._nls_method, 0, 1)
        grid.addWidget(QLabel("image"), 1, 0)
        self._nls_image = QLineEdit(_DEFAULT_NLS_IMAGE)
        grid.addWidget(self._nls_image, 1, 1)
        grid.addWidget(QLabel("threshold"), 2, 0)
        self._nls_threshold = QLineEdit("0.0")
        self._nls_threshold.setToolTip("Used when method = fixed.")
        grid.addWidget(self._nls_threshold, 2, 1)
        layout.addLayout(grid)

    def _build_plots(self, layout) -> None:
        heading = QLabel("PLOTS")
        parameter_heading(heading)
        layout.addWidget(heading)
        self._render_plots = QCheckBox("Render figures from the .iris bundles")
        layout.addWidget(self._render_plots)
        row = QHBoxLayout()
        row.setContentsMargins(12, 0, 0, 0)
        row.addWidget(QLabel("formats"))
        self._formats = QLineEdit("png, svg")
        row.addWidget(self._formats, 1)
        layout.addLayout(row)

    def _build_buttons(self, layout) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._save_btn = QPushButton("Save config…")
        self._save_btn.setToolTip("Write catalog.csv + config.toml without running.")
        action_button(self._save_btn)
        self._save_btn.clicked.connect(self._on_save)
        self._run_btn = QPushButton("Run ▶")
        self._run_btn.setToolTip("Write the files, then run the whole pipeline.")
        action_button(self._run_btn, expand=True)
        self._run_btn.clicked.connect(self._on_run)
        row.addWidget(self._save_btn)
        row.addWidget(self._run_btn, 1)
        layout.addLayout(row)

    # -------------------------------------------------------------------- state
    def choices(self) -> RunChoices:
        quantities = tuple(
            qid for qid, cb in self._quantity_checks.items() if cb.isChecked()
        )
        nls = None
        if self._nls_enabled.isChecked():
            nls = NlsConfig(
                enabled=True,
                image=self._nls_image.text().strip() or _DEFAULT_NLS_IMAGE,
                method=self._nls_method.currentText(),
                threshold=_parse_float(self._nls_threshold.text()),
            )
        formats = tuple(
            part.strip() for part in self._formats.text().split(",") if part.strip()
        )
        return RunChoices(
            quantities=quantities,
            nls=nls,
            render_plots=self._render_plots.isChecked(),
            plot_formats=formats or ("png", "svg"),
        )

    def set_context(self, ctx: object) -> None:
        self._records = list(getattr(ctx, "records", []))
        self._refresh_enabled()

    def set_status(self, message: str) -> None:
        self._status.setText(message)

    def _refresh_enabled(self) -> None:
        ready = bool(self._records) and any(
            cb.isChecked() for cb in self._quantity_checks.values()
        )
        self._run_btn.setEnabled(ready)
        self._save_btn.setEnabled(ready)

    def _on_save(self) -> None:
        if self._save_btn.isEnabled():
            self._save_callback(self.choices())

    def _on_run(self) -> None:
        if self._run_btn.isEnabled():
            self._run_callback(self.choices())


def _parse_float(text: str) -> float:
    try:
        return float(text.strip())
    except ValueError:
        return 0.0
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --frozen pytest tests/napari/test_run_area.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/aggregate_quantification_run_area.py tests/napari/test_run_area.py
git commit -m "feat(napari): RunArea — single Run section authoring run-level choices

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N3vuJPVYYuYWh6EQLh8vrv"
```

---

## Task 4: Wire RunArea into the studio; retire Compute/Aggregate

**Files:**
- Modify: `src/cellflow/napari/aggregate_quantification_studio.py`
- Delete: `src/cellflow/napari/aggregate_quantification_aggregate_area.py`, `tests/napari/test_aggregate_area.py`
- Test: `tests/napari/test_aggregate_quantification_studio.py`

- [ ] **Step 1: Update the studio test (retire Build/Aggregate coupling, characterize Run)**

The studio test (`tests/napari/test_aggregate_quantification_studio.py`) couples to
the machinery this task removes. Read it first, then apply **all** of the
following. The module is imported as `mod`; tests construct
`mod.AggregateQuantificationStudioWidget()` directly (no fixture).

**1a. `test_every_tool_is_its_own_collapsible_collapsed`** — delete the two
trailing lines that assert the Build area:

```python
    # The Build area carries a metric row per quantifier instead.
    assert "contacts" in widget._build_area._rows
```

and replace them with a Run-area assertion (the Run area carries a checkbox per
quantifier):

```python
    # The Run area carries a quantity checkbox per quantifier instead.
    assert "contacts" in widget._run_area._quantity_checks
```

**1b. `test_add_is_register_only_no_build`** — drop the now-gone build hooks. Remove
this line:

```python
    began: list = []
    monkeypatch.setattr(widget, "_begin_build", lambda *a, **k: began.append(a))
```

and remove the two build-worker assertions:

```python
    # No build kicked off; both positions registered, statuses reflect reality.
    assert began == []
    assert widget._build_worker is None
```

Leave the rest (it asserts Add registers positions + statuses). Update the leading
comment line to `# Add registers positions; building is the Run section's job.`

**1c. Delete two build-only tests entirely** (their methods no longer exist):
`test_build_area_run_delegates_to_pipeline` and
`test_build_status_goes_to_compute_section_not_catalogue`.

**1d. `test_section_defaults_and_compute_rename`** — rewrite its body for the new
sections (Run replaces Build/Compute + Aggregate; Run is expanded by default):

```python
def test_section_defaults_and_compute_rename(monkeypatch):
    # The piecemeal Build/Compute + Aggregate sections are replaced by one Run
    # section (expanded by default). Plots stay removed (Iris-only visualization).
    from cellflow.napari.widgets import CollapsibleSection

    app = _app()
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [])
    widget = mod.AggregateQuantificationStudioWidget()
    by_title = {s.title: s for s in widget.findChildren(CollapsibleSection)}
    assert "Run" in by_title
    assert "Build" not in by_title and "Compute" not in by_title
    assert "Aggregate" not in by_title
    assert "Plots" not in by_title
    assert by_title["Parameters"].is_expanded is True
    assert by_title["Tools"].is_expanded is False
    assert by_title["Run"].is_expanded is True
    widget.deleteLater()
    app.processEvents()
```

**1e. Add a Run-wiring test** that the Run button authors a config at the catalogue
root and dispatches `run` (the studio module imports both names, so patch them on
`mod`):

```python
def test_run_section_authors_config_and_dispatches_run(tmp_path, monkeypatch):
    app = _app()
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [])

    pdir = tmp_path / "study" / "p1"
    pdir.mkdir(parents=True)
    widget = mod.AggregateQuantificationStudioWidget()
    widget._records = [
        {"id": "p1", "position_path": pdir, "condition": "ctrl",
         "date": "d", "notes": ""},
    ]
    widget._refresh_table()  # feeds the Run area the catalogue scope

    seen = {}

    def fake_author(out_dir, records, **kw):
        seen["out_dir"] = out_dir
        seen["kw"] = kw
        return out_dir / "config.toml"

    ran = {}
    monkeypatch.setattr(mod, "author_config", fake_author)
    monkeypatch.setattr(mod, "run", lambda p, progress_cb=None: ran.setdefault("p", p) or [])

    widget._run_area._on_run()
    assert seen["out_dir"] == pdir.parent  # catalogue root
    assert "quantities" in seen["kw"] and "params" in seen["kw"]

    widget.deleteLater()
    app.processEvents()
```

(The Run handler authors the config synchronously, then dispatches `run` on a
`thread_worker`; the test only needs `author_config` to have been called with the
catalogue root + choices. If the worker thread makes `run` assertions flaky, drop
the `ran` checks — asserting `author_config` is sufficient.)

- [ ] **Step 2: Run to verify the studio test fails**

Run: `uv run --frozen pytest tests/napari/test_aggregate_quantification_studio.py -q`
Expected: FAIL (asserts `"Run"`, missing `_run_area` / `author_config` symbol).

- [ ] **Step 3: Rewire the studio**

In `src/cellflow/napari/aggregate_quantification_studio.py`:

Imports — replace the build/aggregate imports. Remove:

```python
from cellflow.aggregate_quantification.pipeline import build_quantities
from cellflow.aggregate_quantification.shape_tables import aggregate as aggregate_tables
from cellflow.napari.aggregate_quantification_aggregate_area import AggregateArea
```

and remove `BuildArea` from the `studio_plugins` import (keep `PluginEntry`,
`available_tool_plugins`, `records_satisfying`). Add:

```python
from cellflow.aggregate_quantification.pipeline import author_config, run
from cellflow.napari.aggregate_quantification_run_area import RunArea
```

Keep `from cellflow.aggregate_quantification.shape_tables import catalogue_root`
and `from cellflow.aggregate_quantification.quantifier import (Quantifier, available_quantifiers)` — though `Quantifier` may become unused; drop it if so.

`__init__` — replace the build/aggregate worker state:

Remove these lines:

```python
        #: Background build triggered by a builder plugin.
        self._build_worker = None
        self._build_emitter = _ProgressEmitter(self)
        self._build_emitter.progress.connect(self._on_build_progress)
        #: Background aggregation (pool built products into the shape tables).
        self._aggregate_worker = None
```

with:

```python
        #: Background full-pipeline run (author config, then pipeline.run()).
        self._run_worker = None
        self._run_emitter = _ProgressEmitter(self)
        self._run_emitter.progress.connect(self._on_run_progress)
```

Replace the section-build calls. Find:

```python
        self._build_tools_section(layout)
        self._build_build_section(layout)
        self._build_aggregate_section(layout)
```

with:

```python
        self._build_tools_section(layout)
        self._build_run_section(layout)
```

Delete the methods `_build_build_section`, `_build_aggregate_section`,
`_set_build_status`, `_run_quantity_builds`, `_begin_build`, `_on_build_progress`,
`_on_build_done`, `_on_build_error`, `_run_aggregate`, `_on_aggregate_done`,
`_on_aggregate_error`, and `_reload_build_area`.

Add the Run section + handlers (place near where the old build section methods were):

```python
    def _build_run_section(self, layout) -> None:
        """The single Run area: author catalog.csv + config.toml from the whole
        catalogue and drive ``pipeline.run`` on a worker. Re-created in
        :meth:`_reload_run_area` so a runtime-registered quantity appears."""
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        self._run_host = QWidget()
        self._run_host_layout = QVBoxLayout(self._run_host)
        self._run_host_layout.setContentsMargins(0, 0, 0, 0)
        self._run_host_layout.setSpacing(0)
        self._run_area: RunArea | None = None
        col.addWidget(self._run_host)
        layout.addWidget(CollapsibleSection("Run", container, expanded=True))

    def _reload_run_area(self) -> None:
        """(Re)create the Run area body from the quantifier registry."""
        while self._run_host_layout.count():
            item = self._run_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._run_area = RunArea(
            save_callback=self._on_run_save,
            run_callback=self._on_run_execute,
        )
        self._run_host_layout.addWidget(self._run_area)
        self._push_context_to(self._run_area)

    def _author_run_config(self, choices) -> Path:
        """Write catalog.csv + config.toml for the whole catalogue; return config path."""
        out_dir = catalogue_root(self._records)
        params = self._shared_params.build_params()
        return author_config(
            out_dir,
            self._records,
            quantities=choices.quantities,
            params=params,
            nls=choices.nls,
            render_plots=choices.render_plots,
            plot_formats=choices.plot_formats,
        )

    def _on_run_save(self, choices) -> None:
        if not self._records:
            self._run_area.set_status("Add positions to the catalogue first.")
            return
        try:
            config_path = self._author_run_config(choices)
        except Exception as exc:  # noqa: BLE001 - surface authoring errors in the UI
            self._run_area.set_status(f"Save error: {exc}")
            return
        self._run_area.set_status(f"Wrote {config_path.name} + catalog.csv.")

    def _on_run_execute(self, choices) -> None:
        if self._run_worker is not None:
            self._run_area.set_status("A run is already in progress.")
            return
        if not self._records:
            self._run_area.set_status("Add positions to the catalogue first.")
            return
        try:
            config_path = self._author_run_config(choices)
        except Exception as exc:  # noqa: BLE001 - surface authoring errors in the UI
            self._run_area.set_status(f"Save error: {exc}")
            return
        self._run_area.set_status("Running pipeline…")
        emit = self._run_emitter.progress.emit

        @thread_worker(
            connect={"returned": self._on_run_done, "errored": self._on_run_error}
        )
        def _worker():
            return run(config_path, progress_cb=emit)

        self._run_worker = _worker()

    def _on_run_progress(self, done: int, total: int, label: str) -> None:
        self._run_area.set_status(f"Running: {done}/{total} {label}")

    def _on_run_done(self, written: list) -> None:
        self._run_worker = None
        n = len(written)
        self._run_area.set_status(
            f"Exported {n} .iris → export/." if n else "Run finished; nothing exported."
        )

    def _on_run_error(self, exc: Exception) -> None:
        self._run_worker = None
        self._run_area.set_status(f"Run error: {exc}")
```

In `_reload_plugins`, replace the `self._reload_build_area()` call with
`self._reload_run_area()`.

In `_push_context`, replace the build/aggregate context pushes:

```python
        build_area = getattr(self, "_build_area", None)
        if build_area is not None:
            self._push_context_to(build_area)
        aggregate_area = getattr(self, "_aggregate_area", None)
        if aggregate_area is not None:
            self._push_context_to(aggregate_area)
```

with:

```python
        run_area = getattr(self, "_run_area", None)
        if run_area is not None:
            self._push_context_to(run_area)
```

Add `from pathlib import Path` is already imported (top of file) for the
`_author_run_config` return annotation.

- [ ] **Step 4: Delete the retired aggregate area + its test**

```bash
git rm src/cellflow/napari/aggregate_quantification_aggregate_area.py tests/napari/test_aggregate_area.py
```

- [ ] **Step 5: Run the studio + run-area + headless suites**

Run: `uv run --frozen pytest tests/napari/test_aggregate_quantification_studio.py tests/napari/test_run_area.py tests/aggregate_quantification/ -q`
Expected: PASS. (If a leftover reference to a deleted method/import remains, fix it.)

- [ ] **Step 6: Full napari + aggregate sweep (catch orphans)**

Run: `uv run --frozen pytest tests/napari tests/aggregate_quantification -q`
Expected: PASS (the previously-known `test_aggregate_area` failure is gone with the file). Investigate any new failure — a green sweep is the gate.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(napari): single Run section drives pipeline.run from UI-authored config

Replaces the piecemeal Compute (BuildArea) + Aggregate (AggregateArea) sections
with one Run section that authors catalog.csv + config.toml and runs the whole
pipeline. Deletes the aggregate-only AggregateArea module + test.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N3vuJPVYYuYWh6EQLh8vrv"
```

---

## Final review

After all four tasks: dispatch a final code reviewer over the whole branch diff
(`git diff main...HEAD`), confirm `make_aggregate_quantification_studio` still
imports cleanly, and that `available_analysis_plugins()` is unchanged
(`catalog_summary`, `curation`, `visualize_contacts`). Then merge via
finishing-a-development-branch.
```
