"""Tests for cellflow.napari.meta_widget.MetaSourceBrowserWidget."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QComboBox, QPushButton


# ---------------------------------------------------------------------------
# fake viewer (mirrors tests/napari/test_analysis_widget.py pattern)
# ---------------------------------------------------------------------------

class _LayerCollection(dict):
    def remove(self, layer):
        name = getattr(layer, "name", layer)
        self.pop(name, None)


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()

    def add_points(self, data, *, name, **kwargs):
        layer = types.SimpleNamespace(data=data, name=name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_labels(self, data, *, name, **kwargs):
        layer = types.SimpleNamespace(data=data, name=name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_shapes(self, data, *, name, **kwargs):
        layer = types.SimpleNamespace(data=data, name=name, **kwargs)
        self.layers[name] = layer
        return layer


# ---------------------------------------------------------------------------
# module loader
# ---------------------------------------------------------------------------

def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.meta_widget", None)
    return importlib.import_module("cellflow.napari.meta_widget")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_ready_position(root: Path, condition: str, experiment: str, position: str) -> Path:
    """Create a position with all required files and return its path."""
    pos_dir = root / condition / experiment / position
    (pos_dir / "4_analysis").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "3_cell").mkdir()
    (pos_dir / "4_analysis" / "position_analysis.h5").touch()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    return pos_dir


def _make_position_missing_artifact(root: Path, condition: str, experiment: str, position: str) -> Path:
    """Create a position with labels but no artifact."""
    pos_dir = root / condition / experiment / position
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir(parents=True)
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    return pos_dir


def _combo_items(combo: QComboBox) -> list[str]:
    """Return all visible item texts from a QComboBox (excluding separators)."""
    return [combo.itemText(i) for i in range(combo.count())]


def _combo_current(combo: QComboBox) -> str:
    return combo.currentText()


# ---------------------------------------------------------------------------
# widget instantiation
# ---------------------------------------------------------------------------

def test_widget_creates_selector_combos_and_load_button(monkeypatch):
    """Widget should expose selectors, catalog action buttons, and a Load Source button."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.MetaSourceBrowserWidget()

    assert hasattr(widget, "condition_combo")
    assert isinstance(widget.condition_combo, QComboBox)
    assert hasattr(widget, "experiment_combo")
    assert isinstance(widget.experiment_combo, QComboBox)
    assert hasattr(widget, "position_combo")
    assert isinstance(widget.position_combo, QComboBox)
    assert hasattr(widget, "load_source_btn")
    assert isinstance(widget.load_source_btn, QPushButton)
    assert widget.load_source_btn.text() == "Load Source"
    assert hasattr(widget, "open_catalog_btn")
    assert isinstance(widget.open_catalog_btn, QPushButton)
    assert widget.open_catalog_btn.text() == "Open catalog"
    assert hasattr(widget, "save_catalog_btn")
    assert isinstance(widget.save_catalog_btn, QPushButton)
    assert widget.save_catalog_btn.text() == "Save catalog"
    assert hasattr(widget, "add_h5_btn")
    assert isinstance(widget.add_h5_btn, QPushButton)
    assert widget.add_h5_btn.text() == "Add H5"
    assert hasattr(widget, "autodiscover_folder_btn")
    assert isinstance(widget.autodiscover_folder_btn, QPushButton)
    assert widget.autodiscover_folder_btn.text() == "Autodiscover folder"

    widget.deleteLater()
    app.processEvents()


def test_widget_accepts_optional_viewer(monkeypatch):
    """Passing a viewer should store it for later use."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.MetaSourceBrowserWidget(viewer)

    assert widget.viewer is viewer

    widget.deleteLater()
    app.processEvents()


# ---------------------------------------------------------------------------
# refresh populating selectors
# ---------------------------------------------------------------------------

def test_refresh_populates_condition_combo_sorted(monkeypatch, tmp_path):
    """After refresh, the condition combo lists all conditions from the root."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.MetaSourceBrowserWidget()

    root = tmp_path / "study"
    _make_ready_position(root, "wildtype", "day1", "pos00")
    _make_ready_position(root, "mutant", "day1", "pos00")

    widget.refresh(root)

    assert _combo_items(widget.condition_combo) == ["mutant", "wildtype"]

    widget.deleteLater()
    app.processEvents()


def test_refresh_populates_experiment_combo_for_selected_condition(monkeypatch, tmp_path):
    """Experiment combo should list experiments belonging to the currently selected condition."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.MetaSourceBrowserWidget()

    root = tmp_path / "study"
    _make_ready_position(root, "cond_a", "exp_1", "pos00")
    _make_ready_position(root, "cond_a", "exp_2", "pos00")
    _make_ready_position(root, "cond_b", "exp_3", "pos00")

    widget.refresh(root)

    # cond_a should be first alphabetically
    assert _combo_current(widget.condition_combo) == "cond_a"
    assert _combo_items(widget.experiment_combo) == ["exp_1", "exp_2"]

    widget.deleteLater()
    app.processEvents()


def test_refresh_populates_position_combo_for_selected_experiment(monkeypatch, tmp_path):
    """Position combo should list positions for the selected condition+experiment."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.MetaSourceBrowserWidget()

    root = tmp_path / "study"
    _make_ready_position(root, "c", "e", "pos00")
    _make_ready_position(root, "c", "e", "pos01")
    _make_ready_position(root, "c", "e", "pos02")

    widget.refresh(root)

    assert _combo_items(widget.position_combo) == ["pos00", "pos01", "pos02"]

    widget.deleteLater()
    app.processEvents()


def test_refresh_changing_condition_updates_experiment_and_position(monkeypatch, tmp_path):
    """When the user changes the condition, experiment and position combos update."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.MetaSourceBrowserWidget()

    root = tmp_path / "study"
    _make_ready_position(root, "cond_a", "exp_a1", "pos00")
    _make_ready_position(root, "cond_b", "exp_b1", "pos10")

    widget.refresh(root)

    # current condition is cond_a
    assert _combo_items(widget.experiment_combo) == ["exp_a1"]
    assert _combo_items(widget.position_combo) == ["pos00"]

    # switch to cond_b
    widget.condition_combo.setCurrentText("cond_b")
    assert _combo_items(widget.experiment_combo) == ["exp_b1"]
    assert _combo_items(widget.position_combo) == ["pos10"]

    widget.deleteLater()
    app.processEvents()


def test_refresh_changing_experiment_updates_position(monkeypatch, tmp_path):
    """When the user changes the experiment, the position combo updates."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.MetaSourceBrowserWidget()

    root = tmp_path / "study"
    _make_ready_position(root, "c", "exp_a", "pos00")
    _make_ready_position(root, "c", "exp_b", "pos10")

    widget.refresh(root)

    assert _combo_items(widget.position_combo) == ["pos00"]

    widget.experiment_combo.setCurrentText("exp_b")
    assert _combo_items(widget.position_combo) == ["pos10"]

    widget.deleteLater()
    app.processEvents()


def test_refresh_empty_root_clears_combos(monkeypatch, tmp_path):
    """Refreshing with an empty root should leave all combos empty."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.MetaSourceBrowserWidget()

    root = tmp_path / "study"
    _make_ready_position(root, "c", "e", "p")
    widget.refresh(root)
    assert len(_combo_items(widget.condition_combo)) == 1

    # refresh with empty root
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    widget.refresh(empty_root)

    assert _combo_items(widget.condition_combo) == []
    assert _combo_items(widget.experiment_combo) == []
    assert _combo_items(widget.position_combo) == []

    widget.deleteLater()
    app.processEvents()


# ---------------------------------------------------------------------------
# Load Source button enabled state
# ---------------------------------------------------------------------------

def test_load_source_enabled_when_selected_record_is_ready(monkeypatch, tmp_path):
    """Load Source button should be enabled only when the selected position is ready."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.MetaSourceBrowserWidget()

    root = tmp_path / "study"
    _make_ready_position(root, "c", "e", "pos00")

    widget.refresh(root)

    assert widget.load_source_btn.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


def test_load_source_disabled_when_selected_record_is_not_ready(monkeypatch, tmp_path):
    """Load Source button should be disabled when the position has missing files."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.MetaSourceBrowserWidget()

    root = tmp_path / "study"
    _make_position_missing_artifact(root, "c", "e", "pos00")

    widget.refresh(root)

    assert widget.load_source_btn.isEnabled() is False

    widget.deleteLater()
    app.processEvents()


def test_load_source_disabled_when_no_record_selected(monkeypatch, tmp_path):
    """Load Source should be disabled when the combos are empty."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.MetaSourceBrowserWidget()

    root = tmp_path / "study"
    root.mkdir()
    widget.refresh(root)

    assert widget.load_source_btn.isEnabled() is False

    widget.deleteLater()
    app.processEvents()


def test_load_source_enabled_updates_when_switching_to_ready_position(monkeypatch, tmp_path):
    """Toggling between a non-ready and ready position should update the button."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.MetaSourceBrowserWidget()

    root = tmp_path / "study"
    _make_ready_position(root, "c", "e", "pos_ready")
    _make_position_missing_artifact(root, "c", "e", "pos_bad")

    widget.refresh(root)

    # default is first alphabetically: pos_bad
    assert _combo_current(widget.position_combo) == "pos_bad"
    assert widget.load_source_btn.isEnabled() is False

    widget.position_combo.setCurrentText("pos_ready")
    assert widget.load_source_btn.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


# ---------------------------------------------------------------------------
# Load Source action
# ---------------------------------------------------------------------------

def test_load_source_reads_artifact_and_calls_add_artifact_layers(monkeypatch, tmp_path):
    """Clicking Load Source reads the selected artifact and visualizes with '[Meta] ' prefix."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.MetaSourceBrowserWidget(viewer)

    root = tmp_path / "study"
    pos_dir = _make_ready_position(root, "cond", "exp", "pos00")
    artifact_path = pos_dir / "4_analysis" / "position_analysis.h5"

    artifact = {"cells": [10, 20, 30]}
    read_calls = []
    add_calls = []

    def fake_read(path):
        read_calls.append(path)
        return artifact

    def fake_add(viewer_arg, artifact_arg, **kwargs):
        add_calls.append((viewer_arg, artifact_arg, kwargs))
        prefix = kwargs.get("prefix", "")
        viewer_arg.layers[f"{prefix}Cell labels"] = types.SimpleNamespace(
            name=f"{prefix}Cell labels", data=artifact_arg["cells"]
        )

    monkeypatch.setattr(mod, "read_position_artifact", fake_read)
    monkeypatch.setattr(mod, "add_artifact_layers", fake_add)

    widget.refresh(root)
    widget.load_source_btn.click()

    assert read_calls == [artifact_path]
    assert len(add_calls) == 1
    assert add_calls[0][0] is viewer
    assert add_calls[0][1] is artifact
    assert add_calls[0][2]["prefix"] == "[Meta] "
    assert "[Meta] Cell labels" in viewer.layers

    widget.deleteLater()
    app.processEvents()


def test_load_source_does_nothing_when_disabled(monkeypatch, tmp_path):
    """If Load Source is disabled, clicking it should not call any functions."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.MetaSourceBrowserWidget(viewer)

    root = tmp_path / "study"
    root.mkdir()

    read_calls = []
    add_calls = []

    monkeypatch.setattr(mod, "read_position_artifact", lambda p: read_calls.append(p))
    monkeypatch.setattr(mod, "add_artifact_layers", lambda v, a, **kw: add_calls.append(True))

    widget.refresh(root)
    assert widget.load_source_btn.isEnabled() is False

    widget.load_source_btn.click()

    assert read_calls == []
    assert add_calls == []

    widget.deleteLater()
    app.processEvents()


def test_load_source_uses_real_stored_record_paths(monkeypatch, tmp_path):
    """The record's artifact_path and label paths are forwarded to read_position_artifact."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.MetaSourceBrowserWidget(viewer)

    root = tmp_path / "study"
    pos_dir = _make_ready_position(root, "c", "e", "p")
    artifact_path = pos_dir / "4_analysis" / "position_analysis.h5"

    read_calls = []

    def fake_read(path):
        read_calls.append(path)
        return {"cells": []}

    monkeypatch.setattr(mod, "read_position_artifact", fake_read)
    monkeypatch.setattr(mod, "add_artifact_layers", lambda v, a, **kw: None)

    widget.refresh(root)
    widget.load_source_btn.click()

    assert read_calls == [artifact_path]

    widget.deleteLater()
    app.processEvents()


# ---------------------------------------------------------------------------
# CSV catalog actions
# ---------------------------------------------------------------------------

def test_open_catalog_loads_csv_and_populates_selectors(monkeypatch, tmp_path):
    """Opening a CSV catalog should populate the same cascading selectors."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    source = tmp_path / "position_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "path,date,condition,id,labels\n"
        "position_analysis.h5,day1,treated,pos00,\n"
    )
    widget = mod.MetaSourceBrowserWidget()

    monkeypatch.setattr(
        mod.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(csv_path), "CSV Files (*.csv)"),
    )

    widget.open_catalog_btn.click()

    assert widget._csv_path == csv_path
    assert _combo_items(widget.condition_combo) == ["treated"]
    assert _combo_items(widget.experiment_combo) == ["day1"]
    assert _combo_items(widget.position_combo) == ["pos00"]
    assert widget.load_source_btn.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


def test_add_h5_appends_ready_record(monkeypatch, tmp_path):
    """Adding one H5 file should append a generated ready catalog record."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    source = tmp_path / "position_analysis.h5"
    source.touch()
    widget = mod.MetaSourceBrowserWidget()

    monkeypatch.setattr(
        mod.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(source), "H5 Files (*.h5 *.hdf5)"),
    )

    widget.add_h5_btn.click()

    assert len(widget._records) == 1
    assert widget._records[0]["artifact_path"] == source
    assert widget._records[0]["analysis_status"] == "ready"
    assert _combo_items(widget.condition_combo) == ["unknown_condition"]
    assert _combo_items(widget.experiment_combo) == ["unknown_date"]
    assert _combo_items(widget.position_combo) == ["position_analysis"]

    widget.deleteLater()
    app.processEvents()


def test_autodiscover_folder_appends_multiple_h5_records_and_skips_duplicates(monkeypatch, tmp_path):
    """Autodiscovery should recursively add H5 files while keeping existing records unique."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    root = tmp_path / "study"
    first = root / "pos_a" / "position_analysis.h5"
    second = root / "pos_b" / "position_analysis.h5"
    first.parent.mkdir(parents=True)
    second.parent.mkdir()
    first.touch()
    second.touch()
    widget = mod.MetaSourceBrowserWidget()
    widget._set_records(mod.records_from_h5_paths([first]))

    monkeypatch.setattr(
        mod.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(root),
    )

    widget.autodiscover_folder_btn.click()

    assert [record["artifact_path"] for record in widget._records] == [first, second]
    assert _combo_items(widget.position_combo) == ["pos_a", "pos_b"]

    widget.deleteLater()
    app.processEvents()


def test_save_catalog_writes_current_records_to_active_csv(monkeypatch, tmp_path):
    """Saving with an active CSV path should write through the backend helper."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    source = tmp_path / "position_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    widget = mod.MetaSourceBrowserWidget()
    widget._csv_path = csv_path
    widget._set_records(mod.records_from_h5_paths([source]))

    widget.save_catalog_btn.click()

    assert csv_path.exists()
    assert "position_analysis.h5" in csv_path.read_text()

    widget.deleteLater()
    app.processEvents()


def test_save_catalog_prompts_for_path_when_no_active_csv(monkeypatch, tmp_path):
    """Saving a new catalog should ask for a destination path once."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    source = tmp_path / "position_analysis.h5"
    source.touch()
    csv_path = tmp_path / "new_catalog.csv"
    widget = mod.MetaSourceBrowserWidget()
    widget._set_records(mod.records_from_h5_paths([source]))

    monkeypatch.setattr(
        mod.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(csv_path), "CSV Files (*.csv)"),
    )

    widget.save_catalog_btn.click()

    assert widget._csv_path == csv_path
    assert csv_path.exists()

    widget.deleteLater()
    app.processEvents()
