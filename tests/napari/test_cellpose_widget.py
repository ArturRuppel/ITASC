"""Tests for the local Cellpose widget."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QToolButton


class _LayerCollection(dict):
    def remove(self, layer):
        self.pop(layer.name, None)


class _FakeEvent:
    def connect(self, cb):
        pass

    def disconnect(self, cb):
        pass


class _FakeEvents:
    def __init__(self) -> None:
        self.data = _FakeEvent()
        self.paint = _FakeEvent()
        self.mode = _FakeEvent()
        self.removed = _FakeEvent()


class _FakeSelection:
    def __init__(self) -> None:
        self.active = None


class _FakeLayer:
    def __init__(self, data, name, **kwargs) -> None:
        self.data = np.asarray(data)
        self.name = name
        self.events = _FakeEvents()
        self.kwargs = kwargs


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()
        self.layers.selection = _FakeSelection()
        self.layers.events = _FakeEvents()
        self.dims = SimpleNamespace(
            current_step=(0, 0),
            events=SimpleNamespace(
                current_step=SimpleNamespace(connect=lambda cb: None)
            ),
        )

    def add_image(self, data, *, name, **kwargs):
        layer = _FakeLayer(data, name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_labels(self, data, *, name, **kwargs):
        layer = _FakeLayer(data, name, **kwargs)
        self.layers[name] = layer
        return layer


@pytest.fixture
def _mock_cellpose(monkeypatch):
    """Install a fake cellpose so the runner imports cleanly."""
    fake_cellpose = types.ModuleType("cellpose")
    fake_models = types.ModuleType("cellpose.models")

    class _FakeModel:
        def __init__(self, *_, **__):
            pass

        def eval(self, img, **_kwargs):
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 2:
                dp = np.zeros((2, *arr.shape), dtype=np.float32)
                prob = np.zeros(arr.shape, dtype=np.float32)
            else:
                dp = np.zeros((3, *arr.shape), dtype=np.float32)
                prob = np.zeros(arr.shape, dtype=np.float32)
            return None, (None, dp, prob), None

    fake_models.CellposeModel = _FakeModel
    fake_cellpose.models = fake_models
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)
    monkeypatch.delitem(sys.modules, "cellflow.segmentation.cellpose_runner", raising=False)


def _load_widget(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.cellpose_widget", None)
    return importlib.import_module("cellflow.napari.cellpose_widget")


def test_widget_exposes_stage_rows_and_buttons(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    for name in (
        "nucleus_params_btn",
        "nucleus_preview_btn",
        "nucleus_run_btn",
        "cell_params_btn",
        "cell_preview_btn",
        "cell_run_btn",
    ):
        btn = getattr(w, name)
        assert isinstance(btn, QToolButton), name
    assert w.nucleus_run_btn.text() == "▶"
    assert w.cell_run_btn.text() == "▶"
    w.deleteLater()


def test_params_button_toggles_section(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    assert not w.nucleus_section.is_expanded
    w.nucleus_params_btn.setChecked(True)
    assert w.nucleus_section.is_expanded
    w.nucleus_params_btn.setChecked(False)
    assert not w.nucleus_section.is_expanded
    w.deleteLater()


def test_get_set_state_round_trips(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    new_state = {
        "nucleus": {
            "do_3d": False,
            "anisotropy": 2.25,
            "diameter": 42.0,
            "min_size": 7,
            "gamma": 1.5,
        },
        "cell": {
            "diameter": 18.0,
            "min_size": 3,
            "gamma": 0.8,
        },
    }
    w.set_state(new_state)
    got = w.get_state()
    assert got == new_state
    w.deleteLater()


def test_set_running_stage_disables_other_row(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    w._set_running_stage("nucleus")
    assert w.nucleus_run_btn.text() == "✕"
    assert w.nucleus_run_btn.isEnabled()
    assert not w.cell_run_btn.isEnabled()
    assert not w.cell_params_btn.isEnabled()
    assert not w.cell_preview_btn.isEnabled()
    w._set_running_stage(None)
    assert w.nucleus_run_btn.text() == "▶"
    assert w.cell_run_btn.isEnabled()
    assert w.cell_params_btn.isEnabled()
    w.deleteLater()


def test_exposes_output_files_tracker(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    assert hasattr(w, "output_files_tracker")
    # Same attribute name kept for main_widget.pipeline_status_from_files.
    assert w.output_files_tracker is w._files_widget
    w.deleteLater()


def test_refresh_with_none_does_not_raise(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    w.refresh(None)
    w.deleteLater()
