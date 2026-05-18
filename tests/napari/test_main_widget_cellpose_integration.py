"""Integration test: CellFlowMainWidget uses the new CellposeWidget."""
from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def _mock_cellpose(monkeypatch):
    fake_cellpose = types.ModuleType("cellpose")
    fake_models = types.ModuleType("cellpose.models")

    class _FakeModel:
        def __init__(self, *_, **__): pass

        def eval(self, img, **_kwargs):
            arr = np.asarray(img, dtype=np.float32)
            ndim = 3 if arr.ndim == 3 else 2
            chans = 3 if ndim == 3 else 2
            dp = np.zeros((chans, *arr.shape), dtype=np.float32)
            prob = np.zeros(arr.shape, dtype=np.float32)
            return None, (None, dp, prob), None

    fake_models.CellposeModel = _FakeModel
    fake_cellpose.models = fake_models
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)
    monkeypatch.delitem(sys.modules, "cellflow.segmentation.cellpose_runner", raising=False)
    monkeypatch.delitem(sys.modules, "cellflow.napari.cellpose_widget", raising=False)
    monkeypatch.delitem(sys.modules, "cellflow.napari.main_widget", raising=False)


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


def test_main_widget_constructs_new_cellpose_widget():
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    cellpose_mod = importlib.import_module("cellflow.napari.cellpose_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())
    assert isinstance(w._cellpose_widget, cellpose_mod.CellposeWidget)
    # Old placeholder class no longer exists.
    assert not hasattr(main_mod, "_CellposePanel")
    w.deleteLater()


def test_main_widget_wires_divergence_maps_widget(tmp_path):
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    divergence_mod = importlib.import_module("cellflow.napari.divergence_maps_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())

    assert isinstance(w._divergence_maps_widget, divergence_mod.DivergenceMapsWidget)
    assert w.divergence_maps_section is not None

    state = {
        "nucleus": {"smoothing_sigma": 2.5, "median_radius": 2},
        "cell": {"foreground_z_reduction": "max"},
    }
    w.set_state({"divergence_maps": state})
    got = w.get_state()
    assert got["divergence_maps"]["nucleus"]["smoothing_sigma"] == pytest.approx(2.5)
    assert got["divergence_maps"]["nucleus"]["median_radius"] == 2
    assert got["divergence_maps"]["cell"]["foreground_z_reduction"] == "max"

    w.path_label.setText(str(tmp_path))
    w.pos_spin.setValue(0)
    w._refresh_all()
    assert w._divergence_maps_widget._pos_dir == tmp_path / "pos00"
    w.deleteLater()


def test_main_widget_state_round_trips_cellpose():
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())
    cellpose_state = {
        "nucleus": {
            "do_3d": False,
            "anisotropy": 1.25,
            "diameter": 33.0,
            "min_size": 9,
            "gamma": 1.2,
        },
        "cell": {"diameter": 17.0, "min_size": 4, "gamma": 0.9},
    }
    w.set_state({"cellpose": cellpose_state})
    got = w.get_state()
    assert "cellpose" in got
    assert got["cellpose"] == cellpose_state
    w.deleteLater()


def test_main_widget_pipeline_status_uses_output_files_tracker(tmp_path):
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())
    # Should reach pipeline_status_from_files without error (tracker exists).
    assert w._cellpose_widget.output_files_tracker is not None
    w._update_section_statuses()
    w.deleteLater()
