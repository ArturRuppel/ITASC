"""Tests for the cell workflow widget — Flow-Following Segmentation section."""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QScrollArea


class _LayerCollection(dict):
    def remove(self, layer):
        self.pop(layer.name, None)


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()
        self.dims = SimpleNamespace(
            current_step=(0,),
            events=SimpleNamespace(
                current_step=SimpleNamespace(connect=lambda cb: None)
            ),
        )

    def add_image(self, data, *, name, **kwargs):
        layer = SimpleNamespace(data=np.asarray(data), name=name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_labels(self, data, *, name, **kwargs):
        layer = SimpleNamespace(data=np.asarray(data), name=name, **kwargs)
        self.layers[name] = layer
        return layer


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.cell_workflow_widget", None)
    return importlib.import_module("cellflow.napari.cell_workflow_widget")


def _make_sync_thread_worker():
    def fake_thread_worker(connect=None):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    if connect and "errored" in connect:
                        connect["errored"](exc)
                    return None
                if inspect.isgenerator(result):
                    return_value = None
                    while True:
                        try:
                            yielded = next(result)
                        except StopIteration as exc:
                            return_value = exc.value
                            break
                        if connect and "yielded" in connect:
                            connect["yielded"](yielded)
                    if connect and "returned" in connect:
                        connect["returned"](return_value)
                else:
                    if connect and "returned" in connect:
                        connect["returned"](result)
                return None
            return wrapper
        return decorator
    return fake_thread_worker


def test_widget_exposes_flow_following_section_with_default_params(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    assert not hasattr(widget, "flow_section")
    assert widget.filtered_flow_section.title == "Filtered Flow"
    assert widget.tracked_labels_section.title == "Tracked Cell Labels"
    assert widget.filtered_flow_section.is_expanded is True
    assert widget.tracked_labels_section.is_expanded is True
    assert widget.findChildren(QScrollArea) == []

    assert widget.ff_median_time_spin.value() == 3
    assert widget.ff_median_space_spin.value() == 5
    assert widget.ff_gauss_time_spin.value() == 0.0
    assert widget.ff_gauss_space_spin.value() == 0.0

    assert widget.ff_flow_weight_spin.value() == 0.5
    assert widget.ff_step_scale_spin.value() == 0.2
    assert widget.ff_max_iter_spin.value() == 100
    assert widget.ff_capture_radius_spin.value() == 3.0

    assert widget.ff_flow_mag_btn.text() == "Create filtered_dp"
    assert widget.ff_labels_btn.text() == "Create tracked_labels"
    assert widget.ff_cancel_btn.text() == "Cancel"
    assert widget.ff_cancel_btn.isEnabled() is False

    assert widget.ff_median_time_spin.parent() is widget.filtered_flow_params_widget
    assert widget.ff_median_space_spin.parent() is widget.filtered_flow_params_widget
    assert widget.ff_gauss_time_spin.parent() is widget.filtered_flow_params_widget
    assert widget.ff_gauss_space_spin.parent() is widget.filtered_flow_params_widget
    assert widget.ff_flow_mag_btn.parent() is widget.filtered_flow_params_widget

    assert widget.ff_flow_weight_spin.parent() is widget.tracked_labels_params_widget
    assert widget.ff_step_scale_spin.parent() is widget.tracked_labels_params_widget
    assert widget.ff_max_iter_spin.parent() is widget.tracked_labels_params_widget
    assert widget.ff_capture_radius_spin.parent() is widget.tracked_labels_params_widget
    assert widget.ff_labels_btn.parent() is widget.tracked_labels_params_widget

    widget.deleteLater()
    app.processEvents()


def test_widget_get_set_state_round_trips_flow_following_params(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    state = {
        "flow_following": {
            "median_time": 5, "median_space": 7,
            "gauss_time": 1.5, "gauss_space": 2.0,
            "flow_weight": 0.7, "step_scale": 0.3,
            "max_iter": 200, "capture_radius": 5.0,
        }
    }
    widget.set_state(state)
    got = widget.get_state()

    assert got["flow_following"] == state["flow_following"]

    widget.deleteLater()
    app.processEvents()


def test_widget_input_status_label_shows_check_for_each_required_file(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "3_cell").mkdir()
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif",
                     np.zeros((1, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif",
                     np.zeros((1, 2, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "filtered_dp.tif",
                     np.zeros((1, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif",
                     np.zeros((1, 4, 4), dtype=bool))
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif",
                     np.zeros((1, 4, 4), dtype=np.uint32))

    widget.refresh(pos_dir)

    text = widget.ff_input_lbl.text()
    assert text.count("✓") == 5
    assert "✗" not in text

    widget.deleteLater()
    app.processEvents()


def test_widget_input_status_label_shows_cross_when_files_missing(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget.refresh(pos_dir)

    text = widget.ff_input_lbl.text()
    assert text.count("✗") == 5

    widget.deleteLater()
    app.processEvents()


def test_widget_create_flow_mag_writes_filtered_dp_and_flow_mag(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()

    T, Z, H, W = 2, 2, 6, 6
    prob = np.zeros((T, Z, H, W), dtype=np.float32)
    dp = np.zeros((T, Z, 2, H, W), dtype=np.float32)
    dp[:, :, 0] = 3.0
    dp[:, :, 1] = 4.0

    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif", dp)
    widget = mod.CellWorkflowWidget(_FakeViewer())
    widget.refresh(pos_dir)
    widget.ff_median_time_spin.setValue(1)
    widget.ff_median_space_spin.setValue(1)

    widget._on_create_flow_mag()

    assert (pos_dir / "3_cell" / "filtered_dp.tif").exists()
    assert (pos_dir / "3_cell" / "filtered_flow_mag.tif").exists()
    assert not (pos_dir / "3_cell" / "tracked_labels.tif").exists()
    filtered_dp = tifffile.imread(str(pos_dir / "3_cell" / "filtered_dp.tif"))
    assert filtered_dp.shape == (T, 2, H, W)
    assert filtered_dp.dtype == np.float32
    np.testing.assert_allclose(filtered_dp[:, 0], 3.0)
    np.testing.assert_allclose(filtered_dp[:, 1], 4.0)
    mag = tifffile.imread(str(pos_dir / "3_cell" / "filtered_flow_mag.tif"))
    assert mag.shape == (T, H, W)
    assert mag.dtype == np.float32
    np.testing.assert_allclose(mag, 5.0)
    assert "Filtered Flow Magnitude" in widget.viewer.layers
    assert "Cell Labels" not in widget.viewer.layers

    widget.deleteLater()
    app.processEvents()


def test_widget_create_tracked_labels_calls_compute_flow_following_movie_and_writes_only_labels(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "3_cell").mkdir()

    T, H, W = 2, 6, 6
    filtered_dp = np.zeros((T, 2, H, W), dtype=np.float32)
    filtered_dp[:, 0] = 7.0
    filtered_dp[:, 1] = 8.0
    fg = np.ones((T, H, W), dtype=bool)
    nuc = np.zeros((T, H, W), dtype=np.uint32)
    nuc[:, 3, 3] = 5

    tifffile.imwrite(pos_dir / "3_cell" / "filtered_dp.tif", filtered_dp)
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif", fg)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", nuc)

    fake_filtered = np.full((T, 2, H, W), 0.5, dtype=np.float32)
    fake_labels = np.full((T, H, W), 5, dtype=np.int32)

    captured: dict[str, object] = {}

    def fake_compute(foreground, dp_tcyx, labels, params, progress_cb=None,
                     filter_vectors=True):
        captured["foreground_shape"] = foreground.shape
        captured["dp_shape"] = dp_tcyx.shape
        captured["dp"] = dp_tcyx.copy()
        captured["labels_shape"] = labels.shape
        captured["params"] = params
        captured["filter_vectors"] = filter_vectors
        if progress_cb is not None:
            progress_cb(T, T)
        return fake_filtered, fake_labels

    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    with patch(
        "cellflow.segmentation.compute_flow_following_movie",
        fake_compute,
    ):
        widget._on_create_tracked_labels()

    assert captured["foreground_shape"] == (T, H, W)
    assert captured["dp_shape"] == (T, 2, H, W)
    np.testing.assert_array_equal(captured["dp"], filtered_dp)
    assert captured["labels_shape"] == (T, H, W)
    assert captured["params"].capture_radius == 3.0
    assert captured["filter_vectors"] is False

    assert (pos_dir / "3_cell" / "tracked_labels.tif").exists()
    assert (pos_dir / "3_cell" / "filtered_dp.tif").exists()
    assert not (pos_dir / "3_cell" / "filtered_flow_mag.tif").exists()
    labels_out = tifffile.imread(str(pos_dir / "3_cell" / "tracked_labels.tif"))
    assert labels_out.dtype == np.uint32

    assert "Filtered Flow Magnitude" not in viewer.layers
    assert "Cell Labels" in viewer.layers

    widget.deleteLater()
    app.processEvents()


def test_widget_run_aborts_when_input_file_missing(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget = mod.CellWorkflowWidget(_FakeViewer())
    widget.refresh(pos_dir)

    called = False

    def fake_compute(*args, **kwargs):
        nonlocal called
        called = True
        return None, None

    with patch(
        "cellflow.segmentation.compute_flow_following_movie",
        fake_compute,
    ):
        widget._on_create_tracked_labels()

    assert called is False
    assert "Missing" in widget.ff_status_lbl.text()

    widget.deleteLater()
    app.processEvents()
