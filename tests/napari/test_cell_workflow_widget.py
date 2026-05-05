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

from qtpy.QtWidgets import QApplication


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

    assert widget.flow_section.title == "Flow-Following Segmentation"

    assert widget.ff_median_time_spin.value() == 3
    assert widget.ff_median_space_spin.value() == 5
    assert widget.ff_gauss_time_spin.value() == 0.0
    assert widget.ff_gauss_space_spin.value() == 0.0

    assert widget.ff_flow_weight_spin.value() == 0.5
    assert widget.ff_step_scale_spin.value() == 0.2
    assert widget.ff_max_iter_spin.value() == 100
    assert widget.ff_capture_radius_spin.value() == 3.0

    assert widget.ff_run_btn.text() == "Run"
    assert widget.ff_cancel_btn.text() == "Cancel"
    assert widget.ff_cancel_btn.isEnabled() is False

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
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif",
                     np.zeros((1, 4, 4), dtype=bool))
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif",
                     np.zeros((1, 4, 4), dtype=np.uint32))

    widget.refresh(pos_dir)

    text = widget.ff_input_lbl.text()
    assert text.count("✓") == 4
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
    assert text.count("✗") == 4

    widget.deleteLater()
    app.processEvents()


def test_widget_run_calls_compute_flow_following_movie_and_writes_outputs(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "3_cell").mkdir()

    T, Z, H, W = 2, 2, 6, 6
    prob = np.zeros((T, Z, H, W), dtype=np.float32)
    dp = np.zeros((T, Z, 2, H, W), dtype=np.float32)
    fg = np.ones((T, H, W), dtype=bool)
    nuc = np.zeros((T, H, W), dtype=np.uint32)
    nuc[:, 3, 3] = 5

    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif", dp)
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif", fg)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", nuc)

    fake_filtered = np.full((T, 2, H, W), 0.5, dtype=np.float32)
    fake_labels = np.full((T, H, W), 5, dtype=np.int32)

    captured: dict[str, object] = {}

    def fake_compute(foreground, dp_tcyx, labels, params, progress_cb=None):
        captured["foreground_shape"] = foreground.shape
        captured["dp_shape"] = dp_tcyx.shape
        captured["labels_shape"] = labels.shape
        captured["params"] = params
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
        widget._on_run_flow_following()

    assert captured["foreground_shape"] == (T, H, W)
    assert captured["dp_shape"] == (T, 2, H, W)
    assert captured["labels_shape"] == (T, H, W)
    assert captured["params"].capture_radius == 3.0

    assert (pos_dir / "3_cell" / "filtered_flow_mag.tif").exists()
    assert (pos_dir / "3_cell" / "tracked_labels.tif").exists()
    mag = tifffile.imread(str(pos_dir / "3_cell" / "filtered_flow_mag.tif"))
    assert mag.shape == (T, H, W)
    assert mag.dtype == np.float32
    labels_out = tifffile.imread(str(pos_dir / "3_cell" / "tracked_labels.tif"))
    assert labels_out.dtype == np.uint32

    assert "Filtered Flow Magnitude" in viewer.layers
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
        widget._on_run_flow_following()

    assert called is False
    assert "Missing" in widget.ff_status_lbl.text()

    widget.deleteLater()
    app.processEvents()
