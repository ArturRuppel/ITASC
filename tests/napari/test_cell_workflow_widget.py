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

from qtpy.QtWidgets import QApplication, QLabel, QProgressBar, QScrollArea


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


def _label_texts(widget):
    return [child.text() for child in widget.findChildren(QLabel)]


def _progress_bars(widget):
    return widget.findChildren(QProgressBar)


def test_widget_exposes_flow_following_section_with_default_params(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    assert not hasattr(widget, "flow_section")
    assert widget.filtered_flow_section.title == "Filtered Flow"
    assert widget.foreground_mask_section.title == "Foreground Mask"
    assert widget.tracked_labels_section.title == "Tracked Cell Labels"
    assert widget.filtered_flow_section.is_expanded is True
    assert widget.foreground_mask_section.is_expanded is True
    assert widget.tracked_labels_section.is_expanded is True
    assert widget.findChildren(QScrollArea) == []
    assert widget.layout().indexOf(widget.filtered_flow_section) < widget.layout().indexOf(widget.foreground_mask_section)
    assert widget.layout().indexOf(widget.foreground_mask_section) < widget.layout().indexOf(widget.tracked_labels_section)

    assert not hasattr(widget, "input_files")
    assert not hasattr(widget, "ff_files")
    assert not hasattr(widget, "ff_input_lbl")
    assert not hasattr(widget, "ff_status_lbl")
    assert not hasattr(widget, "ff_progress_bar")

    assert hasattr(widget, "filtered_flow_input_files")
    assert hasattr(widget, "filtered_flow_output_files")
    assert hasattr(widget, "filtered_flow_status_lbl")
    assert hasattr(widget, "filtered_flow_progress_bar")

    assert hasattr(widget, "foreground_mask_input_files")
    assert hasattr(widget, "foreground_mask_output_files")
    assert hasattr(widget, "foreground_mask_status_lbl")
    assert hasattr(widget, "foreground_mask_progress_bar")

    assert hasattr(widget, "tracked_labels_input_files")
    assert hasattr(widget, "tracked_labels_output_files")
    assert hasattr(widget, "tracked_labels_status_lbl")
    assert hasattr(widget, "tracked_labels_progress_bar")

    assert widget.filtered_flow_input_files.parent() is widget.filtered_flow_params_widget
    assert widget.filtered_flow_output_files.parent() is widget.filtered_flow_params_widget
    assert widget.filtered_flow_status_lbl.parent() is widget.filtered_flow_params_widget
    assert widget.filtered_flow_progress_bar.parent() is widget.filtered_flow_params_widget

    assert widget.foreground_mask_input_files.parent() is widget.foreground_mask_params_widget
    assert widget.foreground_mask_output_files.parent() is widget.foreground_mask_params_widget
    assert widget.foreground_mask_status_lbl.parent() is widget.foreground_mask_params_widget
    assert widget.foreground_mask_progress_bar.parent() is widget.foreground_mask_params_widget

    assert widget.tracked_labels_input_files.parent() is widget.tracked_labels_params_widget
    assert widget.tracked_labels_output_files.parent() is widget.tracked_labels_params_widget
    assert widget.tracked_labels_status_lbl.parent() is widget.tracked_labels_params_widget
    assert widget.tracked_labels_progress_bar.parent() is widget.tracked_labels_params_widget

    assert widget.filtered_flow_progress_bar.isVisible() is False
    assert widget.foreground_mask_progress_bar.isVisible() is False
    assert widget.tracked_labels_progress_bar.isVisible() is False

    texts = _label_texts(widget)
    assert "min" not in texts
    assert "max" not in texts
    assert "step" not in texts

    assert widget.ff_median_time_spin.value() == 3
    assert widget.ff_median_space_spin.value() == 5
    assert widget.ff_gauss_time_spin.value() == 0.0
    assert widget.ff_gauss_space_spin.value() == 0.0
    assert widget.fg_cellprob_threshold_spin.value() == 0.0
    assert widget.fg_flow_threshold_spin.value() == 0.0
    assert widget.fg_min_size_spin.value() == 15
    assert widget.fg_niter_spin.value() == 200

    assert widget.ff_flow_weight_spin.value() == 0.5
    assert widget.ff_step_scale_spin.value() == 0.2
    assert widget.ff_max_iter_spin.value() == 100
    assert widget.ff_capture_radius_spin.value() == 3.0

    assert widget.ff_flow_mag_btn.text() == "Create filtered_dp"
    assert widget.preview_fg_masks_btn.text() == "Preview"
    assert widget.fg_masks_btn.text() == "Create foreground_masks"
    assert widget.ff_labels_btn.text() == "Create tracked_labels"
    assert widget.ff_cancel_btn.text() == "Cancel"
    assert widget.ff_cancel_btn.isEnabled() is False

    assert widget.ff_median_time_spin.parent() is widget.filtered_flow_params_widget
    assert widget.ff_median_space_spin.parent() is widget.filtered_flow_params_widget
    assert widget.ff_gauss_time_spin.parent() is widget.filtered_flow_params_widget
    assert widget.ff_gauss_space_spin.parent() is widget.filtered_flow_params_widget
    assert widget.ff_flow_mag_btn.parent() is widget.filtered_flow_params_widget

    assert widget.fg_cellprob_threshold_spin.parent() is widget.foreground_mask_params_widget
    assert widget.fg_flow_threshold_spin.parent() is widget.foreground_mask_params_widget
    assert widget.fg_min_size_spin.parent() is widget.foreground_mask_params_widget
    assert widget.fg_niter_spin.parent() is widget.foreground_mask_params_widget
    assert widget.preview_fg_masks_btn.parent() is widget.foreground_mask_params_widget
    assert widget.fg_masks_btn.parent() is widget.foreground_mask_params_widget

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
        },
        "foreground_mask": {
            "cellprob_threshold": -1.2,
            "flow_threshold": 0.8,
            "min_size": 25,
            "niter": 150,
        }
    }
    widget.set_state(state)
    got = widget.get_state()

    assert got["flow_following"] == state["flow_following"]
    assert got["foreground_mask"] == state["foreground_mask"]

    widget.deleteLater()
    app.processEvents()


def test_widget_stage_file_widgets_show_present_and_missing_files(monkeypatch, tmp_path):
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

    texts = _label_texts(widget)
    assert texts.count("✓") >= 7
    assert "missing" in texts
    assert widget.filtered_flow_input_files.parent() is widget.filtered_flow_params_widget
    assert widget.foreground_mask_output_files.parent() is widget.foreground_mask_params_widget
    assert widget.tracked_labels_input_files.parent() is widget.tracked_labels_params_widget

    widget.deleteLater()
    app.processEvents()


def test_widget_stage_file_widgets_show_missing_when_files_are_absent(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget.refresh(pos_dir)

    texts = _label_texts(widget)
    assert texts.count("✗") >= 9
    assert texts.count("missing") >= 9

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
    assert "Flow magnitude complete." in widget.filtered_flow_status_lbl.text()
    assert widget.filtered_flow_progress_bar.isVisible() is False

    widget.deleteLater()
    app.processEvents()


def test_widget_create_foreground_masks_uses_cellprob_and_filtered_dp(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()

    T, Z, H, W = 2, 3, 5, 5
    prob = np.arange(T * Z * H * W, dtype=np.float32).reshape(T, Z, H, W)
    filtered_dp = np.zeros((T, 2, H, W), dtype=np.float32)
    filtered_dp[:, 0] = 1.0
    filtered_dp[:, 1] = 2.0
    expected_fg = np.zeros((T, H, W), dtype=np.uint8)
    expected_fg[:, 1:4, 2:4] = 1

    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "3_cell" / "filtered_dp.tif", filtered_dp)

    captured: dict[str, object] = {}

    def fake_foreground(prob_arg, dp_arg, **kwargs):
        captured["prob"] = np.asarray(prob_arg).copy()
        captured["dp"] = np.asarray(dp_arg).copy()
        captured.update(kwargs)
        if kwargs["progress_cb"] is not None:
            kwargs["progress_cb"](T, T)
        return expected_fg

    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget.fg_cellprob_threshold_spin.setValue(-0.4)
    widget.fg_flow_threshold_spin.setValue(1.2)
    widget.fg_min_size_spin.setValue(31)
    widget.fg_niter_spin.setValue(123)

    with patch(
        "cellflow.segmentation.compute_cellpose_foreground_masks",
        fake_foreground,
    ):
        widget._on_create_foreground_masks()

    np.testing.assert_array_equal(captured["prob"], prob)
    np.testing.assert_array_equal(captured["dp"], filtered_dp)
    assert captured["cellprob_threshold"] == -0.4
    assert captured["flow_threshold"] == 1.2
    assert captured["min_size"] == 31
    assert captured["niter"] == 123

    fg_path = pos_dir / "3_cell" / "foreground_masks.tif"
    assert fg_path.exists()
    foreground_out = tifffile.imread(str(fg_path))
    np.testing.assert_array_equal(foreground_out, expected_fg)
    assert foreground_out.dtype == np.uint8
    assert "Foreground Mask" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["Foreground Mask"].data, expected_fg)
    assert "Foreground masks complete." in widget.foreground_mask_status_lbl.text()
    assert widget.foreground_mask_progress_bar.isVisible() is False

    widget.deleteLater()
    app.processEvents()


def test_widget_preview_foreground_masks_uses_current_frame_without_writing(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()

    T, Z, H, W = 3, 2, 5, 5
    prob = np.arange(T * Z * H * W, dtype=np.float32).reshape(T, Z, H, W)
    filtered_dp = np.zeros((T, 2, H, W), dtype=np.float32)
    expected_preview = np.zeros((1, H, W), dtype=np.uint8)
    expected_preview[:, 2:4, 1:4] = 1

    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "3_cell" / "filtered_dp.tif", filtered_dp)

    viewer = _FakeViewer()
    viewer.dims.current_step = (2,)
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    captured: dict[str, object] = {}

    def fake_foreground(prob_arg, dp_arg, **kwargs):
        captured["prob"] = np.asarray(prob_arg).copy()
        captured["dp"] = np.asarray(dp_arg).copy()
        captured.update(kwargs)
        return expected_preview

    with patch("cellflow.segmentation.compute_cellpose_foreground_masks", fake_foreground):
        widget._on_preview_foreground_masks()

    np.testing.assert_array_equal(captured["prob"], prob[2:3])
    np.testing.assert_array_equal(captured["dp"], filtered_dp[2:3])
    assert captured["progress_cb"] is None
    assert not (pos_dir / "3_cell" / "foreground_masks.tif").exists()
    assert "Preview: Foreground Mask" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["Preview: Foreground Mask"].data, expected_preview[0])
    assert "Previewed foreground mask at t=2." in widget.foreground_mask_status_lbl.text()

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
    assert "Tracked labels complete." in widget.tracked_labels_status_lbl.text()
    assert widget.tracked_labels_progress_bar.isVisible() is False

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
    assert "Missing" in widget.tracked_labels_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_widget_section_file_load_buttons_load_files_into_viewer(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "3_cell").mkdir(parents=True)
    flow_mag = np.ones((2, 4, 4), dtype=np.float32)
    foreground = np.ones((2, 4, 4), dtype=np.uint8)
    labels = np.ones((2, 4, 4), dtype=np.uint32)
    tifffile.imwrite(pos_dir / "3_cell" / "filtered_flow_mag.tif", flow_mag)
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif", foreground)
    tifffile.imwrite(pos_dir / "3_cell" / "tracked_labels.tif", labels)

    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    for files_widget in (
        widget.filtered_flow_output_files,
        widget.foreground_mask_output_files,
        widget.tracked_labels_output_files,
    ):
        for row in files_widget._rows:
            if row._full_path is not None:
                row._on_load_clicked()

    assert "3_cell_filtered_flow_mag" in viewer.layers
    assert "3_cell_foreground_masks" in viewer.layers
    assert "3_cell_tracked_labels" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["3_cell_filtered_flow_mag"].data, flow_mag)
    np.testing.assert_array_equal(viewer.layers["3_cell_foreground_masks"].data, foreground)
    np.testing.assert_array_equal(viewer.layers["3_cell_tracked_labels"].data, labels)

    widget.deleteLater()
    app.processEvents()
