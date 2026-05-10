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

from qtpy.QtWidgets import QApplication, QLabel, QProgressBar, QPushButton, QScrollArea, QSpinBox


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
        self.mode = "pan_zoom"
        self.contour = 0
        self.visible = True
        self.events = _FakeEvents()
        self.mouse_drag_callbacks = []
        self.kwargs = kwargs

    def bind_key(self, key, fn, overwrite=False):
        pass

    def refresh(self):
        pass

    def _save_history(self, data):
        pass


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()
        self.layers.selection = _FakeSelection()
        self.layers.events = _FakeEvents()
        self.mouse_drag_callbacks = []
        self.dims = SimpleNamespace(
            current_step=(0,),
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

    def add_shapes(self, *, name, **kwargs):
        layer = _FakeLayer([], name, **kwargs)
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
    assert widget.correction_section.title == "Correction"
    assert widget.filtered_flow_section.is_expanded is False
    assert widget.foreground_mask_section.is_expanded is False
    assert widget.tracked_labels_section.is_expanded is False
    assert widget.correction_section.is_expanded is False
    assert widget.correction_shortcuts_section.title == "Correction Shortcuts"
    assert widget.correction_shortcuts_section.is_expanded is False
    assert widget.findChildren(QScrollArea) == []
    assert widget.layout().indexOf(widget.filtered_flow_section) < widget.layout().indexOf(widget.foreground_mask_section)
    assert widget.layout().indexOf(widget.foreground_mask_section) < widget.layout().indexOf(widget.tracked_labels_section)
    assert widget.layout().indexOf(widget.tracked_labels_section) < widget.layout().indexOf(widget.correction_section)

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
    assert widget.ff_max_iter_spin.maximum() == 5000
    assert widget.ff_capture_radius_spin.value() == 3.0
    assert widget.ff_capture_radius_spin.maximum() == 100.0

    assert widget.ff_flow_mag_btn.text() == "Create filtered_dp"
    assert widget.preview_fg_masks_btn.text() == "Preview"
    assert widget.fg_masks_btn.text() == "Create foreground_masks"
    assert widget.ff_labels_btn.text() == "Create tracked_labels"
    assert widget.ff_cancel_btn.text() == "Cancel"
    assert widget.ff_cancel_btn.isEnabled() is False
    correction_button_texts = {
        button.text()
        for button in widget.correction_section.findChildren(QPushButton)
    }
    correction_label_texts = _label_texts(widget.correction_section)
    assert "Load Cell Labels" in correction_button_texts
    assert "Save Cell Labels" in correction_button_texts
    assert "Reassign IDs" in correction_button_texts
    assert "Clean Holes / Islands" not in correction_button_texts
    assert "Fill Holes" in correction_button_texts
    assert "Fix Semiholes" in correction_button_texts
    assert "Clean Fragments" in correction_button_texts
    assert "Artifact cleanup" in correction_label_texts
    assert "Scope:" in correction_label_texts
    assert "Hole radius:" in correction_label_texts
    assert "Max opening:" in correction_label_texts
    assert "◀ Extend (A)" not in correction_button_texts
    assert "Extend (D) ▶" not in correction_button_texts
    assert "◀ Retrack (Q)" not in correction_button_texts
    assert "Retrack (E) ▶" not in correction_button_texts

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
    tifffile.imwrite(pos_dir / "3_cell" / "tracked_labels.tif",
                     np.zeros((1, 4, 4), dtype=np.uint32))

    widget.refresh(pos_dir)

    texts = _label_texts(widget)
    assert texts.count("✓") >= 8
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
    assert texts.count("✗") >= 10
    assert texts.count("missing") >= 10

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


def test_widget_load_cell_correction_loads_cell_labels_and_reference_images(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "0_input").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    labels = np.zeros((2, 4, 4), dtype=np.uint32)
    labels[:, 1:3, 1:3] = 7
    cell_zavg = np.ones((4, 4), dtype=np.float32)
    nuc_zavg = np.full((4, 4), 2.0, dtype=np.float32)
    tifffile.imwrite(pos_dir / "3_cell" / "tracked_labels.tif", labels)
    tifffile.imwrite(pos_dir / "0_input" / "cell_zavg.tif", cell_zavg)
    tifffile.imwrite(pos_dir / "0_input" / "nucleus_zavg.tif", nuc_zavg)

    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    widget._on_load_cell_correction()

    assert "Tracked: Cell" in viewer.layers
    assert "Cell z-avg" in viewer.layers
    assert "Nucleus z-avg" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["Tracked: Cell"].data, labels)
    np.testing.assert_array_equal(
        viewer.layers["Cell z-avg"].data,
        np.broadcast_to(cell_zavg[np.newaxis], labels.shape),
    )
    np.testing.assert_array_equal(
        viewer.layers["Nucleus z-avg"].data,
        np.broadcast_to(nuc_zavg[np.newaxis], labels.shape),
    )
    assert widget.correction_widget._layer is viewer.layers["Tracked: Cell"]
    assert widget.correction_section.is_expanded is True
    assert "Loaded cell label stack" in widget.correction_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_widget_selects_best_overlapping_cell_for_nucleus_selection(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[1, 1:5, 1:5] = 4
    labels[1, 5:7, 5:7] = 8
    source = np.zeros((2, 8, 8), dtype=np.uint32)
    source[1, 2:6, 2:6] = 7
    viewer.add_labels(labels, name="Tracked: Cell")
    widget = mod.CellWorkflowWidget(viewer)
    widget.correction_widget.activate_layer(viewer.layers["Tracked: Cell"])

    widget.select_matching_cell_label(1, 7, source_labels=source)

    assert widget.correction_widget._selected_label == 4
    assert widget.correction_widget._selected_t == 1

    widget.deleteLater()
    app.processEvents()


def test_widget_save_cell_correction_writes_cell_labels(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "3_cell").mkdir(parents=True)
    initial = np.zeros((2, 4, 4), dtype=np.uint32)
    tifffile.imwrite(pos_dir / "3_cell" / "tracked_labels.tif", initial)

    edited = initial.copy()
    edited[0, 1:3, 1:3] = 4
    viewer = _FakeViewer()
    viewer.add_labels(edited, name="Tracked: Cell")
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    widget._on_save_cell_correction()

    saved = tifffile.imread(pos_dir / "3_cell" / "tracked_labels.tif")
    np.testing.assert_array_equal(saved, edited)
    assert saved.dtype == np.uint32
    assert "Saved 2 frame(s)" in widget.correction_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_cell_correction_section_exposes_expand_selected_cell_action(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    correction_button_texts = {
        button.text()
        for button in widget.correction_section.findChildren(QPushButton)
    }
    correction_label_texts = _label_texts(widget.correction_section)

    assert "Expand Selected Cell" in correction_button_texts
    assert "Max expansion px:" in correction_label_texts
    assert isinstance(widget.expand_cell_max_px_spin, QSpinBox)
    assert widget.expand_cell_max_px_spin.value() == 25
    assert widget.expand_cell_max_px_spin.minimum() == 0
    assert widget.expand_cell_max_px_spin.maximum() == 999

    widget.deleteLater()
    app.processEvents()


def test_expand_selected_cell_expands_only_current_frame_from_loaded_foreground(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    viewer = _FakeViewer()
    labels = np.zeros((2, 7, 7), dtype=np.uint32)
    labels[0, 1, 1] = 4
    labels[1, 3, 3] = 4
    foreground = np.zeros_like(labels, dtype=np.uint8)
    foreground[1, 2:5, 2:5] = 1
    labels_layer = viewer.add_labels(labels.copy(), name="Tracked: Cell")
    viewer.add_labels(foreground, name="Foreground Mask")
    viewer.dims.current_step = (1,)

    history = []
    labels_layer._save_history = history.append
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(tmp_path / "pos00")
    widget.correction_widget.activate_layer(labels_layer)
    widget.correction_widget.select_label(1, 4)
    widget.expand_cell_max_px_spin.setValue(1)

    widget.expand_selected_cell_btn.click()

    edited = viewer.layers["Tracked: Cell"].data
    np.testing.assert_array_equal(edited[0], labels[0])
    assert int(np.sum(edited[1] == 4)) == 5
    assert len(history) == 1
    assert "Expanded cell 4 at t=1 by 4 px" in widget.correction_status_lbl.text()
    assert widget.correction_widget._selected_label == 4
    assert widget.correction_widget._selected_t == 1

    widget.deleteLater()
    app.processEvents()


def test_expand_selected_cell_falls_back_to_foreground_mask_file(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "3_cell").mkdir(parents=True)
    labels = np.zeros((1, 5, 5), dtype=np.uint32)
    labels[0, 2, 2] = 3
    foreground = np.ones_like(labels, dtype=np.uint8)
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif", foreground)

    viewer = _FakeViewer()
    labels_layer = viewer.add_labels(labels.copy(), name="Tracked: Cell")
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget.correction_widget.activate_layer(labels_layer)
    widget.correction_widget.select_label(0, 3)
    widget.expand_cell_max_px_spin.setValue(0)

    widget.expand_selected_cell_btn.click()

    assert "Foreground Mask" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["Foreground Mask"].data, foreground)
    assert np.all(viewer.layers["Tracked: Cell"].data == 3)

    widget.deleteLater()
    app.processEvents()


def test_expand_selected_cell_reports_missing_selection(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    viewer = _FakeViewer()
    viewer.add_labels(np.zeros((1, 5, 5), dtype=np.uint32), name="Tracked: Cell")
    viewer.add_labels(np.ones((1, 5, 5), dtype=np.uint8), name="Foreground Mask")
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(tmp_path / "pos00")
    widget.correction_widget.activate_layer(viewer.layers["Tracked: Cell"])

    widget.expand_selected_cell_btn.click()

    assert "No cell selected" in widget.correction_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_expand_selected_cell_reports_missing_foreground_mask(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    viewer = _FakeViewer()
    labels = np.zeros((1, 5, 5), dtype=np.uint32)
    labels[0, 2, 2] = 5
    viewer.add_labels(labels, name="Tracked: Cell")
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(tmp_path / "pos00")
    widget.correction_widget.activate_layer(viewer.layers["Tracked: Cell"])
    widget.correction_widget.select_label(0, 5)

    widget.expand_selected_cell_btn.click()

    assert "Foreground mask not found" in widget.correction_status_lbl.text()

    widget.deleteLater()
    app.processEvents()
