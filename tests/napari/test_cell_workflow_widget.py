"""Tests for the cell workflow widget — flat action-button layout."""
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

from qtpy.QtCore import QPoint
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
)


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


def test_cell_params_controller_does_not_cover_pipeline_header(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    widget.resize(360, 700)
    widget.show()
    app.processEvents()

    toggle = widget.pipeline_files_toggle_btn
    assert widget.pipeline_files_header_lbl.text() == "Pipeline Files"
    assert toggle.text() == "🔍"
    assert not widget._pipeline_files_section._toggle.isVisible()
    header_point = toggle.mapTo(
        widget,
        QPoint(toggle.width() // 2, toggle.height() // 2),
    )

    assert widget.childAt(header_point) is toggle
    assert not widget.cell_params_widget.isVisible()
    assert widget.flow_filter_section.isVisible()
    assert widget.foreground_section.isVisible()
    assert widget.contour_section.isVisible()
    assert widget.segmentation_section.isVisible()
    assert not widget.cell_correction_widget.isVisible()
    assert widget.correction_header.isVisible()
    assert widget.correction_mode_section.isVisible()

    widget.deleteLater()


def test_widget_exposes_stage_rows_with_inline_params(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    for name in (
        "flow_params_btn",
        "flow_run_btn",
        "foreground_params_btn",
        "foreground_run_btn",
        "contour_params_btn",
        "contour_preview_btn",
        "contour_run_btn",
        "segmentation_params_btn",
        "segmentation_run_btn",
    ):
        assert isinstance(getattr(widget, name), QToolButton)
    assert widget.preview_contour_btn is widget.contour_preview_btn

    # Single shared status/progress (not per-section)
    assert isinstance(widget.pipeline_status_lbl, QLabel)
    assert isinstance(widget.pipeline_progress_bar, QProgressBar)
    assert widget.pipeline_progress_bar.isVisible() is False

    # Inline stage params start collapsed and toggle independently.
    assert widget.flow_filter_section.is_expanded is False
    assert widget.foreground_section.is_expanded is False
    assert widget.contour_section.is_expanded is False
    assert widget.segmentation_section.is_expanded is False

    widget.flow_params_btn.setChecked(True)
    assert widget.flow_filter_section.is_expanded is True
    assert widget.foreground_section.is_expanded is False
    widget.contour_params_btn.setChecked(True)
    assert widget.contour_section.is_expanded is True
    assert widget.segmentation_section.is_expanded is False

    # No scroll areas
    assert widget.findChildren(QScrollArea) == []

    # Parameter spin boxes (aliased from CellParamsWidget)
    assert widget.ff_median_time_spin.value() == 3
    assert widget.ff_median_space_spin.value() == 5
    assert widget.ff_gauss_time_spin.value() == 0.0
    assert widget.ff_gauss_space_spin.value() == 0.0
    assert widget.fg_cellprob_threshold_spin.value() == 0.5

    # Correction shortcuts still exist
    assert widget.correction_shortcuts_section.title == "Correction Shortcuts"
    assert widget.correction_shortcuts_section.is_expanded is False

    # Correction buttons
    assert widget.load_labels_btn.text() == "Load Labels"
    assert widget.save_labels_btn.text() == "Save Labels"
    assert widget.fill_holes_btn.text() == "Fill Holes"
    assert widget.fix_semiholes_btn.text() == "Fix Semi Holes"
    assert widget.cleanup_btn.text() == "Clean Up"
    assert widget.expand_cell_btn.text() == "Expand Cell"

    widget.deleteLater()
    app.processEvents()


def test_cell_stage_running_state_disables_other_rows(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    widget._set_running_stage("contour")

    assert widget.contour_run_btn.text() == "✕"
    assert widget.contour_run_btn.isEnabled() is True
    assert widget.contour_params_btn.isEnabled() is True
    assert widget.flow_run_btn.isEnabled() is False
    assert widget.foreground_params_btn.isEnabled() is False
    assert widget.segmentation_run_btn.isEnabled() is False
    assert widget.contour_preview_btn.isEnabled() is False

    widget._set_running_stage(None)
    assert widget.contour_run_btn.text() == "▶"
    assert widget.flow_run_btn.isEnabled() is True
    assert widget.contour_preview_btn.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


def test_cell_correction_uses_stage_style_header(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    assert widget.correction_header_lbl.text() == "Correction"
    assert widget.correction_shortcuts_btn.text() == "📖"
    assert widget.correction_params_btn.text() == "⚙"
    assert widget.correction_active_btn.text() == "⏻"
    assert isinstance(widget.correction_shortcuts_btn, QToolButton)
    assert isinstance(widget.correction_params_btn, QToolButton)
    assert isinstance(widget.correction_active_btn, QToolButton)
    assert widget.cell_correction_widget.isVisible() is False

    widget.correction_params_btn.setChecked(True)
    assert widget.correction_mode_section.is_expanded is True
    widget.correction_params_btn.setChecked(False)
    assert widget.correction_mode_section.is_expanded is False

    widget.deleteLater()
    app.processEvents()


def test_widget_get_set_state_round_trips_flow_filtering_params(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    state = {
        "flow_filtering": {
            "median_time": 5, "median_space": 7,
            "gauss_time": 1.5, "gauss_space": 2.0,
        },
        "foreground": {
            "cellprob_threshold": 0.8,
        }
    }
    widget.set_state(state)
    got = widget.get_state()

    assert got["flow_filtering"] == state["flow_filtering"]
    assert got["foreground"]["cellprob_threshold"] == state["foreground"]["cellprob_threshold"]

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
    assert texts.count("✓") >= 5
    assert "missing" in texts

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


def test_widget_filter_flow_writes_filtered_dp_and_shows_magnitude_layer(monkeypatch, tmp_path):
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

    widget._on_filter_flow()

    assert (pos_dir / "3_cell" / "filtered_dp.tif").exists()
    assert not (pos_dir / "3_cell" / "tracked_labels.tif").exists()
    filtered_dp = tifffile.imread(str(pos_dir / "3_cell" / "filtered_dp.tif"))
    assert filtered_dp.shape == (T, 2, H, W)
    assert filtered_dp.dtype == np.float32
    assert "Filtered Flow Magnitude" in widget.viewer.layers
    assert "Cell Labels" not in widget.viewer.layers
    assert "Flow filtering complete." in widget.pipeline_status_lbl.text()
    assert widget.pipeline_progress_bar.isVisible() is False

    widget.deleteLater()
    app.processEvents()


def test_widget_build_foreground_uses_cellprob_and_filtered_dp(monkeypatch, tmp_path):
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
    widget.fg_cellprob_threshold_spin.setValue(0.7)

    with patch(
        "cellflow.segmentation.cell_foreground.compute_cellpose_foreground_masks",
        fake_foreground,
    ):
        widget._on_build_foreground()

    np.testing.assert_array_equal(captured["prob"], prob)
    np.testing.assert_array_equal(captured["dp"], filtered_dp)
    assert captured["cellprob_threshold"] == 0.7
    assert captured["flow_threshold"] == 0.0
    assert captured["min_size"] == 15
    assert captured["niter"] == 200

    fg_path = pos_dir / "3_cell" / "foreground_masks.tif"
    assert fg_path.exists()
    foreground_out = tifffile.imread(str(fg_path))
    np.testing.assert_array_equal(foreground_out, expected_fg)
    assert foreground_out.dtype == np.uint8
    assert "Foreground Mask: Cell" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["Foreground Mask: Cell"].data, expected_fg)
    assert "Foreground masks complete." in widget.pipeline_status_lbl.text()
    assert widget.pipeline_progress_bar.isVisible() is False

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

    widget._on_segment()

    assert "Missing" in widget.pipeline_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_widget_files_widget_load_buttons_load_files_into_viewer(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "3_cell").mkdir(parents=True)
    filtered_dp = np.ones((2, 2, 4, 4), dtype=np.float32)
    foreground = np.ones((2, 4, 4), dtype=np.uint8)
    labels = np.ones((2, 4, 4), dtype=np.uint32)
    tifffile.imwrite(pos_dir / "3_cell" / "filtered_dp.tif", filtered_dp)
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif", foreground)
    tifffile.imwrite(pos_dir / "3_cell" / "tracked_labels.tif", labels)

    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    for row in widget._files_widget._rows:
        if row._full_path is not None and row._full_path.exists():
            row._on_load_clicked()

    assert "3_cell_filtered_dp" in viewer.layers
    assert "3_cell_foreground_masks" in viewer.layers
    assert "3_cell_tracked_labels" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["3_cell_filtered_dp"].data, filtered_dp)
    np.testing.assert_array_equal(viewer.layers["3_cell_foreground_masks"].data, foreground)
    np.testing.assert_array_equal(viewer.layers["3_cell_tracked_labels"].data, labels)

    widget.deleteLater()
    app.processEvents()


def test_widget_load_cell_correction_loads_cell_labels_and_reference_images(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    labels = np.zeros((2, 4, 4), dtype=np.uint32)
    labels[:, 1:3, 1:3] = 7
    cell_foreground = np.ones((4, 4), dtype=np.float32)
    nuc_foreground = np.full((4, 4), 2.0, dtype=np.float32)
    tifffile.imwrite(pos_dir / "3_cell" / "tracked_labels.tif", labels)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_foreground.tif", cell_foreground)
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_foreground.tif", nuc_foreground)

    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    correction_mod = sys.modules["cellflow.napari.cell_correction_widget"]
    monkeypatch.setattr(correction_mod, "_thread_worker", _make_sync_thread_worker())

    widget.cell_correction_widget._on_load_labels()

    assert "[Correction] Cell Labels" in viewer.layers
    assert "[Correction] Cell z-avg" in viewer.layers
    assert "[Correction] Nucleus z-avg" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["[Correction] Cell Labels"].data, labels)
    np.testing.assert_array_equal(
        viewer.layers["[Correction] Cell z-avg"].data,
        np.broadcast_to(cell_foreground[np.newaxis], labels.shape),
    )
    np.testing.assert_array_equal(
        viewer.layers["[Correction] Nucleus z-avg"].data,
        np.broadcast_to(nuc_foreground[np.newaxis], labels.shape),
    )
    assert widget.correction_widget._layer is viewer.layers["[Correction] Cell Labels"]
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
    viewer.add_labels(labels, name="[Correction] Cell Labels")
    widget = mod.CellWorkflowWidget(viewer)
    widget.correction_widget.activate_layer(viewer.layers["[Correction] Cell Labels"])

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
    viewer.add_labels(edited, name="[Correction] Cell Labels")
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    widget.cell_correction_widget._on_save_labels()

    saved = tifffile.imread(pos_dir / "3_cell" / "tracked_labels.tif")
    np.testing.assert_array_equal(saved, edited)
    assert saved.dtype == np.uint32
    assert "Saved 2 frames" in widget.correction_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_cell_correction_widget_exposes_expand_cell_action(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    correction_button_texts = {
        button.text()
        for button in widget.correction_mode_section.findChildren(QPushButton)
    }
    correction_label_texts = _label_texts(widget.correction_mode_section)

    assert "Expand Cell" in correction_button_texts
    assert "Max expand px:" in correction_label_texts
    assert isinstance(widget.expand_max_px_spin, QSpinBox)
    assert widget.expand_max_px_spin.value() == 25
    assert widget.expand_max_px_spin.minimum() == 0
    assert widget.expand_max_px_spin.maximum() == 999

    widget.deleteLater()
    app.processEvents()


def test_expand_cell_expands_only_current_frame_from_loaded_foreground(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    viewer = _FakeViewer()
    labels = np.zeros((2, 7, 7), dtype=np.uint32)
    labels[0, 1, 1] = 4
    labels[1, 3, 3] = 4
    foreground = np.zeros_like(labels, dtype=np.uint8)
    foreground[1, 2:5, 2:5] = 1
    labels_layer = viewer.add_labels(labels.copy(), name="[Correction] Cell Labels")
    viewer.add_labels(foreground, name="[Correction] Foreground Mask: Cell")
    viewer.dims.current_step = (1,)

    history = []
    labels_layer._save_history = history.append
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(tmp_path / "pos00")
    widget.correction_widget.activate_layer(labels_layer)
    widget.correction_widget.select_label(1, 4)
    widget.expand_max_px_spin.setValue(1)

    widget.expand_cell_btn.click()

    edited = viewer.layers["[Correction] Cell Labels"].data
    np.testing.assert_array_equal(edited[0], labels[0])
    assert int(np.sum(edited[1] == 4)) == 5
    assert len(history) == 1
    assert "Expanded cell 4 at t=1 by 4 px" in widget.correction_status_lbl.text()
    assert widget.correction_widget._selected_label == 4
    assert widget.correction_widget._selected_t == 1

    widget.deleteLater()
    app.processEvents()


def test_expand_cell_falls_back_to_foreground_mask_file(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "3_cell").mkdir(parents=True)
    labels = np.zeros((1, 5, 5), dtype=np.uint32)
    labels[0, 2, 2] = 3
    foreground = np.ones_like(labels, dtype=np.uint8)
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif", foreground)

    viewer = _FakeViewer()
    labels_layer = viewer.add_labels(labels.copy(), name="[Correction] Cell Labels")
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget.correction_widget.activate_layer(labels_layer)
    widget.correction_widget.select_label(0, 3)
    widget.expand_max_px_spin.setValue(0)

    widget.expand_cell_btn.click()

    assert "[Correction] Foreground Mask: Cell" in viewer.layers
    np.testing.assert_array_equal(
        viewer.layers["[Correction] Foreground Mask: Cell"].data, foreground
    )
    assert np.all(viewer.layers["[Correction] Cell Labels"].data == 3)

    widget.deleteLater()
    app.processEvents()


def test_expand_cell_reports_missing_selection(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    viewer = _FakeViewer()
    viewer.add_labels(np.zeros((1, 5, 5), dtype=np.uint32), name="[Correction] Cell Labels")
    viewer.add_labels(np.ones((1, 5, 5), dtype=np.uint8), name="Foreground Mask")
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(tmp_path / "pos00")
    widget.correction_widget.activate_layer(viewer.layers["[Correction] Cell Labels"])

    widget.expand_cell_btn.click()

    assert "No cell selected" in widget.correction_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_expand_cell_reports_missing_foreground_mask(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    viewer = _FakeViewer()
    labels = np.zeros((1, 5, 5), dtype=np.uint32)
    labels[0, 2, 2] = 5
    viewer.add_labels(labels, name="[Correction] Cell Labels")
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(tmp_path / "pos00")
    widget.correction_widget.activate_layer(viewer.layers["[Correction] Cell Labels"])
    widget.correction_widget.select_label(0, 5)

    widget.expand_cell_btn.click()

    assert "Foreground mask not found" in widget.correction_status_lbl.text()

    widget.deleteLater()
    app.processEvents()
