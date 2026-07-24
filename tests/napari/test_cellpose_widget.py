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

from qtpy.QtWidgets import QApplication, QLabel, QComboBox, QLineEdit, QPushButton, QToolButton


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
    monkeypatch.delitem(sys.modules, "itasc.cellpose.cellpose_runner", raising=False)


def _load_widget(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "itasc" / "napari"
    napari_pkg = types.ModuleType("itasc.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "itasc.napari", napari_pkg)
    sys.modules.pop("itasc.napari.cellpose_widget", None)
    return importlib.import_module("itasc.napari.cellpose_widget")


def _layout_items(layout):
    return [layout.itemAt(i) for i in range(layout.count())]


def _layout_widgets_from_items(items):
    return [item.widget() for item in items if item.widget() is not None]


def _layout_for_row_items_containing(widget, *targets):
    target_set = set(targets)
    for index in range(widget.layout().count()):
        item = widget.layout().itemAt(index)
        row = item.layout()
        if row is None:
            continue
        items = _layout_items(row)
        widgets = set(_layout_widgets_from_items(items))
        if target_set.issubset(widgets):
            return items
    raise AssertionError("Could not find matching Cellpose stage row")


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
    for name in ("nuc_model_combo", "cell_model_combo"):
        combo = getattr(w, name)
        assert isinstance(combo, QComboBox), name
        assert combo.currentText() == "Cellpose-SAM"
    for name in ("nuc_model_edit", "cell_model_edit"):
        edit = getattr(w, name)
        assert isinstance(edit, QLineEdit), name
        assert edit.isHidden()
    for name in ("nuc_model_browse_btn", "cell_model_browse_btn"):
        browse = getattr(w, name)
        assert isinstance(browse, QPushButton), name
        assert browse.isHidden()
    assert w.nucleus_run_btn.text() == "▶"
    assert w.cell_run_btn.text() == "▶"
    w.deleteLater()


def test_stage_row_buttons_are_clustered_and_use_header_style(
    _mock_cellpose, monkeypatch
):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())

    nucleus_label = next(
        child for child in w.findChildren(QLabel) if child.text() == "Nucleus Cellpose"
    )
    cell_label = next(
        child for child in w.findChildren(QLabel) if child.text() == "Cell Cellpose"
    )

    nucleus_items = _layout_for_row_items_containing(
        w,
        nucleus_label,
        w.nucleus_params_btn,
        w.nucleus_preview_btn,
        w.nucleus_run_btn,
    )
    assert _layout_widgets_from_items(nucleus_items[:4]) == [
        nucleus_label,
        w.nucleus_params_btn,
        w.nucleus_preview_btn,
        w.nucleus_run_btn,
    ]
    assert nucleus_items[4].spacerItem() is not None

    cell_items = _layout_for_row_items_containing(
        w,
        cell_label,
        w.cell_params_btn,
        w.cell_preview_btn,
        w.cell_run_btn,
    )
    assert _layout_widgets_from_items(cell_items[:4]) == [
        cell_label,
        w.cell_params_btn,
        w.cell_preview_btn,
        w.cell_run_btn,
    ]
    assert cell_items[4].spacerItem() is not None

    for button in (
        w.nucleus_params_btn,
        w.nucleus_preview_btn,
        w.nucleus_run_btn,
        w.cell_params_btn,
        w.cell_preview_btn,
        w.cell_run_btn,
    ):
        assert button.property("itasc_stage_header_action") is True
        assert "border: none" in button.styleSheet()
        assert "text-align: center" in button.styleSheet()

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
            "model": "Custom",
            "custom_model": "/tmp/nucleus_model.pt",
            "layout": "3D",
            "do_3d": False,
            "anisotropy": 2.25,
            "diameter": 42.0,
            "min_size": 7,
            "gamma": 1.5,
        },
        "cell": {
            "model": "Cellpose-SAM",
            "custom_model": "",
            "layout": "2D+t",
            "diameter": 18.0,
            "min_size": 3,
            "gamma": 0.8,
        },
    }
    w.set_state(new_state)
    got = w.get_state()
    assert got["nucleus"] == new_state["nucleus"]
    assert got["cell"] == new_state["cell"]
    assert "divergence_maps" in got
    w.deleteLater()


def test_channel_model_controls_toggle_custom_file_row(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    assert w.nuc_model_edit.isHidden()
    assert w.nuc_model_browse_btn.isHidden()
    assert w.cell_model_edit.isHidden()
    assert w.cell_model_browse_btn.isHidden()

    w.nuc_model_combo.setCurrentText("Custom")
    assert not w.nuc_model_edit.isHidden()
    assert not w.nuc_model_browse_btn.isHidden()
    assert w.cell_model_edit.isHidden()

    w.cell_model_combo.setCurrentText("Custom")
    assert not w.cell_model_edit.isHidden()
    assert not w.cell_model_browse_btn.isHidden()
    w.nuc_model_combo.setCurrentText("Cellpose-SAM")
    assert w.nuc_model_edit.isHidden()
    w.deleteLater()


def test_apply_model_selection_is_per_channel(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    seen = []
    monkeypatch.setattr(
        mod.cellpose_runner, "set_pretrained_model", lambda value: seen.append(value)
    )
    w = mod.CellposeWidget(_FakeViewer())
    nucleus_model = tmp_path / "nucleus.pt"
    cell_model = tmp_path / "cell.pt"
    nucleus_model.write_bytes(b"fake")
    cell_model.write_bytes(b"fake")

    w.nuc_model_combo.setCurrentText("Custom")
    w.nuc_model_edit.setText(str(nucleus_model))
    w.cell_model_combo.setCurrentText("Custom")
    w.cell_model_edit.setText(str(cell_model))

    assert w._apply_model_selection("nucleus") is True
    assert seen[-1] == nucleus_model
    assert w._apply_model_selection("cell") is True
    assert seen[-1] == cell_model
    w.cell_model_combo.setCurrentText("Cellpose-SAM")
    assert w._apply_model_selection("cell") is True
    assert seen[-1] == mod.cellpose_runner.DEFAULT_PRETRAINED_MODEL
    w.deleteLater()


def test_cellpose_parameter_controls_are_sliders(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())

    for slider in (
        w.nuc_anisotropy_spin,
        w.nuc_diameter_spin,
        w.nuc_min_size_spin,
        w.nuc_gamma_spin,
        w.cell_diameter_spin,
        w.cell_min_size_spin,
        w.cell_gamma_spin,
    ):
        buttons = {
            button.objectName(): button
            for button in slider.findChildren(QToolButton)
        }
        start = slider.value()

        buttons["slider_increment_button"].click()
        assert slider.value() == pytest.approx(start + slider.singleStep())

        buttons["slider_decrement_button"].click()
        assert slider.value() == pytest.approx(start)

    w.deleteLater()


def test_set_running_stage_disables_other_row(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    w._set_running_stage("nucleus")
    assert w.nucleus_run_btn.text() == "✕"
    assert w.nucleus_run_btn.isEnabled()
    assert not w.cell_run_btn.isEnabled()
    assert not w.cell_preview_btn.isEnabled()
    # ⚙ params just toggle a parameter panel — harmless — so they stay enabled
    # even while the other channel is running.
    assert w.cell_params_btn.isEnabled()
    w._set_running_stage(None)
    assert w.nucleus_run_btn.text() == "▶"
    assert w.cell_run_btn.isEnabled()
    assert w.cell_params_btn.isEnabled()
    w.deleteLater()


def test_non_cancellable_stage_shows_no_cancel_button(_mock_cellpose, monkeypatch):
    # Previews can't be interrupted, so the running row shows no ✕ and its
    # run button is disabled rather than masquerading as a cancel control.
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    w._set_running_stage("nucleus", cancellable=False)
    assert w.nucleus_run_btn.text() == "▶"
    assert not w.nucleus_run_btn.isEnabled()
    assert not w.cell_run_btn.isEnabled()
    assert not w.cell_preview_btn.isEnabled()
    w._set_running_stage(None)
    assert w.nucleus_run_btn.isEnabled()
    w.deleteLater()


def test_exposes_output_files_tracker(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    assert hasattr(w, "output_files_tracker")
    # Same attribute name kept for main_widget.pipeline_status_from_files.
    assert w.output_files_tracker is w._files_widget
    assert not hasattr(w, "zavg_viz_widget")
    assert all("prob_zavg" not in row._rel_path for row in w._files_widget._rows)
    w.deleteLater()


def test_cellpose_embeds_single_divergence_subwidget_without_duplicate_file_panel(
    _mock_cellpose, monkeypatch
):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    divergence_mod = importlib.import_module("itasc.napari.divergence_maps_widget")

    w = mod.CellposeWidget(_FakeViewer())

    assert isinstance(w.divergence_maps_widget, divergence_mod.DivergenceMapsWidget)
    assert not hasattr(w, "nucleus_divergence_maps_widget")
    assert not hasattr(w, "cell_divergence_maps_widget")
    assert not hasattr(w.divergence_maps_widget, "output_files_tracker")
    assert not hasattr(w.divergence_maps_widget, "pipeline_files_header")
    assert not hasattr(w, "zavg_viz_widget")
    assert all("prob_zavg" not in row._rel_path for row in w._files_widget._rows)
    assert {row._rel_path for row in w._files_widget._rows} >= {
        "1_cellpose/nucleus_prob.tif",
        "1_cellpose/nucleus_dp.tif",
        "1_cellpose/nucleus_contours.tif",
        "1_cellpose/nucleus_foreground.tif",
        "1_cellpose/cell_prob.tif",
        "1_cellpose/cell_dp.tif",
        "1_cellpose/cell_contours.tif",
        "1_cellpose/cell_foreground.tif",
    }
    w.deleteLater()
    app.processEvents()


def test_embeds_divergence_maps_subwidget_and_state(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    divergence_mod = importlib.import_module("itasc.napari.divergence_maps_widget")
    from itasc.cellpose.divergence_maps import DivergenceMapsReport
    import tifffile

    w = mod.CellposeWidget(_FakeViewer())
    assert isinstance(w.divergence_maps_widget, divergence_mod.DivergenceMapsWidget)

    state = {
        "divergence_maps": {
            "nucleus": {"smoothing_sigma": 2.5, "median_radius": 2},
            "cell": {"foreground_z_reduction": "max"},
        }
    }
    w.set_state(state)
    got = w.get_state()
    assert got["divergence_maps"]["nucleus"]["smoothing_sigma"] == pytest.approx(2.5)
    assert got["divergence_maps"]["nucleus"]["median_radius"] == 2
    assert got["divergence_maps"]["cell"]["foreground_z_reduction"] == "max"

    pos = tmp_path / "pos00"
    cellpose_dir = pos / "1_cellpose"
    cellpose_dir.mkdir(parents=True)
    tifffile.imwrite(
        cellpose_dir / "cell_prob.tif",
        np.zeros((1, 1, 2, 2), dtype=np.float32),
    )
    tifffile.imwrite(
        cellpose_dir / "cell_dp.tif",
        np.zeros((1, 1, 2, 2, 2), dtype=np.float32),
    )
    captured = {}

    def _fake_build(
        prob_path,
        dp_path,
        contours_out,
        foreground_out,
        *,
        foreground_z_reduction,
        contour_z_reduction,
        smoothing_sigma,
        median_radius,
        foreground_smoothing_sigma=0.0,
        foreground_median_radius=0,
        progress_cb=None,
        cancel=None,
    ):
        captured["prob_path"] = str(prob_path)
        captured["dp_path"] = str(dp_path)
        captured["contours_out"] = str(contours_out)
        captured["foreground_out"] = str(foreground_out)
        return DivergenceMapsReport(
            frames=1,
            foreground_z_reduction=foreground_z_reduction,
            contour_z_reduction=contour_z_reduction,
            smoothing_sigma=smoothing_sigma,
            median_radius=median_radius,
            contours_path=contours_out,
            foreground_path=foreground_out,
        )

    monkeypatch.setattr(divergence_mod, "build_divergence_maps", _fake_build)
    w.refresh(pos)
    w.divergence_maps_widget._run_blocking("cell")
    assert captured["prob_path"].endswith("cell_prob.tif")
    assert captured["dp_path"].endswith("cell_dp.tif")
    assert captured["contours_out"].endswith("cell_contours.tif")
    assert captured["foreground_out"].endswith("cell_foreground.tif")
    w.deleteLater()


def test_refresh_with_none_does_not_raise(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    w.refresh(None)
    w.deleteLater()


def _make_sync_thread_worker():
    """Patch thread_worker so workers execute synchronously."""
    import inspect

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


def _write_test_stack(path: Path, shape):
    arr = np.zeros(shape, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tifffile as tf

    tf.imwrite(str(path), arr)


def test_pipeline_files_track_configured_input(monkeypatch, tmp_path):
    """The Inputs rows must follow the configured input, not the hardcoded name."""
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())  # integrated mode

    nuc_row = w._files_widget._rows_by_group["Inputs"][0]

    # No canonical 0_input/nucleus.tif exists → the row is missing.
    w.refresh(tmp_path)
    assert nuc_row._full_path is None

    # Configure a non-canonical, not-behind-0_input input and put the file there.
    _write_test_stack(tmp_path / "raw" / "nuc.tif", (2, 3, 6, 6))
    w.set_input_names({"nucleus": "raw/nuc.tif"})

    # The row now tracks the real file — even though 0_input/nucleus.tif was
    # never created — and its label reflects the resolved location.
    assert not (tmp_path / "0_input" / "nucleus.tif").exists()
    assert nuc_row._full_path == tmp_path / "raw" / "nuc.tif"
    assert nuc_row._name_lbl.text() == "raw/nuc.tif"

    # An absolute input outside the position dir falls back to its basename.
    elsewhere = tmp_path.parent / "elsewhere" / "nuc_stack.tif"
    _write_test_stack(elsewhere, (2, 3, 6, 6))
    w.set_input_names({"nucleus": str(elsewhere)})
    assert nuc_row._full_path == elsewhere
    assert nuc_row._name_lbl.text() == "nuc_stack.tif"

    w.deleteLater()
    app.processEvents()


def test_run_nucleus_writes_outputs_and_updates_status(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    _write_test_stack(tmp_path / "0_input" / "nucleus.tif", (2, 3, 6, 6))
    w.refresh(tmp_path)
    w.nucleus_run_btn.click()
    out = tmp_path / "1_cellpose"
    assert (out / "nucleus_prob.tif").exists()
    assert (out / "nucleus_dp.tif").exists()
    assert "complete" in w.status_lbl.text().lower()
    assert w.nucleus_run_btn.text() == "▶"
    w.deleteLater()


def test_run_cell_writes_outputs(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    _write_test_stack(tmp_path / "0_input" / "cell.tif", (2, 3, 6, 6))
    w.refresh(tmp_path)
    w.cell_run_btn.click()
    out = tmp_path / "1_cellpose"
    assert (out / "cell_prob.tif").exists()
    assert (out / "cell_dp.tif").exists()
    w.deleteLater()


def test_run_reports_missing_input(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    w.refresh(tmp_path)  # no input tif written
    w.nucleus_run_btn.click()
    assert "missing" in w.status_lbl.text().lower()
    w.deleteLater()


def test_run_with_no_project_reports_status(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    w.nucleus_run_btn.click()
    assert "no project" in w.status_lbl.text().lower()
    w.deleteLater()


def test_nucleus_preview_2d_creates_layers(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    viewer = _FakeViewer()
    viewer.dims.current_step = (1, 2)  # t=1, z=2
    w = mod.CellposeWidget(viewer)
    _write_test_stack(tmp_path / "0_input" / "nucleus.tif", (3, 4, 6, 6))
    w.refresh(tmp_path)
    w.nuc_3d_chk.setChecked(False)
    w.nucleus_preview_btn.click()
    assert "Preview: Nucleus prob" in viewer.layers
    assert "Preview: Nucleus flow" in viewer.layers
    prob = viewer.layers["Preview: Nucleus prob"].data
    flow = viewer.layers["Preview: Nucleus flow"].data
    assert prob.shape == (3, 4, 6, 6)
    assert flow.shape == (3, 4, 6, 6)
    w.deleteLater()


def test_nucleus_preview_2d_keeps_reference_time_and_z_axes(
    _mock_cellpose, monkeypatch, tmp_path
):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    viewer = _FakeViewer()
    viewer.dims.current_step = (1, 2)
    w = mod.CellposeWidget(viewer)
    _write_test_stack(tmp_path / "0_input" / "nucleus.tif", (3, 4, 6, 6))
    w.refresh(tmp_path)
    w.nuc_3d_chk.setChecked(False)

    monkeypatch.setattr(
        mod.cellpose_runner,
        "run_nucleus_frame",
        lambda frame, z, params: (
            np.ones(frame.shape[-2:], dtype=np.float32),
            np.ones((2, *frame.shape[-2:]), dtype=np.float32),
        ),
    )

    w.nucleus_preview_btn.click()

    reference = viewer.layers["Reference: Nucleus 3D+t"].data
    prob = viewer.layers["Preview: Nucleus prob"].data
    assert prob.shape == reference.shape
    assert np.count_nonzero(prob) == 6 * 6
    assert np.all(prob[1, 2] > 0)
    w.deleteLater()


def test_nucleus_preview_3d_creates_volume_layers(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    viewer = _FakeViewer()
    viewer.dims.current_step = (1, 0)
    w = mod.CellposeWidget(viewer)
    _write_test_stack(tmp_path / "0_input" / "nucleus.tif", (3, 4, 6, 6))
    w.refresh(tmp_path)
    w.nuc_3d_chk.setChecked(True)
    w.nucleus_preview_btn.click()
    prob = viewer.layers["Preview: Nucleus prob"].data
    flow = viewer.layers["Preview: Nucleus flow"].data
    assert prob.shape == (3, 4, 6, 6)
    assert flow.shape == (3, 4, 6, 6)
    w.deleteLater()


def test_nucleus_preview_3d_reports_status_before_inference(
    _mock_cellpose, monkeypatch, tmp_path
):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    viewer = _FakeViewer()
    viewer.dims.current_step = (1, 0)
    w = mod.CellposeWidget(viewer)
    _write_test_stack(tmp_path / "0_input" / "nucleus.tif", (3, 4, 6, 6))
    w.refresh(tmp_path)
    w.nuc_3d_chk.setChecked(True)

    seen = {}

    def _slow_preview(frame, z, params):
        seen["status"] = w.status_lbl.text()
        seen["progress_hidden"] = w.progress_bar.isHidden()
        seen["progress_range"] = (w.progress_bar.minimum(), w.progress_bar.maximum())
        prob = np.zeros(frame.shape, dtype=np.float32)
        dp = np.zeros((3, *frame.shape), dtype=np.float32)
        return prob, dp

    monkeypatch.setattr(mod.cellpose_runner, "run_nucleus_frame", _slow_preview)
    w.nucleus_preview_btn.click()

    assert seen["status"].startswith("Previewing nucleus 3D")
    assert seen["progress_hidden"] is False
    assert seen["progress_range"] == (0, 0)
    w.deleteLater()


def test_cell_preview_creates_2d_layers(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    viewer = _FakeViewer()
    viewer.dims.current_step = (2, 1)
    w = mod.CellposeWidget(viewer)
    _write_test_stack(tmp_path / "0_input" / "cell.tif", (3, 4, 5, 5))
    w.refresh(tmp_path)
    w.cell_preview_btn.click()
    prob = viewer.layers["Preview: Cell prob"].data
    flow = viewer.layers["Preview: Cell flow"].data
    assert prob.shape == (3, 4, 5, 5)
    assert flow.shape == (3, 4, 5, 5)
    w.deleteLater()


def test_cell_preview_loads_reference_stack_before_selecting_slice(
    _mock_cellpose, monkeypatch, tmp_path
):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    class _MiddleSliceViewer(_FakeViewer):
        def add_image(self, data, *, name, **kwargs):
            layer = super().add_image(data, name=name, **kwargs)
            if name == "Reference: Cell 3D+t":
                arr = np.asarray(data)
                self.dims.current_step = (0, arr.shape[1] // 2)
            return layer

    viewer = _MiddleSliceViewer()
    viewer.dims.current_step = (0, 0)
    w = mod.CellposeWidget(viewer)
    _write_test_stack(tmp_path / "0_input" / "cell.tif", (1, 5, 6, 6))
    w.refresh(tmp_path)

    seen = {}

    def _record_preview(frame, z, params):
        seen["shape"] = frame.shape
        seen["z"] = z
        return (
            np.zeros(frame.shape[-2:], dtype=np.float32),
            np.zeros((2, *frame.shape[-2:]), dtype=np.float32),
        )

    monkeypatch.setattr(mod.cellpose_runner, "run_cell_frame", _record_preview)
    w.cell_preview_btn.click()

    assert "Reference: Cell 3D+t" in viewer.layers
    assert viewer.layers["Reference: Cell 3D+t"].data.shape == (1, 5, 6, 6)
    assert seen == {"shape": (5, 6, 6), "z": 2}
    w.deleteLater()


def test_cell_preview_keeps_reference_time_and_z_axes(
    _mock_cellpose, monkeypatch, tmp_path
):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    viewer = _FakeViewer()
    viewer.dims.current_step = (2, 1)
    w = mod.CellposeWidget(viewer)
    _write_test_stack(tmp_path / "0_input" / "cell.tif", (3, 4, 5, 5))
    w.refresh(tmp_path)

    monkeypatch.setattr(
        mod.cellpose_runner,
        "run_cell_frame",
        lambda frame, z, params: (
            np.ones(frame.shape[-2:], dtype=np.float32),
            np.ones((2, *frame.shape[-2:]), dtype=np.float32),
        ),
    )

    w.cell_preview_btn.click()

    reference = viewer.layers["Reference: Cell 3D+t"].data
    prob = viewer.layers["Preview: Cell prob"].data
    assert prob.shape == reference.shape
    assert np.count_nonzero(prob) == 5 * 5
    assert np.all(prob[2, 1] > 0)
    w.deleteLater()


def test_preview_reports_missing_input(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    w.refresh(tmp_path)
    w.nucleus_preview_btn.click()
    assert "missing" in w.status_lbl.text().lower()
    w.deleteLater()


# ── standalone seam (itasc-cellpose distribution) ───────────────────────────

def test_standalone_shows_pickers_and_keeps_files_panel(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer(), standalone=True)

    # The three standalone input/output pickers exist and are visible …
    for edit in (w._nucleus_edit, w._cell_edit, w._output_dir_edit):
        assert edit is not None
    assert w._paths_container.isVisibleTo(w)
    # … and the staged pipeline-files panel stays (the 0_input/1_cellpose layout
    # is the on-disk contract even standalone).
    assert w.pipeline_files_header.isVisibleTo(w)

    w.deleteLater()
    app.processEvents()


def test_standalone_paths_resolve_to_explicit_files(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer(), standalone=True)

    nuc = tmp_path / "nucleus.tif"
    cel = tmp_path / "cell.tif"
    out = tmp_path / "out"
    w._nucleus_edit.setText(str(nuc))
    w._cell_edit.setText(str(cel))
    w._output_dir_edit.setText(str(out))
    w._apply_standalone_paths()

    assert w._input_path("nucleus") == nuc
    assert w._input_path("cell") == cel
    assert w._output_dir() == out
    # _pos_dir is set to the output dir so the run/preview guards pass, and the
    # embedded divergence widget resolves its maps flatly under the same dir.
    assert w._pos_dir == out
    assert w.divergence_maps_widget._resolved_maps_dir() == out

    w.deleteLater()
    app.processEvents()


def test_standalone_factory_returns_standalone_widget(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    import napari

    monkeypatch.setattr(napari, "current_viewer", lambda: _FakeViewer())
    w = mod.make_cellpose_widget()

    assert isinstance(w, mod.CellposeWidget)
    assert w._standalone is True
    assert w._paths_container.isVisibleTo(w)

    w.deleteLater()
    app.processEvents()


# ── input-layout support (2D / 2D+t / 3D / 3D+t) ───────────────────────────────

def test_layout_combos_exist_and_default_to_3dt(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())
    for combo in (w.nuc_layout_combo, w.cell_layout_combo):
        items = [combo.itemText(i) for i in range(combo.count())]
        assert items == ["2D", "2D+t", "3D", "3D+t"]
        assert combo.currentText() == "3D+t"
    w.deleteLater()


def test_nucleus_zless_layout_disables_3d_controls_and_forces_2d(_mock_cellpose, monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())

    # A 3D+t layout keeps the true-3D controls live.
    w.nuc_layout_combo.setCurrentText("3D+t")
    assert w.nuc_3d_chk.isEnabled()
    assert w.nuc_anisotropy_spin.isEnabled()

    # A Z-less layout disables them and forces do_3d off even if checked.
    w.nuc_layout_combo.setCurrentText("2D+t")
    assert not w.nuc_3d_chk.isEnabled()
    assert not w.nuc_anisotropy_spin.isEnabled()
    w.nuc_3d_chk.setChecked(True)
    assert w._build_nucleus_params().do_3d is False
    w.deleteLater()


def test_run_accepts_2d_input(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    _write_test_stack(tmp_path / "0_input" / "nucleus.tif", (6, 6))  # 2-D
    w.refresh(tmp_path)
    # ndim 2 is unambiguous → layout auto-selected.
    assert w.nuc_layout_combo.currentText() == "2D"
    w.nucleus_run_btn.click()
    out = tmp_path / "1_cellpose"
    assert (out / "nucleus_prob.tif").exists()
    assert (out / "nucleus_dp.tif").exists()
    assert "complete" in w.status_lbl.text().lower()
    w.deleteLater()


def test_run_accepts_3d_zstack_with_explicit_layout(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    _write_test_stack(tmp_path / "0_input" / "nucleus.tif", (4, 6, 6))  # 3-D z-stack
    w.refresh(tmp_path)
    # ndim 3 is ambiguous → auto-select leaves the default; user disambiguates.
    w.nuc_layout_combo.setCurrentText("3D")
    w.nucleus_run_btn.click()
    out = tmp_path / "1_cellpose"
    assert (out / "nucleus_prob.tif").exists()
    assert (out / "nucleus_dp.tif").exists()
    assert "complete" in w.status_lbl.text().lower()
    w.deleteLater()


def test_channels_run_independently_when_only_one_present(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    # Only the cell channel is supplied.
    _write_test_stack(tmp_path / "0_input" / "cell.tif", (2, 3, 6, 6))
    w.refresh(tmp_path)

    # Nucleus has no input → reports missing, writes nothing.
    w.nucleus_run_btn.click()
    assert "missing" in w.status_lbl.text().lower()
    assert not (tmp_path / "1_cellpose" / "nucleus_prob.tif").exists()

    # Cell still runs to completion.
    w.cell_run_btn.click()
    assert (tmp_path / "1_cellpose" / "cell_prob.tif").exists()
    assert "complete" in w.status_lbl.text().lower()
    w.deleteLater()


def test_integrated_inputs_are_hidden_and_default_to_0input(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    w = mod.CellposeWidget(_FakeViewer())  # integrated (orchestrated) mode

    # In the full app every path is fixed by the project structure, so the whole
    # picker block (both input rows and the output-dir row) is hidden.
    assert not w._paths_container.isVisibleTo(w)

    w.refresh(tmp_path)

    # Blank fields → canonical 0_input defaults.
    assert w._input_path("nucleus") == tmp_path / "0_input" / "nucleus.tif"
    assert w._input_path("cell") == tmp_path / "0_input" / "cell.tif"

    # The resolution logic still honours a set value (used programmatically): a
    # relative one resolves under the position dir, an absolute one verbatim.
    w._nucleus_edit.setText("raw/my_nuc.tif")
    assert w._input_path("nucleus") == tmp_path / "raw" / "my_nuc.tif"

    elsewhere = tmp_path.parent / "elsewhere" / "cell_stack.tif"
    w._cell_edit.setText(str(elsewhere))
    assert w._input_path("cell") == elsewhere

    w.deleteLater()
    app.processEvents()
