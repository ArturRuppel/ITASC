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
    assert got["nucleus"] == new_state["nucleus"]
    assert got["cell"] == new_state["cell"]
    assert "divergence_maps" in got
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
    assert not hasattr(w, "zavg_viz_widget")
    assert all("prob_zavg" not in row._rel_path for row in w._files_widget._rows)
    w.deleteLater()


def test_embeds_divergence_maps_subwidget_and_state(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    divergence_mod = importlib.import_module("cellflow.napari.divergence_maps_widget")
    from cellflow.segmentation.divergence_maps import DivergenceMapsReport
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
        cellpose_dir / "cell_prob_3dt.tif",
        np.zeros((1, 1, 2, 2), dtype=np.float32),
    )
    tifffile.imwrite(
        cellpose_dir / "cell_dp_3dt.tif",
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
    assert captured["prob_path"].endswith("cell_prob_3dt.tif")
    assert captured["dp_path"].endswith("cell_dp_3dt.tif")
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


def test_run_nucleus_writes_outputs_and_updates_status(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    _write_test_stack(tmp_path / "0_input" / "nucleus_3dt.tif", (2, 3, 6, 6))
    w.refresh(tmp_path)
    w.nucleus_run_btn.click()
    out = tmp_path / "1_cellpose"
    assert (out / "nucleus_prob_3dt.tif").exists()
    assert (out / "nucleus_dp_3dt.tif").exists()
    assert "complete" in w.status_lbl.text().lower()
    assert w.nucleus_run_btn.text() == "▶"
    w.deleteLater()


def test_run_cell_writes_outputs(_mock_cellpose, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    w = mod.CellposeWidget(_FakeViewer())
    _write_test_stack(tmp_path / "0_input" / "cell_3dt.tif", (2, 3, 6, 6))
    w.refresh(tmp_path)
    w.cell_run_btn.click()
    out = tmp_path / "1_cellpose"
    assert (out / "cell_prob_3dt.tif").exists()
    assert (out / "cell_dp_3dt.tif").exists()
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
    _write_test_stack(tmp_path / "0_input" / "nucleus_3dt.tif", (3, 4, 6, 6))
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
    _write_test_stack(tmp_path / "0_input" / "nucleus_3dt.tif", (3, 4, 6, 6))
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
    _write_test_stack(tmp_path / "0_input" / "nucleus_3dt.tif", (3, 4, 6, 6))
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
    _write_test_stack(tmp_path / "0_input" / "nucleus_3dt.tif", (3, 4, 6, 6))
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
    _write_test_stack(tmp_path / "0_input" / "cell_3dt.tif", (3, 4, 5, 5))
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
    _write_test_stack(tmp_path / "0_input" / "cell_3dt.tif", (1, 5, 6, 6))
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
    _write_test_stack(tmp_path / "0_input" / "cell_3dt.tif", (3, 4, 5, 5))
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
