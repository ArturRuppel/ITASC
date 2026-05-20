"""Focused tests for CellParamsWidget."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QLabel, QToolButton


def _install_import_stubs() -> None:
    src_root = Path(__file__).resolve().parents[2] / "src" / "cellflow"
    package_root = src_root / "napari"

    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    sys.modules["cellflow.napari"] = napari_pkg

    # Stub cellflow.segmentation so FlowFollowingParams import works without heavy deps
    class _FakeFlowFollowingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    seg_module = types.ModuleType("cellflow.segmentation")
    seg_module.FlowFollowingParams = _FakeFlowFollowingParams
    sys.modules["cellflow.segmentation"] = seg_module


def _load_widget_class():
    _install_import_stubs()
    # Force reimport in case stubs changed
    sys.modules.pop("cellflow.napari.cell_params_widget", None)
    module = importlib.import_module("cellflow.napari.cell_params_widget")
    return module.CellParamsWidget, module


# ── Structure ─────────────────────────────────────────────────────────────────


def test_cell_params_widget_section_title_and_collapsed_by_default():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    assert widget.section.title == "Parameters"
    assert widget.section.is_expanded is False

    widget.deleteLater()


def test_cell_params_widget_section_headings_cover_all_stages():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    section_labels = {lbl.text() for lbl in widget.section.findChildren(QLabel)}

    assert {
        "Flow Filtering",
        "Foreground",
        "Contour — Cellprob Sweep",
        "Contour — Flow-Following",
        "Contour — Gamma Averaging",
        "Contour — Temporal Stabilization",
        "Segmentation",
    } <= section_labels

    widget.deleteLater()


def test_cell_params_widget_exposes_all_flow_filtering_spinboxes_with_defaults():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    assert widget.ff_median_time_spin.value() == 3
    assert widget.ff_median_space_spin.value() == 5
    assert widget.ff_gauss_time_spin.value() == pytest.approx(0.0)
    assert widget.ff_gauss_space_spin.value() == pytest.approx(0.0)

    widget.deleteLater()


def test_cell_params_widget_exposes_foreground_spinbox_with_default():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    assert widget.fg_cellprob_threshold_spin.value() == pytest.approx(0.5)

    widget.deleteLater()


def test_cell_params_widget_exposes_contour_sweep_spinboxes_with_defaults():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    assert widget.cp_min_spin.value() == pytest.approx(0.05)
    assert widget.cp_max_spin.value() == pytest.approx(0.50)
    assert widget.cp_step_spin.value() == pytest.approx(0.05)

    widget.deleteLater()


def test_cell_params_widget_exposes_contour_ff_spinboxes_with_defaults():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    assert widget.ff_flow_weight_spin.value() == pytest.approx(0.5)
    assert widget.ff_step_scale_spin.value() == pytest.approx(0.2)
    assert widget.ff_max_iter_spin.value() == 100

    widget.deleteLater()


def test_cell_params_widget_exposes_gamma_spinboxes_with_defaults():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    assert widget.gamma_min_spin.value() == pytest.approx(1.0)
    assert widget.gamma_max_spin.value() == pytest.approx(1.0)
    assert widget.gamma_step_spin.value() == pytest.approx(0.25)

    widget.deleteLater()


def test_cell_params_widget_exposes_temporal_stabilization_spinboxes():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    assert widget.memory_tau_spin.value() == pytest.approx(0.0)
    assert widget.memory_floor_spin.value() == pytest.approx(0.01)

    widget.deleteLater()


def test_cell_params_widget_exposes_segmentation_spinboxes_with_defaults():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    assert widget.alpha_unary_spin.value() == pytest.approx(4.0)
    assert widget.lambda_s_spin.value() == pytest.approx(1.0)
    assert widget.beta_s_spin.value() == pytest.approx(5.0)
    assert widget.lambda_t_spin.value() == pytest.approx(1.0)
    assert widget.gamma_unary_spin.value() == pytest.approx(0.0)
    assert widget.n_workers_spin.value() >= 1

    widget.deleteLater()


# ── Config builder methods ────────────────────────────────────────────────────


def test_flow_filter_params_reflects_current_spinbox_values():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    widget.ff_median_time_spin.setValue(5)
    widget.ff_median_space_spin.setValue(7)
    widget.ff_gauss_time_spin.setValue(1.5)
    widget.ff_gauss_space_spin.setValue(2.0)

    params = widget.flow_filter_params()

    assert params.median_kernel_time == 5
    assert params.median_kernel_space == 7
    assert params.gaussian_sigma_time == pytest.approx(1.5)
    assert params.gaussian_sigma_space == pytest.approx(2.0)

    widget.deleteLater()


def test_cellprob_thresholds_produces_correct_arange():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    widget.cp_min_spin.setValue(0.1)
    widget.cp_max_spin.setValue(0.3)
    widget.cp_step_spin.setValue(0.1)

    thresholds = widget.cellprob_thresholds()

    assert len(thresholds) == 3
    assert thresholds[0] == pytest.approx(0.1)
    assert thresholds[1] == pytest.approx(0.2)
    assert thresholds[2] == pytest.approx(0.3)

    widget.deleteLater()


def test_gammas_produces_correct_arange():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    widget.gamma_min_spin.setValue(0.5)
    widget.gamma_max_spin.setValue(1.0)
    widget.gamma_step_spin.setValue(0.25)

    gammas = widget.gammas()

    assert len(gammas) == 3
    assert gammas[0] == pytest.approx(0.5)
    assert gammas[1] == pytest.approx(0.75)
    assert gammas[2] == pytest.approx(1.0)

    widget.deleteLater()


def test_contour_ff_params_reflects_current_spinbox_values():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    widget.ff_flow_weight_spin.setValue(0.8)
    widget.ff_step_scale_spin.setValue(0.3)
    widget.ff_max_iter_spin.setValue(200)

    params = widget.contour_ff_params()

    assert params.flow_weight == pytest.approx(0.8)
    assert params.flow_step_scale == pytest.approx(0.3)
    assert params.max_iterations == 200
    # These are fixed for contour flow-following
    assert params.median_kernel_time == 1
    assert params.median_kernel_space == 1
    assert params.gaussian_sigma_time == pytest.approx(0.0)
    assert params.gaussian_sigma_space == pytest.approx(0.0)

    widget.deleteLater()


def test_ff_max_iter_slider_has_step_buttons():
    _app = QApplication.instance() or QApplication([])
    widget_class, _mod = _load_widget_class()
    widget = widget_class()

    buttons = {
        button.objectName(): button
        for button in widget.ff_max_iter_spin.findChildren(QToolButton)
    }

    decrement = buttons["slider_decrement_button"]
    increment = buttons["slider_increment_button"]

    widget.ff_max_iter_spin.setValue(100)
    increment.click()
    assert widget.ff_max_iter_spin.value() == 110

    decrement.click()
    assert widget.ff_max_iter_spin.value() == 100

    widget.ff_max_iter_spin.setValue(widget.ff_max_iter_spin.minimum())
    decrement.click()
    assert widget.ff_max_iter_spin.value() == widget.ff_max_iter_spin.minimum()

    widget.deleteLater()


# ── Delegation seam test ──────────────────────────────────────────────────────


def test_cell_workflow_delegates_params_controls_to_child_widget(monkeypatch):
    """CellWorkflowWidget composes CellParamsWidget and exposes control aliases."""
    _app = QApplication.instance() or QApplication([])

    src_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(src_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    for key in list(sys.modules):
        if "cell_workflow_widget" in key or "cell_params_widget" in key:
            sys.modules.pop(key)

    # Stubs for heavy optional deps pulled in by cell_workflow_widget
    class _FakeFlowFollowingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    seg_module = types.ModuleType("cellflow.segmentation")
    seg_module.FlowFollowingParams = _FakeFlowFollowingParams
    seg_module.apply_gamma = lambda logits, gamma: logits
    seg_module.build_consensus_boundary_flow_following = lambda *a, **kw: (None, None)
    monkeypatch.setitem(sys.modules, "cellflow.segmentation", seg_module)

    seg_arr_module = types.ModuleType("cellflow.segmentation._array_utils")
    seg_arr_module.normalize_seeded_watershed_dp_stack = lambda dp, shape: dp
    monkeypatch.setitem(sys.modules, "cellflow.segmentation._array_utils", seg_arr_module)

    contour_filt_module = types.ModuleType("cellflow.segmentation.contour_filtering")
    contour_filt_module.contour_memory_filter = lambda arr, **kw: arr
    monkeypatch.setitem(sys.modules, "cellflow.segmentation.contour_filtering", contour_filt_module)

    workflow_mod = importlib.import_module("cellflow.napari.cell_workflow_widget")
    params_mod = importlib.import_module("cellflow.napari.cell_params_widget")

    from types import SimpleNamespace

    class _FakeEvent:
        def connect(self, cb): pass
        def disconnect(self, cb): pass

    class _FakeEvents:
        def __init__(self):
            self.data = _FakeEvent()
            self.paint = _FakeEvent()
            self.mode = _FakeEvent()
            self.removed = _FakeEvent()

    class _FakeSelection:
        def __init__(self):
            self.active = None

    class _FakeLayerList(dict):
        def __init__(self):
            super().__init__()
            self.selection = _FakeSelection()
            self.events = _FakeEvents()

        def remove(self, layer):
            self.pop(layer.name, None)

    class _FakeViewer:
        def __init__(self):
            self.layers = _FakeLayerList()
            self.mouse_drag_callbacks = []
            self.dims = SimpleNamespace(
                current_step=(0,),
                events=SimpleNamespace(
                    current_step=SimpleNamespace(connect=lambda cb: None)
                ),
            )

        def add_image(self, data, *, name, **kwargs):
            pass

        def add_labels(self, data, *, name, **kwargs):
            pass

        def add_shapes(self, *, name, **kwargs):
            pass

    viewer = _FakeViewer()
    widget = workflow_mod.CellWorkflowWidget(viewer)

    assert isinstance(widget.cell_params_widget, params_mod.CellParamsWidget)
    # All control aliases point into the child widget
    p = widget.cell_params_widget
    assert widget.ff_median_time_spin is p.ff_median_time_spin
    assert widget.ff_median_space_spin is p.ff_median_space_spin
    assert widget.ff_gauss_time_spin is p.ff_gauss_time_spin
    assert widget.ff_gauss_space_spin is p.ff_gauss_space_spin
    assert widget.fg_cellprob_threshold_spin is p.fg_cellprob_threshold_spin
    assert widget.cp_min_spin is p.cp_min_spin
    assert widget.cp_max_spin is p.cp_max_spin
    assert widget.cp_step_spin is p.cp_step_spin
    assert widget.ff_flow_weight_spin is p.ff_flow_weight_spin
    assert widget.ff_step_scale_spin is p.ff_step_scale_spin
    assert widget.ff_max_iter_spin is p.ff_max_iter_spin
    assert widget.gamma_min_spin is p.gamma_min_spin
    assert widget.gamma_max_spin is p.gamma_max_spin
    assert widget.gamma_step_spin is p.gamma_step_spin
    assert widget.memory_tau_spin is p.memory_tau_spin
    assert widget.memory_floor_spin is p.memory_floor_spin
    assert widget.alpha_unary_spin is p.alpha_unary_spin
    assert widget.lambda_s_spin is p.lambda_s_spin
    assert widget.beta_s_spin is p.beta_s_spin
    assert widget.lambda_t_spin is p.lambda_t_spin
    assert widget.gamma_unary_spin is p.gamma_unary_spin
    assert widget.n_workers_spin is p.n_workers_spin

    widget.deleteLater()
