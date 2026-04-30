from __future__ import annotations

import importlib
import os
import shlex
import sys
import types
from pathlib import Path

import pytest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import napari
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import QApplication, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget


def _make_viewer():
    app = QApplication.instance() or QApplication([])
    viewer = napari.Viewer(show=False)
    return app, viewer


def _install_import_stubs() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    sys.modules["cellflow.napari"] = napari_pkg

    tracking_pkg = types.ModuleType("cellflow.tracking_ultrack")
    tracking_pkg.__path__ = []
    sys.modules["cellflow.tracking_ultrack"] = tracking_pkg

    class _StubTrackingConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    stub_exports = {
        "cellflow.tracking_ultrack.config": {"TrackingConfig": _StubTrackingConfig},
        "cellflow.tracking_ultrack.export": {"export_tracked_labels": lambda *args, **kwargs: None},
        "cellflow.tracking_ultrack.ingest": {
            "ingest_hypotheses_to_db": lambda *args, **kwargs: None,
            "_select_solver": lambda: "CBC",
        },
        "cellflow.tracking_ultrack.linking": {"run_linking": lambda *args, **kwargs: iter(())},
        "cellflow.tracking_ultrack.extend": {"extend_track": lambda *args, **kwargs: None},
        "cellflow.tracking_ultrack.reseed": {"resolve_with_validation": lambda *args, **kwargs: None},
        "cellflow.tracking_ultrack.solve": {"run_solve": lambda *args, **kwargs: iter(())},
    }

    for module_name, attrs in stub_exports.items():
        module = types.ModuleType(module_name)
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        sys.modules[module_name] = module


def _install_main_widget_stubs() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    sys.modules["cellflow.napari"] = napari_pkg

    class _StubWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()

    stub_modules = {
        "cellflow.napari.analysis_widget": {"AnalysisWidget": _StubWidget},
        "cellflow.napari.cell_workflow_widget": {"CellWorkflowWidget": _StubWidget},
        "cellflow.napari.cellpose_widget": {"CellposeWidget": _StubWidget},
        "cellflow.napari.correction_widget": {"CorrectionWidget": _StubWidget},
        "cellflow.napari.data_panel_widget": {"ProjectStatusPanel": _StubWidget},
        "cellflow.napari.data_prep_widget": {"DataPrepWidget": _StubWidget},
        "cellflow.napari.nucleus_workflow_widget": {"NucleusWorkflowWidget": _StubWidget},
    }

    for module_name, attrs in stub_modules.items():
        module = types.ModuleType(module_name)
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        sys.modules[module_name] = module


def _load_widget_class():
    _install_import_stubs()
    module = importlib.import_module("cellflow.napari.nucleus_workflow_widget")
    return module.NucleusWorkflowWidget


def _load_main_widget_class():
    _install_main_widget_stubs()
    module = importlib.import_module("cellflow.napari.main_widget")
    for module_name in (
        "cellflow.napari.analysis_widget",
        "cellflow.napari.cell_workflow_widget",
        "cellflow.napari.cellpose_widget",
        "cellflow.napari.correction_widget",
        "cellflow.napari.data_panel_widget",
        "cellflow.napari.data_prep_widget",
        "cellflow.napari.nucleus_workflow_widget",
    ):
        sys.modules.pop(module_name, None)
    return module.CellFlowMainWidget


def _load_correction_widget_class():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    sys.modules["cellflow.napari"] = napari_pkg

    module = importlib.import_module("cellflow.napari.correction_widget")
    return module.CorrectionWidget


def _install_terminal_capture(monkeypatch):
    captured = {}
    utils_module = types.ModuleType("cellflow.napari.utils")

    def _capture_launch(cmd):
        captured["cmd"] = cmd

    utils_module.launch_in_terminal = _capture_launch
    monkeypatch.setitem(sys.modules, "cellflow.napari.utils", utils_module)
    return captured


def _read_launched_script(captured: dict) -> str:
    cmd_parts = shlex.split(captured["cmd"])
    return Path(cmd_parts[-1]).read_text()


def test_main_widget_labels_the_outer_nucleus_workflow_section():
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    assert widget.nucleus_section.title == "3. Nucleus Segmentation & Tracking"
    assert widget.nucleus_section._toggle.text() == "3. Nucleus Segmentation && Tracking"

    widget.deleteLater()
    viewer.close()


def test_main_widget_project_header_uses_compact_action_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    for button in (
        widget.project_btn,
        widget.save_btn,
        widget.save_as_btn,
        widget.load_btn,
        widget.load_from_btn,
    ):
        assert button.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed
        assert "font-size: 8pt" in button.styleSheet()
        assert "padding: 1px 4px" in button.styleSheet()

    assert widget.refresh_btn.minimumWidth() == 24
    assert widget.refresh_btn.maximumWidth() == 24

    widget.deleteLater()
    viewer.close()


def test_tracking_correction_shell_exposes_stable_section_attributes():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.tracking_correction_section.title == "4. Tracking & Correction"
    assert widget.tracking_correction_section._toggle.text() == "4. Tracking && Correction"
    assert widget.ultrack_section.title == "Ultrack Tracking"
    assert widget.correction_section.title == "Correction"
    assert widget.correction_shortcuts_section.title == "Correction Shortcuts"
    assert widget.correction_shortcuts_section.is_expanded is True
    assert widget.correction_shortcuts_section.findChildren(QScrollArea) == []
    correction_inner = widget.correction_section._content_frame.layout().itemAt(0).widget()
    correction_layout = correction_inner.layout()
    assert correction_layout.indexOf(widget.correction_widget) != -1
    assert correction_layout.indexOf(widget.correction_shortcuts_section) != -1
    assert (
        correction_layout.indexOf(widget.correction_shortcuts_section)
        > correction_layout.indexOf(widget.correction_widget)
    )

    correction_button_texts = {
        button.text()
        for button in widget.correction_section.findChildren(QPushButton)
    }
    ultrack_button_texts = {
        button.text()
        for button in widget.ultrack_section.findChildren(QPushButton)
    }
    route_checkboxes = widget.ultrack_section.findChildren(type(widget.ultrack_route_check))

    assert "Save Tracked Labels" in correction_button_texts
    assert "Load Tracked Labels" in correction_button_texts
    assert "Reassign IDs" in correction_button_texts
    assert "◀ Extend (A)" in correction_button_texts
    assert "Extend (D) ▶" in correction_button_texts
    assert "◀ Retrack (Q)" in correction_button_texts
    assert "Retrack (E) ▶" in correction_button_texts
    shortcut_keys = {
        shortcut.key().toString(QKeySequence.SequenceFormat.PortableText)
        for shortcut in widget.findChildren(QShortcut)
    }
    assert {"A", "D", "Q", "E"} <= shortcut_keys
    assert "Save Tracked Labels" not in ultrack_button_texts
    assert "Load Tracked Labels" not in ultrack_button_texts
    assert "Reassign IDs" not in ultrack_button_texts
    assert widget.ultrack_route_check in route_checkboxes
    assert widget.ultrack_route_check not in widget.correction_section.findChildren(
        type(widget.ultrack_route_check)
    )

    widget.deleteLater()
    viewer.close()


def test_tracking_correction_action_buttons_expand_horizontally():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    tracked_buttons = [
        widget.run_ultrack_btn,
        widget.ultrack_terminal_btn,
        widget.extend_back_btn,
        widget.extend_fwd_btn,
        widget.retrack_back_btn,
        widget.retrack_fwd_btn,
        widget.save_tracked_btn,
        widget.load_tracked_btn,
        widget.reassign_ids_btn,
    ]

    for button in tracked_buttons:
        assert button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding

    widget.deleteLater()
    viewer.close()


def test_ultrack_section_exposes_route_selector_and_local_status():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_route_check.text() == "Resolve from validated"
    assert widget.ultrack_route_check in widget.ultrack_section.findChildren(
        type(widget.ultrack_route_check)
    )
    assert widget.ultrack_status_lbl.text() == ""
    assert widget.ultrack_progress_bar.isVisible() is False

    widget.deleteLater()
    viewer.close()


def test_ultrack_section_exposes_validated_seed_prior_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_power_spin.value() == 4.0
    assert widget.ultrack_quality_exp_spin.value() == 8.0
    assert widget.ultrack_seed_weight_spin.value() == 0.5
    assert widget.ultrack_seed_space_spin.value() == 25.0
    assert widget.ultrack_seed_time_spin.value() == 2.0
    assert widget.ultrack_seed_window_spin.value() == 5

    assert "solver transform" in widget.ultrack_power_spin.toolTip()
    assert "node_prob" in widget.ultrack_quality_exp_spin.toolTip()
    assert "validated cells" in widget.ultrack_seed_weight_spin.toolTip()

    widget.deleteLater()
    viewer.close()


def test_validated_seed_prior_controls_follow_resolve_checkbox():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    controls = [
        widget.ultrack_quality_exp_spin,
        widget.ultrack_seed_weight_spin,
        widget.ultrack_seed_space_spin,
        widget.ultrack_seed_time_spin,
        widget.ultrack_seed_window_spin,
    ]

    widget.ultrack_route_check.setChecked(False)
    _app.processEvents()
    assert all(not control.isEnabled() for control in controls)
    assert widget.ultrack_power_spin.isEnabled()

    widget.ultrack_route_check.setChecked(True)
    _app.processEvents()
    assert all(control.isEnabled() for control in controls)
    assert widget.ultrack_power_spin.isEnabled()

    widget.deleteLater()
    viewer.close()


def test_ultrack_seed_prior_controls_persist_through_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.ultrack_route_check.setChecked(True)
    widget.ultrack_power_spin.setValue(3.5)
    widget.ultrack_quality_exp_spin.setValue(9.0)
    widget.ultrack_seed_weight_spin.setValue(0.75)
    widget.ultrack_seed_space_spin.setValue(30.0)
    widget.ultrack_seed_time_spin.setValue(3.0)
    widget.ultrack_seed_window_spin.setValue(7)

    state = widget.get_state()
    widget.deleteLater()

    widget = widget_class(viewer)
    widget.set_state(state)

    assert widget.ultrack_route_check.isChecked()
    assert widget.ultrack_power_spin.value() == 3.5
    assert widget.ultrack_quality_exp_spin.value() == 9.0
    assert widget.ultrack_seed_weight_spin.value() == 0.75
    assert widget.ultrack_seed_space_spin.value() == 30.0
    assert widget.ultrack_seed_time_spin.value() == 3.0
    assert widget.ultrack_seed_window_spin.value() == 7

    widget.deleteLater()
    viewer.close()


def test_ultrack_terminal_script_includes_visible_config_controls(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    captured = _install_terminal_capture(monkeypatch)

    pos_dir = tmp_path / "pos00"
    nucleus_dir = pos_dir / "2_nucleus"
    nucleus_dir.mkdir(parents=True)
    (nucleus_dir / "hypotheses.h5").touch()
    widget._pos_dir = pos_dir
    widget.ultrack_power_spin.setValue(3.25)
    widget.ultrack_quality_exp_spin.setValue(9.5)
    widget.ultrack_seed_weight_spin.setValue(0.85)
    widget.ultrack_seed_space_spin.setValue(35.0)
    widget.ultrack_seed_time_spin.setValue(4.0)
    widget.ultrack_seed_window_spin.setValue(8)

    widget._on_ultrack_terminal()
    script = _read_launched_script(captured)

    assert "power=3.25" in script
    assert "quality_exponent=9.5" in script
    assert "seed_weight=0.85" in script
    assert "seed_sigma_space=35.0" in script
    assert "seed_tau_time=4.0" in script
    assert "seed_max_dt=8" in script

    widget.deleteLater()
    viewer.close()


def test_resolve_terminal_script_includes_validated_seed_prior_controls(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    captured = _install_terminal_capture(monkeypatch)
    monkeypatch.setattr(module, "read_validated_tracks", lambda _pos_dir: {12: {0, 1}})

    pos_dir = tmp_path / "pos00"
    nucleus_dir = pos_dir / "2_nucleus"
    nucleus_dir.mkdir(parents=True)
    (nucleus_dir / "hypotheses.h5").touch()
    (nucleus_dir / "tracked_labels.tif").touch()
    widget._pos_dir = pos_dir
    widget.ultrack_power_spin.setValue(3.75)
    widget.ultrack_quality_exp_spin.setValue(10.5)
    widget.ultrack_seed_weight_spin.setValue(0.9)
    widget.ultrack_seed_space_spin.setValue(40.0)
    widget.ultrack_seed_time_spin.setValue(4.5)
    widget.ultrack_seed_window_spin.setValue(9)

    widget._on_resolve_terminal()
    script = _read_launched_script(captured)

    assert "power=3.75" in script
    assert "quality_exponent=10.5" in script
    assert "seed_weight=0.9" in script
    assert "seed_sigma_space=40.0" in script
    assert "seed_tau_time=4.5" in script
    assert "seed_max_dt=9" in script

    widget.deleteLater()
    viewer.close()


def test_tracking_correction_widget_allows_horizontal_scrolling_when_narrow():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    outer = QWidget()
    outer_layout = QVBoxLayout(outer)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    outer_layout.addWidget(scroll)
    outer.show()
    scroll.resize(196, 260)
    outer.resize(220, 300)
    _app.processEvents()

    assert scroll.horizontalScrollBar().maximum() > 0
    assert widget.minimumSizeHint().width() > scroll.viewport().width()

    outer.deleteLater()
    viewer.close()


def test_tracking_correction_widget_minimum_width_stays_compact():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.tracking_correction_section.expand()
    widget.ultrack_section.expand()
    widget.correction_section.expand()
    widget.correction_shortcuts_section.expand()
    _app.processEvents()

    # The old broken layout measured well above 600px because the shortcut
    # help was unwrapped and multiple long controls forced wide horizontal
    # rows. Keep a generous ceiling so the test remains stable across Qt
    # styles while still catching a return to the original oversized layout.
    assert widget.minimumSizeHint().width() < 560

    widget.deleteLater()
    viewer.close()


def test_contour_maps_parameters_expand_and_scroll_when_narrow():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.contour_section.expand()
    _app.processEvents()

    scroll_areas = widget.contour_section.findChildren(QScrollArea)
    assert len(scroll_areas) == 1

    params_scroll = scroll_areas[0]
    assert params_scroll.widgetResizable() is True
    assert params_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded
    assert params_scroll.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert params_scroll.widget().sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding

    for spin in (
        widget.cp_min_spin,
        widget.cp_max_spin,
        widget.cp_step_spin,
        widget.cp_gamma_min_spin,
        widget.cp_gamma_max_spin,
        widget.cp_gamma_step_spin,
    ):
        assert spin.minimumWidth() == 54
        assert spin.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding

    host = QWidget()
    host_layout = QVBoxLayout(host)
    host_layout.setContentsMargins(0, 0, 0, 0)
    host_layout.addWidget(widget)
    host.resize(180, 320)
    host.show()
    _app.processEvents()

    assert widget.save_source_check.text() == "Save label images"
    assert widget.save_source_check.x() == widget.preview_contour_btn.x()
    assert widget.preview_contour_btn.x() == widget.cp_min_spin.x()
    assert widget.build_btn.x() == widget.cp_max_spin.x()
    assert widget.cancel_build_btn.x() == widget.cp_step_spin.x()
    assert abs(widget.preview_contour_btn.width() - widget.cp_min_spin.width()) <= 1
    assert abs(widget.build_btn.width() - widget.cp_max_spin.width()) <= 1
    assert abs(widget.cancel_build_btn.width() - widget.cp_step_spin.width()) <= 1

    assert params_scroll.horizontalScrollBar().maximum() > 0

    host.deleteLater()
    viewer.close()


def test_hypothesis_tuning_spinboxes_expand_equally_in_the_top_grid():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.gen_section.expand()
    widget.gen_tabs.setCurrentIndex(0)
    _app.processEvents()

    host = QWidget()
    host_layout = QVBoxLayout(host)
    host_layout.setContentsMargins(0, 0, 0, 0)
    host_layout.addWidget(widget)
    host.resize(220, 280)
    host.show()
    _app.processEvents()

    for spin in (
        widget.min_size_spin,
        widget.min_circularity_spin,
        widget.noise_scale,
        widget.noise_blur,
    ):
        assert spin.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding

    assert abs(widget.min_size_spin.width() - widget.min_circularity_spin.width()) <= 1
    assert abs(widget.noise_scale.width() - widget.noise_blur.width()) <= 1
    assert abs(widget.min_size_spin.width() - widget.noise_scale.width()) <= 1
    assert abs(widget.preview_btn.x() - widget.min_size_spin.x()) <= 2
    assert abs(widget.save_db_btn.x() - widget.min_circularity_spin.x()) <= 20
    assert widget.preview_btn.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert widget.save_db_btn.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert widget.preview_btn.width() >= 80
    assert widget.save_db_btn.width() >= 80

    host.deleteLater()
    viewer.close()


def test_tracking_correction_restores_two_column_button_and_parameter_layouts():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.tracking_correction_section.expand()
    widget.ultrack_section.expand()
    widget.correction_section.expand()
    widget.show()
    _app.processEvents()

    # Ultrack parameters should present as two side-by-side columns again.
    assert widget.ultrack_min_area_spin.y() == widget.ultrack_appear_spin.y()
    assert widget.ultrack_min_area_spin.x() < widget.ultrack_appear_spin.x()

    # Paired correction actions should also sit side-by-side.
    assert widget.extend_back_btn.y() == widget.extend_fwd_btn.y()
    assert widget.extend_back_btn.x() < widget.extend_fwd_btn.x()
    assert widget.retrack_back_btn.y() == widget.retrack_fwd_btn.y()
    assert widget.retrack_back_btn.x() < widget.retrack_fwd_btn.x()
    assert widget.save_tracked_btn.y() == widget.load_tracked_btn.y()
    assert widget.save_tracked_btn.x() < widget.load_tracked_btn.x()

    widget.deleteLater()
    viewer.close()


@pytest.mark.xfail(reason="Pending nucleus workflow refactor in nucleus_workflow_widget", strict=False)
def test_correction_section_has_no_separate_resolve_action_group():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    correction_button_texts = {
        button.text()
        for button in widget.correction_section.findChildren(QPushButton)
    }

    assert "Re-solve from validated" not in correction_button_texts
    assert "Resolve from validated" not in correction_button_texts
    assert "Run in Terminal" not in correction_button_texts

    widget.deleteLater()
    viewer.close()


def test_correction_section_exposes_extend_and_retrack_parameters():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.extend_params_section.title == "Extend Parameters"
    assert widget.extend_params_section.is_expanded is False
    assert widget.retrack_params_section.title == "Retrack Parameters"
    assert widget.retrack_params_section.is_expanded is False
    assert widget.extend_max_dist_spin.value() == 40.0
    assert widget.retrack_max_dist_spin.value() == 20.0

    widget.deleteLater()
    viewer.close()


def test_validated_overlay_uses_green_fill_at_full_opacity():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    viewer.add_labels(np.array([[[0, 1], [1, 0]]], dtype=np.uint8), name="Tracked: Nucleus")
    widget._add_validated_overlay(np.array([[[0, 1], [0, 0]]], dtype=np.uint8))

    layer = viewer.layers["Validated: Nucleus"]
    color = layer.get_color(1)

    assert layer.contour == 0
    assert layer.opacity == 1.0
    assert np.allclose(color[:3], [0.0, 1.0, 0.0], atol=1e-6)
    assert color[3] == 1.0

    widget.deleteLater()
    viewer.close()


def test_correction_widget_top_buttons_expand_horizontally():
    _app, viewer = _make_viewer()
    widget_class = _load_correction_widget_class()
    widget = widget_class(viewer)

    for button in (
        widget._activate_btn,
        widget._outline_btn,
        widget._reset_mode_btn,
        widget._goto_btn,
    ):
        assert button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding

    widget.deleteLater()
    viewer.close()
