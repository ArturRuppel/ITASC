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
from qtpy.QtWidgets import QApplication, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget


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
            self.min_area = 100
            self.max_distance = 15.0
            self.max_neighbors = 5
            self.linking_mode = "default"
            self.iou_weight = 1.0
            self.power = 4.0
            self.quality_exponent = 8.0
            self.appear_weight = -0.001
            self.disappear_weight = -0.001
            self.division_weight = -0.001
            self.seed_weight = 0.5
            self.seed_sigma_space = 25.0
            self.seed_tau_time = 2.0
            self.seed_max_dt = 5
            self.max_segments_per_time = 1_000_000
            self.__dict__.update(kwargs)

    stub_exports = {
        "cellflow.tracking_ultrack.config": {"TrackingConfig": _StubTrackingConfig},
        "cellflow.tracking_ultrack.export": {"export_tracked_labels": lambda *args, **kwargs: None},
        "cellflow.tracking_ultrack.ingest": {
            "ingest_hypotheses_to_db": lambda *args, **kwargs: None,
            "_select_solver": lambda: "CBC",
            "_build_ultrack_config": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.linking": {"run_linking": lambda *args, **kwargs: iter(())},
        "cellflow.tracking_ultrack.extend": {
            "extend_track": lambda *args, **kwargs: None,
            "extend_track_from_db": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.reseed": {
            "resolve_with_validation": lambda *args, **kwargs: None,
            "resolve_with_canonical_segment": lambda *args, **kwargs: (None, {}),
        },
        "cellflow.tracking_ultrack.seed_prior": {"write_seed_prior_node_probs": lambda *args, **kwargs: None},
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
            self.refreshed_pos_dir = None

        def refresh(self, pos_dir):
            self.refreshed_pos_dir = pos_dir

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


def test_cell_workflow_required_inputs_exclude_optional_flow_vectors():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    source = (package_root / "cell_workflow_widget.py").read_text()
    input_section = source.split("self.input_files = PipelineFilesWidget([", 1)[1].split(
        "layout.addWidget(self.input_files)", 1
    )[0]

    assert "1_cellpose/cell_prob_3dt.tif" in input_section
    assert "1_cellpose/cell_dp_3dt.tif" in input_section
    assert "3_cell/foreground_masks.tif" in input_section
    assert "2_nucleus/tracked_labels.tif" in input_section


def test_main_widget_refreshes_cell_workflow_with_same_position_dir_as_project_status(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget.path_label.setText(str(tmp_path))

    widget._refresh_all()

    assert widget.data_panel.refreshed_pos_dir == pos_dir
    assert widget.cell_workflow_widget.refreshed_pos_dir == pos_dir

    widget.deleteLater()
    viewer.close()
    _app.processEvents()  # flush deferred deletion before next test


def test_tracking_correction_shell_exposes_stable_section_attributes():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    # tracking_correction_section wrapper removed — sections are now top-level
    assert not hasattr(widget, "tracking_correction_section")
    assert widget.ultrack_section.title == "5. Ultrack Tracking"
    assert widget.correction_section.title == "6. Correction"
    assert widget.correction_shortcuts_section.title == "Correction Shortcuts"
    assert widget.correction_shortcuts_section.is_expanded is False
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


def test_nucleus_workflow_status_labels_are_section_local():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "status_lbl")

    local_statuses = [
        (widget.contour_section, widget.contour_status_lbl),
        (widget.foreground_section, widget.fg_status_lbl),
        (widget.db_gen_section, widget.db_gen_status_lbl),
        (widget.ultrack_db_browser_section, widget.ultrack_db_section_status_lbl),
        (widget.ultrack_section, widget.ultrack_status_lbl),
        (widget.correction_section, widget.correction_status_lbl),
    ]
    for section, label in local_statuses:
        assert label.text() == ""
        assert label in section.findChildren(QLabel)

    widget._set_ultrack_status("Ultrack stayed local")

    assert widget.ultrack_status_lbl.text() == "Ultrack stayed local"
    assert not hasattr(widget, "status_lbl")

    widget.deleteLater()
    viewer.close()


def test_db_gen_section_exposes_quality_and_power_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.db_gen_power_spin.value() == 4.0
    assert widget.db_gen_quality_exp_spin.value() == 8.0
    assert "solver transform" in widget.db_gen_power_spin.toolTip()
    assert "node_prob" in widget.db_gen_quality_exp_spin.toolTip()

    widget.deleteLater()
    viewer.close()


def test_foreground_mask_section_exposes_controls_and_output_row():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.foreground_section.title == "2. Foreground Mask"
    assert widget.fg_source_combo.currentText() == "Sigmoid probability"
    assert [
        widget.fg_source_combo.itemText(i)
        for i in range(widget.fg_source_combo.count())
    ] == ["Sigmoid probability", "Flow DP"]
    assert widget.fg_threshold_spin.minimum() == 0.0
    assert widget.fg_threshold_spin.maximum() == 1.0
    assert widget.fg_threshold_spin.value() == 0.5
    assert widget.fg_threshold_spin.singleStep() == 0.01
    assert widget.fg_gamma_spin.minimum() == 0.05
    assert widget.fg_gamma_spin.maximum() == 5.0
    assert widget.fg_gamma_spin.value() == 1.0
    assert widget.fg_gamma_spin.singleStep() == 0.05
    assert widget.fg_preview_btn.text() == "Preview"
    assert widget.fg_build_btn.text() == "Build"
    assert widget.fg_cancel_btn.text() == "Cancel"
    assert widget.fg_status_lbl in widget.foreground_section.findChildren(QLabel)
    assert widget.fg_progress_bar.isVisible() is False

    output_text = " ".join(
        label.text()
        for label in widget.foreground_files.findChildren(QLabel)
    )
    assert "2_nucleus/foreground_masks.tif" in output_text

    widget.deleteLater()
    viewer.close()


def test_foreground_mask_gamma_stays_enabled_for_flow_dp():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.fg_source_combo.setCurrentText("Flow DP")
    _app.processEvents()

    assert widget.fg_gamma_spin.isEnabled()

    widget.deleteLater()
    viewer.close()


def test_foreground_mask_controls_persist_through_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.fg_source_combo.setCurrentText("Flow DP")
    widget.fg_threshold_spin.setValue(0.42)
    widget.fg_gamma_spin.setValue(1.75)

    state = widget.get_state()
    widget.deleteLater()

    widget = widget_class(viewer)
    widget.set_state(state)

    assert widget.fg_source_combo.currentText() == "Flow DP"
    assert abs(widget.fg_threshold_spin.value() - 0.42) < 0.01
    assert abs(widget.fg_gamma_spin.value() - 1.75) < 0.01

    widget.deleteLater()
    viewer.close()


def test_ultrack_section_still_exposes_seed_prior_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_seed_weight_spin.value() == 0.5
    assert widget.ultrack_seed_space_spin.value() == 25.0
    assert widget.ultrack_seed_time_spin.value() == 2.0
    assert widget.ultrack_seed_window_spin.value() == 5
    assert "validated cells" in widget.ultrack_seed_weight_spin.toolTip()

    widget.deleteLater()
    viewer.close()


def test_validated_seed_prior_controls_follow_resolve_checkbox():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    controls = [
        widget.ultrack_seed_weight_spin,
        widget.ultrack_seed_space_spin,
        widget.ultrack_seed_time_spin,
        widget.ultrack_seed_window_spin,
    ]

    widget.ultrack_route_check.setChecked(False)
    _app.processEvents()
    assert all(not control.isEnabled() for control in controls)

    widget.ultrack_route_check.setChecked(True)
    _app.processEvents()
    assert all(control.isEnabled() for control in controls)

    widget.deleteLater()
    viewer.close()


def test_ultrack_seed_prior_controls_persist_through_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.ultrack_route_check.setChecked(True)
    widget.ultrack_seed_weight_spin.setValue(0.75)
    widget.ultrack_seed_space_spin.setValue(30.0)
    widget.ultrack_seed_time_spin.setValue(3.0)
    widget.ultrack_seed_window_spin.setValue(7)

    state = widget.get_state()
    widget.deleteLater()

    widget = widget_class(viewer)
    widget.set_state(state)

    assert widget.ultrack_route_check.isChecked()
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
    workdir = nucleus_dir / "ultrack_workdir"
    workdir.mkdir(parents=True)
    (workdir / "data.db").touch()
    widget._pos_dir = pos_dir
    widget.db_gen_power_spin.setValue(3.25)
    widget.db_gen_quality_exp_spin.setValue(9.5)
    widget.ultrack_seed_weight_spin.setValue(0.85)
    widget.ultrack_seed_space_spin.setValue(35.0)
    widget.ultrack_seed_time_spin.setValue(4.0)
    widget.ultrack_seed_window_spin.setValue(8)

    widget._on_ultrack_terminal()
    script = _read_launched_script(captured)

    assert "power=3.25" in script
    assert "run_solve(working_dir, cfg, overwrite=True)" in script
    assert "export_tracked_labels(working_dir, cfg, tracked_path)" in script
    assert "ingest_hypotheses_to_db" not in script
    assert "write_seed_prior_node_probs" not in script

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
    cellpose_dir = pos_dir / "1_cellpose"
    nucleus_dir = pos_dir / "2_nucleus"
    cellpose_dir.mkdir(parents=True)
    nucleus_dir.mkdir()
    (nucleus_dir / "contour_maps.tif").touch()
    (nucleus_dir / "foreground_masks.tif").touch()
    (nucleus_dir / "tracked_labels.tif").touch()
    (cellpose_dir / "nucleus_prob_zavg.tif").touch()
    widget._pos_dir = pos_dir
    widget.db_gen_power_spin.setValue(3.75)
    widget.db_gen_quality_exp_spin.setValue(10.5)
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


def test_resolve_done_updates_layer_without_saving_tracked_labels():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    source = (package_root / "nucleus_workflow_widget.py").read_text()
    resolve_done = source.split("        def _on_resolve_done(result: tuple) -> None:", 1)[1].split(
        "        def _on_resolve_progress", 1
    )[0]

    assert "write_tracked_frame" not in resolve_done
    assert "invalidate_track(pos_dir" not in resolve_done
    assert "validate_track(pos_dir" not in resolve_done
    assert "Unsaved" in resolve_done


def test_resolve_terminal_script_does_not_autosave_tracked_labels(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    captured = _install_terminal_capture(monkeypatch)
    monkeypatch.setattr(module, "read_validated_tracks", lambda _pos_dir: {12: {0, 1}})

    pos_dir = tmp_path / "pos00"
    cellpose_dir = pos_dir / "1_cellpose"
    nucleus_dir = pos_dir / "2_nucleus"
    cellpose_dir.mkdir(parents=True)
    nucleus_dir.mkdir()
    (nucleus_dir / "contour_maps.tif").touch()
    (nucleus_dir / "foreground_masks.tif").touch()
    (nucleus_dir / "tracked_labels.tif").touch()
    (cellpose_dir / "nucleus_prob_zavg.tif").touch()
    widget._pos_dir = pos_dir

    widget._on_resolve_terminal()
    script = _read_launched_script(captured)

    assert "write_tracked_frame" not in script
    assert "invalidate_track(pos_dir" not in script
    assert "validate_track(pos_dir" not in script
    assert "tracked_labels_resolve_preview.tif" in script
    assert "tifffile.imwrite(str(preview_path), new_labels" in script
    assert "Preview saved" in script
    assert "not saved" in script

    widget.deleteLater()
    viewer.close()


def test_resolve_terminal_script_includes_seed_prior_and_cellprob_zavg(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    captured = _install_terminal_capture(monkeypatch)
    monkeypatch.setattr(module, "read_validated_tracks", lambda _pos_dir: {7: {0}})

    pos_dir = tmp_path / "pos00"
    cellpose_dir = pos_dir / "1_cellpose"
    nucleus_dir = pos_dir / "2_nucleus"
    cellpose_dir.mkdir(parents=True)
    nucleus_dir.mkdir()
    (nucleus_dir / "contour_maps.tif").touch()
    (nucleus_dir / "foreground_masks.tif").touch()
    (nucleus_dir / "tracked_labels.tif").touch()
    (cellpose_dir / "nucleus_prob_zavg.tif").touch()
    widget._pos_dir = pos_dir

    widget.ultrack_route_check.setChecked(True)
    widget.db_gen_power_spin.setValue(3.5)
    widget.db_gen_quality_exp_spin.setValue(9.0)
    widget.ultrack_seed_weight_spin.setValue(0.75)
    widget.ultrack_seed_space_spin.setValue(30.0)
    widget.ultrack_seed_time_spin.setValue(3.0)
    widget.ultrack_seed_window_spin.setValue(7)

    widget._on_resolve_terminal()
    script = _read_launched_script(captured)

    assert "nucleus_prob_zavg.tif" in script
    assert "power=3.5" in script
    assert "quality_exponent=9.0" in script
    assert "seed_weight=0.75" in script
    assert "seed_sigma_space=30.0" in script
    assert "seed_tau_time=3.0" in script
    assert "seed_max_dt=7" in script
    assert "intensity_image_path=nucleus_prob_zavg_path" in script

    widget.deleteLater()
    viewer.close()


def test_tracking_correction_widget_allows_horizontal_scrolling_when_narrow():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    widget.db_gen_section.expand()
    widget.ultrack_section.expand()
    widget.correction_section.expand()
    _app.processEvents()

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
    for button in (
        widget.preview_contour_btn,
        widget.build_btn,
        widget.contour_terminal_btn,
        widget.cancel_build_btn,
    ):
        assert button.minimumWidth() == 54
        assert button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding

    assert params_scroll.horizontalScrollBar().maximum() > 0

    host.deleteLater()
    viewer.close()


def test_db_generation_spinboxes_expand_equally_in_the_top_grid():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.db_gen_section.expand()
    _app.processEvents()

    host = QWidget()
    host_layout = QVBoxLayout(host)
    host_layout.setContentsMargins(0, 0, 0, 0)
    host_layout.addWidget(widget)
    host.resize(220, 280)
    host.show()
    _app.processEvents()

    for spin in (
        widget.db_gen_min_area_spin,
        widget.db_gen_max_area_spin,
        widget.db_gen_max_dist_spin,
        widget.db_gen_max_neighbors_spin,
    ):
        assert spin.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed

    assert widget.db_gen_min_area_spin.y() == widget.db_gen_max_area_spin.y()
    assert widget.db_gen_min_area_spin.x() < widget.db_gen_max_area_spin.x()
    assert widget.db_gen_max_dist_spin.y() == widget.db_gen_max_neighbors_spin.y()
    assert widget.run_db_gen_btn.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert widget.db_gen_terminal_btn.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding

    host.deleteLater()
    viewer.close()


def test_tracking_correction_restores_two_column_button_and_parameter_layouts():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.db_gen_section.expand()
    widget.ultrack_section.expand()
    widget.correction_section.expand()
    widget.show()
    _app.processEvents()

    # DB gen parameters should present as two side-by-side columns
    assert widget.db_gen_min_area_spin.y() == widget.db_gen_max_area_spin.y()
    assert widget.db_gen_min_area_spin.x() < widget.db_gen_max_area_spin.x()

    # Paired correction actions should also sit side-by-side
    assert widget.extend_back_btn.y() == widget.extend_fwd_btn.y()
    assert widget.extend_back_btn.x() < widget.extend_fwd_btn.x()
    assert widget.retrack_back_btn.y() == widget.retrack_fwd_btn.y()
    assert widget.save_tracked_btn.y() == widget.load_tracked_btn.y()

    widget.deleteLater()
    viewer.close()


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


def test_tracking_config_has_segmentation_fields():
    sys.modules.pop("cellflow.tracking_ultrack.config", None)
    sys.modules.pop("cellflow.tracking_ultrack", None)
    from cellflow.tracking_ultrack.config import TrackingConfig
    cfg = TrackingConfig()
    assert cfg.seg_min_area == 300
    assert cfg.seg_max_area == 100_000
    assert cfg.seg_foreground_threshold == 0.5
    assert cfg.seg_min_frontier == 0.0
    assert cfg.seg_ws_hierarchy == "area"
    assert cfg.seg_n_workers == 1


def test_build_ultrack_config_applies_segmentation_fields(tmp_path):
    for mod in ["cellflow.tracking_ultrack.config", "cellflow.tracking_ultrack.ingest", "cellflow.tracking_ultrack"]:
        sys.modules.pop(mod, None)
    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.ingest import _build_ultrack_config
    cfg = TrackingConfig(
        seg_min_area=500,
        seg_max_area=50_000,
        seg_foreground_threshold=0.3,
        seg_min_frontier=0.05,
        seg_ws_hierarchy="dynamics",
        seg_n_workers=2,
    )
    ultrack_cfg = _build_ultrack_config(cfg, tmp_path)
    sc = ultrack_cfg.segmentation_config
    assert sc.min_area == 500
    assert sc.max_area == 50_000
    assert abs(sc.threshold - 0.3) < 1e-6
    assert abs(sc.min_frontier - 0.05) < 1e-6
    assert sc.n_workers == 2


def test_resolve_with_canonical_segment_exists():
    for mod in ["cellflow.tracking_ultrack.reseed", "cellflow.tracking_ultrack"]:
        sys.modules.pop(mod, None)
    from cellflow.tracking_ultrack.reseed import resolve_with_canonical_segment
    import inspect
    sig = inspect.signature(resolve_with_canonical_segment)
    params = set(sig.parameters)
    assert "contour_maps_path" in params
    assert "foreground_masks_path" in params
    assert "validated_tracks" in params
    assert "tracked_labels" in params
    assert "cfg" in params
    assert "intensity_image_path" in params
    # must NOT require hypotheses_path
    assert "hypotheses_path" not in params


def test_extend_track_from_db_missing_db_raises(tmp_path):
    sys.modules.pop("cellflow.tracking_ultrack.extend", None)
    sys.modules.pop("cellflow.tracking_ultrack", None)
    from cellflow.tracking_ultrack.extend import extend_track_from_db
    import numpy as np
    tracked = np.zeros((3, 10, 10), dtype=np.uint32)
    tracked[0, 2:5, 2:5] = 7
    result = extend_track_from_db(
        source_id=7,
        source_frame=0,
        direction="forward",
        tracked_labels=tracked,
        db_path=tmp_path / "data.db",
        d_max=40.0,
    )
    assert result is None  # missing DB returns None; widget shows clear error


# ── Task 5: New layout tests ──────────────────────────────────────────────────

def test_nucleus_workflow_has_six_canonical_top_level_sections():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.contour_section.title == "1. Contour Maps"
    assert widget.foreground_section.title == "2. Foreground Mask"
    assert widget.db_gen_section.title == "3. Ultrack Database Generation"
    assert widget.ultrack_db_browser_section.title == "4. Ultrack Database Browser"
    assert widget.ultrack_section.title == "5. Ultrack Tracking"
    assert widget.correction_section.title == "6. Correction"

    assert not hasattr(widget, "tracking_correction_section")

    widget.deleteLater()
    viewer.close()


def test_deprecated_sections_are_hidden():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert hasattr(widget, "gen_section")
    assert hasattr(widget, "db_section")
    assert not widget.gen_section.isVisible()
    assert not widget.db_section.isVisible()

    widget.deleteLater()
    viewer.close()


def test_canonical_sections_expose_required_elements():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    # Section 1: Contour Maps
    assert hasattr(widget, "build_btn")
    assert hasattr(widget, "contour_terminal_btn")
    assert hasattr(widget, "contour_status_lbl")
    assert hasattr(widget, "build_progress_bar")

    # Section 2: Foreground Mask
    assert hasattr(widget, "fg_source_combo")
    assert hasattr(widget, "fg_threshold_spin")
    assert hasattr(widget, "fg_gamma_spin")
    assert hasattr(widget, "fg_preview_btn")
    assert hasattr(widget, "fg_build_btn")
    assert hasattr(widget, "fg_cancel_btn")
    assert hasattr(widget, "fg_status_lbl")
    assert hasattr(widget, "fg_progress_bar")

    # Section 3: Ultrack Database Generation
    assert hasattr(widget, "run_db_gen_btn")
    assert hasattr(widget, "db_gen_terminal_btn")
    assert hasattr(widget, "db_gen_status_lbl")
    assert hasattr(widget, "db_gen_progress_bar")

    # Section 4: Ultrack Database Browser
    assert hasattr(widget, "ultrack_db_info_lbl")
    assert hasattr(widget, "ultrack_db_active_btn")
    assert hasattr(widget, "ultrack_db_refresh_btn")
    assert hasattr(widget, "ultrack_db_mode_combo")
    assert hasattr(widget, "ultrack_db_hierarchy_slider")
    assert hasattr(widget, "ultrack_db_height_lbl")
    assert hasattr(widget, "ultrack_db_section_status_lbl")

    # Section 5: Ultrack Tracking
    assert hasattr(widget, "run_ultrack_btn")
    assert hasattr(widget, "ultrack_terminal_btn")
    assert hasattr(widget, "ultrack_status_lbl")
    assert hasattr(widget, "ultrack_progress_bar")

    # Section 6: Correction (no Run button per spec)
    assert hasattr(widget, "correction_status_lbl")

    widget.deleteLater()
    viewer.close()


def test_ultrack_section_is_top_level_and_has_route_selector():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_section.title == "5. Ultrack Tracking"
    assert widget.ultrack_route_check.text() == "Resolve from validated"
    assert widget.ultrack_route_check in widget.ultrack_section.findChildren(
        type(widget.ultrack_route_check)
    )
    assert widget.ultrack_status_lbl.text() == ""
    assert widget.ultrack_progress_bar.isVisible() is False

    widget.deleteLater()
    viewer.close()


def test_correction_section_is_top_level():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.correction_section.title == "6. Correction"
    correction_button_texts = {
        button.text()
        for button in widget.correction_section.findChildren(QPushButton)
    }
    assert "Save Tracked Labels" in correction_button_texts
    assert "Load Tracked Labels" in correction_button_texts
    assert "◀ Extend (A)" in correction_button_texts
    assert "Extend (D) ▶" in correction_button_texts
    assert "◀ Retrack (Q)" in correction_button_texts
    assert "Retrack (E) ▶" in correction_button_texts

    widget.deleteLater()
    viewer.close()


def test_correction_shortcuts_are_still_installed():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    shortcut_keys = {
        shortcut.key().toString(QKeySequence.SequenceFormat.PortableText)
        for shortcut in widget.findChildren(QShortcut)
    }
    assert {"A", "D", "Q", "E"} <= shortcut_keys

    widget.deleteLater()
    viewer.close()


# ── Task 6: DB generation run + terminal ─────────────────────────────────────

def _install_sync_thread_worker(monkeypatch, module):
    def _sync_thread_worker(*, connect):
        def _decorator(func):
            def _runner():
                yielded = connect.get("yielded")
                returned = connect.get("returned")
                errored = connect.get("errored")
                try:
                    result = func()
                    if hasattr(result, "__next__"):
                        while True:
                            try:
                                item = next(result)
                            except StopIteration as stop:
                                if returned is not None:
                                    returned(stop.value)
                                break
                            else:
                                if yielded is not None:
                                    yielded(item)
                    elif returned is not None:
                        returned(result)
                except Exception as exc:
                    if errored is not None:
                        errored(exc)
                    else:
                        raise

            return _runner

        return _decorator

    monkeypatch.setattr(module, "thread_worker", _sync_thread_worker)


def test_db_gen_section_calls_ultrack_segment_on_run(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    calls = []

    def fake_segment(foreground, contours, ultrack_cfg, **kwargs):
        calls.append((foreground.shape, contours.shape, kwargs))

    monkeypatch.setattr(module, "_ultrack_segment", fake_segment, raising=False)
    monkeypatch.setattr(module, "write_seed_prior_node_probs", lambda *a, **kw: calls.append(("score", a)))
    monkeypatch.setattr(module, "run_linking", lambda *a, **kw: iter([(1, 1, "linked")]))

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    dummy = np.zeros((2, 1, 4, 4), dtype=np.float32)
    import tifffile

    tifffile.imwrite(str(pos_dir / "2_nucleus" / "contour_maps.tif"), dummy)
    tifffile.imwrite(str(pos_dir / "2_nucleus" / "foreground_masks.tif"), dummy)
    tifffile.imwrite(str(pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif"), dummy)
    widget._pos_dir = pos_dir

    widget._on_run_db_generation()

    assert calls[0][0] == (2, 4, 4)
    assert calls[0][1] == (2, 4, 4)
    assert calls[0][2]["overwrite"] is True
    assert calls[0][2]["max_segments_per_time"] == 1_000_000
    assert calls[1][0] == "score"
    assert widget.run_db_gen_btn.isEnabled()
    assert widget.db_gen_terminal_btn.isEnabled()
    assert "complete" in widget.db_gen_status_lbl.text().lower()

    widget.deleteLater()
    viewer.close()


def test_db_gen_section_terminal_script_includes_canonical_segment(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    captured = _install_terminal_capture(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "2_nucleus" / "contour_maps.tif").touch()
    (pos_dir / "2_nucleus" / "foreground_masks.tif").touch()
    (pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif").touch()
    widget._pos_dir = pos_dir

    widget._on_db_gen_terminal()
    script = _read_launched_script(captured)

    assert "ultrack.core.segmentation.processing" in script
    assert "foreground_masks" in script
    assert "contour_maps" in script
    assert "nucleus_prob_zavg" in script
    assert "write_seed_prior_node_probs" in script
    assert "run_linking" in script
    assert "if __name__ == '__main__':" in script

    widget.deleteLater()
    viewer.close()


def test_db_gen_section_fails_clearly_if_foreground_masks_missing(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "2_nucleus" / "contour_maps.tif").touch()
    widget._pos_dir = pos_dir

    widget._on_run_db_generation()

    text = widget.db_gen_status_lbl.text().lower()
    assert "foreground_masks" in text or "missing" in text
    assert "foreground mask" in text
    assert "run" in text

    widget.deleteLater()
    viewer.close()


# ── Task 7: Ultrack DB browser ────────────────────────────────────────────────

def test_ultrack_db_browser_shows_missing_db_status(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget._pos_dir = pos_dir
    widget._ultrack_db_browser_active = True

    widget._refresh_ultrack_db_browser()

    text = widget.ultrack_db_section_status_lbl.text().lower()
    assert "data.db" in text or "missing" in text or "not found" in text

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_exposes_two_modes():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    modes = [
        widget.ultrack_db_mode_combo.itemText(i)
        for i in range(widget.ultrack_db_mode_combo.count())
    ]

    assert modes == ["Summary only", "Hierarchy cut"]
    assert widget.ultrack_db_hierarchy_slider.minimum() == 0
    assert widget.ultrack_db_hierarchy_slider.maximum() == 100
    assert widget.ultrack_db_hierarchy_slider.value() == 50

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_exposes_connected_focus_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_db_connected_focus_check.text() == "Connected focus"
    assert widget.ultrack_db_edge_alpha_check.text() == "Edge weight transparency"
    assert widget.ultrack_db_prob_alpha_check.text() == "Node prob transparency"
    assert not widget.ultrack_db_connected_focus_check.isEnabled()
    assert not widget.ultrack_db_edge_alpha_check.isEnabled()

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_normalizes_preview_metadata():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    labels = np.array([[1, 0]], dtype=np.uint32)
    normalized = widget._normalize_ultrack_db_preview(
        (labels, "status", {1: 0.5}, {1: 101}, {101: 1})
    )

    assert normalized[0] is labels
    assert normalized[1] == "status"
    assert normalized[2] == {1: 0.5}
    assert normalized[3] == {1: 101}
    assert normalized[4] == {101: 1}

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_builds_display_label_node_id_metadata():
    widget_class = _load_widget_class()

    prob_dict, label_to_node_id, node_id_to_label = (
        widget_class._ultrack_db_node_preview_metadata(
            [
                types.SimpleNamespace(id=101, node_prob=0.25),
                types.SimpleNamespace(id=202, node_prob=0.75),
            ]
        )
    )

    assert prob_dict == {1: 0.25, 2: 0.75}
    assert label_to_node_id == {1: 101, 2: 202}
    assert node_id_to_label == {101: 1, 202: 2}


def test_ultrack_db_browser_click_selects_node_id_from_display_label(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_mode_combo.setCurrentText("Hierarchy cut")
    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    monkeypatch.setattr(widget, "_query_distinct_heights", lambda *a: (0.5,))
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *a: (
            np.array([[0, 2], [0, 0]], dtype=np.uint32),
            "rendered",
            {2: 0.8},
            {2: 222},
            {222: 2},
        ),
    )

    widget._refresh_ultrack_db_browser()
    widget._select_ultrack_db_preview_label(2, frame=4)

    assert widget._ultrack_db_selected_node_id == 222
    assert widget._ultrack_db_selected_frame == 4
    assert "Selected node 222" in widget.ultrack_db_section_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_selection_highlight_uses_cyan_contour():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    labels = np.zeros((6, 6), dtype=np.uint32)
    labels[2:5, 1:4] = 7

    widget._update_ultrack_db_highlight(labels, 7)

    layer = viewer.layers["Ultrack DB Selection"]
    assert layer.visible
    assert len(layer.data) == 1
    assert layer.name == "Ultrack DB Selection"

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_connected_focus_filters_by_viewer_frame(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_mode_combo.setCurrentText("Hierarchy cut")
    widget.ultrack_db_connected_focus_check.setChecked(True)
    widget._ultrack_db_selected_node_id = 222
    widget._ultrack_db_selected_frame = 4
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    monkeypatch.setattr(widget, "_query_distinct_heights", lambda *a: (0.5,))
    monkeypatch.setattr(
        widget,
        "_query_ultrack_db_connected_nodes",
        lambda *a: ({111: 0.25}, {333: 0.9}),
    )

    labels_by_frame = {
        3: (
            np.array([[1, 0], [2, 0]], dtype=np.uint32),
            {1: 111, 2: 999},
            {111: 1, 999: 2},
        ),
        4: (
            np.array([[1, 0], [2, 0]], dtype=np.uint32),
            {1: 222, 2: 999},
            {222: 1, 999: 2},
        ),
        5: (
            np.array([[1, 0], [2, 0]], dtype=np.uint32),
            {1: 333, 2: 999},
            {333: 1, 999: 2},
        ),
    }
    current = {"t": 4}
    monkeypatch.setattr(widget, "_current_t", lambda: current["t"])

    def _render(*args):
        labels, label_to_node_id, node_id_to_label = labels_by_frame[current["t"]]
        return labels, "rendered", {1: 1.0, 2: 1.0}, label_to_node_id, node_id_to_label

    monkeypatch.setattr(widget, "_render_hierarchy_cut", _render)

    widget._refresh_ultrack_db_browser()
    assert set(np.unique(viewer.layers["Ultrack DB Preview"].data)) == {0, 1}

    current["t"] = 3
    widget._refresh_ultrack_db_browser()
    assert set(np.unique(viewer.layers["Ultrack DB Preview"].data)) == {0, 1}

    current["t"] = 5
    widget._refresh_ultrack_db_browser()
    assert set(np.unique(viewer.layers["Ultrack DB Preview"].data)) == {0, 1}

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_edge_and_node_prob_transparency_multiply(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_mode_combo.setCurrentText("Hierarchy cut")
    widget.ultrack_db_connected_focus_check.setChecked(True)
    widget.ultrack_db_edge_alpha_check.setChecked(True)
    widget.ultrack_db_prob_alpha_check.setChecked(True)
    widget._ultrack_db_selected_node_id = 222
    widget._ultrack_db_selected_frame = 4
    monkeypatch.setattr(widget, "_current_t", lambda: 5)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    monkeypatch.setattr(widget, "_query_distinct_heights", lambda *a: (0.5,))
    monkeypatch.setattr(
        widget,
        "_query_ultrack_db_connected_nodes",
        lambda *a: ({}, {333: 0.5, 444: 1.0}),
    )
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *a: (
            np.array([[1, 0], [2, 0]], dtype=np.uint32),
            "rendered",
            {1: 0.2, 2: 0.8},
            {1: 333, 2: 444},
            {333: 1, 444: 2},
        ),
    )

    widget._refresh_ultrack_db_browser()

    layer = viewer.layers["Ultrack DB Preview"]
    assert layer.data.shape == (2, 2, 4)
    assert layer.data[0, 0, 3] == pytest.approx(0.075)
    assert layer.data[1, 0, 3] == pytest.approx(1.0)
    assert layer.data[0, 1, 3] == 0

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_refresh_reanchors_selection_contour(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_mode_combo.setCurrentText("Hierarchy cut")
    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    monkeypatch.setattr(widget, "_query_distinct_heights", lambda *a: (0.5,))
    renders = [
        (
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint32),
            "rendered",
            {1: 0.8},
            {1: 222},
            {222: 1},
        ),
        (
            np.array([[0, 0, 0], [0, 2, 2], [0, 2, 2]], dtype=np.uint32),
            "rendered",
            {2: 0.8},
            {2: 222},
            {222: 2},
        ),
    ]
    monkeypatch.setattr(widget, "_render_hierarchy_cut", lambda *a: renders.pop(0))

    widget._refresh_ultrack_db_browser()
    widget._select_ultrack_db_preview_label(1, frame=4)
    first_contour = np.asarray(viewer.layers["Ultrack DB Selection"].data[0]).copy()
    widget._ultrack_db_preview_cache.clear()
    widget._refresh_ultrack_db_browser()
    second_contour = np.asarray(viewer.layers["Ultrack DB Selection"].data[0])

    assert widget._ultrack_db_node_id_to_label == {222: 2}
    assert second_contour[:, 0].max() > first_contour[:, 0].max()
    assert second_contour[:, 1].max() > first_contour[:, 1].max()

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_connected_focus_reports_hidden_selected_node(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_mode_combo.setCurrentText("Hierarchy cut")
    widget.ultrack_db_connected_focus_check.setChecked(True)
    widget._ultrack_db_selected_node_id = 222
    widget._ultrack_db_selected_frame = 4
    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    monkeypatch.setattr(widget, "_query_distinct_heights", lambda *a: (0.5,))
    monkeypatch.setattr(widget, "_query_ultrack_db_connected_nodes", lambda *a: ({}, {}))
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *a: (
            np.array([[1, 0]], dtype=np.uint32),
            "rendered",
            {1: 0.8},
            {1: 999},
            {999: 1},
        ),
    )

    widget._refresh_ultrack_db_browser()

    assert "hidden" in widget.ultrack_db_section_status_lbl.text().lower()
    assert "222" in widget.ultrack_db_section_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_hierarchy_cut_caches_by_frame_and_slider(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget.ultrack_db_mode_combo.setCurrentText("Hierarchy cut")
    widget.ultrack_db_hierarchy_slider.setValue(1)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True  # skip middle-frame jump; no real DB to query

    calls = []

    def _fake_summary(path, frame):
        return "3 nodes | 2 links | frame 0: 1 nodes"

    def _fake_render(path, frame, slider_int):
        calls.append((path, frame, slider_int))
        return np.zeros((5, 5), dtype=np.uint32), "rendered hierarchy cut"

    monkeypatch.setattr(widget, "_ultrack_db_summary_text", _fake_summary)
    monkeypatch.setattr(widget, "_query_distinct_heights", lambda path, mtime_ns: (0.25, 0.75))
    monkeypatch.setattr(widget, "_render_hierarchy_cut", _fake_render)

    widget._refresh_ultrack_db_browser()
    widget._refresh_ultrack_db_browser()

    assert len(calls) == 1
    assert calls[0] == (db_path, 0, 0.75)
    assert widget.ultrack_db_info_lbl.text() == "3 nodes | 2 links | frame 0: 1 nodes"
    assert widget.ultrack_db_section_status_lbl.text() == "rendered hierarchy cut"
    assert "Ultrack DB Preview" in viewer.layers

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_probability_transparency_renders_rgba_preview(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget.ultrack_db_mode_combo.setCurrentText("Hierarchy cut")
    widget.ultrack_db_hierarchy_slider.setValue(0)
    widget.ultrack_db_prob_alpha_check.setChecked(True)
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda path, frame: "summary")
    monkeypatch.setattr(widget, "_query_distinct_heights", lambda path, mtime_ns: (0.5,))

    labels = np.array(
        [
            [1, 0],
            [0, 2],
        ],
        dtype=np.uint32,
    )
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *args: (labels, "rendered hierarchy cut", {1: 0.2, 2: 0.8}),
    )

    widget._refresh_ultrack_db_browser()

    layer = viewer.layers["Ultrack DB Preview"]
    assert layer.data.shape == (2, 2, 4)
    assert layer.data[0, 0, 3] < layer.data[1, 1, 3]
    assert layer.data[0, 1, 3] == 0

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_summary_mode_does_not_render(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True  # skip middle-frame jump; no real DB to query

    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda path, frame: "summary")
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *args, **kwargs: pytest.fail("summary mode must not render"),
    )

    widget._refresh_ultrack_db_browser()

    assert widget.ultrack_db_info_lbl.text() == "summary"
    assert widget.ultrack_db_section_status_lbl.text() == "Summary refreshed."
    assert "Ultrack DB Preview" not in viewer.layers

    widget.deleteLater()
    viewer.close()


# ── Task 8: Tracking solves existing DB; extend uses DB ──────────────────────

def test_ultrack_tracking_solve_fails_clearly_if_db_missing(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    widget._pos_dir = pos_dir

    widget._on_run_ultrack()

    text = widget.ultrack_status_lbl.text().lower()
    assert "data.db" in text or "missing" in text or "database" in text

    widget.deleteLater()
    viewer.close()


def test_extend_fails_clearly_if_db_missing(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget._pos_dir = pos_dir
    viewer.add_labels(np.zeros((2, 8, 8), dtype=np.uint32), name="Tracked: Nucleus")

    widget._on_extend(direction="forward")

    text = widget.correction_status_lbl.text().lower()
    assert "data.db" in text or "database" in text

    widget.deleteLater()
    viewer.close()


# ── Task 9: Resolve from validated uses canonical segmentation ───────────────

def test_resolve_from_validated_fails_clearly_if_foreground_masks_missing(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    monkeypatch.setattr(module, "read_validated_tracks", lambda _pos_dir: {1: {0, 1}})

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "2_nucleus" / "contour_maps.tif").touch()
    (pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif").touch()
    widget._pos_dir = pos_dir
    viewer.add_labels(np.zeros((2, 8, 8), dtype=np.uint32), name="Tracked: Nucleus")

    widget.ultrack_route_check.setChecked(True)
    widget._on_run_tracking_route()

    text = widget.ultrack_status_lbl.text().lower()
    assert "foreground_masks" in text or "missing" in text

    widget.deleteLater()
    viewer.close()


def test_resolve_terminal_script_uses_canonical_segment_not_hypotheses_h5(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    captured = _install_terminal_capture(monkeypatch)
    monkeypatch.setattr(module, "read_validated_tracks", lambda _pos_dir: {12: {0, 1}})

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "2_nucleus" / "contour_maps.tif").touch()
    (pos_dir / "2_nucleus" / "foreground_masks.tif").touch()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif").touch()
    widget._pos_dir = pos_dir

    widget._on_resolve_terminal()
    script = _read_launched_script(captured)

    assert "foreground_masks" in script
    assert "contour_maps" in script
    assert "nucleus_prob_zavg" in script
    assert "resolve_with_canonical_segment" in script
    assert "hypotheses.h5" not in script
    assert "ingest_hypotheses_to_db" not in script

    widget.deleteLater()
    viewer.close()


# ── Task 10: DB generation state persistence ─────────────────────────────────

def test_db_gen_controls_persist_through_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.db_gen_min_area_spin.setValue(500)
    widget.db_gen_max_area_spin.setValue(80_000)
    widget.db_gen_fg_thr_spin.setValue(0.4)
    widget.db_gen_min_frontier_spin.setValue(0.05)
    widget.db_gen_ws_hierarchy_combo.setCurrentText("dynamics")
    widget.db_gen_max_dist_spin.setValue(20.0)
    widget.db_gen_max_neighbors_spin.setValue(8)
    widget.db_gen_linking_mode_combo.setCurrentText("iou")
    widget.db_gen_iou_weight_spin.setValue(0.8)
    widget.db_gen_quality_exp_spin.setValue(6.0)
    widget.db_gen_power_spin.setValue(3.0)
    widget.db_gen_n_workers_spin.setValue(4)

    state = widget.get_state()
    widget.deleteLater()

    widget = widget_class(viewer)
    widget.set_state(state)

    assert widget.db_gen_min_area_spin.value() == 500
    assert widget.db_gen_max_area_spin.value() == 80_000
    assert abs(widget.db_gen_fg_thr_spin.value() - 0.4) < 0.01
    assert abs(widget.db_gen_min_frontier_spin.value() - 0.05) < 0.01
    assert widget.db_gen_ws_hierarchy_combo.currentText() == "dynamics"
    assert widget.db_gen_max_dist_spin.value() == 20.0
    assert widget.db_gen_max_neighbors_spin.value() == 8
    assert widget.db_gen_linking_mode_combo.currentText() == "iou"
    assert abs(widget.db_gen_iou_weight_spin.value() - 0.8) < 0.01
    assert abs(widget.db_gen_quality_exp_spin.value() - 6.0) < 0.01
    assert abs(widget.db_gen_power_spin.value() - 3.0) < 0.01
    assert widget.db_gen_n_workers_spin.value() == 4

    widget.deleteLater()
    viewer.close()


# ── DB Browser hierarchy slider: discrete height-index tests ────────────


def _make_ultrack_db_with_heights(db_path: Path, heights: list[float]) -> None:
    import pickle
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import Base
    from ultrack.core.database import NodeDB
    from ultrack.core.segmentation.node import Node
    from ultrack.utils.constants import NO_PARENT

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for i, height in enumerate(heights, start=1):
            mask = np.ones((1, 2, 2), dtype=bool)
            bbox = np.array([0, i, i, 1, i + 2, i + 2], dtype=np.int64)
            node_obj = Node.from_mask(time=0, mask=mask, bbox=bbox, node_id=i)
            session.add(
                NodeDB(
                    id=i,
                    t=0,
                    t_node_id=i,
                    t_hier_id=1,
                    z=0,
                    y=i + 1,
                    x=i + 1,
                    area=4,
                    height=float(height),
                    hier_parent_id=NO_PARENT,
                    pickle=pickle.dumps(node_obj),
                )
            )
        session.commit()
        assert session.query(NodeDB.height).distinct().count() == len(set(heights))
    engine.dispose()


def test_ultrack_db_hierarchy_slider_uses_distinct_db_heights(tmp_path, monkeypatch):
    _make_ultrack_db_with_heights(
        tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db",
        [0.1, 0.4, 0.9],
    )
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    calls = []

    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_mode_combo.setCurrentText("Hierarchy cut")
    widget.ultrack_db_hierarchy_slider.setValue(1)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda _path, _frame: "ok")

    def _render(db_path, frame, db_height):
        calls.append((db_path, frame, db_height))
        return np.zeros((5, 5), dtype=np.uint32), f"rendered h={db_height:.2f}"

    monkeypatch.setattr(widget, "_render_hierarchy_cut", _render)

    widget._refresh_ultrack_db_browser()

    assert widget.ultrack_db_hierarchy_slider.minimum() == 0
    assert widget.ultrack_db_hierarchy_slider.maximum() == 2
    assert widget.ultrack_db_hierarchy_slider.value() == 1
    assert "1" in widget.ultrack_db_height_lbl.text()
    assert "0.40" in widget.ultrack_db_height_lbl.text()
    assert calls == [
        (
            tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db",
            0,
            0.4,
        )
    ]

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_hierarchy_slider_clamps_when_db_heights_shrink(tmp_path, monkeypatch):
    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    _make_ultrack_db_with_heights(db_path, [0.1, 0.3, 0.5, 0.7])
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    calls = []

    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_mode_combo.setCurrentText("Hierarchy cut")
    widget.ultrack_db_hierarchy_slider.setValue(3)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda _path, _frame: "ok")

    def _render(_db_path, _frame, db_height):
        calls.append(db_height)
        return np.zeros((5, 5), dtype=np.uint32), "ok"

    monkeypatch.setattr(widget, "_render_hierarchy_cut", _render)
    widget._refresh_ultrack_db_browser()

    db_path.unlink()
    _make_ultrack_db_with_heights(db_path, [0.2, 0.6])
    widget._ultrack_db_preview_cache.clear()
    widget._refresh_ultrack_db_browser()

    assert widget.ultrack_db_hierarchy_slider.maximum() == 1
    assert widget.ultrack_db_hierarchy_slider.value() == 1
    assert "1" in widget.ultrack_db_height_lbl.text()
    assert "0.60" in widget.ultrack_db_height_lbl.text()
    assert calls == [0.7, 0.6]

    widget.deleteLater()
    viewer.close()
