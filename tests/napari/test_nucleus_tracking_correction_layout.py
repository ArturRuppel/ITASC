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
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QPushButton,
    QCheckBox,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


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
            self.quality_weight = 1.0
            self.quality_exponent = 8.0
            self.circularity_weight = 0.25
            self.appear_weight = -0.001
            self.disappear_weight = -0.001
            self.division_weight = -0.001
            self.solution_gap = 0.001
            self.time_limit = 36000
            self.window_size = 0
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
        "cellflow.tracking_ultrack.db_build": {"build_ultrack_database": lambda *args, **kwargs: None},
        "cellflow.tracking_ultrack.extend": {
            "extend_track": lambda *args, **kwargs: None,
            "extend_track_from_db": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.reseed": {
            "resolve_with_validation": lambda *args, **kwargs: None,
            "resolve_with_canonical_segment": lambda *args, **kwargs: (None, {}),
        },
        "cellflow.tracking_ultrack.seed_prior": {
            "boost_validated_edges": lambda *args, **kwargs: None,
            "write_seed_prior_node_probs": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.solve": {
            "database_has_annotations": lambda *args, **kwargs: False,
            "run_solve": lambda *args, **kwargs: iter(()),
        },
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
            self.state_received = None
            self.selection_callback = None
            self.match_requests = []

        def refresh(self, pos_dir):
            self.refreshed_pos_dir = pos_dir

        def get_state(self):
            return {"widget": type(self).__name__}

        def set_state(self, state):
            self.state_received = state

        def set_selection_callback(self, fn):
            self.selection_callback = fn

        def select_matching_cell_label(self, t, source_label):
            self.match_requests.append(("cell", t, source_label))

        def select_matching_nucleus_label(self, t, source_label):
            self.match_requests.append(("nucleus", t, source_label))

    class _StubCellposeWidget(_StubWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hpc_cellpose_widget = _StubWidget()

        def refresh(self, pos_dir):
            super().refresh(pos_dir)
            self.hpc_cellpose_widget.refresh(pos_dir)

    stub_modules = {
        "cellflow.napari.analysis_widget": {"AnalysisWidget": _StubWidget},
        "cellflow.napari.cell_workflow_widget": {"CellWorkflowWidget": _StubWidget},
        "cellflow.napari.cellpose_widget": {"CellposeWidget": _StubCellposeWidget},
        "cellflow.napari.correction_widget": {"CorrectionWidget": _StubWidget},
        "cellflow.napari.data_panel_widget": {"ProjectStatusPanel": _StubWidget},
        "cellflow.napari.data_prep_widget": {"DataPrepWidget": _StubWidget},
        "cellflow.napari.hpc_cellpose_widget": {"HpcCellposeWidget": _StubWidget},
        "cellflow.napari.meta_widget": {"MetaSourceBrowserWidget": _StubWidget},
        "cellflow.napari.nucleus_workflow_widget": {"NucleusWorkflowWidget": _StubWidget},
        "cellflow.napari.nls_classification_widget": {"NLSClassificationWidget": _StubWidget},
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
    sys.modules.pop("cellflow.napari.main_widget", None)
    module = importlib.import_module("cellflow.napari.main_widget")
    for module_name in (
        "cellflow.napari.analysis_widget",
        "cellflow.napari.cell_workflow_widget",
        "cellflow.napari.cellpose_widget",
        "cellflow.napari.correction_widget",
        "cellflow.napari.data_panel_widget",
        "cellflow.napari.data_prep_widget",
        "cellflow.napari.hpc_cellpose_widget",
        "cellflow.napari.meta_widget",
        "cellflow.napari.nucleus_workflow_widget",
        "cellflow.napari.nls_classification_widget",
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


def test_main_widget_sections_are_collapsed_by_default():
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    sections = (
        widget.data_section,
        widget.prep_section,
        widget.cellpose_section,
        widget.nucleus_section,
        widget.cell_section,
        widget.analysis_section,
        widget.nls_classification_section,
        widget.meta_section,
    )
    assert all(section.is_expanded is False for section in sections)

    widget.deleteLater()
    viewer.close()


def test_main_widget_synchronizes_cell_and_nucleus_selection_callbacks():
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    widget.nucleus_workflow_widget.selection_callback(2, 7)
    widget.cell_workflow_widget.selection_callback(3, 11)

    assert widget.cell_workflow_widget.match_requests == [("cell", 2, 7)]
    assert widget.nucleus_workflow_widget.match_requests == [("nucleus", 3, 11)]

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


def test_main_widget_embeds_hpc_cellpose_inside_cellpose_stage():
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "hpc_cellpose_section")
    assert widget.hpc_cellpose_widget is widget._cellpose_widget.hpc_cellpose_widget
    assert widget.scroll_layout.indexOf(widget.cellpose_section) < widget.scroll_layout.indexOf(
        widget.nucleus_section
    )

    widget.deleteLater()
    viewer.close()


def test_main_widget_includes_nls_classification_section_after_analysis():
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    assert widget.nls_classification_section.title == "5b. NLS Classification"
    assert widget.scroll_layout.indexOf(widget.analysis_section) < widget.scroll_layout.indexOf(
        widget.nls_classification_section
    )

    widget.deleteLater()
    viewer.close()


def test_main_widget_includes_meta_source_browser_after_nls_classification(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    project_root = tmp_path / "study"
    project_root.mkdir()
    widget.path_label.setText(str(project_root))
    widget.pos_spin.setValue(3)
    widget._refresh_all()

    assert widget.meta_section.title == "6. Meta Analyzer"
    assert widget.meta_source_browser.refreshed_pos_dir == project_root
    assert widget.scroll_layout.indexOf(widget.nls_classification_section) < widget.scroll_layout.indexOf(
        widget.meta_section
    )

    widget.deleteLater()
    viewer.close()


def test_nucleus_workflow_uses_stage_local_file_widgets():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "input_files")

    assert hasattr(widget, "contour_input_files")
    assert hasattr(widget, "contour_output_files")
    assert hasattr(widget, "contour_status_lbl")
    assert hasattr(widget, "build_progress_bar")

    assert hasattr(widget, "db_gen_input_files")
    assert hasattr(widget, "db_gen_output_files")
    assert hasattr(widget, "db_gen_status_lbl")
    assert hasattr(widget, "db_gen_progress_bar")

    assert hasattr(widget, "ultrack_input_files")
    assert hasattr(widget, "ultrack_output_files")
    assert hasattr(widget, "ultrack_status_lbl")
    assert hasattr(widget, "ultrack_progress_bar")

    assert widget.contour_input_files in widget.contour_section.findChildren(type(widget.contour_input_files))
    assert widget.contour_output_files in widget.contour_section.findChildren(type(widget.contour_output_files))
    assert widget.db_gen_input_files in widget.db_gen_section.findChildren(type(widget.db_gen_input_files))
    assert widget.db_gen_output_files in widget.db_gen_section.findChildren(type(widget.db_gen_output_files))
    assert widget.ultrack_input_files in widget.ultrack_section.findChildren(type(widget.ultrack_input_files))
    assert widget.ultrack_output_files in widget.ultrack_section.findChildren(type(widget.ultrack_output_files))

    assert widget.build_progress_bar.isVisible() is False
    assert widget.db_gen_progress_bar.isVisible() is False
    assert widget.ultrack_progress_bar.isVisible() is False

    texts = _label_texts(widget.contour_section)
    assert "min" not in texts
    assert "max" not in texts
    assert "step" not in texts

    widget.deleteLater()
    viewer.close()


def test_main_widget_refreshes_cell_workflow_with_same_position_dir_as_project_status(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget.path_label.setText(str(tmp_path))

    widget._refresh_all()

    assert widget.data_panel.refreshed_pos_dir == pos_dir
    assert widget.hpc_cellpose_widget.refreshed_pos_dir == pos_dir
    assert widget.cell_workflow_widget.refreshed_pos_dir == pos_dir
    assert widget.analysis_widget.refreshed_pos_dir == pos_dir
    assert widget.nls_classification_widget.refreshed_pos_dir == pos_dir

    widget.deleteLater()
    viewer.close()
    _app.processEvents()  # flush deferred deletion before next test


def test_main_widget_persists_hpc_cellpose_state():
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    state = widget.get_state()
    assert state["hpc_cellpose"] == {"widget": "_StubWidget"}

    widget.set_state({"hpc_cellpose": {"frames": "0,2", "remote_host": "maestro.pasteur.fr"}})
    assert widget.hpc_cellpose_widget.state_received == {
        "frames": "0,2",
        "remote_host": "maestro.pasteur.fr",
    }

    widget.deleteLater()
    viewer.close()


def test_tracking_correction_shell_exposes_stable_section_attributes():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    # tracking_correction_section wrapper removed — sections are now top-level
    assert not hasattr(widget, "tracking_correction_section")
    assert widget.ultrack_section.title == "4. Ultrack Tracking"
    assert widget.correction_section.title == "5. Correction"
    assert widget.ultrack_db_browser_section.title == "Ultrack Database Browser"
    assert widget.layout().indexOf(widget.correction_section) < widget.layout().indexOf(
        widget.ultrack_db_browser_section
    )
    assert widget.correction_shortcuts_section.title == "Correction Shortcuts"
    assert widget.ultrack_section.is_expanded is False
    assert widget.correction_section.is_expanded is False
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
    correction_label_texts = _label_texts(widget.correction_section)
    ultrack_button_texts = {
        button.text()
        for button in widget.ultrack_section.findChildren(QPushButton)
    }
    ultrack_checkbox_texts = {
        checkbox.text() for checkbox in widget.ultrack_section.findChildren(QCheckBox)
    }

    assert "Save Tracked Labels" in correction_button_texts
    assert "Load Tracked Labels" in correction_button_texts
    assert "Reassign IDs" in correction_button_texts
    assert "Clean Holes / Islands" not in correction_button_texts
    assert "Fill Holes" in correction_button_texts
    assert "Fix Semiholes" in correction_button_texts
    assert "Clean Fragments" in correction_button_texts
    assert "Artifact cleanup" in correction_label_texts
    assert "Scope:" in correction_label_texts
    assert "Hole radius:" in correction_label_texts
    assert "Max opening:" in correction_label_texts
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
    assert "Resolve from validated" not in ultrack_checkbox_texts

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


def test_ultrack_section_has_no_legacy_resolve_route_and_keeps_local_status():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "ultrack_route_check")
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


def test_db_gen_exposes_node_probability_weight_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert hasattr(widget, "db_gen_quality_weight_spin")
    assert hasattr(widget, "db_gen_quality_exp_spin")
    assert hasattr(widget, "db_gen_circularity_weight_spin")
    assert widget.db_gen_quality_weight_spin.value() == pytest.approx(1.0)
    assert widget.db_gen_quality_exp_spin.value() == pytest.approx(8.0)
    assert widget.db_gen_circularity_weight_spin.value() == pytest.approx(0.25)

    widget.db_gen_quality_weight_spin.setValue(0.8)
    widget.db_gen_quality_exp_spin.setValue(6.0)
    widget.db_gen_circularity_weight_spin.setValue(0.35)

    cfg = widget._db_gen_config_from_controls()

    assert cfg.quality_weight == pytest.approx(0.8)
    assert cfg.quality_exponent == pytest.approx(6.0)
    assert cfg.circularity_weight == pytest.approx(0.35)

    widget.deleteLater()
    viewer.close()


def test_contour_maps_section_exposes_stage_files_and_foreground_threshold():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "foreground_section")
    assert widget.contour_fg_threshold_spin.minimum() == 0.0
    assert widget.contour_fg_threshold_spin.maximum() == 1.0
    assert widget.contour_fg_threshold_spin.value() == 0.5
    assert widget.contour_fg_threshold_spin.singleStep() == 0.01
    assert widget.contour_flow_threshold_spin.minimum() == 0.0
    assert widget.contour_flow_threshold_spin.maximum() == 10.0
    assert widget.contour_flow_threshold_spin.value() == 0.0
    assert widget.contour_flow_threshold_spin.singleStep() == 0.1

    input_text = " ".join(
        label.text()
        for label in widget.contour_input_files.findChildren(QLabel)
    )
    output_text = " ".join(
        label.text()
        for label in widget.contour_output_files.findChildren(QLabel)
    )
    assert "1_cellpose/nucleus_prob_3dt.tif" in input_text
    assert "1_cellpose/nucleus_dp_3dt.tif" in input_text
    assert "Flow threshold:" in _label_texts(widget.contour_section)
    assert "2_nucleus/foreground_masks.tif" in output_text
    assert "2_nucleus/foreground_scores.tif" in output_text
    assert "2_nucleus/contour_maps.tif" in output_text

    widget.deleteLater()
    viewer.close()


def test_nucleus_db_gen_and_ultrack_sections_expose_stage_file_rows():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_input_text = " ".join(
        label.text()
        for label in widget.db_gen_input_files.findChildren(QLabel)
    )
    db_output_text = " ".join(
        label.text()
        for label in widget.db_gen_output_files.findChildren(QLabel)
    )
    ultrack_input_text = " ".join(
        label.text()
        for label in widget.ultrack_input_files.findChildren(QLabel)
    )
    ultrack_output_text = " ".join(
        label.text()
        for label in widget.ultrack_output_files.findChildren(QLabel)
    )

    assert "2_nucleus/contour_maps.tif" in db_input_text
    assert "2_nucleus/foreground_masks.tif" in db_input_text
    assert "1_cellpose/nucleus_prob_zavg.tif" in db_input_text
    assert "2_nucleus/ultrack_workdir/data.db" in db_output_text
    assert "2_nucleus/ultrack_workdir/data.db" in ultrack_input_text
    assert "2_nucleus/tracked_labels.tif" in ultrack_output_text

    widget.deleteLater()
    viewer.close()


def test_nucleus_stage_file_widgets_show_present_and_missing_files(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus" / "ultrack_workdir").mkdir(parents=True)
    import tifffile

    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif", np.zeros((1, 1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif", np.zeros((1, 1, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "2_nucleus" / "contour_maps.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_masks.tif", np.zeros((1, 4, 4), dtype=np.uint8))
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", np.zeros((1, 4, 4), dtype=np.uint32))
    (pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db").write_bytes(b"sqlite placeholder")

    widget.refresh(pos_dir)

    texts = _label_texts(widget)
    assert texts.count("✓") >= 8
    assert "missing" in texts
    assert widget.contour_input_files in widget.contour_section.findChildren(type(widget.contour_input_files))
    assert widget.db_gen_output_files in widget.db_gen_section.findChildren(type(widget.db_gen_output_files))
    assert widget.ultrack_output_files in widget.ultrack_section.findChildren(type(widget.ultrack_output_files))

    widget.deleteLater()
    viewer.close()


def test_nucleus_stage_file_load_buttons_load_files_into_viewer(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    contours = np.ones((2, 4, 4), dtype=np.float32)
    masks = np.ones((2, 4, 4), dtype=np.uint8)
    labels = np.ones((2, 4, 4), dtype=np.uint32)
    import tifffile

    tifffile.imwrite(pos_dir / "2_nucleus" / "contour_maps.tif", contours)
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_masks.tif", masks)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", labels)

    widget.refresh(pos_dir)

    for files_widget in (
        widget.contour_output_files,
        widget.ultrack_output_files,
    ):
        for row in files_widget._rows:
            if row._full_path is not None:
                row._on_load_clicked()

    assert "2_nucleus_contour_maps" in viewer.layers
    assert "2_nucleus_foreground_masks" in viewer.layers
    assert "2_nucleus_tracked_labels" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["2_nucleus_contour_maps"].data, contours)
    np.testing.assert_array_equal(viewer.layers["2_nucleus_foreground_masks"].data, masks)
    np.testing.assert_array_equal(viewer.layers["2_nucleus_tracked_labels"].data, labels)

    widget.deleteLater()
    viewer.close()


def test_contour_foreground_threshold_persists_without_old_foreground_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.contour_fg_threshold_spin.setValue(0.42)

    state = widget.get_state()
    assert "foreground_mask" not in state
    assert state["cellprob"]["foreground_threshold"] == pytest.approx(0.42)
    widget.deleteLater()

    widget = widget_class(viewer)
    widget.set_state({
        **state,
        "cellprob": {**state["cellprob"], "foreground_threshold": 0.73},
        "foreground_mask": {"threshold": 0.1, "gamma": 2.0, "niter": 50},
    })

    assert widget.contour_fg_threshold_spin.value() == pytest.approx(0.73)

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


def test_ultrack_tracker_hides_db_build_duplicate_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    for control in (
        widget.ultrack_min_area_spin,
        widget.ultrack_max_dist_spin,
        widget.ultrack_max_neighbors_spin,
        widget.ultrack_linking_mode_combo,
        widget.ultrack_iou_weight_spin,
    ):
        assert control.parent() is None

    ultrack_labels = {
        label.text()
        for label in widget.ultrack_section.findChildren(QLabel)
    }
    assert "Min Area (px):" not in ultrack_labels
    assert "Max Distance (px):" not in ultrack_labels
    assert "Max Neighbors:" not in ultrack_labels
    assert "Linking Mode:" not in ultrack_labels
    assert "IoU Weight:" not in ultrack_labels
    assert {
        "min_area",
        "max_distance",
        "max_neighbors",
        "linking_mode",
        "iou_weight",
    }.isdisjoint(widget.get_state()["ultrack"])

    widget.deleteLater()
    viewer.close()


def test_ultrack_parameters_are_grouped_by_workflow_stage():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_group_labels = {
        label.text()
        for label in widget.db_gen_section.findChildren(QLabel)
    }
    ultrack_group_labels = {
        label.text()
        for label in widget.ultrack_section.findChildren(QLabel)
    }

    assert {"Candidate extraction", "Candidate linking", "Node scoring"} <= db_group_labels
    assert {"Track scope", "Event penalties", "Solver scoring"} <= ultrack_group_labels
    assert "Validated seed prior" in db_group_labels
    assert "Validated seed prior" not in ultrack_group_labels
    assert widget.ultrack_quality_exp_spin.parent() is None

    widget.deleteLater()
    viewer.close()


def test_validated_seed_prior_controls_follow_db_generation_checkbox():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    controls = [
        widget.ultrack_seed_weight_spin,
        widget.ultrack_seed_space_spin,
        widget.ultrack_seed_time_spin,
        widget.ultrack_seed_window_spin,
    ]

    widget.db_gen_use_validated_check.setChecked(False)
    _app.processEvents()
    assert all(not control.isEnabled() for control in controls)

    widget.db_gen_use_validated_check.setChecked(True)
    _app.processEvents()
    assert all(control.isEnabled() for control in controls)

    widget.deleteLater()
    viewer.close()


def test_db_scoring_controls_stay_enabled_without_validation():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    seed_controls = [
        widget.ultrack_seed_weight_spin,
        widget.ultrack_seed_space_spin,
        widget.ultrack_seed_time_spin,
        widget.ultrack_seed_window_spin,
    ]
    scoring_controls = [
        widget.db_gen_quality_weight_spin,
        widget.db_gen_quality_exp_spin,
        widget.db_gen_circularity_weight_spin,
    ]

    widget.db_gen_use_validated_check.setChecked(False)
    _app.processEvents()

    assert all(control.isEnabled() for control in scoring_controls)
    assert all(not control.isEnabled() for control in seed_controls)

    widget.deleteLater()
    viewer.close()


def test_old_ultrack_linking_state_migrates_to_db_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.set_state({
        "ultrack": {
            "min_area": 700,
            "max_distance": 21.0,
            "max_neighbors": 9,
            "linking_mode": "iou",
            "iou_weight": 0.65,
        }
    })

    assert widget.db_gen_min_area_spin.value() == 700
    assert widget.db_gen_max_dist_spin.value() == pytest.approx(21.0)
    assert widget.db_gen_max_neighbors_spin.value() == 9
    assert widget.db_gen_linking_mode_combo.currentText() == "iou"
    assert widget.db_gen_iou_weight_spin.value() == pytest.approx(0.65)

    widget.deleteLater()
    viewer.close()


def test_ultrack_seed_prior_controls_persist_through_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.db_gen_use_validated_check.setChecked(True)
    widget.ultrack_seed_weight_spin.setValue(0.75)
    widget.ultrack_seed_space_spin.setValue(30.0)
    widget.ultrack_seed_time_spin.setValue(3.0)
    widget.ultrack_seed_window_spin.setValue(7)

    state = widget.get_state()
    widget.deleteLater()

    widget = widget_class(viewer)
    widget.set_state(state)

    assert widget.db_gen_use_validated_check.isChecked()
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
    widget.ultrack_power_spin.setValue(4.25)
    widget.db_gen_quality_exp_spin.setValue(9.5)
    widget.ultrack_seed_weight_spin.setValue(0.85)
    widget.ultrack_seed_space_spin.setValue(35.0)
    widget.ultrack_seed_time_spin.setValue(4.0)
    widget.ultrack_seed_window_spin.setValue(8)

    widget._on_ultrack_terminal()
    script = _read_launched_script(captured)

    assert "power=4.25" in script
    assert "seg_min_area=" not in script
    assert "max_distance=" not in script
    assert "run_solve(working_dir, cfg, overwrite=True)" in script
    assert "export_tracked_labels(" in script
    assert "ingest_hypotheses_to_db" not in script
    assert "write_seed_prior_node_probs" not in script

    widget.deleteLater()
    viewer.close()


def test_db_generation_terminal_script_includes_validated_seed_prior_controls(tmp_path, monkeypatch):
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
    widget.db_gen_quality_exp_spin.setValue(10.5)
    widget.db_gen_use_validated_check.setChecked(True)
    widget.ultrack_seed_weight_spin.setValue(0.9)
    widget.ultrack_seed_space_spin.setValue(40.0)
    widget.ultrack_seed_time_spin.setValue(4.5)
    widget.ultrack_seed_window_spin.setValue(9)

    widget._on_db_gen_terminal()
    script = _read_launched_script(captured)

    assert "quality_exponent=10.5" in script
    assert "seed_weight=0.9" in script
    assert "seed_sigma_space=40.0" in script
    assert "seed_tau_time=4.5" in script
    assert "seed_max_dt=9" in script

    widget.deleteLater()
    viewer.close()
def test_db_generation_terminal_script_includes_seed_prior_and_nucleus_prob_zavg(tmp_path, monkeypatch):
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

    widget.db_gen_quality_exp_spin.setValue(9.0)
    widget.db_gen_use_validated_check.setChecked(True)
    widget.ultrack_seed_weight_spin.setValue(0.75)
    widget.ultrack_seed_space_spin.setValue(30.0)
    widget.ultrack_seed_time_spin.setValue(3.0)
    widget.ultrack_seed_window_spin.setValue(7)

    widget._on_db_gen_terminal()
    script = _read_launched_script(captured)

    assert "nucleus_prob_zavg.tif" in script
    assert "quality_exponent=9.0" in script
    assert "seed_weight=0.75" in script
    assert "seed_sigma_space=30.0" in script
    assert "seed_tau_time=3.0" in script
    assert "seed_max_dt=7" in script
    assert "nucleus_prob_zavg_path=nucleus_prob_zavg_path" in script

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

    if widget.minimumSizeHint().width() > scroll.viewport().width():
        assert scroll.horizontalScrollBar().maximum() > 0
    else:
        assert scroll.horizontalScrollBar().maximum() == 0

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
        widget.contour_flow_threshold_spin,
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

    texts = _label_texts(widget.contour_section)
    assert "min" not in texts
    assert "max" not in texts
    assert "step" not in texts
    assert params_scroll.horizontalScrollBar().maximum() >= 0

    host.deleteLater()
    viewer.close()


def test_contour_maps_build_writes_contour_scores_and_thresholded_masks(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    widget._pos_dir = pos_dir
    widget.contour_fg_threshold_spin.setValue(0.5)

    import tifffile

    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif", np.zeros((2, 1, 3, 3), dtype=np.float32))
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif", np.zeros((2, 1, 2, 3, 3), dtype=np.float32))

    def fake_build(
        prob_3d,
        _dp_3d,
        _thresholds,
        _gammas,
        *,
        flow_threshold=0.0,
        mask_callback=None,
    ):
        frame_value = 0.25 if float(prob_3d[0, 0, 0]) == 0.0 else 0.75
        boundary = np.full((3, 3), frame_value, dtype=np.float32)
        foreground = np.array(
            [[0.0, 0.49, 0.5], [0.51, 0.75, 1.0], [0.2, 0.8, 0.4]],
            dtype=np.float32,
        )
        return boundary, foreground

    monkeypatch.setattr(widget, "_build_consensus_boundary_averaged", fake_build)
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif", np.stack([
        np.zeros((1, 3, 3), dtype=np.float32),
        np.ones((1, 3, 3), dtype=np.float32),
    ]))

    widget._on_build_contour_maps()

    contours = tifffile.imread(pos_dir / "2_nucleus" / "contour_maps.tif")
    scores = tifffile.imread(pos_dir / "2_nucleus" / "foreground_scores.tif")
    masks = tifffile.imread(pos_dir / "2_nucleus" / "foreground_masks.tif")
    assert contours.shape == (2, 3, 3)
    assert contours.dtype == np.float32
    assert scores.dtype == np.float32
    assert masks.dtype == np.uint8
    np.testing.assert_array_equal(masks, (scores >= 0.5).astype(np.uint8))
    output_texts = _label_texts(widget.contour_output_files)
    assert "✓" in output_texts
    assert "Contour maps and foreground masks built." in widget.contour_status_lbl.text()
    assert widget.build_progress_bar.isVisible() is False

    widget.deleteLater()
    viewer.close()


def test_contour_maps_build_passes_flow_threshold_to_builder(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    widget._pos_dir = pos_dir
    widget.contour_flow_threshold_spin.setValue(1.2)

    import tifffile

    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif",
        np.zeros((1, 1, 3, 3), dtype=np.float32),
    )
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif",
        np.zeros((1, 1, 2, 3, 3), dtype=np.float32),
    )
    captured: dict[str, float] = {}

    def fake_build(
        _prob_3d,
        _dp_3d,
        _thresholds,
        _gammas,
        *,
        flow_threshold,
        mask_callback=None,
    ):
        captured["flow_threshold"] = flow_threshold
        return (
            np.zeros((3, 3), dtype=np.float32),
            np.ones((3, 3), dtype=np.float32),
        )

    monkeypatch.setattr(widget, "_build_consensus_boundary_averaged", fake_build)

    widget._on_build_contour_maps()

    assert captured["flow_threshold"] == pytest.approx(1.2)

    widget.deleteLater()
    viewer.close()


def test_contour_terminal_script_includes_flow_threshold(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    captured = _install_terminal_capture(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    widget._pos_dir = pos_dir
    widget.contour_flow_threshold_spin.setValue(1.4)

    import tifffile

    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif",
        np.zeros((1, 1, 3, 3), dtype=np.float32),
    )
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif",
        np.zeros((1, 1, 2, 3, 3), dtype=np.float32),
    )

    widget._on_run_contour_terminal()
    script = _read_launched_script(captured)

    assert "flow_threshold = 1.4" in script
    assert "flow_threshold=flow_threshold" in script

    widget.deleteLater()
    viewer.close()


def test_contour_filter_preview_updates_layer_without_overwriting(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    widget._pos_dir = pos_dir

    import tifffile

    original = np.zeros((2, 4, 4), dtype=np.float32)
    original[1, 2, 2] = 10.0
    tifffile.imwrite(pos_dir / "2_nucleus" / "contour_maps.tif", original)

    def fake_filter(contours, params):
        assert params.median_kernel_time == 3
        return np.asarray(contours, dtype=np.float32) + 1.0

    monkeypatch.setattr("cellflow.segmentation.compute_filtered_contour_maps", fake_filter)
    widget.contour_filter_median_time_spin.setValue(3)

    widget._on_preview_contour_filter()

    np.testing.assert_array_equal(
        tifffile.imread(pos_dir / "2_nucleus" / "contour_maps.tif"),
        original,
    )
    assert "Contour Map: Nucleus" in viewer.layers
    np.testing.assert_array_equal(
        viewer.layers["Contour Map: Nucleus"].data,
        original + 1.0,
    )

    widget.deleteLater()
    viewer.close()


def test_contour_filter_run_overwrites_contour_maps(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    widget._pos_dir = pos_dir

    import tifffile

    original = np.zeros((2, 4, 4), dtype=np.float32)
    original[0, 1, 1] = 5.0
    tifffile.imwrite(pos_dir / "2_nucleus" / "contour_maps.tif", original)

    monkeypatch.setattr(
        "cellflow.segmentation.compute_filtered_contour_maps",
        lambda contours, params: np.asarray(contours, dtype=np.float32) + 2.0,
    )

    widget._on_run_contour_filter()

    written = tifffile.imread(pos_dir / "2_nucleus" / "contour_maps.tif")
    assert written.dtype == np.float32
    np.testing.assert_array_equal(written, original + 2.0)
    assert "Contour Map: Nucleus" in viewer.layers
    np.testing.assert_array_equal(
        viewer.layers["Contour Map: Nucleus"].data,
        original + 2.0,
    )
    assert "Filtered contour maps written to contour_maps.tif." in widget.contour_status_lbl.text()
    assert "✓" in _label_texts(widget.contour_output_files)

    widget.deleteLater()
    viewer.close()


def test_contour_filter_controls_persist_through_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.contour_filter_median_time_spin.setValue(3)
    widget.contour_filter_median_space_spin.setValue(5)
    widget.contour_filter_gauss_time_spin.setValue(1.5)
    widget.contour_filter_gauss_space_spin.setValue(2.5)

    state = widget.get_state()
    widget.deleteLater()

    widget = widget_class(viewer)
    widget.set_state(state)

    assert widget.contour_filter_median_time_spin.value() == 3
    assert widget.contour_filter_median_space_spin.value() == 5
    assert abs(widget.contour_filter_gauss_time_spin.value() - 1.5) < 0.01
    assert abs(widget.contour_filter_gauss_space_spin.value() - 2.5) < 0.01

    widget.deleteLater()
    viewer.close()


def test_contour_flow_threshold_persists_through_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.contour_flow_threshold_spin.setValue(1.7)

    state = widget.get_state()
    widget.deleteLater()

    widget = widget_class(viewer)
    widget.set_state(state)

    assert widget.contour_flow_threshold_spin.value() == pytest.approx(1.7)

    widget.deleteLater()
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
    assert widget.extend_area_weight_spin.value() == 1.0
    assert widget.extend_iou_weight_spin.value() == 1.0
    assert widget.extend_distance_weight_spin.value() == 0.25
    assert widget.extend_overlap_penalty_spin.value() == 1.0
    assert widget.extend_greedy_overwrite_check.isChecked() is False
    assert widget.retrack_max_dist_spin.value() == 20.0

    widget.deleteLater()
    viewer.close()


def test_validated_overlay_uses_green_fill_at_default_opacity_below_spotlight():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    viewer.add_labels(np.array([[[0, 1], [1, 0]]], dtype=np.uint8), name="Tracked: Nucleus")
    viewer.add_image(np.zeros((2, 2, 4), dtype=np.float32), name="CellSpotlight", rgb=True)
    widget._add_validated_overlay(np.array([[[0, 1], [0, 0]]], dtype=np.uint8))

    layer = viewer.layers["Validated: Nucleus"]
    color = layer.get_color(1)

    assert layer.contour == 0
    assert layer.opacity == 0.4
    assert np.allclose(color[:3], [0.0, 1.0, 0.0], atol=1e-6)
    assert color[3] == 1.0
    assert viewer.layers.index("Validated: Nucleus") < viewer.layers.index("CellSpotlight")

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
        widget._fill_holes_btn,
        widget._fix_semiholes_btn,
        widget._clean_fragments_btn,
        widget._goto_btn,
    ):
        assert button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding

    widget.deleteLater()
    viewer.close()


def test_correction_widget_fill_holes_uses_configured_radius():
    _app, viewer = _make_viewer()
    widget_class = _load_correction_widget_class()
    widget = widget_class(viewer)
    labels = np.zeros((1, 12, 12), dtype=np.uint32)
    labels[0, 1:11, 1:4] = 1
    labels[0, 1:11, 8:11] = 2
    labels[0, 1:3, 4:8] = 1
    labels[0, 9:11, 4:8] = 2
    labels[0, 3:9, 4:8] = 0
    layer = viewer.add_labels(labels, name="Tracked: Nucleus")

    widget.activate_layer(layer)
    widget._hole_radius_spin.setValue(1)
    widget._fill_holes()

    gap = np.asarray(layer.data)[0, 3:9, 4:8]
    assert np.any(gap == 0)
    assert np.any(gap == 1)
    assert np.any(gap == 2)
    np.testing.assert_array_equal(np.asarray(layer.data)[0, 0, :], np.zeros(12, dtype=np.uint32))

    widget._hole_radius_spin.setValue(999)
    widget._fill_holes()

    assert not np.any(np.asarray(layer.data)[0, 3:9, 4:8] == 0)
    np.testing.assert_array_equal(np.asarray(layer.data)[0, 0, :], np.zeros(12, dtype=np.uint32))

    widget.deleteLater()
    viewer.close()


def test_correction_widget_artifact_cleanup_current_frame_scope_only_changes_selected_frame():
    _app, viewer = _make_viewer()
    widget_class = _load_correction_widget_class()
    widget = widget_class(viewer)
    labels = np.ones((2, 6, 6), dtype=np.uint32)
    labels[:, 2:4, 2:4] = 0
    layer = viewer.add_labels(labels, name="Tracked: Nucleus")

    widget.activate_layer(layer)
    viewer.dims.current_step = (1, 0, 0)
    widget._cleanup_scope_combo.setCurrentText("Current frame")
    widget._hole_radius_spin.setValue(999)
    widget._fill_holes()

    assert np.any(np.asarray(layer.data)[0] == 0)
    assert not np.any(np.asarray(layer.data)[1] == 0)

    widget.deleteLater()
    viewer.close()


def test_correction_widget_artifact_cleanup_all_frames_records_each_changed_frame():
    _app, viewer = _make_viewer()
    widget_class = _load_correction_widget_class()
    widget = widget_class(viewer)
    labels = np.ones((2, 6, 6), dtype=np.uint32)
    labels[:, 2:4, 2:4] = 0
    layer = viewer.add_labels(labels, name="Tracked: Nucleus")
    recorded = []
    original_record_history = widget._record_history

    def record_history(layer_arg, t, before):
        recorded.append(t)
        original_record_history(layer_arg, t, before)

    widget.activate_layer(layer)
    widget._record_history = record_history
    widget._cleanup_scope_combo.setCurrentText("All frames")
    widget._hole_radius_spin.setValue(999)
    widget._fill_holes()

    assert not np.any(np.asarray(layer.data) == 0)
    assert recorded == [0, 1]
    assert "Filled holes in 2 frame(s)" in widget._status.text()

    widget.deleteLater()
    viewer.close()


def test_correction_widget_notifies_when_selected_label_changes():
    _app, viewer = _make_viewer()
    widget_class = _load_correction_widget_class()
    widget = widget_class(viewer)
    labels = np.zeros((1, 8, 8), dtype=np.uint32)
    labels[0, 1:4, 1:4] = 7
    labels[0, 4:7, 4:7] = 9
    layer = viewer.add_labels(labels, name="Tracked: Nucleus")
    calls = []

    widget.activate_layer(layer)
    widget.set_selection_callback(lambda t, label: calls.append((t, label)))

    widget._update_highlight(0, 7)
    widget._update_highlight(0, 7)
    widget._update_highlight(0, 9)
    widget._update_highlight(0, 0)

    assert calls == [(0, 7), (0, 9), (0, 0)]

    widget.deleteLater()
    viewer.close()


def test_correction_widget_defaults_to_one_pixel_outlines():
    _app, viewer = _make_viewer()
    widget_class = _load_correction_widget_class()
    widget = widget_class(viewer)
    layer = viewer.add_labels(np.zeros((1, 4, 4), dtype=np.uint8), name="Tracked: Nucleus")

    widget.activate_layer(layer)

    assert widget._outline_btn.isChecked() is True
    assert layer.contour == 1

    widget.deleteLater()
    viewer.close()


def test_correction_widget_deactivation_restores_previous_outline_width():
    _app, viewer = _make_viewer()
    widget_class = _load_correction_widget_class()
    widget = widget_class(viewer)
    layer = viewer.add_labels(np.zeros((1, 4, 4), dtype=np.uint8), name="Tracked: Nucleus")
    layer.contour = 3

    widget.activate_layer(layer)
    widget.deactivate()

    assert layer.contour == 3

    widget.deleteLater()
    viewer.close()


def test_correction_widget_highlight_adds_soft_spotlight():
    _app, viewer = _make_viewer()
    widget_class = _load_correction_widget_class()
    widget = widget_class(viewer)
    labels = np.zeros((1, 81, 81), dtype=np.uint8)
    labels[0, 30:50, 30:50] = 7
    layer = viewer.add_labels(labels, name="Tracked: Nucleus")

    widget.activate_layer(layer)
    widget._update_highlight(0, 7)

    spotlight = viewer.layers["CellSpotlight"]
    data = np.asarray(spotlight.data)

    assert spotlight.visible is True
    assert data.shape == (81, 81, 4)
    assert np.allclose(data[..., :3], 0.0)
    assert data[40, 40, 3] == pytest.approx(0.0)
    assert data[0, 40, 3] == pytest.approx(0.7)
    assert data[20, 40, 3] == pytest.approx(0.35, abs=0.08)

    widget.deleteLater()
    viewer.close()


def test_correction_widget_spotlight_is_not_used_as_intensity_image():
    _app, viewer = _make_viewer()
    widget_class = _load_correction_widget_class()
    widget = widget_class(viewer)
    labels = np.zeros((1, 9, 9), dtype=np.uint8)
    labels[0, 3:6, 3:6] = 7
    layer = viewer.add_labels(labels, name="Tracked: Nucleus")

    widget.activate_layer(layer)
    widget._update_highlight(0, 7)

    assert widget._image_frame(0) is None

    widget.deleteLater()
    viewer.close()


def test_nucleus_workflow_selects_best_overlapping_nucleus_for_cell_selection():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[1, 1:5, 1:5] = 3
    labels[1, 5:7, 5:7] = 6
    source = np.zeros((2, 8, 8), dtype=np.uint32)
    source[1, 2:6, 2:6] = 10
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget.activate_layer(viewer.layers["Tracked: Nucleus"])

    widget.select_matching_nucleus_label(1, 10, source_labels=source)

    assert widget.correction_widget._selected_label == 3
    assert widget.correction_widget._selected_t == 1

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

def test_nucleus_workflow_has_five_canonical_sections_plus_optional_db_browser():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.contour_section.title == "1. Contour Maps"
    assert not hasattr(widget, "foreground_section")
    assert widget.db_gen_section.title == "2. Ultrack Database Generation"
    assert widget.ultrack_section.title == "4. Ultrack Tracking"
    assert widget.correction_section.title == "5. Correction"
    assert widget.ultrack_db_browser_section.title == "Ultrack Database Browser"
    assert widget.layout().indexOf(widget.correction_section) < widget.layout().indexOf(
        widget.ultrack_db_browser_section
    )

    assert not hasattr(widget, "tracking_correction_section")

    widget.deleteLater()
    viewer.close()


def test_deprecated_sections_are_removed():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "gen_section")
    assert not hasattr(widget, "db_section")

    widget.deleteLater()
    viewer.close()


def test_canonical_sections_expose_required_elements():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    # Section 1: Contour Maps
    assert hasattr(widget, "contour_input_files")
    assert hasattr(widget, "contour_output_files")
    assert hasattr(widget, "build_btn")
    assert hasattr(widget, "contour_terminal_btn")
    assert hasattr(widget, "contour_status_lbl")
    assert hasattr(widget, "build_progress_bar")
    assert hasattr(widget, "contour_fg_threshold_spin")
    assert hasattr(widget, "contour_flow_threshold_spin")
    assert hasattr(widget, "contour_filter_median_time_spin")
    assert hasattr(widget, "contour_filter_median_space_spin")
    assert hasattr(widget, "contour_filter_gauss_time_spin")
    assert hasattr(widget, "contour_filter_gauss_space_spin")
    assert hasattr(widget, "preview_contour_filter_btn")
    assert hasattr(widget, "run_contour_filter_btn")
    assert not hasattr(widget, "fg_threshold_spin")

    # Section 3: Ultrack Database Generation
    assert hasattr(widget, "db_gen_input_files")
    assert hasattr(widget, "db_gen_output_files")
    assert hasattr(widget, "run_db_gen_btn")
    assert hasattr(widget, "db_gen_terminal_btn")
    assert hasattr(widget, "db_gen_status_lbl")
    assert hasattr(widget, "db_gen_progress_bar")

    # Section 4: Ultrack Database Browser
    assert hasattr(widget, "ultrack_db_info_lbl")
    assert hasattr(widget, "ultrack_db_active_btn")
    assert hasattr(widget, "ultrack_db_refresh_btn")
    assert not hasattr(widget, "ultrack_db_mode_combo")
    assert hasattr(widget, "ultrack_db_hierarchy_slider")
    assert hasattr(widget, "ultrack_db_height_lbl")
    assert hasattr(widget, "ultrack_db_section_status_lbl")

    # Section 5: Ultrack Tracking
    assert hasattr(widget, "ultrack_input_files")
    assert hasattr(widget, "ultrack_output_files")
    assert hasattr(widget, "run_ultrack_btn")
    assert hasattr(widget, "ultrack_terminal_btn")
    assert hasattr(widget, "ultrack_status_lbl")
    assert hasattr(widget, "ultrack_progress_bar")

    # Correction has no Run button per spec.
    assert hasattr(widget, "correction_status_lbl")

    widget.deleteLater()
    viewer.close()


def test_ultrack_section_is_top_level_without_legacy_route_selector():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_section.title == "4. Ultrack Tracking"
    assert not hasattr(widget, "ultrack_route_check")
    assert widget.ultrack_status_lbl.text() == ""
    assert widget.ultrack_progress_bar.isVisible() is False

    widget.deleteLater()
    viewer.close()


def test_correction_section_is_top_level():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.correction_section.title == "5. Correction"
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


def _label_texts(widget):
    return [child.text() for child in widget.findChildren(QLabel)]


def _progress_bars(widget):
    return widget.findChildren(QProgressBar)


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
    def fake_build_database(**kwargs):
        foreground = np.asarray(tifffile.imread(kwargs["foreground_masks_path"]))
        contours = np.asarray(tifffile.imread(kwargs["contour_maps_path"]))
        if foreground.ndim == 4 and foreground.shape[1] == 1:
            foreground = foreground[:, 0]
        if contours.ndim == 4 and contours.shape[1] == 1:
            contours = contours[:, 0]
        module._ultrack_segment(
            foreground,
            contours,
            kwargs["cfg"],
            overwrite=True,
            max_segments_per_time=1_000_000,
        )
        calls.append(("build", kwargs["use_validated"]))
        data_db = kwargs["working_dir"] / "data.db"
        data_db.parent.mkdir(parents=True, exist_ok=True)
        data_db.write_bytes(b"sqlite placeholder")
        return {"database": str(data_db)}

    monkeypatch.setattr(module, "build_ultrack_database", fake_build_database)

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
    assert calls[1] == ("build", False)
    assert widget.run_db_gen_btn.isEnabled()
    assert widget.db_gen_terminal_btn.isEnabled()
    assert "complete" in widget.db_gen_status_lbl.text().lower()
    assert widget.db_gen_progress_bar.isVisible() is False
    assert "✓" in _label_texts(widget.db_gen_output_files)

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

    assert "foreground_masks" in script
    assert "contour_maps" in script
    assert "nucleus_prob_zavg" in script
    assert "quality_weight=" in script
    assert "circularity_weight=" in script
    assert "build_ultrack_database" in script
    assert "write_seed_prior_node_probs" not in script
    assert "run_linking" not in script
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


def _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, heights):
    module = sys.modules[widget_class.__module__]
    states = tuple(module._HierarchyCutState((), height) for height in heights)
    monkeypatch.setattr(widget, "_query_hierarchy_cut_states", lambda *a: states)
    return states


def test_ultrack_db_browser_summary_label_wraps_instead_of_widening():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_db_info_lbl.wordWrap() is True
    assert (
        widget.ultrack_db_info_lbl.sizePolicy().horizontalPolicy()
        != QSizePolicy.Policy.Expanding
    )

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_exposes_hierarchy_only_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "ultrack_db_mode_combo")
    assert widget.ultrack_db_hierarchy_slider.minimum() == 0
    assert widget.ultrack_db_hierarchy_slider.maximum() == 100
    assert widget.ultrack_db_hierarchy_slider.value() == 50
    assert widget._ultrack_db_slider_row.isHidden() is False

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
    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
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
    widget.ultrack_db_connected_focus_check.setChecked(True)
    widget._ultrack_db_selected_node_id = 222
    widget._ultrack_db_selected_frame = 4
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
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
    widget.ultrack_db_connected_focus_check.setChecked(True)
    widget.ultrack_db_edge_alpha_check.setChecked(True)
    widget.ultrack_db_prob_alpha_check.setChecked(True)
    widget._ultrack_db_selected_node_id = 222
    widget._ultrack_db_selected_frame = 4
    monkeypatch.setattr(widget, "_current_t", lambda: 5)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
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
    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
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
    widget.ultrack_db_connected_focus_check.setChecked(True)
    widget._ultrack_db_selected_node_id = 222
    widget._ultrack_db_selected_frame = 4
    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
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
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.25, 0.75))
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
    widget.ultrack_db_hierarchy_slider.setValue(0)
    widget.ultrack_db_prob_alpha_check.setChecked(True)
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda path, frame: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))

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


def test_ultrack_db_browser_shows_summary_while_rendering_hierarchy(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True

    calls = []
    labels = np.zeros((5, 5), dtype=np.uint32)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda path, frame: "summary stats")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda path, frame, height: calls.append((path, frame, height))
        or (labels, "rendered hierarchy cut"),
    )

    widget._refresh_ultrack_db_browser()

    assert calls == [(db_path, 0, 0.5)]
    assert widget.ultrack_db_info_lbl.text() == "summary stats"
    assert widget.ultrack_db_section_status_lbl.text() == "rendered hierarchy cut"
    assert "Ultrack DB Preview" in viewer.layers

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_does_not_add_contour_or_foreground_layers(tmp_path, monkeypatch):
    import tifffile

    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    db_path = pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    tifffile.imwrite(pos_dir / "2_nucleus" / "contour_maps.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_masks.tif", np.zeros((1, 4, 4), dtype=np.uint8))
    widget._pos_dir = pos_dir
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True

    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda path, frame: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *args: (np.zeros((4, 4), dtype=np.uint32), "rendered hierarchy cut"),
    )

    widget._refresh_ultrack_db_browser()

    assert "Ultrack DB Preview" in viewer.layers
    assert "Ultrack DB Annotations" not in viewer.layers
    assert "Contour Maps: Nucleus" not in viewer.layers
    assert "Foreground Masks: Nucleus" not in viewer.layers

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


def test_ultrack_tracking_refreshes_stage_output_files(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus" / "ultrack_workdir").mkdir(parents=True)
    (pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db").write_bytes(b"sqlite placeholder")
    widget._pos_dir = pos_dir

    labels = np.ones((2, 4, 4), dtype=np.uint32)

    def fake_export(_working_dir, _cfg, tracked_path, **_kwargs):
        tracked_path.parent.mkdir(parents=True, exist_ok=True)
        import tifffile
        tifffile.imwrite(tracked_path, labels)
        return labels

    monkeypatch.setattr(module, "run_solve", lambda *a, **kw: iter([(1, 1, "solved")]))
    monkeypatch.setattr(module, "export_tracked_labels", fake_export)

    widget._on_run_ultrack()

    assert "Tracked: Nucleus" in viewer.layers
    assert (pos_dir / "2_nucleus" / "tracked_labels.tif").exists()
    assert "✓" in _label_texts(widget.ultrack_output_files)
    assert widget.ultrack_progress_bar.isVisible() is False

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


def test_extend_passes_weight_parameters_to_db_tracker(tmp_path, monkeypatch):
    from cellflow.database.validation import validate_track

    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]

    pos_dir = tmp_path / "pos00"
    db_path = pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = pos_dir
    validate_track(pos_dir, 9, [1])

    labels = np.zeros((2, 12, 12), dtype=np.uint32)
    labels[0, 3:6, 3:6] = 7
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 7

    widget.extend_max_dist_spin.setValue(31.0)
    widget.extend_area_weight_spin.setValue(0.7)
    widget.extend_iou_weight_spin.setValue(1.5)
    widget.extend_distance_weight_spin.setValue(0.2)
    widget.extend_overlap_penalty_spin.setValue(2.0)
    widget.extend_greedy_overwrite_check.setChecked(True)

    captured = {}

    def fake_extend_track_from_db(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(module, "extend_track_from_db", fake_extend_track_from_db)

    widget._on_extend(direction="forward")

    assert captured["d_max"] == pytest.approx(31.0)
    assert captured["area_weight"] == pytest.approx(0.7)
    assert captured["iou_weight"] == pytest.approx(1.5)
    assert captured["distance_weight"] == pytest.approx(0.2)
    assert captured["overlap_penalty"] == pytest.approx(2.0)
    assert captured["greedy_overwrite"] is True
    assert captured["validated_tracks"] == {9: {1}}

    widget.deleteLater()
    viewer.close()


def test_extend_greedy_overwrite_paints_combined_assignments(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]

    pos_dir = tmp_path / "pos00"
    db_path = pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = pos_dir

    labels = np.zeros((2, 32, 32), dtype=np.uint32)
    labels[0, 5:10, 5:10] = 7
    labels[1, 6:11, 6:11] = 9
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 7
    widget.extend_greedy_overwrite_check.setChecked(True)

    source_mask = np.zeros((32, 32), dtype=bool)
    source_mask[6:11, 6:11] = True
    displaced_mask = np.zeros((32, 32), dtype=bool)
    displaced_mask[20:25, 20:25] = True

    result = types.SimpleNamespace(
        target_frame=1,
        candidate_label=101,
        candidate_partition=0,
        mask_2d=source_mask,
        bbox=(6, 6, 11, 11),
        centroid_distance=1.0,
        area_ratio=1.0,
        centroid_corrected_iou=1.0,
        existing_overlap=1.0,
        assignments=(
            types.SimpleNamespace(cell_id=7, mask_2d=source_mask),
            types.SimpleNamespace(cell_id=9, mask_2d=displaced_mask),
        ),
    )
    monkeypatch.setattr(module, "extend_track_from_db", lambda **_kwargs: result)

    widget._on_extend(direction="forward")

    frame = viewer.layers["Tracked: Nucleus"].data[1]
    assert np.all(frame[6:11, 6:11] == 7)
    assert np.all(frame[20:25, 20:25] == 9)
    assert "reassigned 1 conflict" in widget.correction_status_lbl.text()

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
    widget.db_gen_quality_weight_spin.setValue(0.8)
    widget.db_gen_quality_exp_spin.setValue(6.0)
    widget.db_gen_circularity_weight_spin.setValue(0.35)
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
    assert abs(widget.db_gen_quality_weight_spin.value() - 0.8) < 0.01
    assert abs(widget.db_gen_quality_exp_spin.value() - 6.0) < 0.01
    assert abs(widget.db_gen_circularity_weight_spin.value() - 0.35) < 0.01
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


def _add_ultrack_node(
    session,
    *,
    node_id: int,
    parent_id: int,
    height: float,
    bbox: tuple[int, int, int, int],
) -> None:
    import pickle
    from ultrack.core.database import NodeDB
    from ultrack.core.segmentation.node import Node

    y0, x0, y1, x1 = bbox
    mask = np.ones((1, y1 - y0, x1 - x0), dtype=bool)
    node_obj = Node.from_mask(
        time=0,
        mask=mask,
        bbox=np.array([0, y0, x0, 1, y1, x1], dtype=np.int64),
        node_id=node_id,
    )
    session.add(
        NodeDB(
            id=node_id,
            t=0,
            t_node_id=node_id,
            t_hier_id=1,
            z=0,
            y=(y0 + y1) / 2,
            x=(x0 + x1) / 2,
            area=int(mask.sum()),
            height=float(height),
            hier_parent_id=parent_id,
            pickle=pickle.dumps(node_obj),
        )
    )


def _make_ultrack_db_with_equal_height_plateau(db_path: Path) -> None:
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import Base
    from ultrack.utils.constants import NO_PARENT

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _add_ultrack_node(
            session,
            node_id=100,
            parent_id=NO_PARENT,
            height=10.0,
            bbox=(0, 0, 2, 4),
        )
        _add_ultrack_node(
            session,
            node_id=101,
            parent_id=100,
            height=10.0,
            bbox=(0, 0, 2, 2),
        )
        _add_ultrack_node(
            session,
            node_id=102,
            parent_id=100,
            height=10.0,
            bbox=(0, 2, 2, 4),
        )
        _add_ultrack_node(
            session,
            node_id=201,
            parent_id=101,
            height=1.0,
            bbox=(0, 0, 1, 1),
        )
        session.commit()
    engine.dispose()


def test_ultrack_db_hierarchy_slider_uses_frame_cut_states(tmp_path, monkeypatch):
    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    _make_ultrack_db_with_equal_height_plateau(db_path)
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_hierarchy_slider.setValue(1)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda _path, _frame: "ok")

    widget._refresh_ultrack_db_browser()

    assert widget.ultrack_db_hierarchy_slider.minimum() == 0
    assert widget.ultrack_db_hierarchy_slider.maximum() == 2
    assert widget.ultrack_db_hierarchy_slider.value() == 1
    assert "1" in widget.ultrack_db_height_lbl.text()
    assert "10.00" in widget.ultrack_db_height_lbl.text()
    assert set(widget._ultrack_db_label_to_node_id.values()) == {101, 102}

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_hierarchy_cut_keeps_equal_height_intermediate_nodes(tmp_path):
    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    _make_ultrack_db_with_equal_height_plateau(db_path)
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    _labels, status, _probs, label_to_node_id, _node_id_to_label, _annotations = (
        widget._render_hierarchy_cut(db_path, frame=0, h_actual=10.0)
    )

    assert set(label_to_node_id.values()) == {101, 102}
    assert 100 not in label_to_node_id.values()
    assert 201 not in label_to_node_id.values()
    assert "2 segment(s)" in status

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_hierarchy_slider_states_eventually_show_equal_height_parent(tmp_path):
    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    _make_ultrack_db_with_equal_height_plateau(db_path)
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    states = widget._query_hierarchy_cut_states(
        db_path, db_path.stat().st_mtime_ns, frame=0
    )
    rendered_node_ids = []
    for index, state in enumerate(states):
        labels, status, _probs, label_to_node_id, _node_id_to_label, _annotations = (
            widget._render_hierarchy_cut_state(db_path, frame=0, state=state)
        )
        assert labels.size > 0
        assert f"i={index}" not in status
        rendered_node_ids.append(set(label_to_node_id.values()))

    assert rendered_node_ids == [
        {102, 201},
        {101, 102},
        {100},
    ]
    assert {100, 101, 102, 201} == set().union(*rendered_node_ids)

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_hierarchy_slider_clamps_when_cut_states_shrink(tmp_path, monkeypatch):
    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    module = sys.modules[widget_class.__module__]
    widget = widget_class(viewer)
    calls = []

    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_hierarchy_slider.setValue(3)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda _path, _frame: "ok")
    state_sets = [
        (
            module._HierarchyCutState((1,), 0.1),
            module._HierarchyCutState((2,), 0.3),
            module._HierarchyCutState((3,), 0.5),
            module._HierarchyCutState((4,), 0.7),
        ),
        (
            module._HierarchyCutState((5,), 0.2),
            module._HierarchyCutState((6,), 0.6),
        ),
    ]

    def _states(*_args):
        return state_sets[0]

    def _render(_db_path, _frame, state):
        calls.append(state.height)
        return np.zeros((5, 5), dtype=np.uint32), "ok"

    monkeypatch.setattr(widget, "_query_hierarchy_cut_states", _states)
    monkeypatch.setattr(widget, "_render_hierarchy_cut_state", _render)
    widget._refresh_ultrack_db_browser()

    state_sets.pop(0)
    widget._ultrack_db_preview_cache.clear()
    widget._refresh_ultrack_db_browser()

    assert widget.ultrack_db_hierarchy_slider.maximum() == 1
    assert widget.ultrack_db_hierarchy_slider.value() == 1
    assert "1" in widget.ultrack_db_height_lbl.text()
    assert "0.60" in widget.ultrack_db_height_lbl.text()
    assert calls == [0.7, 0.6]

    widget.deleteLater()
    viewer.close()
