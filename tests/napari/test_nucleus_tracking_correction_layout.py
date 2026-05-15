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
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def _make_viewer():
    app = QApplication.instance() or QApplication([])
    viewer = napari.Viewer(show=False)
    return app, viewer


def _install_import_stubs() -> None:
    src_root = Path(__file__).resolve().parents[2] / "src" / "cellflow"
    package_root = src_root / "napari"

    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    sys.modules["cellflow.napari"] = napari_pkg

    tracking_pkg = types.ModuleType("cellflow.tracking_ultrack")
    tracking_pkg.__path__ = [str(src_root / "tracking_ultrack")]
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
            self.bias = 0.0
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
        "cellflow.tracking_ultrack.corrections": {
            "Correction": __import__(
                "cellflow.tracking_ultrack.corrections",
                fromlist=["Correction"],
            ).Correction,
        },
        "cellflow.tracking_ultrack.db_build": {
            "apply_annotations_and_score": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.export": {"export_tracked_labels": lambda *args, **kwargs: None},
        "cellflow.tracking_ultrack.ingest": {
            "ingest_hypotheses_to_db": lambda *args, **kwargs: None,
            "_select_solver": lambda: "CBC",
            "_build_ultrack_config": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.linking": {"run_linking": lambda *args, **kwargs: iter(())},
        "cellflow.tracking_ultrack.multi_threshold": {
            "build_ultrack_database_from_sources": lambda *args, **kwargs: None,
            "preview_ultrack_source_stack_frame": lambda *args, **kwargs: (None, None, 0, []),
            "write_ultrack_source_stacks": lambda *args, **kwargs: [],
        },
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
        "cellflow.segmentation": {
            "apply_gamma": lambda logits, gamma: logits,
            "build_nucleus_averaged_maps": lambda *args, **kwargs: None,
            "build_consensus_boundary": lambda *args, **kwargs: (None, None),
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
        "cellflow.napari.cell_boundary_workflow_widget": {
            "CellBoundaryWorkflowWidget": _StubWidget,
        },
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
        "cellflow.napari.cell_boundary_workflow_widget",
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

    assert widget.nucleus_section.title == "Nucleus Segmentation & Tracking"
    assert widget.nucleus_section._toggle.text() == "Nucleus Segmentation && Tracking"

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


def test_main_widget_embeds_nls_classification_inside_analysis_stage():
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "nls_classification_section")
    assert not hasattr(widget, "nls_classification_widget")
    assert widget.analysis_section.title == "Analysis"

    widget.deleteLater()
    viewer.close()


def test_main_widget_includes_meta_source_browser_after_analysis(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    project_root = tmp_path / "study"
    project_root.mkdir()
    widget.path_label.setText(str(project_root))
    widget.pos_spin.setValue(3)
    widget._refresh_all()

    assert widget.meta_section.title == "Meta Analyzer"
    assert widget.meta_source_browser.refreshed_pos_dir == project_root
    assert widget.scroll_layout.indexOf(widget.analysis_section) < widget.scroll_layout.indexOf(
        widget.meta_section
    )

    widget.deleteLater()
    viewer.close()


def test_nucleus_workflow_uses_unified_file_status_and_progress_widgets():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "input_files")
    assert hasattr(widget, "_files_widget")
    assert hasattr(widget, "pipeline_status_lbl")
    assert hasattr(widget, "pipeline_progress_bar")

    removed_stage_attrs = {
        "contour_input_files",
        "contour_output_files",
        "contour_status_lbl",
        "build_progress_bar",
        "db_gen_input_files",
        "db_gen_output_files",
        "db_gen_status_lbl",
        "db_gen_progress_bar",
        "ultrack_input_files",
        "ultrack_output_files",
        "ultrack_status_lbl",
        "ultrack_progress_bar",
    }
    for attr in removed_stage_attrs:
        assert not hasattr(widget, attr)

    assert widget.pipeline_progress_bar.isVisible() is False

    texts = _label_texts(widget)
    assert "Cellprob:" in texts
    assert "Z:" in texts
    assert "Contour:" in texts
    assert "Foreground:" in texts

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

    widget.deleteLater()
    viewer.close()
    _app.processEvents()  # flush deferred deletion before next test


def test_main_widget_no_longer_includes_track_conditioned_cell_boundary_section():
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "cell_boundary_section")
    assert not hasattr(widget, "cell_boundary_workflow_widget")
    assert widget.scroll_layout.indexOf(widget.cell_section) < widget.scroll_layout.indexOf(
        widget.analysis_section
    )

    state = widget.get_state()
    assert "cell_boundary" not in state
    widget.set_state({"cell_boundary": {"mode": "track-conditioned"}})

    widget.deleteLater()
    viewer.close()


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


def test_main_widget_top_level_sections_use_unnumbered_mocha_titles():
    _app, viewer = _make_viewer()
    widget_class = _load_main_widget_class()
    from cellflow.napari.ui_style import semantic_color

    widget = widget_class(viewer)

    expected = {
        widget.data_section: "Project Status",
        widget.prep_section: "Data Preparation",
        widget.cellpose_section: "Cellpose",
        widget.nucleus_section: "Nucleus Segmentation & Tracking",
        widget.cell_section: "Cell Segmentation",
        widget.analysis_section: "Analysis",
        widget.meta_section: "Meta Analyzer",
    }

    for section, title in expected.items():
        assert section.title == title
        toggle = section.findChild(QToolButton)
        assert toggle is not None
        assert toggle.text() == title.replace("&", "&&")
        assert f"color: {semantic_color('stage', 0)};" in toggle.styleSheet()

    widget.deleteLater()
    viewer.close()


def test_nucleus_workflow_uses_semantic_colors_by_role_and_level():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    from cellflow.napari.ui_style import semantic_color

    widget = widget_class(viewer)

    files_toggle = widget.findChild(QToolButton)
    assert files_toggle is not None
    assert f"color: {semantic_color('stage', 1)};" in files_toggle.styleSheet()

    segmentation_toggle = next(
        toggle
        for toggle in widget.findChildren(QToolButton)
        if toggle.text() == "Segmentation Input Parameters"
    )
    assert segmentation_toggle is not None
    assert f"color: {semantic_color('params', 1)};" in segmentation_toggle.styleSheet()
    assert "Segmentation Inputs" not in {
        label.text() for label in widget.segmentation_inputs_section.findChildren(QLabel)
    }
    assert widget.run_db_gen_btn.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert semantic_color("indicators", 0) in widget.pipeline_status_lbl.styleSheet()

    widget.deleteLater()
    viewer.close()


def test_tracking_correction_shell_exposes_stable_section_attributes():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    # tracking_correction_section wrapper removed — sections are now top-level
    assert not hasattr(widget, "tracking_correction_section")
    assert widget.segmentation_inputs_section.title == "Segmentation Input Parameters"
    assert widget.tracking_ultrack_section.title == "Ultrack Parameters"
    assert widget.correction_mode_section.title == "Correction"
    assert widget.ultrack_db_browser_section.title == "Database Browser"
    assert widget.layout().indexOf(widget.tracking_ultrack_section) < widget.layout().indexOf(
        widget.ultrack_db_browser_section
    )
    assert widget.layout().indexOf(widget.ultrack_db_browser_section) < widget.layout().indexOf(
        widget.correction_mode_section
    )
    assert widget.correction_shortcuts_section.title == "Correction Shortcuts"
    assert widget.segmentation_inputs_section.is_expanded is True
    assert widget.tracking_ultrack_section.is_expanded is False
    assert widget.correction_mode_section.is_expanded is False
    assert widget.correction_shortcuts_section.is_expanded is True
    assert widget.correction_shortcuts_section.findChildren(QScrollArea) == []
    correction_inner = widget.correction_mode_section._content_frame.layout().itemAt(0).widget()
    correction_layout = correction_inner.layout()
    assert correction_layout.indexOf(widget.correction_widget) != -1
    assert correction_layout.indexOf(widget.correction_shortcuts_section) != -1
    assert (
        correction_layout.indexOf(widget.correction_shortcuts_section)
        < correction_layout.indexOf(widget.correction_widget)
    )

    correction_button_texts = {
        button.text()
        for button in widget.correction_mode_section.findChildren(QPushButton)
    }
    ultrack_button_texts = {
        button.text()
        for button in widget.tracking_ultrack_section.findChildren(QPushButton)
    }
    ultrack_checkbox_texts = {
        checkbox.text() for checkbox in widget.tracking_ultrack_section.findChildren(QCheckBox)
    }

    assert "Activate Correction" not in correction_button_texts
    assert "Save tracked" in correction_button_texts
    assert "Load Labels" not in correction_button_texts
    assert "Save Labels" not in correction_button_texts
    assert "Extend selected" not in correction_button_texts
    assert "Retrack selected" not in correction_button_texts
    assert "Reassign ID" in correction_button_texts
    assert "Remove unvalidated" in correction_button_texts
    assert "Clean Holes / Islands" not in correction_button_texts
    assert "◀ Extend (A)" in correction_button_texts
    assert "Extend (D) ▶" in correction_button_texts
    assert "◀ Retrack (Q)" in correction_button_texts
    assert "Retrack (E) ▶" in correction_button_texts
    shortcut_keys = {
        shortcut.key().toString(QKeySequence.SequenceFormat.PortableText)
        for shortcut in widget.findChildren(QShortcut)
    }
    assert {"A", "D", "Q", "E"} <= shortcut_keys
    assert "Save tracked" not in ultrack_button_texts
    assert "Load Labels" not in ultrack_button_texts
    assert "Reassign ID" not in ultrack_button_texts
    assert "Resolve from validated" not in ultrack_checkbox_texts

    widget.deleteLater()
    viewer.close()


def test_tracking_correction_action_buttons_expand_horizontally():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    tracked_buttons = [
        widget.run_ultrack_btn,
        widget.extend_back_btn,
        widget.extend_fwd_btn,
        widget.retrack_back_btn,
        widget.retrack_fwd_btn,
        widget.save_tracked_btn,
        widget.reassign_ids_btn,
        widget.remove_unvalidated_btn,
    ]

    for button in tracked_buttons:
        assert button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding

    widget.deleteLater()
    viewer.close()


def test_correction_activation_button_is_top_level_and_section_header_is_passive():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    top_level_layout = widget.layout()
    assert top_level_layout.indexOf(widget.correction_active_btn) != -1
    assert top_level_layout.indexOf(widget.correction_active_btn) < top_level_layout.indexOf(
        widget.correction_mode_section
    )
    assert widget.correction_active_btn not in widget.correction_mode_section.findChildren(
        QPushButton
    )

    assert widget.correction_mode_section.is_expanded is False
    assert widget.correction_mode_section._content_frame.isVisible() is False
    assert widget.correction_mode_section._toggle.isVisible() is False
    assert widget.correction_mode_section._toggle.isEnabled() is False

    widget.deleteLater()
    viewer.close()


def test_ultrack_section_has_no_legacy_resolve_route_and_uses_pipeline_status():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "ultrack_route_check")
    assert widget.pipeline_status_lbl.text() == ""
    assert widget.pipeline_progress_bar.isVisible() is False

    widget.deleteLater()
    viewer.close()


def test_nucleus_workflow_status_labels_use_unified_pipeline_status():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "status_lbl")
    assert hasattr(widget, "pipeline_status_lbl")
    assert widget.pipeline_status_lbl.text() == ""

    assert widget.ultrack_db_section_status_lbl.text() == ""
    assert widget.ultrack_db_section_status_lbl in widget.ultrack_db_browser_section.findChildren(QLabel)
    assert not hasattr(widget, "correction_section")
    assert widget.correction_status_lbl.text() == ""

    widget._status("Ultrack stayed local")

    assert widget.pipeline_status_lbl.text() == "Ultrack stayed local"
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


def test_db_gen_exposes_threshold_sweep_controls_and_ignores_progressive_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    try:
        assert not hasattr(widget, "db_gen_progressive_check")
        assert "Progressive inputs" not in {
            checkbox.text() for checkbox in widget.findChildren(QCheckBox)
        }
        assert widget.db_gen_threshold_min_spin.value() == pytest.approx(0.1)
        assert widget.db_gen_threshold_max_spin.value() == pytest.approx(0.5)
        assert widget.db_gen_threshold_step_spin.value() == pytest.approx(0.1)
        np.testing.assert_allclose(
            widget._db_gen_thresholds_from_controls(),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        )

        state = widget.get_state()
        assert "progressive" not in state["db_generation"]
        widget.set_state({"db_generation": {"progressive": True}})
        assert "progressive" not in widget.get_state()["db_generation"]
    finally:
        widget.deleteLater()
        viewer.close()
        for module_name in tuple(sys.modules):
            if module_name == "cellflow.tracking_ultrack" or module_name.startswith(
                "cellflow.tracking_ultrack."
            ):
                sys.modules.pop(module_name, None)


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


def test_contour_maps_panel_exposes_source_stack_artifacts_without_masks_or_fg_threshold():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "foreground_section")
    assert not hasattr(widget, "contour_fg_threshold_spin")

    file_text = " ".join(label.text() for label in widget._files_widget.findChildren(QLabel))
    assert "1_cellpose/nucleus_prob_3dt.tif" in file_text
    assert "1_cellpose/nucleus_dp_3dt.tif" in file_text
    assert "2_nucleus/contours.tif" in file_text
    assert "2_nucleus/foreground_scores.tif" in file_text
    assert "2_nucleus/contour_sources.tif" in file_text
    assert "2_nucleus/foreground_sources.tif" in file_text
    assert "2_nucleus/foreground_masks.tif" not in file_text

    widget.deleteLater()
    viewer.close()


def test_nucleus_pipeline_files_expose_db_generation_and_ultrack_rows():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    file_text = " ".join(label.text() for label in widget._files_widget.findChildren(QLabel))

    assert "2_nucleus/contours.tif" in file_text
    assert "2_nucleus/foreground_scores.tif" in file_text
    assert "2_nucleus/contour_sources.tif" in file_text
    assert "2_nucleus/foreground_sources.tif" in file_text
    assert "2_nucleus/foreground_masks.tif" not in file_text
    assert "1_cellpose/nucleus_prob_zavg.tif" in file_text
    assert "2_nucleus/ultrack_workdir/data.db" in file_text
    assert "2_nucleus/tracked_labels.tif" in file_text

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
    tifffile.imwrite(pos_dir / "2_nucleus" / "contours.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_scores.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "2_nucleus" / "contour_sources.tif", np.zeros((1, 1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_sources.tif", np.zeros((1, 1, 4, 4), dtype=np.uint8))
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", np.zeros((1, 4, 4), dtype=np.uint32))
    (pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db").write_bytes(b"sqlite placeholder")

    widget.refresh(pos_dir)

    texts = _label_texts(widget)
    assert texts.count("✓") >= 9
    assert "missing" in texts
    assert len(widget._files_widget._rows) == 11

    widget.deleteLater()
    viewer.close()


def test_nucleus_stage_file_load_buttons_load_files_into_viewer(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    contours = np.ones((2, 4, 4), dtype=np.float32)
    scores = np.ones((2, 4, 4), dtype=np.float32) * 0.5
    contour_sources = np.ones((1, 2, 4, 4), dtype=np.float32) * 0.25
    foreground_sources = np.ones((1, 2, 4, 4), dtype=np.uint8)
    labels = np.ones((2, 4, 4), dtype=np.uint32)
    import tifffile

    tifffile.imwrite(pos_dir / "2_nucleus" / "contours.tif", contours)
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_scores.tif", scores)
    tifffile.imwrite(pos_dir / "2_nucleus" / "contour_sources.tif", contour_sources)
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_sources.tif", foreground_sources)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", labels)

    widget.refresh(pos_dir)

    for row in widget._files_widget._rows:
        if row._full_path is not None:
            row._on_load_clicked()

    assert "2_nucleus_contours" in viewer.layers
    assert "2_nucleus_foreground_scores" in viewer.layers
    assert "2_nucleus_contour_sources" in viewer.layers
    assert "2_nucleus_foreground_sources" in viewer.layers
    assert "2_nucleus_foreground_masks" not in viewer.layers
    assert "2_nucleus_tracked_labels" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["2_nucleus_contours"].data, contours)
    np.testing.assert_array_equal(viewer.layers["2_nucleus_foreground_scores"].data, scores)
    np.testing.assert_array_equal(viewer.layers["2_nucleus_contour_sources"].data, contour_sources)
    np.testing.assert_array_equal(viewer.layers["2_nucleus_foreground_sources"].data, foreground_sources)
    np.testing.assert_array_equal(viewer.layers["2_nucleus_tracked_labels"].data, labels)

    widget.deleteLater()
    viewer.close()


def test_removed_segmentation_source_controls_are_not_persisted_or_restored():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    state = widget.get_state()
    for attr in (
        "contour_fg_threshold_spin",
        "cp_min_spin",
        "cp_max_spin",
        "cp_step_spin",
        "cp_gamma_min_spin",
        "cp_gamma_max_spin",
        "cp_gamma_step_spin",
        "contour_flow_threshold_spin",
        "save_source_check",
    ):
        assert not hasattr(widget, attr)
    assert "cellprob" not in state
    assert "save_source" not in state
    assert "foreground_mask" not in state

    widget.set_state({
        **state,
        "save_source": True,
        "cellprob": {
            "min": -3.0,
            "max": 0.0,
            "step": 1.0,
            "gamma_min": 0.5,
            "gamma_max": 2.0,
            "gamma_step": 0.5,
            "flow_threshold": 1.7,
            "foreground_threshold": 0.73,
        },
        "foreground_mask": {"threshold": 0.1, "gamma": 2.0, "niter": 50},
    })

    new_state = widget.get_state()
    assert "cellprob" not in new_state
    assert "save_source" not in new_state
    assert "foreground_mask" not in new_state

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

    for attr in (
        "ultrack_min_area_spin",
        "ultrack_max_dist_spin",
        "ultrack_max_neighbors_spin",
        "ultrack_linking_mode_combo",
        "ultrack_iou_weight_spin",
    ):
        assert not hasattr(widget, attr)

    ultrack_labels = {
        label.text()
        for label in widget.tracking_ultrack_section.findChildren(QLabel)
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

    tracking_labels = {
        label.text()
        for label in widget.tracking_ultrack_section.findChildren(QLabel)
    }

    assert {
        "DB Generation — Candidates",
        "DB Generation — Linking",
        "DB Generation — Scoring",
        "DB Generation — Validated Seed Prior",
        "Ultrack — Track Scope",
        "Ultrack — Event Penalties",
        "Ultrack — Solver",
    } <= tracking_labels

    widget.deleteLater()
    viewer.close()


def test_ultrack_solver_bias_control_updates_config_and_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert hasattr(widget, "ultrack_bias_spin")
    assert widget.ultrack_bias_spin.value() == pytest.approx(0.0)

    widget.ultrack_bias_spin.setValue(-0.5)
    cfg = widget._ultrack_config_from_controls()

    assert cfg.bias == pytest.approx(-0.5)
    assert widget.get_state()["ultrack"]["bias"] == pytest.approx(-0.5)

    widget.set_state({"ultrack": {"bias": -0.25}})

    assert widget.ultrack_bias_spin.value() == pytest.approx(-0.25)

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

    assert not hasattr(widget, "ultrack_terminal_btn")
    assert not hasattr(widget, "_on_ultrack_terminal")

    widget.deleteLater()
    viewer.close()


def test_db_generation_terminal_controls_are_removed():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "db_gen_terminal_btn")
    assert not hasattr(widget, "_on_db_gen_terminal")

    widget.deleteLater()
    viewer.close()


def test_tracking_correction_widget_allows_horizontal_scrolling_when_narrow():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    widget.segmentation_inputs_section.expand()
    widget.tracking_ultrack_section.expand()
    widget.correction_mode_section.expand()
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

    widget.tracking_ultrack_section.expand()
    widget.correction_mode_section.expand()
    widget.correction_shortcuts_section.expand()
    _app.processEvents()

    # The old broken layout measured well above 600px because the shortcut
    # help was unwrapped and multiple long controls forced wide horizontal
    # rows. Keep a generous ceiling so the test remains stable across Qt
    # styles while still catching a return to the original oversized layout.
    assert widget.minimumSizeHint().width() < 560

    widget.deleteLater()
    viewer.close()


def test_contour_maps_parameters_expand_when_narrow():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    params_section = widget.segmentation_inputs_section
    params_section.expand()
    _app.processEvents()

    assert params_section.findChildren(QScrollArea) == []

    for spin in (
        widget.db_gen_threshold_min_spin,
        widget.db_gen_threshold_max_spin,
        widget.db_gen_threshold_step_spin,
    ):
        assert spin.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed

    host = QWidget()
    host_layout = QVBoxLayout(host)
    host_layout.setContentsMargins(0, 0, 0, 0)
    host_layout.addWidget(widget)
    host.resize(180, 320)
    host.show()
    _app.processEvents()

    for button in (
        widget.preview_contour_btn,
        widget.build_btn,
    ):
        assert button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding

    texts = _label_texts(params_section)
    assert "Contour — Cellprob Sweep" not in texts
    assert "Contour — Gamma Averaging" not in texts
    assert "Contour — Output" not in texts
    assert "Segmentation Inputs" not in texts

    host.deleteLater()
    viewer.close()


def test_nucleus_contours_path_prefers_contours_tif_over_contour_maps_tif(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "2_nucleus" / "contours.tif").touch()
    (pos_dir / "2_nucleus" / "contour_maps.tif").touch()
    widget._pos_dir = pos_dir

    assert widget._contours_path() == pos_dir / "2_nucleus" / "contours.tif"
    assert widget._contour_maps_path() == pos_dir / "2_nucleus" / "contours.tif"

    widget.deleteLater()
    viewer.close()


def test_nucleus_contours_path_falls_back_to_contour_maps_tif(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "2_nucleus" / "contour_maps.tif").touch()
    widget._pos_dir = pos_dir

    assert widget._contours_path() == pos_dir / "2_nucleus" / "contour_maps.tif"

    widget.deleteLater()
    viewer.close()


def test_nucleus_source_stack_path_helpers_point_to_stage_b_artifacts(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    widget._pos_dir = pos_dir

    assert widget._contours_path() == pos_dir / "2_nucleus" / "contours.tif"
    assert widget._contour_sources_path() == pos_dir / "2_nucleus" / "contour_sources.tif"
    assert widget._foreground_sources_path() == pos_dir / "2_nucleus" / "foreground_sources.tif"
    assert widget._foreground_scores_path() == pos_dir / "2_nucleus" / "foreground_scores.tif"

    widget.deleteLater()
    viewer.close()


def test_contour_maps_build_writes_source_stacks_from_contours_and_foreground_scores(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    widget._pos_dir = pos_dir
    widget.db_gen_threshold_min_spin.setValue(0.2)
    widget.db_gen_threshold_max_spin.setValue(0.4)
    widget.db_gen_threshold_step_spin.setValue(0.2)
    widget.source_foreground_threshold_min_spin.setValue(0.2)
    widget.source_foreground_threshold_max_spin.setValue(0.4)
    widget.source_foreground_threshold_step_spin.setValue(0.2)

    import tifffile

    tifffile.imwrite(pos_dir / "2_nucleus" / "contours.tif", np.zeros((2, 3, 3), dtype=np.float32))
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_scores.tif", np.ones((2, 3, 3), dtype=np.float32))
    calls = []

    def fake_write(*args, **kwargs):
        calls.append((args, kwargs))
        (pos_dir / "2_nucleus" / "contour_sources.tif").touch()
        (pos_dir / "2_nucleus" / "foreground_sources.tif").touch()
        return [{"contour_threshold": 0.2, "foreground_threshold": 0.2}]

    monkeypatch.setattr(module, "write_ultrack_source_stacks", fake_write)

    widget._on_build_contour_maps()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:4] == (
        pos_dir / "2_nucleus" / "contours.tif",
        pos_dir / "2_nucleus" / "foreground_scores.tif",
        pos_dir / "2_nucleus" / "contour_sources.tif",
        pos_dir / "2_nucleus" / "foreground_sources.tif",
    )
    np.testing.assert_allclose(kwargs["contour_thresholds"], np.array([0.2, 0.4]))
    np.testing.assert_allclose(kwargs["foreground_thresholds"], np.array([0.2, 0.4]))
    assert "Ultrack source stacks built" in widget.pipeline_status_lbl.text()
    assert widget.pipeline_progress_bar.isVisible() is False

    widget.deleteLater()
    viewer.close()


def test_source_stack_build_uses_independent_contour_and_foreground_thresholds(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    widget._pos_dir = pos_dir
    widget.source_contour_threshold_min_spin.setValue(0.2)
    widget.source_contour_threshold_max_spin.setValue(0.4)
    widget.source_contour_threshold_step_spin.setValue(0.2)
    widget.source_foreground_threshold_min_spin.setValue(0.3)
    widget.source_foreground_threshold_max_spin.setValue(0.7)
    widget.source_foreground_threshold_step_spin.setValue(0.4)

    import tifffile

    tifffile.imwrite(pos_dir / "2_nucleus" / "contours.tif", np.zeros((2, 3, 3), dtype=np.float32))
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_scores.tif", np.ones((2, 3, 3), dtype=np.float32))
    calls = []

    def fake_write(*args, **kwargs):
        calls.append((args, kwargs))
        (pos_dir / "2_nucleus" / "contour_sources.tif").touch()
        (pos_dir / "2_nucleus" / "foreground_sources.tif").touch()
        return [
            {"contour_threshold": 0.2, "foreground_threshold": 0.3},
            {"contour_threshold": 0.2, "foreground_threshold": 0.7},
            {"contour_threshold": 0.4, "foreground_threshold": 0.3},
            {"contour_threshold": 0.4, "foreground_threshold": 0.7},
        ]

    monkeypatch.setattr(module, "write_ultrack_source_stacks", fake_write)

    widget._on_build_contour_maps()

    assert len(calls) == 1
    _, kwargs = calls[0]
    np.testing.assert_allclose(kwargs["contour_thresholds"], np.array([0.2, 0.4]))
    np.testing.assert_allclose(kwargs["foreground_thresholds"], np.array([0.3, 0.7]))
    assert "4 sources" in widget.pipeline_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_contour_maps_preview_displays_source_stack_planes(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    widget._pos_dir = pos_dir
    widget.db_gen_threshold_min_spin.setValue(0.2)
    widget.db_gen_threshold_max_spin.setValue(0.2)
    widget.db_gen_threshold_step_spin.setValue(0.2)
    widget.source_foreground_threshold_min_spin.setValue(0.2)
    widget.source_foreground_threshold_max_spin.setValue(0.2)
    widget.source_foreground_threshold_step_spin.setValue(0.2)

    import tifffile

    prob = np.zeros((2, 1, 3, 3), dtype=np.float32)
    dp = np.zeros((2, 1, 2, 3, 3), dtype=np.float32)
    contour_sources = np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3)
    foreground_sources = (contour_sources > 8).astype(np.uint8)
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif", dp)
    write_calls = []

    def fail_write(*args, **kwargs):
        write_calls.append((args, kwargs))
        raise AssertionError("Preview must not write source-stack TIFFs")

    def fake_build_consensus_boundary(prob_frame, dp_frame, thresholds, **kwargs):
        assert prob_frame.shape == (1, 3, 3)
        assert dp_frame.shape == (1, 2, 3, 3)
        np.testing.assert_allclose(thresholds, [-3.0, -2.0, -1.0, 0.0])
        return np.zeros((3, 3), dtype=np.float32), np.ones((3, 3), dtype=np.float32)

    def fake_preview(*args, **kwargs):
        np.testing.assert_allclose(args[0], np.zeros((1, 3, 3), dtype=np.float32))
        np.testing.assert_allclose(args[1], np.ones((1, 3, 3), dtype=np.float32))
        assert kwargs["frame_index"] == 0
        np.testing.assert_allclose(kwargs["contour_thresholds"], np.array([0.2]))
        np.testing.assert_allclose(kwargs["foreground_thresholds"], np.array([0.2]))
        return contour_sources[:, 0], foreground_sources[:, 0], 0, [
            {"contour_threshold": 0.2, "foreground_threshold": 0.2}
        ]

    monkeypatch.setattr(module, "write_ultrack_source_stacks", fail_write)
    monkeypatch.setattr(module, "build_consensus_boundary", fake_build_consensus_boundary)
    monkeypatch.setattr(module, "preview_ultrack_source_stack_frame", fake_preview)

    widget._on_preview_contour_maps()

    expected_contours = np.zeros_like(contour_sources)
    expected_contours[:, 0] = contour_sources[:, 0]
    expected_foreground = np.zeros_like(foreground_sources)
    expected_foreground[:, 0] = foreground_sources[:, 0]
    np.testing.assert_allclose(
        viewer.layers["Contour Map: Nucleus"].data,
        expected_contours,
    )
    np.testing.assert_array_equal(
        viewer.layers["Foreground Score: Nucleus"].data,
        expected_foreground,
    )
    assert isinstance(viewer.layers["Foreground Score: Nucleus"], napari.layers.Labels)
    assert "Preview segmentation inputs" in widget.pipeline_status_lbl.text()
    assert write_calls == []
    assert not (pos_dir / "2_nucleus" / "contour_sources.tif").exists()
    assert not (pos_dir / "2_nucleus" / "foreground_sources.tif").exists()

    widget.deleteLater()
    viewer.close()


def test_preview_segmentation_inputs_uses_full_time_layers_and_loads_nucleus_zavg(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "0_input").mkdir(parents=True)
    (pos_dir / "1_cellpose").mkdir()
    (pos_dir / "2_nucleus").mkdir()
    widget._pos_dir = pos_dir
    widget.map_cellprob_min_spin.setValue(-2.0)
    widget.map_cellprob_max_spin.setValue(-2.0)
    widget.map_cellprob_step_spin.setValue(1.0)
    widget.map_z_start_spin.setValue(0)
    widget.map_z_stop_spin.setValue(-1)
    widget.map_z_step_spin.setValue(1)
    widget.source_contour_threshold_min_spin.setValue(0.2)
    widget.source_contour_threshold_max_spin.setValue(0.2)
    widget.source_contour_threshold_step_spin.setValue(0.2)
    widget.source_foreground_threshold_min_spin.setValue(0.3)
    widget.source_foreground_threshold_max_spin.setValue(0.3)
    widget.source_foreground_threshold_step_spin.setValue(0.3)

    import tifffile

    prob = np.zeros((3, 2, 4, 5), dtype=np.float32)
    dp = np.zeros((3, 2, 2, 4, 5), dtype=np.float32)
    zavg = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif", dp)
    tifffile.imwrite(pos_dir / "0_input" / "nucleus_zavg.tif", zavg)

    viewer.add_image(np.zeros((3, 4, 5), dtype=np.float32), name="time scaffold")
    viewer.dims.current_step = (1, 0, 0)

    def fake_build_consensus_boundary(prob_frame, dp_frame, thresholds, **kwargs):
        assert prob_frame.shape == (2, 4, 5)
        assert dp_frame.shape == (2, 2, 4, 5)
        np.testing.assert_allclose(thresholds, [-2.0])
        return np.full((4, 5), 4.0, dtype=np.float32), np.full((4, 5), 5.0, dtype=np.float32)

    def fake_preview(contours, foreground_scores, **kwargs):
        np.testing.assert_allclose(contours, np.full((1, 4, 5), 4.0, dtype=np.float32))
        np.testing.assert_allclose(foreground_scores, np.full((1, 4, 5), 5.0, dtype=np.float32))
        assert kwargs["frame_index"] == 0
        return (
            np.full((1, 4, 5), 7.0, dtype=np.float32),
            np.full((1, 4, 5), 1.0, dtype=np.uint8),
            0,
            [{"contour_threshold": 0.2, "foreground_threshold": 0.3}],
        )

    monkeypatch.setattr(module, "build_consensus_boundary", fake_build_consensus_boundary)
    monkeypatch.setattr(module, "preview_ultrack_source_stack_frame", fake_preview)

    widget._on_preview_contour_maps()

    assert "Nucleus z-avg" in viewer.layers
    np.testing.assert_allclose(viewer.layers["Nucleus z-avg"].data, zavg)
    contour_layer = viewer.layers["Contour Map: Nucleus"].data
    foreground_preview_layer = viewer.layers["Foreground Score: Nucleus"]
    assert isinstance(foreground_preview_layer, napari.layers.Labels)
    foreground_layer = foreground_preview_layer.data
    assert contour_layer.shape == (1, 3, 4, 5)
    assert foreground_layer.shape == (1, 3, 4, 5)
    np.testing.assert_allclose(contour_layer[:, 0], 0)
    np.testing.assert_allclose(contour_layer[:, 1], 7)
    np.testing.assert_allclose(contour_layer[:, 2], 0)
    np.testing.assert_array_equal(foreground_layer[:, 0], 0)
    np.testing.assert_array_equal(foreground_layer[:, 1], 1)
    np.testing.assert_array_equal(foreground_layer[:, 2], 0)

    widget.deleteLater()
    viewer.close()


def test_preview_segmentation_inputs_reads_time_from_preview_time_axis(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    widget._pos_dir = pos_dir
    widget.map_cellprob_min_spin.setValue(-2.0)
    widget.map_cellprob_max_spin.setValue(-2.0)
    widget.map_cellprob_step_spin.setValue(1.0)
    widget.map_z_start_spin.setValue(0)
    widget.map_z_stop_spin.setValue(-1)
    widget.map_z_step_spin.setValue(1)
    widget.source_contour_threshold_min_spin.setValue(0.2)
    widget.source_contour_threshold_max_spin.setValue(0.2)
    widget.source_contour_threshold_step_spin.setValue(0.2)
    widget.source_foreground_threshold_min_spin.setValue(0.3)
    widget.source_foreground_threshold_max_spin.setValue(0.3)
    widget.source_foreground_threshold_step_spin.setValue(0.3)

    import tifffile

    prob = np.zeros((3, 1, 4, 5), dtype=np.float32)
    prob[2] = 2.0
    dp = np.zeros((3, 1, 2, 4, 5), dtype=np.float32)
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif", dp)

    viewer.add_image(
        np.zeros((2, 3, 4, 5), dtype=np.float32),
        name="Contour Map: Nucleus",
    )
    viewer.add_labels(
        np.zeros((2, 3, 4, 5), dtype=np.uint8),
        name="Foreground Score: Nucleus",
    )
    viewer.dims.current_step = (1, 2, 0, 0)

    def fake_build_consensus_boundary(prob_frame, dp_frame, thresholds, **kwargs):
        np.testing.assert_allclose(prob_frame, np.full((1, 4, 5), 2.0, dtype=np.float32))
        return np.full((4, 5), 4.0, dtype=np.float32), np.full((4, 5), 5.0, dtype=np.float32)

    def fake_preview(contours, foreground_scores, **kwargs):
        return (
            np.full((1, 4, 5), 7.0, dtype=np.float32),
            np.full((1, 4, 5), 1.0, dtype=np.uint8),
            0,
            [{"contour_threshold": 0.2, "foreground_threshold": 0.3}],
        )

    monkeypatch.setattr(module, "build_consensus_boundary", fake_build_consensus_boundary)
    monkeypatch.setattr(module, "preview_ultrack_source_stack_frame", fake_preview)

    widget._on_preview_contour_maps()

    assert "t=2" in widget.pipeline_status_lbl.text()
    np.testing.assert_allclose(viewer.layers["Contour Map: Nucleus"].data[:, 2], 7)
    np.testing.assert_array_equal(viewer.layers["Foreground Score: Nucleus"].data[:, 2], 1)

    widget.deleteLater()
    viewer.close()


def test_z_stop_minus_one_means_all_z_slices():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.map_z_start_spin.setValue(0)
    widget.map_z_stop_spin.setValue(-1)
    widget.map_z_step_spin.setValue(1)

    assert widget.map_z_stop_spin.value() == -1
    assert widget._map_z_indices_from_controls() is None

    widget.deleteLater()
    viewer.close()


def test_pipeline_actions_and_status_are_top_level_not_in_parameter_sections():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.preview_contour_btn.parentWidget() is widget
    assert widget.build_btn.parentWidget() is widget
    assert widget.pipeline_status_lbl.parentWidget() is widget
    assert widget.pipeline_progress_bar.parentWidget() is widget
    assert widget.run_db_gen_btn.parentWidget() is widget
    assert widget.run_ultrack_btn.parentWidget() is widget
    assert widget.cancel_btn.parentWidget() is widget

    assert widget.preview_contour_btn not in widget.segmentation_inputs_section.findChildren(QPushButton)
    assert widget.build_btn not in widget.segmentation_inputs_section.findChildren(QPushButton)
    assert widget.pipeline_status_lbl not in widget.segmentation_inputs_section.findChildren(QLabel)
    assert widget.pipeline_progress_bar not in widget.segmentation_inputs_section.findChildren(QProgressBar)
    assert widget.run_db_gen_btn not in widget.tracking_ultrack_section.findChildren(QPushButton)
    assert widget.run_ultrack_btn not in widget.tracking_ultrack_section.findChildren(QPushButton)
    assert widget.cancel_btn not in widget.tracking_ultrack_section.findChildren(QPushButton)

    widget.deleteLater()
    viewer.close()


def test_contour_terminal_controls_are_removed():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "contour_terminal_btn")
    assert not hasattr(widget, "_on_run_contour_terminal")

    widget.deleteLater()
    viewer.close()


def test_contour_filter_controls_and_actions_are_removed_from_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    removed_attrs = {
        "contour_filter_median_time_spin",
        "contour_filter_median_space_spin",
        "contour_filter_gauss_time_spin",
        "contour_filter_gauss_space_spin",
        "preview_contour_filter_btn",
        "run_contour_filter_btn",
        "_on_preview_contour_filter",
        "_on_run_contour_filter",
    }
    for attr in removed_attrs:
        assert not hasattr(widget, attr)

    widget.set_state({
        "contour_filter": {
            "median_time": 3,
            "median_space": 5,
            "gauss_time": 1.5,
            "gauss_space": 2.5,
        }
    })
    assert "contour_filter" not in widget.get_state()

    widget.deleteLater()
    viewer.close()


def test_source_stack_actions_use_explicit_source_language():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.preview_contour_btn.text() == "Preview Segmentation Inputs"
    assert widget.build_btn.text() == "Build Segmentation Inputs"
    assert "segmentation input source sweep" in widget.preview_contour_btn.toolTip().lower()
    assert "averaged maps" in widget.build_btn.toolTip().lower()
    assert "source-stack" in widget.run_db_gen_btn.toolTip().lower()

    widget.deleteLater()
    viewer.close()


def test_segmentation_inputs_expose_stage_a_map_builder_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.build_maps_btn.text() == "Build Segmentation Inputs"
    assert "averaged maps" in widget.build_maps_btn.toolTip()
    assert "source stacks" in widget.build_maps_btn.toolTip()
    assert widget.map_cellprob_min_spin.value() == -3.0
    assert widget.map_cellprob_max_spin.value() == 0.0
    assert widget.map_cellprob_step_spin.value() == 1.0
    assert widget.map_z_start_spin.value() == 0
    assert widget.map_z_stop_spin.value() == -1
    assert widget.map_z_step_spin.value() == 1
    assert widget.build_maps_btn not in widget.segmentation_inputs_section.findChildren(QPushButton)
    assert widget.build_btn not in widget.segmentation_inputs_section.findChildren(QPushButton)

    widget.deleteLater()
    viewer.close()


def test_build_maps_calls_stage_a_backend_without_building_sources(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    prob_path = pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif"
    dp_path = pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif"
    prob_path.touch()
    dp_path.touch()
    widget._pos_dir = pos_dir
    widget.map_cellprob_min_spin.setValue(-2.0)
    widget.map_cellprob_max_spin.setValue(0.0)
    widget.map_cellprob_step_spin.setValue(1.0)
    widget.map_z_start_spin.setValue(1)
    widget.map_z_stop_spin.setValue(3)
    widget.map_z_step_spin.setValue(2)
    calls = []
    source_calls = []

    class DummyReport:
        frames = 4

    def fake_build_maps(*args, **kwargs):
        calls.append((args, kwargs))
        (pos_dir / "2_nucleus" / "contours.tif").touch()
        (pos_dir / "2_nucleus" / "foreground_scores.tif").touch()
        return DummyReport()

    monkeypatch.setattr(module, "build_nucleus_averaged_maps", fake_build_maps)
    monkeypatch.setattr(
        module,
        "write_ultrack_source_stacks",
        lambda *args, **kwargs: source_calls.append((args, kwargs)),
    )

    widget._on_build_nucleus_maps()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:4] == (
        prob_path,
        dp_path,
        pos_dir / "2_nucleus" / "contours.tif",
        pos_dir / "2_nucleus" / "foreground_scores.tif",
    )
    np.testing.assert_allclose(kwargs["cellprob_thresholds"], np.array([-2.0, -1.0, 0.0]))
    assert kwargs["z_indices"] == [1, 3]
    assert source_calls == []
    assert "Averaged maps built" in widget.pipeline_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_stage_a_map_controls_round_trip_through_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.map_cellprob_min_spin.setValue(-4.0)
    widget.map_cellprob_max_spin.setValue(1.0)
    widget.map_cellprob_step_spin.setValue(0.5)
    widget.map_z_start_spin.setValue(2)
    widget.map_z_stop_spin.setValue(7)
    widget.map_z_step_spin.setValue(2)
    state = widget.get_state()

    widget.deleteLater()
    widget = widget_class(viewer)
    widget.set_state(state)

    assert widget.map_cellprob_min_spin.value() == pytest.approx(-4.0)
    assert widget.map_cellprob_max_spin.value() == pytest.approx(1.0)
    assert widget.map_cellprob_step_spin.value() == pytest.approx(0.5)
    assert widget.map_z_start_spin.value() == 2
    assert widget.map_z_stop_spin.value() == 7
    assert widget.map_z_step_spin.value() == 2

    widget.deleteLater()
    viewer.close()


def test_db_generation_spinboxes_expand_equally_in_the_top_grid():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.tracking_ultrack_section.expand()
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
    assert not hasattr(widget, "db_gen_terminal_btn")

    host.deleteLater()
    viewer.close()


def test_tracking_correction_restores_two_column_button_and_parameter_layouts():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.segmentation_inputs_section.expand()
    widget.tracking_ultrack_section.expand()
    widget.correction_mode_section.expand()
    widget.show()
    _app.processEvents()

    # DB gen parameters should present as two side-by-side columns
    assert widget.db_gen_min_area_spin.y() == widget.db_gen_max_area_spin.y()
    assert widget.db_gen_min_area_spin.x() < widget.db_gen_max_area_spin.x()

    # Paired correction actions should also sit side-by-side
    assert widget.extend_back_btn.y() == widget.extend_fwd_btn.y()
    assert widget.extend_back_btn.x() < widget.extend_fwd_btn.x()
    assert widget.retrack_back_btn.y() == widget.retrack_fwd_btn.y()
    assert widget.validate_track_btn.y() == widget.anchor_here_btn.y()

    widget.deleteLater()
    viewer.close()


def test_correction_section_has_no_separate_resolve_action_group():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    correction_button_texts = {
        button.text()
        for button in widget.correction_mode_section.findChildren(QPushButton)
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

    assert widget.extend_params_section.title == "Extend / Retrack Parameters"
    assert widget.extend_params_section.is_expanded is False
    assert widget.retrack_params_section.title == "Extend / Retrack Parameters"
    assert widget.retrack_params_section is widget.extend_params_section
    assert widget.retrack_params_section.is_expanded is False
    assert widget.extend_max_dist_spin.value() == 40.0
    assert widget.extend_area_weight_spin.value() == 1.0
    assert widget.extend_iou_weight_spin.value() == 1.0
    assert widget.extend_distance_weight_spin.value() == 0.25
    assert widget.extend_overlap_penalty_spin.value() == 1.0
    assert widget.extend_greedy_overwrite_check.isChecked() is False
    assert widget.retrack_max_dist_spin.value() == 20.0
    assert not hasattr(widget, "extend_before_spin")
    assert not hasattr(widget, "extend_after_spin")
    assert not hasattr(widget, "retrack_window_spin")

    widget.deleteLater()
    viewer.close()


def test_validated_overlay_uses_green_fill_at_default_opacity_below_spotlight():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    viewer.add_labels(np.array([[[0, 1], [1, 0]]], dtype=np.uint8), name="Tracked: Nucleus")
    viewer.add_image(np.zeros((2, 2, 4), dtype=np.float32), name="[Correction] CellSpotlight", rgb=True)
    widget._add_validated_overlay(np.array([[[0, 1], [0, 0]]], dtype=np.uint8))

    layer = viewer.layers["[Correction] Validated: Nucleus"]
    color = layer.get_color(1)

    assert layer.contour == 0
    assert layer.opacity == 0.4
    assert np.allclose(color[:3], [0.0, 1.0, 0.0], atol=1e-6)
    assert color[3] == 1.0
    assert viewer.layers.index("[Correction] Validated: Nucleus") < viewer.layers.index("[Correction] CellSpotlight")

    widget.deleteLater()
    viewer.close()


def test_nucleus_correction_button_removes_unvalidated_label_instances(tmp_path):
    from cellflow.database.validation import validate_track

    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    pos_dir = tmp_path / "pos000"
    labels = np.zeros((2, 5, 5), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 7
    labels[0, 3:5, 3:5] = 9
    labels[1, 1:3, 1:3] = 7
    labels[1, 3:5, 3:5] = 11
    viewer.add_labels(labels, name="Tracked: Nucleus")
    validate_track(pos_dir, 7, [0])

    widget.refresh(pos_dir)
    widget.remove_unvalidated_btn.click()

    edited = np.asarray(viewer.layers["Tracked: Nucleus"].data)
    assert np.all(edited[0, 1:3, 1:3] == 7)
    assert not np.any(edited[0, 3:5, 3:5] == 9)
    assert not np.any(edited[1] == 7)
    assert not np.any(edited[1] == 11)
    assert "Removed unvalidated labels" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_validate_track_button_writes_validated_corrections_for_each_present_frame(tmp_path):
    from cellflow.database.validation import read_corrections, read_validated_tracks

    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget._pos_dir = pos_dir
    labels = np.zeros((3, 8, 8), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 5
    labels[2, 4:6, 4:6] = 5
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 5
    viewer.dims.current_step = (0, 0, 0)

    widget.validate_track_btn.click()

    corrections = read_corrections(pos_dir)
    assert [(c.cell_id, c.t, c.kind) for c in corrections] == [
        (5, 0, "validated"),
        (5, 2, "validated"),
    ]
    assert read_validated_tracks(pos_dir) == {5: {0, 2}}
    assert "validated track" in widget.correction_status_lbl.text().lower()

    widget.deleteLater()
    viewer.close()


def test_anchor_here_button_writes_anchor_correction_at_selected_cell_centroid(tmp_path):
    from cellflow.database.validation import read_corrections

    app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget._pos_dir = pos_dir
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[1, 4:6, 1:4] = 3
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 3
    viewer.dims.current_step = (1, 0, 0)

    widget.anchor_here_btn.click()
    app.processEvents()

    corrections = read_corrections(pos_dir)
    assert [(c.cell_id, c.t, c.kind) for c in corrections] == [(3, 1, "anchor")]
    assert corrections[0].y == pytest.approx(4.5)
    assert corrections[0].x == pytest.approx(2.0)
    anchor_layer = viewer.layers["[Correction] Anchors: Nucleus"]
    expected = np.zeros_like(labels, dtype=np.uint8)
    expected[1, 4:6, 1:4] = 1
    np.testing.assert_array_equal(anchor_layer.data, expected)
    assert anchor_layer.opacity == 0.4
    assert np.allclose(anchor_layer.get_color(1)[:3], [1.0, 1.0, 0.0], atol=1e-6)
    assert "anchor" in widget.correction_status_lbl.text().lower()

    widget.deleteLater()
    viewer.close()


def test_b_shortcut_writes_anchor_correction_at_selected_cell_centroid(tmp_path):
    from cellflow.database.validation import read_corrections

    app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget._pos_dir = pos_dir
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[1, 4:6, 1:4] = 3
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 3
    viewer.dims.current_step = (1, 0, 0)
    shortcut = next(
        shortcut
        for shortcut in widget.findChildren(QShortcut)
        if shortcut.key().toString(QKeySequence.SequenceFormat.PortableText) == "B"
    )

    shortcut.activated.emit()
    app.processEvents()

    corrections = read_corrections(pos_dir)
    assert [(c.cell_id, c.t, c.kind) for c in corrections] == [(3, 1, "anchor")]
    assert corrections[0].y == pytest.approx(4.5)
    assert corrections[0].x == pytest.approx(2.0)
    assert "[Correction] Anchors: Nucleus" in viewer.layers

    widget.deleteLater()
    viewer.close()


def test_b_shortcut_toggles_existing_anchor_off(tmp_path):
    from cellflow.database.validation import read_corrections

    app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget._pos_dir = pos_dir
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[1, 4:6, 1:4] = 3
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 3
    viewer.dims.current_step = (1, 0, 0)
    shortcut = next(
        shortcut
        for shortcut in widget.findChildren(QShortcut)
        if shortcut.key().toString(QKeySequence.SequenceFormat.PortableText) == "B"
    )

    shortcut.activated.emit()
    app.processEvents()
    shortcut.activated.emit()
    app.processEvents()

    assert read_corrections(pos_dir) == []
    anchor_layer = viewer.layers["[Correction] Anchors: Nucleus"]
    np.testing.assert_array_equal(anchor_layer.data, np.zeros_like(labels, dtype=np.uint8))
    assert "unanchored" in widget.correction_status_lbl.text().lower()

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


def test_correction_widget_preserves_selection_across_frame_change_for_relabel():
    _app, viewer = _make_viewer()
    widget_class = _load_correction_widget_class()
    widget = widget_class(viewer)
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[0, 1:4, 1:4] = 7
    labels[1, 4:7, 4:7] = 9
    layer = viewer.add_labels(labels, name="Tracked: Nucleus")

    widget.activate_layer(layer)
    widget.select_label(0, 7)
    viewer.dims.current_step = (1, 0, 0)
    widget._on_dims_change()

    assert widget._selected_label == 7
    assert widget._selected_t == 0

    event = types.SimpleNamespace(
        type="mouse_press",
        button=2,
        modifiers=(),
        position=(1, 5, 5),
    )
    callback_result = layer.mouse_drag_callbacks[-1](layer, event)
    if callback_result is not None:
        try:
            next(callback_result)
        except StopIteration:
            pass

    edited = np.asarray(layer.data)
    assert np.all(edited[1, 4:7, 4:7] == 7)
    assert "Relabelled" in widget._status.text()

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

    spotlight = viewer.layers["[Correction] CellSpotlight"]
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
    pytest.importorskip("ultrack")
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
    pytest.importorskip("ultrack")
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

def test_nucleus_workflow_uses_flat_source_stack_layout():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "contour_section")
    assert not hasattr(widget, "foreground_section")
    assert not hasattr(widget, "db_gen_section")
    assert not hasattr(widget, "ultrack_section")
    assert not hasattr(widget, "correction_section")
    assert not hasattr(widget, "tracking_correction_section")
    assert widget.ultrack_db_browser_section.title == "Database Browser"
    assert widget.correction_mode_section.title == "Correction"
    toggles = {toggle.text() for toggle in widget.findChildren(QToolButton)}
    assert "Pipeline Files" in toggles
    assert "Segmentation Input Parameters" in toggles
    assert "Ultrack Parameters" in toggles
    assert "Database Browser" in toggles
    assert "Correction Shortcuts" in toggles

    widget.deleteLater()
    viewer.close()


def test_nucleus_workflow_uses_simplified_four_section_layout():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.segmentation_inputs_section.title == "Segmentation Input Parameters"
    assert widget.tracking_ultrack_section.title == "Ultrack Parameters"
    assert widget.ultrack_db_browser_section.title == "Database Browser"
    assert widget.correction_mode_section.title == "Correction"

    assert widget.segmentation_inputs_section.is_expanded is True
    assert widget.tracking_ultrack_section.is_expanded is False
    assert widget.ultrack_db_browser_section.is_expanded is False
    assert widget.correction_mode_section.is_expanded is False

    top_level_widgets = [
        widget.layout().itemAt(index).widget()
        for index in range(widget.layout().count())
        if widget.layout().itemAt(index).widget() is not None
    ]
    assert top_level_widgets.index(widget.segmentation_inputs_section) < top_level_widgets.index(
        widget.tracking_ultrack_section
    )
    assert top_level_widgets.index(widget.tracking_ultrack_section) < top_level_widgets.index(
        widget.ultrack_db_browser_section
    )
    assert top_level_widgets.index(widget.ultrack_db_browser_section) < top_level_widgets.index(
        widget.correction_mode_section
    )

    assert widget.preview_contour_btn not in widget.segmentation_inputs_section.findChildren(QPushButton)
    assert widget.build_btn not in widget.segmentation_inputs_section.findChildren(QPushButton)
    assert widget.run_db_gen_btn not in widget.tracking_ultrack_section.findChildren(QPushButton)
    assert widget.run_ultrack_btn not in widget.tracking_ultrack_section.findChildren(QPushButton)
    assert widget.cancel_btn not in widget.tracking_ultrack_section.findChildren(QPushButton)

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

    assert hasattr(widget, "_files_widget")
    assert hasattr(widget, "pipeline_status_lbl")
    assert hasattr(widget, "pipeline_progress_bar")

    # Section 1: Segmentation Inputs / Source Stacks
    assert hasattr(widget, "build_btn")
    assert not hasattr(widget, "contour_flow_threshold_spin")
    assert not hasattr(widget, "contour_terminal_btn")
    assert not hasattr(widget, "contour_fg_threshold_spin")
    assert not hasattr(widget, "contour_filter_median_time_spin")
    assert not hasattr(widget, "contour_filter_median_space_spin")
    assert not hasattr(widget, "contour_filter_gauss_time_spin")
    assert not hasattr(widget, "contour_filter_gauss_space_spin")
    assert not hasattr(widget, "preview_contour_filter_btn")
    assert not hasattr(widget, "run_contour_filter_btn")
    assert not hasattr(widget, "fg_threshold_spin")

    # Section 2: Ultrack Database Generation
    assert hasattr(widget, "run_db_gen_btn")
    assert hasattr(widget, "db_gen_threshold_min_spin")
    assert hasattr(widget, "db_gen_threshold_max_spin")
    assert hasattr(widget, "db_gen_threshold_step_spin")
    assert not hasattr(widget, "db_gen_terminal_btn")
    assert not hasattr(widget, "db_gen_fg_thr_spin")

    # Section 4: Ultrack Database Browser
    assert hasattr(widget, "ultrack_db_info_lbl")
    assert hasattr(widget, "ultrack_db_active_btn")
    assert hasattr(widget, "ultrack_db_refresh_btn")
    assert not hasattr(widget, "ultrack_db_mode_combo")
    assert hasattr(widget, "ultrack_db_hierarchy_slider")
    assert hasattr(widget, "ultrack_db_height_lbl")
    assert hasattr(widget, "ultrack_db_section_status_lbl")

    # Section 5: Ultrack Tracking
    assert hasattr(widget, "run_ultrack_btn")
    assert not hasattr(widget, "ultrack_terminal_btn")

    assert hasattr(widget, "correction_status_lbl")
    assert not hasattr(widget, "correction_section")

    widget.deleteLater()
    viewer.close()


def test_ultrack_section_is_top_level_without_legacy_route_selector():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.tracking_ultrack_section.title == "Ultrack Parameters"
    assert not hasattr(widget, "ultrack_route_check")
    assert widget.pipeline_status_lbl.text() == ""
    assert widget.pipeline_progress_bar.isVisible() is False

    widget.deleteLater()
    viewer.close()


def test_correction_section_is_top_level():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.correction_mode_section.title == "Correction"
    correction_button_texts = {
        button.text()
        for button in widget.correction_mode_section.findChildren(QPushButton)
    }
    assert "Save tracked" in correction_button_texts
    assert "Load Labels" not in correction_button_texts
    assert "Extend selected" not in correction_button_texts
    assert "Retrack selected" not in correction_button_texts
    assert "◀ Extend (A)" in correction_button_texts
    assert "Extend (D) ▶" in correction_button_texts
    assert "◀ Retrack (Q)" in correction_button_texts
    assert "Retrack (E) ▶" in correction_button_texts

    widget.deleteLater()
    viewer.close()


def test_correction_activate_button_expands_activates_and_deactivates_layers(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "Position_1"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    tracked = np.zeros((1, 4, 4), dtype=np.uint32)
    tracked[0, 1:3, 1:3] = 1
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", tracked)
    widget.refresh(pos_dir)

    assert widget.correction_mode_section.is_expanded is False

    widget.correction_active_btn.setChecked(True)

    assert widget.correction_mode_section.is_expanded is True
    assert widget.correction_widget._layer is viewer.layers["[Correction] Tracked: Nucleus"]
    assert "[Correction] CorrectionDraw" in viewer.layers
    assert "[Correction] CellHighlight" in viewer.layers
    assert "[Correction] CellSpotlight" in viewer.layers

    widget.correction_active_btn.setChecked(False)

    assert widget.correction_mode_section.is_expanded is False
    assert widget.correction_widget._layer is None
    assert "[Correction] CorrectionDraw" not in viewer.layers
    assert "[Correction] CellHighlight" not in viewer.layers
    assert "[Correction] CellSpotlight" not in viewer.layers
    assert "[Correction] Tracked: Nucleus" not in viewer.layers

    widget.deleteLater()
    viewer.close()


def test_correction_activation_loads_owned_layers_from_disk(monkeypatch, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "Position_1"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "0_input").mkdir(parents=True)
    tracked = np.zeros((2, 4, 5), dtype=np.uint32)
    tracked[0, 1:3, 1:3] = 7
    cell = np.arange(20, dtype=np.float32).reshape(4, 5)
    nucleus = np.full((4, 5), 3, dtype=np.float32)
    nls = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", tracked)
    tifffile.imwrite(pos_dir / "0_input" / "cell_zavg.tif", cell)
    tifffile.imwrite(pos_dir / "0_input" / "nucleus_zavg.tif", nucleus)
    tifffile.imwrite(pos_dir / "0_input" / "NLS_zavg.tif", nls)

    stale = viewer.add_labels(np.ones((1, 2, 2), dtype=np.uint32), name="[Correction] stale")
    stale.visible = True
    existing = viewer.add_image(np.ones((4, 5), dtype=np.float32), name="Existing")
    existing.visible = True
    widget.refresh(pos_dir)

    widget.correction_active_btn.setChecked(True)

    assert "[Correction] stale" not in viewer.layers
    assert "[Correction] Tracked: Nucleus" in viewer.layers
    assert "[Correction] Cell z-avg" in viewer.layers
    assert "[Correction] Nucleus z-avg" in viewer.layers
    assert "[Correction] NLS z-avg" in viewer.layers
    assert viewer.layers["Existing"].visible is False
    assert widget.correction_widget._layer is viewer.layers["[Correction] Tracked: Nucleus"]
    assert widget.correction_mode_section.is_expanded is True
    assert set(widget._correction_owned_layers) == {
        "[Correction] Tracked: Nucleus",
        "[Correction] Cell z-avg",
        "[Correction] Nucleus z-avg",
        "[Correction] NLS z-avg",
    }
    np.testing.assert_array_equal(viewer.layers["[Correction] Tracked: Nucleus"].data, tracked)
    assert viewer.layers["[Correction] Cell z-avg"].blending == "additive"
    assert viewer.layers["[Correction] Nucleus z-avg"].blending == "additive"
    assert viewer.layers["[Correction] NLS z-avg"].blending == "additive"
    assert viewer.layers["[Correction] NLS z-avg"].colormap.name == "bop_blue"

    widget.deleteLater()
    viewer.close()


def test_correction_deactivation_removes_registered_layers_and_restores_viewer_state(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "Position_1"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    tracked = np.zeros((1, 4, 4), dtype=np.uint32)
    tracked[0, 1:3, 1:3] = 5
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", tracked)

    existing = viewer.add_image(np.ones((4, 4), dtype=np.float32), name="Existing")
    unrelated_prefixed = viewer.add_image(np.ones((4, 4), dtype=np.float32), name="[Correction] User Layer")
    existing.visible = True
    unrelated_prefixed.visible = True
    viewer.layers.selection.active = existing
    widget.refresh(pos_dir)

    widget.correction_active_btn.setChecked(True)
    widget._add_validated_overlay(np.ones_like(tracked, dtype=np.uint8))
    assert "[Correction] Validated: Nucleus" in viewer.layers
    widget.correction_active_btn.setChecked(False)

    assert "[Correction] Tracked: Nucleus" not in viewer.layers
    assert "[Correction] Validated: Nucleus" not in viewer.layers
    assert "[Correction] User Layer" in viewer.layers
    assert viewer.layers["Existing"].visible is True
    assert viewer.layers["[Correction] User Layer"].visible is True
    assert viewer.layers.selection.active is existing
    assert widget._correction_owned_layers == set()
    assert widget.correction_widget._layer is None
    assert widget.correction_mode_section.is_expanded is False

    widget.deleteLater()
    viewer.close()


def test_correction_activation_without_tracked_file_restores_button_and_viewer(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "Position_1"
    pos_dir.mkdir()
    existing = viewer.add_image(np.ones((4, 4), dtype=np.float32), name="Existing")
    existing.visible = True
    widget.refresh(pos_dir)

    widget.correction_active_btn.setChecked(True)

    assert widget.correction_active_btn.isChecked() is False
    assert widget.correction_mode_section.is_expanded is False
    assert viewer.layers["Existing"].visible is True
    assert widget._correction_owned_layers == set()
    assert "No tracked labels file found" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_correction_save_writes_correction_owned_tracked_layer(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "Position_1"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    original = np.zeros((2, 4, 4), dtype=np.uint32)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", original)
    widget.refresh(pos_dir)
    widget.correction_active_btn.setChecked(True)

    edited = np.zeros((2, 4, 4), dtype=np.uint32)
    edited[1, 1:3, 1:3] = 9
    viewer.layers["[Correction] Tracked: Nucleus"].data = edited
    widget._on_save_tracked()

    np.testing.assert_array_equal(
        tifffile.imread(pos_dir / "2_nucleus" / "tracked_labels.tif"),
        edited,
    )
    assert "Saved 2 frame(s)" in widget.correction_status_lbl.text()

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
    assert {"A", "D", "Q", "E", "B"} <= shortcut_keys

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


def test_database_browser_activate_button_expands_and_deactivation_removes_layers():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    viewer.add_labels(np.ones((1, 4, 4), dtype=np.uint8), name="Ultrack DB Preview")
    viewer.add_labels(np.ones((1, 4, 4), dtype=np.uint8), name="Ultrack DB Selection")
    viewer.add_labels(np.ones((1, 4, 4), dtype=np.uint8), name="Ultrack DB Annotations")

    assert widget.ultrack_db_browser_section.is_expanded is False

    widget.ultrack_db_active_btn.setChecked(True)

    assert widget.ultrack_db_browser_section.is_expanded is True
    assert widget._ultrack_db_browser_active is True

    widget.ultrack_db_active_btn.setChecked(False)

    assert widget.ultrack_db_browser_section.is_expanded is False
    assert widget._ultrack_db_browser_active is False
    assert "Ultrack DB Preview" not in viewer.layers
    assert "Ultrack DB Selection" not in viewer.layers
    assert "Ultrack DB Annotations" not in viewer.layers

    widget.deleteLater()
    viewer.close()


def test_db_gen_section_calls_source_stack_builder_on_run(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    calls = []

    def fake_build_database(**kwargs):
        calls.append(kwargs)
        data_db = kwargs["working_dir"] / "data.db"
        data_db.parent.mkdir(parents=True, exist_ok=True)
        data_db.write_bytes(b"sqlite placeholder")
        return {"database": str(data_db)}

    monkeypatch.setattr(module, "build_ultrack_database_from_sources", fake_build_database)
    monkeypatch.setattr(module, "_ultrack_segment", object(), raising=False)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    dummy = np.zeros((2, 1, 4, 4), dtype=np.float32)
    import tifffile

    tifffile.imwrite(str(pos_dir / "2_nucleus" / "contour_sources.tif"), dummy)
    tifffile.imwrite(str(pos_dir / "2_nucleus" / "foreground_sources.tif"), dummy.astype(np.uint8))
    widget._pos_dir = pos_dir

    widget._on_run_db_generation()

    assert len(calls) == 1
    call = calls[0]
    assert call["contour_sources_path"] == pos_dir / "2_nucleus" / "contour_sources.tif"
    assert call["foreground_sources_path"] == pos_dir / "2_nucleus" / "foreground_sources.tif"
    # DB generation builds candidates only — no annotation/scoring inputs.
    assert "score_signal_path" not in call
    assert "corrections" not in call
    assert "validated_tracks" not in call
    assert "tracked_labels" not in call
    assert "use_validated" not in call
    assert "nucleus_prob_zavg_path" not in call
    assert "thresholds" not in call
    assert call["cfg"].seg_foreground_threshold == pytest.approx(0.0)
    assert widget.run_db_gen_btn.isEnabled()
    assert "complete" in widget.pipeline_status_lbl.text().lower()
    assert widget.pipeline_progress_bar.isVisible() is False
    assert "✓" in _label_texts(widget._files_widget)

    widget.deleteLater()
    viewer.close()


def test_db_gen_section_has_no_terminal_launcher():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "db_gen_terminal_btn")
    assert not hasattr(widget, "_on_db_gen_terminal")

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
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut_state",
        lambda path, frame, state: widget._render_hierarchy_cut(path, frame, state.height),
    )
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
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_scores.tif", np.zeros((1, 4, 4), dtype=np.float32))
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

    text = widget.pipeline_status_lbl.text().lower()
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
    import tifffile
    tifffile.imwrite(
        str(pos_dir / "2_nucleus" / "foreground_scores.tif"),
        np.zeros((2, 4, 4), dtype=np.float32),
    )
    widget._pos_dir = pos_dir

    labels = np.ones((2, 4, 4), dtype=np.uint32)

    def fake_export(_working_dir, _cfg, tracked_path, **_kwargs):
        tracked_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(tracked_path, labels)
        return labels

    monkeypatch.setattr(module, "run_solve", lambda *a, **kw: iter([(1, 1, "solved")]))
    monkeypatch.setattr(module, "export_tracked_labels", fake_export)
    monkeypatch.setattr(module, "apply_annotations_and_score", lambda **kwargs: None)

    widget._on_run_ultrack()

    assert "Tracked: Nucleus" in viewer.layers
    assert (pos_dir / "2_nucleus" / "tracked_labels.tif").exists()
    assert "✓" in _label_texts(widget._files_widget)
    assert widget.pipeline_progress_bar.isVisible() is False

    widget.deleteLater()
    viewer.close()


def test_ultrack_tracking_passes_corrections_to_export(tmp_path, monkeypatch):
    from cellflow.database.validation import add_correction
    from cellflow.tracking_ultrack.corrections import Correction

    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    module = sys.modules[widget_class.__module__]
    _install_sync_thread_worker(monkeypatch, module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus" / "ultrack_workdir").mkdir(parents=True)
    (pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db").write_bytes(b"sqlite placeholder")
    import tifffile
    tifffile.imwrite(
        str(pos_dir / "2_nucleus" / "foreground_scores.tif"),
        np.zeros((2, 4, 4), dtype=np.float32),
    )
    widget._pos_dir = pos_dir
    tracked = np.ones((2, 4, 4), dtype=np.uint32)
    viewer.add_labels(tracked, name="Tracked: Nucleus")
    add_correction(pos_dir, Correction(cell_id=2, t=1, kind="anchor", y=2.0, x=3.0))
    widget.db_gen_use_validated_check.setChecked(True)
    captured = {}
    annotate_calls = []

    def fake_export(_working_dir, _cfg, _tracked_path, **kwargs):
        captured.update(kwargs)
        return tracked

    monkeypatch.setattr(module, "run_solve", lambda *a, **kw: iter([(1, 1, "solved")]))
    monkeypatch.setattr(module, "export_tracked_labels", fake_export)
    monkeypatch.setattr(
        module,
        "apply_annotations_and_score",
        lambda **kwargs: annotate_calls.append(kwargs) or None,
    )

    widget._on_run_ultrack()

    assert [(c.cell_id, c.t, c.kind) for c in captured["corrections"]] == [
        (2, 1, "anchor")
    ]
    assert captured["tracked_labels"].shape == (2, 4, 4)
    # apply_annotations_and_score must run before solve, receiving the same corrections.
    assert len(annotate_calls) == 1
    assert [(c.cell_id, c.t, c.kind) for c in annotate_calls[0]["corrections"]] == [
        (2, 1, "anchor")
    ]
    assert annotate_calls[0]["score_signal_path"] == pos_dir / "2_nucleus" / "foreground_scores.tif"

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


def test_extend_greedy_overwrite_paints_source_and_overwrites_non_validated(tmp_path, monkeypatch):
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
        assignments=(types.SimpleNamespace(cell_id=7, mask_2d=source_mask),),
    )
    monkeypatch.setattr(module, "extend_track_from_db", lambda **_kwargs: result)

    widget._on_extend(direction="forward")

    frame = viewer.layers["Tracked: Nucleus"].data[1]
    # Greedy overwrite paints source cell wherever it doesn't overlap validated cells
    assert np.all(frame[6:11, 6:11] == 7)
    # No reassigned conflict text in status
    status = widget.correction_status_lbl.text()
    assert "reassigned" not in status

    widget.deleteLater()
    viewer.close()


def test_extend_greedy_overwrite_preserves_validated_cells(tmp_path, monkeypatch):
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
        assignments=(types.SimpleNamespace(cell_id=7, mask_2d=source_mask),),
    )
    monkeypatch.setattr(module, "extend_track_from_db", lambda **_kwargs: result)
    monkeypatch.setattr(module, "read_validated_tracks", lambda _pos_dir: {9: {1}})

    widget._on_extend(direction="forward")

    frame = viewer.layers["Tracked: Nucleus"].data[1]
    # Cell 9 is validated at frame 1, so it should not be overwritten
    assert np.all(frame[6:11, 6:11] == 9)

    widget.deleteLater()
    viewer.close()


# ── Task 10: DB generation state persistence ─────────────────────────────────

def test_db_gen_controls_persist_through_state():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget.db_gen_min_area_spin.setValue(500)
    widget.db_gen_max_area_spin.setValue(80_000)
    widget.db_gen_threshold_min_spin.setValue(0.2)
    widget.db_gen_threshold_max_spin.setValue(0.6)
    widget.db_gen_threshold_step_spin.setValue(0.2)
    widget.source_foreground_threshold_min_spin.setValue(0.3)
    widget.source_foreground_threshold_max_spin.setValue(0.9)
    widget.source_foreground_threshold_step_spin.setValue(0.3)
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
    assert widget.db_gen_threshold_min_spin.value() == pytest.approx(0.2)
    assert widget.db_gen_threshold_max_spin.value() == pytest.approx(0.6)
    assert widget.db_gen_threshold_step_spin.value() == pytest.approx(0.2)
    assert widget.source_foreground_threshold_min_spin.value() == pytest.approx(0.3)
    assert widget.source_foreground_threshold_max_spin.value() == pytest.approx(0.9)
    assert widget.source_foreground_threshold_step_spin.value() == pytest.approx(0.3)
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
    pytest.importorskip("ultrack")
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


def test_ultrack_db_hierarchy_states_treat_null_parent_as_root(tmp_path):
    pytest.importorskip("ultrack")
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import Base

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _add_ultrack_node(
            session,
            node_id=100,
            parent_id=None,
            height=10.0,
            bbox=(0, 0, 2, 4),
        )
        _add_ultrack_node(
            session,
            node_id=101,
            parent_id=100,
            height=1.0,
            bbox=(0, 0, 2, 2),
        )
        session.commit()
    engine.dispose()

    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    states = widget._query_hierarchy_cut_states(
        db_path, db_path.stat().st_mtime_ns, frame=0
    )

    assert [state.node_ids for state in states] == [(101,), (100,)]

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
