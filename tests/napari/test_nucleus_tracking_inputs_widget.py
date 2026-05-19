"""Focused tests for NucleusTrackingInputsWidget."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import napari
from qtpy.QtWidgets import QApplication, QCheckBox


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
            self.area_weight = 1.0
            self.iou_weight = 1.0
            self.distance_weight = 0.25
            self.min_area_ratio = 0.3
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
            self.max_segments_per_time = 1_000_000
            self.__dict__.update(kwargs)

    stub_exports = {
        "cellflow.tracking_ultrack.config": {"TrackingConfig": _StubTrackingConfig},
    }

    for module_name, attrs in stub_exports.items():
        module = types.ModuleType(module_name)
        if module_name == "cellflow.segmentation":
            segmentation_dir = (
                Path(__file__).resolve().parents[2] / "src" / "cellflow" / "segmentation"
            )
            module.__path__ = [str(segmentation_dir)]
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        sys.modules[module_name] = module


def _load_widget_class():
    _install_import_stubs()
    module = importlib.import_module("cellflow.napari.nucleus_tracking_inputs_widget")
    return module.NucleusTrackingInputsWidget, module


# ── Structure tests ───────────────────────────────────────────────────────────


def test_tracking_inputs_widget_exposes_quality_controls_without_deprecated_power():
    _app = QApplication.instance() or QApplication([])
    widget_class, _module = _load_widget_class()
    widget = widget_class()

    assert not hasattr(widget, "db_gen_power_spin")
    assert widget.db_gen_quality_exp_spin.value() == pytest.approx(8.0)
    assert "node_prob" in widget.db_gen_quality_exp_spin.toolTip()

    widget.deleteLater()


def test_tracking_inputs_widget_exposes_node_probability_weight_controls():
    _app = QApplication.instance() or QApplication([])
    widget_class, _module = _load_widget_class()
    widget = widget_class()

    assert hasattr(widget, "db_gen_quality_weight_spin")
    assert hasattr(widget, "db_gen_quality_exp_spin")
    assert hasattr(widget, "db_gen_circularity_weight_spin")
    assert widget.db_gen_quality_weight_spin.value() == pytest.approx(1.0)
    assert widget.db_gen_quality_exp_spin.value() == pytest.approx(8.0)
    assert widget.db_gen_circularity_weight_spin.value() == pytest.approx(0.25)

    widget.deleteLater()


def test_tracking_inputs_widget_db_gen_config_applies_all_controls():
    _app = QApplication.instance() or QApplication([])
    widget_class, _module = _load_widget_class()
    widget = widget_class()

    widget.db_gen_quality_weight_spin.setValue(0.8)
    widget.db_gen_quality_exp_spin.setValue(6.0)
    widget.db_gen_circularity_weight_spin.setValue(0.35)
    widget.db_gen_min_area_spin.setValue(500)
    widget.db_gen_max_dist_spin.setValue(20.0)

    cfg = widget.db_gen_config()

    assert cfg.quality_weight == pytest.approx(0.8)
    assert cfg.quality_exponent == pytest.approx(6.0)
    assert cfg.circularity_weight == pytest.approx(0.35)
    assert cfg.seg_min_area == 500
    assert cfg.max_distance == pytest.approx(20.0)

    widget.deleteLater()


def test_tracking_inputs_widget_ultrack_config_includes_solver_controls():
    _app = QApplication.instance() or QApplication([])
    widget_class, _module = _load_widget_class()
    widget = widget_class()

    widget.ultrack_bias_spin.setValue(-0.5)
    widget.ultrack_power_spin.setValue(3.0)
    widget.ultrack_appear_spin.setValue(-0.05)

    cfg = widget.ultrack_config()

    assert cfg.bias == pytest.approx(-0.5)
    assert cfg.power == pytest.approx(3.0)
    assert cfg.appear_weight == pytest.approx(-0.05)

    widget.deleteLater()


def test_tracking_inputs_widget_shape_mode_enables_weight_controls():
    _app = QApplication.instance() or QApplication([])
    widget_class, _module = _load_widget_class()
    widget = widget_class()

    assert not widget.db_gen_area_weight_spin.isEnabled()
    assert not widget.db_gen_iou_weight_spin.isEnabled()
    assert not widget.db_gen_distance_weight_spin.isEnabled()

    widget.db_gen_linking_mode_combo.setCurrentText("shape")

    assert widget.db_gen_area_weight_spin.isEnabled()
    assert widget.db_gen_iou_weight_spin.isEnabled()
    assert widget.db_gen_distance_weight_spin.isEnabled()

    widget.db_gen_linking_mode_combo.setCurrentText("default")

    assert not widget.db_gen_area_weight_spin.isEnabled()
    assert not widget.db_gen_iou_weight_spin.isEnabled()
    assert not widget.db_gen_distance_weight_spin.isEnabled()

    widget.deleteLater()


def test_tracking_inputs_widget_scoring_controls_always_enabled():
    _app = QApplication.instance() or QApplication([])
    widget_class, _module = _load_widget_class()
    widget = widget_class()

    scoring_controls = [
        widget.db_gen_quality_weight_spin,
        widget.db_gen_quality_exp_spin,
        widget.db_gen_circularity_weight_spin,
    ]

    widget.db_gen_use_validated_check.setChecked(False)

    assert all(control.isEnabled() for control in scoring_controls)

    widget.deleteLater()


def test_tracking_inputs_widget_threshold_pair_list_starts_empty_and_adds_pairs():
    _app = QApplication.instance() or QApplication([])
    widget_class, _module = _load_widget_class()
    widget = widget_class()

    assert widget.threshold_pairs() == []

    widget.source_contour_threshold_spin.setValue(0.2)
    widget.source_foreground_threshold_spin.setValue(0.7)

    assert widget.current_threshold_pair() == {
        "contour_threshold": pytest.approx(0.2),
        "foreground_threshold": pytest.approx(0.7),
    }

    assert widget.add_threshold_pair()
    pairs = widget.threshold_pairs()
    assert len(pairs) == 1
    assert pairs[0]["contour_threshold"] == pytest.approx(0.2)
    assert pairs[0]["foreground_threshold"] == pytest.approx(0.7)
    assert widget.source_threshold_pairs_table.rowCount() == 1

    widget.deleteLater()


def test_tracking_inputs_widget_preview_control_is_checkbox():
    _app = QApplication.instance() or QApplication([])
    widget_class, _module = _load_widget_class()
    widget = widget_class()

    assert isinstance(widget.source_threshold_preview_check, QCheckBox)
    assert widget.source_threshold_preview_check.text() == "Preview"
    assert not hasattr(widget, "source_threshold_preview_btn")

    widget.deleteLater()


def test_tracking_inputs_widget_threshold_pair_list_rejects_duplicates():
    _app = QApplication.instance() or QApplication([])
    widget_class, _module = _load_widget_class()
    widget = widget_class()

    widget.source_contour_threshold_spin.setValue(0.2)
    widget.source_foreground_threshold_spin.setValue(0.7)

    assert widget.add_threshold_pair()
    assert not widget.add_threshold_pair()

    assert len(widget.threshold_pairs()) == 1
    assert widget.source_threshold_status_lbl.text()

    widget.deleteLater()


def test_tracking_inputs_widget_removes_and_clears_threshold_pairs():
    _app = QApplication.instance() or QApplication([])
    widget_class, _module = _load_widget_class()
    widget = widget_class()

    widget.set_threshold_pairs(
        [
            {"contour_threshold": 0.2, "foreground_threshold": 0.7},
            {"contour_threshold": 0.4, "foreground_threshold": 0.8},
        ]
    )
    widget.source_threshold_pairs_table.selectRow(0)

    assert widget.remove_selected_threshold_pair()
    pairs = widget.threshold_pairs()
    assert len(pairs) == 1
    assert pairs[0]["contour_threshold"] == pytest.approx(0.4)

    widget.clear_threshold_pairs()

    assert widget.threshold_pairs() == []
    assert widget.source_threshold_pairs_table.rowCount() == 0

    widget.deleteLater()


# ── Delegation seam test ──────────────────────────────────────────────────────


def test_nucleus_workflow_delegates_tracking_inputs_to_child_widget():
    """NucleusWorkflowWidget composes NucleusTrackingInputsWidget and exposes aliases."""
    _install_import_stubs()

    # Install the full set of stubs needed for the workflow widget
    src_root = Path(__file__).resolve().parents[2] / "src" / "cellflow"

    tracking_pkg = types.ModuleType("cellflow.tracking_ultrack")
    tracking_pkg.__path__ = [str(src_root / "tracking_ultrack")]
    sys.modules["cellflow.tracking_ultrack"] = tracking_pkg

    class _StubTrackingConfig:
        def __init__(self, **kwargs):
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
            "build_ultrack_database_from_thresholds": lambda *args, **kwargs: None,
            "build_ultrack_database_from_threshold_pairs": lambda *args, **kwargs: None,
            "build_ultrack_database_from_sources": lambda *args, **kwargs: None,
            "build_ultrack_source_stacks": lambda *args, **kwargs: (
                np.zeros((1, 1, 1, 1), dtype=np.float32),
                np.zeros((1, 1, 1, 1), dtype=np.uint8),
                [],
            ),
            "build_ultrack_source_stacks_from_pairs": lambda *args, **kwargs: (
                np.zeros((1, 1, 1, 1), dtype=np.float32),
                np.zeros((1, 1, 1, 1), dtype=np.uint8),
                [],
            ),
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
            "write_seed_prior_node_probs": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.solve": {
            "database_has_annotations": lambda *args, **kwargs: False,
            "run_solve": lambda *args, **kwargs: iter(()),
        },
        "cellflow.segmentation": {
            "apply_gamma": lambda logits, gamma: logits,
            "build_consensus_boundary_flow_following": lambda *args, **kwargs: (None, None),
            "CancelledError": type("CancelledError", (Exception,), {}),
        },
    }

    for module_name, attrs in stub_exports.items():
        module = types.ModuleType(module_name)
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        sys.modules[module_name] = module

    _app, viewer = _make_viewer()
    workflow_module = importlib.import_module("cellflow.napari.nucleus_workflow_widget")
    tracking_module = importlib.import_module("cellflow.napari.nucleus_tracking_inputs_widget")
    widget = workflow_module.NucleusWorkflowWidget(viewer)

    assert isinstance(
        widget.nucleus_tracking_inputs_widget,
        tracking_module.NucleusTrackingInputsWidget,
    )
    assert widget.tracking_db_section is widget.nucleus_tracking_inputs_widget.db_section
    assert widget.tracking_solve_section is widget.nucleus_tracking_inputs_widget.solve_section
    assert widget.db_gen_min_area_spin is widget.nucleus_tracking_inputs_widget.db_gen_min_area_spin
    assert widget.db_gen_quality_weight_spin is widget.nucleus_tracking_inputs_widget.db_gen_quality_weight_spin
    assert widget.ultrack_bias_spin is widget.nucleus_tracking_inputs_widget.ultrack_bias_spin
    assert widget.ultrack_solver_lbl is widget.nucleus_tracking_inputs_widget.ultrack_solver_lbl
    assert widget.db_gen_linking_mode_combo is widget.nucleus_tracking_inputs_widget.db_gen_linking_mode_combo

    widget.deleteLater()
    viewer.close()
