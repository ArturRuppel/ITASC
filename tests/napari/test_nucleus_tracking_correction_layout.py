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
from qtpy.QtCore import QPoint
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
            "CancelledError": type("CancelledError", (Exception,), {}),
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
            self._files_widget = types.SimpleNamespace(
                presence_count_by_group=lambda: {"Output": (0, 1)}
            )

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

    stub_modules = {
        "cellflow.napari.contact_analysis_widget": {"ContactAnalysisWidget": _StubWidget},
        "cellflow.napari.cell_boundary_workflow_widget": {
            "CellBoundaryWorkflowWidget": _StubWidget,
        },
        "cellflow.napari.cell_workflow_widget": {"CellWorkflowWidget": _StubWidget},
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
        "cellflow.napari.contact_analysis_widget",
        "cellflow.napari.cell_boundary_workflow_widget",
        "cellflow.napari.cell_workflow_widget",
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


def test_ultrack_terminal_script_includes_visible_config_controls():
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


def test_deprecated_sections_are_removed():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "gen_section")
    assert not hasattr(widget, "db_section")

    widget.deleteLater()
    viewer.close()


def test_correction_section_uses_stage_header_params_activate_and_active_toolbar():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    header_layout = widget.correction_header.layout()
    header_widgets = [
        header_layout.itemAt(i).widget()
        for i in range(header_layout.count())
        if header_layout.itemAt(i).widget() is not None
    ]

    assert widget.correction_header_lbl.text() == "Correction"
    assert widget.correction_shortcuts_btn.text() == "📖"
    assert widget.correction_params_btn.text() == "⚙"
    assert widget.correction_active_btn.text() == "⏻"
    assert isinstance(widget.correction_shortcuts_btn, QToolButton)
    assert isinstance(widget.correction_params_btn, QToolButton)
    assert isinstance(widget.correction_active_btn, QToolButton)
    assert header_widgets == [
        widget.correction_header_lbl,
        widget.correction_shortcuts_btn,
        widget.correction_params_btn,
        widget.correction_active_btn,
    ]
    assert widget.correction_shortcuts_btn in header_widgets
    assert widget.correction_params_btn in header_widgets
    assert widget.correction_active_btn in header_widgets
    assert widget.save_tracked_btn not in header_widgets
    assert widget.remove_unvalidated_btn not in header_widgets

    assert widget.correction_widget._outline_btn.parent() is widget.extend_retrack_params_section._inner
    assert widget.correction_widget._status.isVisible() is False
    assert widget.commit_btn.parent() is None
    assert widget.correction_toolbar.isHidden() is True
    assert widget.save_tracked_btn.parent() is widget.correction_toolbar
    assert widget.remove_unvalidated_btn.parent() is widget.correction_toolbar

    for section in (
        widget.extend_retrack_params_section,
        widget.correction_shortcuts_section,
    ):
        assert section._toggle.isHidden() is True
        assert "border: none" in section._content_frame.styleSheet()
        assert section._content_frame.layout().contentsMargins().left() == 0
        assert section.layout().contentsMargins().top() == 0

    assert widget.extend_retrack_params_section.is_expanded is False
    assert widget.correction_shortcuts_section.is_expanded is False
    widget.correction_params_btn.setChecked(True)
    assert widget.extend_retrack_params_section.is_expanded is True
    assert widget.correction_shortcuts_section.is_expanded is False
    assert widget.correction_mode_section.is_expanded is True
    assert widget.correction_widget.isHidden() is True
    assert widget.correction_toolbar.isHidden() is True
    widget.correction_params_btn.setChecked(False)
    assert widget.extend_retrack_params_section.is_expanded is False
    assert widget.correction_mode_section.is_expanded is False

    widget.correction_shortcuts_btn.setChecked(True)
    assert widget.correction_shortcuts_section.is_expanded is True
    assert widget.extend_retrack_params_section.is_expanded is False
    assert widget.correction_mode_section.is_expanded is True
    assert widget.correction_widget.isHidden() is True
    assert widget.correction_toolbar.isHidden() is True
    widget.correction_shortcuts_btn.setChecked(False)
    assert widget.correction_shortcuts_section.is_expanded is False
    assert widget.correction_mode_section.is_expanded is False

    correction = widget.nucleus_correction_widget
    viewer.add_labels(
        np.zeros((1, 2, 2), dtype=np.uint8),
        name="[Correction] Nucleus Labels",
    )
    correction._capture_correction_view_state = lambda: None
    correction._restore_correction_view_state = lambda: None
    correction._load_correction_layers_from_disk = lambda: True
    correction._refresh_refinement_widget = lambda: None
    correction._refresh_tracked_layer_from_disk = lambda: None
    correction._remove_correction_owned_layers = lambda: None

    widget.correction_params_btn.setChecked(True)
    assert widget.extend_retrack_params_section.is_expanded is True
    widget.correction_active_btn.setChecked(True)
    assert widget.correction_active_btn.isChecked() is True
    assert widget.extend_retrack_params_section.is_expanded is False
    assert widget.correction_shortcuts_section.is_expanded is False
    assert widget.correction_mode_section.is_expanded is True
    assert widget.correction_widget.isHidden() is False
    assert widget.correction_toolbar.isHidden() is False

    widget.deleteLater()
    viewer.close()


def test_nucleus_workflow_stage_headers_are_compact_and_evenly_spaced():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    from cellflow.napari.ui_style import stage_accent

    nucleus_accent = stage_accent("nucleus")
    for text in (
        "Pipeline Files",
        "Ultrack Inputs",
        "Ultrack database",
        "Ultrack solve",
        "Database Browser",
        "Correction",
    ):
        label = next(child for child in widget.findChildren(QLabel) if child.text() == text)
        style = label.styleSheet()
        assert "font-size: 9pt" in style
        assert f"color: {nucleus_accent}" not in style

    pipeline_layout = widget.segmentation_inputs_section.parentWidget().layout()
    assert pipeline_layout.spacing() == widget.layout().spacing()

    assert widget.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Maximum
    root_layout = widget.layout()
    assert root_layout.itemAt(root_layout.count() - 1).spacerItem() is None

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


def test_db_gen_section_has_no_terminal_launcher():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "db_gen_terminal_btn")
    assert not hasattr(widget, "_on_db_gen_terminal")

    widget.deleteLater()
    viewer.close()


# ── Task 8: Tracking solves existing DB; extend uses DB ──────────────────────


# ── Task 10: DB generation state persistence ─────────────────────────────────
