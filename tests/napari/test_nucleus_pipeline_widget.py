"""Focused tests for NucleusPipelineWidget."""
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
from napari.layers import Labels
from qtpy.QtWidgets import QApplication, QProgressBar, QToolButton


@pytest.fixture(scope="module", autouse=True)
def _restore_import_stubs_after_module():
    """Keep this module's synthetic imports from leaking into later tests."""
    tracked_prefixes = (
        "cellflow.napari.",
        "cellflow.tracking_ultrack.",
        "cellflow.segmentation.",
    )
    tracked_modules = {
        name
        for name in sys.modules
        if name in {"cellflow.napari", "cellflow.tracking_ultrack", "cellflow.segmentation"}
        or name.startswith(tracked_prefixes)
    }
    originals = {name: sys.modules[name] for name in tracked_modules}

    yield

    for name in list(sys.modules):
        if (
            name in {"cellflow.napari", "cellflow.tracking_ultrack", "cellflow.segmentation"}
            or name.startswith(tracked_prefixes)
        ) and name not in originals:
            sys.modules.pop(name, None)
    for name, module in originals.items():
        sys.modules[name] = module


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
            self.seg_min_area = 300
            self.seg_max_area = 100_000
            self.seg_foreground_threshold = 0.0
            self.seg_min_frontier = 0.0
            self.seg_ws_hierarchy = "area"
            self.seg_n_workers = 1
            self.max_distance = 15.0
            self.max_neighbors = 5
            self.linking_mode = "default"
            self.area_weight = 1.0
            self.iou_weight = 1.0
            self.distance_weight = 0.05
            self.quality_weight = 1.0
            self.quality_exponent = 8.0
            self.circularity_weight = 0.25
            self.link_n_workers = 1
            self.power = 4.0
            self.bias = 0.0
            self.appear_weight = -0.1
            self.disappear_weight = -0.1
            self.division_weight = -0.001
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
            "_select_solver": lambda: "CBC",
        },
        "cellflow.tracking_ultrack.multi_threshold": {
            "build_ultrack_database_from_sources": lambda *args, **kwargs: None,
            "build_ultrack_database_from_thresholds": lambda *args, **kwargs: None,
            "build_ultrack_database_from_threshold_pairs": lambda *args, **kwargs: None,
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
            "write_ultrack_source_stacks": lambda *args, **kwargs: [],
        },
        "cellflow.tracking_ultrack.extend": {
            "extend_track_from_db": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.solve": {
            "run_solve": lambda *args, **kwargs: iter(()),
        },
        "cellflow.segmentation": {
            "CancelledError": type("CancelledError", (Exception,), {}),
        },
        "cellflow.segmentation.nucleus_segmentation": {
            "_check_cancel": lambda is_cancelled: None,
        },
    }

    for module_name, attrs in stub_exports.items():
        mod = types.ModuleType(module_name)
        if module_name == "cellflow.segmentation":
            mod.__path__ = []
        for attr_name, value in attrs.items():
            setattr(mod, attr_name, value)
        sys.modules[module_name] = mod


def _load_workflow_widget_class():
    _install_import_stubs()
    module = importlib.import_module("cellflow.napari.nucleus_workflow_widget")
    return module.NucleusWorkflowWidget


def _get_pipeline_module():
    return sys.modules["cellflow.napari.nucleus_pipeline_widget"]


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


# ── Delegation seam tests ─────────────────────────────────────────────────────


def test_nucleus_workflow_composes_pipeline_widget():
    """NucleusWorkflowWidget.nucleus_pipeline_widget exists and is a NucleusPipelineWidget."""
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    pipeline_module = _get_pipeline_module()
    widget = widget_class(viewer)

    assert isinstance(
        widget.nucleus_pipeline_widget,
        pipeline_module.NucleusPipelineWidget,
    )
    assert widget.seg_run_btn is widget.nucleus_pipeline_widget.seg_run_btn
    assert widget.db_run_btn is widget.nucleus_pipeline_widget.db_run_btn
    assert widget.solve_run_btn is widget.nucleus_pipeline_widget.solve_run_btn
    assert widget.seg_params_btn is widget.nucleus_pipeline_widget.seg_params_btn
    assert widget.db_params_btn is widget.nucleus_pipeline_widget.db_params_btn
    assert widget.solve_params_btn is widget.nucleus_pipeline_widget.solve_params_btn
    assert widget.pipeline_status_lbl is widget.nucleus_pipeline_widget.pipeline_status_lbl
    assert widget.pipeline_progress_bar is widget.nucleus_pipeline_widget.pipeline_progress_bar

    widget.deleteLater()
    viewer.close()


def test_nucleus_pipeline_files_header_uses_magnifier_button():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    assert widget.pipeline_files_header_lbl.text() == "Pipeline Files"
    assert isinstance(widget.pipeline_files_toggle_btn, QToolButton)
    assert widget.pipeline_files_toggle_btn.text() == "🔍"
    assert not widget._pipeline_files_section._toggle.isVisible()

    assert widget._pipeline_files_section.is_expanded is False
    widget.pipeline_files_toggle_btn.click()
    assert widget._pipeline_files_section.is_expanded is True
    widget.pipeline_files_toggle_btn.click()
    assert widget._pipeline_files_section.is_expanded is False

    widget.deleteLater()
    viewer.close()


def test_nucleus_pipeline_files_omit_source_stack_artifacts():
    _install_import_stubs()
    workflow_module = importlib.import_module("cellflow.napari.nucleus_workflow_widget")

    tracked_paths = [
        path
        for group in workflow_module._NUCLEUS_PIPELINE_FILE_GROUPS
        for path, _label in group[1]
    ]

    assert "2_nucleus/contour_sources.tif" not in tracked_paths
    assert "2_nucleus/foreground_sources.tif" not in tracked_paths
    assert "2_nucleus/ultrack_workdir/data.db" in tracked_paths


def test_project_status_panel_omits_source_stack_artifacts():
    _install_import_stubs()
    data_panel_module = importlib.import_module("cellflow.napari.data_panel_widget")

    tracked_paths = [
        path
        for group in data_panel_module._TRACKED_FILE_GROUPS
        for path, _label in group[1]
    ]

    assert "2_nucleus/contour_sources.tif" not in tracked_paths
    assert "2_nucleus/foreground_sources.tif" not in tracked_paths
    assert "2_nucleus/ultrack_workdir/data.db" in tracked_paths


def test_pipeline_widget_handler_methods_aliased_on_workflow():
    """Pipeline handler methods are reachable on the workflow widget via instance aliases."""
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    pipeline_module = _get_pipeline_module()
    widget = widget_class(viewer)
    pl = widget.nucleus_pipeline_widget

    for method_name in (
        "_on_build_segmentation_inputs",
        "_on_run_db_generation",
        "_on_run_ultrack",
        "_on_cancel",
        "_status",
        "_set_running_stage",
    ):
        widget_method = getattr(widget, method_name)
        pl_method = getattr(pl, method_name)
        assert getattr(widget_method, "__func__", widget_method) is getattr(pl_method, "__func__", pl_method), (
            f"{method_name} not aliased to pipeline widget"
        )

    widget.deleteLater()
    viewer.close()


# ── Structure tests ───────────────────────────────────────────────────────────


def test_pipeline_widget_initial_run_buttons_enabled():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    assert widget.seg_run_btn.isEnabled()
    assert widget.db_run_btn.isEnabled()
    assert widget.solve_run_btn.isEnabled()
    assert widget.seg_run_btn.text() == "▶"
    assert widget.db_run_btn.text() == "▶"
    assert widget.solve_run_btn.text() == "▶"

    widget.deleteLater()
    viewer.close()


def test_nucleus_workflow_exposes_threshold_pair_controls_in_db_parameters():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    for name in (
        "map_cellprob_min_spin",
        "map_cellprob_max_spin",
        "map_cellprob_step_spin",
        "map_cellprob_range",
        "map_z_range",
        "map_z_start_spin",
        "map_z_stop_spin",
        "map_z_step_spin",
    ):
        assert not hasattr(widget, name)
        assert not hasattr(widget.nucleus_segmentation_inputs_widget, name)

    for name in (
        "source_contour_threshold_min_spin",
        "source_contour_threshold_max_spin",
        "source_contour_threshold_step_spin",
        "source_foreground_threshold_min_spin",
        "source_foreground_threshold_max_spin",
        "source_foreground_threshold_step_spin",
        "source_contour_threshold_range",
        "source_foreground_threshold_range",
    ):
        assert not hasattr(widget, name)
        assert not hasattr(widget.nucleus_segmentation_inputs_widget, name)

    assert hasattr(widget, "source_contour_threshold_spin")
    assert hasattr(widget, "source_foreground_threshold_spin")
    assert widget.threshold_pairs() == []
    assert "map_generation" not in widget.get_state()

    widget.deleteLater()
    viewer.close()


def test_pipeline_widget_status_label_starts_empty():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    assert widget.pipeline_status_lbl.text() == ""
    assert widget.pipeline_progress_bar.isVisible() is False

    widget.deleteLater()
    viewer.close()


def test_pipeline_widget_progress_bar_is_a_progress_bar():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    assert isinstance(widget.pipeline_progress_bar, QProgressBar)

    widget.deleteLater()
    viewer.close()


# ── Status helpers ────────────────────────────────────────────────────────────


def test_pipeline_status_method_updates_label():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    widget._status("hello pipeline")
    assert widget.pipeline_status_lbl.text() == "hello pipeline"
    assert not widget.pipeline_status_lbl.isHidden()

    widget._status("")
    assert widget.pipeline_status_lbl.isHidden()

    widget.deleteLater()
    viewer.close()


def test_nucleus_state_persists_explicit_threshold_pairs():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    widget.set_threshold_pairs(
        [
            {"contour_threshold": 0.2, "foreground_threshold": 0.7},
            {"contour_threshold": 0.4, "foreground_threshold": 0.8},
        ]
    )

    state = widget.get_state()
    db_state = state["db_generation"]

    assert db_state["threshold_pairs"] == [
        {"contour_threshold": 0.2, "foreground_threshold": 0.7},
        {"contour_threshold": 0.4, "foreground_threshold": 0.8},
    ]
    assert "threshold_min" not in db_state
    assert "contour_threshold_min" not in db_state
    assert "foreground_threshold_min" not in db_state

    restored = widget_class(viewer)
    restored.set_state(state)

    assert restored.threshold_pairs() == db_state["threshold_pairs"]

    restored.deleteLater()
    widget.deleteLater()
    viewer.close()


def test_nucleus_state_loads_legacy_sweep_state_as_empty_threshold_pairs():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    widget.set_state(
        {
            "db_generation": {
                "threshold_min": 0.1,
                "threshold_max": 0.5,
                "threshold_step": 0.1,
                "contour_threshold_min": 0.2,
                "foreground_threshold_min": 0.3,
            }
        }
    )

    assert widget.threshold_pairs() == []

    widget.deleteLater()
    viewer.close()


def test_pipeline_clear_progress_hides_bar():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    widget.pipeline_progress_bar.setVisible(True)
    widget._clear_progress()
    assert not widget.pipeline_progress_bar.isVisible()

    widget.deleteLater()
    viewer.close()


def test_set_running_stage_disables_other_rows():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pl = widget.nucleus_pipeline_widget

    pl._set_running_stage("seg")
    assert widget.seg_run_btn.text() == "✕"
    assert widget.seg_run_btn.isEnabled()
    assert not widget.db_run_btn.isEnabled()
    assert not widget.solve_run_btn.isEnabled()
    assert not widget.db_params_btn.isEnabled()
    assert not widget.solve_params_btn.isEnabled()

    pl._set_running_stage("db")
    assert widget.db_run_btn.text() == "✕"
    assert widget.db_run_btn.isEnabled()
    assert not widget.seg_run_btn.isEnabled()
    assert not widget.solve_run_btn.isEnabled()

    pl._set_running_stage("ultrack")
    assert widget.solve_run_btn.text() == "✕"
    assert widget.solve_run_btn.isEnabled()
    assert not widget.seg_run_btn.isEnabled()
    assert not widget.db_run_btn.isEnabled()

    pl._set_running_stage(None)
    assert widget.seg_run_btn.text() == "▶"
    assert widget.db_run_btn.text() == "▶"
    assert widget.solve_run_btn.text() == "▶"
    assert widget.seg_run_btn.isEnabled()
    assert widget.db_run_btn.isEnabled()
    assert widget.solve_run_btn.isEnabled()
    assert widget.seg_params_btn.isEnabled()
    assert widget.db_params_btn.isEnabled()
    assert widget.solve_params_btn.isEnabled()

    widget.deleteLater()
    viewer.close()


def test_running_stage_params_btn_stays_enabled():
    """While a stage is running, its own ⚙ button stays enabled."""
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pl = widget.nucleus_pipeline_widget

    pl._set_running_stage("seg")
    assert widget.seg_params_btn.isEnabled()

    pl._set_running_stage("db")
    assert widget.db_params_btn.isEnabled()

    pl._set_running_stage("ultrack")
    assert widget.solve_params_btn.isEnabled()

    widget.deleteLater()
    viewer.close()


# ── Handler tests ─────────────────────────────────────────────────────────────


def test_preview_threshold_pair_updates_layers_without_mutating_pair_list(
    tmp_path, monkeypatch
):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    _install_sync_thread_worker(monkeypatch, pipeline_module)

    pos_dir = tmp_path / "pos00"
    widget._pos_dir = pos_dir
    widget.source_contour_threshold_spin.setValue(0.2)
    widget.source_foreground_threshold_spin.setValue(0.6)

    import tifffile
    (pos_dir / "1_cellpose").mkdir(parents=True)
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_contours.tif",
        np.ones((2, 3, 3), dtype=np.float32),
    )
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_foreground.tif",
        np.ones((2, 3, 3), dtype=np.float32),
    )

    calls = []
    contour_preview = np.full((1, 2, 3, 3), 0.75, dtype=np.float32)
    foreground_preview = np.ones((1, 2, 3, 3), dtype=np.uint8)
    metadata = [{"contour_threshold": 0.2, "foreground_threshold": 0.6}]

    def fake_build(contours, foreground_scores, **kwargs):
        calls.append((contours.copy(), foreground_scores.copy(), kwargs))
        return contour_preview, foreground_preview, metadata

    def fail_write(*args, **kwargs):
        raise AssertionError("source-stack TIFF writer should not be called")

    monkeypatch.setattr(
        pipeline_module,
        "build_ultrack_source_stacks_from_pairs",
        fake_build,
    )
    monkeypatch.setattr(
        pipeline_module,
        "write_ultrack_source_stacks",
        fail_write,
        raising=False,
    )

    widget._on_preview_threshold_pair()

    assert len(calls) == 1
    _contours, _foreground_scores, kwargs = calls[0]
    assert kwargs["threshold_pairs"] == metadata
    assert "Ultrack Preview: Contours" in viewer.layers
    assert "Ultrack Preview: Foreground" in viewer.layers
    contour_layer = viewer.layers["Ultrack Preview: Contours"]
    foreground_layer = viewer.layers["Ultrack Preview: Foreground"]
    assert isinstance(foreground_layer, Labels)
    assert contour_layer.data.shape[:2] == (2, 1)
    assert foreground_layer.data.shape[:2] == (2, 1)
    np.testing.assert_allclose(contour_layer.data, np.moveaxis(contour_preview, 0, 1))
    np.testing.assert_array_equal(
        foreground_layer.data,
        np.moveaxis(foreground_preview, 0, 1),
    )
    assert contour_layer.metadata["thresholds"] == metadata
    assert foreground_layer.metadata["thresholds"] == metadata
    assert contour_layer.metadata["axis_order"] == ("time", "source", "y", "x")
    assert foreground_layer.metadata["axis_order"] == ("time", "source", "y", "x")
    viewer.dims.set_current_step(0, 1)
    viewer.dims.set_current_step(1, 0)
    assert widget._current_t() == 1
    assert widget.threshold_pairs() == []
    assert not (pos_dir / "2_nucleus" / "contour_sources.tif").exists()
    assert "preview" in widget.pipeline_status_lbl.text().lower()
    assert not widget.pipeline_progress_bar.isVisible()

    widget.deleteLater()
    viewer.close()


def test_preview_checkbox_auto_updates_when_threshold_changes(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    _install_sync_thread_worker(monkeypatch, pipeline_module)

    pos_dir = tmp_path / "pos00"
    widget._pos_dir = pos_dir

    import tifffile
    (pos_dir / "1_cellpose").mkdir(parents=True)
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_contours.tif",
        np.ones((2, 3, 3), dtype=np.float32),
    )
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_foreground.tif",
        np.ones((2, 3, 3), dtype=np.float32),
    )

    calls = []

    def fake_build(contours, foreground_scores, **kwargs):
        calls.append(kwargs["threshold_pairs"][0])
        return (
            np.ones((1, 2, 3, 3), dtype=np.float32),
            np.ones((1, 2, 3, 3), dtype=np.uint8),
            kwargs["threshold_pairs"],
        )

    monkeypatch.setattr(
        pipeline_module,
        "build_ultrack_source_stacks_from_pairs",
        fake_build,
    )

    widget.source_threshold_preview_check.setChecked(True)
    calls.clear()
    widget.source_contour_threshold_spin.setValue(0.35)

    assert calls[-1]["contour_threshold"] == pytest.approx(0.35)

    widget.source_threshold_preview_check.setChecked(False)
    calls.clear()
    widget.source_foreground_threshold_spin.setValue(0.75)

    assert calls == []

    widget.deleteLater()
    viewer.close()


def test_run_db_generation_calls_build_database(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    _install_sync_thread_worker(monkeypatch, pipeline_module)
    widget.set_threshold_pairs(
        [
            {"contour_threshold": 0.4, "foreground_threshold": 0.8},
            {"contour_threshold": 0.2, "foreground_threshold": 0.6},
        ]
    )

    calls = []

    def fake_build_database(**kwargs):
        calls.append(kwargs)
        data_db = kwargs["working_dir"] / "data.db"
        data_db.parent.mkdir(parents=True, exist_ok=True)
        data_db.write_bytes(b"sqlite placeholder")
        return {"database": str(data_db)}

    monkeypatch.setattr(
        pipeline_module,
        "build_ultrack_database_from_threshold_pairs",
        fake_build_database,
    )
    monkeypatch.setattr(pipeline_module, "_ultrack_available", lambda: True)
    monkeypatch.setattr(pipeline_module, "_ultrack_segment", object(), raising=False)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    dummy = np.zeros((2, 1, 4, 4), dtype=np.float32)
    import tifffile
    tifffile.imwrite(str(pos_dir / "1_cellpose" / "nucleus_contours.tif"), dummy)
    tifffile.imwrite(str(pos_dir / "1_cellpose" / "nucleus_foreground.tif"), dummy)
    widget._pos_dir = pos_dir

    widget._on_run_db_generation()

    assert len(calls) == 1
    call = calls[0]
    assert call["contours_path"] == pos_dir / "1_cellpose" / "nucleus_contours.tif"
    assert call["foreground_scores_path"] == pos_dir / "1_cellpose" / "nucleus_foreground.tif"
    assert call["working_dir"] == pos_dir / "2_nucleus" / "ultrack_workdir"
    assert call["threshold_pairs"] == [
        {"contour_threshold": 0.4, "foreground_threshold": 0.8},
        {"contour_threshold": 0.2, "foreground_threshold": 0.6},
    ]
    assert "score_signal_path" not in call
    assert call["cfg"].seg_foreground_threshold == pytest.approx(0.0)
    assert widget.db_run_btn.isEnabled()
    assert widget.db_run_btn.text() == "▶"
    assert "complete" in widget.pipeline_status_lbl.text().lower()
    assert not widget.pipeline_progress_bar.isVisible()

    widget.deleteLater()
    viewer.close()


def test_run_db_generation_reports_empty_threshold_pair_list(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    monkeypatch.setattr(pipeline_module, "_ultrack_available", lambda: True)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    dummy = np.zeros((2, 4, 4), dtype=np.float32)
    import tifffile
    tifffile.imwrite(str(pos_dir / "1_cellpose" / "nucleus_contours.tif"), dummy)
    tifffile.imwrite(str(pos_dir / "1_cellpose" / "nucleus_foreground.tif"), dummy)
    widget._pos_dir = pos_dir

    widget._on_run_db_generation()

    assert widget.pipeline_status_lbl.text() == (
        "Add at least one threshold pair before DB generation."
    )

    widget.deleteLater()
    viewer.close()


def test_run_db_generation_reports_missing_canonical_contours(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    monkeypatch.setattr(pipeline_module, "_ultrack_available", lambda: True)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    widget._pos_dir = pos_dir

    widget._on_run_db_generation()

    assert "nucleus_contours.tif" in widget.pipeline_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_run_db_generation_reports_missing_canonical_foreground(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    monkeypatch.setattr(pipeline_module, "_ultrack_available", lambda: True)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    import tifffile
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "nucleus_contours.tif",
        np.ones((2, 4, 4), dtype=np.float32),
    )
    widget._pos_dir = pos_dir

    widget._on_run_db_generation()

    assert "nucleus_foreground.tif" in widget.pipeline_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_run_ultrack_fails_clearly_if_db_missing(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    widget._pos_dir = pos_dir

    widget._on_run_ultrack()

    text = widget.pipeline_status_lbl.text().lower()
    assert "data.db" in text or "missing" in text or "database" in text

    widget.deleteLater()
    viewer.close()


def test_run_ultrack_updates_tracked_layer(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    _install_sync_thread_worker(monkeypatch, pipeline_module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus" / "ultrack_workdir").mkdir(parents=True)
    (pos_dir / "1_cellpose").mkdir()
    (pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db").write_bytes(b"sqlite placeholder")
    import tifffile
    tifffile.imwrite(
        str(pos_dir / "1_cellpose" / "nucleus_foreground.tif"),
        np.zeros((2, 4, 4), dtype=np.float32),
    )
    widget._pos_dir = pos_dir

    labels = np.ones((2, 4, 4), dtype=np.uint32)

    def fake_export(_working_dir, _cfg, tracked_path, **_kwargs):
        tracked_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(tracked_path, labels)
        return labels

    monkeypatch.setattr(pipeline_module, "run_solve", lambda *a, **kw: iter([(1, 1, "solved")]))
    monkeypatch.setattr(pipeline_module, "export_tracked_labels", fake_export)
    monkeypatch.setattr(pipeline_module, "apply_annotations_and_score", lambda **kwargs: None)

    widget._on_run_ultrack()

    assert "Tracked: Nucleus" in viewer.layers
    assert (pos_dir / "2_nucleus" / "tracked_labels.tif").exists()
    assert not widget.pipeline_progress_bar.isVisible()

    widget.deleteLater()
    viewer.close()


def test_cancel_stops_workers():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    pl = widget.nucleus_pipeline_widget

    widget._status("running")
    pl._set_running_stage("db")

    class _FakeWorker:
        def __init__(self):
            self.quit_called = False

        def quit(self):
            self.quit_called = True

    fake_worker = _FakeWorker()
    pl._db_gen_worker = fake_worker

    widget._on_cancel()

    assert fake_worker.quit_called
    assert pl._db_gen_worker is None
    assert widget.db_run_btn.isEnabled()
    assert widget.db_run_btn.text() == "▶"
    assert "Cancelled" in widget.pipeline_status_lbl.text()

    widget.deleteLater()
    viewer.close()


# ── New tests: per-stage rows ─────────────────────────────────────────────────


def test_params_btn_toggles_section_expansion():
    """Clicking each ⚙ button toggles its inline params section."""
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    db_section = widget.tracking_db_section
    solve_section = widget.tracking_solve_section

    # DB and solve parameter blocks are collapsed by default.
    assert not db_section.is_expanded
    assert not solve_section.is_expanded

    # Toggle db params
    widget.db_params_btn.setChecked(True)
    assert db_section.is_expanded
    widget.db_params_btn.setChecked(False)
    assert not db_section.is_expanded

    # Toggle solve params
    widget.solve_params_btn.setChecked(True)
    assert solve_section.is_expanded
    widget.solve_params_btn.setChecked(False)
    assert not solve_section.is_expanded

    widget.deleteLater()
    viewer.close()


def test_seg_run_btn_invokes_build_segmentation_inputs(monkeypatch):
    """Clicking ▶ on the segmentation row calls _on_build_segmentation_inputs."""
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pl = widget.nucleus_pipeline_widget

    called = []
    monkeypatch.setattr(pl, "_on_build_segmentation_inputs", lambda: called.append("seg"))

    widget.seg_run_btn.click()

    assert called == ["seg"]

    widget.deleteLater()
    viewer.close()


def test_db_run_btn_invokes_run_db_generation(monkeypatch):
    """Clicking ▶ on the DB row calls _on_run_db_generation."""
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pl = widget.nucleus_pipeline_widget

    called = []
    monkeypatch.setattr(pl, "_on_run_db_generation", lambda: called.append("db"))

    widget.db_run_btn.click()

    assert called == ["db"]

    widget.deleteLater()
    viewer.close()


def test_solve_run_btn_invokes_run_ultrack(monkeypatch):
    """Clicking ▶ on the solve row calls _on_run_ultrack."""
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pl = widget.nucleus_pipeline_widget

    called = []
    monkeypatch.setattr(pl, "_on_run_ultrack", lambda: called.append("ultrack"))

    widget.solve_run_btn.click()

    assert called == ["ultrack"]

    widget.deleteLater()
    viewer.close()


def test_running_row_shows_cancel_icon_others_disabled():
    """While seg is running, its row shows ✕ and the other rows are disabled."""
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pl = widget.nucleus_pipeline_widget

    pl._set_running_stage("seg")

    assert widget.seg_run_btn.text() == "✕"
    assert widget.seg_run_btn.isEnabled()
    assert not widget.db_run_btn.isEnabled()
    assert not widget.solve_run_btn.isEnabled()
    assert not widget.db_params_btn.isEnabled()
    assert not widget.solve_params_btn.isEnabled()

    widget.deleteLater()
    viewer.close()
