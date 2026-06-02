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


def _stub_missing_attr(name: str):
    """Fallback for stub modules: resolve any unlisted import to a no-op callable."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return lambda *args, **kwargs: None


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
            "corrections_from_validated_tracks": __import__(
                "cellflow.tracking_ultrack.corrections",
                fromlist=["corrections_from_validated_tracks"],
            ).corrections_from_validated_tracks,
        },
        "cellflow.tracking_ultrack.db_build": {
            "apply_annotations_and_score": lambda *args, **kwargs: None,
            "build_atom_union_database": lambda *args, **kwargs: None,
            "annotate_database_from_corrections": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.export": {"export_tracked_labels": lambda *args, **kwargs: None},
        "cellflow.tracking_ultrack.ingest": {
            "_select_solver": lambda: "CBC",
        },
        "cellflow.tracking_ultrack.multi_threshold": {
            "build_ultrack_database_from_sources": lambda *args, **kwargs: None,
            "build_ultrack_database_from_thresholds": lambda *args, **kwargs: None,
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
        mod.__getattr__ = _stub_missing_attr
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


def test_project_status_panel_tracks_only_required_input_stacks():
    _install_import_stubs()
    data_panel_module = importlib.import_module("cellflow.napari.data_panel_widget")

    tracked_paths = [
        path
        for group in data_panel_module._TRACKED_FILE_GROUPS
        for path, _label in group[1]
    ]

    assert "2_nucleus/contour_sources.tif" not in tracked_paths
    assert "2_nucleus/foreground_sources.tif" not in tracked_paths
    assert "0_input/NLS_zavg.tif" not in tracked_paths
    assert "0_input/NLS_3dt.tif" not in tracked_paths
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


def test_dims_step_change_defers_correction_refresh_until_event_loop(monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    calls = []

    monkeypatch.setattr(
        widget.nucleus_correction_widget,
        "on_dims_step_changed",
        lambda: calls.append("correction refreshed"),
    )

    widget._on_dims_step_changed()

    assert calls == []
    _app.processEvents()
    assert calls == ["correction refreshed"]

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


def test_nucleus_state_persists_atom_union_params():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    widget.atom_union_max_atoms_spin.setValue(5)
    widget.atom_union_max_area_spin.setValue(4242)

    state = widget.get_state()
    db_state = state["db_generation"]

    assert db_state["max_atoms"] == 5
    assert db_state["atom_union_max_area"] == 4242
    assert "threshold_pairs" not in db_state
    assert "max_area" not in db_state

    restored = widget_class(viewer)
    restored.set_state(state)

    assert restored.atom_union_max_atoms_spin.value() == 5
    assert restored.atom_union_max_area_spin.value() == 4242

    restored.deleteLater()
    widget.deleteLater()
    viewer.close()


def test_nucleus_state_does_not_persist_validated_corrections_as_solve_option():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    state = widget.get_state()

    assert "use_validated" not in state["db_generation"]
    assert "resolve_from_validated" not in state["ultrack"]

    restored = widget_class(viewer)
    restored.set_state(state)

    assert not hasattr(restored, "solve_use_validated_check")

    restored.deleteLater()
    widget.deleteLater()
    viewer.close()


def test_nucleus_state_loads_legacy_db_generation_validated_option():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    widget.set_state({"db_generation": {"use_validated": True}})

    assert "use_validated" not in widget.get_state()["db_generation"]
    assert not hasattr(widget, "solve_use_validated_check")

    widget.deleteLater()
    viewer.close()


def test_nucleus_state_loads_legacy_validated_options_without_solve_checkbox():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    widget.set_state(
        {
            "db_generation": {"use_validated": True},
            "ultrack": {"resolve_from_validated": True},
        }
    )

    assert "resolve_from_validated" not in widget.get_state()["ultrack"]
    assert "use_validated" not in widget.get_state()["db_generation"]
    assert not hasattr(widget, "solve_use_validated_check")

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


def test_run_db_generation_calls_build_atom_union_database(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    _install_sync_thread_worker(monkeypatch, pipeline_module)

    widget.atom_union_max_atoms_spin.setValue(4)
    widget.atom_union_max_area_spin.setValue(7000)

    calls = []

    def fake_build_atom_union_database(atoms_path, working_dir, cfg, progress_cb=None):
        calls.append((atoms_path, working_dir, cfg))
        data_db = Path(working_dir) / "data.db"
        data_db.parent.mkdir(parents=True, exist_ok=True)
        data_db.write_bytes(b"sqlite placeholder")
        return types.SimpleNamespace()

    monkeypatch.setattr(
        pipeline_module,
        "build_atom_union_database",
        fake_build_atom_union_database,
    )
    monkeypatch.setattr(pipeline_module, "apply_annotations_and_score", lambda **kwargs: None)
    monkeypatch.setattr(pipeline_module, "_ultrack_available", lambda: True)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    import tifffile
    tifffile.imwrite(
        str(pos_dir / "2_nucleus" / "atoms.tif"),
        np.zeros((2, 4, 4), dtype=np.int32),
    )
    widget._pos_dir = pos_dir

    widget._on_run_db_generation()

    assert len(calls) == 1
    atoms_path, working_dir, cfg = calls[0]
    assert atoms_path == pos_dir / "2_nucleus" / "atoms.tif"
    assert working_dir == pos_dir / "2_nucleus" / "ultrack_workdir"
    assert cfg.atom_union_max_atoms == 4
    assert cfg.atom_union_max_area == 7000
    assert widget.db_run_btn.isEnabled()
    assert widget.db_run_btn.text() == "▶"
    assert "complete" in widget.pipeline_status_lbl.text().lower()
    assert not widget.pipeline_progress_bar.isVisible()

    widget.deleteLater()
    viewer.close()


def test_run_db_generation_reports_missing_atoms_tif(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    monkeypatch.setattr(pipeline_module, "_ultrack_available", lambda: True)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    widget._pos_dir = pos_dir

    widget._on_run_db_generation()

    assert "atoms.tif" in widget.pipeline_status_lbl.text()

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


def test_run_ultrack_solve_does_not_apply_annotations_or_require_tracked_labels(
    tmp_path,
    monkeypatch,
):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    _install_sync_thread_worker(monkeypatch, pipeline_module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus" / "ultrack_workdir").mkdir(parents=True)
    (pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db").write_bytes(
        b"sqlite placeholder"
    )
    widget._pos_dir = pos_dir

    calls = []
    labels = np.ones((2, 4, 4), dtype=np.uint32)
    monkeypatch.setattr(pipeline_module, "read_corrections", lambda _pos_dir: [])
    monkeypatch.setattr(pipeline_module, "read_validated_tracks", lambda _pos_dir: {})
    monkeypatch.setattr(
        pipeline_module,
        "apply_annotations_and_score",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("solve path should not apply annotations")
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_solve",
        lambda working_dir, cfg, overwrite: calls.append(
            ("solve", working_dir, cfg, overwrite)
        )
        or iter([(1, 1, "solved")]),
    )
    monkeypatch.setattr(
        pipeline_module,
        "export_tracked_labels",
        lambda working_dir, cfg, tracked_path, **kwargs: calls.append(
            ("export", working_dir, cfg, tracked_path, kwargs)
        )
        or labels,
    )

    widget._on_run_ultrack()

    assert [call[0] for call in calls] == ["solve", "export"]
    assert "Tracked: Nucleus" in viewer.layers
    assert "tracked_labels" not in widget.pipeline_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_run_ultrack_export_receives_saved_validation_geometry_without_solve_checkbox(
    tmp_path,
    monkeypatch,
):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    _install_sync_thread_worker(monkeypatch, pipeline_module)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus" / "ultrack_workdir").mkdir(parents=True)
    (pos_dir / "1_cellpose").mkdir()
    (pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db").write_bytes(
        b"sqlite placeholder"
    )
    import tifffile

    tracked_labels = np.zeros((2, 4, 4), dtype=np.uint32)
    tracked_labels[0, 1:3, 1:3] = 7
    tracked_labels[1, 2:4, 2:4] = 8
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_foreground.tif", tracked_labels)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", tracked_labels)
    widget._pos_dir = pos_dir

    from cellflow.tracking_ultrack.corrections import Correction

    correction = Correction(cell_id=7, t=0, kind="validated", y=1.5, x=1.5)
    captured = {}
    monkeypatch.setattr(pipeline_module, "read_corrections", lambda _pos_dir: [correction])
    monkeypatch.setattr(pipeline_module, "read_validated_tracks", lambda _pos_dir: {8: {1}})
    monkeypatch.setattr(
        pipeline_module,
        "apply_annotations_and_score",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("solve path should not apply annotations")
        ),
    )
    monkeypatch.setattr(pipeline_module, "run_solve", lambda *a, **kw: iter(()))

    def fake_export(_working_dir, _cfg, _tracked_path, **kwargs):
        captured.update(kwargs)
        return tracked_labels

    monkeypatch.setattr(pipeline_module, "export_tracked_labels", fake_export)

    widget._on_run_ultrack()

    assert [(c.cell_id, c.t, c.kind) for c in captured["corrections"]] == [
        (7, 0, "validated"),
        (8, 1, "validated"),
    ]
    assert captured["validated_tracks"] is None
    np.testing.assert_array_equal(captured["tracked_labels"], tracked_labels)

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
