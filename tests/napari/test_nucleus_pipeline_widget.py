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
from qtpy.QtWidgets import QApplication, QProgressBar


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
            "preview_ultrack_source_stack_frame": lambda *args, **kwargs: (None, None, 0, []),
            "write_ultrack_source_stacks": lambda *args, **kwargs: [],
        },
        "cellflow.tracking_ultrack.extend": {
            "extend_track_from_db": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.solve": {
            "run_solve": lambda *args, **kwargs: iter(()),
        },
        "cellflow.segmentation": {
            "apply_gamma": lambda logits, gamma: logits,
            "build_nucleus_averaged_maps": lambda *args, **kwargs: None,
            "build_consensus_boundary": lambda *args, **kwargs: (None, None),
        },
    }

    for module_name, attrs in stub_exports.items():
        mod = types.ModuleType(module_name)
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
    assert widget.preview_contour_btn is widget.nucleus_pipeline_widget.preview_contour_btn
    assert widget.build_btn is widget.nucleus_pipeline_widget.build_btn
    assert widget.run_db_gen_btn is widget.nucleus_pipeline_widget.run_db_gen_btn
    assert widget.run_ultrack_btn is widget.nucleus_pipeline_widget.run_ultrack_btn
    assert widget.cancel_btn is widget.nucleus_pipeline_widget.cancel_btn
    assert widget.pipeline_status_lbl is widget.nucleus_pipeline_widget.pipeline_status_lbl
    assert widget.pipeline_progress_bar is widget.nucleus_pipeline_widget.pipeline_progress_bar

    widget.deleteLater()
    viewer.close()


def test_pipeline_widget_handler_methods_aliased_on_workflow():
    """Pipeline handler methods are reachable on the workflow widget via instance aliases."""
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    pipeline_module = _get_pipeline_module()
    widget = widget_class(viewer)
    pl = widget.nucleus_pipeline_widget

    for method_name in (
        "_on_build_segmentation_inputs",
        "_on_build_contour_maps",
        "_on_preview_contour_maps",
        "_on_run_db_generation",
        "_on_run_ultrack",
        "_on_cancel",
        "_status",
        "_set_pipeline_buttons_enabled",
    ):
        widget_method = getattr(widget, method_name)
        pl_method = getattr(pl, method_name)
        # Instance aliases set via setattr store bound methods; compare via
        # __func__/__self__ rather than identity because each getattr on a
        # class creates a new bound-method wrapper.
        assert getattr(widget_method, "__func__", widget_method) is getattr(pl_method, "__func__", pl_method), (
            f"{method_name} not aliased to pipeline widget"
        )

    widget.deleteLater()
    viewer.close()


# ── Structure tests ───────────────────────────────────────────────────────────


def test_pipeline_widget_initial_cancel_button_disabled():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    assert not widget.cancel_btn.isEnabled()
    assert widget.run_db_gen_btn.isEnabled()
    assert widget.run_ultrack_btn.isEnabled()

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
    # isVisible() returns False for unshown top-level widgets even after
    # setVisible(True); use isHidden() to check the widget's own visibility flag.
    assert not widget.pipeline_status_lbl.isHidden()

    widget._status("")
    assert widget.pipeline_status_lbl.isHidden()

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


def test_set_pipeline_buttons_enabled_toggles_cancel():
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)

    widget._set_pipeline_buttons_enabled(False)
    assert not widget.build_btn.isEnabled()
    assert not widget.run_db_gen_btn.isEnabled()
    assert widget.cancel_btn.isEnabled()

    widget._set_pipeline_buttons_enabled(True)
    assert widget.build_btn.isEnabled()
    assert widget.run_db_gen_btn.isEnabled()
    assert not widget.cancel_btn.isEnabled()

    widget.deleteLater()
    viewer.close()


# ── Handler tests ─────────────────────────────────────────────────────────────


def test_build_contour_maps_calls_write_source_stacks(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    _install_sync_thread_worker(monkeypatch, pipeline_module)

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

    monkeypatch.setattr(pipeline_module, "write_ultrack_source_stacks", fake_write)

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
    assert not widget.pipeline_progress_bar.isVisible()

    widget.deleteLater()
    viewer.close()


def test_run_db_generation_calls_build_database(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_workflow_widget_class()
    widget = widget_class(viewer)
    pipeline_module = _get_pipeline_module()
    _install_sync_thread_worker(monkeypatch, pipeline_module)

    calls = []

    def fake_build_database(**kwargs):
        calls.append(kwargs)
        data_db = kwargs["working_dir"] / "data.db"
        data_db.parent.mkdir(parents=True, exist_ok=True)
        data_db.write_bytes(b"sqlite placeholder")
        return {"database": str(data_db)}

    monkeypatch.setattr(pipeline_module, "build_ultrack_database_from_sources", fake_build_database)
    monkeypatch.setattr(pipeline_module, "_ultrack_segment", object(), raising=False)

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
    assert "score_signal_path" not in call
    assert call["cfg"].seg_foreground_threshold == pytest.approx(0.0)
    assert widget.run_db_gen_btn.isEnabled()
    assert "complete" in widget.pipeline_status_lbl.text().lower()
    assert not widget.pipeline_progress_bar.isVisible()

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
    widget.cancel_btn.setEnabled(True)

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
    assert widget.run_db_gen_btn.isEnabled()
    assert not widget.cancel_btn.isEnabled()
    assert "Cancelled" in widget.pipeline_status_lbl.text()

    widget.deleteLater()
    viewer.close()
