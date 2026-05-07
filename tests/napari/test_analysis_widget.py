from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication


class _LayerCollection(dict):
    def remove(self, layer):
        name = getattr(layer, "name", layer)
        self.pop(name, None)


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()

    def add_points(self, data, *, name, **kwargs):
        layer = types.SimpleNamespace(data=data, name=name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_shapes(self, data, *, name, **kwargs):
        layer = types.SimpleNamespace(data=data, name=name, **kwargs)
        self.layers[name] = layer
        return layer


class _FakeWorker:
    def __init__(self) -> None:
        self.quit_calls = 0

    def quit(self) -> None:
        self.quit_calls += 1


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.analysis_widget", None)
    return importlib.import_module("cellflow.napari.analysis_widget")


def _make_sync_thread_worker():
    def fake_thread_worker(connect=None):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    if connect and "errored" in connect:
                        connect["errored"](exc)
                    return _FakeWorker()

                if inspect.isgenerator(result):
                    return_value = None
                    while True:
                        try:
                            yielded = next(result)
                        except StopIteration as exc:
                            return_value = exc.value
                            break
                        if connect and "yielded" in connect:
                            connect["yielded"](yielded)
                    if connect and "returned" in connect:
                        connect["returned"](return_value)
                else:
                    if connect and "returned" in connect:
                        connect["returned"](result)
                if connect and "finished" in connect:
                    connect["finished"]()
                return _FakeWorker()

            return wrapper

        return decorator

    return fake_thread_worker


def test_analysis_widget_refresh_tracks_inputs_output_and_button_states(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.AnalysisWidget()

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()

    widget.refresh(pos_dir)

    assert widget._pos_dir == pos_dir
    assert widget.artifact_out_path == pos_dir / "4_analysis" / "position_analysis.h5"
    assert widget.artifact_path_lbl.text() == f"Output: {pos_dir / '4_analysis' / 'position_analysis.h5'}"
    assert "cell labels" in widget.input_status_lbl.text()
    assert "nucleus labels" in widget.input_status_lbl.text()
    assert widget.build_artifact_btn.isEnabled() is False
    assert widget.show_artifact_btn.isEnabled() is False
    assert widget.clear_artifact_btn.isEnabled() is False

    (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    widget.refresh(pos_dir)

    assert widget.build_artifact_btn.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


def test_analysis_widget_build_runs_in_worker_and_reports_progress(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos03"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "3_cell" / "tracked_labels.tif").touch()

    progress_events: list[tuple[int, int, str]] = []
    captured: dict[str, object] = {}

    def fake_build(position_path, output_path, **kwargs):
        captured["position_path"] = position_path
        captured["output_path"] = output_path
        captured["kwargs"] = kwargs
        progress_cb = kwargs["progress_cb"]
        progress_cb(2, 5, "Indexing records")
        progress_events.append((2, 5, "Indexing records"))
        assert widget.artifact_progress_bar.maximum() == 5
        assert widget.artifact_progress_bar.value() == 2
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"h5")
        return output_path

    monkeypatch.setattr(mod, "build_position_analysis_artifact", fake_build)

    widget = mod.AnalysisWidget()
    widget.refresh(pos_dir)
    widget._on_build_artifact()

    assert progress_events == [(2, 5, "Indexing records")]
    assert captured["position_path"] == pos_dir
    assert captured["output_path"] == pos_dir / "4_analysis" / "position_analysis.h5"
    assert captured["kwargs"]["cell_tracked_labels_path"] == pos_dir / "3_cell" / "tracked_labels.tif"
    assert captured["kwargs"]["nucleus_tracked_labels_path"] == pos_dir / "2_nucleus" / "tracked_labels.tif"
    assert callable(captured["kwargs"]["progress_cb"])
    assert "Wrote" in widget.artifact_status_lbl.text()
    assert widget.artifact_out_path.exists()

    widget.deleteLater()
    app.processEvents()


def test_analysis_widget_cancel_calls_worker_quit_when_active(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.AnalysisWidget()

    worker = _FakeWorker()
    widget._build_worker = worker
    widget.cancel_build_btn.setEnabled(True)

    widget._on_cancel_build()

    assert worker.quit_calls == 1
    assert widget._build_worker is None

    widget.deleteLater()
    app.processEvents()


def test_analysis_widget_shows_and_clears_artifact_layers(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.AnalysisWidget(viewer)

    pos_dir = tmp_path / "pos08"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    artifact_path = pos_dir / "4_analysis" / "position_analysis.h5"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"h5")
    widget.refresh(pos_dir)

    assert widget.show_artifact_btn.isEnabled() is True
    assert widget.clear_artifact_btn.isEnabled() is True

    artifact = {"cells": [1, 2, 3]}
    read_calls = []
    add_calls = []

    def fake_read(path):
        read_calls.append(path)
        return artifact

    def fake_add(viewer_arg, artifact_arg, *, prefix):
        add_calls.append((viewer_arg, artifact_arg, prefix))
        viewer_arg.layers[f"{prefix}Cells"] = types.SimpleNamespace(
            name=f"{prefix}Cells"
        )
        viewer_arg.layers[f"{prefix}Edges"] = types.SimpleNamespace(
            name=f"{prefix}Edges"
        )
        viewer_arg.layers["Background"] = types.SimpleNamespace(name="Background")

    monkeypatch.setattr(mod, "read_position_artifact", fake_read)
    monkeypatch.setattr(mod, "add_artifact_layers", fake_add)

    widget.show_artifact_btn.click()

    assert read_calls == [artifact_path]
    assert add_calls == [(viewer, artifact, "[Artifact] ")]
    assert f"[Artifact] Cells" in viewer.layers
    assert "Background" in viewer.layers

    widget.clear_artifact_btn.click()

    assert "[Artifact] Cells" not in viewer.layers
    assert "[Artifact] Edges" not in viewer.layers
    assert "Background" in viewer.layers

    widget.deleteLater()
    app.processEvents()


def test_analysis_widget_show_artifact_uses_real_reader_and_visualizer(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.AnalysisWidget(viewer)

    pos_dir = tmp_path / "pos09"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    labels = np.zeros((1, 4, 4), dtype=np.uint16)
    labels[0, :, :2] = 1
    labels[0, :, 2:] = 2

    import tifffile

    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", labels)
    tifffile.imwrite(pos_dir / "3_cell" / "tracked_labels.tif", labels)
    widget.refresh(pos_dir)
    mod.build_position_analysis_artifact(
        pos_dir,
        widget.artifact_out_path,
        cell_tracked_labels_path=pos_dir / "3_cell" / "tracked_labels.tif",
        nucleus_tracked_labels_path=pos_dir / "2_nucleus" / "tracked_labels.tif",
    )
    widget.refresh(pos_dir)

    widget.show_artifact_btn.click()

    assert "[Artifact] Cell centroids" in viewer.layers
    assert "[Artifact] Edges" in viewer.layers
    assert "[Artifact] T1 events" in viewer.layers
    assert viewer.layers["[Artifact] Cell centroids"].data.shape == (2, 3)
    assert len(viewer.layers["[Artifact] Edges"].data) >= 1

    widget.deleteLater()
    app.processEvents()
