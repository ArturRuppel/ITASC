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

    def add_labels(self, data, *, name, **kwargs):
        layer = types.SimpleNamespace(data=data, name=name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_image(self, data, *, name, **kwargs):
        layer = types.SimpleNamespace(data=data, name=name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_shapes(self, data, *, name, **kwargs):
        layer = types.SimpleNamespace(data=data, name=name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_tracks(self, data, *, name, **kwargs):
        layer = types.SimpleNamespace(data=np.asarray(data), name=name, **kwargs)
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
    sys.modules.pop("cellflow.napari.contact_analysis_widget", None)
    sys.modules.pop("cellflow.napari.nls_classification_widget", None)
    return importlib.import_module("cellflow.napari.contact_analysis_widget")


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


def test_contact_analysis_widget_refresh_tracks_inputs_output_and_button_states(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.ContactAnalysisWidget()

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()

    widget.refresh(pos_dir)

    assert widget._pos_dir == pos_dir
    assert widget.contact_analysis_out_path == pos_dir / "4_contact_analysis" / "contact_analysis.h5"
    assert hasattr(widget, "_files_widget")
    assert widget.build_contact_analysis_btn.isEnabled() is False
    assert widget.show_contact_analysis_btn.isEnabled() is False
    assert widget.clear_contact_analysis_btn.isEnabled() is False

    (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    widget.refresh(pos_dir)

    assert widget.build_contact_analysis_btn.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_does_not_embed_personal_nls_classification(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.ContactAnalysisWidget(viewer)

    pos_dir = tmp_path / "pos04"
    widget.refresh(pos_dir)

    assert not hasattr(widget, "nls_classification_widget")
    assert "cellflow.napari.nls_classification_widget" not in sys.modules

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_build_runs_in_worker_and_reports_progress(monkeypatch, tmp_path):
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
        assert widget.contact_analysis_progress_bar.maximum() == 5
        assert widget.contact_analysis_progress_bar.value() == 2
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"h5")
        return output_path

    monkeypatch.setattr(mod, "build_position_contact_analysis", fake_build)

    widget = mod.ContactAnalysisWidget()
    widget.refresh(pos_dir)
    widget._on_build_contact_analysis()

    assert progress_events == [(2, 5, "Indexing records")]
    assert captured["position_path"] == pos_dir
    assert captured["output_path"] == pos_dir / "4_contact_analysis" / "contact_analysis.h5"
    assert captured["kwargs"]["cell_tracked_labels_path"] == pos_dir / "3_cell" / "tracked_labels.tif"
    assert captured["kwargs"]["nucleus_tracked_labels_path"] == pos_dir / "2_nucleus" / "tracked_labels.tif"
    assert callable(captured["kwargs"]["progress_cb"])
    assert "Wrote" in widget.contact_analysis_status_lbl.text()
    assert widget.contact_analysis_out_path.exists()

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_cancel_calls_worker_quit_when_active(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.ContactAnalysisWidget()

    worker = _FakeWorker()
    widget._build_worker = worker
    widget.cancel_build_btn.setEnabled(True)

    widget._on_cancel_build()

    assert worker.quit_calls == 1
    assert widget._build_worker is None

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_shows_and_clears_contact_analysis_layers(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.ContactAnalysisWidget(viewer)

    pos_dir = tmp_path / "pos08"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    contact_analysis_path = pos_dir / "4_contact_analysis" / "contact_analysis.h5"
    contact_analysis_path.parent.mkdir(parents=True, exist_ok=True)
    contact_analysis_path.write_bytes(b"h5")
    widget.refresh(pos_dir)

    assert widget.show_contact_analysis_btn.isEnabled() is True
    assert widget.clear_contact_analysis_btn.isEnabled() is True

    contact_analysis = {"cells": [1, 2, 3]}
    read_calls = []
    add_calls = []

    def fake_read(path):
        read_calls.append(path)
        return contact_analysis

    def fake_add(viewer_arg, contact_analysis_arg, **kwargs):
        prefix = kwargs["prefix"]
        add_calls.append((viewer_arg, contact_analysis_arg, kwargs))
        viewer_arg.layers[f"{prefix}Cells"] = types.SimpleNamespace(
            name=f"{prefix}Cells"
        )
        viewer_arg.layers[f"{prefix}Edges"] = types.SimpleNamespace(
            name=f"{prefix}Edges"
        )
        viewer_arg.layers["Background"] = types.SimpleNamespace(name="Background")

    monkeypatch.setattr(mod, "read_position_contact_analysis", fake_read)
    monkeypatch.setattr(mod, "add_contact_analysis_layers", fake_add)

    widget.show_contact_analysis_btn.click()

    assert read_calls == [contact_analysis_path]
    assert add_calls == [
        (
            viewer,
            contact_analysis,
            {
                "prefix": "[Contact Analysis] ",
                "color_cells_by_label": False,
                "color_edges_by_id": False,
                "color_edges_by_label": False,
                "hide_border_edges": False,
            },
        )
    ]
    assert f"[Contact Analysis] Cells" in viewer.layers
    assert "Background" in viewer.layers

    widget.clear_contact_analysis_btn.click()

    assert "[Contact Analysis] Cells" not in viewer.layers
    assert "[Contact Analysis] Edges" not in viewer.layers
    assert "Background" in viewer.layers

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_forwards_visualizer_options(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.ContactAnalysisWidget(viewer)

    pos_dir = tmp_path / "pos10"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    contact_analysis_path = pos_dir / "4_contact_analysis" / "contact_analysis.h5"
    contact_analysis_path.parent.mkdir(parents=True, exist_ok=True)
    contact_analysis_path.write_bytes(b"h5")
    widget.refresh(pos_dir)

    contact_analysis = {"cells": [1]}
    add_calls = []
    monkeypatch.setattr(mod, "read_position_contact_analysis", lambda _path: contact_analysis)

    def fake_add(viewer_arg, contact_analysis_arg, **kwargs):
        add_calls.append((viewer_arg, contact_analysis_arg, kwargs))
        viewer_arg.layers[f"{kwargs['prefix']}Cells"] = types.SimpleNamespace(
            name=f"{kwargs['prefix']}Cells"
        )

    monkeypatch.setattr(mod, "add_contact_analysis_layers", fake_add)

    widget.color_cells_by_label_cb.setChecked(True)
    widget.color_edges_by_id_cb.setChecked(True)
    widget.color_edges_by_label_cb.setChecked(True)
    widget.hide_border_edges_cb.setChecked(True)
    widget.show_contact_analysis_btn.click()

    assert add_calls == [
        (
            viewer,
            contact_analysis,
            {
                "prefix": "[Contact Analysis] ",
                "color_cells_by_label": True,
                "color_edges_by_id": True,
                "color_edges_by_label": True,
                "hide_border_edges": True,
            },
        )
    ]

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_checkbox_does_not_live_update_visualization(monkeypatch, tmp_path):
    """Checkboxes no longer trigger automatic reloads; user must click Show Contact Analysis."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.ContactAnalysisWidget(viewer)

    pos_dir = tmp_path / "pos11"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    contact_analysis_path = pos_dir / "4_contact_analysis" / "contact_analysis.h5"
    contact_analysis_path.parent.mkdir(parents=True, exist_ok=True)
    contact_analysis_path.write_bytes(b"h5")
    widget.refresh(pos_dir)

    contact_analysis = {"cells": [1]}
    monkeypatch.setattr(mod, "read_position_contact_analysis", lambda _path: contact_analysis)
    add_calls = []

    def fake_add(viewer_arg, contact_analysis_arg, **kwargs):
        add_calls.append(kwargs)
        viewer_arg.layers[f"{kwargs['prefix']}Cells"] = types.SimpleNamespace(
            name=f"{kwargs['prefix']}Cells"
        )

    monkeypatch.setattr(mod, "add_contact_analysis_layers", fake_add)

    # First show with default settings
    widget.show_contact_analysis_btn.click()
    # Changing checkbox must NOT trigger a reload on its own
    widget.color_cells_by_label_cb.setChecked(True)
    assert len(add_calls) == 1, "checkbox change must not auto-reload"
    assert add_calls[0]["color_cells_by_label"] is False

    # Clicking Show Contact Analysis again picks up the updated checkbox state
    widget.show_contact_analysis_btn.click()
    assert len(add_calls) == 2
    assert add_calls[1]["color_cells_by_label"] is True

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_show_contact_analysis_uses_real_reader_and_visualizer(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.ContactAnalysisWidget(viewer)

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
    mod.build_position_contact_analysis(
        pos_dir,
        widget.contact_analysis_out_path,
        cell_tracked_labels_path=pos_dir / "3_cell" / "tracked_labels.tif",
        nucleus_tracked_labels_path=pos_dir / "2_nucleus" / "tracked_labels.tif",
    )
    widget.refresh(pos_dir)

    widget._on_show_contact_analysis()

    assert "[Contact Analysis] Cell labels" in viewer.layers
    assert "[Contact Analysis] Nucleus labels" in viewer.layers
    assert "[Contact Analysis] Edges" in viewer.layers
    assert "[Contact Analysis] T1 edges" in viewer.layers
    assert viewer.layers["[Contact Analysis] Cell labels"].data.shape == (1, 4, 4)
    assert viewer.layers["[Contact Analysis] Nucleus labels"].data.shape == (1, 4, 4)
    assert len(viewer.layers["[Contact Analysis] Edges"].data) >= 1

    widget.deleteLater()
    app.processEvents()
