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


def _set_pos(widget, pos_dir):
    """Drive the widget the way the orchestrator does: staged paths + status root."""
    widget.set_context(
        cell_labels=pos_dir / "3_cell" / "tracked_labels.tif",
        nucleus_labels=pos_dir / "2_nucleus" / "tracked_labels.tif",
        out_path=pos_dir / "4_contact_analysis" / "contact_analysis.h5",
        status_root=pos_dir,
    )


def _staged_pos(tmp_path, name, *, cell=True, nucleus=True, h5=False):
    pos_dir = tmp_path / name
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    if nucleus:
        (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    if cell:
        (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    if h5:
        out = pos_dir / "4_contact_analysis" / "contact_analysis.h5"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"h5")
    return pos_dir


def test_contact_analysis_widget_refresh_tracks_inputs_output_and_button_states(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.ContactAnalysisWidget()

    pos_dir = _staged_pos(tmp_path, "pos00", cell=False, nucleus=True)
    _set_pos(widget, pos_dir)

    assert widget.cell_labels_path == pos_dir / "3_cell" / "tracked_labels.tif"
    assert widget.nucleus_labels_path == pos_dir / "2_nucleus" / "tracked_labels.tif"
    assert widget.contact_analysis_out_path == pos_dir / "4_contact_analysis" / "contact_analysis.h5"
    assert hasattr(widget, "_files_widget")
    # Cell labels not on disk yet -> Visualize/Recompute disabled.
    assert widget.visualize_btn.isEnabled() is False
    assert widget.recompute_btn.isEnabled() is False
    assert widget.clear_contact_analysis_btn.isEnabled() is False

    (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    _set_pos(widget, pos_dir)

    assert widget.visualize_btn.isEnabled() is True
    assert widget.recompute_btn.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_does_not_embed_personal_nls_classification(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.ContactAnalysisWidget(viewer)

    pos_dir = tmp_path / "pos04"
    _set_pos(widget, pos_dir)

    assert not hasattr(widget, "nls_classification_widget")
    assert "cellflow.napari.nls_classification_widget" not in sys.modules

    widget.deleteLater()
    app.processEvents()


def test_visualize_builds_when_missing_then_shows(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = _staged_pos(tmp_path, "pos03", cell=True, nucleus=True)

    progress_events: list[tuple[int, int, str]] = []
    captured: dict[str, object] = {}

    def fake_ensure(*, cell_labels_path, output_path, nucleus_labels_path=None,
                    overwrite=False, progress_cb=None, **kwargs):
        captured["cell_labels_path"] = cell_labels_path
        captured["output_path"] = output_path
        captured["nucleus_labels_path"] = nucleus_labels_path
        captured["overwrite"] = overwrite
        progress_cb(2, 5, "Indexing records")
        progress_events.append((2, 5, "Indexing records"))
        assert widget.contact_analysis_progress_bar.maximum() == 5
        assert widget.contact_analysis_progress_bar.value() == 2
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"h5")
        return output_path, True

    add_calls = []

    def fake_add(viewer_arg, contact_analysis_arg, **kwargs):
        add_calls.append(kwargs)
        viewer_arg.layers[f"{kwargs['prefix']}Cells"] = types.SimpleNamespace(
            name=f"{kwargs['prefix']}Cells"
        )

    monkeypatch.setattr(mod, "ensure_contact_analysis", fake_ensure)
    monkeypatch.setattr(mod, "read_position_contact_analysis", lambda _p: {"cells": [1]})
    monkeypatch.setattr(mod, "add_contact_analysis_layers", fake_add)

    viewer = _FakeViewer()
    widget = mod.ContactAnalysisWidget(viewer)
    _set_pos(widget, pos_dir)
    widget._on_visualize(overwrite=False)

    # Built (missing -> compute) then showed.
    assert progress_events == [(2, 5, "Indexing records")]
    assert captured["output_path"] == pos_dir / "4_contact_analysis" / "contact_analysis.h5"
    assert captured["cell_labels_path"] == pos_dir / "3_cell" / "tracked_labels.tif"
    assert captured["nucleus_labels_path"] == pos_dir / "2_nucleus" / "tracked_labels.tif"
    assert captured["overwrite"] is False
    assert widget.contact_analysis_out_path.exists()
    assert len(add_calls) == 1
    assert "[Contact Analysis] Cells" in viewer.layers

    widget.deleteLater()
    app.processEvents()


def test_visualize_uses_existing_h5_without_rebuild(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = _staged_pos(tmp_path, "pos05", cell=True, nucleus=True, h5=True)

    def boom(*args, **kwargs):
        raise AssertionError("ensure_contact_analysis must not be called when .h5 exists")

    add_calls = []

    def fake_add(viewer_arg, contact_analysis_arg, **kwargs):
        add_calls.append(kwargs)

    monkeypatch.setattr(mod, "ensure_contact_analysis", boom)
    monkeypatch.setattr(mod, "read_position_contact_analysis", lambda _p: {"cells": [1]})
    monkeypatch.setattr(mod, "add_contact_analysis_layers", fake_add)

    viewer = _FakeViewer()
    widget = mod.ContactAnalysisWidget(viewer)
    _set_pos(widget, pos_dir)
    widget.visualize_btn.click()

    # Fast path: no rebuild, straight to show.
    assert len(add_calls) == 1

    widget.deleteLater()
    app.processEvents()


def test_recompute_forces_rebuild_even_when_h5_exists(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = _staged_pos(tmp_path, "pos06", cell=True, nucleus=True, h5=True)

    captured: dict[str, object] = {}

    def fake_ensure(*, cell_labels_path, output_path, nucleus_labels_path=None,
                    overwrite=False, progress_cb=None, **kwargs):
        captured["overwrite"] = overwrite
        return output_path, True

    monkeypatch.setattr(mod, "ensure_contact_analysis", fake_ensure)
    monkeypatch.setattr(mod, "read_position_contact_analysis", lambda _p: {"cells": [1]})
    monkeypatch.setattr(mod, "add_contact_analysis_layers", lambda *a, **k: None)

    viewer = _FakeViewer()
    widget = mod.ContactAnalysisWidget(viewer)
    _set_pos(widget, pos_dir)
    widget.recompute_btn.click()

    assert captured["overwrite"] is True

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

    pos_dir = _staged_pos(tmp_path, "pos08", cell=True, nucleus=True, h5=True)
    contact_analysis_path = pos_dir / "4_contact_analysis" / "contact_analysis.h5"
    _set_pos(widget, pos_dir)

    assert widget.visualize_btn.isEnabled() is True
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

    widget.visualize_btn.click()

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
    assert "[Contact Analysis] Cells" in viewer.layers
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

    pos_dir = _staged_pos(tmp_path, "pos10", cell=True, nucleus=True, h5=True)
    _set_pos(widget, pos_dir)

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
    widget.visualize_btn.click()

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
    """Checkboxes no longer trigger automatic reloads; the user must click Visualize."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.ContactAnalysisWidget(viewer)

    pos_dir = _staged_pos(tmp_path, "pos11", cell=True, nucleus=True, h5=True)
    _set_pos(widget, pos_dir)

    contact_analysis = {"cells": [1]}
    monkeypatch.setattr(mod, "read_position_contact_analysis", lambda _path: contact_analysis)
    add_calls = []

    def fake_add(viewer_arg, contact_analysis_arg, **kwargs):
        add_calls.append(kwargs)
        viewer_arg.layers[f"{kwargs['prefix']}Cells"] = types.SimpleNamespace(
            name=f"{kwargs['prefix']}Cells"
        )

    monkeypatch.setattr(mod, "add_contact_analysis_layers", fake_add)

    # First Visualize with default settings
    widget.visualize_btn.click()
    # Changing a checkbox must NOT trigger a reload on its own
    widget.color_cells_by_label_cb.setChecked(True)
    assert len(add_calls) == 1, "checkbox change must not auto-reload"
    assert add_calls[0]["color_cells_by_label"] is False

    # Clicking Visualize again picks up the updated checkbox state
    widget.visualize_btn.click()
    assert len(add_calls) == 2
    assert add_calls[1]["color_cells_by_label"] is True

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_show_uses_real_reader_and_visualizer(monkeypatch, tmp_path):
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
    _set_pos(widget, pos_dir)
    mod.build_contact_analysis(
        cell_labels_path=pos_dir / "3_cell" / "tracked_labels.tif",
        output_path=widget.contact_analysis_out_path,
        nucleus_labels_path=pos_dir / "2_nucleus" / "tracked_labels.tif",
    )
    _set_pos(widget, pos_dir)

    widget._show_from_disk()

    assert "[Contact Analysis] Cell labels" in viewer.layers
    assert "[Contact Analysis] Nucleus labels" in viewer.layers
    assert "[Contact Analysis] Edges" in viewer.layers
    assert "[Contact Analysis] T1 edges" in viewer.layers
    assert viewer.layers["[Contact Analysis] Cell labels"].data.shape == (1, 4, 4)
    assert viewer.layers["[Contact Analysis] Nucleus labels"].data.shape == (1, 4, 4)
    assert len(viewer.layers["[Contact Analysis] Edges"].data) >= 1

    widget.deleteLater()
    app.processEvents()


def _flat_pos(tmp_path, name, *, nucleus=False, h5=False):
    pos = tmp_path / name
    pos.mkdir(parents=True)
    labels = np.zeros((1, 4, 4), dtype=np.uint16)
    labels[0, :, :2] = 1
    labels[0, :, 2:] = 2
    import tifffile

    tifffile.imwrite(pos / "cell_labels.tif", labels)
    if nucleus:
        tifffile.imwrite(pos / "nucleus_labels.tif", labels)
    if h5:
        (pos / "contact_analysis.h5").write_bytes(b"h5")
    return pos


def _discover(widget, root):
    widget._batch_root_edit.setText(str(root))
    widget._rediscover()


def test_standalone_shows_discovery_panel_and_hides_staged(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.ContactAnalysisWidget(standalone=True)

    assert widget._discovery_container.isVisibleTo(widget) is True
    assert widget._pipeline_files_section.isVisibleTo(widget) is False
    # The old single-file pickers are gone.
    assert not hasattr(widget, "_pickers_container")

    widget.deleteLater()
    app.processEvents()


def test_discovery_populates_list_with_status(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.ContactAnalysisWidget(standalone=True)

    _flat_pos(tmp_path, "posA", nucleus=True, h5=True)
    _flat_pos(tmp_path, "posB", nucleus=False)
    _discover(widget, tmp_path)

    assert widget._discovery_list.count() == 2
    labels = [widget._discovery_list.item(i).text() for i in range(2)]
    assert labels[0] == "posA    cell+nucleus    [built]"
    assert labels[1] == "posB    cell only    [missing]"

    widget.deleteLater()
    app.processEvents()


def test_double_click_visualizes_and_computes_when_missing(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    _flat_pos(tmp_path, "posB", nucleus=False)  # no .h5 yet

    captured = {}

    def fake_ensure(*, cell_labels_path, output_path, nucleus_labels_path=None,
                    overwrite=False, progress_cb=None, **kwargs):
        captured["output_path"] = output_path
        captured["overwrite"] = overwrite
        output_path.write_bytes(b"h5")
        return output_path, True

    add_calls = []

    def fake_add(viewer_arg, contact_analysis_arg, **kwargs):
        add_calls.append(kwargs)
        viewer_arg.layers[f"{kwargs['prefix']}Cells"] = types.SimpleNamespace(
            name=f"{kwargs['prefix']}Cells"
        )

    monkeypatch.setattr(mod, "ensure_contact_analysis", fake_ensure)
    monkeypatch.setattr(mod, "read_position_contact_analysis", lambda _p: {"cells": [1]})
    monkeypatch.setattr(mod, "add_contact_analysis_layers", fake_add)

    viewer = _FakeViewer()
    widget = mod.ContactAnalysisWidget(viewer, standalone=True)
    _discover(widget, tmp_path)

    item = widget._discovery_list.item(0)
    widget._on_job_activated(item)

    assert captured["output_path"] == tmp_path / "posB" / "contact_analysis.h5"
    assert captured["overwrite"] is False
    assert (tmp_path / "posB" / "contact_analysis.h5").exists()
    assert len(add_calls) == 1
    # Badge flips to built.
    assert widget._discovery_list.item(0).text().endswith("[built]")

    widget.deleteLater()
    app.processEvents()


def test_recompute_selected_forces_rebuild(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    _flat_pos(tmp_path, "posA", nucleus=True, h5=True)  # .h5 already present

    captured = {}

    def fake_ensure(*, overwrite=False, output_path, **kwargs):
        captured["overwrite"] = overwrite
        return output_path, True

    monkeypatch.setattr(mod, "ensure_contact_analysis", fake_ensure)
    monkeypatch.setattr(mod, "read_position_contact_analysis", lambda _p: {"cells": [1]})
    monkeypatch.setattr(mod, "add_contact_analysis_layers", lambda *a, **k: None)

    viewer = _FakeViewer()
    widget = mod.ContactAnalysisWidget(viewer, standalone=True)
    _discover(widget, tmp_path)
    widget._discovery_list.setCurrentRow(0)  # selects -> set_context
    widget.recompute_btn.click()

    assert captured["overwrite"] is True

    widget.deleteLater()
    app.processEvents()


def test_process_all_builds_every_position_and_refreshes_badges(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    _flat_pos(tmp_path, "posA", nucleus=True)
    _flat_pos(tmp_path, "posB", nucleus=False)

    widget = mod.ContactAnalysisWidget(standalone=True)
    _discover(widget, tmp_path)
    assert all(
        widget._discovery_list.item(i).text().endswith("[missing]") for i in range(2)
    )

    widget.run_batch_btn.click()

    # Real backend ran on the synthetic stacks: both .h5 written, badges flipped.
    assert (tmp_path / "posA" / "contact_analysis.h5").exists()
    assert (tmp_path / "posB" / "contact_analysis.h5").exists()
    assert all(
        widget._discovery_list.item(i).text().endswith("[built]") for i in range(2)
    )
    assert "built 2" in widget.batch_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_state_round_trips(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.ContactAnalysisWidget()

    widget.set_state(
        {
            "color_cells_by_label": True,
            "color_edges_by_id": True,
            "color_edges_by_label": False,
            "hide_border_edges": True,
        }
    )
    state = widget.get_state()
    assert state == {
        "color_cells_by_label": True,
        "color_edges_by_id": True,
        "color_edges_by_label": False,
        "hide_border_edges": True,
    }

    widget.deleteLater()
    app.processEvents()
