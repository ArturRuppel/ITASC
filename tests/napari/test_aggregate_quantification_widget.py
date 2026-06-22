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
    sys.modules.pop("cellflow.napari.aggregate_quantification_widget", None)
    sys.modules.pop("cellflow.napari.nls_classification_widget", None)
    return importlib.import_module("cellflow.napari.aggregate_quantification_widget")


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
        out_path=pos_dir / "aggregate_quantification" / "contact_analysis.h5",
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
        out = pos_dir / "aggregate_quantification" / "contact_analysis.h5"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"h5")
    return pos_dir


def test_contact_analysis_widget_refresh_tracks_inputs_output_and_button_states(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.AggregateQuantificationWidget()

    pos_dir = _staged_pos(tmp_path, "pos00", cell=False, nucleus=True)
    _set_pos(widget, pos_dir)

    assert widget.cell_labels_path == pos_dir / "3_cell" / "tracked_labels.tif"
    assert widget.nucleus_labels_path == pos_dir / "2_nucleus" / "tracked_labels.tif"
    assert widget.contact_analysis_out_path == pos_dir / "aggregate_quantification" / "contact_analysis.h5"
    assert hasattr(widget, "_files_widget")
    # Cell labels not on disk yet -> Visualize disabled.
    assert widget.visualize_btn.isEnabled() is False

    (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    _set_pos(widget, pos_dir)

    assert widget.visualize_btn.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_visualize_enabled_with_only_cell_labels(monkeypatch, tmp_path):
    """Nucleus is optional: cell labels alone should enable Visualize.

    The orchestrator always wires a nucleus path, but it may point at a file
    that does not exist yet. A missing nucleus file must not gate Visualize.
    """
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.AggregateQuantificationWidget()

    # cell present, nucleus file absent (path is still wired by _set_pos).
    pos_dir = _staged_pos(tmp_path, "pos00", cell=True, nucleus=False)
    _set_pos(widget, pos_dir)

    assert widget.nucleus_labels_path == pos_dir / "2_nucleus" / "tracked_labels.tif"
    assert not widget.nucleus_labels_path.exists()
    assert widget._effective_nucleus_path() is None
    assert widget.visualize_btn.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_does_not_embed_personal_nls_classification(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.AggregateQuantificationWidget(viewer)

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
    widget = mod.AggregateQuantificationWidget(viewer)
    _set_pos(widget, pos_dir)
    widget._on_visualize(overwrite=False)
    app.processEvents()  # overlay add is deferred to the next event-loop tick

    # Built (missing -> compute) then showed.
    assert progress_events == [(2, 5, "Indexing records")]
    assert captured["output_path"] == pos_dir / "aggregate_quantification" / "contact_analysis.h5"
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
    widget = mod.AggregateQuantificationWidget(viewer)
    _set_pos(widget, pos_dir)
    widget.visualize_btn.click()
    app.processEvents()  # overlay add is deferred to the next event-loop tick

    # Fast path: no rebuild, straight to show.
    assert len(add_calls) == 1

    widget.deleteLater()
    app.processEvents()


def test_revisualize_same_position_skips_layer_rebuild(monkeypatch, tmp_path):
    """Re-Visualize of the same position + options must not remove/re-add layers.

    The remove/re-add churn both flickers and can leave phantom empty rows in
    napari's layer list, so an unchanged re-Show is a no-op; only a changed
    position or display option rebuilds.
    """
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = _staged_pos(tmp_path, "pos05b", cell=True, nucleus=True, h5=True)

    add_calls = []
    cleared = {"n": 0}

    def fake_add(viewer_arg, contact_analysis_arg, **kwargs):
        add_calls.append(kwargs)
        viewer_arg.layers[f"{kwargs['prefix']}Cells"] = types.SimpleNamespace(
            name=f"{kwargs['prefix']}Cells"
        )

    monkeypatch.setattr(mod, "ensure_contact_analysis", lambda **k: (k["output_path"], True))
    monkeypatch.setattr(mod, "read_position_contact_analysis", lambda _p: {"cells": [1]})
    monkeypatch.setattr(mod, "add_contact_analysis_layers", fake_add)

    viewer = _FakeViewer()
    widget = mod.AggregateQuantificationWidget(viewer)
    _set_pos(widget, pos_dir)

    widget.visualize_btn.click()
    app.processEvents()
    assert len(add_calls) == 1

    # Same position, same options, layers still present -> skipped entirely.
    widget.visualize_btn.click()
    app.processEvents()
    assert len(add_calls) == 1, "unchanged re-Visualize must not re-add layers"

    # Changing a display option breaks the fast path and rebuilds.
    widget.color_edges_by_id_cb.setChecked(True)
    widget.visualize_btn.click()
    app.processEvents()
    assert len(add_calls) == 2

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
    widget = mod.AggregateQuantificationWidget(viewer)
    _set_pos(widget, pos_dir)
    assert widget.recompute_btn.isEnabled() is True
    widget.recompute_btn.click()

    assert captured["overwrite"] is True

    widget.deleteLater()
    app.processEvents()


def test_recompute_rereads_h5_instead_of_serving_stale_cache(monkeypatch, tmp_path):
    """A same-path rebuild must invalidate the path-keyed .h5 cache.

    The cache is keyed by output path, but Recompute rewrites the .h5 in place, so
    without invalidation _show_from_disk would serve the stale cached read.
    """
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = _staged_pos(tmp_path, "pos07", cell=True, nucleus=True, h5=True)

    def fake_ensure(*, cell_labels_path, output_path, nucleus_labels_path=None,
                    overwrite=False, progress_cb=None, **kwargs):
        return output_path, True

    # Each read returns a distinct payload so a stale cache is observable.
    reads: dict[str, int] = {"n": 0}

    def fake_read(_p):
        reads["n"] += 1
        return {"cells": [reads["n"]]}

    shown: list = []
    monkeypatch.setattr(mod, "ensure_contact_analysis", fake_ensure)
    monkeypatch.setattr(mod, "read_position_contact_analysis", fake_read)
    monkeypatch.setattr(
        mod, "add_contact_analysis_layers", lambda _v, ca, **k: shown.append(ca)
    )

    viewer = _FakeViewer()
    widget = mod.AggregateQuantificationWidget(viewer)
    _set_pos(widget, pos_dir)

    # First Visualize: existing .h5 is read once and shown. The overlay add is
    # deferred to the next event-loop tick, so flush before asserting it ran.
    widget.visualize_btn.click()
    app.processEvents()
    assert reads["n"] == 1
    assert shown[-1] == {"cells": [1]}

    # Recompute rewrites the same path → the cache must be dropped and re-read,
    # so the freshly built analysis is shown rather than the stale {"cells": [1]}.
    widget.recompute_btn.click()
    app.processEvents()
    assert reads["n"] == 2
    assert shown[-1] == {"cells": [2]}

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_shows_and_clears_contact_analysis_layers(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.AggregateQuantificationWidget(viewer)

    pos_dir = _staged_pos(tmp_path, "pos08", cell=True, nucleus=True, h5=True)
    contact_analysis_path = pos_dir / "aggregate_quantification" / "contact_analysis.h5"
    _set_pos(widget, pos_dir)

    assert widget.visualize_btn.isEnabled() is True

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
    app.processEvents()  # overlay add is deferred to the next event-loop tick

    assert read_calls == [contact_analysis_path]
    assert add_calls == [
        (
            viewer,
            contact_analysis,
            {
                "prefix": "[Contact Analysis] ",
                "color_edges_by_id": False,
                "color_edges_by_label": False,
                "hide_border_edges": False,
            },
        )
    ]
    assert "[Contact Analysis] Cells" in viewer.layers
    assert "Background" in viewer.layers

    # Visualize clears stale contact-analysis layers before re-adding; exercise
    # that internal clear directly (the standalone "Clear Layers" button is gone).
    widget._clear_contact_analysis_layers(set_status=True)

    assert "[Contact Analysis] Cells" not in viewer.layers
    assert "[Contact Analysis] Edges" not in viewer.layers
    assert "Background" in viewer.layers

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_forwards_visualizer_options(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.AggregateQuantificationWidget(viewer)

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

    widget.color_edges_by_id_cb.setChecked(True)
    widget.color_edges_by_label_cb.setChecked(True)
    widget.hide_border_edges_cb.setChecked(True)
    widget.visualize_btn.click()
    app.processEvents()  # overlay add is deferred to the next event-loop tick

    assert add_calls == [
        (
            viewer,
            contact_analysis,
            {
                "prefix": "[Contact Analysis] ",
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
    widget = mod.AggregateQuantificationWidget(viewer)

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
    app.processEvents()  # overlay add is deferred to the next event-loop tick
    # Changing a checkbox must NOT trigger a reload on its own
    widget.color_edges_by_id_cb.setChecked(True)
    assert len(add_calls) == 1, "checkbox change must not auto-reload"
    assert add_calls[0]["color_edges_by_id"] is False

    # Clicking Visualize again picks up the updated checkbox state
    widget.visualize_btn.click()
    app.processEvents()  # overlay add is deferred to the next event-loop tick
    assert len(add_calls) == 2
    assert add_calls[1]["color_edges_by_id"] is True

    widget.deleteLater()
    app.processEvents()


def test_contact_analysis_widget_show_uses_real_reader_and_visualizer(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    viewer = _FakeViewer()
    widget = mod.AggregateQuantificationWidget(viewer)

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
    app.processEvents()  # overlay add is deferred to the next event-loop tick

    assert "[Contact Analysis] Cell labels" in viewer.layers
    assert "[Contact Analysis] Nucleus labels" in viewer.layers
    assert "[Contact Analysis] Edges" in viewer.layers
    # Single-frame data can have no T1 transitions, so the (globally empty) T1
    # edges layer is skipped rather than added as a blank "ghost" layer.
    assert "[Contact Analysis] T1 edges" not in viewer.layers
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
    widget = mod.AggregateQuantificationWidget(standalone=True)

    assert widget._discovery_container.isVisibleTo(widget) is True
    assert widget._pipeline_files_section.isVisibleTo(widget) is False
    # The old single-file pickers are gone.
    assert not hasattr(widget, "_pickers_container")

    widget.deleteLater()
    app.processEvents()


def test_discovery_populates_list_with_status(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.AggregateQuantificationWidget(standalone=True)

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
    widget = mod.AggregateQuantificationWidget(viewer, standalone=True)
    _discover(widget, tmp_path)

    item = widget._discovery_list.item(0)
    widget._on_job_activated(item)
    app.processEvents()  # overlay add is deferred to the next event-loop tick

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
    widget = mod.AggregateQuantificationWidget(viewer, standalone=True)
    _discover(widget, tmp_path)
    widget._discovery_list.setCurrentRow(0)  # selects -> set_context
    widget._on_visualize(overwrite=True)  # was the Recompute button

    assert captured["overwrite"] is True

    widget.deleteLater()
    app.processEvents()


def test_process_all_builds_every_position_and_refreshes_badges(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    _flat_pos(tmp_path, "posA", nucleus=True)
    _flat_pos(tmp_path, "posB", nucleus=False)

    widget = mod.AggregateQuantificationWidget(standalone=True)
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
    widget = mod.AggregateQuantificationWidget()

    widget.set_state(
        {
            "color_edges_by_id": True,
            "color_edges_by_label": False,
            "hide_border_edges": True,
        }
    )
    state = widget.get_state()
    assert state == {
        "color_edges_by_id": True,
        "color_edges_by_label": False,
        "hide_border_edges": True,
    }

    widget.deleteLater()
    app.processEvents()
