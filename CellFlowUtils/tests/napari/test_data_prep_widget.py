"""Tests for the raw data preparation napari widget."""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QComboBox, QLabel, QLineEdit, QWidget


class _FakeSignal:
    def connect(self, _callback) -> None:
        pass


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = {}


class _FakeMainWidget:
    def __init__(self, root_dir: Path) -> None:
        self.path_label = QLabel(str(root_dir))
        self.px_edit = QLineEdit()
        self.dt_edit = QLineEdit()
        self.pos_spin = SimpleNamespace(value=lambda: 0)
        self.refresh_requested = _FakeSignal()


class _FakeWorker:
    aborted = _FakeSignal()


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow_utils" / "napari"
    napari_pkg = types.ModuleType("cellflow_utils.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow_utils.napari", napari_pkg)
    sys.modules.pop("cellflow_utils.napari.data_prep_widget", None)
    return importlib.import_module("cellflow_utils.napari.data_prep_widget")


def _make_widget(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.DataPrepWidget(_FakeViewer(), _FakeMainWidget(tmp_path))
    return app, mod, widget


def _load_standalone_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow_utils" / "napari"
    napari_pkg = types.ModuleType("cellflow_utils.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow_utils.napari", napari_pkg)

    class _StubHpcCellposeWidget(QWidget):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.refreshed_pos_dir = None

        def refresh(self, pos_dir):
            self.refreshed_pos_dir = pos_dir

    hpc_module = types.ModuleType("cellflow_utils.napari.hpc_cellpose_widget")
    hpc_module.HpcCellposeWidget = _StubHpcCellposeWidget
    monkeypatch.setitem(sys.modules, "cellflow_utils.napari.hpc_cellpose_widget", hpc_module)
    sys.modules.pop("cellflow_utils.napari.data_prep_standalone_widget", None)
    return importlib.import_module("cellflow_utils.napari.data_prep_standalone_widget")


def _combo_items(combo: QComboBox) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


def _sync_thread_worker(connect=None):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if inspect.isgenerator(result):
                for yielded in result:
                    if connect and "yielded" in connect:
                        connect["yielded"](yielded)
            elif connect and "returned" in connect:
                connect["returned"](result)
            if connect and "finished" in connect:
                connect["finished"]()
            return _FakeWorker()
        return wrapper
    return decorator


def test_get_set_state_round_trips_frame_range(monkeypatch, tmp_path):
    app, _mod, widget = _make_widget(monkeypatch, tmp_path)

    state = {
        "ndtiff_path": "/data/acq",
        "positions": "0,2",
        "xy_downsample": 4,
        "z_downsample": 2,
        "frame_start": 2,
        "frame_end": 8,
        "nucleus_channel": 1,
        "cell_channel": 2,
        "nls_channel": 3,
        "overwrite": True,
    }

    widget.set_state(state)

    assert widget.get_state() == state

    widget.deleteLater()
    app.processEvents()


def test_run_passes_frame_range_to_dataset_config(monkeypatch, tmp_path):
    app, mod, widget = _make_widget(monkeypatch, tmp_path)
    captured = []

    def fake_run(config, pos, *, overwrite):
        captured.append((config, pos, overwrite))
        yield (1, 1, "done")

    monkeypatch.setattr(mod, "thread_worker", _sync_thread_worker)
    monkeypatch.setattr(mod, "run_prep", fake_run)

    widget.ndtiff_edit.setText("/data/acq")
    widget.pos_edit.setText("0,2")
    widget.ds_spin.setValue(3)
    widget.z_ds_spin.setValue(2)
    widget.frame_start_spin.setValue(4)
    widget.frame_end_spin.setValue(9)
    widget._set_channel_options("nucleus", ["CSUTRANS", "CSU405", "CSU488", "CSU561"], selected=1)
    widget._set_channel_options("cell", ["CSUTRANS", "CSU405", "CSU488", "CSU561"], selected=2)
    widget._set_channel_options("nls", ["CSUTRANS", "CSU405", "CSU488", "CSU561"], selected=3)
    widget.overwrite_check.setChecked(True)

    widget._on_run()

    assert [(pos, overwrite) for _config, pos, overwrite in captured] == [(0, True), (2, True)]
    assert [config.xy_downsample for config, _pos, _overwrite in captured] == [3, 3]
    assert [config.z_downsample for config, _pos, _overwrite in captured] == [2, 2]
    assert [config.nucleus_channel for config, _pos, _overwrite in captured] == [1, 1]
    assert [config.cell_channel for config, _pos, _overwrite in captured] == [2, 2]
    assert [config.nls_channel for config, _pos, _overwrite in captured] == [3, 3]
    assert [config.frame_start for config, _pos, _overwrite in captured] == [4, 4]
    assert [config.frame_end for config, _pos, _overwrite in captured] == [9, 9]

    widget.deleteLater()
    app.processEvents()


def test_run_in_terminal_includes_frame_range(monkeypatch, tmp_path):
    app, mod, widget = _make_widget(monkeypatch, tmp_path)
    launched = []
    monkeypatch.setattr(mod, "launch_in_terminal", launched.append)

    widget.ndtiff_edit.setText("/data/acq")
    widget.pos_edit.setText("1")
    widget.ds_spin.setValue(2)
    widget.z_ds_spin.setValue(3)
    widget.frame_start_spin.setValue(5)
    widget.frame_end_spin.setValue(12)
    widget._set_channel_options("nucleus", ["CSUTRANS", "CSU405", "CSU488", "CSU561"], selected=1)
    widget._set_channel_options("cell", ["CSUTRANS", "CSU405", "CSU488", "CSU561"], selected=2)
    widget._set_channel_options("nls", ["CSUTRANS", "CSU405", "CSU488", "CSU561"], selected=3)

    widget._on_run_in_terminal()

    assert len(launched) == 1
    assert "z_downsample=3" in launched[0]
    assert "nucleus_channel=1" in launched[0]
    assert "cell_channel=2" in launched[0]
    assert "nls_channel=3" in launched[0]
    assert "frame_start=5" in launched[0]
    assert "frame_end=12" in launched[0]

    widget.deleteLater()
    app.processEvents()


def test_standalone_data_prep_embeds_hpc_cellpose_section(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_standalone_module(monkeypatch)
    widget = mod.DataPrepStandaloneWidget(_FakeViewer())

    assert widget.hpc_cellpose_section.title == "HPC Cellpose"
    assert widget.hpc_cellpose_section.is_expanded is False

    widget.deleteLater()
    app.processEvents()


def test_standalone_data_prep_refreshes_hpc_cellpose(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_standalone_module(monkeypatch)
    widget = mod.DataPrepStandaloneWidget(_FakeViewer())
    widget.path_label.setText(str(tmp_path))
    widget.pos_spin.setValue(2)

    widget._refresh()

    assert widget.hpc_cellpose_widget.refreshed_pos_dir == tmp_path / "pos02"

    widget.deleteLater()
    app.processEvents()



def test_metadata_populates_channel_dropdowns(monkeypatch, tmp_path):
    app, _mod, widget = _make_widget(monkeypatch, tmp_path)

    widget._on_metadata_returned({
        "positions": [0, 1],
        "pixel_size_um": 0.25,
        "time_interval_s": 120.0,
        "channel_names": ["CSUTRANS", "CSU405", "CSU488", "CSU561"],
    })

    for combo_name in ("nucleus_channel_combo", "cell_channel_combo", "nls_channel_combo"):
        combo = getattr(widget, combo_name)
        assert isinstance(combo, QComboBox)
        assert _combo_items(combo) == [
            "0: CSUTRANS",
            "1: CSU405",
            "2: CSU488",
            "3: CSU561",
        ]

    widget.nucleus_channel_combo.setCurrentIndex(1)
    widget.cell_channel_combo.setCurrentIndex(2)
    widget.nls_channel_combo.setCurrentIndex(3)
    assert widget.get_state()["nucleus_channel"] == 1
    assert widget.get_state()["cell_channel"] == 2
    assert widget.get_state()["nls_channel"] == 3

    widget.deleteLater()
    app.processEvents()
