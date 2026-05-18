from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from cellflow_personal.contact_analysis.nls_classification import NLSClassificationSummary


class _FakeWorker:
    def quit(self) -> None:
        pass


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow_personal" / "napari"
    napari_pkg = types.ModuleType("cellflow_personal.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow_personal.napari", napari_pkg)
    sys.modules.pop("cellflow_personal.napari.nls_classification_widget", None)
    return importlib.import_module("cellflow_personal.napari.nls_classification_widget")


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
                            next(result)
                        except StopIteration as exc:
                            return_value = exc.value
                            break
                    result = return_value
                if connect and "returned" in connect:
                    connect["returned"](result)
                return _FakeWorker()

            return wrapper

        return decorator

    return fake_thread_worker


def test_nls_classification_widget_button_runs_classifier_with_position_paths(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())
    calls = []

    def fake_patch(h5_path, nls_zavg_path, nucleus_labels_path):
        calls.append((h5_path, nls_zavg_path, nucleus_labels_path))
        return NLSClassificationSummary(
            h5_path=h5_path,
            threshold=42.0,
            track_count=4,
            high_track_count=2,
            low_track_count=2,
        )

    monkeypatch.setattr(mod, "patch_position_contact_analysis_nls_classes", fake_patch)
    pos_dir = tmp_path / "pos00"
    (pos_dir / "0_input").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "4_contact_analysis").mkdir()
    (pos_dir / "0_input" / "NLS_zavg.tif").touch()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "4_contact_analysis" / "contact_analysis.h5").touch()

    widget = mod.NLSClassificationWidget()
    widget.refresh(pos_dir)
    widget.classify_btn.click()

    assert calls == [
        (
            pos_dir / "4_contact_analysis" / "contact_analysis.h5",
            pos_dir / "0_input" / "NLS_zavg.tif",
            pos_dir / "2_nucleus" / "tracked_labels.tif",
        )
    ]
    assert "classified 4 tracks" in widget.status_lbl.text()
    assert "high=2" in widget.status_lbl.text()
    assert widget.classify_btn.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


def test_nls_classification_widget_disables_button_until_required_files_exist(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.NLSClassificationWidget()
    pos_dir = tmp_path / "pos01"
    (pos_dir / "0_input").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "4_contact_analysis").mkdir()

    widget.refresh(pos_dir)

    assert widget.classify_btn.isEnabled() is False
    assert "NLS image" in widget.status_lbl.text()
    assert "contact analysis" in widget.status_lbl.text()

    widget.deleteLater()
    app.processEvents()
