from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from cellflow_utils.contact_analysis.nls_classification import NLSClassificationSummary


class _FakeWorker:
    def quit(self) -> None:
        pass


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow_utils" / "napari"
    napari_pkg = types.ModuleType("cellflow_utils.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow_utils.napari", napari_pkg)
    sys.modules.pop("cellflow_utils.napari.nls_classification_widget", None)
    return importlib.import_module("cellflow_utils.napari.nls_classification_widget")


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


def test_nls_classification_widget_button_runs_classifier_with_picked_paths(monkeypatch, tmp_path):
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
    h5_path = tmp_path / "contact_analysis.h5"
    nls_path = tmp_path / "NLS_zavg.tif"
    labels_path = tmp_path / "tracked_labels.tif"
    h5_path.touch()
    nls_path.touch()
    labels_path.touch()

    widget = mod.NLSClassificationWidget()
    widget.h5_edit.setText(str(h5_path))
    widget.nls_edit.setText(str(nls_path))
    widget.labels_edit.setText(str(labels_path))
    widget.classify_btn.click()

    assert calls == [(h5_path, nls_path, labels_path)]
    assert "classified 4 tracks" in widget.status_lbl.text()
    assert "high=2" in widget.status_lbl.text()
    assert widget.classify_btn.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


def test_nls_classification_widget_disables_button_until_required_files_exist(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.NLSClassificationWidget()

    assert widget.classify_btn.isEnabled() is False
    assert "NLS image" in widget.status_lbl.text()
    assert "contact analysis" in widget.status_lbl.text()
    assert "nucleus labels" in widget.status_lbl.text()

    # A path that points at a non-existent file keeps the button disabled.
    widget.h5_edit.setText(str(tmp_path / "missing.h5"))
    assert widget.classify_btn.isEnabled() is False

    widget.deleteLater()
    app.processEvents()
