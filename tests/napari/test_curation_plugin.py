"""The napari curation tool widget — Qt glue over the CurationController."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from itasc.napari.contact_analysis.plugins import (
    AnalysisContext,
    available_analysis_plugins,
)
from itasc.napari.contact_analysis.plugins.curation import CurationWidget


def _app():
    return QApplication.instance() or QApplication([])


class _FakeDims:
    def __init__(self, frame):
        self.current_step = (frame, 0, 0)


class _FakeViewer:
    def __init__(self, frame=0):
        self.dims = _FakeDims(frame)


def _record(tmp_path, pid="p1"):
    pdir = tmp_path / "study" / pid
    pdir.mkdir(parents=True, exist_ok=True)
    return {
        "id": pid,
        "experiment_id": "EXP1",
        "position_path": pdir,
        "cell_tracked_labels_path": None,
        "nucleus_tracked_labels_path": None,
        "contact_analysis_path": None,
    }


def test_curation_plugin_is_registered():
    assert CurationWidget in available_analysis_plugins()
    assert "curation" in {cls.plugin_id for cls in available_analysis_plugins()}


def test_single_position_enables_actions_only_with_reason(tmp_path, monkeypatch):
    app = _app()
    plugin = CurationWidget(viewer=_FakeViewer(frame=2))
    monkeypatch.setattr(plugin, "_update_display", lambda record: None)

    plugin.set_context(AnalysisContext(records=[_record(tmp_path)]))
    assert not plugin._exclude_frame_btn.isEnabled()
    assert not plugin._exclude_position_btn.isEnabled()
    plugin._reason_edit.setText("out of focus")
    assert plugin._exclude_frame_btn.isEnabled()
    assert plugin._exclude_position_btn.isEnabled()

    plugin.deleteLater()
    app.processEvents()


def test_zero_or_many_positions_disables_actions(tmp_path, monkeypatch):
    app = _app()
    plugin = CurationWidget(viewer=_FakeViewer())
    monkeypatch.setattr(plugin, "_update_display", lambda record: None)
    plugin._reason_edit.setText("x")

    plugin.set_context(AnalysisContext(records=[]))
    assert not plugin._exclude_frame_btn.isEnabled()

    plugin.set_context(AnalysisContext(records=[_record(tmp_path, "p1"),
                                               _record(tmp_path, "p2")]))
    assert not plugin._exclude_frame_btn.isEnabled()

    plugin.deleteLater()
    app.processEvents()


def test_exclude_frame_calls_controller_with_current_frame(tmp_path, monkeypatch):
    app = _app()
    plugin = CurationWidget(viewer=_FakeViewer(frame=5))
    monkeypatch.setattr(plugin, "_update_display", lambda record: None)
    plugin.set_context(AnalysisContext(records=[_record(tmp_path)]))
    plugin._reason_edit.setText("blurry")

    calls = []
    monkeypatch.setattr(plugin._controller, "exclude_frame",
                        lambda **kw: calls.append(kw))
    plugin._exclude_frame_btn.click()

    assert calls == [{"experiment_id": "EXP1", "position_id": "p1",
                      "frame": 5, "reason": "blurry"}]

    plugin.deleteLater()
    app.processEvents()


def test_exclude_position_calls_controller(tmp_path, monkeypatch):
    app = _app()
    plugin = CurationWidget(viewer=_FakeViewer())
    monkeypatch.setattr(plugin, "_update_display", lambda record: None)
    plugin.set_context(AnalysisContext(records=[_record(tmp_path)]))
    plugin._reason_edit.setText("all bad")

    calls = []
    monkeypatch.setattr(plugin._controller, "exclude_position",
                        lambda **kw: calls.append(kw))
    plugin._exclude_position_btn.click()

    assert calls == [{"experiment_id": "EXP1", "position_id": "p1",
                      "reason": "all bad"}]

    plugin.deleteLater()
    app.processEvents()


def test_exclude_frame_writes_through_to_csv(tmp_path, monkeypatch):
    """End-to-end through the real controller: the action lands in the CSV."""
    app = _app()
    plugin = CurationWidget(viewer=_FakeViewer(frame=7))
    monkeypatch.setattr(plugin, "_update_display", lambda record: None)
    plugin.set_context(AnalysisContext(records=[_record(tmp_path)]))
    plugin._reason_edit.setText("blurry")
    plugin._exclude_frame_btn.click()

    from itasc.contact_analysis.curation import read_curation
    # Single position at <tmp>/study/p1 -> catalogue_root is <tmp>/study.
    csv_path = tmp_path / "study" / "curation.csv"
    back = read_curation(csv_path)
    assert back is not None and len(back) == 1
    assert int(back.iloc[0]["frame"]) == 7

    plugin.deleteLater()
    app.processEvents()
