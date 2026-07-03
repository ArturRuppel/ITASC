from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from cellflow.napari.contact_analysis.plugins import (
    AnalysisContext,
    available_analysis_plugins,
)
from cellflow.napari.contact_analysis.plugins.visualize_contacts import (
    VisualizeContactsPlugin,
)


def _app():
    return QApplication.instance() or QApplication([])


def test_visualize_contacts_is_registered():
    assert VisualizeContactsPlugin in available_analysis_plugins()
    ids = {cls.plugin_id for cls in available_analysis_plugins()}
    assert "visualize_contacts" in ids


def test_single_position_drives_the_embedded_view(monkeypatch):
    app = _app()
    plugin = VisualizeContactsPlugin()
    calls: list[dict] = []
    monkeypatch.setattr(plugin._view, "set_context", lambda **kw: calls.append(kw))

    record = {
        "id": "p1",
        "contact_analysis_path": Path("/study/p1/4_contact_analysis/contact_analysis.h5"),
        "cell_tracked_labels_path": Path("/study/p1/3_cell/tracked_labels.tif"),
        "nucleus_tracked_labels_path": Path("/study/p1/2_nucleus/tracked_labels.tif"),
        "position_path": Path("/study/p1"),
    }
    plugin.set_context(AnalysisContext(records=[record]))
    assert calls[-1]["out_path"] == record["contact_analysis_path"]
    assert calls[-1]["cell_labels"] == record["cell_tracked_labels_path"]
    assert calls[-1]["status_root"] == record["position_path"]

    plugin.deleteLater()
    app.processEvents()


def test_zero_or_many_positions_clears_the_view(monkeypatch):
    app = _app()
    plugin = VisualizeContactsPlugin()
    calls: list[dict] = []
    monkeypatch.setattr(plugin._view, "set_context", lambda **kw: calls.append(kw))

    # Several positions: ambiguous for a single viewer -> cleared.
    plugin.set_context(
        AnalysisContext(
            records=[
                {"id": "p1", "contact_analysis_path": Path("/a.h5")},
                {"id": "p2", "contact_analysis_path": Path("/b.h5")},
            ]
        )
    )
    assert calls[-1]["cell_labels"] is None and calls[-1]["out_path"] is None

    # No positions: also cleared.
    plugin.set_context(AnalysisContext(records=[]))
    assert calls[-1]["cell_labels"] is None and calls[-1]["out_path"] is None

    plugin.deleteLater()
    app.processEvents()
