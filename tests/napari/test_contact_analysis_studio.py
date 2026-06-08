from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QLabel

from cellflow.napari import contact_analysis_studio as mod
from cellflow.napari.meta_plugins import (
    MetaAnalysisPlugin,
    MetaContext,
    available_meta_plugins,
)
from cellflow.napari.meta_plugins.catalog_summary import CatalogSummaryPlugin


def _app():
    return QApplication.instance() or QApplication([])


def _make_ready_position(root: Path, condition: str, experiment: str, position: str) -> None:
    pos = root / condition / experiment / position
    (pos / "4_contact_analysis").mkdir(parents=True)
    (pos / "2_nucleus").mkdir()
    (pos / "3_cell").mkdir()
    (pos / "4_contact_analysis" / "contact_analysis.h5").touch()
    (pos / "2_nucleus" / "tracked_labels.tif").touch()
    (pos / "3_cell" / "tracked_labels.tif").touch()


# --------------------------------------------------------------------- registry


def test_subclassing_registers_plugin():
    classes = available_meta_plugins()
    assert CatalogSummaryPlugin in classes
    ids = {cls.plugin_id for cls in classes}
    assert "catalog_summary" in ids


def test_single_contact_analysis_entry_no_meta_or_plugins_in_manifest():
    # The two standalone tools are merged into one Contact Analysis entry; plugins
    # and the old Meta Analysis entry must not appear as top-level dock widgets.
    import cellflow
    from npe2 import PluginManifest

    manifest = PluginManifest.from_file(Path(cellflow.__file__).parent / "napari.yaml")
    commands = {cmd.id for cmd in manifest.contributions.commands}
    assert "cellflow.contact_analysis_widget" in commands  # the merged tool
    assert "cellflow.meta_analysis_widget" not in commands  # merged away
    assert not any(cmd.startswith("cellflow.meta_plugin") for cmd in commands)


# ---------------------------------------------------------------- plugin hosting


def test_widget_mounts_registered_plugin():
    app = _app()
    widget = mod.ContactAnalysisStudioWidget()
    assert isinstance(widget._active_plugin, CatalogSummaryPlugin)
    widget.deleteLater()
    app.processEvents()


def test_selection_scope_forwarded_to_plugin(monkeypatch):
    app = _app()

    received: list[list[dict]] = []

    class _RecordingPlugin(MetaAnalysisPlugin):
        plugin_id = "recording_test"
        display_name = "Recording"

        def set_context(self, ctx: MetaContext) -> None:
            received.append(list(ctx.records))

    monkeypatch.setattr(mod, "available_meta_plugins", lambda: [_RecordingPlugin])

    widget = mod.ContactAnalysisStudioWidget()
    widget._records = [
        {"condition": "ctrl", "date": "d1", "id": "p1", "contact_analysis_path": Path("/a.h5"), "contact_analysis_status": "ready"},
        {"condition": "drug", "date": "d1", "id": "p2", "contact_analysis_path": Path("/b.h5"), "contact_analysis_status": "incomplete"},
    ]
    widget._refresh_table()  # empty selection -> whole catalog in scope
    assert received[-1] and len(received[-1]) == 2

    widget._table.selectRow(0)
    app.processEvents()
    assert len(received[-1]) == 1
    assert received[-1][0]["id"] == "p1"

    widget.deleteLater()
    app.processEvents()


def test_no_plugins_shows_placeholder(monkeypatch):
    app = _app()
    monkeypatch.setattr(mod, "available_meta_plugins", lambda: [])
    widget = mod.ContactAnalysisStudioWidget()
    assert isinstance(widget._active_plugin, QLabel)
    assert "No meta-analysis plugins" in widget._active_plugin.text()
    widget.deleteLater()
    app.processEvents()


# ---------------------------------------------------------------- contact view


def test_single_selection_drives_contact_view(monkeypatch):
    app = _app()
    monkeypatch.setattr(mod, "available_meta_plugins", lambda: [])

    widget = mod.ContactAnalysisStudioWidget()

    calls: list[dict] = []
    monkeypatch.setattr(
        widget._contact_widget, "set_context", lambda **kw: calls.append(kw)
    )

    widget._records = [
        {
            "condition": "ctrl", "date": "d1", "id": "p1",
            "contact_analysis_path": Path("/study/ctrl/d1/p1/4_contact_analysis/contact_analysis.h5"),
            "cell_tracked_labels_path": Path("/study/ctrl/d1/p1/3_cell/tracked_labels.tif"),
            "nucleus_tracked_labels_path": Path("/study/ctrl/d1/p1/2_nucleus/tracked_labels.tif"),
            "position_path": Path("/study/ctrl/d1/p1"),
            "contact_analysis_status": "ready",
        },
        {
            "condition": "drug", "date": "d1", "id": "p2",
            "contact_analysis_path": Path("/b.h5"),
            "contact_analysis_status": "incomplete",
        },
    ]
    widget._refresh_table()

    # One row selected -> contact view targets that position with its paths.
    widget._table.selectRow(0)
    app.processEvents()
    assert calls[-1]["out_path"] == widget._records[0]["contact_analysis_path"]
    assert calls[-1]["cell_labels"] == widget._records[0]["cell_tracked_labels_path"]

    # Multi-selection is ambiguous for a single view -> cleared.
    widget._table.selectAll()
    app.processEvents()
    assert calls[-1]["cell_labels"] is None and calls[-1]["out_path"] is None

    widget.deleteLater()
    app.processEvents()


def test_factory_resolves_to_studio_in_full_install():
    app = _app()
    import cellflow.napari.contact_analysis_widget as caw

    widget = caw.make_contact_analysis_widget(napari_viewer=None)
    assert isinstance(widget, mod.ContactAnalysisStudioWidget)
    widget.deleteLater()
    app.processEvents()


# -------------------------------------------------------------- catalog actions


def test_discover_annotate_add_then_csv_roundtrip(tmp_path):
    app = _app()
    study = tmp_path / "study"
    _make_ready_position(study, "ctrl", "exp1", "pos01")
    _make_ready_position(study, "ctrl", "exp1", "pos02")

    widget = mod.ContactAnalysisStudioWidget()
    widget._root_edit.setText(str(study))
    widget._contact_name_edit.setText("4_contact_analysis/contact_analysis.h5")
    widget._cell_name_edit.setText("3_cell/tracked_labels.tif")
    widget._nucleus_name_edit.setText("2_nucleus/tracked_labels.tif")

    # Discover stages a collection but does not add it yet.
    widget._on_discover()
    assert len(widget._pending_entries) == 2
    assert widget._discovered_list.count() == 2
    assert widget._records == []

    # Annotate the collection and add it.
    widget._condition_edit.setText("ctrl")
    widget._date_edit.setText("2026-05-09")
    widget._notes_edit.setText("pilot")
    widget._on_add_to_catalogue()
    assert len(widget._records) == 2
    assert widget._table.rowCount() == 2
    assert widget._pending_entries == []
    assert all(r["condition"] == "ctrl" and r["notes"] == "pilot" for r in widget._records)
    assert all(r["cell_tracked_labels_path"] is not None for r in widget._records)

    # CSV round-trip preserves the label paths.
    csv_path = tmp_path / "catalog.csv"
    widget._save_csv_to(csv_path)
    widget._on_clear_catalog()
    assert widget._records == []
    widget._load_csv_from(csv_path)
    assert len(widget._records) == 2
    assert all(r["cell_tracked_labels_path"] is not None for r in widget._records)
    assert all(r["nucleus_tracked_labels_path"] is not None for r in widget._records)

    widget.deleteLater()
    app.processEvents()


def test_catalog_summary_plugin_reports_counts():
    app = _app()
    plugin = CatalogSummaryPlugin()
    plugin.set_context(
        MetaContext(
            records=[
                {"condition": "ctrl", "contact_analysis_status": "ready"},
                {"condition": "ctrl", "contact_analysis_status": "incomplete"},
                {"condition": "drug", "contact_analysis_status": "ready"},
            ]
        )
    )
    text = plugin._summary_lbl.text()
    assert "3 position(s)" in text
    assert "ready: 2" in text
    assert "incomplete: 1" in text
    assert "ctrl: 2" in text
    plugin.deleteLater()
    app.processEvents()
