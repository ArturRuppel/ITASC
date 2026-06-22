from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QLabel

from cellflow.aggregate_quantification.quantifiers.contacts import ContactsQuantifier
from cellflow.napari import aggregate_quantification_studio as mod
from cellflow.napari.aggregate_quantification.plugins import (
    AnalysisPlugin,
    AnalysisContext,
    available_analysis_plugins,
)
from cellflow.napari.aggregate_quantification.plugins.catalog_summary import CatalogSummaryPlugin


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
    classes = available_analysis_plugins()
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
    assert "cellflow.aggregate_quantification_widget" in commands  # the merged tool
    assert "cellflow.meta_analysis_widget" not in commands  # merged away
    assert not any(cmd.startswith("cellflow.meta_plugin") for cmd in commands)


# ---------------------------------------------------------------- plugin hosting


def test_every_tool_is_its_own_collapsible_collapsed():
    app = _app()
    widget = mod.AggregateQuantificationStudioWidget()
    # Tools are the analysis plugins (building moved to the Build area), each its
    # own collapsible section, all collapsed (off) until expanded.
    assert "catalog_summary" in widget._plugin_sections
    assert not any(pid.startswith("build:") for pid in widget._plugin_sections)
    assert all(
        not plugin.section.is_expanded for plugin in widget._plugin_sections.values()
    )
    # The Build area carries a metric row per quantifier instead.
    assert "contacts" in widget._build_area._rows
    widget.deleteLater()
    app.processEvents()


def test_collapsible_header_toggles_a_plugin():
    app = _app()
    widget = mod.AggregateQuantificationStudioWidget()

    plugin = widget._plugin_sections["catalog_summary"]
    assert isinstance(plugin.body, CatalogSummaryPlugin)
    # The section's own collapse header is the on/off control (no checkbox).
    assert not plugin.section.is_expanded
    plugin.section.expand()
    assert plugin.section.is_expanded
    plugin.section.collapse()
    assert not plugin.section.is_expanded

    widget.deleteLater()
    app.processEvents()


def test_selection_scope_forwarded_to_plugins(monkeypatch):
    app = _app()

    received: list[list[dict]] = []

    class _RecordingPlugin(AnalysisPlugin):
        plugin_id = "recording_test"
        display_name = "Recording"

        def set_context(self, ctx: AnalysisContext) -> None:
            received.append(list(ctx.records))

    entry = mod.PluginEntry(
        plugin_id="recording_test",
        display_name="Recording",
        requires=(),
        factory=lambda viewer: _RecordingPlugin(viewer),
    )
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [entry])

    widget = mod.AggregateQuantificationStudioWidget()
    # Plugins are always present and fed the scope; no checkbox to mount first.
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


def test_plugin_header_greyed_when_inputs_missing(monkeypatch):
    app = _app()

    entry = mod.PluginEntry(
        plugin_id="needs_nucleus",
        display_name="Needs nucleus",
        requires=("nucleus_labels_path",),  # a PositionInputs field name
        factory=lambda viewer: QLabel("x"),
    )
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [entry])

    widget = mod.AggregateQuantificationStudioWidget()
    widget._records = [
        {"condition": "c", "date": "d", "id": "p1", "contact_analysis_path": Path("/a.h5"),
         "cell_tracked_labels_path": Path("/cells.tif"), "contact_analysis_status": "ready"},
    ]
    widget._refresh_table()
    toggle = widget._plugin_sections["needs_nucleus"].section._toggle
    assert toggle.isEnabled() is False

    widget._records[0]["nucleus_tracked_labels_path"] = Path("/nuc.tif")
    widget._refresh_table()
    assert toggle.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


def test_no_plugins_shows_placeholder(monkeypatch):
    app = _app()
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [])
    widget = mod.AggregateQuantificationStudioWidget()
    assert widget._plugin_sections == {}
    widget.deleteLater()
    app.processEvents()


# ----------------------------------------------- contact view (now a plugin)


def test_single_selection_drives_visualize_contacts_plugin(monkeypatch):
    app = _app()
    widget = mod.AggregateQuantificationStudioWidget()

    # The per-position visualizer is the Visualize Contacts plugin now; drive
    # its embedded view by intercepting the widget-level set_context.
    plugin = widget._plugin_sections["visualize_contacts"].body
    calls: list[dict] = []
    monkeypatch.setattr(plugin._view, "set_context", lambda **kw: calls.append(kw))

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
    import cellflow.napari.aggregate_quantification_widget as caw

    widget = caw.make_aggregate_quantification_widget(napari_viewer=None)
    assert isinstance(widget, mod.AggregateQuantificationStudioWidget)
    widget.deleteLater()
    app.processEvents()


# -------------------------------------------------------------- catalog actions


def test_discover_annotate_add_then_csv_roundtrip(tmp_path):
    app = _app()
    study = tmp_path / "study"
    _make_ready_position(study, "ctrl", "exp1", "pos01")
    _make_ready_position(study, "ctrl", "exp1", "pos02")

    widget = mod.AggregateQuantificationStudioWidget()
    widget._root_edit.setText(str(study))
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


def test_add_is_register_only_no_build(tmp_path, monkeypatch):
    """Add registers positions synchronously and never builds (that's a plugin)."""
    app = _app()
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [])

    study = tmp_path / "study"
    p1 = study / "pos01"
    p2 = study / "pos02"
    for p in (p1, p2):
        p.mkdir(parents=True)
        (p / "cell_labels.tif").touch()
    # pos01 already built (in the shared aggregate_quantification/ folder); pos02 missing.
    (p1 / "aggregate_quantification").mkdir()
    (p1 / "aggregate_quantification" / "contact_analysis.h5").touch()

    widget = mod.AggregateQuantificationStudioWidget()
    began: list = []
    monkeypatch.setattr(widget, "_begin_build", lambda *a, **k: began.append(a))

    widget._root_edit.setText(str(study))
    widget._cell_name_edit.setText("cell_labels.tif")
    widget._on_discover()
    assert len(widget._pending_entries) == 2

    widget._on_add_to_catalogue()
    # No build kicked off; both positions registered, statuses reflect reality.
    assert began == []
    assert widget._build_worker is None
    assert len(widget._records) == 2
    by_id = {r["id"]: r for r in widget._records}
    assert by_id["pos01"]["contact_analysis_status"] == "ready"
    assert by_id["pos02"]["contact_analysis_status"] == "incomplete"

    widget.deleteLater()
    app.processEvents()


def test_build_area_run_delegates_to_pipeline(tmp_path, monkeypatch):
    """The Build area Run forwards the checked quantifiers + in-scope records (and
    the shared build params) to the build loop. Which positions actually build —
    skipping those lacking inputs — is now ``pipeline.build_quantities``' job (see
    test_pipeline), not the studio's."""
    app = _app()
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [])

    widget = mod.AggregateQuantificationStudioWidget()
    quantifier = ContactsQuantifier()

    p1, p2 = tmp_path / "p1", tmp_path / "p2"
    for p in (p1, p2):
        p.mkdir()
    records = [
        {"id": "p1", "position_path": p1, "contact_analysis_path": p1 / "contact_analysis.h5",
         "cell_tracked_labels_path": p1 / "cells.tif"},
        {"id": "p2", "position_path": p2, "contact_analysis_path": p2 / "contact_analysis.h5",
         "cell_tracked_labels_path": p2 / "cells.tif"},
    ]

    captured: list = []
    monkeypatch.setattr(widget, "_begin_build", lambda *args: captured.append(args))

    widget._run_quantity_builds([quantifier], records, overwrite=True)
    quants, recs, _params = captured[-1]
    assert [q.quantity_id for q in quants] == ["contacts"]
    assert [r["id"] for r in recs] == ["p1", "p2"]

    # Empty scope or no metric checked is a no-op (no worker queued).
    captured.clear()
    widget._run_quantity_builds([quantifier], [], overwrite=True)
    widget._run_quantity_builds([], records, overwrite=True)
    assert captured == []

    widget.deleteLater()
    app.processEvents()


def test_build_status_goes_to_compute_section_not_catalogue(monkeypatch):
    # Bug 13: computing/status text belongs in the Compute section, beside the
    # build controls — not in the catalogue status line.
    app = _app()
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [])
    widget = mod.AggregateQuantificationStudioWidget()
    widget._catalog_status_lbl.setText("catalogue idle")

    # "Nothing to build" is a build outcome -> Compute status, catalogue untouched.
    widget._run_quantity_builds([ContactsQuantifier()], [], overwrite=False)
    assert "Nothing to build" in widget._build_status_lbl.text()
    assert widget._catalog_status_lbl.text() == "catalogue idle"

    # Progress + completion also land on the Compute status.
    widget._on_build_progress(1, 2, "p1")
    assert "Computing" in widget._build_status_lbl.text()
    widget._on_build_done([])
    assert "Built" in widget._build_status_lbl.text()
    assert widget._catalog_status_lbl.text() == "catalogue idle"

    widget.deleteLater()
    app.processEvents()


def test_section_defaults_and_compute_rename(monkeypatch):
    # Bugs 16/18/20: Parameters expanded; Build renamed to Compute; Tools /
    # Compute collapsed by default. The in-napari Plots section is removed
    # (visualization is Iris-only).
    from cellflow.napari.widgets import CollapsibleSection

    app = _app()
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [])
    widget = mod.AggregateQuantificationStudioWidget()
    by_title = {
        s.title: s for s in widget.findChildren(CollapsibleSection)
    }
    assert "Build" not in by_title and "Compute" in by_title
    assert "Plots" not in by_title
    assert by_title["Parameters"].is_expanded is True
    assert by_title["Tools"].is_expanded is False
    assert by_title["Compute"].is_expanded is False
    widget.deleteLater()
    app.processEvents()


def test_collapsing_everything_shrinks_the_studio_to_fit(monkeypatch):
    # Bug 25: once every collapsible is collapsed the studio's content must
    # shrink back to the stacked headers — not retain the tall expanded height.
    from qtpy.QtWidgets import QScrollArea

    from cellflow.napari.widgets import CollapsibleSection

    app = _app()
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [])
    widget = mod.AggregateQuantificationStudioWidget()
    widget.resize(360, 900)
    sections = widget.findChildren(CollapsibleSection)

    for section in sections:
        section.expand()
    app.processEvents()
    content = widget.findChild(QScrollArea).widget()
    expanded_min = content.minimumSizeHint().height()

    for section in sections:
        section.collapse()
    app.processEvents()
    app.processEvents()
    collapsed_min = content.minimumSizeHint().height()

    # Collapsing reclaims the height: the content is now a small fraction of its
    # expanded minimum, roughly the stacked section headers.
    assert collapsed_min < expanded_min / 3
    assert collapsed_min <= 40 * len(sections)
    widget.deleteLater()
    app.processEvents()


def test_catalog_summary_plugin_reports_counts():
    app = _app()
    plugin = CatalogSummaryPlugin()
    plugin.set_context(
        AnalysisContext(
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


# --------------------------------------------------------------- catalog editing


def test_save_csv_appends_csv_extension(tmp_path, monkeypatch):
    app = _app()
    widget = mod.AggregateQuantificationStudioWidget()
    widget._records = [
        {"condition": "ctrl", "date": "d1", "id": "p1",
         "contact_analysis_path": Path("/a.h5"), "contact_analysis_status": "ready"},
    ]

    # User picks a name without an extension; getSaveFileName returns it verbatim.
    target = tmp_path / "my_catalog"
    monkeypatch.setattr(
        mod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "")
    )
    widget._on_save_csv()

    assert (tmp_path / "my_catalog.csv").is_file()
    assert not (tmp_path / "my_catalog").exists()

    widget.deleteLater()
    app.processEvents()


def test_save_csv_keeps_existing_csv_extension(tmp_path, monkeypatch):
    app = _app()
    widget = mod.AggregateQuantificationStudioWidget()
    widget._records = [
        {"condition": "ctrl", "date": "d1", "id": "p1",
         "contact_analysis_path": Path("/a.h5"), "contact_analysis_status": "ready"},
    ]
    target = tmp_path / "cat.csv"
    monkeypatch.setattr(
        mod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "")
    )
    widget._on_save_csv()

    assert target.is_file()
    assert not (tmp_path / "cat.csv.csv").exists()

    widget.deleteLater()
    app.processEvents()


def test_remove_selected_drops_only_selected_rows():
    app = _app()
    widget = mod.AggregateQuantificationStudioWidget()
    widget._records = [
        {"condition": "ctrl", "date": "d1", "id": "p1",
         "contact_analysis_path": Path("/a.h5"), "contact_analysis_status": "ready"},
        {"condition": "drug", "date": "d1", "id": "p2",
         "contact_analysis_path": Path("/b.h5"), "contact_analysis_status": "ready"},
        {"condition": "drug", "date": "d2", "id": "p3",
         "contact_analysis_path": Path("/c.h5"), "contact_analysis_status": "ready"},
    ]
    widget._refresh_table()
    # Seed the analysis cache so we can confirm dropped rows are evicted.
    widget._analysis_cache[Path("/b.h5")] = object()

    widget._table.selectRow(1)
    app.processEvents()
    widget._on_remove_selected()

    assert [r["id"] for r in widget._records] == ["p1", "p3"]
    assert widget._table.rowCount() == 2
    assert Path("/b.h5") not in widget._analysis_cache

    widget.deleteLater()
    app.processEvents()


def test_remove_selected_without_selection_is_a_noop():
    app = _app()
    widget = mod.AggregateQuantificationStudioWidget()
    widget._records = [
        {"condition": "ctrl", "date": "d1", "id": "p1",
         "contact_analysis_path": Path("/a.h5"), "contact_analysis_status": "ready"},
    ]
    widget._refresh_table()
    widget._table.clearSelection()
    app.processEvents()
    widget._on_remove_selected()

    assert [r["id"] for r in widget._records] == ["p1"]

    widget.deleteLater()
    app.processEvents()
