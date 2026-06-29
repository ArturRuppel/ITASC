from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QLabel

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
    # The Run area carries a quantity checkbox per quantifier instead.
    assert "contacts" in widget._run_area._quantity_checks
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


def test_discover_levels_add_then_csv_roundtrip(tmp_path):
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
    # The three nesting levels (study → condition → experiment → position) are
    # seeded with the recognized identity axes, innermost = position_id.
    assert widget._level_names() == ["condition", "experiment_id", "position_id"]

    # A manual constant tag rides onto every added row alongside the folder levels.
    widget._add_manual_column("operator", "Ada")
    widget._on_add_to_catalogue()
    assert len(widget._records) == 2
    assert widget._pending_entries == []
    by_id = {r["id"]: r for r in widget._records}
    assert set(by_id) == {"pos01", "pos02"}
    assert all(r["condition"] == "ctrl" for r in widget._records)
    assert all(r["experiment_id"] == "exp1" for r in widget._records)
    assert all(r["columns"]["operator"] == "Ada" for r in widget._records)
    assert all(r["cell_tracked_labels_path"] is not None for r in widget._records)

    # CSV round-trip preserves the label paths and the free-form columns.
    csv_path = tmp_path / "catalog.csv"
    widget._save_csv_to(csv_path)
    assert "operator" in csv_path.read_text().splitlines()[0]
    widget._on_clear_catalog()
    assert widget._records == []
    widget._load_csv_from(csv_path)
    assert len(widget._records) == 2
    assert all(r["cell_tracked_labels_path"] is not None for r in widget._records)
    assert all(r["nucleus_tracked_labels_path"] is not None for r in widget._records)
    assert all(r["columns"]["operator"] == "Ada" for r in widget._records)
    assert all(r["experiment_id"] == "exp1" for r in widget._records)

    widget.deleteLater()
    app.processEvents()


def test_renaming_a_level_rederives_staged_columns(tmp_path):
    app = _app()
    study = tmp_path / "study"
    _make_ready_position(study, "ctrl", "exp1", "pos01")

    widget = mod.AggregateQuantificationStudioWidget()
    widget._root_edit.setText(str(study))
    widget._cell_name_edit.setText("3_cell/tracked_labels.tif")
    widget._on_discover()

    # Rename the outermost level before committing; the column follows live.
    widget._level_name_fields[0].setText("genotype")
    cols = widget._columns_for_entry(widget._pending_entries[0])
    assert cols["genotype"] == "ctrl"
    assert "condition" not in cols
    # position_id still resolves to the innermost folder name (kept unique).
    assert cols["position_id"] == "pos01"

    widget._on_add_to_catalogue()
    assert widget._records[0]["columns"]["genotype"] == "ctrl"

    widget.deleteLater()
    app.processEvents()


def test_discover_refuses_mixed_depth_positions(tmp_path):
    app = _app()
    study = tmp_path / "study"
    _make_ready_position(study, "ctrl", "exp1", "pos01")  # depth 3
    shallow = study / "loose_pos"  # depth 1
    (shallow / "3_cell").mkdir(parents=True)
    (shallow / "3_cell" / "tracked_labels.tif").touch()

    widget = mod.AggregateQuantificationStudioWidget()
    widget._root_edit.setText(str(study))
    widget._cell_name_edit.setText("3_cell/tracked_labels.tif")
    widget._on_discover()

    # Differing depths can't line up to named levels → nothing staged, a warning.
    assert widget._pending_entries == []
    assert widget._level_name_fields == []
    assert not widget._add_btn.isEnabled()
    assert "differing folder depths" in widget._discover_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_dynamic_discover_tooltip_names_filled_inputs():
    app = _app()
    widget = mod.AggregateQuantificationStudioWidget()

    widget._cell_name_edit.setText("")
    widget._nucleus_name_edit.setText("")
    assert "at least one input file" in widget._discover_btn.toolTip()

    widget._cell_name_edit.setText("cells.tif")
    assert "cells.tif" in widget._discover_btn.toolTip()
    widget._nucleus_name_edit.setText("nuc.tif")
    tip = widget._discover_btn.toolTip()
    assert "cells.tif" in tip and "nuc.tif" in tip

    widget.deleteLater()
    app.processEvents()


def test_group_separators_and_remove_with_separators(tmp_path):
    app = _app()
    widget = mod.AggregateQuantificationStudioWidget()
    # Two batches (distinct column captions) → a separator row precedes each.
    widget._records = [
        {"condition": "WT", "date": "d1", "id": "p1",
         "columns": {"condition": "WT", "position_id": "p1"},
         "contact_analysis_path": Path("/a.h5"), "contact_analysis_status": "ready"},
        {"condition": "WT", "date": "d1", "id": "p2",
         "columns": {"condition": "WT", "position_id": "p2"},
         "contact_analysis_path": Path("/b.h5"), "contact_analysis_status": "ready"},
        {"condition": "KO", "date": "d1", "id": "p3",
         "columns": {"condition": "KO", "position_id": "p3"},
         "contact_analysis_path": Path("/c.h5"), "contact_analysis_status": "ready"},
    ]
    widget._refresh_table()
    # 3 records + 2 group separators = 5 table rows.
    assert widget._table.rowCount() == 5
    assert widget._row_to_record == [None, 0, 1, None, 2]

    # Selecting the KO data row (table row 4) maps to record index 2, and removing
    # it drops exactly p3 despite the separator offset.
    widget._table.selectRow(4)
    app.processEvents()
    assert widget._selected_rows() == [2]
    widget._on_remove_selected()
    assert [r["id"] for r in widget._records] == ["p1", "p2"]

    widget.deleteLater()
    app.processEvents()


def test_add_is_register_only_no_build(tmp_path, monkeypatch):
    """Add registers positions synchronously and never builds (that's a plugin)."""
    # Add registers positions; building is the Run section's job.
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

    widget._root_edit.setText(str(study))
    widget._cell_name_edit.setText("cell_labels.tif")
    widget._on_discover()
    assert len(widget._pending_entries) == 2

    widget._on_add_to_catalogue()
    assert len(widget._records) == 2
    by_id = {r["id"]: r for r in widget._records}
    assert by_id["pos01"]["contact_analysis_status"] == "ready"
    assert by_id["pos02"]["contact_analysis_status"] == "incomplete"

    widget.deleteLater()
    app.processEvents()



def test_section_defaults_and_compute_rename(monkeypatch):
    # The piecemeal Build/Compute + Aggregate sections are replaced by one Run
    # section (expanded by default). Plots stay removed (Iris-only visualization).
    from cellflow.napari.widgets import CollapsibleSection

    app = _app()
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [])
    widget = mod.AggregateQuantificationStudioWidget()
    by_title = {s.title: s for s in widget.findChildren(CollapsibleSection)}
    assert "Run" in by_title
    assert "Build" not in by_title and "Compute" not in by_title
    assert "Aggregate" not in by_title
    assert "Plots" not in by_title
    assert by_title["Parameters"].is_expanded is True
    assert by_title["Tools"].is_expanded is False
    assert by_title["Run"].is_expanded is True
    widget.deleteLater()
    app.processEvents()


def test_run_section_authors_config_and_dispatches_run(tmp_path, monkeypatch):
    app = _app()
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [])

    pdir = tmp_path / "study" / "p1"
    pdir.mkdir(parents=True)
    widget = mod.AggregateQuantificationStudioWidget()
    widget._records = [
        {"id": "p1", "position_path": pdir, "condition": "ctrl",
         "date": "d", "notes": ""},
    ]
    widget._refresh_table()  # feeds the Run area the catalogue scope

    seen = {}

    def fake_author(out_dir, records, **kw):
        seen["out_dir"] = out_dir
        seen["kw"] = kw
        return out_dir / "config.toml"

    ran = {}
    monkeypatch.setattr(mod, "author_config", fake_author)
    monkeypatch.setattr(mod, "run", lambda p, progress_cb=None: ran.setdefault("p", p) or [])

    widget._run_area._on_run()
    assert seen["out_dir"] == pdir.parent  # catalogue root
    assert "quantities" in seen["kw"] and "params" in seen["kw"]

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
