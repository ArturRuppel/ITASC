from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QLabel

from cellflow.napari import contact_analysis_studio as mod
from cellflow.napari.contact_analysis.plugins import (
    AnalysisPlugin,
    AnalysisContext,
    available_analysis_plugins,
)
from cellflow.napari.contact_analysis.plugins.catalog_summary import CatalogSummaryPlugin


def _app():
    # A bare QApplication([]) aborts under pytest here; go through napari's own
    # bootstrap so the shared QApplication is created the way the app expects.
    from napari.qt import get_qapp

    return get_qapp()


def _entry(record: dict) -> dict:
    """A shared-panel entry (key/columns/payload) for a catalog record dict."""
    key = record.get("position_path") or record.get("id") or record.get("contact_analysis_path")
    return {
        "key": str(key),
        "columns": dict(record.get("columns") or {}),
        "payload": record,
    }


def _load(widget, records: list[dict]) -> None:
    """Seed the studio's ExperimentsPanel with committed rows from records."""
    widget._panel.set_records([_entry(r) for r in records])


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
    assert "cellflow.contact_analysis_widget" in commands  # the merged tool
    assert "cellflow.meta_analysis_widget" not in commands  # merged away
    assert not any(cmd.startswith("cellflow.meta_plugin") for cmd in commands)


# ---------------------------------------------------------------- plugin hosting


def test_every_tool_is_its_own_collapsible_collapsed():
    app = _app()
    widget = mod.ContactAnalysisStudioWidget()
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
    widget = mod.ContactAnalysisStudioWidget()

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

    widget = mod.ContactAnalysisStudioWidget()
    # Plugins are always present and fed the scope; no checkbox to mount first.
    _load(widget, [
        {"condition": "ctrl", "date": "d1", "id": "p1", "contact_analysis_path": Path("/a.h5"), "contact_analysis_status": "ready"},
        {"condition": "drug", "date": "d1", "id": "p2", "contact_analysis_path": Path("/b.h5"), "contact_analysis_status": "incomplete"},
    ])
    # Empty selection -> whole catalog in scope.
    assert received[-1] and len(received[-1]) == 2

    widget._panel.set_active("p1")
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

    widget = mod.ContactAnalysisStudioWidget()
    record = {
        "condition": "c", "date": "d", "id": "p1", "contact_analysis_path": Path("/a.h5"),
        "cell_tracked_labels_path": Path("/cells.tif"), "contact_analysis_status": "ready",
    }
    _load(widget, [record])
    toggle = widget._plugin_sections["needs_nucleus"].section._toggle
    assert toggle.isEnabled() is False

    with_nucleus = {**record, "nucleus_tracked_labels_path": Path("/nuc.tif")}
    _load(widget, [with_nucleus])
    assert toggle.isEnabled() is True

    widget.deleteLater()
    app.processEvents()


def test_no_plugins_shows_placeholder(monkeypatch):
    app = _app()
    monkeypatch.setattr(mod, "available_tool_plugins", lambda: [])
    widget = mod.ContactAnalysisStudioWidget()
    assert widget._plugin_sections == {}
    widget.deleteLater()
    app.processEvents()


# ----------------------------------------------- contact view (now a plugin)


def test_single_selection_drives_visualize_contacts_plugin(monkeypatch):
    app = _app()
    widget = mod.ContactAnalysisStudioWidget()

    # The per-position visualizer is the Visualize Contacts plugin now; drive
    # its embedded view by intercepting the widget-level set_context.
    plugin = widget._plugin_sections["visualize_contacts"].body
    calls: list[dict] = []
    monkeypatch.setattr(plugin._view, "set_context", lambda **kw: calls.append(kw))

    records = [
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
    _load(widget, records)

    # One row selected -> contact view targets that position with its paths.
    widget._panel.set_active(str(records[0]["position_path"]))
    app.processEvents()
    assert calls[-1]["out_path"] == records[0]["contact_analysis_path"]
    assert calls[-1]["cell_labels"] == records[0]["cell_tracked_labels_path"]

    # Multi-selection is ambiguous for a single view -> cleared.
    widget._panel.select_all()
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


def test_discover_levels_add_then_csv_roundtrip(tmp_path):
    app = _app()
    study = tmp_path / "study"
    _make_ready_position(study, "ctrl", "exp1", "pos01")
    _make_ready_position(study, "ctrl", "exp1", "pos02")

    widget = mod.ContactAnalysisStudioWidget()
    panel = widget._panel
    panel.input_name_fields["cell"].setText("3_cell/tracked_labels.tif")
    panel.input_name_fields["nucleus"].setText("2_nucleus/tracked_labels.tif")

    # One additive Find adds both folders straight to the list.
    found = panel.discover(str(study))
    assert len(found) == 2
    assert len(panel.keys()) == 2
    # The three nesting levels (study → condition → experiment → position) are
    # seeded with the recognized identity axes, innermost = position_id.
    assert set(found[0]["columns"]) >= {"condition", "experiment_id", "position_id"}

    records = widget._all_records()
    assert len(records) == 2
    by_id = {r["id"]: r for r in records}
    assert set(by_id) == {"pos01", "pos02"}
    assert all(r["condition"] == "ctrl" for r in records)
    assert all(r["experiment_id"] == "exp1" for r in records)
    assert all(r["cell_tracked_labels_path"] is not None for r in records)

    # CSV round-trip preserves the label paths and the folder-derived columns.
    csv_path = tmp_path / "catalog.csv"
    widget._save_csv_to(csv_path)
    widget._on_clear_catalog()
    assert widget._panel.keys() == []
    widget._load_csv_from(csv_path)
    records = widget._all_records()
    assert len(records) == 2
    assert all(r["cell_tracked_labels_path"] is not None for r in records)
    assert all(r["nucleus_tracked_labels_path"] is not None for r in records)
    assert all(r["columns"]["position_id"] in {"pos01", "pos02"} for r in records)
    assert all(r["experiment_id"] == "exp1" for r in records)

    widget.deleteLater()
    app.processEvents()


def test_find_accumulates_across_roots(tmp_path):
    app = _app()
    study = tmp_path / "study"
    _make_ready_position(study, "wt", "exp1", "pos01")
    _make_ready_position(study, "ko", "exp1", "pos02")

    widget = mod.ContactAnalysisStudioWidget()
    panel = widget._panel
    panel.input_name_fields["cell"].setText("3_cell/tracked_labels.tif")

    # Two separate Finds accumulate into one list (deduped by folder path).
    panel.discover(str(study / "wt"))
    panel.discover(str(study / "ko"))
    assert len(panel.keys()) == 2
    # Re-scanning a root already listed adds nothing.
    panel.discover(str(study / "wt"))
    assert len(panel.keys()) == 2

    widget.deleteLater()
    app.processEvents()


def test_renaming_a_column_carries_folder_values(tmp_path):
    app = _app()
    study = tmp_path / "study"
    _make_ready_position(study, "ctrl", "exp1", "pos01")

    widget = mod.ContactAnalysisStudioWidget()
    panel = widget._panel
    panel.input_name_fields["cell"].setText("3_cell/tracked_labels.tif")
    panel.discover(str(study))

    # The folder-derived columns seed to the recognized identity axes; renaming
    # the outermost column (condition → genotype) carries the folder value across
    # while position_id stays pinned to the innermost folder name.
    assert panel.column_names()[0] == "condition"
    panel.rename_column(0, "genotype")
    cols = panel.records()[0]["columns"]
    assert cols["genotype"] == "ctrl"
    assert "condition" not in cols
    assert cols["position_id"] == "pos01"

    widget.deleteLater()
    app.processEvents()


def test_discover_refuses_mixed_depth_positions(tmp_path, monkeypatch):
    app = _app()
    study = tmp_path / "study"
    _make_ready_position(study, "ctrl", "exp1", "pos01")  # depth 3
    shallow = study / "loose_pos"  # depth 1
    (shallow / "3_cell").mkdir(parents=True)
    (shallow / "3_cell" / "tracked_labels.tif").touch()

    widget = mod.ContactAnalysisStudioWidget()
    widget._panel.input_name_fields["cell"].setText("3_cell/tracked_labels.tif")
    # The Discover button opens a folder dialog; feed it the mixed-depth root.
    monkeypatch.setattr(
        mod.QFileDialog, "getExistingDirectory", lambda *a, **k: str(study)
    )
    widget._on_discover_requested()

    # Differing depths can't line up to named levels → nothing added, a warning.
    assert widget._panel.keys() == []
    assert "differing folder depths" in widget._catalog_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_discover_button_opens_dialog_and_adds(tmp_path, monkeypatch):
    app = _app()
    study = tmp_path / "study"
    _make_ready_position(study, "ctrl", "exp1", "pos01")
    _make_ready_position(study, "ctrl", "exp1", "pos02")

    widget = mod.ContactAnalysisStudioWidget()
    widget._panel.input_name_fields["cell"].setText("3_cell/tracked_labels.tif")
    monkeypatch.setattr(
        mod.QFileDialog, "getExistingDirectory", lambda *a, **k: str(study)
    )
    widget._on_discover_requested()
    # Uniform-depth root → both folders added straight to the list.
    assert len(widget._panel.keys()) == 2

    widget.deleteLater()
    app.processEvents()


def test_delete_via_panel_drops_only_that_row(tmp_path):
    app = _app()
    widget = mod.ContactAnalysisStudioWidget()
    _load(widget, [
        {"condition": "WT", "date": "d1", "id": "p1",
         "contact_analysis_path": Path("/a.h5"), "contact_analysis_status": "ready"},
        {"condition": "WT", "date": "d1", "id": "p2",
         "contact_analysis_path": Path("/b.h5"), "contact_analysis_status": "ready"},
        {"condition": "KO", "date": "d1", "id": "p3",
         "contact_analysis_path": Path("/c.h5"), "contact_analysis_status": "ready"},
    ])
    assert len(widget._panel.keys()) == 3

    # Select the KO row and delete it via the panel's own Delete-selected path.
    widget._panel.set_active("p3")
    app.processEvents()
    widget._panel.delete_selected()
    assert [r["id"] for r in widget._all_records()] == ["p1", "p2"]

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
    # pos01 already built (in the shared 4_contact_analysis/ folder); pos02 missing.
    (p1 / "4_contact_analysis").mkdir()
    (p1 / "4_contact_analysis" / "contact_analysis.h5").touch()

    widget = mod.ContactAnalysisStudioWidget()
    panel = widget._panel
    panel.input_name_fields["cell"].setText("cell_labels.tif")
    panel.discover(str(study))
    records = widget._all_records()
    assert len(records) == 2
    by_id = {r["id"]: r for r in records}
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
    widget = mod.ContactAnalysisStudioWidget()
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
    widget = mod.ContactAnalysisStudioWidget()
    _load(widget, [
        {"id": "p1", "position_path": pdir, "condition": "ctrl",
         "date": "d", "notes": ""},
    ])

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
    widget = mod.ContactAnalysisStudioWidget()
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
    widget = mod.ContactAnalysisStudioWidget()
    _load(widget, [
        {"condition": "ctrl", "date": "d1", "id": "p1",
         "contact_analysis_path": Path("/a.h5"), "contact_analysis_status": "ready"},
    ])

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
    widget = mod.ContactAnalysisStudioWidget()
    _load(widget, [
        {"condition": "ctrl", "date": "d1", "id": "p1",
         "contact_analysis_path": Path("/a.h5"), "contact_analysis_status": "ready"},
    ])
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
    widget = mod.ContactAnalysisStudioWidget()
    _load(widget, [
        {"condition": "ctrl", "date": "d1", "id": "p1",
         "contact_analysis_path": Path("/a.h5"), "contact_analysis_status": "ready"},
        {"condition": "drug", "date": "d1", "id": "p2",
         "contact_analysis_path": Path("/b.h5"), "contact_analysis_status": "ready"},
        {"condition": "drug", "date": "d2", "id": "p3",
         "contact_analysis_path": Path("/c.h5"), "contact_analysis_status": "ready"},
    ])

    widget._panel.set_active("p2")
    app.processEvents()
    widget._panel.delete_selected()

    assert [r["id"] for r in widget._all_records()] == ["p1", "p3"]
    assert len(widget._panel.keys()) == 2

    widget.deleteLater()
    app.processEvents()


def test_remove_selected_without_selection_is_a_noop():
    app = _app()
    widget = mod.ContactAnalysisStudioWidget()
    _load(widget, [
        {"condition": "ctrl", "date": "d1", "id": "p1",
         "contact_analysis_path": Path("/a.h5"), "contact_analysis_status": "ready"},
    ])
    widget._panel.clear_selection()
    app.processEvents()
    widget._panel.delete_selected()

    assert [r["id"] for r in widget._all_records()] == ["p1"]

    widget.deleteLater()
    app.processEvents()


def test_panel_reports_positions_and_selection():
    app = _app()
    widget = mod.ContactAnalysisStudioWidget()
    assert widget._panel.keys() == []  # empty catalog
    _load(widget, [
        {"condition": "c", "date": "d", "id": "p1", "contact_analysis_path": Path("/a.h5")},
        {"condition": "c", "date": "d", "id": "p2", "contact_analysis_path": Path("/b.h5")},
    ])
    assert len(widget._panel.keys()) == 2
    widget._panel.set_active("p1")
    app.processEvents()
    assert len(widget._panel.selected_payloads()) == 1
    widget.deleteLater()
    app.processEvents()


def test_clicking_nucleus_dot_loads_stage_into_viewer(tmp_path):
    import numpy as np
    import tifffile

    app = _app()

    class _Viewer:
        def __init__(self):
            self.layers = {}
            self.added = []

        def add_labels(self, data, name):
            self.layers[name] = data
            self.added.append(name)

        def add_image(self, data, name, colormap="gray"):
            self.layers[name] = data
            self.added.append(name)

    pos = tmp_path / "p1"
    pos.mkdir()
    tifffile.imwrite(str(pos / "nucleus_labels.tif"), np.zeros((4, 4), np.uint16))

    viewer = _Viewer()
    widget = mod.ContactAnalysisStudioWidget(viewer=viewer)
    record = {"condition": "c", "date": "d", "id": "p1", "position_path": pos,
              "contact_analysis_path": pos / "x.h5", "contact_analysis_status": "incomplete"}
    _load(widget, [record])

    # A rail-dot click emits stage_load_requested(payload, stage); the studio
    # loads that stage's outputs for the position.
    widget._on_stage_load_requested(record, "nucleus")
    assert viewer.added == ["p1:nucleus_labels"]

    # A row with no canonical folder reports it can't load.
    widget._on_stage_load_requested({"id": "loose", "position_path": None}, "nucleus")
    assert "cannot load" in widget._catalog_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_discovery_fields_default_to_committed_final_output_names():
    app = _app()
    widget = mod.ContactAnalysisStudioWidget()
    # Finalized positions (base-folder *_labels.tif) are auto-discovered by
    # default; the user overrides to a working stage file when uncommitted.
    fields = widget._panel.input_name_fields
    assert fields["cell"].text() == "cell_labels.tif"
    assert fields["nucleus"].text() == "nucleus_labels.tif"
    widget.deleteLater()
    app.processEvents()
