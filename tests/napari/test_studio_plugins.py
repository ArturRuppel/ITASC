from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from cellflow.napari import studio_plugins as sp


def _app():
    return QApplication.instance() or QApplication([])


def test_available_studio_plugins_has_contacts_builder_and_analysis_plugins():
    entries = sp.available_studio_plugins(build_callback=lambda *a: None)
    ids = {e.plugin_id for e in entries}
    assert "build:contacts" in ids  # one builder per quantifier
    assert "catalog_summary" in ids  # meta plugins folded into the same list
    builder = next(e for e in entries if e.plugin_id == "build:contacts")
    assert builder.display_name == "Build: Cell–cell contacts"
    assert builder.requires == ("cell_labels_path",)


def test_group_plugin_suppresses_its_quantitys_auto_builder():
    entries = sp.available_studio_plugins(build_callback=lambda *a: None)
    ids = {e.plugin_id for e in entries}
    # The Cell Shape group plugin owns cell_shape's build, so no auto-builder.
    assert "cell_shape" in ids
    assert "build:cell_shape" not in ids
    # Quantities without a dedicated plugin keep their free auto-builder.
    assert "build:contacts" in ids


def test_position_inputs_from_record_maps_catalogue_keys():
    inputs = sp.position_inputs_from_record(
        {
            "position_path": Path("/study/p1"),
            "cell_tracked_labels_path": Path("/study/p1/cells.tif"),
            "nucleus_tracked_labels_path": None,
        }
    )
    assert inputs.position_dir == Path("/study/p1")
    assert inputs.cell_labels_path == Path("/study/p1/cells.tif")
    assert inputs.nucleus_labels_path is None


def test_output_for_record_routes_each_quantifier_to_its_own_artifact():
    from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier
    from cellflow.aggregate_quantification.quantifiers.contacts import ContactsQuantifier

    # A record whose explicit (possibly nested) contacts path differs from the
    # bare default; contacts must honour the column, cell_shape must not.
    record = {
        "position_path": Path("/study/p1"),
        "contact_analysis_path": Path("/study/p1/4_contact_analysis/contact_analysis.h5"),
        "cell_tracked_labels_path": Path("/study/p1/cells.tif"),
    }

    assert sp.output_for_record(ContactsQuantifier(), record) == Path(
        "/study/p1/4_contact_analysis/contact_analysis.h5"
    )
    # The second quantifier derives its own path — never the contacts artifact.
    assert sp.output_for_record(CellShapeQuantifier(), record) == Path(
        "/study/p1/cell_shape.h5"
    )


def test_records_satisfying_filters_by_requires():
    records = [
        {"id": "a", "contact_analysis_path": Path("/a.h5"), "cell_tracked_labels_path": Path("/a/c.tif")},
        {"id": "b", "contact_analysis_path": Path("/b.h5"), "cell_tracked_labels_path": None},
    ]
    assert [r["id"] for r in sp.records_satisfying(("cell_labels_path",), records)] == ["a"]
    # No requirement -> everything qualifies.
    assert len(sp.records_satisfying((), records)) == 2


def test_builder_plugin_build_button_gating_and_callback():
    app = _app()
    from cellflow.aggregate_quantification.quantifiers.contacts import ContactsQuantifier
    from cellflow.napari.aggregate_quantification.plugins import AnalysisContext

    captured: list = []
    plugin = sp.BuilderPlugin(ContactsQuantifier(), lambda q, recs, ow: captured.append((q, recs, ow)))

    # No buildable records -> button disabled.
    plugin.set_context(AnalysisContext(records=[{"id": "x", "contact_analysis_path": Path("/x.h5")}]))
    assert plugin._build_btn.isEnabled() is False

    # A record with cell labels -> enabled; clicking forwards (quantifier, records, overwrite).
    records = [{"id": "y", "contact_analysis_path": Path("/y.h5"), "cell_tracked_labels_path": Path("/y/c.tif")}]
    plugin.set_context(AnalysisContext(records=records))
    assert plugin._build_btn.isEnabled() is True
    plugin._overwrite_cb.setChecked(True)
    plugin._on_build()
    assert len(captured) == 1
    quantifier, recs, overwrite = captured[0]
    assert quantifier.quantity_id == "contacts"
    assert [r["id"] for r in recs] == ["y"]
    assert overwrite is True

    plugin.deleteLater()
    app.processEvents()
