from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from cellflow.napari import studio_plugins as sp


def _app():
    return QApplication.instance() or QApplication([])


def test_available_tool_plugins_are_analysis_tools_not_builders():
    entries = sp.available_tool_plugins()
    ids = {e.plugin_id for e in entries}
    assert "catalog_summary" in ids  # analysis tools surface here
    # Building is no longer a per-tool plugin — metrics live in the Build area.
    assert not any(pid.startswith("build:") for pid in ids)
    # The retired plot plugins are gone from the tool list too.
    assert "shape" not in ids
    assert "track_dynamics" not in ids


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
        "/study/p1/aggregate_quantification/cell_shape.csv"
    )


def test_records_satisfying_filters_by_requires():
    records = [
        {"id": "a", "contact_analysis_path": Path("/a.h5"), "cell_tracked_labels_path": Path("/a/c.tif")},
        {"id": "b", "contact_analysis_path": Path("/b.h5"), "cell_tracked_labels_path": None},
    ]
    assert [r["id"] for r in sp.records_satisfying(("cell_labels_path",), records)] == ["a"]
    # No requirement -> everything qualifies.
    assert len(sp.records_satisfying((), records)) == 2


def test_build_area_status_dots_and_run_callback(tmp_path):
    app = _app()
    from cellflow.aggregate_quantification.quantifiers.contacts import ContactsQuantifier
    from cellflow.napari.aggregate_quantification.plugins import AnalysisContext

    captured: list = []
    quantifier = ContactsQuantifier()
    area = sp.BuildArea([quantifier], lambda qs, recs, ow: captured.append((qs, recs, ow)))
    row = area._rows["contacts"]

    # No in-scope position has the inputs -> grey dot, checkbox disabled.
    area.set_context(AnalysisContext(records=[{"id": "x", "contact_analysis_path": Path("/x.h5")}]))
    assert sp._DOT_NONE in row.dot.styleSheet()
    assert row.checkbox.isEnabled() is False

    # One built + one missing among applicable positions -> red (not all built).
    p1, p2 = tmp_path / "p1", tmp_path / "p2"
    p1.mkdir()
    p2.mkdir()
    (p1 / "contact_analysis.h5").touch()
    records = [
        {"id": "p1", "position_path": p1, "contact_analysis_path": p1 / "contact_analysis.h5",
         "cell_tracked_labels_path": p1 / "c.tif"},
        {"id": "p2", "position_path": p2, "contact_analysis_path": p2 / "contact_analysis.h5",
         "cell_tracked_labels_path": p2 / "c.tif"},
    ]
    area.set_context(AnalysisContext(records=records))
    assert sp._DOT_MISSING in row.dot.styleSheet()
    assert row.checkbox.isEnabled() is True

    # All applicable positions built -> green.
    (p2 / "contact_analysis.h5").touch()
    area.set_context(AnalysisContext(records=records))
    assert sp._DOT_ALL in row.dot.styleSheet()

    # Run forwards the checked metrics, the scope, and overwrite=True.
    row.checkbox.setChecked(True)
    area._on_run()
    assert len(captured) == 1
    quantifiers, recs, overwrite = captured[0]
    assert [q.quantity_id for q in quantifiers] == ["contacts"]
    assert [r["id"] for r in recs] == ["p1", "p2"]
    assert overwrite is True

    area.deleteLater()
    app.processEvents()
