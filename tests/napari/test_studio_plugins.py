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


def test_contacts_declares_it_produces_the_contact_analysis_input():
    from cellflow.aggregate_quantification.quantifiers.contacts import ContactsQuantifier

    # The contacts-derived metrics consume this field; the producer link is what
    # lets the Build area draw the dependency generically.
    assert ContactsQuantifier.produces == "contact_analysis_path"


def test_group_build_metrics_nests_derived_under_their_producer():
    from cellflow.aggregate_quantification.quantifier import available_quantifiers

    quantifiers = [cls() for cls in available_quantifiers()]
    groups = sp.group_build_metrics(quantifiers)
    labels = [g.label for g in groups]

    # Raw metrics are grouped by their source layer; a derived group trails its
    # producer's group.
    assert "Cell" in labels and "Nucleus" in labels and "Cell + nucleus" in labels
    derived = next(g for g in groups if g.derived)
    assert derived.label == "Derived from Cell–cell contacts"
    member_ids = {q.quantity_id for q in derived.members}
    assert {"neighbor_count", "neighbor_enrichment", "contact_type_zscore"} <= member_ids
    # The derived group is placed right after the group that holds its producer.
    cell_idx = labels.index("Cell")
    assert labels[cell_idx + 1] == "Derived from Cell–cell contacts"
    # Contacts itself stays a raw metric (it reads cell labels, not a product);
    # cell density now counts off the cell labels too, so it is a raw Cell metric.
    cell_group = next(g for g in groups if g.label == "Cell")
    cell_ids = {q.quantity_id for q in cell_group.members}
    assert {"contacts", "cell_density"} <= cell_ids


def test_metric_input_labels_name_the_producer_for_derived_inputs():
    from cellflow.aggregate_quantification.quantifier import available_quantifiers

    quantifiers = [cls() for cls in available_quantifiers()]
    producers = sp.producers_by_field(quantifiers)
    by_id = {q.quantity_id: q for q in quantifiers}

    # A raw metric shows its source-file inputs by friendly name.
    assert sp.metric_input_labels(by_id["cell_shape"], producers) == [
        "cell labels",
        "pixel size",
    ]
    # A derived metric shows its dependency by the producer's display name.
    assert sp.metric_input_labels(by_id["neighbor_count"], producers) == [
        "Cell–cell contacts"
    ]


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
