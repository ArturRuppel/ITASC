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
    from cellflow.contact_analysis.quantifiers.contacts import ContactsQuantifier

    # The contacts-derived metrics consume this field; the producer link is what
    # lets the Build area draw the dependency generically.
    assert ContactsQuantifier.produces == "contact_analysis_path"


def test_group_build_metrics_nests_derived_under_their_producer():
    from cellflow.contact_analysis.quantifier import available_quantifiers

    quantifiers = [cls() for cls in available_quantifiers()]
    groups = sp.group_build_metrics(quantifiers)
    labels = [g.label for g in groups]

    # Raw metrics are grouped by their source layer; a derived group trails its
    # producer's group.
    assert "Cell" in labels and "Nucleus" in labels and "Cell + nucleus" in labels
    derived = next(g for g in groups if g.derived)
    assert derived.label == "Derived from Cell–cell contacts"
    member_ids = {q.quantity_id for q in derived.members}
    assert {"neighbor_count", "signed_contact_length"} <= member_ids
    # The derived group is placed right after the group that holds its producer.
    cell_idx = labels.index("Cell")
    assert labels[cell_idx + 1] == "Derived from Cell–cell contacts"
    # Contacts itself stays a raw metric (it reads cell labels, not a product);
    # cell density now counts off the cell labels too, so it is a raw Cell metric.
    cell_group = next(g for g in groups if g.label == "Cell")
    cell_ids = {q.quantity_id for q in cell_group.members}
    assert {"contacts", "cell_density"} <= cell_ids


def test_metric_dependencies_list_files_then_params_in_registry_order():
    from cellflow.contact_analysis.quantifier import available_quantifiers

    quantifiers = [cls() for cls in available_quantifiers()]
    by_id = {q.quantity_id: q for q in quantifiers}

    # Pixel size is now a global param (required_build_params), not a file input —
    # cell_shape's dependencies are its file input then its param, registry order.
    assert sp.metric_dependencies(by_id["cell_shape"]) == [
        "cell_labels_path",
        "pixel_size_um",
    ]
    # Cell dynamics adds the frame-interval param after pixel size.
    assert sp.metric_dependencies(by_id["cell_dynamics"]) == [
        "cell_labels_path",
        "pixel_size_um",
        "time_interval_s",
    ]
    # A derived metric depends on the produced contacts intermediate (CA).
    assert sp.metric_dependencies(by_id["neighbor_count"]) == ["contact_analysis_path"]


def test_referenced_dependencies_dedupes_and_keeps_registry_order():
    from cellflow.contact_analysis.quantifier import available_quantifiers

    quantifiers = [cls() for cls in available_quantifiers()]
    referenced = sp.referenced_dependencies(quantifiers)

    # Deduped across all metrics, ordered by the registry, and only what's used.
    assert referenced == [
        "cell_labels_path",
        "nucleus_labels_path",
        "contact_analysis_path",
        "pixel_size_um",
        "time_interval_s",
        "fov_area_mm2",
    ]


def test_coverage_badge_marks_full_coverage_with_a_check():
    assert sp.coverage_badge(0, 5) == "0/5"
    assert sp.coverage_badge(2, 5) == "2/5"
    assert sp.coverage_badge(5, 5) == "5/5 ✓"
    # An empty applicable set is not "full" — no check.
    assert sp.coverage_badge(0, 0) == "0/0"


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
    from cellflow.contact_analysis.quantifiers.cell_shape import CellShapeQuantifier
    from cellflow.contact_analysis.quantifiers.contacts import ContactsQuantifier

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
        "/study/p1/4_contact_analysis/cell_shape.csv"
    )


def test_contact_analysis_path_input_requires_the_file_to_exist(tmp_path):
    # The catalogue stamps a contacts artifact path on every position whether or
    # not it has been built, so a derived metric must gate on the file actually
    # existing — otherwise its status dot reads red for positions that can never
    # build (no contacts product). See position_inputs_from_record.
    built, unbuilt = tmp_path / "built.h5", tmp_path / "unbuilt.h5"
    built.touch()
    records = [
        {"id": "has", "contact_analysis_path": built},
        {"id": "missing", "contact_analysis_path": unbuilt},
    ]
    assert sp.position_inputs_from_record(records[0]).contact_analysis_path == built
    assert sp.position_inputs_from_record(records[1]).contact_analysis_path is None
    # Only the position whose contacts artifact exists is applicable to a
    # contacts-derived metric (which ``requires`` the produced input).
    satisfied = sp.records_satisfying(("contact_analysis_path",), records)
    assert [r["id"] for r in satisfied] == ["has"]


def test_records_satisfying_filters_by_requires():
    records = [
        {"id": "a", "contact_analysis_path": Path("/a.h5"), "cell_tracked_labels_path": Path("/a/c.tif")},
        {"id": "b", "contact_analysis_path": Path("/b.h5"), "cell_tracked_labels_path": None},
    ]
    assert [r["id"] for r in sp.records_satisfying(("cell_labels_path",), records)] == ["a"]
    # No requirement -> everything qualifies.
    assert len(sp.records_satisfying((), records)) == 2


def test_build_area_coverage_badges_and_run_callback(tmp_path):
    app = _app()
    from cellflow.contact_analysis.quantifiers.contacts import ContactsQuantifier
    from cellflow.napari.contact_analysis.plugins import AnalysisContext

    captured: list = []
    quantifier = ContactsQuantifier()
    area = sp.BuildArea([quantifier], lambda qs, recs, ow: captured.append((qs, recs, ow)))
    row = area._rows["contacts"]

    # No in-scope position has the inputs -> 0/0 badge, checkbox disabled, and the
    # cell-labels legend entry reads 0/1 (one in-scope position, none with labels).
    area.set_context(AnalysisContext(records=[{"id": "x", "contact_analysis_path": Path("/x.h5")}]))
    assert row.badge.text() == "0/0"
    assert row.checkbox.isEnabled() is False
    assert area._legend._entries["cell_labels_path"][2].text() == "0/1"

    # One built + one missing among applicable positions -> partial 1/2, buildable.
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
    assert row.badge.text() == "1/2"
    assert row.checkbox.isEnabled() is True
    # Both positions carry cell labels -> the legend shows full coverage.
    assert area._legend._entries["cell_labels_path"][2].text() == "2/2 ✓"

    # All applicable positions built -> full coverage with a check.
    (p2 / "contact_analysis.h5").touch()
    area.set_context(AnalysisContext(records=records))
    assert row.badge.text() == "2/2 ✓"

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


def test_build_area_param_gating_blocks_until_param_set(tmp_path):
    # A metric with a required global param (pixel size) is not buildable until the
    # param is set, even when every file input is present. The chip dims when unset.
    app = _app()
    from cellflow.contact_analysis.quantifiers.cell_shape import CellShapeQuantifier
    from cellflow.napari.contact_analysis.plugins import AnalysisContext

    params: dict = {"pixel_size_um": None}
    area = sp.BuildArea(
        [CellShapeQuantifier()], lambda *a: None, params_provider=lambda: dict(params)
    )
    row = area._rows["cell_shape"]

    p1 = tmp_path / "p1"
    p1.mkdir()
    records = [{"id": "p1", "position_path": p1, "cell_tracked_labels_path": p1 / "c.tif"}]

    # Files present but pixel size unset -> not buildable; the px chip is struck.
    area.set_context(AnalysisContext(records=records))
    assert row.checkbox.isEnabled() is False
    assert "line-through" in row.chips["pixel_size_um"].styleSheet()
    assert area._legend._entries["pixel_size_um"][2].text() == "unset"

    # Set the param -> buildable; the px chip is no longer struck.
    params["pixel_size_um"] = 0.5
    area._refresh()
    assert row.checkbox.isEnabled() is True
    assert "line-through" not in row.chips["pixel_size_um"].styleSheet()
    assert area._legend._entries["pixel_size_um"][2].text() == "set ✓"

    area.deleteLater()
    app.processEvents()


def test_build_area_check_all_toggles_buildable_metrics(tmp_path):
    # Bug 22: a single button checks every buildable metric, then flips to
    # "Uncheck all" and clears them; disabled (no-input) metrics are left alone.
    app = _app()
    from cellflow.contact_analysis.quantifiers.contacts import ContactsQuantifier
    from cellflow.napari.contact_analysis.plugins import AnalysisContext

    quantifier = ContactsQuantifier()
    area = sp.BuildArea([quantifier], lambda *a: None)
    row = area._rows["contacts"]

    p1 = tmp_path / "p1"
    p1.mkdir()
    area.set_context(AnalysisContext(records=[
        {"id": "p1", "position_path": p1, "contact_analysis_path": p1 / "contact_analysis.h5",
         "cell_tracked_labels_path": p1 / "c.tif"},
    ]))
    assert row.checkbox.isEnabled() is True
    assert area._check_all_btn.text() == "Check all"

    area._check_all_btn.click()
    assert row.checkbox.isChecked() is True
    assert area._check_all_btn.text() == "Uncheck all"

    area._check_all_btn.click()
    assert row.checkbox.isChecked() is False
    assert area._check_all_btn.text() == "Check all"

    area.deleteLater()
    app.processEvents()
