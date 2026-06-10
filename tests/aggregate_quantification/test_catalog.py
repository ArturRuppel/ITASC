"""Tests for cellflow.aggregate_quantification.catalog – CSV catalog + name-based discovery."""

from __future__ import annotations


def test_save_and_load_catalog_round_trip_with_relative_paths(tmp_path):
    """Saved catalogs should use relative paths and load them as resolved contact-analysis paths."""
    from cellflow.aggregate_quantification.catalog import load_catalog, save_catalog

    source = tmp_path / "analysis" / "contact_analysis.h5"
    source.parent.mkdir()
    source.touch()
    csv_path = tmp_path / "catalog.csv"

    cell = tmp_path / "analysis" / "cell_labels.tif"
    nucleus = tmp_path / "analysis" / "nucleus_labels.tif"
    cell.touch()
    nucleus.touch()

    save_catalog(csv_path, [{
        "path": source,
        "date": "2026-05-09",
        "condition": "treated",
        "id": "pos01",
        "notes": "edge,bright",
        "cell_tracked_labels_path": cell,
        "nucleus_tracked_labels_path": nucleus,
    }])

    csv_text = csv_path.read_text()
    assert "analysis/contact_analysis.h5" in csv_text
    # Label paths are persisted (relative) so a reloaded catalog can recompute.
    assert "analysis/cell_labels.tif" in csv_text
    assert "analysis/nucleus_labels.tif" in csv_text
    assert str(source) not in csv_text

    records = load_catalog(csv_path)

    assert len(records) == 1
    record = records[0]
    assert record["contact_analysis_path"] == source
    assert record["cell_tracked_labels_path"] == cell
    assert record["nucleus_tracked_labels_path"] == nucleus
    assert record["date"] == "2026-05-09"
    assert record["condition"] == "treated"
    assert record["id"] == "pos01"
    assert record["notes"] == "edge,bright"
    assert record["contact_analysis_status"] == "ready"


def test_save_load_round_trip_without_label_paths(tmp_path):
    """A record with no label paths round-trips with empty cells and None paths."""
    from cellflow.aggregate_quantification.catalog import load_catalog, save_catalog

    source = tmp_path / "contact_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"

    save_catalog(csv_path, [{
        "path": source, "date": "d1", "condition": "ctrl", "id": "p1", "notes": "",
    }])

    record = load_catalog(csv_path)[0]
    assert record["cell_tracked_labels_path"] is None
    assert record["nucleus_tracked_labels_path"] is None


def test_discover_catalog_entries_by_name_and_relative_path(tmp_path):
    """A folder's inputs are grouped into one entry; the contact path is derived."""
    from cellflow.aggregate_quantification.catalog import discover_catalog_entries

    # Two positions in a nested layout; one missing the nucleus labels.
    p1 = tmp_path / "expA" / "pos01"
    p2 = tmp_path / "expA" / "pos02"
    for p in (p1, p2):
        (p / "4_contact_analysis").mkdir(parents=True)
        (p / "3_cell").mkdir()
        (p / "4_contact_analysis" / "contact_analysis.h5").touch()
        (p / "3_cell" / "tracked_labels.tif").touch()
    (p1 / "2_nucleus").mkdir()
    (p1 / "2_nucleus" / "tracked_labels.tif").touch()

    entries = discover_catalog_entries(
        tmp_path,
        cell_name="3_cell/tracked_labels.tif",
        contact_name="4_contact_analysis/contact_analysis.h5",
        nucleus_name="2_nucleus/tracked_labels.tif",
    )

    assert [e["id"] for e in entries] == ["pos01", "pos02"]
    assert all(e["position_path"].name.startswith("pos") for e in entries)
    assert entries[0]["cell_tracked_labels_path"] == p1 / "3_cell" / "tracked_labels.tif"
    assert entries[0]["contact_analysis_path"] == p1 / "4_contact_analysis" / "contact_analysis.h5"
    assert entries[0]["nucleus_tracked_labels_path"] == p1 / "2_nucleus" / "tracked_labels.tif"
    # pos02 has no nucleus labels -> not associated.
    assert entries[1]["nucleus_tracked_labels_path"] is None
    # No metadata is assigned at discovery time.
    assert "condition" not in entries[0] and "date" not in entries[0]


def test_discover_catalog_entries_derives_missing_contact_path(tmp_path):
    """A position with cell labels but no .h5 is still discovered; the contact
    path is derived from the cell labels so it can be computed later."""
    from cellflow.aggregate_quantification.catalog import discover_catalog_entries

    pos = tmp_path / "pos01"
    pos.mkdir()
    (pos / "cell_labels.tif").touch()  # no contact_analysis.h5 yet

    entries = discover_catalog_entries(tmp_path, cell_name="cell_labels.tif")

    assert len(entries) == 1
    contact = entries[0]["contact_analysis_path"]
    assert contact == pos / "contact_analysis.h5"
    assert not contact.exists()
    assert entries[0]["cell_tracked_labels_path"] == pos / "cell_labels.tif"


def test_discover_catalog_entries_by_nucleus_only(tmp_path):
    """Inputs are optional: a folder with only nucleus labels is still a position."""
    from cellflow.aggregate_quantification.catalog import discover_catalog_entries

    pos = tmp_path / "pos01"
    pos.mkdir()
    (pos / "nucleus_labels.tif").touch()  # no cell labels at all

    entries = discover_catalog_entries(tmp_path, nucleus_name="nucleus_labels.tif")

    assert len(entries) == 1
    assert entries[0]["id"] == "pos01"
    assert entries[0]["nucleus_tracked_labels_path"] == pos / "nucleus_labels.tif"
    assert entries[0]["cell_tracked_labels_path"] is None
    # The contact-analysis output path is still derived even with no cell labels.
    assert entries[0]["contact_analysis_path"] == pos / "contact_analysis.h5"


def test_discover_catalog_entries_groups_inputs_from_different_subfolders(tmp_path):
    """Cell and nucleus inputs in different subfolders collapse to one entry."""
    from cellflow.aggregate_quantification.catalog import discover_catalog_entries

    pos = tmp_path / "pos01"
    (pos / "3_cell").mkdir(parents=True)
    (pos / "2_nucleus").mkdir()
    (pos / "3_cell" / "tracked_labels.tif").touch()
    (pos / "2_nucleus" / "tracked_labels.tif").touch()

    entries = discover_catalog_entries(
        tmp_path,
        cell_name="3_cell/tracked_labels.tif",
        nucleus_name="2_nucleus/tracked_labels.tif",
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry["cell_tracked_labels_path"] == pos / "3_cell" / "tracked_labels.tif"
    assert entry["nucleus_tracked_labels_path"] == pos / "2_nucleus" / "tracked_labels.tif"


def test_discover_catalog_entries_skips_folders_without_inputs(tmp_path):
    """A folder with none of the recognized inputs is not a position."""
    from cellflow.aggregate_quantification.catalog import discover_catalog_entries

    (tmp_path / "empty").mkdir()
    (tmp_path / "pos01").mkdir()
    (tmp_path / "pos01" / "cell_labels.tif").touch()

    entries = discover_catalog_entries(
        tmp_path, cell_name="cell_labels.tif", nucleus_name="nucleus_labels.tif"
    )

    assert [e["id"] for e in entries] == ["pos01"]


def test_discover_catalog_entries_without_any_input_names_is_empty(tmp_path):
    """With no input names supplied there is nothing to anchor discovery on."""
    from cellflow.aggregate_quantification.catalog import discover_catalog_entries

    (tmp_path / "pos01").mkdir()
    (tmp_path / "pos01" / "cell_labels.tif").touch()

    assert discover_catalog_entries(tmp_path) == []


def test_load_catalog_resolves_relative_paths_from_csv_parent(tmp_path):
    """Relative path cells should be resolved against the catalog file directory."""
    from cellflow.aggregate_quantification.catalog import load_catalog

    source = tmp_path / "nested" / "contact_analysis.h5"
    source.parent.mkdir()
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "path,date,condition,id,labels\n"
        "nested/contact_analysis.h5,day1,control,pos00,\n"
    )

    records = load_catalog(csv_path)

    assert records[0]["path"] == source
    assert records[0]["contact_analysis_path"] == source
    assert records[0]["contact_analysis_status"] == "ready"


def test_load_catalog_preserves_extra_columns(tmp_path):
    """Extra CSV columns should remain available in loaded record dictionaries."""
    from cellflow.aggregate_quantification.catalog import load_catalog

    source = tmp_path / "contact_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "path,date,condition,id,labels,operator\n"
        "contact_analysis.h5,day1,control,pos00,,Ada\n"
    )

    records = load_catalog(csv_path)

    assert records[0]["operator"] == "Ada"


def test_load_catalog_reports_missing_required_columns(tmp_path):
    """Catalog validation errors should name missing required columns."""
    import pytest

    from cellflow.aggregate_quantification.catalog import load_catalog

    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text("path,date,condition,labels\nmissing.h5,day1,c,\n")

    with pytest.raises(ValueError, match="id"):
        load_catalog(csv_path)


def test_load_catalog_marks_missing_h5_as_incomplete(tmp_path):
    """Explicit CSV records do not require labels, but missing H5 files are incomplete."""
    from cellflow.aggregate_quantification.catalog import load_catalog

    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "path,date,condition,id,labels\n"
        "missing.h5,day1,control,pos00,\n"
    )

    records = load_catalog(csv_path)

    assert records[0]["contact_analysis_path"] == tmp_path / "missing.h5"
    assert records[0]["contact_analysis_status"] == "incomplete"


def test_merge_catalog_records_skips_duplicate_resolved_paths(tmp_path):
    """Merging should avoid duplicate H5 sources by resolved contact-analysis path."""
    from cellflow.aggregate_quantification.catalog import merge_catalog_records

    source = tmp_path / "contact_analysis.h5"
    other = tmp_path / "other.h5"
    source.touch()
    other.touch()

    merged = merge_catalog_records(
        [{"path": source, "date": "d", "condition": "c", "id": "p", "labels": ""}],
        [
            {"path": source, "date": "d", "condition": "c", "id": "dupe", "labels": ""},
            {"path": other, "date": "d", "condition": "c", "id": "other", "labels": ""},
        ],
    )

    assert [record["path"] for record in merged] == [source, other]
