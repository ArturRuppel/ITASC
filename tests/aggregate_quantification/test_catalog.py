"""Tests for cellflow.aggregate_quantification.catalog – CSV catalog + name-based discovery."""

from __future__ import annotations

import pytest


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


def test_save_catalog_stores_absolute_position_and_relative_files(tmp_path):
    """A record with a position folder stores it absolute and files relative to it."""
    from cellflow.aggregate_quantification.catalog import load_catalog, save_catalog

    position = tmp_path / "expA" / "pos00"
    (position / "4_contact_analysis").mkdir(parents=True)
    (position / "3_cell").mkdir()
    (position / "2_nucleus").mkdir()
    contact = position / "4_contact_analysis" / "contact_analysis.h5"
    cell = position / "3_cell" / "tracked_labels.tif"
    nucleus = position / "2_nucleus" / "tracked_labels.tif"
    for p in (contact, cell, nucleus):
        p.touch()
    csv_path = tmp_path / "catalog.csv"

    save_catalog(csv_path, [{
        "position_path": position,
        "path": contact,
        "date": "2026-05-09",
        "condition": "treated",
        "id": "pos00",
        "notes": "",
        "cell_tracked_labels_path": cell,
        "nucleus_tracked_labels_path": nucleus,
    }])

    csv_text = csv_path.read_text()
    # Position folder is absolute; files are relative to it.
    assert str(position) in csv_text
    assert "4_contact_analysis/contact_analysis.h5" in csv_text
    assert "3_cell/tracked_labels.tif" in csv_text
    assert "2_nucleus/tracked_labels.tif" in csv_text
    assert str(contact) not in csv_text

    record = load_catalog(csv_path)[0]
    assert record["position_path"] == position
    assert record["contact_analysis_path"] == contact
    assert record["cell_tracked_labels_path"] == cell
    assert record["nucleus_tracked_labels_path"] == nucleus
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


def test_experiment_id_defaults_to_date_when_absent(tmp_path):
    """experiment_id falls back to date as a *default* when not supplied."""
    from cellflow.aggregate_quantification.catalog import load_catalog, save_catalog

    source = tmp_path / "contact_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    save_catalog(csv_path, [{
        "path": source, "date": "2026-05-09", "condition": "c", "id": "p1", "notes": "",
    }])

    record = load_catalog(csv_path)[0]
    assert record["experiment_id"] == "2026-05-09"


def test_experiment_id_is_distinct_from_date_when_provided(tmp_path):
    """experiment_id is its own field, not an alias for date: the same experiment
    (paired) can carry two conditions / dates while sharing one experiment_id."""
    from cellflow.aggregate_quantification.catalog import load_catalog, save_catalog

    source = tmp_path / "contact_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    save_catalog(csv_path, [{
        "path": source, "experiment_id": "EXP-01", "date": "2026-05-09",
        "condition": "treated", "id": "p1", "notes": "",
    }])

    record = load_catalog(csv_path)[0]
    assert record["experiment_id"] == "EXP-01"
    assert record["date"] == "2026-05-09"


def test_experiment_id_column_is_optional_for_legacy_catalogs(tmp_path):
    """A hand-written CSV lacking the experiment_id column still loads, with
    experiment_id defaulting to date (backward compatible)."""
    from cellflow.aggregate_quantification.catalog import load_catalog

    source = tmp_path / "contact_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text("path,date,condition,id\ncontact_analysis.h5,day1,ctrl,p1\n")

    record = load_catalog(csv_path)[0]
    assert record["experiment_id"] == "day1"


def test_load_catalog_rejects_duplicate_identity(tmp_path):
    """Two rows sharing (experiment_id, condition, position_id) would silently
    merge two positions' cells downstream; loading must error instead."""
    from cellflow.aggregate_quantification.catalog import load_catalog

    (tmp_path / "a.h5").touch()
    (tmp_path / "b.h5").touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "path,date,condition,experiment_id,id\n"
        "a.h5,d1,ctrl,EXP-01,Pos0\n"
        "b.h5,d1,ctrl,EXP-01,Pos0\n"
    )

    with pytest.raises(ValueError, match="Pos0"):
        load_catalog(csv_path)


def test_load_catalog_allows_same_position_id_in_different_condition(tmp_path):
    """The same position_id under a different condition is a distinct identity."""
    from cellflow.aggregate_quantification.catalog import load_catalog

    (tmp_path / "a.h5").touch()
    (tmp_path / "b.h5").touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "path,date,condition,experiment_id,id\n"
        "a.h5,d1,ctrl,EXP-01,Pos0\n"
        "b.h5,d1,drug,EXP-01,Pos0\n"
    )

    assert len(load_catalog(csv_path)) == 2


def test_discover_catalog_entries_by_name_and_relative_path(tmp_path):
    """A folder's inputs are grouped into one entry; the contact path is derived."""
    from cellflow.aggregate_quantification.catalog import discover_catalog_entries

    # Two positions in a nested layout; one missing the nucleus labels.
    p1 = tmp_path / "expA" / "pos01"
    p2 = tmp_path / "expA" / "pos02"
    for p in (p1, p2):
        (p / "3_cell").mkdir(parents=True)
        (p / "3_cell" / "tracked_labels.tif").touch()
    (p1 / "2_nucleus").mkdir()
    (p1 / "2_nucleus" / "tracked_labels.tif").touch()

    # Contact analysis is a derived output, not a discovery input — no contact_name.
    entries = discover_catalog_entries(
        tmp_path,
        cell_name="3_cell/tracked_labels.tif",
        nucleus_name="2_nucleus/tracked_labels.tif",
    )

    assert [e["id"] for e in entries] == ["pos01", "pos02"]
    assert all(e["position_path"].name.startswith("pos") for e in entries)
    assert entries[0]["cell_tracked_labels_path"] == p1 / "3_cell" / "tracked_labels.tif"
    # The contact path is always the fixed default output location.
    assert entries[0]["contact_analysis_path"] == (
        p1 / "aggregate_quantification" / "contact_analysis.h5"
    )
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
    assert contact == pos / "aggregate_quantification" / "contact_analysis.h5"
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
    assert entries[0]["contact_analysis_path"] == pos / "aggregate_quantification" / "contact_analysis.h5"


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


def test_relative_levels_returns_segments_from_root_to_position(tmp_path):
    """The path from root (exclusive) to the position folder (inclusive)."""
    from cellflow.aggregate_quantification.catalog import relative_levels

    pos = tmp_path / "WT" / "2024-01-15" / "pos3"
    pos.mkdir(parents=True)
    assert relative_levels(tmp_path, pos) == ("WT", "2024-01-15", "pos3")


def test_columns_from_levels_zips_names_to_segments(tmp_path):
    """Each level name maps to its folder-name value; blank names drop out."""
    from cellflow.aggregate_quantification.catalog import columns_from_levels

    cols = columns_from_levels(
        ["condition", "experiment_id", "position_id"], ("WT", "2024-01-15", "pos3")
    )
    assert cols == {"condition": "WT", "experiment_id": "2024-01-15", "position_id": "pos3"}
    # A blank level name contributes no column; extra segments without a name drop.
    assert columns_from_levels(["", "experiment_id"], ("WT", "2024-01-15")) == {
        "experiment_id": "2024-01-15"
    }


def test_discovered_level_depth_uniform_and_mixed(tmp_path):
    """Uniform depth -> that depth; differing depths -> None (caller warns)."""
    from cellflow.aggregate_quantification.catalog import discovered_level_depth

    a = tmp_path / "WT" / "e1" / "pos1"
    b = tmp_path / "KO" / "e2" / "pos2"
    for p in (a, b):
        p.mkdir(parents=True)
    assert discovered_level_depth(tmp_path, [a, b]) == 3

    shallow = tmp_path / "KO" / "pos3"
    shallow.mkdir()
    assert discovered_level_depth(tmp_path, [a, shallow]) is None


def test_save_catalog_persists_extra_free_form_columns(tmp_path):
    """A record's free-form columns are written to the CSV and reload-preserved."""
    from cellflow.aggregate_quantification.catalog import load_catalog, save_catalog

    source = tmp_path / "contact_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    save_catalog(csv_path, [{
        "path": source,
        "columns": {
            "condition": "WT",
            "experiment_id": "E1",
            "position_id": "pos3",
            "replicate": "r2",
        },
    }])

    csv_text = csv_path.read_text()
    assert "replicate" in csv_text.splitlines()[0]  # extra column in the header
    assert "r2" in csv_text

    record = load_catalog(csv_path)[0]
    # Recognized levels become the identity/axis fields.
    assert record["condition"] == "WT"
    assert record["experiment_id"] == "E1"
    assert record["id"] == "pos3"
    # The extra column round-trips, both flat and in the columns bag.
    assert record["columns"]["replicate"] == "r2"
    assert record["replicate"] == "r2"


def test_extra_columns_do_not_join_identity(tmp_path):
    """Two rows differing only in an extra column still share an identity (error)."""
    from cellflow.aggregate_quantification.catalog import load_catalog

    (tmp_path / "a.h5").touch()
    (tmp_path / "b.h5").touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "path,date,condition,experiment_id,id,replicate\n"
        "a.h5,d1,ctrl,EXP-01,Pos0,r1\n"
        "b.h5,d1,ctrl,EXP-01,Pos0,r2\n"
    )
    with pytest.raises(ValueError, match="Pos0"):
        load_catalog(csv_path)


def test_legacy_flat_record_gains_a_columns_bag(tmp_path):
    """Old-style flat records (no columns key) still save/load and expose a bag."""
    from cellflow.aggregate_quantification.catalog import load_catalog, save_catalog

    source = tmp_path / "contact_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    save_catalog(csv_path, [{
        "path": source, "date": "d1", "condition": "ctrl", "id": "p1", "notes": "",
    }])

    record = load_catalog(csv_path)[0]
    assert record["columns"]["condition"] == "ctrl"
    assert record["columns"]["position_id"] == "p1"
    # No spurious extra columns for a plain record.
    assert "replicate" not in record["columns"]


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
