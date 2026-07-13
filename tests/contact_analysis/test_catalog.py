"""Tests for cellflow.contact_analysis.catalog – CSV catalog + name-based discovery.

The catalog holds exactly the structural path columns plus the *classification
columns* the widget defined (whatever they are named) — no privileged
``condition`` / ``experiment_id`` / ``id`` columns. A position is identified by the
combination of its own columns; uniqueness is enforced at aggregate time, not on
load (see ``tests/contact_analysis/test_shape_tables.py``).
"""

from __future__ import annotations

import csv


def _structural_header():
    return ["position_path", "contact_analysis", "cell_labels", "nucleus_labels"]


def test_saved_catalog_header_is_structural_plus_widget_columns(tmp_path):
    """The CSV header is the fixed path columns followed by the record's columns —
    one column per widget column, verbatim, and nothing else."""
    from cellflow.contact_analysis.catalog import save_catalog

    pos = tmp_path / "WT" / "p1"
    pos.mkdir(parents=True)
    record = {
        "position_path": pos,
        "contact_analysis_path": pos / "4_contact_analysis" / "contact_analysis.h5",
        "cell_tracked_labels_path": pos / "cell_labels.tif",
        "nucleus_tracked_labels_path": pos / "nucleus_labels.tif",
        "columns": {"genotype": "WT", "position_id": "p1"},
    }
    out = tmp_path / "catalog.csv"
    save_catalog(out, [record])

    with out.open(newline="") as fh:
        header = next(csv.reader(fh))
    # No privileged identity columns, no date / notes / path.
    assert "condition" not in header
    assert "experiment_id" not in header
    assert "id" not in header
    assert "date" not in header and "notes" not in header and "path" not in header
    assert header == _structural_header() + ["genotype", "position_id"]


def test_columns_are_named_exactly_as_the_widget(tmp_path):
    """Renamed widget columns land in the CSV under their own names — the point of
    the whole change: as many columns as the widget shows, named as it names them."""
    from cellflow.contact_analysis.catalog import load_catalog, save_catalog

    pos = tmp_path / "study" / "p1"
    pos.mkdir(parents=True)
    out = tmp_path / "catalog.csv"
    save_catalog(out, [{
        "position_path": pos,
        "contact_analysis_path": pos / "4_contact_analysis" / "contact_analysis.h5",
        "columns": {"drug": "nocodazole", "replicate": "r2", "field": "f07"},
    }])

    header = out.read_text().splitlines()[0].split(",")
    assert header == _structural_header() + ["drug", "replicate", "field"]

    record = load_catalog(out)[0]
    assert record["columns"] == {"drug": "nocodazole", "replicate": "r2", "field": "f07"}


def test_catalog_roundtrip(tmp_path):
    from cellflow.contact_analysis.catalog import load_catalog, save_catalog

    pos = tmp_path / "ctrl" / "p1"
    pos.mkdir(parents=True)
    record = {
        "position_path": pos,
        "contact_analysis_path": pos / "4_contact_analysis" / "contact_analysis.h5",
        "cell_tracked_labels_path": pos / "cell_labels.tif",
        "nucleus_tracked_labels_path": pos / "nucleus_labels.tif",
        "columns": {"condition": "ctrl", "position_id": "p1"},
    }
    out = tmp_path / "catalog.csv"
    save_catalog(out, [record])
    loaded = load_catalog(out)

    assert len(loaded) == 1
    r = loaded[0]
    assert r["columns"] == {"condition": "ctrl", "position_id": "p1"}
    assert r["contact_analysis_path"].name == "contact_analysis.h5"
    assert r["cell_tracked_labels_path"].name == "cell_labels.tif"
    assert r["nucleus_tracked_labels_path"].name == "nucleus_labels.tif"


def test_load_requires_position_path(tmp_path):
    import pytest
    from cellflow.contact_analysis.catalog import load_catalog

    out = tmp_path / "bad.csv"
    out.write_text("condition,pos\nctrl,p1\n")
    with pytest.raises(ValueError, match="position_path"):
        load_catalog(out)


def test_save_and_load_catalog_round_trip_with_relative_paths(tmp_path):
    """Saved catalogs use relative paths and reload them as resolved paths."""
    from cellflow.contact_analysis.catalog import load_catalog, save_catalog

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
        "columns": {"condition": "treated", "position_id": "pos01"},
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
    assert record["columns"]["condition"] == "treated"
    assert record["columns"]["position_id"] == "pos01"
    assert record["contact_analysis_status"] == "ready"


def test_save_catalog_stores_absolute_position_and_relative_files(tmp_path):
    """A record with a position folder stores it absolute and files relative to it."""
    from cellflow.contact_analysis.catalog import load_catalog, save_catalog

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
        "columns": {"condition": "treated", "position_id": "pos00"},
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
    from cellflow.contact_analysis.catalog import load_catalog, save_catalog

    source = tmp_path / "contact_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"

    save_catalog(csv_path, [{
        "path": source, "columns": {"condition": "ctrl", "position_id": "p1"},
    }])

    record = load_catalog(csv_path)[0]
    assert record["cell_tracked_labels_path"] is None
    assert record["nucleus_tracked_labels_path"] is None


def test_no_columns_still_saves_and_loads(tmp_path):
    """A record with no classification columns is legal to persist; the CSV is just
    the structural columns. (Aggregating >1 such position is what fails, later.)"""
    from cellflow.contact_analysis.catalog import load_catalog, save_catalog

    source = tmp_path / "contact_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    save_catalog(csv_path, [{"path": source, "position_path": tmp_path}])

    header = csv_path.read_text().splitlines()[0].split(",")
    assert header == _structural_header()
    record = load_catalog(csv_path)[0]
    assert record["columns"] == {}


def test_load_maps_legacy_id_column_to_position_id(tmp_path):
    """A catalog written by the old code carried a privileged ``id`` column; it must
    still load, its value carried under ``position_id`` (which is what it meant)."""
    from cellflow.contact_analysis.catalog import load_catalog

    (tmp_path / "contact_analysis.h5").touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "position_path,path,condition,experiment_id,id\n"
        ",contact_analysis.h5,ctrl,EXP-01,Pos0\n"
    )

    record = load_catalog(csv_path)[0]
    assert record["columns"]["position_id"] == "Pos0"
    assert record["columns"]["condition"] == "ctrl"
    assert record["columns"]["experiment_id"] == "EXP-01"
    # The reserved ``id`` name never becomes a classification column.
    assert "id" not in record["columns"]


def test_load_does_not_reject_duplicate_identity(tmp_path):
    """Load is permissive: a catalog may hold more positions than one run pools, so
    identity uniqueness is checked at aggregate time, not on load."""
    from cellflow.contact_analysis.catalog import load_catalog

    (tmp_path / "a.h5").touch()
    (tmp_path / "b.h5").touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "position_path,path,condition,position_id\n"
        ",a.h5,ctrl,Pos0\n"
        ",b.h5,ctrl,Pos0\n"
    )
    # No raise — two identical identities load fine (the aggregator refuses them).
    assert len(load_catalog(csv_path)) == 2


def test_discover_catalog_entries_by_name_and_relative_path(tmp_path):
    """A folder's inputs are grouped into one entry; the contact path is derived."""
    from cellflow.contact_analysis.catalog import discover_catalog_entries

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
    assert entries[0]["contact_analysis_path"] == p1 / "contact_analysis.h5"
    assert entries[0]["nucleus_tracked_labels_path"] == p1 / "2_nucleus" / "tracked_labels.tif"
    # pos02 has no nucleus labels -> not associated.
    assert entries[1]["nucleus_tracked_labels_path"] is None
    # No metadata is assigned at discovery time.
    assert "condition" not in entries[0] and "date" not in entries[0]


def test_discover_entry_id_becomes_position_id_column_on_save(tmp_path):
    """The discovery skeleton's per-folder ``id`` persists as a ``position_id``
    classification column, so a discovered catalog has a usable identity."""
    from cellflow.contact_analysis.catalog import (
        discover_catalog_entries,
        load_catalog,
        save_catalog,
    )

    pos = tmp_path / "pos01"
    pos.mkdir()
    (pos / "cell_labels.tif").touch()
    entries = discover_catalog_entries(tmp_path, cell_name="cell_labels.tif")
    csv_path = tmp_path / "catalog.csv"
    save_catalog(csv_path, entries)

    assert "position_id" in csv_path.read_text().splitlines()[0].split(",")
    assert load_catalog(csv_path)[0]["columns"]["position_id"] == "pos01"


def test_discover_catalog_entries_derives_missing_contact_path(tmp_path):
    """A position with cell labels but no .h5 is still discovered; the contact
    path is derived from the cell labels so it can be computed later."""
    from cellflow.contact_analysis.catalog import discover_catalog_entries

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
    from cellflow.contact_analysis.catalog import discover_catalog_entries

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
    from cellflow.contact_analysis.catalog import discover_catalog_entries

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
    from cellflow.contact_analysis.catalog import discover_catalog_entries

    (tmp_path / "empty").mkdir()
    (tmp_path / "pos01").mkdir()
    (tmp_path / "pos01" / "cell_labels.tif").touch()

    entries = discover_catalog_entries(
        tmp_path, cell_name="cell_labels.tif", nucleus_name="nucleus_labels.tif"
    )

    assert [e["id"] for e in entries] == ["pos01"]


def test_discover_catalog_entries_without_any_input_names_is_empty(tmp_path):
    """With no input names supplied there is nothing to anchor discovery on."""
    from cellflow.contact_analysis.catalog import discover_catalog_entries

    (tmp_path / "pos01").mkdir()
    (tmp_path / "pos01" / "cell_labels.tif").touch()

    assert discover_catalog_entries(tmp_path) == []


def test_load_catalog_resolves_relative_paths_from_csv_parent(tmp_path):
    """Relative path cells should be resolved against the catalog file directory."""
    from cellflow.contact_analysis.catalog import load_catalog

    source = tmp_path / "nested" / "contact_analysis.h5"
    source.parent.mkdir()
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "position_path,path,condition,position_id\n"
        ",nested/contact_analysis.h5,control,pos00\n"
    )

    records = load_catalog(csv_path)

    assert records[0]["path"] == source
    assert records[0]["contact_analysis_path"] == source
    assert records[0]["contact_analysis_status"] == "ready"


def test_load_catalog_preserves_extra_columns(tmp_path):
    """Every non-structural CSV column is a classification column, available both as
    a flat key and in the columns bag."""
    from cellflow.contact_analysis.catalog import load_catalog

    source = tmp_path / "contact_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "position_path,path,condition,position_id,operator\n"
        ",contact_analysis.h5,control,pos00,Ada\n"
    )

    record = load_catalog(csv_path)[0]

    assert record["operator"] == "Ada"
    assert record["columns"]["operator"] == "Ada"
    assert record["columns"]["condition"] == "control"


def test_load_catalog_marks_missing_h5_as_incomplete(tmp_path):
    """Explicit CSV records do not require labels, but missing H5 files are incomplete."""
    from cellflow.contact_analysis.catalog import load_catalog

    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "position_path,path,condition,position_id\n"
        ",missing.h5,control,pos00\n"
    )

    records = load_catalog(csv_path)

    assert records[0]["contact_analysis_path"] == tmp_path / "missing.h5"
    assert records[0]["contact_analysis_status"] == "incomplete"


def test_relative_levels_returns_segments_from_root_to_position(tmp_path):
    """The path from root (exclusive) to the position folder (inclusive)."""
    from cellflow.contact_analysis.catalog import relative_levels

    pos = tmp_path / "WT" / "2024-01-15" / "pos3"
    pos.mkdir(parents=True)
    assert relative_levels(tmp_path, pos) == ("WT", "2024-01-15", "pos3")


def test_columns_from_levels_zips_names_to_segments(tmp_path):
    """Each level name maps to its folder-name value; blank names drop out."""
    from cellflow.contact_analysis.catalog import columns_from_levels

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
    from cellflow.contact_analysis.catalog import discovered_level_depth

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
    from cellflow.contact_analysis.catalog import load_catalog, save_catalog

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
    assert record["columns"] == {
        "condition": "WT",
        "experiment_id": "E1",
        "position_id": "pos3",
        "replicate": "r2",
    }
    # And each column is also available as a flat key.
    assert record["replicate"] == "r2"


def test_legacy_flat_record_gains_a_columns_bag(tmp_path):
    """Old-style flat records (no columns key) still save/load and expose a bag,
    with the legacy ``id`` carried as ``position_id``."""
    from cellflow.contact_analysis.catalog import load_catalog, save_catalog

    source = tmp_path / "contact_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    save_catalog(csv_path, [{
        "path": source, "condition": "ctrl", "id": "p1",
    }])

    record = load_catalog(csv_path)[0]
    assert record["columns"]["condition"] == "ctrl"
    assert record["columns"]["position_id"] == "p1"
    assert "id" not in record["columns"]
    assert "replicate" not in record["columns"]


def test_merge_catalog_records_skips_duplicate_resolved_paths(tmp_path):
    """Merging should avoid duplicate H5 sources by resolved contact-analysis path."""
    from cellflow.contact_analysis.catalog import merge_catalog_records

    source = tmp_path / "contact_analysis.h5"
    other = tmp_path / "other.h5"
    source.touch()
    other.touch()

    merged = merge_catalog_records(
        [{"path": source, "columns": {"condition": "c", "position_id": "p"}}],
        [
            {"path": source, "columns": {"condition": "c", "position_id": "dupe"}},
            {"path": other, "columns": {"condition": "c", "position_id": "other"}},
        ],
    )

    assert [record["path"] for record in merged] == [source, other]
