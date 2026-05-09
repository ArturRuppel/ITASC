"""Tests for cellflow.meta.catalog.discover_study – study directory discovery."""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_ready_position(root: Path, condition: str, experiment: str, position: str) -> Path:
    """Create a fully-populated position directory and return its path."""
    pos_dir = root / condition / experiment / position
    (pos_dir / "4_analysis").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "3_cell").mkdir()
    (pos_dir / "4_analysis" / "position_analysis.h5").touch()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    return pos_dir


def _make_position_missing_artifact(root: Path, condition: str, experiment: str, position: str) -> Path:
    """Create a position missing the artifact file but with labels present."""
    pos_dir = root / condition / experiment / position
    (pos_dir / "4_analysis").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "3_cell").mkdir()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()
    (pos_dir / "3_cell" / "tracked_labels.tif").touch()
    return pos_dir


def _make_position_missing_labels(root: Path, condition: str, experiment: str, position: str) -> Path:
    """Create a position with the artifact but no label files."""
    pos_dir = root / condition / experiment / position
    (pos_dir / "4_analysis").mkdir(parents=True)
    (pos_dir / "4_analysis" / "position_analysis.h5").touch()
    return pos_dir


# ---------------------------------------------------------------------------
# discovery of a complete study
# ---------------------------------------------------------------------------

def test_discover_study_returns_ready_records_for_all_positions(tmp_path):
    """Every position with all three files should appear as analysis_status='ready'."""
    from cellflow.meta.catalog import discover_study

    root = tmp_path / "study"
    _make_ready_position(root, "wildtype", "2024-01-01", "pos00")
    _make_ready_position(root, "wildtype", "2024-01-01", "pos01")
    _make_ready_position(root, "mutant", "2024-01-02", "pos00")

    records = discover_study(root)

    assert len(records) == 3

    for rec in records:
        assert rec["analysis_status"] == "ready"
        assert rec["position_path"] == root / rec["condition_id"] / rec["experiment_id"] / rec["position_id"]
        assert rec["artifact_path"] == rec["position_path"] / "4_analysis" / "position_analysis.h5"
        assert rec["nucleus_tracked_labels_path"] == rec["position_path"] / "2_nucleus" / "tracked_labels.tif"
        assert rec["cell_tracked_labels_path"] == rec["position_path"] / "3_cell" / "tracked_labels.tif"

    ids = {(r["condition_id"], r["experiment_id"], r["position_id"]) for r in records}
    assert ids == {
        ("wildtype", "2024-01-01", "pos00"),
        ("wildtype", "2024-01-01", "pos01"),
        ("mutant", "2024-01-02", "pos00"),
    }


def test_discover_study_returns_sorted_records(tmp_path):
    """Records should be sorted by condition, experiment, then position."""
    from cellflow.meta.catalog import discover_study

    root = tmp_path / "study"
    _make_ready_position(root, "z_cond", "b_exp", "pos02")
    _make_ready_position(root, "a_cond", "c_exp", "pos01")
    _make_ready_position(root, "a_cond", "a_exp", "pos00")

    records = discover_study(root)

    compound = [(r["condition_id"], r["experiment_id"], r["position_id"]) for r in records]
    assert compound == [
        ("a_cond", "a_exp", "pos00"),
        ("a_cond", "c_exp", "pos01"),
        ("z_cond", "b_exp", "pos02"),
    ]


# ---------------------------------------------------------------------------
# positions with missing files
# ---------------------------------------------------------------------------

def test_discover_study_returns_not_ready_when_artifact_is_missing(tmp_path):
    """If 4_analysis/position_analysis.h5 is missing, status != 'ready'."""
    from cellflow.meta.catalog import discover_study

    root = tmp_path / "study"
    _make_position_missing_artifact(root, "cond", "exp", "pos00")

    records = discover_study(root)

    assert len(records) == 1
    assert records[0]["analysis_status"] != "ready"
    assert records[0]["condition_id"] == "cond"
    assert records[0]["experiment_id"] == "exp"
    assert records[0]["position_id"] == "pos00"


def test_discover_study_returns_not_ready_when_nucleus_labels_missing(tmp_path):
    """If 2_nucleus/tracked_labels.tif is missing, status != 'ready'."""
    from cellflow.meta.catalog import discover_study

    root = tmp_path / "study"
    pos_dir = root / "cond" / "exp" / "pos00"
    (pos_dir / "4_analysis").mkdir(parents=True)
    (pos_dir / "4_analysis" / "position_analysis.h5").touch()
    (pos_dir / "3_cell").mkdir()
    (pos_dir / "3_cell" / "tracked_labels.tif").touch()

    records = discover_study(root)

    assert len(records) == 1
    assert records[0]["analysis_status"] != "ready"


def test_discover_study_returns_not_ready_when_cell_labels_missing(tmp_path):
    """If 3_cell/tracked_labels.tif is missing, status != 'ready'."""
    from cellflow.meta.catalog import discover_study

    root = tmp_path / "study"
    pos_dir = root / "cond" / "exp" / "pos00"
    (pos_dir / "4_analysis").mkdir(parents=True)
    (pos_dir / "4_analysis" / "position_analysis.h5").touch()
    (pos_dir / "2_nucleus").mkdir()
    (pos_dir / "2_nucleus" / "tracked_labels.tif").touch()

    records = discover_study(root)

    assert len(records) == 1
    assert records[0]["analysis_status"] != "ready"


def test_discover_study_mixes_ready_and_not_ready_in_same_study(tmp_path):
    """Within one root, some positions may be ready and others not."""
    from cellflow.meta.catalog import discover_study

    root = tmp_path / "study"
    _make_ready_position(root, "cond", "exp", "pos00")
    _make_position_missing_artifact(root, "cond", "exp", "pos01")

    records = discover_study(root)

    assert len(records) == 2
    by_id = {(r["condition_id"], r["experiment_id"], r["position_id"]): r["analysis_status"] for r in records}
    assert by_id[("cond", "exp", "pos00")] == "ready"
    assert by_id[("cond", "exp", "pos01")] != "ready"


# ---------------------------------------------------------------------------
# empty / shallow roots
# ---------------------------------------------------------------------------

def test_discover_study_returns_empty_list_when_no_study_dirs_exist(tmp_path):
    """An empty directory produces no records."""
    from cellflow.meta.catalog import discover_study

    root = tmp_path / "study"
    root.mkdir()

    records = discover_study(root)

    assert records == []


def test_discover_study_skips_non_directory_children(tmp_path):
    """Regular files at any level should be skipped."""
    from cellflow.meta.catalog import discover_study

    root = tmp_path / "study"
    root.mkdir()
    (root / "README.txt").touch()

    records = discover_study(root)

    assert records == []


def test_discover_study_skips_condition_with_no_position_dirs(tmp_path):
    """An empty condition directory should produce no records."""
    from cellflow.meta.catalog import discover_study

    root = tmp_path / "study"
    (root / "empty_cond").mkdir(parents=True)

    records = discover_study(root)

    assert records == []


# ---------------------------------------------------------------------------
# record field completeness
# ---------------------------------------------------------------------------

def test_discover_study_record_has_all_required_keys(tmp_path):
    """Every record must contain exactly the expected set of fields."""
    from cellflow.meta.catalog import discover_study

    root = tmp_path / "study"
    _make_ready_position(root, "c", "e", "p")

    records = discover_study(root)

    assert len(records) == 1
    expected_keys = {
        "condition_id",
        "experiment_id",
        "position_id",
        "position_path",
        "artifact_path",
        "nucleus_tracked_labels_path",
        "cell_tracked_labels_path",
        "analysis_status",
    }
    assert set(records[0].keys()) == expected_keys


def test_discover_study_record_paths_are_pathlib_objects(tmp_path):
    """Path fields should be Path instances, not strings."""
    from cellflow.meta.catalog import discover_study

    root = tmp_path / "study"
    _make_ready_position(root, "c", "e", "p")

    records = discover_study(root)

    rec = records[0]
    assert isinstance(rec["position_path"], Path)
    assert isinstance(rec["artifact_path"], Path)
    assert isinstance(rec["nucleus_tracked_labels_path"], Path)
    assert isinstance(rec["cell_tracked_labels_path"], Path)


# ---------------------------------------------------------------------------
# CSV catalog helpers
# ---------------------------------------------------------------------------

def test_save_and_load_meta_catalog_round_trip_with_relative_paths(tmp_path):
    """Saved catalogs should use relative paths and load them as resolved artifact paths."""
    from cellflow.meta.catalog import load_meta_catalog, save_meta_catalog

    source = tmp_path / "analysis" / "position_analysis.h5"
    source.parent.mkdir()
    source.touch()
    csv_path = tmp_path / "catalog.csv"

    save_meta_catalog(csv_path, [{
        "path": source,
        "date": "2026-05-09",
        "condition": "treated",
        "id": "pos01",
        "labels": "edge,bright",
    }])

    csv_text = csv_path.read_text()
    assert "analysis/position_analysis.h5" in csv_text
    assert str(source) not in csv_text

    records = load_meta_catalog(csv_path)

    assert records == [{
        "path": source,
        "date": "2026-05-09",
        "condition": "treated",
        "id": "pos01",
        "labels": "edge,bright",
        "condition_id": "treated",
        "experiment_id": "2026-05-09",
        "position_id": "pos01",
        "artifact_path": source,
        "analysis_status": "ready",
    }]


def test_load_meta_catalog_resolves_relative_paths_from_csv_parent(tmp_path):
    """Relative path cells should be resolved against the catalog file directory."""
    from cellflow.meta.catalog import load_meta_catalog

    source = tmp_path / "nested" / "position_analysis.h5"
    source.parent.mkdir()
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "path,date,condition,id,labels\n"
        "nested/position_analysis.h5,day1,control,pos00,\n"
    )

    records = load_meta_catalog(csv_path)

    assert records[0]["path"] == source
    assert records[0]["artifact_path"] == source
    assert records[0]["analysis_status"] == "ready"


def test_load_meta_catalog_preserves_extra_columns(tmp_path):
    """Extra CSV columns should remain available in loaded record dictionaries."""
    from cellflow.meta.catalog import load_meta_catalog

    source = tmp_path / "position_analysis.h5"
    source.touch()
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "path,date,condition,id,labels,operator\n"
        "position_analysis.h5,day1,control,pos00,,Ada\n"
    )

    records = load_meta_catalog(csv_path)

    assert records[0]["operator"] == "Ada"


def test_load_meta_catalog_reports_missing_required_columns(tmp_path):
    """Catalog validation errors should name missing required columns."""
    import pytest

    from cellflow.meta.catalog import load_meta_catalog

    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text("path,date,condition,labels\nmissing.h5,day1,c,\n")

    with pytest.raises(ValueError, match="id"):
        load_meta_catalog(csv_path)


def test_load_meta_catalog_marks_missing_h5_as_incomplete(tmp_path):
    """Explicit CSV records do not require labels, but missing H5 files are incomplete."""
    from cellflow.meta.catalog import load_meta_catalog

    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "path,date,condition,id,labels\n"
        "missing.h5,day1,control,pos00,\n"
    )

    records = load_meta_catalog(csv_path)

    assert records[0]["artifact_path"] == tmp_path / "missing.h5"
    assert records[0]["analysis_status"] == "incomplete"


def test_discover_h5_files_returns_nested_h5_paths_sorted(tmp_path):
    """Folder discovery should find .h5 files recursively by default."""
    from cellflow.meta.catalog import discover_h5_files

    nested = tmp_path / "a" / "4_analysis"
    nested.mkdir(parents=True)
    first = nested / "position_analysis.h5"
    second = tmp_path / "b.h5"
    ignored = tmp_path / "notes.txt"
    second.touch()
    first.touch()
    ignored.touch()

    assert discover_h5_files(tmp_path) == sorted([first, second])
    assert discover_h5_files(tmp_path, recursive=False) == [second]


def test_records_from_h5_paths_uses_defaults_and_unique_ids(tmp_path):
    """Generated rows should use conservative defaults and unique identifiers."""
    from cellflow.meta.catalog import records_from_h5_paths

    first = tmp_path / "a" / "position_analysis.h5"
    second = tmp_path / "b" / "position_analysis.h5"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()

    records = records_from_h5_paths(
        [first, second],
        defaults={"date": "day2", "condition": "treated", "labels": "manual"},
    )

    assert [record["date"] for record in records] == ["day2", "day2"]
    assert [record["condition"] for record in records] == ["treated", "treated"]
    assert [record["labels"] for record in records] == ["manual", "manual"]
    assert records[0]["id"] == "a"
    assert records[1]["id"] == "b"
    assert all(record["analysis_status"] == "ready" for record in records)


def test_merge_catalog_records_skips_duplicate_resolved_paths(tmp_path):
    """Merging should avoid duplicate H5 sources by resolved artifact path."""
    from cellflow.meta.catalog import merge_catalog_records

    source = tmp_path / "position_analysis.h5"
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
