"""Aggregate capstone: readiness partition + engine drive."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cellflow.napari.aggregate_widget import partition_ready


def _record(pos_dir: Path, *, ready: bool) -> dict:
    """A main_widget-shaped catalog record (see ``_catalog_record_for_position``).

    Identity lives in the ``columns`` bag under the seed level names
    (``condition`` / ``experiment_id`` / ``position_id``), which is what
    ``save_catalog`` reads via its ``_BAG_TO_CSV`` mapping. ``ready`` controls
    whether the per-position ``contacts.h5`` exists on disk.
    """
    h5 = pos_dir / "4_contact_analysis" / "contact_analysis.h5"
    if ready:
        h5.parent.mkdir(parents=True, exist_ok=True)
        h5.write_bytes(b"")
    return {
        "position_path": pos_dir,
        "contact_analysis_path": h5,
        "cell_tracked_labels_path": pos_dir / "cell_labels.tif",
        "nucleus_tracked_labels_path": pos_dir / "nucleus_labels.tif",
        "columns": {
            "condition": "ctrl",
            "experiment_id": "exp1",
            "position_id": pos_dir.name,
        },
    }


def test_partition_ready_splits_by_h5_presence(tmp_path):
    a = _record(tmp_path / "posA", ready=True)
    b = _record(tmp_path / "posB", ready=False)
    ready, not_ready = partition_ready([a, b])
    assert ready == [a]
    assert not_ready == [b]


def test_partition_ready_empty():
    assert partition_ready([]) == ([], [])


from cellflow.contact_analysis import load_catalog
from cellflow.napari import aggregate_widget as aw


def test_pool_positions_authors_ready_subset_and_runs(tmp_path, monkeypatch):
    ready = [
        _record(tmp_path / "study" / "posA", ready=True),
        _record(tmp_path / "study" / "posB", ready=True),
    ]
    seen = {}

    def _fake_run(config_path):
        seen["config_path"] = Path(config_path)
        return {"object_table": Path(config_path).parent / "object_table.csv"}

    monkeypatch.setattr(aw, "run", _fake_run)

    result = aw.pool_positions(ready, skipped_names=["posC"])

    # The engine was driven with the authored config.
    config_path = seen["config_path"]
    assert config_path.name == "config.toml"
    # The authored catalog contains exactly the two ready positions.
    catalog = load_catalog(config_path.parent / "catalog.csv")
    names = sorted(Path(rec["position_path"]).name for rec in catalog)
    assert names == ["posA", "posB"]
    # The result carries the skipped names and table map for the UI.
    assert result["skipped"] == ["posC"]
    assert "object_table" in result["tables"]
    # Tables land in the ready positions' common ancestor.
    assert result["project_dir"] == (tmp_path / "study")


from napari.qt import get_qapp


def test_widget_readout_reports_ready_split_and_names_not_ready(tmp_path):
    get_qapp()
    from cellflow.napari.aggregate_widget import AggregateWidget

    w = AggregateWidget()
    w.set_records([
        _record(tmp_path / "posA", ready=True),
        _record(tmp_path / "posB", ready=False),
    ])
    assert "1 of 2" in w.readout.text()
    assert "posB" in w.readout.text()
    assert w.run_btn.isEnabled() is True
    assert w.section_status() == "in_progress"


def test_widget_run_button_disabled_when_nothing_ready(tmp_path):
    get_qapp()
    from cellflow.napari.aggregate_widget import AggregateWidget

    w = AggregateWidget()
    w.set_records([_record(tmp_path / "posA", ready=False)])
    assert w.run_btn.isEnabled() is False
    assert w.section_status() == "not_started"
