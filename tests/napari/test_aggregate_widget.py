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
