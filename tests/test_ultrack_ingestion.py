from __future__ import annotations

import json
import sys
import types

import numpy as np

from cellflow.ultrack.ingestion import (
    labels_batch_to_foreground_contours,
    load_hypothesis_labelmaps,
    write_hypothesis_labelmaps,
)
from cellflow.ultrack.config import TrackingConfig
from cellflow.ultrack.stages.tracking import run_nucleus_ultrack


def test_write_hypothesis_labelmaps_round_trip(tmp_path):
    labelmaps = [
        np.array(
            [
                [0, 1, 1],
                [0, 1, 2],
                [3, 3, 2],
            ],
            dtype=np.uint32,
        ),
        np.array(
            [
                [0, 4, 4],
                [5, 5, 4],
                [5, 0, 0],
            ],
            dtype=np.uint32,
        ),
    ]

    manifest_path = write_hypothesis_labelmaps(
        tmp_path,
        labelmaps,
        stage_name="nucleus_ultrack",
        source="cellpose-sweep",
    )

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["stage"] == "nucleus_ultrack"
    assert manifest["source"] == "cellpose-sweep"
    assert manifest["labelmap_count"] == 2
    assert [entry["path"] for entry in manifest["labelmaps"]] == [
        "labelmaps/labelmap_000.tif",
        "labelmaps/labelmap_001.tif",
    ]

    loaded_labelmaps, loaded_manifest = load_hypothesis_labelmaps(tmp_path)
    assert loaded_manifest["stage"] == "nucleus_ultrack"
    assert len(loaded_labelmaps) == 2
    assert np.array_equal(loaded_labelmaps[0], labelmaps[0])
    assert np.array_equal(loaded_labelmaps[1], labelmaps[1])


def test_labels_batch_to_foreground_contours_averages_maps():
    labels_a = np.array(
        [
            [0, 1, 1],
            [0, 1, 2],
            [3, 3, 2],
        ],
        dtype=np.uint32,
    )
    labels_b = np.array(
        [
            [0, 4, 4],
            [5, 5, 4],
            [5, 0, 0],
        ],
        dtype=np.uint32,
    )

    foreground, contours = labels_batch_to_foreground_contours([labels_a, labels_b], smooth_sigma=0.0)

    assert foreground.shape == labels_a.shape
    assert contours.shape == labels_a.shape
    assert foreground.dtype == np.float32
    assert contours.dtype == np.float32
    assert foreground.min() >= 0.0
    assert foreground.max() <= 1.0
    assert contours.min() >= 0.0
    assert contours.max() <= 1.0
    assert foreground[0, 0] == 0.0
    assert foreground[0, 1] == 1.0


def test_labels_batch_to_foreground_contours_rejects_empty_batch():
    try:
        labels_batch_to_foreground_contours([])
    except ValueError as exc:
        assert "at least one array" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty batch")


def test_run_nucleus_ultrack_uses_persisted_labelmaps(tmp_path, monkeypatch):
    labelmaps = [
        np.array([[0, 1], [2, 2]], dtype=np.uint32),
        np.array([[0, 3], [3, 4]], dtype=np.uint32),
    ]
    write_hypothesis_labelmaps(
        tmp_path,
        labelmaps,
        stage_name="nucleus_ultrack",
        source="test-source",
    )

    captured = {}

    def fake_clear_all_data(_database_path):
        captured["cleared"] = True

    def fake_segment(foreground, contours, _cfg, overwrite=False):
        captured["foreground"] = np.asarray(foreground)
        captured["contours"] = np.asarray(contours)
        captured["overwrite"] = overwrite

    fake_ultrack = types.ModuleType("ultrack")
    fake_ultrack.__path__ = []
    fake_core = types.ModuleType("ultrack.core")
    fake_core.__path__ = []
    fake_seg_pkg = types.ModuleType("ultrack.core.segmentation")
    fake_seg_pkg.__path__ = []
    fake_db = types.ModuleType("ultrack.core.database")
    fake_db.clear_all_data = fake_clear_all_data
    fake_seg = types.ModuleType("ultrack.core.segmentation.processing")
    fake_seg.segment = fake_segment

    monkeypatch.setitem(sys.modules, "ultrack", fake_ultrack)
    monkeypatch.setitem(sys.modules, "ultrack.core", fake_core)
    monkeypatch.setitem(sys.modules, "ultrack.core.database", fake_db)
    monkeypatch.setitem(sys.modules, "ultrack.core.segmentation", fake_seg_pkg)
    monkeypatch.setitem(sys.modules, "ultrack.core.segmentation.processing", fake_seg)
    fake_config = types.ModuleType("ultrack.config")
    fake_config.__package__ = "ultrack"
    monkeypatch.setitem(sys.modules, "ultrack.config", fake_config)

    class FakeMainConfig:
        def __init__(self, **kwargs):
            self.data_config = types.SimpleNamespace(database_path=tmp_path / "data.db")
            self.kwargs = kwargs

    fake_config.MainConfig = FakeMainConfig

    cfg = TrackingConfig(n_workers=1)
    progress = list(run_nucleus_ultrack(tmp_path, cfg, overwrite=True))

    assert captured["cleared"] is True
    assert captured["overwrite"] is True
    assert captured["foreground"].shape == labelmaps[0].shape
    assert captured["contours"].shape == labelmaps[0].shape
    assert progress[-1][2] == "Segmentation done."
    assert (tmp_path / "hypotheses_manifest.json").exists()
