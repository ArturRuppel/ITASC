from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "core" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ultrack" / "src"))

from cellflow.ultrack.config import TrackingConfig
from cellflow.ultrack.linking import compute_weighted_links
from cellflow.ultrack.stages.tracking import run_linking


class StubNode:
    def __init__(self, node_id: int, centroid: tuple[float, float], ious: dict[int, float]):
        self.id = node_id
        self.centroid = np.asarray(centroid, dtype=np.float32)
        self._ious = dict(ious)

    def IoU(self, other: "StubNode") -> float:
        return float(self._ious.get(other.id, 0.0))


def test_compute_weighted_links_prefers_iou_when_weight_is_one():
    source_nodes = [
        StubNode(1, (0.0, 0.0), {10: 0.2}),
        StubNode(2, (9.0, 0.0), {10: 0.8}),
    ]
    target_nodes = [StubNode(10, (1.0, 0.0), {})]

    links = compute_weighted_links(
        source_nodes,
        target_nodes,
        max_distance=20.0,
        max_neighbors=1,
        iou_weight=1.0,
        min_link_iou=0.0,
    )

    assert len(links) == 1
    assert links[0].source_id == 2
    assert links[0].target_id == 10
    assert links[0].iou == 0.8
    assert links[0].weight == 0.8


def test_compute_weighted_links_prefers_distance_when_weight_is_zero():
    source_nodes = [
        StubNode(1, (0.0, 0.0), {10: 0.2}),
        StubNode(2, (9.0, 0.0), {10: 0.8}),
    ]
    target_nodes = [StubNode(10, (1.0, 0.0), {})]

    links = compute_weighted_links(
        source_nodes,
        target_nodes,
        max_distance=20.0,
        max_neighbors=1,
        iou_weight=0.0,
        min_link_iou=0.0,
    )

    assert len(links) == 1
    assert links[0].source_id == 1
    assert links[0].target_id == 10
    assert links[0].distance == 1.0


def test_compute_weighted_links_respects_min_iou_and_neighbor_limit():
    source_nodes = [
        StubNode(1, (0.0, 0.0), {10: 0.4}),
        StubNode(2, (2.0, 0.0), {10: 0.6}),
        StubNode(3, (4.0, 0.0), {10: 0.05}),
    ]
    target_nodes = [StubNode(10, (1.0, 0.0), {})]

    links = compute_weighted_links(
        source_nodes,
        target_nodes,
        max_distance=10.0,
        max_neighbors=2,
        iou_weight=1.0,
        min_link_iou=0.1,
    )

    assert [link.source_id for link in links] == [2, 1]
    assert all(link.iou >= 0.1 for link in links)
    assert len(links) == 2


def test_run_linking_dispatches_to_iou_mode(tmp_path, monkeypatch):
    seen = {}

    def fake_run_iou_linking(working_dir, cfg, overwrite=True):
        seen["working_dir"] = working_dir
        seen["cfg"] = cfg
        seen["overwrite"] = overwrite
        yield (0, 1, "custom iou linker")

    monkeypatch.setattr("cellflow.ultrack.linking.run_iou_linking", fake_run_iou_linking)

    cfg = TrackingConfig(linking_mode="iou")
    progress = list(run_linking(tmp_path, cfg, overwrite=False))

    assert seen["working_dir"] == tmp_path
    assert seen["cfg"] == cfg
    assert seen["overwrite"] is False
    assert progress == [(0, 1, "custom iou linker")]
