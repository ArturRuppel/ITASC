"""Unit tests for the shape-mode linker."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

ultrack = pytest.importorskip("ultrack")


def _make_engine(db_path):
    import sqlalchemy as sqla
    from ultrack.core.database import Base

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine


def _make_node_row(node_id: int, t: int, y: float, x: float, h: int, w: int):
    """Create a NodeDB row with a rectangular mask of size h×w centred near (y, x)."""
    from ultrack.core.database import NodeDB
    from ultrack.core.segmentation.node import Node

    y0, x0 = int(y) - h // 2, int(x) - w // 2
    mask = np.ones((1, h, w), dtype=bool)
    bbox_3d = np.array([0, y0, x0, 1, y0 + h, x0 + w], dtype=np.int64)
    node = Node.from_mask(time=t, mask=mask, bbox=bbox_3d, node_id=node_id)
    blob = pickle.dumps(node)
    return NodeDB(
        id=node_id,
        t=t,
        t_node_id=node_id,
        t_hier_id=1,
        z=0,
        y=float(node.centroid[-2]),
        x=float(node.centroid[-1]),
        area=h * w,
        pickle=blob,
    )


def test_shape_correct_candidate_wins_over_closer_wrong_shape(tmp_path):
    """With shape mode, the candidate with matching shape beats the closer but wrong-shape one."""
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.linking import run_linking

    engine = _make_engine(tmp_path / "data.db")

    # t=0: one source node — 5×5 square at (50, 50)
    src = _make_node_row(10, t=0, y=50, x=50, h=5, w=5)
    # t=1: two candidates
    # A — closer centroid (~4 px away) but elongated 10×2 shape → low IoU after alignment
    cand_a = _make_node_row(20, t=1, y=54, x=51, h=10, w=2)
    # B — further centroid (~8 px away) but same 5×5 shape → IoU ≈ 1.0 after alignment
    cand_b = _make_node_row(30, t=1, y=58, x=51, h=5, w=5)

    with Session(engine) as session:
        session.add_all([src, cand_a, cand_b])
        session.commit()

    cfg = TrackingConfig(
        linking_mode="shape",
        max_distance=20.0,
        max_neighbors=5,
        min_area_ratio=0.3,
        min_link_iou=0.05,
        area_weight=1.0,
        iou_weight=1.0,
        distance_weight=0.25,
    )
    list(run_linking(tmp_path, cfg))

    with Session(engine) as session:
        links = {int(lnk.target_id): float(lnk.weight) for lnk in session.query(LinkDB).all()}

    assert 20 in links, "Candidate A should have a link"
    assert 30 in links, "Candidate B should have a link"
    assert links[30] > links[20], (
        f"Shape-correct B (w={links[30]:.4f}) should score higher than wrong-shape A (w={links[20]:.4f})"
    )


def test_area_ratio_prefilter_removes_candidate(tmp_path):
    """Candidate with area_ratio < min_area_ratio produces no LinkDB row."""
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.linking import run_linking

    engine = _make_engine(tmp_path / "data.db")

    # t=0: 5×5 source (area=25)
    src = _make_node_row(10, t=0, y=50, x=50, h=5, w=5)
    # t=1: very large candidate (30×30 = area=900); ratio = 25/900 ≈ 0.028 << min_area_ratio=0.3
    cand_large = _make_node_row(20, t=1, y=52, x=52, h=30, w=30)

    with Session(engine) as session:
        session.add_all([src, cand_large])
        session.commit()

    cfg = TrackingConfig(
        linking_mode="shape",
        max_distance=20.0,
        max_neighbors=5,
        min_area_ratio=0.3,
        min_link_iou=0.05,
        area_weight=1.0,
        iou_weight=1.0,
        distance_weight=0.25,
    )
    list(run_linking(tmp_path, cfg))

    with Session(engine) as session:
        count = session.query(LinkDB).count()

    assert count == 0, f"Area-ratio prefilter should have blocked the edge, got {count} links"
