"""Tests for boost_validated_edges in seed_prior.py."""
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


def _make_node_row(node_id: int, t: int, y: float, x: float, area: int, *, real: bool = False):
    from ultrack.core.database import NodeDB, VarAnnotation
    from ultrack.core.segmentation.node import Node

    h = w = max(1, int(area ** 0.5))
    y0, x0 = int(y - h // 2), int(x - w // 2)
    y1, x1 = y0 + h, x0 + w
    mask_crop = np.ones((1, h, w), dtype=bool)
    bbox_3d = np.array([0, y0, x0, 1, y1, x1], dtype=np.int64)
    node = Node.from_mask(time=t, mask=mask_crop, bbox=bbox_3d, node_id=node_id)
    blob = pickle.dumps(node)

    row = NodeDB(
        id=node_id,
        t=t,
        t_node_id=node_id,
        t_hier_id=0 if real else 1,
        z=0,
        y=y,
        x=x,
        area=area,
        pickle=blob,
    )
    if real:
        from ultrack.core.database import VarAnnotation
        row.node_annot = VarAnnotation.REAL
    return row


def test_boost_validated_edges_increases_weight_and_returns_correct_counts(tmp_path):
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, NodeDB

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.seed_prior import boost_validated_edges

    engine = _make_engine(tmp_path / "data.db")

    real_node = _make_node_row(1001, t=0, y=10.0, x=10.0, area=25, real=True)
    candidate_node = _make_node_row(1002, t=1, y=11.0, x=11.0, area=25, real=False)

    with Session(engine) as session:
        session.add_all([real_node, candidate_node])
        session.commit()

    initial_weight = 0.3
    with Session(engine) as session:
        session.add(LinkDB(source_id=1001, target_id=1002, weight=initial_weight))
        session.commit()

    cfg = TrackingConfig(
        seed_weight=0.5,
        seed_sigma_space=25.0,
        seed_tau_time=2.0,
        seed_max_dt=5,
        seed_sigma_area=0.5,
    )

    report = boost_validated_edges(tmp_path, cfg)

    assert report.seeds == 1
    assert report.boosted == 1

    with Session(engine) as session:
        link = session.query(LinkDB).first()
        assert link.weight > initial_weight


def test_boost_validated_edges_no_real_nodes_returns_zero(tmp_path):
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.seed_prior import boost_validated_edges

    engine = _make_engine(tmp_path / "data.db")

    n1 = _make_node_row(1, t=0, y=5.0, x=5.0, area=9, real=False)
    n2 = _make_node_row(2, t=1, y=6.0, x=6.0, area=9, real=False)

    with Session(engine) as session:
        session.add_all([n1, n2])
        session.commit()
        session.add(LinkDB(source_id=1, target_id=2, weight=0.5))
        session.commit()

    cfg = TrackingConfig()
    report = boost_validated_edges(tmp_path, cfg)

    assert report.seeds == 0
    assert report.boosted == 0

    with Session(engine) as session:
        link = session.query(LinkDB).first()
        assert link.weight == pytest.approx(0.5)


def test_boost_validated_edges_skips_link_outside_seed_max_dt(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.seed_prior import boost_validated_edges

    engine = _make_engine(tmp_path / "data.db")

    real_node = _make_node_row(1, t=0, y=10.0, x=10.0, area=25, real=True)
    far_node = _make_node_row(2, t=10, y=10.0, x=10.0, area=25, real=False)

    with Session(engine) as session:
        session.add_all([real_node, far_node])
        session.commit()
        session.add(LinkDB(source_id=1, target_id=2, weight=0.4))
        session.commit()

    cfg = TrackingConfig(seed_max_dt=3)  # dt=10 exceeds max
    report = boost_validated_edges(tmp_path, cfg)

    assert report.seeds == 1
    assert report.boosted == 0

    with Session(engine) as session:
        link = session.query(LinkDB).first()
        assert link.weight == pytest.approx(0.4)
