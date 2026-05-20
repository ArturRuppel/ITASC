"""Tests for ensure_anchor_incident_links in corrections.py."""
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
        row.node_annot = VarAnnotation.REAL
    return row


def test_inserts_outgoing_and_incoming_edges_within_max_distance(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import ensure_anchor_incident_links

    engine = _make_engine(tmp_path / "data.db")

    anchor = _make_node_row(100, t=1, y=20.0, x=20.0, area=25, real=True)
    pred_in_range = _make_node_row(200, t=0, y=22.0, x=22.0, area=25)
    pred_far = _make_node_row(201, t=0, y=80.0, x=80.0, area=25)
    succ_in_range = _make_node_row(300, t=2, y=18.0, x=18.0, area=25)
    succ_far = _make_node_row(301, t=2, y=80.0, x=80.0, area=25)

    with Session(engine) as session:
        session.add_all([anchor, pred_in_range, pred_far, succ_in_range, succ_far])
        session.commit()

    cfg = TrackingConfig(max_distance=15.0, linking_mode="default", distance_weight=0.0)
    report = ensure_anchor_incident_links(tmp_path, cfg)

    assert report.anchors_processed == 1
    assert report.inserted == 2

    with Session(engine) as session:
        links = session.query(LinkDB.source_id, LinkDB.target_id).all()
        link_pairs = {(int(s), int(t)) for s, t in links}

    assert (200, 100) in link_pairs
    assert (100, 300) in link_pairs
    assert (201, 100) not in link_pairs
    assert (100, 301) not in link_pairs


def test_skips_existing_link_pair(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import ensure_anchor_incident_links

    engine = _make_engine(tmp_path / "data.db")

    anchor = _make_node_row(1, t=0, y=10.0, x=10.0, area=25, real=True)
    neighbor = _make_node_row(2, t=1, y=11.0, x=11.0, area=25)

    with Session(engine) as session:
        session.add_all([anchor, neighbor])
        session.commit()
        session.add(LinkDB(source_id=1, target_id=2, weight=0.99))
        session.commit()

    cfg = TrackingConfig(max_distance=15.0, linking_mode="default", distance_weight=0.0)
    report = ensure_anchor_incident_links(tmp_path, cfg)

    assert report.inserted == 0
    with Session(engine) as session:
        link = session.query(LinkDB).filter_by(source_id=1, target_id=2).one()
        assert link.weight == pytest.approx(0.99)


def test_does_not_insert_duplicate_pair_when_adjacent_anchors_share_edge(tmp_path):
    from collections import Counter

    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import ensure_anchor_incident_links

    engine = _make_engine(tmp_path / "data.db")

    first_anchor = _make_node_row(1, t=0, y=10.0, x=10.0, area=25, real=True)
    second_anchor = _make_node_row(2, t=1, y=11.0, x=11.0, area=25, real=True)

    with Session(engine) as session:
        session.add_all([first_anchor, second_anchor])
        session.commit()

    cfg = TrackingConfig(max_distance=15.0, linking_mode="default", distance_weight=0.0)
    report = ensure_anchor_incident_links(tmp_path, cfg)

    assert report.inserted == 1
    with Session(engine) as session:
        pairs = [
            (int(source_id), int(target_id))
            for source_id, target_id in session.query(
                LinkDB.source_id, LinkDB.target_id
            ).all()
        ]

    assert pairs == [(1, 2)]
    assert Counter(pairs)[(1, 2)] == 1


def test_removes_existing_duplicate_link_pairs_before_solve(tmp_path):
    from collections import Counter

    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, VarAnnotation

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import ensure_anchor_incident_links

    engine = _make_engine(tmp_path / "data.db")

    first_anchor = _make_node_row(1, t=0, y=10.0, x=10.0, area=25, real=True)
    second_anchor = _make_node_row(2, t=1, y=11.0, x=11.0, area=25, real=True)

    with Session(engine) as session:
        session.add_all([first_anchor, second_anchor])
        session.commit()
        session.add_all(
            [
                LinkDB(source_id=1, target_id=2, weight=0.5),
                LinkDB(
                    source_id=1,
                    target_id=2,
                    weight=0.75,
                    annotation=VarAnnotation.REAL,
                ),
            ]
        )
        session.commit()

    cfg = TrackingConfig(max_distance=15.0, linking_mode="default", distance_weight=0.0)
    report = ensure_anchor_incident_links(tmp_path, cfg)

    assert report.inserted == 0
    with Session(engine) as session:
        links = session.query(LinkDB).all()
        pairs = [(int(link.source_id), int(link.target_id)) for link in links]

    assert Counter(pairs)[(1, 2)] == 1
    assert links[0].annotation == VarAnnotation.REAL
    assert links[0].weight == pytest.approx(0.75)


def test_no_anchors_no_inserts(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import ensure_anchor_incident_links

    engine = _make_engine(tmp_path / "data.db")

    a = _make_node_row(1, t=0, y=5.0, x=5.0, area=9)
    b = _make_node_row(2, t=1, y=6.0, x=6.0, area=9)
    with Session(engine) as session:
        session.add_all([a, b])
        session.commit()

    cfg = TrackingConfig(max_distance=15.0, linking_mode="default")
    report = ensure_anchor_incident_links(tmp_path, cfg)

    assert report.inserted == 0
    assert report.anchors_processed == 0
    with Session(engine) as session:
        assert session.query(LinkDB).count() == 0
