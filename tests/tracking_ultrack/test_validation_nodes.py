from __future__ import annotations

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.validation_nodes import inject_validated_nodes


def test_inject_validated_nodes_creates_real_node_with_reserved_hierarchy(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        candidate = _make_node_row(1_000_001, 0, 30, 30, 40, 40)
        candidate.t_node_id = 1
        session.add(candidate)
        session.commit()

    tracked = np.zeros((1, 64, 64), dtype=np.uint32)
    tracked[0, 5:15, 7:17] = 42

    report = inject_validated_nodes(tmp_path, {42: {0}}, tracked, TrackingConfig())

    assert report.inserted == 1
    assert report.skipped_missing == 0
    with Session(engine) as session:
        rows = session.query(NodeDB).order_by(NodeDB.id).all()

    injected = [row for row in rows if row.t_hier_id == 0][0]
    assert injected.t == 0
    assert injected.t_node_id == 2
    assert injected.node_annot == VarAnnotation.REAL
    assert injected.node_prob == 1.0
    assert injected.area == 100


def test_inject_validated_nodes_marks_intersecting_candidates_fake_and_adds_overlap(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, OverlapDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        overlap_candidate = _make_node_row(1_000_001, 0, 0, 0, 10, 10)
        overlap_candidate.t_node_id = 1
        unrelated_candidate = _make_node_row(1_000_002, 0, 30, 30, 40, 40)
        unrelated_candidate.t_node_id = 2
        session.add_all([overlap_candidate, unrelated_candidate])
        session.commit()

    tracked = np.zeros((1, 64, 64), dtype=np.uint32)
    tracked[0, 5:15, 5:15] = 7

    report = inject_validated_nodes(tmp_path, {7: {0}}, tracked, TrackingConfig())

    assert report.inserted == 1
    with Session(engine) as session:
        rows = {row.id: row for row in session.query(NodeDB).all()}
        injected_id = [row.id for row in rows.values() if row.t_hier_id == 0][0]
        assert rows[1_000_001].node_annot == VarAnnotation.FAKE
        assert rows[1_000_002].node_annot == VarAnnotation.UNKNOWN
        assert rows[injected_id].node_annot == VarAnnotation.REAL
        overlap_pairs = {
            (row.node_id, row.ancestor_id)
            for row in session.query(OverlapDB).all()
        }

    assert (max(injected_id, 1_000_001), min(injected_id, 1_000_001)) in overlap_pairs


def test_inject_validated_nodes_reports_missing_cell_frames(tmp_path):
    from tests.tracking_ultrack.test_reseed import _make_engine

    _make_engine(tmp_path / "data.db")
    tracked = np.zeros((1, 32, 32), dtype=np.uint32)

    report = inject_validated_nodes(tmp_path, {9: {0}}, tracked, TrackingConfig())

    assert report.inserted == 0
    assert report.skipped_missing == 1
    assert report.skipped == [(9, 0)]
