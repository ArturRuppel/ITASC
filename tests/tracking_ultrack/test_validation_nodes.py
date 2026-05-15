from __future__ import annotations

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack._node_geometry import node_bbox_and_mask, node_pickle_ndim
from cellflow.tracking_ultrack.validation_nodes import inject_validated_nodes


def test_inject_validated_nodes_replaces_best_iou_candidate_and_preserves_hierarchy(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        candidate = _make_node_row(1_000_001, 0, 4, 6, 16, 18)
        candidate.t_node_id = 1
        candidate.t_hier_id = 17
        candidate.height = 0.25
        session.add(candidate)
        session.commit()

    tracked = np.zeros((1, 64, 64), dtype=np.uint32)
    tracked[0, 5:15, 7:17] = 42

    report = inject_validated_nodes(tmp_path, {42: {0}}, tracked, TrackingConfig())

    assert report.inserted == 1
    assert report.skipped_missing == 0
    with Session(engine) as session:
        rows = session.query(NodeDB).order_by(NodeDB.id).all()

    assert len(rows) == 1
    replaced = rows[0]
    assert replaced.id == 1_000_001
    assert replaced.t == 0
    assert replaced.t_node_id == 1
    assert replaced.t_hier_id == 17
    assert replaced.height == 0.25
    assert replaced.node_annot == VarAnnotation.REAL
    assert replaced.node_prob == 1.0
    assert replaced.area == 100
    assert node_pickle_ndim(replaced.pickle) == 3
    bbox, mask = node_bbox_and_mask(replaced.id, replaced.pickle)
    assert bbox == (5, 7, 15, 17)
    assert mask.shape == (10, 10)
    assert mask.all()


def test_inject_validated_nodes_falls_back_to_reserved_node_when_frame_has_no_candidates(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine

    engine = _make_engine(tmp_path / "data.db")

    tracked = np.zeros((1, 64, 64), dtype=np.uint32)
    tracked[0, 5:15, 7:17] = 42

    report = inject_validated_nodes(tmp_path, {42: {0}}, tracked, TrackingConfig())

    assert report.inserted == 1
    assert report.skipped_missing == 0
    with Session(engine) as session:
        rows = session.query(NodeDB).order_by(NodeDB.id).all()

    injected = rows[0]
    assert injected.t == 0
    assert injected.t_hier_id == 0
    assert injected.node_annot == VarAnnotation.REAL
    assert injected.node_prob == 1.0
    assert injected.area == 100


def test_inject_validated_nodes_marks_intersecting_unmatched_candidates_fake_and_adds_overlap(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, OverlapDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        matched_candidate = _make_node_row(1_000_001, 0, 5, 5, 15, 15)
        matched_candidate.t_node_id = 1
        overlap_candidate = _make_node_row(1_000_002, 0, 0, 0, 10, 10)
        overlap_candidate.t_node_id = 2
        unrelated_candidate = _make_node_row(1_000_003, 0, 30, 30, 40, 40)
        unrelated_candidate.t_node_id = 3
        session.add_all([matched_candidate, overlap_candidate, unrelated_candidate])
        session.commit()

    tracked = np.zeros((1, 64, 64), dtype=np.uint32)
    tracked[0, 5:15, 5:15] = 7

    report = inject_validated_nodes(tmp_path, {7: {0}}, tracked, TrackingConfig())

    assert report.inserted == 1
    with Session(engine) as session:
        rows = {row.id: row for row in session.query(NodeDB).all()}
        assert rows[1_000_001].node_annot == VarAnnotation.REAL
        assert rows[1_000_002].node_annot == VarAnnotation.FAKE
        assert rows[1_000_003].node_annot == VarAnnotation.UNKNOWN
        overlap_pairs = {
            (row.node_id, row.ancestor_id)
            for row in session.query(OverlapDB).all()
        }

    assert (1_000_002, 1_000_001) in overlap_pairs


def test_inject_validated_nodes_assigns_candidates_one_to_one_by_iou(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        first = _make_node_row(1_000_001, 0, 0, 0, 10, 10)
        second = _make_node_row(1_000_002, 0, 20, 20, 30, 30)
        session.add_all([first, second])
        session.commit()

    tracked = np.zeros((1, 40, 40), dtype=np.uint32)
    tracked[0, 0:10, 0:10] = 7
    tracked[0, 20:30, 20:30] = 8

    report = inject_validated_nodes(tmp_path, {7: {0}, 8: {0}}, tracked, TrackingConfig())

    assert report.inserted == 2
    with Session(engine) as session:
        rows = {row.id: row for row in session.query(NodeDB).all()}

    assert rows[1_000_001].node_annot == VarAnnotation.REAL
    assert rows[1_000_002].node_annot == VarAnnotation.REAL
    first_bbox, _first_mask = node_bbox_and_mask(
        rows[1_000_001].id, rows[1_000_001].pickle
    )
    second_bbox, _second_mask = node_bbox_and_mask(
        rows[1_000_002].id, rows[1_000_002].pickle
    )
    assert first_bbox == (0, 0, 10, 10)
    assert second_bbox == (20, 20, 30, 30)


def test_inject_validated_nodes_replaces_stale_overlaps_for_replaced_candidate(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import OverlapDB
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        matched_candidate = _make_node_row(1_000_001, 0, 5, 5, 15, 15)
        stale_overlap_candidate = _make_node_row(1_000_002, 0, 40, 40, 50, 50)
        new_overlap_candidate = _make_node_row(1_000_003, 0, 0, 0, 10, 10)
        session.add_all([matched_candidate, stale_overlap_candidate, new_overlap_candidate])
        session.add(OverlapDB(node_id=1_000_002, ancestor_id=1_000_001))
        session.commit()

    tracked = np.zeros((1, 64, 64), dtype=np.uint32)
    tracked[0, 5:15, 5:15] = 7

    inject_validated_nodes(tmp_path, {7: {0}}, tracked, TrackingConfig())

    with Session(engine) as session:
        overlap_pairs = {
            (row.node_id, row.ancestor_id)
            for row in session.query(OverlapDB).all()
        }

    assert (1_000_002, 1_000_001) not in overlap_pairs
    assert (1_000_003, 1_000_001) in overlap_pairs


def test_inject_validated_nodes_reports_missing_cell_frames(tmp_path):
    from tests.tracking_ultrack.test_reseed import _make_engine

    _make_engine(tmp_path / "data.db")
    tracked = np.zeros((1, 32, 32), dtype=np.uint32)

    report = inject_validated_nodes(tmp_path, {9: {0}}, tracked, TrackingConfig())

    assert report.inserted == 0
    assert report.skipped_missing == 1
    assert report.skipped == [(9, 0)]
