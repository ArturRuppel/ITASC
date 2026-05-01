import numpy as np
from sqlalchemy.orm import Session

from tests.tracking_ultrack.test_anchor import _make_engine
from tests.tracking_ultrack.test_reseed import _insert_overlaps, _make_node_row


def test_diagnose_anchor_frame_candidates_reports_best_node_state(tmp_path):
    from ultrack.core.database import NodeDB, VarAnnotation

    from cellflow.tracking_ultrack.anchor_diagnostics import (
        diagnose_anchor_frame_candidates,
    )

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        session.add_all(
            [
                _make_node_row(101, 1, 0, 0, 10, 10),
                _make_node_row(102, 1, 20, 20, 30, 30),
                _make_node_row(103, 1, 40, 40, 50, 50),
            ]
        )
        session.query(NodeDB).where(NodeDB.id == 101).update(
            {NodeDB.selected: True, NodeDB.node_annot: VarAnnotation.REAL},
            synchronize_session=False,
        )
        session.query(NodeDB).where(NodeDB.id == 102).update(
            {NodeDB.node_annot: VarAnnotation.FAKE},
            synchronize_session=False,
        )
        session.commit()

    labels = np.zeros((3, 64, 64), dtype=np.uint32)
    labels[1, 0:10, 0:10] = 7
    labels[1, 20:30, 20:30] = 8

    report = diagnose_anchor_frame_candidates(tmp_path, labels, frame_index=1)

    assert report.frame_index == 1
    assert [match.gt_label for match in report.matches] == [7, 8]
    assert report.matches[0].best_node_id == 101
    assert report.matches[0].best_iou == 1.0
    assert report.matches[0].selected is True
    assert report.matches[0].node_annot == VarAnnotation.REAL
    assert report.matches[1].best_node_id == 102
    assert report.matches[1].best_iou == 1.0
    assert report.matches[1].selected is False
    assert report.matches[1].node_annot == VarAnnotation.FAKE


def test_diagnose_anchor_frame_candidates_reports_links_and_selected_overlaps(tmp_path):
    from ultrack.core.database import LinkDB, NodeDB

    from cellflow.tracking_ultrack.anchor_diagnostics import (
        diagnose_anchor_frame_candidates,
    )

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        session.add_all(
            [
                _make_node_row(100, 0, 0, 0, 10, 10),
                _make_node_row(101, 1, 0, 0, 10, 10),
                _make_node_row(102, 1, 0, 0, 10, 8),
                _make_node_row(200, 2, 0, 0, 10, 10),
            ]
        )
        session.add_all(
            [
                LinkDB(source_id=100, target_id=101, weight=0.7),
                LinkDB(source_id=101, target_id=200, weight=0.9),
            ]
        )
        session.query(NodeDB).where(NodeDB.id == 102).update(
            {NodeDB.selected: True},
            synchronize_session=False,
        )
        session.commit()
    _insert_overlaps(engine, [(101, 102)])

    labels = np.zeros((3, 64, 64), dtype=np.uint32)
    labels[1, 0:10, 0:10] = 7

    report = diagnose_anchor_frame_candidates(tmp_path, labels, frame_index=1)

    match = report.matches[0]
    assert match.best_node_id == 101
    assert match.selected is False
    assert match.incoming_link_count == 1
    assert match.best_incoming_weight == 0.7
    assert match.outgoing_link_count == 1
    assert match.best_outgoing_weight == 0.9
    assert len(match.selected_overlaps) == 1
    assert match.selected_overlaps[0].node_id == 102
    assert match.selected_overlaps[0].iou == 0.8
    assert match.selected_overlaps[0].area == 80
