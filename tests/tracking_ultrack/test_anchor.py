from pathlib import Path

import numpy as np
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from cellflow.tracking_ultrack.anchor import (
    annotate_anchor_frame,
    suppress_anchor_adjacent_fragments,
)
from tests.tracking_ultrack.test_reseed import _make_node_row


def _make_engine(db_path: Path):
    from ultrack.core.database import Base

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine


def test_annotate_anchor_frame_pins_matching_nodes_and_fakes_other_anchor_nodes(tmp_path):
    from ultrack.core.database import NodeDB, VarAnnotation

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        session.add_all(
            [
                _make_node_row(101, 1, 0, 0, 10, 10),
                _make_node_row(102, 1, 20, 20, 30, 30),
                _make_node_row(103, 1, 40, 40, 50, 50),
                _make_node_row(201, 0, 0, 0, 10, 10),
            ]
        )
        session.commit()

    labels = np.zeros((3, 64, 64), dtype=np.uint32)
    labels[1, 0:10, 0:10] = 7
    labels[1, 20:30, 20:30] = 8

    report = annotate_anchor_frame(tmp_path, labels, frame_index=1, min_iou=1.0)

    assert report.frame_index == 1
    assert report.n_gt_labels == 2
    assert report.n_matched == 2
    assert report.n_unmatched == 0
    assert report.matched_node_ids == [101, 102]

    with Session(engine) as session:
        rows = {
            row.id: row.node_annot
            for row in session.query(NodeDB).order_by(NodeDB.id).all()
        }

    assert rows[101] == VarAnnotation.REAL
    assert rows[102] == VarAnnotation.REAL
    assert rows[103] == VarAnnotation.FAKE
    assert rows[201] == VarAnnotation.UNKNOWN


def test_annotate_anchor_frame_does_not_match_same_shape_at_wrong_location(tmp_path):
    from ultrack.core.database import NodeDB, VarAnnotation

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        session.add(_make_node_row(101, 1, 20, 20, 30, 30))
        session.commit()

    labels = np.zeros((3, 64, 64), dtype=np.uint32)
    labels[1, 0:10, 0:10] = 7

    report = annotate_anchor_frame(tmp_path, labels, frame_index=1, min_iou=0.5)

    assert report.n_matched == 0
    assert report.unmatched_labels == [7]
    with Session(engine) as session:
        row = session.query(NodeDB).one()
    assert row.node_annot == VarAnnotation.FAKE


def test_suppress_anchor_adjacent_fragments_fakes_subobject_alternatives(tmp_path):
    from ultrack.core.database import NodeDB, VarAnnotation

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        session.add_all(
            [
                _make_node_row(101, 2, 0, 0, 20, 20),  # best whole-object candidate
                _make_node_row(102, 2, 0, 0, 10, 20),  # top fragment
                _make_node_row(103, 2, 10, 0, 20, 20),  # bottom fragment
                _make_node_row(104, 2, 30, 30, 40, 40),  # unrelated object
                _make_node_row(201, 1, 0, 0, 20, 20),  # anchor frame is untouched here
            ]
        )
        session.commit()

    labels = np.zeros((3, 64, 64), dtype=np.uint32)
    labels[1, 0:20, 0:20] = 7

    report = suppress_anchor_adjacent_fragments(
        tmp_path,
        labels,
        frame_index=1,
        neighbor_offsets=(1,),
        min_best_iou=0.75,
        fragment_max_iou_fraction=0.75,
        min_fragment_containment=0.95,
    )

    assert report.frame_index == 1
    assert report.suppressed_node_ids == [102, 103]
    assert report.by_frame == {2: 2}

    with Session(engine) as session:
        rows = {
            row.id: row.node_annot
            for row in session.query(NodeDB).order_by(NodeDB.id).all()
        }

    assert rows[101] == VarAnnotation.UNKNOWN
    assert rows[102] == VarAnnotation.FAKE
    assert rows[103] == VarAnnotation.FAKE
    assert rows[104] == VarAnnotation.UNKNOWN
    assert rows[201] == VarAnnotation.UNKNOWN
