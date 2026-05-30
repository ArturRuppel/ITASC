from pathlib import Path

import numpy as np
import pytest


def test_annotation_name_handles_ultrack_var_annotation_enum_shape():
    from enum import Enum

    from cellflow.tracking_ultrack.db_query import annotation_name

    class VarAnnotationLike(Enum):
        UNKNOWN = 0
        REAL = 1
        FAKE = 2

    assert annotation_name(VarAnnotationLike.REAL) == "REAL"
    assert annotation_name(VarAnnotationLike.FAKE) == "FAKE"
    assert annotation_name(VarAnnotationLike.UNKNOWN) == "UNKNOWN"


def _add_ultrack_node(
    session,
    *,
    node_id: int,
    frame: int = 0,
    parent_id: int | None,
    height: float,
    bbox: tuple[int, int, int, int],
    node_prob: float | None = None,
) -> None:
    import pickle

    from ultrack.core.database import NodeDB
    from ultrack.core.segmentation.node import Node

    y0, x0, y1, x1 = bbox
    mask = np.ones((1, y1 - y0, x1 - x0), dtype=bool)
    node_obj = Node.from_mask(
        time=frame,
        mask=mask,
        bbox=np.array([0, y0, x0, 1, y1, x1], dtype=np.int64),
        node_id=node_id,
    )
    session.add(
        NodeDB(
            id=node_id,
            t=frame,
            t_node_id=node_id,
            t_hier_id=1,
            z=0,
            y=(y0 + y1) / 2,
            x=(x0 + x1) / 2,
            area=int(mask.sum()),
            height=float(height),
            hier_parent_id=parent_id,
            node_prob=node_prob,
            pickle=pickle.dumps(node_obj),
        )
    )


def _make_hierarchy_db(db_path: Path) -> None:
    pytest.importorskip("ultrack")
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import Base, LinkDB
    from ultrack.utils.constants import NO_PARENT

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _add_ultrack_node(
            session,
            node_id=100,
            parent_id=NO_PARENT,
            height=10.0,
            bbox=(0, 0, 2, 4),
        )
        _add_ultrack_node(
            session,
            node_id=101,
            parent_id=100,
            height=10.0,
            bbox=(0, 0, 2, 2),
        )
        _add_ultrack_node(
            session,
            node_id=102,
            parent_id=100,
            height=10.0,
            bbox=(0, 2, 2, 4),
        )
        _add_ultrack_node(
            session,
            node_id=201,
            parent_id=101,
            height=1.0,
            bbox=(0, 0, 1, 1),
        )
        _add_ultrack_node(
            session,
            node_id=301,
            frame=1,
            parent_id=NO_PARENT,
            height=1.0,
            bbox=(0, 0, 1, 1),
        )
        session.add(LinkDB(source_id=201, target_id=301, weight=0.25))
        session.commit()
    engine.dispose()


def test_summary_text_reports_node_and_edge_probability_statistics(tmp_path):
    pytest.importorskip("ultrack")

    from cellflow.tracking_ultrack.db_query import summary_text
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, NodeDB

    db_path = tmp_path / "data.db"
    _make_hierarchy_db(db_path)

    import sqlalchemy as sqla

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        session.query(NodeDB).filter_by(id=100).update({"node_prob": 0.1})
        session.query(NodeDB).filter_by(id=101).update({"node_prob": 0.3})
        session.query(NodeDB).filter_by(id=102).update({"node_prob": 0.7})
        session.query(NodeDB).filter_by(id=201).update({"node_prob": 0.9})
        session.query(NodeDB).filter_by(id=301).update({"node_prob": None})
        session.add(LinkDB(source_id=101, target_id=301, weight=0.75))
        session.commit()
    engine.dispose()

    text = summary_text(db_path, frame=0)

    assert "node prob 4/5 scored" in text
    assert "min 0.100" in text
    assert "median 0.500" in text
    assert "mean 0.500" in text
    assert "max 0.900" in text
    assert "edge weight 2 links" in text
    assert "min 0.250" in text
    assert "median 0.500" in text
    assert "max 0.750" in text


def test_query_hierarchy_cut_states_promotes_equal_height_plateau(tmp_path):
    from cellflow.tracking_ultrack.db_query import query_hierarchy_cut_states

    db_path = tmp_path / "data.db"
    _make_hierarchy_db(db_path)

    states = query_hierarchy_cut_states(db_path, frame=0)

    assert [state.node_ids for state in states] == [
        (201, 102),
        (101, 102),
        (100,),
    ]
    assert [state.height for state in states] == [10.0, 10.0, 10.0]


def test_query_connected_nodes_returns_predecessor_successor_weights(tmp_path):
    from cellflow.tracking_ultrack.db_query import query_connected_nodes

    db_path = tmp_path / "data.db"
    _make_hierarchy_db(db_path)

    predecessors, successors = query_connected_nodes(db_path, selected_node_id=201)

    assert predecessors == {}
    assert successors == {301: pytest.approx(0.25)}


def test_render_hierarchy_cut_state_returns_preview_metadata(tmp_path):
    from cellflow.tracking_ultrack.db_query import (
        HierarchyCutState,
        render_hierarchy_cut_state,
    )

    db_path = tmp_path / "data.db"
    _make_hierarchy_db(db_path)

    preview = render_hierarchy_cut_state(
        db_path,
        frame=0,
        state=HierarchyCutState((101, 102), 10.0),
        plane_shape=(1, 1),
    )

    assert set(preview.label_to_node_id.values()) == {101, 102}
    assert preview.labels.shape == (2, 4)
    assert "2 segment(s)" in preview.status
