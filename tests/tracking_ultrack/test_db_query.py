from pathlib import Path

import numpy as np
import pytest


def _add_ultrack_node(
    session,
    *,
    node_id: int,
    frame: int = 0,
    parent_id: int | None,
    height: float,
    bbox: tuple[int, int, int, int],
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
