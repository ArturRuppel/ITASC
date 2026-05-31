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

    lines = text.splitlines()
    assert len(lines) >= 5
    assert all("|" not in line for line in lines)
    assert any(line.startswith("Database:") for line in lines)
    assert any(line.startswith("Node annotations:") for line in lines)
    assert any(line.startswith("Link annotations:") for line in lines)
    assert any(line.startswith("Frame 0:") for line in lines)
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


# ── Atom-union browser model ──────────────────────────────────────────────────


def test_greedy_color_classes_packs_non_overlapping_candidates():
    from cellflow.tracking_ultrack.db_query import greedy_color_classes

    # 4 atoms in a row -> size-2 unions AB=11, BC=12, CD=13; 11-12 and 12-13 overlap.
    classes = greedy_color_classes([11, 12, 13], [(11, 12), (12, 13)])

    # Every candidate appears exactly once; 12 conflicts with both so it is alone,
    # while the disjoint 11 and 13 share a class -> 2 slider positions, not 3.
    assert sorted(sum(classes, ())) == [11, 12, 13]
    assert (12,) in classes and (11, 13) in classes
    # No class contains an overlapping pair.
    edge_set = {(11, 12), (12, 13)}
    for cls in classes:
        for a in cls:
            for b in cls:
                assert (a, b) not in edge_set and (b, a) not in edge_set


def test_greedy_color_classes_edge_cases():
    from cellflow.tracking_ultrack.db_query import greedy_color_classes

    assert greedy_color_classes([], []) == ()
    assert greedy_color_classes([1, 2, 3], []) == ((1, 2, 3),)  # no overlaps -> one class
    assert len(greedy_color_classes([1, 2, 3], [(1, 2), (2, 3), (1, 3)])) == 3  # triangle


def _add_atom_union_node(session, *, node_id, height, bbox, frame=0):
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
            frontier=-1.0,
            pickle=pickle.dumps(node_obj),
        )
    )


def _make_atom_union_db(db_path: Path) -> None:
    """4 atoms A,B,C,D in a row, plus the size-2 unions AB, BC, CD."""
    pytest.importorskip("ultrack")
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import Base, OverlapDB

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _add_atom_union_node(session, node_id=1, height=1, bbox=(0, 0, 2, 2))
        _add_atom_union_node(session, node_id=2, height=1, bbox=(0, 2, 2, 4))
        _add_atom_union_node(session, node_id=3, height=1, bbox=(0, 4, 2, 6))
        _add_atom_union_node(session, node_id=4, height=1, bbox=(0, 6, 2, 8))
        _add_atom_union_node(session, node_id=11, height=2, bbox=(0, 0, 2, 4))  # AB
        _add_atom_union_node(session, node_id=12, height=2, bbox=(0, 2, 2, 6))  # BC
        _add_atom_union_node(session, node_id=13, height=2, bbox=(0, 4, 2, 8))  # CD
        # AB-BC share atom B, BC-CD share atom C (node_id < ancestor_id).
        session.add(OverlapDB(node_id=11, ancestor_id=12))
        session.add(OverlapDB(node_id=12, ancestor_id=13))
        session.commit()
    engine.dispose()


def test_query_union_sizes_returns_distinct_sizes_for_frame(tmp_path):
    from cellflow.tracking_ultrack.db_query import query_union_sizes

    db_path = tmp_path / "data.db"
    _make_atom_union_db(db_path)

    assert query_union_sizes(db_path, frame=0) == (1, 2)


def test_query_union_color_classes_groups_disjoint_size2_candidates(tmp_path):
    from cellflow.tracking_ultrack.db_query import query_union_color_classes

    db_path = tmp_path / "data.db"
    _make_atom_union_db(db_path)

    classes = query_union_color_classes(db_path, frame=0, union_size=2)

    # BC overlaps both neighbors, so it is alone; AB and CD pack together.
    assert sorted(sum(classes, ())) == [11, 12, 13]
    assert (12,) in classes and (11, 13) in classes


def test_query_union_color_classes_size1_atoms_are_one_class(tmp_path):
    from cellflow.tracking_ultrack.db_query import query_union_color_classes

    db_path = tmp_path / "data.db"
    _make_atom_union_db(db_path)

    # Atoms are disjoint -> a single full-frame partition of individual atoms.
    assert query_union_color_classes(db_path, frame=0, union_size=1) == ((1, 2, 3, 4),)


def test_render_union_partition_merges_class_and_fills_leftover_atoms(tmp_path):
    from cellflow.tracking_ultrack.db_query import render_union_partition

    db_path = tmp_path / "data.db"
    _make_atom_union_db(db_path)

    # Class {AB, CD} covers all 4 atoms -> 2 merged regions, fully tiled, no leftovers.
    full = render_union_partition(db_path, 0, (11, 13), plane_shape=(2, 8))
    assert set(full.label_to_node_id.values()) == {11, 13}
    assert set(np.unique(full.labels)) == {1, 2}

    # Class {BC} merges B+C; A and D have no other available union of size <= 2
    # (AB needs B, CD needs C, both taken) so they fall back to individual atoms.
    partial = render_union_partition(db_path, 0, (12,), plane_shape=(2, 8))
    assert set(partial.label_to_node_id.values()) == {12, 1, 4}
    assert (partial.labels == 0).sum() == 0  # still a full-frame partition


def _make_atom_union_db_with_lower_merge(db_path: Path) -> None:
    """6 atoms A..F in a row, the size-3 union ABC, and the size-2 unions AB and DE."""
    pytest.importorskip("ultrack")
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import Base

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for i in range(6):  # atoms A..F = ids 1..6
            _add_atom_union_node(
                session, node_id=i + 1, height=1, bbox=(0, 2 * i, 2, 2 * i + 2)
            )
        _add_atom_union_node(session, node_id=11, height=2, bbox=(0, 0, 2, 4))  # AB
        _add_atom_union_node(session, node_id=14, height=2, bbox=(0, 6, 2, 10))  # DE
        _add_atom_union_node(session, node_id=21, height=3, bbox=(0, 0, 2, 6))  # ABC
        session.commit()
    engine.dispose()


def test_render_union_partition_fills_leftover_with_most_merged_union(tmp_path):
    from cellflow.tracking_ultrack.db_query import render_union_partition

    db_path = tmp_path / "data.db"
    _make_atom_union_db_with_lower_merge(db_path)

    # Viewing the size-3 group {ABC}: leftover D,E must show as the size-2 union DE,
    # not as two separate atoms; only F (no available union) stays an atom.
    preview = render_union_partition(
        db_path, 0, (21,), plane_shape=(2, 12), union_size=3
    )
    assert set(preview.label_to_node_id.values()) == {21, 14, 6}
    assert 4 not in preview.label_to_node_id.values()  # atom D not shown raw
    assert 5 not in preview.label_to_node_id.values()  # atom E not shown raw
    assert (preview.labels == 0).sum() == 0  # still a full-frame partition
