from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from cellflow.tracking_ultrack.swap_candidate import (
    SwapCandidate,
    cycle_index,
    list_swap_candidates,
    nearest_area_index,
)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _make_engine(db_path: Path):
    pytest.importorskip("ultrack")
    import sqlalchemy as sqla
    from ultrack.core.database import Base

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine


def _make_node_row(
    node_id: int,
    t: int,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    *,
    hier_parent_id: int | None = None,
    height: float = 0.0,
):
    from ultrack.core.database import NodeDB
    from ultrack.core.segmentation.node import Node
    from ultrack.utils.constants import NO_PARENT

    h, w = y1 - y0, x1 - x0
    mask_crop = np.ones((1, h, w), dtype=bool)
    bbox_3d = np.array([0, y0, x0, 1, y1, x1], dtype=np.int64)
    node = Node.from_mask(time=t, mask=mask_crop, bbox=bbox_3d, node_id=node_id)
    blob = pickle.dumps(node)
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    return NodeDB(
        id=node_id,
        t=t,
        t_node_id=node_id,
        t_hier_id=1,
        hier_parent_id=NO_PARENT if hier_parent_id is None else hier_parent_id,
        height=height,
        z=0,
        y=cy,
        x=cx,
        area=h * w,
        pickle=blob,
    )


def _full_mask(y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _insert_nodes(engine, nodes):
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        session.bulk_save_objects(nodes)
        session.commit()


# Frame shape used throughout
H, W = 100, 100


# ---------------------------------------------------------------------------
# Hierarchy fixture
# ---------------------------------------------------------------------------
# root 1: (0:20, 0:20)  area 400
#   node 2: (0:20, 0:10)  area 200   parent 1
#     node 4: (0:10, 0:10) area 100  parent 2   <- source matches here
#     node 5: (10:20, 0:10) area 100 parent 2
#   node 3: (0:20, 10:20) area 200   parent 1
# unrelated root 9: (50:70, 50:70) area 400 (separate lattice)

def _insert_hierarchy(engine):
    _insert_nodes(engine, [
        _make_node_row(1, 0, 0, 0, 20, 20, height=3.0),
        _make_node_row(2, 0, 0, 0, 20, 10, hier_parent_id=1, height=2.0),
        _make_node_row(3, 0, 0, 10, 20, 20, hier_parent_id=1, height=2.0),
        _make_node_row(4, 0, 0, 0, 10, 10, hier_parent_id=2, height=1.0),
        _make_node_row(5, 0, 10, 0, 20, 10, hier_parent_id=2, height=1.0),
        _make_node_row(9, 0, 50, 50, 70, 70, height=3.0),
    ])


# ---------------------------------------------------------------------------
# test_list_swap_candidates_returns_lattice_branch
# ---------------------------------------------------------------------------

class TestListSwapCandidatesReturnsLatticeBranch:
    def test_returns_matched_node_lineage_excluding_siblings(self, tmp_path):
        db_path = tmp_path / "data.db"
        engine = _make_engine(db_path)
        _insert_hierarchy(engine)

        # source mask matches leaf node 4 exactly
        candidates = list_swap_candidates(
            db_path=db_path,
            frame=0,
            source_mask=_full_mask(0, 0, 10, 10),
            frame_shape=(H, W),
        )
        node_ids = {c.node_id for c in candidates}
        # lineage of node 4: itself + ancestors (2, 1). No descendants (leaf).
        assert node_ids == {1, 2, 4}
        # sibling (5), cousin (3) and the unrelated lattice (9) are excluded
        assert node_ids.isdisjoint({3, 5, 9})

    def test_includes_descendants_of_matched_node(self, tmp_path):
        db_path = tmp_path / "data.db"
        engine = _make_engine(db_path)
        _insert_hierarchy(engine)

        # source mask matches interior node 2 exactly
        candidates = list_swap_candidates(
            db_path=db_path,
            frame=0,
            source_mask=_full_mask(0, 0, 20, 10),  # == node 2
            frame_shape=(H, W),
        )
        node_ids = {c.node_id for c in candidates}
        # lineage of node 2: itself, ancestor (1) and descendants (4, 5).
        assert node_ids == {1, 2, 4, 5}
        # node 2's sibling (3) is excluded
        assert 3 not in node_ids

    def test_match_is_by_overlap_not_proximity(self, tmp_path):
        db_path = tmp_path / "data.db"
        engine = _make_engine(db_path)
        _insert_hierarchy(engine)

        # A mask overlapping node 3's region matches that branch; node 3 is a
        # leaf-ish node whose only ancestor is the root.
        candidates = list_swap_candidates(
            db_path=db_path,
            frame=0,
            source_mask=_full_mask(0, 10, 20, 20),  # == node 3
            frame_shape=(H, W),
        )
        node_ids = {c.node_id for c in candidates}
        assert node_ids == {1, 3}
        # node 3's nephews (4, 5) and sibling subtree (2) are excluded
        assert node_ids.isdisjoint({2, 4, 5})

    def test_no_overlap_returns_empty(self, tmp_path):
        db_path = tmp_path / "data.db"
        engine = _make_engine(db_path)
        _insert_hierarchy(engine)

        candidates = list_swap_candidates(
            db_path=db_path,
            frame=0,
            source_mask=_full_mask(80, 80, 90, 90),  # empty region
            frame_shape=(H, W),
        )
        assert candidates == []


# ---------------------------------------------------------------------------
# test_list_swap_candidates_sorted_by_area_asc
# ---------------------------------------------------------------------------

class TestListSwapCandidatesSortedByAreaAsc:
    def test_sorted_asc(self, tmp_path):
        db_path = tmp_path / "data.db"
        engine = _make_engine(db_path)
        _insert_hierarchy(engine)

        candidates = list_swap_candidates(
            db_path=db_path,
            frame=0,
            source_mask=_full_mask(0, 0, 10, 10),
            frame_shape=(H, W),
        )
        areas = [c.area for c in candidates]
        assert areas == sorted(areas), f"Expected area-ASC order, got {areas}"


# ---------------------------------------------------------------------------
# test_list_swap_candidates_empty_when_db_missing
# ---------------------------------------------------------------------------

class TestListSwapCandidatesEmptyWhenDbMissing:
    def test_missing_db_returns_empty(self, tmp_path):
        db_path = tmp_path / "nonexistent.db"
        result = list_swap_candidates(
            db_path=db_path,
            frame=0,
            source_mask=_full_mask(0, 0, 10, 10),
            frame_shape=(H, W),
        )
        assert result == []


# ---------------------------------------------------------------------------
# Cursor seeding + cycling
# ---------------------------------------------------------------------------

def _make_candidates(areas: list[int]) -> tuple[SwapCandidate, ...]:
    return tuple(
        SwapCandidate(
            node_id=i,
            mask_2d=np.zeros((H, W), dtype=bool),
            bbox=(0, 0, 1, 1),
            centroid=(0.0, 0.0),
            area=a,
        )
        for i, a in enumerate(areas)
    )


class TestNearestAreaIndex:
    def test_exact_match(self):
        candidates = _make_candidates([10, 20, 30, 50])
        assert nearest_area_index(candidates, 30) == 2

    def test_closest_when_no_exact(self):
        # 23 is closer to 20 (index 1) than 30 (index 2)
        candidates = _make_candidates([10, 20, 30, 50])
        assert nearest_area_index(candidates, 23) == 1

    def test_clamps_to_ends(self):
        candidates = _make_candidates([10, 20, 30])
        assert nearest_area_index(candidates, 5) == 0
        assert nearest_area_index(candidates, 999) == 2


class TestCycleIndex:
    def test_larger_steps_up(self):
        # area-sorted, so "larger" moves toward higher index
        assert cycle_index(4, 1, larger=True) == 2

    def test_smaller_steps_down(self):
        assert cycle_index(4, 2, larger=False) == 1

    def test_larger_wraps_at_top(self):
        assert cycle_index(4, 3, larger=True) == 0

    def test_smaller_wraps_at_bottom(self):
        assert cycle_index(4, 0, larger=False) == 3

    def test_full_cycle_visits_every_index(self):
        # Repeated "larger" steps reach all four indices including ties-free wrap
        seen = []
        idx = 0
        for _ in range(4):
            seen.append(idx)
            idx = cycle_index(4, idx, larger=True)
        assert sorted(seen) == [0, 1, 2, 3]
        assert idx == 0  # back to start
