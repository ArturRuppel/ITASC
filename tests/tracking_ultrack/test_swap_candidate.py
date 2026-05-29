from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from cellflow.tracking_ultrack.swap_candidate import (
    SwapCandidate,
    list_swap_candidates,
    step_larger,
    step_smaller,
)
from cellflow.tracking_ultrack._node_geometry import make_node_pickle


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


def _make_node_row(node_id: int, t: int, y0: int, x0: int, y1: int, x1: int):
    from ultrack.core.database import NodeDB
    from ultrack.core.segmentation.node import Node

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
        z=0,
        y=cy,
        x=cx,
        area=h * w,
        pickle=blob,
    )


def _insert_nodes(engine, nodes):
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        session.bulk_save_objects(nodes)
        session.commit()


# Frame shape used throughout
H, W = 100, 100


# ---------------------------------------------------------------------------
# test_list_swap_candidates_filters_by_radius
# ---------------------------------------------------------------------------

class TestListSwapCandidatesFiltersByRadius:
    def test_filters_by_radius(self, tmp_path):
        db_path = tmp_path / "data.db"
        engine = _make_engine(db_path)

        # node 1: centroid at (10,10), within 40px of source_centroid=(10,10)
        # node 2: centroid at (50,50), dist=56.6 from (10,10) — outside 40px
        # node 3: centroid at (20,20), within 40px (dist=14.1)
        _insert_nodes(engine, [
            _make_node_row(1, 0, 5, 5, 15, 15),    # centroid=(10, 10)
            _make_node_row(2, 0, 45, 45, 55, 55),  # centroid=(50, 50)
            _make_node_row(3, 0, 15, 15, 25, 25),  # centroid=(20, 20)
        ])

        candidates = list_swap_candidates(
            db_path=db_path,
            frame=0,
            source_centroid=(10.0, 10.0),
            radius_px=40.0,
            frame_shape=(H, W),
        )
        node_ids = {c.node_id for c in candidates}
        assert 1 in node_ids
        assert 3 in node_ids
        assert 2 not in node_ids

    def test_filters_distant_nodes_before_reading_masks(self, tmp_path):
        """Candidate search should not deserialize masks outside the radius window."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB

        db_path = tmp_path / "data.db"
        engine = _make_engine(db_path)
        near_pickle = make_node_pickle(
            0,
            np.ones((5, 5), dtype=bool),
            np.array([8, 8, 13, 13], dtype=np.int64),
            1,
            ndim=3,
        )
        with Session(engine) as session:
            session.add_all(
                [
                    NodeDB(
                        id=1,
                        t=0,
                        t_node_id=1,
                        t_hier_id=0,
                        z=0,
                        y=10.0,
                        x=10.0,
                        area=25,
                        pickle=near_pickle,
                    ),
                    NodeDB(
                        id=2,
                        t=0,
                        t_node_id=2,
                        t_hier_id=0,
                        z=0,
                        y=80.0,
                        x=80.0,
                        area=25,
                        pickle=b"far",
                    ),
                ]
            )
            session.commit()

        candidates = list_swap_candidates(
            db_path=db_path,
            frame=0,
            source_centroid=(10.0, 10.0),
            radius_px=20.0,
            frame_shape=(H, W),
        )

        assert [c.node_id for c in candidates] == [1]


# ---------------------------------------------------------------------------
# test_list_swap_candidates_sorted_by_area_asc
# ---------------------------------------------------------------------------

class TestListSwapCandidatesSortedByAreaAsc:
    def test_sorted_asc(self, tmp_path):
        db_path = tmp_path / "data.db"
        engine = _make_engine(db_path)

        # Three nodes near origin with different sizes
        _insert_nodes(engine, [
            _make_node_row(10, 0, 0, 0, 20, 20),   # area=400
            _make_node_row(11, 0, 0, 0, 5, 5),     # area=25
            _make_node_row(12, 0, 0, 0, 10, 10),   # area=100
        ])

        candidates = list_swap_candidates(
            db_path=db_path,
            frame=0,
            source_centroid=(10.0, 10.0),
            radius_px=200.0,
            frame_shape=(H, W),
        )
        areas = [c.area for c in candidates]
        assert areas == sorted(areas), f"Expected area-ASC order, got {areas}"
        assert len(areas) == 3


# ---------------------------------------------------------------------------
# test_list_swap_candidates_empty_when_db_missing
# ---------------------------------------------------------------------------

class TestListSwapCandidatesEmptyWhenDbMissing:
    def test_missing_db_returns_empty(self, tmp_path):
        db_path = tmp_path / "nonexistent.db"
        result = list_swap_candidates(
            db_path=db_path,
            frame=0,
            source_centroid=(0.0, 0.0),
            radius_px=100.0,
            frame_shape=(H, W),
        )
        assert result == []


# ---------------------------------------------------------------------------
# test_directional_step_z_finds_strictly_smaller
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


class TestDirectionalStepZ:
    def test_finds_strictly_smaller(self):
        # Areas sorted ASC: [10, 20, 30, 50]
        # displayed_area=35 → largest area < 35 is 30 at index 2
        candidates = _make_candidates([10, 20, 30, 50])
        idx = step_smaller(candidates, displayed_area=35)
        assert idx == 2
        assert candidates[idx].area == 30

    def test_finds_largest_strictly_smaller(self):
        # displayed_area=25 → largest strictly smaller is 20 at index 1
        candidates = _make_candidates([10, 20, 30, 50])
        idx = step_smaller(candidates, displayed_area=25)
        assert idx == 1
        assert candidates[idx].area == 20


# ---------------------------------------------------------------------------
# test_directional_step_c_finds_strictly_larger
# ---------------------------------------------------------------------------

class TestDirectionalStepC:
    def test_finds_strictly_larger(self):
        # Areas: [10, 20, 30, 50]
        # displayed_area=25 → smallest area > 25 is 30 at index 2
        candidates = _make_candidates([10, 20, 30, 50])
        idx = step_larger(candidates, displayed_area=25)
        assert idx == 2
        assert candidates[idx].area == 30

    def test_finds_smallest_strictly_larger(self):
        # displayed_area=10 → smallest > 10 is 20 at index 1
        candidates = _make_candidates([10, 20, 30, 50])
        idx = step_larger(candidates, displayed_area=10)
        assert idx == 1
        assert candidates[idx].area == 20


# ---------------------------------------------------------------------------
# test_directional_step_bounds
# ---------------------------------------------------------------------------

class TestDirectionalStepBounds:
    def test_z_at_bottom_returns_none(self):
        # displayed_area=10 == smallest candidate — no smaller exists
        candidates = _make_candidates([10, 20, 30])
        assert step_smaller(candidates, displayed_area=10) is None

    def test_z_below_all_returns_none(self):
        candidates = _make_candidates([10, 20, 30])
        assert step_smaller(candidates, displayed_area=5) is None

    def test_c_at_top_returns_none(self):
        # displayed_area=30 == largest candidate — no larger exists
        candidates = _make_candidates([10, 20, 30])
        assert step_larger(candidates, displayed_area=30) is None

    def test_c_above_all_returns_none(self):
        candidates = _make_candidates([10, 20, 30])
        assert step_larger(candidates, displayed_area=100) is None
