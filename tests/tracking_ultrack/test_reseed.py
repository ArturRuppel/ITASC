"""Tests for cellflow.tracking_ultrack.reseed.

Three test groups:
  - Unit: prune_validated_overlaps — synthetic NodeDB, assert correct deletions
  - Unit: merge_validated_into_export — synthetic export + validated tracks
"""
from __future__ import annotations

import pickle
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.reseed import (
    merge_validated_into_export,
    prune_validated_overlaps,
)


def _has_ultrack() -> bool:
    try:
        return importlib.util.find_spec("ultrack") is not None
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_engine(db_path: Path):
    from ultrack.core.database import Base
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine


def _make_node_row(node_id: int, t: int, y0: int, x0: int, y1: int, x1: int):
    """Build a NodeDB row with a (1, h, w) mask crop as stored by ingest.py."""
    from ultrack.core.database import NodeDB
    from ultrack.core.segmentation.node import Node

    h, w = y1 - y0, x1 - x0
    mask_crop = np.ones((1, h, w), dtype=bool)  # fully filled rectangle
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
    from ultrack.core.database import NodeDB  # noqa: F401 needed for session
    with Session(engine) as session:
        session.bulk_save_objects(nodes)
        session.commit()


def _insert_overlaps(engine, pairs: list[tuple[int, int]]):
    """Insert (node_id, ancestor_id) pairs into OverlapDB."""
    from ultrack.core.database import OverlapDB
    with Session(engine) as session:
        for nid, aid in pairs:
            session.add(OverlapDB(node_id=nid, ancestor_id=aid))
        session.commit()


def _count_nodes(engine) -> int:
    from ultrack.core.database import NodeDB
    with Session(engine) as session:
        return session.query(NodeDB).count()


def _count_overlaps(engine) -> int:
    from ultrack.core.database import OverlapDB
    with Session(engine) as session:
        return session.query(OverlapDB).count()


def _node_ids(engine) -> set[int]:
    from ultrack.core.database import NodeDB
    with Session(engine) as session:
        return {r.id for r in session.query(NodeDB).all()}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracking_cfg():
    return TrackingConfig()


# ===========================================================================
# Unit tests — prune_validated_overlaps
# ===========================================================================

@pytest.mark.skipif(not _has_ultrack(), reason="Ultrack is required for DB pruning tests")
class TestPruneValidatedOverlaps:
    """Unit tests for prune_validated_overlaps."""

    def test_no_validated_tracks_is_noop(self, tmp_path):
        """Empty validated_tracks dict → nothing deleted."""
        engine = _make_engine(tmp_path / "data.db")
        _insert_nodes(engine, [_make_node_row(1, 0, 0, 0, 10, 10)])

        tracked = np.zeros((1, 20, 20), dtype=np.uint32)
        n = prune_validated_overlaps(tmp_path, {}, tracked)

        assert n == 0
        assert _count_nodes(engine) == 1

    def test_overlapping_node_is_deleted(self, tmp_path):
        """A node whose bbox overlaps the validated mask is deleted."""
        engine = _make_engine(tmp_path / "data.db")
        # Node at t=0, pixels [5:15, 5:15]
        _insert_nodes(engine, [_make_node_row(1, 0, 5, 5, 15, 15)])

        # validated cell 99 occupies [8:12, 8:12] at t=0 — overlaps node bbox
        tracked = np.zeros((1, 30, 30), dtype=np.uint32)
        tracked[0, 8:12, 8:12] = 99

        n = prune_validated_overlaps(tmp_path, {99: {0}}, tracked)

        assert n == 1
        assert _count_nodes(engine) == 0

    def test_non_overlapping_node_is_kept(self, tmp_path):
        """A node that doesn't overlap any validated mask is preserved."""
        engine = _make_engine(tmp_path / "data.db")
        # Node far away from validated cell
        _insert_nodes(engine, [_make_node_row(1, 0, 0, 0, 5, 5)])

        # validated cell at [20:25, 20:25]
        tracked = np.zeros((1, 30, 30), dtype=np.uint32)
        tracked[0, 20:25, 20:25] = 7

        n = prune_validated_overlaps(tmp_path, {7: {0}}, tracked)

        assert n == 0
        assert _count_nodes(engine) == 1

    def test_mixed_nodes_only_overlapping_deleted(self, tmp_path):
        """With two nodes at the same frame, only the overlapping one is deleted."""
        engine = _make_engine(tmp_path / "data.db")
        # Node 1: [0:10, 0:10] — overlaps validated cell at [5:8, 5:8]
        # Node 2: [15:25, 15:25] — no overlap
        _insert_nodes(engine, [
            _make_node_row(1, 0, 0, 0, 10, 10),
            _make_node_row(2, 0, 15, 15, 25, 25),
        ])

        tracked = np.zeros((1, 30, 30), dtype=np.uint32)
        tracked[0, 5:8, 5:8] = 42

        n = prune_validated_overlaps(tmp_path, {42: {0}}, tracked)

        assert n == 1
        assert _node_ids(engine) == {2}

    def test_overlapdb_rows_are_deleted_when_node_pruned(self, tmp_path):
        """OverlapDB rows referencing a pruned node are removed in the same transaction."""
        engine = _make_engine(tmp_path / "data.db")
        # Two nodes: N1 and N2, with an OverlapDB pair between them.
        # N1 overlaps the validated cell → should be deleted together with the pair.
        _insert_nodes(engine, [
            _make_node_row(10, 0, 0, 0, 10, 10),   # N1 — overlaps validated cell
            _make_node_row(20, 0, 0, 0, 10, 10),   # N2 — same bbox (alternative hypothesis)
        ])
        _insert_overlaps(engine, [(10, 20)])

        tracked = np.zeros((1, 20, 20), dtype=np.uint32)
        tracked[0, 2:8, 2:8] = 55

        n = prune_validated_overlaps(tmp_path, {55: {0}}, tracked)

        # Both N1 and N2 overlap the validated cell → both deleted
        assert n == 2
        assert _count_nodes(engine) == 0
        assert _count_overlaps(engine) == 0

    def test_overlapdb_rows_referencing_surviving_nodes_kept(self, tmp_path):
        """OverlapDB rows between non-conflicting nodes are not deleted."""
        engine = _make_engine(tmp_path / "data.db")
        # N1 at [0:10, 0:10] — overlaps validated cell
        # N2 at [15:25, 15:25] — safe
        # N3 at [15:25, 15:25] — safe; overlap pair (N2, N3)
        _insert_nodes(engine, [
            _make_node_row(1, 0, 0, 0, 10, 10),
            _make_node_row(2, 0, 15, 15, 25, 25),
            _make_node_row(3, 0, 15, 15, 25, 25),
        ])
        _insert_overlaps(engine, [(2, 3)])

        tracked = np.zeros((1, 30, 30), dtype=np.uint32)
        tracked[0, 3:7, 3:7] = 99

        n = prune_validated_overlaps(tmp_path, {99: {0}}, tracked)

        assert n == 1                      # only N1 deleted
        assert _node_ids(engine) == {2, 3}
        assert _count_overlaps(engine) == 1  # (N2, N3) pair preserved

    def test_multi_frame_prune(self, tmp_path):
        """Validated cell across two frames prunes nodes at each frame independently."""
        engine = _make_engine(tmp_path / "data.db")
        # t=0: N1 overlaps; N2 doesn't
        # t=1: N3 overlaps
        _insert_nodes(engine, [
            _make_node_row(1, 0, 2, 2, 12, 12),   # overlaps at t=0
            _make_node_row(2, 0, 20, 20, 28, 28),  # safe at t=0
            _make_node_row(3, 1, 3, 3, 13, 13),    # overlaps at t=1
        ])

        tracked = np.zeros((2, 30, 30), dtype=np.uint32)
        tracked[0, 5:9, 5:9] = 77
        tracked[1, 5:9, 5:9] = 77

        n = prune_validated_overlaps(tmp_path, {77: {0, 1}}, tracked)

        assert n == 2
        assert _node_ids(engine) == {2}

    def test_cell_absent_at_frame_is_skipped(self, tmp_path):
        """If the validated cell is absent (all-zero) at a claimed frame, skip gracefully."""
        engine = _make_engine(tmp_path / "data.db")
        _insert_nodes(engine, [_make_node_row(1, 0, 0, 0, 10, 10)])

        # tracked has cell 88 at t=1 but not t=0; validated_tracks says t=0 too
        tracked = np.zeros((2, 20, 20), dtype=np.uint32)
        tracked[1, 2:8, 2:8] = 88

        # Even though t=0 is "validated", the cell doesn't appear there
        n = prune_validated_overlaps(tmp_path, {88: {0, 1}}, tracked)

        # t=0: no mask found → skip; t=1: node 1 is at t=0, not t=1 → no overlap
        assert n == 0
        assert _count_nodes(engine) == 1

    def test_pixel_exact_intersection_required(self, tmp_path):
        """Adjacent bboxes (touching but not overlapping pixels) do not count as conflict."""
        engine = _make_engine(tmp_path / "data.db")
        # Node bbox: rows [0:5, 0:5] — ends at row 5 (exclusive)
        # Validated cell: rows [5:10, 0:5] — starts at row 5
        # Bboxes touch at y=5 but no pixel shared
        _insert_nodes(engine, [_make_node_row(1, 0, 0, 0, 5, 5)])

        tracked = np.zeros((1, 15, 15), dtype=np.uint32)
        tracked[0, 5:10, 0:5] = 11

        n = prune_validated_overlaps(tmp_path, {11: {0}}, tracked)

        assert n == 0
        assert _count_nodes(engine) == 1


# ===========================================================================
# Unit tests — merge_validated_into_export
# ===========================================================================

class TestMergeValidatedIntoExport:
    """Unit tests for merge_validated_into_export."""

    def test_no_validated_tracks_noop(self):
        """Empty validated_tracks → exported_labels unchanged."""
        exported = np.array([[[1, 0], [0, 2]]], dtype=np.uint32)
        original = exported.copy()
        result, id_map = merge_validated_into_export(exported, {}, exported)
        assert np.array_equal(result, original)
        assert id_map == {}

    def test_validated_pixels_overwrite_exported(self):
        """Validated mask pixels end up with the validated cell ID.

        When the solver filled everything with ID=5 (degenerate case), solver
        track 5 is inferred to be the same as validated cell 77 via the dominant
        ID in the validated mask region, so pixels carrying solver ID=5 are
        remapped to 77.  The validated mask region always ends up as 77.
        """
        T, H, W = 3, 10, 10
        exported = np.ones((T, H, W), dtype=np.uint32) * 5  # solver filled it all with ID=5

        tracked = np.zeros((T, H, W), dtype=np.uint32)
        tracked[1, 2:5, 2:5] = 77  # cell 77 at t=1

        result, id_map = merge_validated_into_export(exported, {77: {1}}, tracked)

        assert result[1, 2, 2] == 77
        assert result[1, 4, 4] == 77
        assert id_map == {}

    def test_all_frames_of_one_cell_get_same_new_id(self):
        """A multi-frame validated cell uses the same track ID across all its frames."""
        T, H, W = 5, 15, 15
        exported = np.zeros((T, H, W), dtype=np.uint32)

        tracked = np.zeros((T, H, W), dtype=np.uint32)
        for t in range(3):
            tracked[t, 4:8, 4:8] = 33  # cell 33 at t=0,1,2

        result, id_map = merge_validated_into_export(exported, {33: {0, 1, 2}}, tracked)

        ids_used = set()
        for t in range(3):
            ids_at_t = set(result[t].ravel()) - {0}
            ids_used.update(ids_at_t)

        # All frames must use the same single new ID
        assert len(ids_used) == 1
        assert ids_used == {33}
        assert id_map == {}

    def test_different_cells_get_different_ids(self):
        """Two validated cells keep their distinct validated IDs."""
        T, H, W = 2, 20, 20
        exported = np.zeros((T, H, W), dtype=np.uint32)

        tracked = np.zeros((T, H, W), dtype=np.uint32)
        tracked[0, 0:5, 0:5] = 10
        tracked[0, 10:15, 10:15] = 20

        result, id_map = merge_validated_into_export(exported, {10: {0}, 20: {0}}, tracked)

        # Collect IDs placed in the two regions
        id_a = result[0, 2, 2]
        id_b = result[0, 12, 12]
        assert id_a != 0
        assert id_b != 0
        assert id_a != id_b
        assert id_a == 10
        assert id_b == 20
        assert id_map == {}

    def test_validated_ids_are_reserved_from_solver_collisions(self):
        """Solver pixels using a validated ID outside its mask are moved away."""
        T, H, W = 3, 20, 20
        max_solver_id = 1000
        exported = np.zeros((T, H, W), dtype=np.uint32)
        exported[0, 0:3, 0:3] = max_solver_id
        exported[2, 0:3, 0:3] = 11

        tracked = np.zeros((T, H, W), dtype=np.uint32)
        tracked[0, 5:10, 5:10] = 11
        tracked[1, 5:10, 5:10] = 22
        tracked[2, 5:10, 5:10] = 33

        result, id_map = merge_validated_into_export(
            exported,
            {11: {0}, 22: {1}, 33: {2}},
            tracked,
        )

        all_ids = set(result.ravel()) - {0}
        # The original max_solver_id is still there
        assert max_solver_id in all_ids
        assert result[0, 5, 5] == 11
        assert result[1, 5, 5] == 22
        assert result[2, 5, 5] == 33
        assert result[2, 1, 1] > max_solver_id
        assert id_map == {}

    def test_id_map_omits_unchanged_validated_ids(self):
        """Unchanged validated IDs do not force validation JSON remapping."""
        T, H, W = 2, 20, 20
        original_max = 50
        exported = np.zeros((T, H, W), dtype=np.uint32)
        exported[0, 0, 0] = original_max

        tracked = np.zeros((T, H, W), dtype=np.uint32)
        tracked[0, 0:5, 0:5] = 10
        tracked[0, 10:15, 10:15] = 20
        tracked[1, 5:10, 5:10] = 30

        _result, id_map = merge_validated_into_export(
            exported,
            {10: {0}, 20: {0}, 30: {1}},
            tracked,
        )

        assert id_map == {}

    def test_validated_cell_absent_at_frame_skipped_gracefully(self):
        """If a validated cell is absent from tracked_labels at a claimed frame, skip it."""
        T, H, W = 2, 10, 10
        exported = np.zeros((T, H, W), dtype=np.uint32)
        tracked = np.zeros((T, H, W), dtype=np.uint32)
        # Cell 5 only present at t=0, not t=1; validated_tracks claims both
        tracked[0, 2:5, 2:5] = 5

        result, id_map = merge_validated_into_export(exported, {5: {0, 1}}, tracked)

        assert np.any(result[0] != 0)  # t=0: cell placed
        assert np.all(result[1] == 0)  # t=1: nothing placed (cell absent)
        assert id_map == {}

    def test_3d_spatial_input(self):
        """4-D (T, Z, Y, X) labelmap works correctly."""
        T, Z, H, W = 2, 3, 10, 10
        exported = np.zeros((T, Z, H, W), dtype=np.uint32)

        tracked = np.zeros((T, Z, H, W), dtype=np.uint32)
        tracked[0, :, 3:7, 3:7] = 99  # cell 99 across all Z slices at t=0

        result, id_map = merge_validated_into_export(exported, {99: {0}}, tracked)

        assert np.all(result[0, :, 3:7, 3:7] == 99)
        assert np.all(result[1] == 0)
        assert id_map == {}
