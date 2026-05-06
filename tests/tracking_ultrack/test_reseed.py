"""Tests for cellflow.tracking_ultrack.reseed.

Three test groups:
  - Unit: prune_validated_overlaps — synthetic NodeDB, assert correct deletions
  - Unit: merge_validated_into_export — synthetic export + validated tracks
  - Integration: resolve_with_validation round-trip on a small synthetic dataset
"""
from __future__ import annotations

import pickle
import tempfile
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


# ===========================================================================
# Unit tests — resolve_with_validation orchestration
# ===========================================================================

def test_resolve_with_validation_solves_with_annotations(monkeypatch, tmp_path):
    from cellflow.tracking_ultrack.reseed import resolve_with_validation

    calls = {"solve": []}
    tracked = np.zeros((1, 16, 16), dtype=np.uint32)
    tracked[0, 1:5, 1:5] = 7
    image_path = tmp_path / "nucleus_zavg.tif"
    import tifffile
    tifffile.imwrite(image_path, np.ones((1, 16, 16), dtype=np.float32))

    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.ingest_hypotheses_to_db",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.inject_validated_nodes",
        lambda *args, **kwargs: type("Report", (), {"inserted": 1, "skipped_missing": 0, "faked": 0})(),
        raising=False,
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.write_seed_prior_node_probs",
        lambda *args, **kwargs: type("Report", (), {"scored": 1, "seeds": 1})(),
        raising=False,
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.run_linking",
        lambda *args, **kwargs: iter([(1, 1, "linked")]),
        raising=False,
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.run_solve",
        lambda *args, **kwargs: calls["solve"].append(kwargs) or iter([(1, 1, "solved")]),
        raising=False,
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.export_tracked_labels",
        lambda *args, **kwargs: tracked.copy(),
        raising=False,
    )
    result, id_map = resolve_with_validation(
        tmp_path / "hypotheses.h5",
        {7: {0}},
        tracked,
        TrackingConfig(),
        intensity_image_path=image_path,
    )

    assert result.shape == tracked.shape
    assert id_map == {}
    assert np.all(result[0, 1:5, 1:5] == 7)
    assert calls["solve"] == [{"overwrite": True, "use_annotations": True}]


def test_resolve_with_validation_pastes_validated_masks_after_export(monkeypatch, tmp_path):
    from cellflow.tracking_ultrack.reseed import resolve_with_validation

    tracked = np.zeros((1, 16, 16), dtype=np.uint32)
    tracked[0, 1:5, 1:5] = 7
    exported = np.zeros_like(tracked)
    exported[0, 1:5, 1:5] = 99
    image_path = tmp_path / "nucleus_zavg.tif"
    import tifffile
    tifffile.imwrite(image_path, np.ones((1, 16, 16), dtype=np.float32))

    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.ingest_hypotheses_to_db",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.inject_validated_nodes",
        lambda *args, **kwargs: type("Report", (), {"inserted": 1, "skipped_missing": 0, "faked": 0})(),
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.write_seed_prior_node_probs",
        lambda *args, **kwargs: type("Report", (), {"scored": 1, "seeds": 1})(),
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.run_linking",
        lambda *args, **kwargs: iter([(1, 1, "linked")]),
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.run_solve",
        lambda *args, **kwargs: iter([(1, 1, "solved")]),
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.export_tracked_labels",
        lambda *args, **kwargs: exported.copy(),
    )

    result, id_map = resolve_with_validation(
        tmp_path / "hypotheses.h5",
        {7: {0}},
        tracked,
        TrackingConfig(),
        intensity_image_path=image_path,
    )

    assert id_map == {}
    assert np.all(result[0, 1:5, 1:5] == 7)


def test_resolve_with_validation_uses_configured_linker(monkeypatch, tmp_path):
    """Re-solve must honor cfg.linking_mode instead of hard-coding Ultrack link()."""
    from cellflow.tracking_ultrack.reseed import resolve_with_validation

    tracked = np.zeros((2, 12, 12), dtype=np.uint32)
    tracked[0, 1:4, 1:4] = 7
    image_path = tmp_path / "nucleus_zavg.tif"
    import tifffile
    tifffile.imwrite(image_path, np.ones((2, 12, 12), dtype=np.float32))
    cfg = TrackingConfig(min_area=1, linking_mode="iou")
    calls: list[tuple[Path, TrackingConfig]] = []

    def fake_ingest(*args, **kwargs):
        return None

    def fake_run_linking(working_dir, passed_cfg, *, overwrite=True):
        calls.append((Path(working_dir), passed_cfg))
        yield (1, 1, "fake iou linking")

    monkeypatch.setattr("cellflow.tracking_ultrack.reseed.ingest_hypotheses_to_db", fake_ingest)
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.inject_validated_nodes",
        lambda *args, **kwargs: type("Report", (), {"inserted": 1, "skipped_missing": 0, "faked": 0})(),
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.write_seed_prior_node_probs",
        lambda *args, **kwargs: type("Report", (), {"scored": 1, "seeds": 1})(),
    )
    monkeypatch.setattr("cellflow.tracking_ultrack.reseed.run_linking", fake_run_linking)
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.run_solve",
        lambda *args, **kwargs: iter([(1, 1, "fake solve")]),
    )
    monkeypatch.setattr(
        "cellflow.tracking_ultrack.reseed.export_tracked_labels",
        lambda *args, **kwargs: tracked.copy(),
    )

    result, id_map = resolve_with_validation(
        tmp_path / "hypotheses.h5",
        {7: {0}},
        tracked,
        cfg,
        intensity_image_path=image_path,
    )

    assert calls, "Expected resolve_with_validation to delegate through run_linking"
    assert calls[0][1] is cfg
    assert calls[0][1].linking_mode == "iou"
    assert np.array_equal(result, tracked)
    assert id_map == {}


# ===========================================================================
# Integration test — resolve_with_validation round-trip
# ===========================================================================

class TestResolveWithValidation:
    """Integration test: full resolve_with_validation on a tiny synthetic dataset."""

    @pytest.fixture
    def populated_working_dir(self, tmp_path):
        """Ingest a small 3-frame, 2-partition synthetic dataset into Ultrack NodeDB."""
        import h5py
        import tifffile
        from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db

        # Use a low min_area so all synthetic cells (including split halves) pass the filter.
        ingest_cfg = TrackingConfig(min_area=1, max_distance=20.0)

        T, H, W = 3, 60, 60
        h5_path = tmp_path / "hypotheses.h5"

        # Build 3 frames, 2 partitions each.
        # Cells are 12×12 = 144 px so they clear the default min_area=100 filter.
        # Frame layout:
        #   t=0: p0 has cell A (rows 2:14, cols 2:14); p1 has A split into A1+A2
        #   t=1: p0 has cell A migrated slightly (rows 3:15, cols 3:15)
        #   t=2: p0 has cell A (rows 4:16, cols 4:16)
        timepoints = {}
        for t in range(T):
            lm_p0 = np.zeros((1, H, W), dtype=np.uint32)
            lm_p0[0, 2 + t: 14 + t, 2: 14] = 1   # single large cell (12×12 = 144 px)

            lm_p1 = np.zeros((1, H, W), dtype=np.uint32)
            lm_p1[0, 2 + t: 8 + t, 2: 14] = 1    # top half  (6×12 = 72 px each)
            lm_p1[0, 8 + t: 14 + t, 2: 14] = 2   # bottom half

            timepoints[t] = {0: lm_p0, 1: lm_p1}

        with h5py.File(h5_path, "w") as f:
            f.attrs["version"] = 2
            f.attrs["stage"] = "nucleus_hypotheses"
            f.attrs["layout"] = "hypotheses/t{t:03d}/p{p:03d}/labels"
            root = f.create_group("hypotheses")
            for t, parts in timepoints.items():
                tgrp = root.create_group(f"t{t:03d}")
                for p, lm in parts.items():
                    pgrp = tgrp.create_group(f"p{p:03d}")
                    pgrp.create_dataset("labels", data=lm.astype(np.uint32))
                    pgrp.attrs["parameter_index"] = p
                    pgrp.attrs["parameter_json"] = "{}"

        working_dir = tmp_path / "tracking"
        ingest_hypotheses_to_db(h5_path, working_dir, ingest_cfg, overwrite=True)
        image_path = tmp_path / "nucleus_zavg.tif"
        tifffile.imwrite(image_path, np.ones((T, H, W), dtype=np.float32))
        return h5_path, T, H, W, ingest_cfg, image_path

    def test_round_trip_validated_cells_unchanged(
        self, tmp_path, populated_working_dir
    ):
        """After resolve_with_validation, validated pixels are preserved verbatim."""
        from cellflow.tracking_ultrack.reseed import resolve_with_validation

        working_dir, T, H, W, cfg, image_path = populated_working_dir

        # tracked_labels: use a simple labelmap where cell 1 spans all 3 frames.
        # Cells are 12×12 matching the fixture partitions.
        tracked = np.zeros((T, H, W), dtype=np.uint32)
        for t in range(T):
            tracked[t, 2 + t: 14 + t, 2: 14] = 1  # cell 1 present at t=0,1,2

        # Validate cell 1 at t=1 only
        validated_tracks = {1: {1}}

        result, id_map = resolve_with_validation(
            working_dir,
            validated_tracks,
            tracked,
            cfg,
            intensity_image_path=image_path,
        )

        assert result.shape[0] == T
        assert id_map

    def test_validated_cells_have_consistent_id_across_frames(
        self, tmp_path, populated_working_dir
    ):
        """A validated cell spanning multiple frames should use one consistent track ID."""
        from cellflow.tracking_ultrack.reseed import resolve_with_validation

        working_dir, T, H, W, cfg, image_path = populated_working_dir

        tracked = np.zeros((T, H, W), dtype=np.uint32)
        for t in range(T):
            tracked[t, 2 + t: 14 + t, 2: 14] = 1

        # Validate cell 1 across all 3 frames
        validated_tracks = {1: {0, 1, 2}}

        result, id_map = resolve_with_validation(
            working_dir,
            validated_tracks,
            tracked,
            cfg,
            intensity_image_path=image_path,
        )

        assert id_map
        assert np.asarray(result).shape[:1] == tracked.shape[:1]

    def test_resolve_returns_correct_spatial_shape(
        self, tmp_path, populated_working_dir
    ):
        """resolve_with_validation returns a labelmap with the expected (T, H, W) shape."""
        from cellflow.tracking_ultrack.reseed import resolve_with_validation

        working_dir, T, H, W, cfg, image_path = populated_working_dir

        tracked = np.zeros((T, H, W), dtype=np.uint32)
        tracked[0, 5:17, 5:17] = 3

        result, id_map = resolve_with_validation(
            working_dir,
            {3: {0}},
            tracked,
            cfg,
            intensity_image_path=image_path,
        )

        assert result.ndim == tracked.ndim
        assert result.shape == tracked.shape

    def test_no_validated_tracks_still_produces_output(
        self, tmp_path, populated_working_dir
    ):
        """resolve_with_validation with no validated tracks returns a labelmap copy."""
        from cellflow.tracking_ultrack.reseed import resolve_with_validation

        working_dir, T, H, W, cfg, image_path = populated_working_dir

        tracked = np.zeros((T, H, W), dtype=np.uint32)

        result, id_map = resolve_with_validation(
            working_dir,
            {},
            tracked,
            cfg,
            intensity_image_path=image_path,
        )

        assert result.shape[0] == T
        assert result.dtype == np.uint32
        assert id_map == {}
