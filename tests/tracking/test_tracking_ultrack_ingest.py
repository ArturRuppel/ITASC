"""Tests for ultrack hypothesis ingestion."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db
from ultrack.core.database import NodeDB, OverlapDB


@pytest.fixture
def tracking_cfg():
    """Create a default tracking config for tests."""
    return TrackingConfig()


def _create_test_h5(path: Path, timepoints: dict[int, dict[int, np.ndarray]]) -> None:
    """Create a test hypotheses.h5 file.

    Parameters
    ----------
    path : Path
        Output HDF5 file path.
    timepoints : dict[int, dict[int, np.ndarray]]
        Nested dict: timepoint -> partition -> labels (shape Z, Y, X, dtype uint32)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["version"] = 2
        h5.attrs["stage"] = "nucleus_hypotheses"
        h5.attrs["layout"] = "hypotheses/t{t:03d}/p{p:03d}/labels"

        root = h5.create_group("hypotheses")
        for t, partitions in timepoints.items():
            t_grp = root.create_group(f"t{t:03d}")
            for p, labels in partitions.items():
                p_grp = t_grp.create_group(f"p{p:03d}")
                p_grp.create_dataset(
                    "labels",
                    data=np.asarray(labels, dtype=np.uint32),
                    compression="gzip",
                )
                p_grp.attrs["parameter_index"] = int(p)
                p_grp.attrs["parameter_json"] = json.dumps(
                    {"method": "test", "param": p}, sort_keys=True
                )


def _count_nodedb_rows(db_path: Path) -> int:
    """Count rows in NodeDB."""
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        return session.query(NodeDB).count()


def _count_overlapdb_rows(db_path: Path) -> int:
    """Count rows in OverlapDB."""
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        return session.query(OverlapDB).count()


def _get_nodedb_rows(db_path: Path) -> list[NodeDB]:
    """Retrieve all NodeDB rows."""
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        return session.query(NodeDB).all()


def _get_overlapdb_rows(db_path: Path) -> list[OverlapDB]:
    """Retrieve all OverlapDB rows."""
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        return session.query(OverlapDB).all()


def test_single_partition_non_overlapping_cells(tmp_path, tracking_cfg):
    """Single partition with 3 non-overlapping cells yields 3 NodeDB, 0 OverlapDB rows."""
    # Create a labelmap with 3 rectangular cells
    labelmap = np.zeros((1, 64, 64), dtype=np.uint32)
    labelmap[0, 0:20, 0:20] = 1  # Cell 1
    labelmap[0, 0:20, 30:50] = 2  # Cell 2
    labelmap[0, 30:50, 0:20] = 3  # Cell 3

    h5_path = tmp_path / "hypotheses.h5"
    working_dir = tmp_path / "tracking"

    _create_test_h5(h5_path, {0: {0: labelmap}})

    ingest_hypotheses_to_db(h5_path, working_dir, tracking_cfg, overwrite=True)

    db_path = working_dir / "data.db"
    assert _count_nodedb_rows(db_path) == 3
    assert _count_overlapdb_rows(db_path) == 0

    # Verify all NodeDB rows have t=0 and t_hier_id=1 (p=0)
    nodes = _get_nodedb_rows(db_path)
    for node in nodes:
        assert node.t == 0
        assert node.t_hier_id == 1


def test_two_partitions_contained_cells(tmp_path, tracking_cfg):
    """Two partitions where p=1 cells are contained in p=0 cells."""
    # p=0: 2 large cells
    labelmap_p0 = np.zeros((1, 64, 64), dtype=np.uint32)
    labelmap_p0[0, 0:30, 0:30] = 1  # Cell 1: top-left
    labelmap_p0[0, 0:30, 32:62] = 2  # Cell 2: top-right

    # p=1: 2 smaller cells strictly inside the p=0 cells
    labelmap_p1 = np.zeros((1, 64, 64), dtype=np.uint32)
    labelmap_p1[0, 5:25, 5:25] = 1  # Inside cell 1
    labelmap_p1[0, 5:25, 37:57] = 2  # Inside cell 2

    h5_path = tmp_path / "hypotheses.h5"
    working_dir = tmp_path / "tracking"

    _create_test_h5(h5_path, {0: {0: labelmap_p0, 1: labelmap_p1}})

    ingest_hypotheses_to_db(h5_path, working_dir, tracking_cfg, overwrite=True)

    db_path = working_dir / "data.db"
    assert _count_nodedb_rows(db_path) == 4
    assert _count_overlapdb_rows(db_path) == 2


def test_node_pickle_roundtrip(tmp_path, tracking_cfg):
    """Pickle round-trip preserves node mask shape, bbox, and area."""
    # Create a single cell
    labelmap = np.zeros((1, 64, 64), dtype=np.uint32)
    labelmap[0, 10:30, 15:45] = 1  # Rectangle: 20x30 pixels

    h5_path = tmp_path / "hypotheses.h5"
    working_dir = tmp_path / "tracking"

    _create_test_h5(h5_path, {0: {0: labelmap}})

    ingest_hypotheses_to_db(h5_path, working_dir, tracking_cfg, overwrite=True)

    db_path = working_dir / "data.db"
    nodes = _get_nodedb_rows(db_path)
    assert len(nodes) == 1

    node_row = nodes[0]
    node = node_row.pickle  # MaybePickleType auto-deserializes

    # Nodes are stored as 3D (Z=1, h, w) to match (Z, Y, X) export buffers
    assert node.mask.shape == (1, 20, 30)
    assert node.bbox is not None
    assert len(node.bbox) == 6  # (min_z, min_y, min_x, max_z, max_y, max_x)
    assert node.area == 600  # 20 * 30


def test_min_area_filter(tmp_path, tracking_cfg):
    """min_area filter applies correctly to cells."""
    # Create 3 cells with different areas
    labelmap = np.zeros((1, 100, 100), dtype=np.uint32)
    # Cell 1: 10x10 = 100 pixels
    labelmap[0, 0:10, 0:10] = 1
    # Cell 2: 15x15 = 225 pixels
    labelmap[0, 10:25, 0:15] = 2
    # Cell 3: 25x25 = 625 pixels
    labelmap[0, 10:35, 30:55] = 3

    h5_path = tmp_path / "hypotheses.h5"
    working_dir = tmp_path / "tracking"

    _create_test_h5(h5_path, {0: {0: labelmap}})

    # Ingest with min_area=150 (filters out cells < 150)
    ingest_hypotheses_to_db(
        h5_path, working_dir, tracking_cfg, overwrite=True, min_area=150
    )

    db_path = working_dir / "data.db"
    # Only cells 2 and 3 should remain (areas 225 and 625)
    assert _count_nodedb_rows(db_path) == 2


def test_dedup_identical_byte_labelmaps(tmp_path, tracking_cfg):
    """Two byte-identical labelmaps → second is dropped (only one set of nodes inserted)."""
    from cellflow.tracking_ultrack.ingest import _canonical_hash

    labelmap = np.zeros((1, 64, 64), dtype=np.uint32)
    labelmap[0, 0:20, 0:20] = 1
    labelmap[0, 30:50, 30:50] = 2

    h5_path = tmp_path / "hypotheses.h5"
    working_dir = tmp_path / "tracking"
    # p=0 and p=1 are byte-identical
    _create_test_h5(h5_path, {0: {0: labelmap, 1: labelmap.copy()}})

    # Verify the canonical hashes match before ingestion
    assert _canonical_hash(labelmap[0]) == _canonical_hash(labelmap[0].copy())

    ingest_hypotheses_to_db(h5_path, working_dir, tracking_cfg, overwrite=True)

    db_path = working_dir / "data.db"
    # Only the first partition's 2 cells should be ingested; the duplicate is dropped.
    assert _count_nodedb_rows(db_path) == 2
    # Single partition → no overlaps
    assert _count_overlapdb_rows(db_path) == 0


def test_dedup_permuted_labels_same_structure(tmp_path, tracking_cfg):
    """Same regions with permuted label IDs → second is treated as duplicate and dropped."""
    from cellflow.tracking_ultrack.ingest import _canonical_hash

    # p=0: label 1 in top-left, label 2 in bottom-right
    lm_p0 = np.zeros((1, 64, 64), dtype=np.uint32)
    lm_p0[0, 0:20, 0:20] = 1
    lm_p0[0, 30:50, 30:50] = 2

    # p=1: same spatial regions but label IDs swapped (7 and 3 instead of 1 and 2)
    lm_p1 = np.zeros((1, 64, 64), dtype=np.uint32)
    lm_p1[0, 0:20, 0:20] = 7
    lm_p1[0, 30:50, 30:50] = 3

    # Confirm they canonicalize to the same hash
    h0 = _canonical_hash(lm_p0[0])
    h1 = _canonical_hash(lm_p1[0])
    assert h0 == h1, f"Expected same hash for isomorphic labelmaps, got {h0!r} != {h1!r}"

    h5_path = tmp_path / "hypotheses.h5"
    working_dir = tmp_path / "tracking"
    _create_test_h5(h5_path, {0: {0: lm_p0, 1: lm_p1}})

    ingest_hypotheses_to_db(h5_path, working_dir, tracking_cfg, overwrite=True)

    db_path = working_dir / "data.db"
    # Only p=0's 2 cells should be inserted; p=1 is a structural duplicate.
    assert _count_nodedb_rows(db_path) == 2
    assert _count_overlapdb_rows(db_path) == 0


def test_dedup_genuinely_different_partitions_both_kept(tmp_path, tracking_cfg):
    """Two genuinely different partitions → both are kept."""
    from cellflow.tracking_ultrack.ingest import _canonical_hash

    # p=0: single large cell in top-left
    lm_p0 = np.zeros((1, 64, 64), dtype=np.uint32)
    lm_p0[0, 0:30, 0:30] = 1

    # p=1: two smaller cells (different structure entirely)
    lm_p1 = np.zeros((1, 64, 64), dtype=np.uint32)
    lm_p1[0, 0:15, 0:15] = 1
    lm_p1[0, 15:30, 15:30] = 2

    # Confirm they do NOT canonicalize to the same hash
    h0 = _canonical_hash(lm_p0[0])
    h1 = _canonical_hash(lm_p1[0])
    assert h0 != h1, "Different partition structures must not hash identically"

    h5_path = tmp_path / "hypotheses.h5"
    working_dir = tmp_path / "tracking"
    _create_test_h5(h5_path, {0: {0: lm_p0, 1: lm_p1}})

    ingest_hypotheses_to_db(h5_path, working_dir, tracking_cfg, overwrite=True)

    db_path = working_dir / "data.db"
    # 1 cell (p=0) + 2 cells (p=1) = 3 nodes
    assert _count_nodedb_rows(db_path) == 3
    # p=1 cells are both contained within the p=0 cell region → 2 overlap pairs
    assert _count_overlapdb_rows(db_path) == 2


def test_dedup_all_zero_labelmaps(tmp_path, tracking_cfg):
    """Two all-zero labelmaps dedup correctly with no errors or edge cases."""
    from cellflow.tracking_ultrack.ingest import _canonical_hash

    lm_zero_a = np.zeros((1, 64, 64), dtype=np.uint32)
    lm_zero_b = np.zeros((1, 64, 64), dtype=np.uint32)

    # Two all-zero maps must hash identically
    h_a = _canonical_hash(lm_zero_a[0])
    h_b = _canonical_hash(lm_zero_b[0])
    assert h_a == h_b, "All-zero labelmaps must produce identical canonical hashes"

    h5_path = tmp_path / "hypotheses.h5"
    working_dir = tmp_path / "tracking"
    _create_test_h5(h5_path, {0: {0: lm_zero_a, 1: lm_zero_b}})

    # Should complete without div-by-zero or other exceptions
    ingest_hypotheses_to_db(h5_path, working_dir, tracking_cfg, overwrite=True)

    db_path = working_dir / "data.db"
    # No non-zero labels in either partition → 0 nodes (second is deduped anyway)
    assert _count_nodedb_rows(db_path) == 0
    assert _count_overlapdb_rows(db_path) == 0


def test_n_frames_cap(tmp_path, tracking_cfg):
    """n_frames kwarg limits ingestion to only the first N timepoints."""
    labelmap_a = np.zeros((1, 64, 64), dtype=np.uint32)
    labelmap_a[0, 0:20, 0:20] = 1

    labelmap_b = np.zeros((1, 64, 64), dtype=np.uint32)
    labelmap_b[0, 30:50, 30:50] = 2

    h5_path = tmp_path / "hypotheses.h5"
    working_dir = tmp_path / "tracking"
    # 3 timepoints, each with a single partition and one cell
    _create_test_h5(h5_path, {0: {0: labelmap_a}, 1: {0: labelmap_b}, 2: {0: labelmap_a}})

    # Ingest only the first 2 timepoints
    ingest_hypotheses_to_db(h5_path, working_dir, tracking_cfg, overwrite=True, n_frames=2)

    db_path = working_dir / "data.db"
    nodes = _get_nodedb_rows(db_path)
    assert len(nodes) == 2, f"Expected 2 nodes (one per frame), got {len(nodes)}"
    present_t = {n.t for n in nodes}
    assert present_t == {0, 1}, f"Expected t={{0,1}}, got {present_t}"


def test_overlap_encoding_late_timepoint(tmp_path, tracking_cfg):
    """Regression test for MAX_ID overflow in _compute_overlaps_vectorized.

    At t=12 with max_segments_per_time=1_000_000, node IDs start at 13_000_001,
    which exceeds the old hard-coded MAX_ID=10_000_000.  The old code produced
    silently wrong decoded pairs (hi, lo); the fixed code derives MAX_ID from
    the actual maximum node id in the frame, guaranteeing correctness.

    Setup: two partitions at t=12 only.
      p=0: cell A (pixels 0:20, 0:20)  and cell B (pixels 0:20, 30:50)  — no overlap
      p=1: cell C (pixels 5:15, 5:15)  — inside cell A => overlaps A, not B

    Expected OverlapDB: exactly 1 pair (node_id_A, node_id_C).
    Without the fix, the decoded pair ids are wrong so the test fails.
    """
    from cellflow.tracking_ultrack.ingest import _generate_id

    # Use only t=12; _generate_id(i, 12, 1_000_000) = i + 13_000_000 > 10_000_000
    max_segs = tracking_cfg.max_segments_per_time  # default 1_000_000
    TARGET_T = 12

    lm_p0 = np.zeros((1, 64, 64), dtype=np.uint32)
    lm_p0[0, 0:20, 0:20] = 1   # cell A
    lm_p0[0, 0:20, 30:50] = 2  # cell B (non-overlapping with C)

    lm_p1 = np.zeros((1, 64, 64), dtype=np.uint32)
    lm_p1[0, 5:15, 5:15] = 1   # cell C — strictly inside cell A

    h5_path = tmp_path / "hypotheses.h5"
    working_dir = tmp_path / "tracking"
    _create_test_h5(h5_path, {TARGET_T: {0: lm_p0, 1: lm_p1}})

    # min_area=0 so no cells are filtered by area
    ingest_hypotheses_to_db(h5_path, working_dir, tracking_cfg, overwrite=True, min_area=0)

    db_path = working_dir / "data.db"

    # 3 NodeDB rows: A, B (from p=0) and C (from p=1)
    assert _count_nodedb_rows(db_path) == 3

    # Node ids at t=12: p=0 cells get index=1,2; p=1 cell gets index=3
    expected_id_A = _generate_id(1, TARGET_T, max_segs)  # 13_000_001
    expected_id_B = _generate_id(2, TARGET_T, max_segs)  # 13_000_002
    expected_id_C = _generate_id(3, TARGET_T, max_segs)  # 13_000_003

    assert expected_id_A > 10_000_000, "precondition: ids must exceed old MAX_ID"

    # Exactly 1 overlap pair: (A, C). B and C do not share pixels.
    assert _count_overlapdb_rows(db_path) == 1

    overlap_rows = _get_overlapdb_rows(db_path)
    pair = tuple(sorted((overlap_rows[0].node_id, overlap_rows[0].ancestor_id)))
    expected_pair = tuple(sorted((expected_id_A, expected_id_C)))
    assert pair == expected_pair, (
        f"Wrong overlap pair {pair}; expected {expected_pair}. "
        "This indicates MAX_ID overflow in _compute_overlaps_vectorized."
    )
