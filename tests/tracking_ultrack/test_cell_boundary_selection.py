from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import tifffile

from cellflow.tracking_ultrack.cell_boundary_selection import (
    BoundaryCandidate,
    BoundarySelectionParams,
    detect_overlap_conflicts,
    export_selected_boundaries,
    is_candidate_eligible_for_track,
    load_candidates_from_db,
    run_track_conditioned_boundary_selection,
    select_all_track_boundaries,
    select_track_boundaries_dp,
    validate_cell_boundary_inputs,
)


def test_candidate_requires_nucleus_anchor_for_requested_track():
    nuclei = np.zeros((8, 8), dtype=np.uint32)
    nuclei[2:4, 2:4] = 7
    nuclei[5:7, 5:7] = 9

    anchored = BoundaryCandidate(
        node_id=11,
        t=0,
        mask=np.ones((4, 4), dtype=bool),
        bbox=(1, 1, 5, 5),
        score=1.0,
    )
    unanchored = BoundaryCandidate(
        node_id=12,
        t=0,
        mask=np.ones((2, 2), dtype=bool),
        bbox=(0, 5, 2, 7),
        score=10.0,
    )

    assert is_candidate_eligible_for_track(anchored, nuclei, 7)
    assert not is_candidate_eligible_for_track(unanchored, nuclei, 7)


def test_candidate_anchor_fraction_threshold_is_configurable():
    nuclei = np.zeros((8, 8), dtype=np.uint32)
    nuclei[2:6, 2:6] = 7
    partial = BoundaryCandidate(
        node_id=20,
        t=0,
        mask=np.ones((2, 2), dtype=bool),
        bbox=(2, 2, 4, 4),
        score=1.0,
    )

    assert is_candidate_eligible_for_track(partial, nuclei, 7)
    assert not is_candidate_eligible_for_track(
        partial,
        nuclei,
        7,
        min_nucleus_fraction=0.5,
    )


def test_dp_prefers_temporally_smooth_path_over_jumpy_high_unary_candidate():
    nuclei = np.zeros((3, 12, 12), dtype=np.uint32)
    nuclei[:, 4:6, 4:6] = 7
    candidates = [
        BoundaryCandidate(1, 0, np.ones((5, 5), bool), (2, 2, 7, 7), score=1.0),
        BoundaryCandidate(2, 1, np.ones((5, 5), bool), (2, 2, 7, 7), score=1.0),
        BoundaryCandidate(3, 2, np.ones((5, 5), bool), (2, 2, 7, 7), score=1.0),
        BoundaryCandidate(99, 1, np.ones((4, 4), bool), (4, 4, 8, 8), score=8.0),
    ]

    result = select_track_boundaries_dp(
        candidates,
        nuclei,
        track_id=7,
        params=BoundarySelectionParams(
            centroid_jump_weight=2.5,
            area_change_weight=0.0,
            iou_loss_weight=2.0,
            missing_penalty=20.0,
        ),
    )

    assert result.selected_node_ids == {0: 1, 1: 2, 2: 3}
    assert result.missing_frames == set()


def test_dp_uses_missing_state_when_no_eligible_candidate_exists():
    nuclei = np.zeros((3, 10, 10), dtype=np.uint32)
    nuclei[:, 4:6, 4:6] = 7
    candidates = [
        BoundaryCandidate(1, 0, np.ones((4, 4), bool), (3, 3, 7, 7), score=1.0),
        BoundaryCandidate(2, 1, np.ones((2, 2), bool), (0, 0, 2, 2), score=50.0),
        BoundaryCandidate(3, 2, np.ones((4, 4), bool), (3, 3, 7, 7), score=1.0),
    ]

    result = select_track_boundaries_dp(
        candidates,
        nuclei,
        track_id=7,
        params=BoundarySelectionParams(missing_penalty=5.0),
    )

    assert result.selected_node_ids == {0: 1, 1: None, 2: 3}
    assert result.missing_frames == {1}


def test_dp_uses_missing_state_when_all_eligible_candidates_are_too_poor():
    nuclei = np.zeros((1, 10, 10), dtype=np.uint32)
    nuclei[0, 4:6, 4:6] = 7
    candidates = [
        BoundaryCandidate(1, 0, np.ones((4, 4), bool), (3, 3, 7, 7), score=-100.0),
    ]

    result = select_track_boundaries_dp(
        candidates,
        nuclei,
        track_id=7,
        params=BoundarySelectionParams(missing_penalty=5.0),
    )

    assert result.selected_node_ids == {0: None}
    assert result.missing_frames == {0}


def test_overlap_conflict_diagnostics_report_selected_mask_intersections():
    selections = {
        7: {0: 1},
        9: {0: 2},
    }
    candidates = {
        1: BoundaryCandidate(1, 0, np.ones((4, 4), bool), (1, 1, 5, 5), score=1.0),
        2: BoundaryCandidate(2, 0, np.ones((4, 4), bool), (3, 3, 7, 7), score=1.0),
    }

    conflicts = detect_overlap_conflicts(selections, candidates, min_overlap_pixels=1)

    assert len(conflicts) == 1
    assert conflicts[0].t == 0
    assert conflicts[0].track_ids == (7, 9)
    assert conflicts[0].node_ids == (1, 2)
    assert conflicts[0].overlap_pixels == 4


def test_export_selected_boundaries_preserves_nucleus_track_ids():
    selections = {
        7: {0: 1, 1: 2},
        9: {0: 3},
    }
    candidates = {
        1: BoundaryCandidate(1, 0, np.ones((3, 3), bool), (1, 1, 4, 4), score=1.0),
        2: BoundaryCandidate(2, 1, np.ones((3, 3), bool), (2, 2, 5, 5), score=1.0),
        3: BoundaryCandidate(3, 0, np.ones((2, 2), bool), (6, 6, 8, 8), score=1.0),
    }

    labels = export_selected_boundaries(selections, candidates, shape=(2, 10, 10))

    assert labels.dtype == np.uint32
    assert set(np.unique(labels)) == {0, 7, 9}
    assert (labels[0, 1:4, 1:4] == 7).all()
    assert (labels[1, 2:5, 2:5] == 7).all()
    assert (labels[0, 6:8, 6:8] == 9).all()


def test_select_all_tracks_and_export_preserves_connected_track_ids():
    nuclei = np.zeros((2, 12, 12), dtype=np.uint32)
    nuclei[:, 2:4, 2:4] = 7
    nuclei[:, 7:9, 7:9] = 9
    candidates = [
        BoundaryCandidate(1, 0, np.ones((4, 4), bool), (1, 1, 5, 5), score=1.0),
        BoundaryCandidate(2, 1, np.ones((4, 4), bool), (1, 1, 5, 5), score=1.0),
        BoundaryCandidate(3, 0, np.ones((4, 4), bool), (6, 6, 10, 10), score=1.0),
        BoundaryCandidate(4, 1, np.ones((4, 4), bool), (6, 6, 10, 10), score=1.0),
    ]

    results = select_all_track_boundaries(candidates, nuclei)
    selections = {
        track_id: result.selected_node_ids
        for track_id, result in results.items()
    }
    labels = export_selected_boundaries(
        selections,
        {candidate.node_id: candidate for candidate in candidates},
        shape=nuclei.shape,
    )

    assert set(results) == {7, 9}
    assert results[7].missing_frames == set()
    assert results[9].missing_frames == set()
    assert set(np.unique(labels)) == {0, 7, 9}
    for t in range(2):
        assert (labels[t, 1:5, 1:5] == 7).all()
        assert (labels[t, 6:10, 6:10] == 9).all()


def test_export_rejects_candidate_bbox_outside_output_shape():
    selections = {7: {0: 1}}
    candidates = {
        1: BoundaryCandidate(1, 0, np.ones((3, 3), bool), (8, 8, 11, 11), score=1.0),
    }

    try:
        export_selected_boundaries(selections, candidates, shape=(1, 10, 10))
    except ValueError as exc:
        assert "outside output shape" in str(exc)
    else:
        raise AssertionError("expected export to reject out-of-bounds bbox")


# ---------------------------------------------------------------------------
# Step 1: load_candidates_from_db
# ---------------------------------------------------------------------------


def _make_db_engine(db_path: Path):
    sqla = pytest.importorskip("sqlalchemy")
    Base = pytest.importorskip("ultrack.core.database").Base

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine


def _make_node_db_row(node_id: int, t: int, y0: int, x0: int, y1: int, x1: int,
                      node_prob: float | None = None):
    """Build a NodeDB row with a (1, h, w) bool mask crop."""
    NodeDB = pytest.importorskip("ultrack.core.database").NodeDB
    Node = pytest.importorskip("ultrack.core.segmentation.node").Node

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
        node_prob=node_prob,
    )


def test_load_candidates_from_db_creates_boundary_candidates_with_mask_and_bbox(tmp_path):
    pytest.importorskip("sqlalchemy")
    Session = pytest.importorskip("sqlalchemy.orm").Session

    engine = _make_db_engine(tmp_path / "data.db")
    with Session(engine) as session:
        session.add_all([
            _make_node_db_row(101, 0, 2, 3, 7, 8),
            _make_node_db_row(102, 1, 5, 5, 15, 15),
        ])
        session.commit()
    engine.dispose()

    candidates = load_candidates_from_db(tmp_path)

    assert len(candidates) == 2
    by_id = {c.node_id: c for c in candidates}

    c101 = by_id[101]
    assert c101.t == 0
    assert c101.bbox == (2, 3, 7, 8)
    assert c101.mask.shape == (5, 5)
    assert c101.mask.dtype == bool
    assert c101.mask.all()

    c102 = by_id[102]
    assert c102.t == 1
    assert c102.bbox == (5, 5, 15, 15)
    assert c102.mask.shape == (10, 10)


def test_load_candidates_uses_node_prob_as_score(tmp_path):
    pytest.importorskip("sqlalchemy")
    Session = pytest.importorskip("sqlalchemy.orm").Session

    engine = _make_db_engine(tmp_path / "data.db")
    with Session(engine) as session:
        session.add(_make_node_db_row(1, 0, 0, 0, 5, 5, node_prob=0.75))
        session.commit()
    engine.dispose()

    candidates = load_candidates_from_db(tmp_path)
    assert len(candidates) == 1
    assert candidates[0].score == 0.75


def test_load_candidates_defaults_score_to_zero_when_node_prob_is_none(tmp_path):
    pytest.importorskip("sqlalchemy")
    Session = pytest.importorskip("sqlalchemy.orm").Session

    engine = _make_db_engine(tmp_path / "data.db")
    with Session(engine) as session:
        session.add(_make_node_db_row(1, 0, 0, 0, 5, 5, node_prob=None))
        session.commit()
    engine.dispose()

    candidates = load_candidates_from_db(tmp_path)
    assert len(candidates) == 1
    assert candidates[0].score == 0.0


def test_load_candidates_returns_empty_list_for_empty_database(tmp_path):
    pytest.importorskip("sqlalchemy")

    _make_db_engine(tmp_path / "data.db").dispose()

    candidates = load_candidates_from_db(tmp_path)
    assert candidates == []


def test_load_candidates_handles_missing_data_db(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candidates_from_db(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Step 2: validate_cell_boundary_inputs
# ---------------------------------------------------------------------------


def test_validate_inputs_accepts_matching_3d_tiffs(tmp_path):
    contour_path = tmp_path / "contours.tif"
    fg_path = tmp_path / "foreground.tif"
    nuc_path = tmp_path / "nuclei.tif"

    shape = (3, 32, 32)
    tifffile.imwrite(str(contour_path), np.ones(shape, dtype=np.float32))
    tifffile.imwrite(str(fg_path), np.ones(shape, dtype=np.float32))
    tifffile.imwrite(str(nuc_path), np.ones(shape, dtype=np.uint32))

    contours, fg, nuclei = validate_cell_boundary_inputs(
        contour_path, fg_path, nuc_path,
    )
    assert contours.shape == shape
    assert fg.shape == shape
    assert nuclei.shape == shape


def test_validate_inputs_raises_when_path_missing(tmp_path):
    contour_path = tmp_path / "contours.tif"
    fg_path = tmp_path / "foreground.tif"
    nuc_path = tmp_path / "nuclei.tif"

    tifffile.imwrite(str(contour_path), np.ones((1, 8, 8), dtype=np.float32))
    tifffile.imwrite(str(fg_path), np.ones((1, 8, 8), dtype=np.float32))
    # nuclei path does not exist

    with pytest.raises(FileNotFoundError):
        validate_cell_boundary_inputs(contour_path, fg_path, nuc_path)


def test_validate_inputs_normalizes_4d_singleton_z_to_3d(tmp_path):
    contour_path = tmp_path / "contours.tif"
    fg_path = tmp_path / "foreground.tif"
    nuc_path = tmp_path / "nuclei.tif"

    # 4D with singleton Z: (T, 1, Y, X)
    shape_4d = (2, 1, 16, 16)
    tifffile.imwrite(str(contour_path), np.ones(shape_4d, dtype=np.float32))
    tifffile.imwrite(str(fg_path), np.ones(shape_4d, dtype=np.float32))
    tifffile.imwrite(str(nuc_path), np.ones(shape_4d, dtype=np.uint32))

    contours, fg, nuclei = validate_cell_boundary_inputs(
        contour_path, fg_path, nuc_path,
    )
    expected = (2, 16, 16)
    assert contours.shape == expected
    assert fg.shape == expected
    assert nuclei.shape == expected


def test_validate_inputs_normalizes_2d_to_3d(tmp_path):
    contour_path = tmp_path / "contours.tif"
    fg_path = tmp_path / "foreground.tif"
    nuc_path = tmp_path / "nuclei.tif"

    shape_2d = (16, 16)
    tifffile.imwrite(str(contour_path), np.ones(shape_2d, dtype=np.float32))
    tifffile.imwrite(str(fg_path), np.ones(shape_2d, dtype=np.float32))
    tifffile.imwrite(str(nuc_path), np.ones(shape_2d, dtype=np.uint32))

    contours, fg, nuclei = validate_cell_boundary_inputs(
        contour_path, fg_path, nuc_path,
    )
    expected = (1, 16, 16)
    assert contours.shape == expected
    assert fg.shape == expected
    assert nuclei.shape == expected


def test_validate_inputs_raises_on_shape_mismatch(tmp_path):
    contour_path = tmp_path / "contours.tif"
    fg_path = tmp_path / "foreground.tif"
    nuc_path = tmp_path / "nuclei.tif"

    tifffile.imwrite(str(contour_path), np.ones((3, 32, 32), dtype=np.float32))
    tifffile.imwrite(str(fg_path), np.ones((3, 32, 32), dtype=np.float32))
    tifffile.imwrite(str(nuc_path), np.ones((4, 32, 32), dtype=np.uint32))

    with pytest.raises(ValueError, match="shape"):
        validate_cell_boundary_inputs(contour_path, fg_path, nuc_path)


def test_validate_inputs_raises_on_unsupported_ndim(tmp_path):
    contour_path = tmp_path / "contours.tif"
    fg_path = tmp_path / "foreground.tif"
    nuc_path = tmp_path / "nuclei.tif"

    tifffile.imwrite(str(contour_path), np.ones((1, 2, 3, 4, 5), dtype=np.float32))
    tifffile.imwrite(str(fg_path), np.ones((1, 2, 3, 4, 5), dtype=np.float32))
    tifffile.imwrite(str(nuc_path), np.ones((1, 2, 3, 4, 5), dtype=np.uint32))

    with pytest.raises(ValueError, match="ndim"):
        validate_cell_boundary_inputs(contour_path, fg_path, nuc_path)


def test_validate_inputs_raises_on_4d_non_singleton_z(tmp_path):
    contour_path = tmp_path / "contours.tif"
    fg_path = tmp_path / "foreground.tif"
    nuc_path = tmp_path / "nuclei.tif"

    # 4D with Z > 1
    shape_4d = (2, 3, 16, 16)
    tifffile.imwrite(str(contour_path), np.ones(shape_4d, dtype=np.float32))
    tifffile.imwrite(str(fg_path), np.ones(shape_4d, dtype=np.float32))
    tifffile.imwrite(str(nuc_path), np.ones(shape_4d, dtype=np.uint32))

    with pytest.raises(ValueError, match="Z"):
        validate_cell_boundary_inputs(contour_path, fg_path, nuc_path)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def test_run_track_conditioned_boundary_selection_writes_labels_and_diagnostics(tmp_path):
    pos_dir = tmp_path / "pos00"
    cell_dir = pos_dir / "3_cell"
    nucleus_dir = pos_dir / "2_nucleus"
    workdir = cell_dir / "ultrack_workdir"
    cell_dir.mkdir(parents=True)
    nucleus_dir.mkdir(parents=True)
    workdir.mkdir()

    nuclei = np.zeros((2, 10, 10), dtype=np.uint32)
    nuclei[0, 2:4, 2:4] = 7
    nuclei[1, 3:5, 3:5] = 7
    nuclei[0, 6:8, 6:8] = 9

    tifffile.imwrite(cell_dir / "contour_maps.tif", np.zeros(nuclei.shape, dtype=np.float32))
    tifffile.imwrite(cell_dir / "foreground_masks.tif", np.ones(nuclei.shape, dtype=np.uint8))
    tifffile.imwrite(nucleus_dir / "tracked_labels.tif", nuclei)

    candidates = [
        BoundaryCandidate(1, 0, np.ones((4, 4), bool), (1, 1, 5, 5), score=1.0),
        BoundaryCandidate(2, 1, np.ones((4, 4), bool), (2, 2, 6, 6), score=1.0),
        BoundaryCandidate(3, 0, np.ones((3, 3), bool), (5, 5, 8, 8), score=1.0),
    ]

    result = run_track_conditioned_boundary_selection(
        pos_dir,
        candidate_loader=lambda path: candidates,
        params=BoundarySelectionParams(missing_penalty=3.0),
    )

    labels = tifffile.imread(cell_dir / "tracked_labels.tif")
    assert labels.dtype == np.uint32
    assert set(np.unique(labels)) == {0, 7, 9}
    assert (labels[0, 1:5, 1:5] == 7).all()
    assert (labels[1, 2:6, 2:6] == 7).all()
    assert (labels[0, 5:8, 5:8] == 9).all()

    assert result.output_path == cell_dir / "tracked_labels.tif"
    assert result.candidate_count == 3
    assert result.missing_frames_by_track == {7: [], 9: []}
    assert result.diagnostics_path == cell_dir / "boundary_selection_diagnostics.json"
    assert result.diagnostics_path.exists()
    assert '"candidate_count": 3' in result.diagnostics_path.read_text()


def test_run_track_conditioned_boundary_selection_reports_missing_frames_and_conflicts(tmp_path):
    pos_dir = tmp_path / "pos00"
    cell_dir = pos_dir / "3_cell"
    nucleus_dir = pos_dir / "2_nucleus"
    (cell_dir / "ultrack_workdir").mkdir(parents=True)
    nucleus_dir.mkdir(parents=True)

    nuclei = np.zeros((2, 10, 10), dtype=np.uint32)
    nuclei[:, 2:4, 2:4] = 7
    nuclei[0, 4:6, 4:6] = 9

    tifffile.imwrite(cell_dir / "contour_maps.tif", np.zeros(nuclei.shape, dtype=np.float32))
    tifffile.imwrite(cell_dir / "foreground_masks.tif", np.ones(nuclei.shape, dtype=np.uint8))
    tifffile.imwrite(nucleus_dir / "tracked_labels.tif", nuclei)

    candidates = [
        BoundaryCandidate(1, 0, np.ones((5, 5), bool), (1, 1, 6, 6), score=1.0),
        BoundaryCandidate(2, 0, np.ones((4, 4), bool), (3, 3, 7, 7), score=1.0),
    ]

    result = run_track_conditioned_boundary_selection(
        pos_dir,
        candidate_loader=lambda path: candidates,
        params=BoundarySelectionParams(missing_penalty=3.0),
    )

    assert result.missing_frames_by_track == {7: [1], 9: []}
    assert len(result.overlap_conflicts) == 1
    assert result.overlap_conflicts[0].track_ids == (7, 9)
    assert result.overlap_conflicts[0].overlap_pixels == 9

    diagnostics = json.loads(result.diagnostics_path.read_text())
    assert diagnostics["missing_frames_by_track"] == {"7": [1], "9": []}
    assert diagnostics["conflicts"] == [
        {
            "t": 0,
            "track_ids": [7, 9],
            "node_ids": [1, 2],
            "overlap_pixels": 9,
        }
    ]
