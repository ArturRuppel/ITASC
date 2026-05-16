from __future__ import annotations

import numpy as np
import pytest

ultrack = pytest.importorskip("ultrack")


def test_apply_corrections_marks_validated_nodes_fake_and_anchor_nodes_real(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import Correction, apply_corrections_to_database

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        validated_candidate = _make_node_row(1, 0, 4, 4, 8, 8)
        anchor_candidate = _make_node_row(2, 0, 20, 20, 24, 24)
        far_candidate = _make_node_row(3, 0, 40, 40, 44, 44)
        session.add_all([validated_candidate, anchor_candidate, far_candidate])
        session.commit()

    corrections = [
        Correction(cell_id=7, t=0, kind="validated", y=6.0, x=6.0),
        Correction(cell_id=9, t=0, kind="anchor", y=22.0, x=22.0),
    ]

    report = apply_corrections_to_database(
        tmp_path,
        corrections,
        TrackingConfig(anchor_radius_px=5.0),
    )

    assert report.fake_nodes == 1
    assert report.anchor_nodes == 1
    with Session(engine) as session:
        rows = {row.id: row for row in session.query(NodeDB).all()}

    assert rows[1].node_annot == VarAnnotation.FAKE
    assert rows[2].node_annot == VarAnnotation.REAL
    assert rows[3].node_annot == VarAnnotation.UNKNOWN


def test_apply_corrections_uses_iou_to_pick_correct_hierarchical_candidate(tmp_path):
    """When two candidates share nearly the same centroid (parent/child hierarchy),
    centroid-distance matching is ambiguous. IoU-based matching with tracked_labels
    must select the candidate whose mask actually matches the corrected cell,
    regardless of centroid proximity."""
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import Correction, apply_corrections_to_database

    engine = _make_engine(tmp_path / "data.db")
    # Two hierarchically nested candidates at frame 0:
    # - large: rows 0-16, cols 0-16, centroid ~(8, 8)
    # - small: rows 4-12, cols 4-12, centroid ~(8, 8)  ← user wants this one
    with Session(engine) as session:
        large = _make_node_row(1, 0, 0, 0, 16, 16)
        small = _make_node_row(2, 0, 4, 4, 12, 12)
        session.add_all([large, small])
        session.commit()

    # tracked_labels shows the SMALL mask (user swapped via Z/C and saved)
    tracked = np.zeros((1, 20, 20), dtype=np.uint32)
    tracked[0, 4:12, 4:12] = 7

    corrections = [Correction(cell_id=7, t=0, kind="anchor", y=8.0, x=8.0)]
    report = apply_corrections_to_database(
        tmp_path,
        corrections,
        TrackingConfig(anchor_radius_px=5.0),
        tracked_labels=tracked,
    )

    assert report.anchor_nodes == 1
    assert report.unmatched_anchors == ()
    with Session(engine) as session:
        rows = {row.id: row for row in session.query(NodeDB).all()}

    # The SMALL candidate must be REAL; the large one must remain UNKNOWN.
    assert rows[2].node_annot == VarAnnotation.REAL
    assert rows[1].node_annot == VarAnnotation.UNKNOWN


def test_apply_corrections_prunes_overlap_between_anchor_real_nodes(tmp_path):
    """Two anchors at the same frame landing on hierarchical siblings must
    not leave a contradicting OverlapDB row (REAL+REAL <= 1 is infeasible)."""
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, OverlapDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import (
        _count_overlaps,
        _insert_overlaps,
        _make_engine,
        _make_node_row,
    )

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import Correction, apply_corrections_to_database

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        # Two hierarchy siblings at the same frame, plus an unrelated node
        # that shares an OverlapDB row with one of them (must be preserved).
        sibling_a = _make_node_row(1, 0, 10, 10, 14, 14)
        sibling_b = _make_node_row(2, 0, 20, 20, 24, 24)
        unrelated = _make_node_row(3, 0, 60, 60, 64, 64)
        session.add_all([sibling_a, sibling_b, unrelated])
        session.commit()
    _insert_overlaps(engine, [(1, 2), (1, 3)])
    assert _count_overlaps(engine) == 2

    report = apply_corrections_to_database(
        tmp_path,
        [
            Correction(cell_id=5, t=0, kind="anchor", y=12.0, x=12.0),
            Correction(cell_id=6, t=0, kind="anchor", y=22.0, x=22.0),
        ],
        TrackingConfig(anchor_radius_px=5.0),
    )

    assert report.anchor_nodes == 2
    assert report.anchor_overlaps_pruned == 1

    with Session(engine) as session:
        rows = {row.id: row for row in session.query(NodeDB).all()}
        overlap_pairs = {
            (row.node_id, row.ancestor_id) for row in session.query(OverlapDB).all()
        }

    assert rows[1].node_annot == VarAnnotation.REAL
    assert rows[2].node_annot == VarAnnotation.REAL
    # The REAL/REAL overlap row is gone; the REAL/unrelated row is kept.
    assert overlap_pairs == {(1, 3)}


def test_apply_corrections_marks_consecutive_anchor_link_real(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import Correction, apply_corrections_to_database

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        first = _make_node_row(1, 0, 10, 10, 14, 14)
        second = _make_node_row(2, 1, 12, 12, 16, 16)
        session.add_all([first, second])
        session.commit()
        session.add(LinkDB(source_id=1, target_id=2, weight=0.25))
        session.commit()

    report = apply_corrections_to_database(
        tmp_path,
        [
            Correction(cell_id=5, t=0, kind="anchor", y=12.0, x=12.0),
            Correction(cell_id=5, t=1, kind="anchor", y=14.0, x=14.0),
        ],
        TrackingConfig(anchor_radius_px=5.0),
    )

    assert report.anchor_links == 1
    with Session(engine) as session:
        link = session.query(LinkDB).one()

    assert link.annotation == VarAnnotation.REAL


def test_apply_post_solve_corrections_remaps_anchor_track_stamps_missing_anchor_and_pastes_validated():
    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import Correction, apply_post_solve_corrections

    exported = np.zeros((3, 40, 40), dtype=np.uint32)
    exported[:, 10:14, 10:14] = 99
    tracked = np.zeros_like(exported)
    tracked[1, 20:24, 20:24] = 7

    result, report = apply_post_solve_corrections(
        exported,
        [
            Correction(cell_id=5, t=0, kind="anchor", y=12.0, x=12.0),
            Correction(cell_id=6, t=2, kind="anchor", y=32.0, x=32.0),
            Correction(cell_id=7, t=1, kind="validated", y=22.0, x=22.0),
        ],
        tracked,
        TrackingConfig(anchor_radius_px=5.0, anchor_stamp_radius_px=2.0),
    )

    assert report.remapped_anchor_tracks == 1
    assert report.stamped_anchors == 1
    assert report.pasted_validated == 1
    assert 99 not in np.unique(result)
    assert np.all(result[:, 10:14, 10:14] == 5)
    assert result[2, 32, 32] == 6
    assert np.all(result[1, 20:24, 20:24] == 7)


def test_apply_post_solve_corrections_evicts_unrelated_solver_pixels_for_validated_paste():
    """A validated cell_id that collides with an unrelated solver track must
    not produce two disjoint regions sharing the same ID."""
    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import Correction, apply_post_solve_corrections

    exported = np.zeros((3, 40, 40), dtype=np.uint32)
    # Solver independently used ID=7 for an unrelated track across all frames.
    exported[:, 0:4, 0:4] = 7
    # Tracked layer has cell_id=7 at a different location at t=1.
    tracked = np.zeros_like(exported)
    tracked[1, 20:24, 20:24] = 7

    result, report = apply_post_solve_corrections(
        exported,
        [Correction(cell_id=7, t=1, kind="validated", y=22.0, x=22.0)],
        tracked,
        TrackingConfig(),
    )

    assert report.pasted_validated == 1
    # Validated geometry is at the right place with the right ID.
    assert np.all(result[1, 20:24, 20:24] == 7)
    # The unrelated solver pixels MUST NOT still carry ID=7.
    assert not (result[:, 0:4, 0:4] == 7).any()
    # And the only pixels labeled 7 anywhere are the validated mask.
    pixels_labeled_seven = (result == 7).sum()
    assert pixels_labeled_seven == int((tracked == 7).sum())


def test_apply_post_solve_corrections_evicts_unrelated_solver_pixels_for_anchor_stamp():
    """An unmatched anchor with cell_id colliding with an unrelated solver
    track must evict the solver pixels before stamping."""
    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import Correction, apply_post_solve_corrections

    exported = np.zeros((2, 40, 40), dtype=np.uint32)
    # Solver used ID=5 for an unrelated track far from the anchor location.
    exported[:, 0:4, 0:4] = 5
    tracked = np.zeros_like(exported)

    # Anchor at (30, 30) — no solver track within anchor_radius_px, so it
    # falls through to _stamp_disk.
    result, report = apply_post_solve_corrections(
        exported,
        [Correction(cell_id=5, t=0, kind="anchor", y=30.0, x=30.0)],
        tracked,
        TrackingConfig(anchor_radius_px=5.0, anchor_stamp_radius_px=2.0),
    )

    assert report.stamped_anchors == 1
    assert report.remapped_anchor_tracks == 0
    # Stamp must be present.
    assert result[0, 30, 30] == 5
    # The unrelated solver pixels MUST NOT still carry ID=5.
    assert not (result[:, 0:4, 0:4] == 5).any()


def test_apply_corrections_returns_unmatched_anchor_when_no_candidate_in_radius(tmp_path):
    """An anchor correction whose position has no NodeDB candidate within radius
    must be returned in unmatched_anchors rather than silently dropped."""
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import Correction, apply_corrections_to_database

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        far_node = _make_node_row(1, 0, 60, 60, 64, 64)
        session.add(far_node)
        session.commit()

    # Anchor position is far from the only NodeDB node (centroid ~(62,62))
    corrections = [Correction(cell_id=5, t=0, kind="anchor", y=10.0, x=10.0)]
    report = apply_corrections_to_database(
        tmp_path, corrections, TrackingConfig(anchor_radius_px=5.0)
    )

    assert report.anchor_nodes == 0
    assert len(report.unmatched_anchors) == 1
    assert report.unmatched_anchors[0] == corrections[0]
    # The far node must remain UNKNOWN (not accidentally marked REAL or FAKE)
    with Session(engine) as session:
        node = session.query(NodeDB).one()
    assert node.node_annot == VarAnnotation.UNKNOWN


def test_inject_unmatched_anchor_nodes_inserts_real_node_and_overlap_rows(tmp_path):
    """inject_unmatched_anchor_nodes must insert a REAL NodeDB row with the
    cell's mask from tracked_labels and add OverlapDB rows for overlapping
    existing candidates."""
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, OverlapDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import (
        Correction,
        inject_unmatched_anchor_nodes,
    )

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        overlapping = _make_node_row(1, 0, 2, 2, 8, 8)
        non_overlapping = _make_node_row(2, 0, 50, 50, 60, 60)
        session.add_all([overlapping, non_overlapping])
        session.commit()

    # Manually-drawn cell at (0:10, 0:10) in tracked_labels
    tracked = np.zeros((1, 20, 20), dtype=np.uint32)
    tracked[0, 0:10, 0:10] = 99

    report = inject_unmatched_anchor_nodes(
        tmp_path,
        (Correction(cell_id=99, t=0, kind="anchor", y=5.0, x=5.0),),
        tracked,
        TrackingConfig(max_segments_per_time=1000),
    )

    assert report.injected == 1
    assert report.skipped_no_mask == 0

    with Session(engine) as session:
        nodes = {row.id: row for row in session.query(NodeDB).all()}
        overlaps = {
            (int(row.node_id), int(row.ancestor_id))
            for row in session.query(OverlapDB).all()
        }

    injected_ids = {nid for nid, row in nodes.items() if row.node_annot == VarAnnotation.REAL}
    assert len(injected_ids) == 1
    injected_id = next(iter(injected_ids))

    # Injected node must overlap with existing node 1 (same region), but not node 2
    assert any(injected_id in pair for pair in overlaps)
    overlapping_partners = {p[1] if p[0] == injected_id else p[0] for p in overlaps if injected_id in p}
    assert 1 in overlapping_partners
    assert 2 not in overlapping_partners


def test_inject_unmatched_anchor_nodes_skips_when_no_mask_in_tracked_labels(tmp_path):
    """If the anchor's cell_id is not in tracked_labels at frame t, skip it."""
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import (
        Correction,
        inject_unmatched_anchor_nodes,
    )

    engine = _make_engine(tmp_path / "data.db")
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB
    with Session(engine) as session:
        session.add(_make_node_row(1, 0, 0, 0, 5, 5))
        session.commit()

    tracked = np.zeros((1, 20, 20), dtype=np.uint32)
    # cell_id=99 is absent from tracked_labels

    report = inject_unmatched_anchor_nodes(
        tmp_path,
        (Correction(cell_id=99, t=0, kind="anchor", y=5.0, x=5.0),),
        tracked,
        TrackingConfig(),
    )

    assert report.injected == 0
    assert report.skipped_no_mask == 1


def test_apply_corrections_preserves_matched_anchor_when_solver_id_equals_target():
    """When the solver already labeled the matched anchor's track with the
    correct ID, the track must be preserved untouched (no eviction)."""
    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import Correction, apply_post_solve_corrections

    exported = np.zeros((2, 40, 40), dtype=np.uint32)
    # Solver track with ID=5 is at the anchor location.
    exported[:, 20:24, 20:24] = 5
    tracked = np.zeros_like(exported)

    result, report = apply_post_solve_corrections(
        exported,
        [Correction(cell_id=5, t=0, kind="anchor", y=22.0, x=22.0)],
        tracked,
        TrackingConfig(anchor_radius_px=5.0, anchor_stamp_radius_px=2.0),
    )

    assert report.remapped_anchor_tracks == 0
    assert report.stamped_anchors == 0
    # Full track preserved.
    assert np.all(result[:, 20:24, 20:24] == 5)
