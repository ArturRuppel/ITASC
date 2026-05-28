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


def test_annotate_anchor_tail_links_marks_best_successor_real(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import (
        Correction,
        annotate_anchor_tail_links,
        apply_corrections_to_database,
    )

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        anchor = _make_node_row(1, 0, 10, 10, 14, 14)
        good = _make_node_row(2, 1, 11, 11, 15, 15)
        bad = _make_node_row(3, 1, 35, 35, 39, 39)
        session.add_all([anchor, good, bad])
        session.commit()
        session.add_all(
            [
                LinkDB(source_id=1, target_id=2, weight=1.25),
                LinkDB(source_id=1, target_id=3, weight=-2.0),
            ]
        )
        session.commit()

    tracked = np.zeros((2, 50, 50), dtype=np.uint32)
    tracked[0, 10:14, 10:14] = 5
    correction = Correction(cell_id=5, t=0, kind="anchor", y=12.0, x=12.0)
    cfg = TrackingConfig(anchor_radius_px=5.0, max_distance=40.0)

    apply_corrections_to_database(
        tmp_path,
        [correction],
        cfg,
        tracked_labels=tracked,
    )
    report = annotate_anchor_tail_links(
        tmp_path,
        [correction],
        cfg,
        tracked_labels=tracked,
    )

    assert report.annotated == 1
    with Session(engine) as session:
        links = {
            int(row.target_id): row.annotation
            for row in session.query(LinkDB).where(LinkDB.source_id == 1)
        }

    assert links[2] == VarAnnotation.REAL
    assert links[3] == VarAnnotation.UNKNOWN


def test_annotate_anchor_tail_links_walks_successor_chain(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import (
        Correction,
        annotate_anchor_tail_links,
        apply_corrections_to_database,
    )

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        anchor = _make_node_row(1, 0, 10, 10, 14, 14)
        frame_1_good = _make_node_row(2, 1, 11, 11, 15, 15)
        frame_2_good = _make_node_row(3, 2, 12, 12, 16, 16)
        frame_2_bad = _make_node_row(4, 2, 35, 35, 39, 39)
        session.add_all([anchor, frame_1_good, frame_2_good, frame_2_bad])
        session.commit()
        session.add_all(
            [
                LinkDB(source_id=1, target_id=2, weight=1.5),
                LinkDB(source_id=2, target_id=3, weight=1.25),
                LinkDB(source_id=2, target_id=4, weight=-1.0),
            ]
        )
        session.commit()

    tracked = np.zeros((3, 50, 50), dtype=np.uint32)
    tracked[0, 10:14, 10:14] = 5
    correction = Correction(cell_id=5, t=0, kind="anchor", y=12.0, x=12.0)
    cfg = TrackingConfig(anchor_radius_px=5.0, max_distance=40.0)

    apply_corrections_to_database(
        tmp_path,
        [correction],
        cfg,
        tracked_labels=tracked,
    )
    report = annotate_anchor_tail_links(
        tmp_path,
        [correction],
        cfg,
        tracked_labels=tracked,
    )

    assert report.annotated == 2
    with Session(engine) as session:
        links = {
            (int(row.source_id), int(row.target_id)): row.annotation
            for row in session.query(LinkDB).all()
        }

    assert links[(1, 2)] == VarAnnotation.REAL
    assert links[(2, 3)] == VarAnnotation.REAL
    assert links[(2, 4)] == VarAnnotation.UNKNOWN


def test_annotate_anchor_tail_links_walks_predecessor_chain(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import (
        Correction,
        annotate_anchor_tail_links,
        apply_corrections_to_database,
    )

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        frame_0_good = _make_node_row(1, 0, 10, 10, 14, 14)
        frame_1_good = _make_node_row(2, 1, 11, 11, 15, 15)
        anchor = _make_node_row(3, 2, 12, 12, 16, 16)
        frame_1_bad = _make_node_row(4, 1, 35, 35, 39, 39)
        session.add_all([frame_0_good, frame_1_good, anchor, frame_1_bad])
        session.commit()
        session.add_all(
            [
                LinkDB(source_id=1, target_id=2, weight=1.25),
                LinkDB(source_id=2, target_id=3, weight=1.5),
                LinkDB(source_id=4, target_id=3, weight=-1.0),
            ]
        )
        session.commit()

    tracked = np.zeros((3, 50, 50), dtype=np.uint32)
    tracked[2, 12:16, 12:16] = 5
    correction = Correction(cell_id=5, t=2, kind="anchor", y=14.0, x=14.0)
    cfg = TrackingConfig(anchor_radius_px=5.0, max_distance=40.0)

    apply_corrections_to_database(
        tmp_path,
        [correction],
        cfg,
        tracked_labels=tracked,
    )
    report = annotate_anchor_tail_links(
        tmp_path,
        [correction],
        cfg,
        tracked_labels=tracked,
    )

    assert report.annotated == 2
    with Session(engine) as session:
        links = {
            (int(row.source_id), int(row.target_id)): row.annotation
            for row in session.query(LinkDB).all()
        }

    assert links[(1, 2)] == VarAnnotation.REAL
    assert links[(2, 3)] == VarAnnotation.REAL
    assert links[(4, 3)] == VarAnnotation.UNKNOWN


def test_annotate_anchor_tail_links_avoids_converging_anchor_lineages(tmp_path):
    from collections import Counter

    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import (
        Correction,
        annotate_anchor_tail_links,
        apply_corrections_to_database,
    )

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        first_anchor = _make_node_row(1, 0, 10, 10, 14, 14)
        second_anchor = _make_node_row(2, 0, 30, 30, 34, 34)
        common_best = _make_node_row(3, 1, 20, 20, 24, 24)
        second_fallback = _make_node_row(4, 1, 31, 31, 35, 35)
        session.add_all([first_anchor, second_anchor, common_best, second_fallback])
        session.commit()
        session.add_all(
            [
                LinkDB(source_id=1, target_id=3, weight=2.0),
                LinkDB(source_id=2, target_id=3, weight=2.0),
                LinkDB(source_id=2, target_id=4, weight=1.5),
            ]
        )
        session.commit()

    tracked = np.zeros((2, 50, 50), dtype=np.uint32)
    tracked[0, 10:14, 10:14] = 5
    tracked[0, 30:34, 30:34] = 6
    corrections = [
        Correction(cell_id=5, t=0, kind="anchor", y=12.0, x=12.0),
        Correction(cell_id=6, t=0, kind="anchor", y=32.0, x=32.0),
    ]
    cfg = TrackingConfig(anchor_radius_px=5.0, max_distance=40.0)

    apply_corrections_to_database(
        tmp_path,
        corrections,
        cfg,
        tracked_labels=tracked,
    )
    report = annotate_anchor_tail_links(
        tmp_path,
        corrections,
        cfg,
        tracked_labels=tracked,
    )

    assert report.annotated == 2
    with Session(engine) as session:
        real_links = [
            (int(row.source_id), int(row.target_id))
            for row in session.query(LinkDB).where(
                LinkDB.annotation == VarAnnotation.REAL
            )
        ]

    incoming = Counter(target_id for _source_id, target_id in real_links)
    assert incoming[3] == 1
    assert incoming[4] == 1


def test_annotate_anchor_tail_links_avoids_overlapping_forced_successors(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, OverlapDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import (
        Correction,
        annotate_anchor_tail_links,
        apply_corrections_to_database,
    )

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        first_anchor = _make_node_row(1, 0, 10, 10, 14, 14)
        second_anchor = _make_node_row(2, 0, 30, 30, 34, 34)
        first_target = _make_node_row(3, 1, 20, 20, 24, 24)
        overlapping_target = _make_node_row(4, 1, 21, 21, 25, 25)
        fallback_target = _make_node_row(5, 1, 31, 31, 35, 35)
        session.add_all(
            [
                first_anchor,
                second_anchor,
                first_target,
                overlapping_target,
                fallback_target,
            ]
        )
        session.commit()
        session.add_all(
            [
                LinkDB(source_id=1, target_id=3, weight=2.0),
                LinkDB(source_id=2, target_id=4, weight=2.0),
                LinkDB(source_id=2, target_id=5, weight=1.5),
                OverlapDB(node_id=4, ancestor_id=3),
            ]
        )
        session.commit()

    tracked = np.zeros((2, 50, 50), dtype=np.uint32)
    tracked[0, 10:14, 10:14] = 5
    tracked[0, 30:34, 30:34] = 6
    corrections = [
        Correction(cell_id=5, t=0, kind="anchor", y=12.0, x=12.0),
        Correction(cell_id=6, t=0, kind="anchor", y=32.0, x=32.0),
    ]
    cfg = TrackingConfig(anchor_radius_px=5.0, max_distance=40.0)

    apply_corrections_to_database(
        tmp_path,
        corrections,
        cfg,
        tracked_labels=tracked,
    )
    report = annotate_anchor_tail_links(
        tmp_path,
        corrections,
        cfg,
        tracked_labels=tracked,
    )

    assert report.annotated == 2
    with Session(engine) as session:
        links = {
            (int(row.source_id), int(row.target_id)): row.annotation
            for row in session.query(LinkDB).all()
        }

    assert links[(1, 3)] == VarAnnotation.REAL
    assert links[(2, 4)] == VarAnnotation.UNKNOWN
    assert links[(2, 5)] == VarAnnotation.REAL


def test_apply_post_solve_corrections_remaps_anchor_track_stamps_missing_anchor_and_pastes_validated():
    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import (
        Correction,
        apply_post_solve_corrections,
    )

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


def test_apply_post_solve_corrections_remaps_real_link_descendant_tracklet(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, NO_PARENT, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import (
        Correction,
        apply_post_solve_corrections,
    )
    from cellflow.tracking_ultrack.ingest import _build_ultrack_config
    from ultrack.core.export import to_tracks_layer

    engine = _make_engine(tmp_path / "data.db")
    (tmp_path / "metadata.toml").write_text("shape = [ 2, 40, 40,]\n")
    with Session(engine) as session:
        anchor = _make_node_row(1, 0, 10, 10, 14, 14)
        continuation = _make_node_row(2, 1, 12, 12, 16, 16)
        sibling_branch = _make_node_row(3, 1, 28, 28, 32, 32)
        anchor.selected = True
        anchor.parent_id = NO_PARENT
        continuation.selected = True
        continuation.parent_id = 1
        sibling_branch.selected = True
        sibling_branch.parent_id = 1
        session.add_all([anchor, continuation, sibling_branch])
        session.commit()
        session.add(
            LinkDB(
                source_id=1,
                target_id=2,
                weight=1.0,
                annotation=VarAnnotation.REAL,
            )
        )
        session.commit()

    cfg = TrackingConfig()
    tracks_df, _graph = to_tracks_layer(_build_ultrack_config(cfg, tmp_path))
    anchor_track = int(tracks_df.loc[tracks_df["id"] == 1, "track_id"].iloc[0])
    continuation_track = int(tracks_df.loc[tracks_df["id"] == 2, "track_id"].iloc[0])
    sibling_track = int(tracks_df.loc[tracks_df["id"] == 3, "track_id"].iloc[0])
    assert continuation_track != anchor_track

    exported = np.zeros((2, 40, 40), dtype=np.uint32)
    exported[0, 10:14, 10:14] = anchor_track
    exported[1, 12:16, 12:16] = continuation_track
    exported[1, 28:32, 28:32] = sibling_track
    tracked = np.zeros_like(exported)
    tracked[0, 10:14, 10:14] = 5

    result, report = apply_post_solve_corrections(
        exported,
        [Correction(cell_id=5, t=0, kind="anchor", y=12.0, x=12.0)],
        tracked,
        cfg,
        working_dir=tmp_path,
    )

    assert report.remapped_anchor_tracks == 2
    assert np.all(result[0, 10:14, 10:14] == 5)
    assert np.all(result[1, 12:16, 12:16] == 5)
    assert not (result[1, 28:32, 28:32] == 5).any()


def test_apply_post_solve_corrections_remaps_real_link_ancestor_tracklet(tmp_path):
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, NO_PARENT, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row
    from ultrack.core.export import to_tracks_layer

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import (
        Correction,
        apply_post_solve_corrections,
    )
    from cellflow.tracking_ultrack.ingest import _build_ultrack_config

    engine = _make_engine(tmp_path / "data.db")
    (tmp_path / "metadata.toml").write_text("shape = [ 2, 40, 40,]\n")
    with Session(engine) as session:
        predecessor = _make_node_row(1, 0, 10, 10, 14, 14)
        anchor = _make_node_row(2, 1, 12, 12, 16, 16)
        sibling_branch = _make_node_row(3, 1, 28, 28, 32, 32)
        predecessor.selected = True
        predecessor.parent_id = NO_PARENT
        anchor.selected = True
        anchor.parent_id = 1
        sibling_branch.selected = True
        sibling_branch.parent_id = 1
        session.add_all([predecessor, anchor, sibling_branch])
        session.commit()
        session.add(
            LinkDB(
                source_id=1,
                target_id=2,
                weight=1.0,
                annotation=VarAnnotation.REAL,
            )
        )
        session.commit()

    cfg = TrackingConfig()
    tracks_df, _graph = to_tracks_layer(_build_ultrack_config(cfg, tmp_path))
    predecessor_track = int(tracks_df.loc[tracks_df["id"] == 1, "track_id"].iloc[0])
    anchor_track = int(tracks_df.loc[tracks_df["id"] == 2, "track_id"].iloc[0])
    sibling_track = int(tracks_df.loc[tracks_df["id"] == 3, "track_id"].iloc[0])
    assert predecessor_track != anchor_track

    exported = np.zeros((2, 40, 40), dtype=np.uint32)
    exported[0, 10:14, 10:14] = predecessor_track
    exported[1, 12:16, 12:16] = anchor_track
    exported[1, 28:32, 28:32] = sibling_track
    tracked = np.zeros_like(exported)
    tracked[1, 12:16, 12:16] = 5

    result, report = apply_post_solve_corrections(
        exported,
        [Correction(cell_id=5, t=1, kind="anchor", y=14.0, x=14.0)],
        tracked,
        cfg,
        working_dir=tmp_path,
    )

    assert report.remapped_anchor_tracks == 2
    assert np.all(result[0, 10:14, 10:14] == 5)
    assert np.all(result[1, 12:16, 12:16] == 5)
    assert not (result[1, 28:32, 28:32] == 5).any()


def test_apply_post_solve_corrections_preserves_identity_lineage_tracklet(monkeypatch):
    from cellflow.tracking_ultrack import corrections as corrections_module
    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.corrections import (
        Correction,
        apply_post_solve_corrections,
    )

    exported = np.zeros((2, 20, 20), dtype=np.uint32)
    exported[0, 2:6, 2:6] = 10
    exported[1, 3:7, 3:7] = 135
    tracked = np.zeros_like(exported)
    tracked[0, 2:6, 2:6] = 135

    monkeypatch.setattr(
        corrections_module,
        "_anchor_lineage_track_remaps",
        lambda working_dir, corrections, cfg: {10: 135, 135: 135},
    )

    result, report = apply_post_solve_corrections(
        exported,
        [Correction(cell_id=135, t=0, kind="anchor", y=3.5, x=3.5)],
        tracked,
        TrackingConfig(anchor_radius_px=5.0),
        working_dir="unused",
    )

    assert report.remapped_anchor_tracks == 1
    assert np.all(result[0, 2:6, 2:6] == 135)
    assert np.all(result[1, 3:7, 3:7] == 135)


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
