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
