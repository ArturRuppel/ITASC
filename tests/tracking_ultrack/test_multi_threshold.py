from __future__ import annotations

import numpy as np
import pytest
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.db_build import UltrackDatabaseBuildReport

from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row


def _write_source_db(db_path, hierarchy_ids):
    engine = _make_engine(db_path)
    with Session(engine) as session:
        for idx, hierarchy_id in enumerate(hierarchy_ids, start=1):
            y0 = 5 + idx * 8
            x0 = 7 + idx * 8
            row = _make_node_row(idx, 0, y0, x0, y0 + 4, x0 + 4)
            row.t_hier_id = hierarchy_id
            session.add(row)
        session.commit()
    engine.dispose()


def _merged_hierarchy_ids(db_path):
    from ultrack.core.database import NodeDB

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            return [
                row.t_hier_id
                for row in session.query(NodeDB).order_by(NodeDB.id).all()
            ]
    finally:
        engine.dispose()


def _merged_hierarchy_parent_ids(db_path):
    from ultrack.core.database import NodeDB

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            return {
                int(row.id): row.hier_parent_id
                for row in session.query(NodeDB).order_by(NodeDB.id).all()
            }
    finally:
        engine.dispose()


def _merged_overlap_pairs(db_path):
    from ultrack.core.database import OverlapDB

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            return {
                (row.node_id, row.ancestor_id)
                for row in session.query(OverlapDB).all()
            }
    finally:
        engine.dispose()


def test_merge_ultrack_databases_remaps_t_hier_ids_globally(tmp_path):
    pytest.importorskip("ultrack")

    from cellflow.tracking_ultrack.multi_threshold import merge_ultrack_databases

    source_a = tmp_path / "source_a.db"
    source_b = tmp_path / "source_b.db"
    output = tmp_path / "merged.db"

    _write_source_db(source_a, [1, 1])
    _write_source_db(source_b, [1, None, 0])

    merge_ultrack_databases([source_a, source_b], output, frame_shape=(64, 64))

    hier_ids = _merged_hierarchy_ids(output)

    assert len(hier_ids) == 5
    assert hier_ids[0] == hier_ids[1]
    assert hier_ids[2] != hier_ids[0]
    assert hier_ids[3] not in {None, hier_ids[0], hier_ids[2]}
    assert hier_ids[4] not in {None, hier_ids[0], hier_ids[2], hier_ids[3]}


def test_merge_cross_source_overlaps_include_hidden_same_source_nodes(tmp_path):
    pytest.importorskip("ultrack")

    from cellflow.tracking_ultrack.multi_threshold import merge_ultrack_databases

    source_a = tmp_path / "source_a.db"
    source_b = tmp_path / "source_b.db"
    output = tmp_path / "merged.db"

    engine_a = _make_engine(source_a)
    with Session(engine_a) as session:
        session.add(_make_node_row(1, 0, 0, 0, 10, 10))
        session.add(_make_node_row(2, 0, 5, 5, 15, 15))
        session.commit()
    engine_a.dispose()

    engine_b = _make_engine(source_b)
    with Session(engine_b) as session:
        session.add(_make_node_row(1, 0, 6, 6, 9, 9))
        session.commit()
    engine_b.dispose()

    report = merge_ultrack_databases(
        [source_a, source_b], output, frame_shape=(32, 32)
    )

    assert report.cross_source_overlaps == 2
    assert _merged_overlap_pairs(output) == {(3, 1), (3, 2)}


def test_merge_ultrack_databases_remaps_hierarchy_parents_per_source(tmp_path):
    pytest.importorskip("ultrack")

    from ultrack.utils.constants import NO_PARENT

    from cellflow.tracking_ultrack.multi_threshold import merge_ultrack_databases

    source_a = tmp_path / "source_a.db"
    source_b = tmp_path / "source_b.db"
    output = tmp_path / "merged.db"

    _write_source_db(source_a, [1, 1])
    _write_source_db(source_b, [1, 1])

    for db_path in (source_a, source_b):
        engine = sqla.create_engine(f"sqlite:///{db_path}")
        try:
            with Session(engine) as session:
                from ultrack.core.database import NodeDB

                root = session.query(NodeDB).filter(NodeDB.id == 1).one()
                child = session.query(NodeDB).filter(NodeDB.id == 2).one()
                root.hier_parent_id = NO_PARENT
                child.hier_parent_id = 1
                session.commit()
        finally:
            engine.dispose()

    merge_ultrack_databases([source_a, source_b], output, frame_shape=(64, 64))

    assert _merged_hierarchy_parent_ids(output) == {
        1: NO_PARENT,
        2: 1,
        3: NO_PARENT,
        4: 3,
    }


def test_merge_source_metadata_does_not_corrupt_segm_annotation_enum(tmp_path):
    pytest.importorskip("ultrack")

    from ultrack.core.database import NodeDB, NodeSegmAnnotation

    from cellflow.tracking_ultrack.multi_threshold import (
        merge_ultrack_databases,
        query_source_indices,
        query_source_node_ids,
    )

    source_a = tmp_path / "source_a.db"
    source_b = tmp_path / "source_b.db"
    output = tmp_path / "merged.db"

    _write_source_db(source_a, [1])
    _write_source_db(source_b, [1, 1])

    merge_ultrack_databases([source_a, source_b], output, frame_shape=(64, 64))

    engine = sqla.create_engine(f"sqlite:///{output}")
    try:
        with Session(engine) as session:
            rows = session.query(NodeDB).order_by(NodeDB.id).all()
            assert [row.segm_annot for row in rows] == [
                NodeSegmAnnotation.UNKNOWN,
                NodeSegmAnnotation.UNKNOWN,
                NodeSegmAnnotation.UNKNOWN,
            ]
    finally:
        engine.dispose()

    assert query_source_indices(output) == (0, 1)
    assert query_source_node_ids(output, 0) == (1,)
    assert query_source_node_ids(output, 1) == (2, 3)


def test_build_multithreshold_database_normalizes_globally_and_keeps_values(
    tmp_path, monkeypatch
):
    import cellflow.tracking_ultrack.multi_threshold as mt

    contours = np.array(
        [
            [[0.0, 10.0], [20.0, 30.0]],
            [[40.0, 50.0], [60.0, 100.0]],
        ],
        dtype=np.float32,
    )
    foreground = np.array(
        [
            [[100.0, 90.0], [80.0, 70.0]],
            [[60.0, 50.0], [40.0, 0.0]],
        ],
        dtype=np.float32,
    )
    segmented: list[tuple[np.ndarray, np.ndarray]] = []
    merge_calls = []

    monkeypatch.setattr(
        mt,
        "_load_ultrack_inputs",
        lambda contour_path, foreground_path: (contours, foreground),
    )

    def fake_segment(foreground_thr, contours_thr, ultrack_cfg, cfg):
        segmented.append((foreground_thr.copy(), contours_thr.copy()))
        (ultrack_cfg.tmp_dir / "data.db").write_bytes(b"source")

    def fake_build_config(cfg, tmp_dir):
        class DummyConfig:
            pass

        dummy = DummyConfig()
        dummy.tmp_dir = tmp_dir
        return dummy

    monkeypatch.setattr(mt, "_build_ultrack_config", fake_build_config)
    monkeypatch.setattr(mt, "_run_ultrack_segment", fake_segment)

    def fake_merge(temp_dbs, output_path, frame_shape, progress_cb=None):
        merge_calls.append((list(temp_dbs), output_path, frame_shape))
        output_path.write_bytes(b"merged")

    monkeypatch.setattr(mt, "merge_ultrack_databases", fake_merge)
    score_calls = []

    def fake_write_seed_prior_node_probs(working_dir, intensity_image_path, cfg):
        score_calls.append((working_dir, intensity_image_path))
        return type("ScoreReport", (), {"scored": 1, "seeds": 1})()

    monkeypatch.setattr(
        mt,
        "write_seed_prior_node_probs",
        fake_write_seed_prior_node_probs,
    )
    monkeypatch.setattr(mt, "run_linking", lambda working_dir, cfg: [])

    report = mt.build_multithreshold_database(
        tmp_path / "contours.tif",
        tmp_path / "foreground_scores.tif",
        tmp_path / "nucleus_prob_zavg.tif",
        tmp_path / "work",
        TrackingConfig(),
        thresholds=[0.25, 0.75],
    )

    foreground_025, contours_025 = segmented[0]
    foreground_075, contours_075 = segmented[1]

    expected_contours_norm = contours / 100.0
    expected_foreground_norm = foreground / 100.0
    np.testing.assert_allclose(
        contours_025,
        np.where(expected_contours_norm < 0.25, 0.0, expected_contours_norm),
    )
    np.testing.assert_allclose(
        foreground_025,
        np.where(expected_foreground_norm < 0.25, 0.0, expected_foreground_norm),
    )
    np.testing.assert_allclose(
        contours_075,
        np.where(expected_contours_norm < 0.75, 0.0, expected_contours_norm),
    )
    np.testing.assert_allclose(
        foreground_075,
        np.where(expected_foreground_norm < 0.75, 0.0, expected_foreground_norm),
    )
    assert contours_025[0, 1, 1] == pytest.approx(0.3)
    assert foreground_025[0, 1, 1] == pytest.approx(0.7)
    assert merge_calls[0][2] == (2, 2)
    assert score_calls == [
        (tmp_path / "work", tmp_path / "foreground_scores.tif")
    ]
    assert report == UltrackDatabaseBuildReport(scored_nodes=1, seed_nodes=1)


def test_build_multithreshold_database_uses_threshold_order(tmp_path, monkeypatch):
    import cellflow.tracking_ultrack.multi_threshold as mt

    calls = []
    stack = np.arange(8, dtype=np.float32).reshape(2, 2, 2)

    monkeypatch.setattr(mt, "_load_ultrack_inputs", lambda *_: (stack, stack))

    def fake_build_config(cfg, tmp_dir):
        class DummyConfig:
            pass

        dummy = DummyConfig()
        dummy.tmp_dir = tmp_dir
        return dummy

    def fake_segment(foreground_thr, contours_thr, ultrack_cfg, cfg):
        calls.append(ultrack_cfg.tmp_dir.name)
        (ultrack_cfg.tmp_dir / "data.db").write_bytes(b"source")

    monkeypatch.setattr(mt, "_build_ultrack_config", fake_build_config)
    monkeypatch.setattr(mt, "_run_ultrack_segment", fake_segment)
    monkeypatch.setattr(
        mt,
        "merge_ultrack_databases",
        lambda temp_dbs, output_path, frame_shape, progress_cb=None: output_path.write_bytes(
            b"merged"
        ),
    )
    monkeypatch.setattr(
        mt,
        "write_seed_prior_node_probs",
        lambda *_: type("ScoreReport", (), {"scored": 0, "seeds": 0})(),
    )
    monkeypatch.setattr(mt, "run_linking", lambda *_: [])

    mt.build_multithreshold_database(
        tmp_path / "contours.tif",
        tmp_path / "foreground_scores.tif",
        tmp_path / "nucleus_prob_zavg.tif",
        tmp_path / "work",
        TrackingConfig(),
        thresholds=[0.5, 0.1, 0.3],
    )

    assert calls == ["_mt_tmp_0", "_mt_tmp_1", "_mt_tmp_2"]
