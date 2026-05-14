from __future__ import annotations

import numpy as np
import pytest
import sqlalchemy as sqla
import tifffile
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


def test_build_ultrack_source_stacks_preserves_contours_and_binarizes_foreground():
    from cellflow.tracking_ultrack.multi_threshold import build_ultrack_source_stacks

    contours = np.array(
        [
            [[0.1, 0.4], [0.6, 0.9]],
            [[0.2, 0.5], [0.7, 1.0]],
        ],
        dtype=np.float32,
    )
    foreground_scores = np.array(
        [
            [[0.2, 0.3], [0.8, 0.9]],
            [[0.1, 0.5], [0.6, 1.0]],
        ],
        dtype=np.float32,
    )

    contour_sources, foreground_sources, metadata = build_ultrack_source_stacks(
        contours,
        foreground_scores,
        contour_thresholds=[0.5, 0.8],
        foreground_thresholds=[0.4, 0.7],
    )

    assert contour_sources.shape == (4, 2, 2, 2)
    assert foreground_sources.shape == (4, 2, 2, 2)
    assert contour_sources.dtype == np.float32
    assert foreground_sources.dtype == np.uint8
    np.testing.assert_allclose(
        contour_sources[0],
        np.where(contours >= 0.5, contours, 0.0),
    )
    np.testing.assert_array_equal(
        foreground_sources[0],
        (foreground_scores >= 0.4).astype(np.uint8),
    )
    assert set(np.unique(foreground_sources)) <= {0, 1}
    assert metadata == [
        {"contour_threshold": 0.5, "foreground_threshold": 0.4},
        {"contour_threshold": 0.5, "foreground_threshold": 0.7},
        {"contour_threshold": 0.8, "foreground_threshold": 0.4},
        {"contour_threshold": 0.8, "foreground_threshold": 0.7},
    ]


def test_write_ultrack_source_stacks_writes_expected_tiffs(tmp_path):
    from cellflow.tracking_ultrack.multi_threshold import write_ultrack_source_stacks

    contours = np.array([[[0.1, 0.6]]], dtype=np.float32)
    foreground_scores = np.array([[[0.2, 0.7]]], dtype=np.float32)
    contours_path = tmp_path / "contours.tif"
    scores_path = tmp_path / "foreground_scores.tif"
    contour_sources_path = tmp_path / "contour_sources.tif"
    foreground_sources_path = tmp_path / "foreground_sources.tif"
    tifffile.imwrite(contours_path, contours)
    tifffile.imwrite(scores_path, foreground_scores)

    metadata = write_ultrack_source_stacks(
        contours_path,
        scores_path,
        contour_sources_path,
        foreground_sources_path,
        contour_thresholds=[0.5],
        foreground_thresholds=[0.5],
    )

    np.testing.assert_allclose(
        tifffile.imread(contour_sources_path),
        np.array([[[[0.0, 0.6]]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        tifffile.imread(foreground_sources_path),
        np.array([[[[0, 1]]]], dtype=np.uint8),
    )
    assert metadata == [{"contour_threshold": 0.5, "foreground_threshold": 0.5}]


def test_preview_ultrack_source_stack_frame_returns_one_frame_without_writing():
    from cellflow.tracking_ultrack.multi_threshold import preview_ultrack_source_stack_frame

    contours = np.array(
        [
            [[0.1, 0.6], [0.8, 0.2]],
            [[0.4, 0.7], [0.3, 1.0]],
        ],
        dtype=np.float32,
    )
    foreground_scores = np.array(
        [
            [[0.1, 0.9], [0.6, 0.2]],
            [[0.8, 0.4], [0.7, 1.0]],
        ],
        dtype=np.float32,
    )

    contour_frame, foreground_frame, frame_index, metadata = preview_ultrack_source_stack_frame(
        contours,
        foreground_scores,
        contour_thresholds=[0.5, 0.75],
        foreground_thresholds=[0.6],
        frame_index=1,
    )

    assert contour_frame.shape == (2, 2, 2)
    assert foreground_frame.shape == (2, 2, 2)
    np.testing.assert_allclose(
        contour_frame[0],
        np.where(contours[1] >= 0.5, contours[1], 0.0),
    )
    np.testing.assert_array_equal(
        foreground_frame[0],
        (foreground_scores[1] >= 0.6).astype(np.uint8),
    )
    assert frame_index == 1
    assert metadata == [
        {"contour_threshold": 0.5, "foreground_threshold": 0.6},
        {"contour_threshold": 0.75, "foreground_threshold": 0.6},
    ]


def test_preview_ultrack_source_stack_frame_returns_clamped_frame_index():
    from cellflow.tracking_ultrack.multi_threshold import preview_ultrack_source_stack_frame

    contours = np.ones((2, 2, 2), dtype=np.float32)
    foreground_scores = np.ones((2, 2, 2), dtype=np.float32)

    contour_frame, foreground_frame, frame_index, _metadata = preview_ultrack_source_stack_frame(
        contours,
        foreground_scores,
        contour_thresholds=[0.5],
        foreground_thresholds=[0.5],
        frame_index=99,
    )

    assert frame_index == 1
    assert contour_frame.shape == (1, 2, 2)
    assert foreground_frame.shape == (1, 2, 2)


def test_build_ultrack_database_from_sources_segments_sources_in_order(
    tmp_path, monkeypatch
):
    import cellflow.tracking_ultrack.multi_threshold as mt

    contours = np.stack(
        [
            np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4),
            np.linspace(1.0, 2.0, 24, dtype=np.float32).reshape(2, 3, 4),
        ],
        axis=0,
    )
    foreground = np.stack(
        [
            np.zeros((2, 3, 4), dtype=np.uint8),
            np.ones((2, 3, 4), dtype=np.uint8),
        ],
        axis=0,
    )
    contour_sources_path = tmp_path / "contour_sources.tif"
    foreground_sources_path = tmp_path / "foreground_sources.tif"
    tifffile.imwrite(contour_sources_path, contours)
    tifffile.imwrite(foreground_sources_path, foreground)

    segmented: list[tuple[np.ndarray, np.ndarray, str]] = []
    merge_calls = []

    def fake_build_config(cfg, tmp_dir):
        class DummyConfig:
            pass

        dummy = DummyConfig()
        dummy.tmp_dir = tmp_dir
        return dummy

    def fake_segment(foreground_source, contour_source, ultrack_cfg, cfg):
        segmented.append(
            (foreground_source.copy(), contour_source.copy(), ultrack_cfg.tmp_dir.name)
        )
        (ultrack_cfg.tmp_dir / "data.db").write_bytes(b"source")

    def fake_merge(temp_dbs, output_path, frame_shape, progress_cb=None):
        merge_calls.append((list(temp_dbs), output_path, frame_shape))
        output_path.write_bytes(b"merged")

    monkeypatch.setattr(mt, "_build_ultrack_config", fake_build_config)
    monkeypatch.setattr(mt, "_run_ultrack_segment", fake_segment)
    monkeypatch.setattr(mt, "merge_ultrack_databases", fake_merge)
    monkeypatch.setattr(mt, "run_linking", lambda working_dir, cfg: [])

    report = mt.build_ultrack_database_from_sources(
        contour_sources_path,
        foreground_sources_path,
        tmp_path / "work",
        TrackingConfig(),
    )

    assert [item[2] for item in segmented] == ["_source_tmp_0", "_source_tmp_1"]
    np.testing.assert_allclose(segmented[0][1], contours[0])
    np.testing.assert_array_equal(segmented[1][0], foreground[1].astype(np.float32))
    assert merge_calls[0][2] == (3, 4)
    assert report == UltrackDatabaseBuildReport()


def test_build_ultrack_database_from_sources_normalizes_single_source_input(
    tmp_path, monkeypatch
):
    import cellflow.tracking_ultrack.multi_threshold as mt

    tifffile.imwrite(
        tmp_path / "contour_sources.tif",
        np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4),
    )
    tifffile.imwrite(tmp_path / "foreground_sources.tif", np.ones((2, 3, 4), dtype=np.uint8))
    calls = []

    def fake_build_config(cfg, tmp_dir):
        class DummyConfig:
            pass

        dummy = DummyConfig()
        dummy.tmp_dir = tmp_dir
        return dummy

    def fake_segment(foreground_source, contour_source, ultrack_cfg, cfg):
        calls.append((foreground_source.shape, contour_source.shape))
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
    monkeypatch.setattr(mt, "run_linking", lambda *_: [])

    mt.build_ultrack_database_from_sources(
        tmp_path / "contour_sources.tif",
        tmp_path / "foreground_sources.tif",
        tmp_path / "work",
        TrackingConfig(),
    )

    assert calls == [((2, 3, 4), (2, 3, 4))]


def test_build_ultrack_database_from_sources_rejects_shape_mismatch(tmp_path):
    from cellflow.tracking_ultrack.multi_threshold import build_ultrack_database_from_sources

    tifffile.imwrite(
        tmp_path / "contour_sources.tif",
        np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4),
    )
    tifffile.imwrite(tmp_path / "foreground_sources.tif", np.ones((3, 3, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="same shape"):
        build_ultrack_database_from_sources(
            tmp_path / "contour_sources.tif",
            tmp_path / "foreground_sources.tif",
            tmp_path / "work",
            TrackingConfig(),
        )


def test_build_ultrack_database_from_sources_rejects_nonbinary_foreground(tmp_path):
    from cellflow.tracking_ultrack.multi_threshold import build_ultrack_database_from_sources

    tifffile.imwrite(
        tmp_path / "contour_sources.tif",
        np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4),
    )
    tifffile.imwrite(
        tmp_path / "foreground_sources.tif",
        np.full((2, 3, 4), 0.5, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="foreground_sources.*binary"):
        build_ultrack_database_from_sources(
            tmp_path / "contour_sources.tif",
            tmp_path / "foreground_sources.tif",
            tmp_path / "work",
            TrackingConfig(),
        )


def test_build_ultrack_database_from_sources_rejects_empty_contours(tmp_path):
    from cellflow.tracking_ultrack.multi_threshold import build_ultrack_database_from_sources

    tifffile.imwrite(tmp_path / "contour_sources.tif", np.zeros((2, 3, 4), dtype=np.float32))
    tifffile.imwrite(tmp_path / "foreground_sources.tif", np.ones((2, 3, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="contour_sources.*nonzero"):
        build_ultrack_database_from_sources(
            tmp_path / "contour_sources.tif",
            tmp_path / "foreground_sources.tif",
            tmp_path / "work",
            TrackingConfig(),
        )


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
    monkeypatch.setattr(mt, "run_linking", lambda working_dir, cfg: [])

    report = mt.build_multithreshold_database(
        tmp_path / "contours.tif",
        tmp_path / "foreground_scores.tif",
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
    assert report == UltrackDatabaseBuildReport()


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
    monkeypatch.setattr(mt, "run_linking", lambda *_: [])

    mt.build_multithreshold_database(
        tmp_path / "contours.tif",
        tmp_path / "foreground_scores.tif",
        tmp_path / "work",
        TrackingConfig(),
        thresholds=[0.5, 0.1, 0.3],
    )

    assert calls == ["_mt_tmp_0", "_mt_tmp_1", "_mt_tmp_2"]
