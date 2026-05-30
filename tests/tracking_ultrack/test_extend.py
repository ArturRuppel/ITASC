"""Tests for cellflow.tracking_ultrack.extend."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cellflow.tracking_ultrack._node_geometry import make_node_pickle


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestExtendTrack:
    def test_extend_track_from_db_handles_deserialized_2d_node(self, tmp_path):
        """DB-backed extend should accept already-deserialized 2D NodeDB.pickle values."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        from tests.tracking_ultrack.test_reseed import _make_engine

        from cellflow.tracking_ultrack.extend import extend_track_from_db

        engine = _make_engine(tmp_path / "data.db")
        node_id = 101
        y0, x0, y1, x1 = 8, 9, 14, 15
        mask_2d = np.ones((y1 - y0, x1 - x0), dtype=bool)
        node_pickle = make_node_pickle(
            1,
            mask_2d,
            np.array([y0, x0, y1, x1], dtype=np.int64),
            node_id,
        )

        with Session(engine) as session:
            session.add(
                NodeDB(
                    id=node_id,
                    t=1,
                    t_node_id=1,
                    t_hier_id=0,
                    z=0,
                    y=(y0 + y1) / 2.0,
                    x=(x0 + x1) / 2.0,
                    area=int(mask_2d.sum()),
                    pickle=node_pickle,
                )
            )
            session.commit()

        tracked = np.zeros((2, 24, 24), dtype=np.uint32)
        tracked[0, y0:y1, x0:x1] = 7

        result = extend_track_from_db(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            db_path=tmp_path / "data.db",
        )

        assert result is not None
        assert result.candidate_label == node_id
        assert result.bbox == (y0, x0, y1, x1)
        assert result.mask_2d.shape == tracked.shape[1:]
        assert result.mask_2d[y0:y1, x0:x1].all()

    def test_extend_track_from_db_skips_candidate_with_mismatched_mask_shape(self, tmp_path):
        """Malformed DB candidates should be skipped instead of crashing during paint."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        from tests.tracking_ultrack.test_reseed import _make_engine

        from cellflow.tracking_ultrack.extend import extend_track_from_db

        engine = _make_engine(tmp_path / "data.db")
        y0, x0, y1, x1 = 8, 9, 14, 15
        with Session(engine) as session:
            session.add(
                NodeDB(
                    id=102,
                    t=1,
                    t_node_id=1,
                    t_hier_id=0,
                    z=0,
                    y=(y0 + y1) / 2.0,
                    x=(x0 + x1) / 2.0,
                    area=25,
                    pickle=SimpleNamespace(
                        bbox=np.array([y0, x0, y1, x1], dtype=np.int64),
                        mask=np.ones((5, 5), dtype=bool),
                    ),
                )
            )
            session.commit()

        tracked = np.zeros((2, 24, 24), dtype=np.uint32)
        tracked[0, y0:y1, x0:x1] = 7

        result = extend_track_from_db(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            db_path=tmp_path / "data.db",
        )

        assert result is None

    def test_extend_track_from_db_greedy_overwrite_returns_single_best_candidate(self, tmp_path):
        """Greedy overwrite returns the single best candidate regardless of overlap."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        from tests.tracking_ultrack.test_reseed import _make_engine

        from cellflow.tracking_ultrack.extend import extend_track_from_db

        engine = _make_engine(tmp_path / "data.db")

        def add_node(session, node_id, y0, x0, y1, x1):
            mask_2d = np.ones((y1 - y0, x1 - x0), dtype=bool)
            node_pickle = make_node_pickle(
                1,
                mask_2d,
                np.array([y0, x0, y1, x1], dtype=np.int64),
                node_id,
            )
            session.add(
                NodeDB(
                    id=node_id,
                    t=1,
                    t_node_id=node_id,
                    t_hier_id=0,
                    z=0,
                    y=(y0 + y1) / 2.0,
                    x=(x0 + x1) / 2.0,
                    area=int(mask_2d.sum()),
                    pickle=node_pickle,
                )
            )

        with Session(engine) as session:
            add_node(session, 101, 6, 6, 11, 11)    # nearest candidate, overlaps cell 9
            session.commit()

        tracked = np.zeros((2, 32, 32), dtype=np.uint32)
        tracked[0, 5:10, 5:10] = 7
        tracked[1, 6:11, 6:11] = 9

        result = extend_track_from_db(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            db_path=tmp_path / "data.db",
            greedy_overwrite=True,
        )

        assert result is not None
        assert [assignment.cell_id for assignment in result.assignments] == [7]
        assert result.candidate_label == 101
        assert result.mask_2d[6:11, 6:11].all()

    def test_extend_track_from_db_filters_distant_nodes_before_reading_masks(
        self, tmp_path
    ):
        """DB-backed extend should not deserialize masks outside the distance window."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        from tests.tracking_ultrack.test_reseed import _make_engine

        from cellflow.tracking_ultrack import extend as extend_module

        engine = _make_engine(tmp_path / "data.db")
        near_pickle = make_node_pickle(
            1,
            np.ones((5, 5), dtype=bool),
            np.array([5, 5, 10, 10], dtype=np.int64),
            101,
        )
        with Session(engine) as session:
            session.add_all(
                [
                    NodeDB(
                        id=101,
                        t=1,
                        t_node_id=101,
                        t_hier_id=0,
                        z=0,
                        y=7.0,
                        x=7.0,
                        area=25,
                        pickle=near_pickle,
                    ),
                    NodeDB(
                        id=202,
                        t=1,
                        t_node_id=202,
                        t_hier_id=0,
                        z=0,
                        y=50.0,
                        x=50.0,
                        area=25,
                        pickle=b"far",
                    ),
                ]
            )
            session.commit()

        tracked = np.zeros((2, 64, 64), dtype=np.uint32)
        tracked[0, 5:10, 5:10] = 7

        result = extend_module.extend_track_from_db(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            db_path=tmp_path / "data.db",
            d_max=10.0,
        )

        assert result is not None
        assert result.candidate_label == 101

    def test_extend_track_from_db_computes_source_stats_once(
        self, monkeypatch, tmp_path
    ):
        """Source-mask centroid and area should not be recomputed per candidate."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        from tests.tracking_ultrack.test_reseed import _make_engine

        from cellflow.tracking_ultrack import extend as extend_module

        engine = _make_engine(tmp_path / "data.db")

        def add_node(session, node_id, y0, x0, y1, x1):
            mask_2d = np.ones((y1 - y0, x1 - x0), dtype=bool)
            session.add(
                NodeDB(
                    id=node_id,
                    t=1,
                    t_node_id=node_id,
                    t_hier_id=0,
                    z=0,
                    y=(y0 + y1) / 2.0,
                    x=(x0 + x1) / 2.0,
                    area=int(mask_2d.sum()),
                    pickle=make_node_pickle(
                        1,
                        mask_2d,
                        np.array([y0, x0, y1, x1], dtype=np.int64),
                        node_id,
                    ),
                )
            )

        with Session(engine) as session:
            add_node(session, 101, 5, 5, 10, 10)
            add_node(session, 102, 6, 6, 11, 11)
            session.commit()

        original = extend_module._mask_centroid_area
        calls = 0

        def counting_mask_centroid_area(mask):
            nonlocal calls
            calls += 1
            return original(mask)

        monkeypatch.setattr(
            extend_module,
            "_mask_centroid_area",
            counting_mask_centroid_area,
        )

        tracked = np.zeros((2, 64, 64), dtype=np.uint32)
        tracked[0, 5:10, 5:10] = 7

        result = extend_module.extend_track_from_db(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            db_path=tmp_path / "data.db",
            d_max=10.0,
        )

        assert result is not None
        assert calls == 1
