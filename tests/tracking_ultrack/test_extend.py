"""Tests for itasc.tracking_ultrack.extend (LinkDB-driven extend)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from itasc.tracking_ultrack._node_geometry import make_node_pickle


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _add_node(session, node_id, t, y0, x0, y1, x1):
    """Insert a NodeDB row with a real pickled mask at frame ``t``."""
    from ultrack.core.database import NodeDB

    mask_2d = np.ones((y1 - y0, x1 - x0), dtype=bool)
    session.add(
        NodeDB(
            id=node_id,
            t=t,
            t_node_id=node_id,
            t_hier_id=0,
            z=0,
            y=(y0 + y1) / 2.0,
            x=(x0 + x1) / 2.0,
            area=int(mask_2d.sum()),
            pickle=make_node_pickle(
                t, mask_2d, np.array([y0, x0, y1, x1], dtype=np.int64), node_id
            ),
        )
    )


def _add_link(session, source_id, target_id, weight):
    """Insert a LinkDB edge (source at t, target at t+1) with ``weight``."""
    from ultrack.core.database import LinkDB, VarAnnotation

    session.add(
        LinkDB(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            annotation=VarAnnotation.UNKNOWN,
        )
    )


# ---------------------------------------------------------------------------
# extend_track_from_db
# ---------------------------------------------------------------------------

class TestExtendTrackFromDb:
    def test_forward_follows_link_to_target(self, tmp_path):
        """Forward extend matches the source node and follows its LinkDB edge."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from tests.tracking_ultrack.test_reseed import _make_engine

        from itasc.tracking_ultrack.extend import extend_track_from_db

        engine = _make_engine(tmp_path / "data.db")
        y0, x0, y1, x1 = 8, 9, 14, 15
        with Session(engine) as session:
            _add_node(session, 50, 0, y0, x0, y1, x1)       # source node at t=0
            _add_node(session, 101, 1, y0, x0, y1, x1)      # linked candidate at t=1
            _add_link(session, 50, 101, weight=0.9)
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
        assert result.target_frame == 1
        assert result.candidate_label == 101
        assert result.weight == pytest.approx(0.9)
        assert result.bbox == (y0, x0, y1, x1)
        assert result.mask_2d.shape == tracked.shape[1:]
        assert result.mask_2d[y0:y1, x0:x1].all()

    def test_backward_follows_reverse_link(self, tmp_path):
        """Backward extend reads rows where the matched node is the link target."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from tests.tracking_ultrack.test_reseed import _make_engine

        from itasc.tracking_ultrack.extend import extend_track_from_db

        engine = _make_engine(tmp_path / "data.db")
        y0, x0, y1, x1 = 8, 9, 14, 15
        with Session(engine) as session:
            _add_node(session, 200, 0, y0, x0, y1, x1)      # predecessor at t=0
            _add_node(session, 201, 1, y0, x0, y1, x1)      # source node at t=1
            _add_link(session, 200, 201, weight=0.5)        # link points 0 -> 1
            session.commit()

        tracked = np.zeros((2, 24, 24), dtype=np.uint32)
        tracked[1, y0:y1, x0:x1] = 7

        result = extend_track_from_db(
            source_id=7,
            source_frame=1,
            direction="backward",
            tracked_labels=tracked,
            db_path=tmp_path / "data.db",
        )

        assert result is not None
        assert result.target_frame == 0
        assert result.candidate_label == 200

    def test_highest_weight_link_wins(self, tmp_path):
        """When several links exist, the highest-weight candidate is chosen."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from tests.tracking_ultrack.test_reseed import _make_engine

        from itasc.tracking_ultrack.extend import extend_track_from_db

        engine = _make_engine(tmp_path / "data.db")
        with Session(engine) as session:
            _add_node(session, 50, 0, 5, 5, 11, 11)
            _add_node(session, 101, 1, 5, 5, 11, 11)
            _add_node(session, 102, 1, 20, 20, 26, 26)
            _add_link(session, 50, 101, weight=0.2)
            _add_link(session, 50, 102, weight=0.8)
            session.commit()

        tracked = np.zeros((2, 40, 40), dtype=np.uint32)
        tracked[0, 5:11, 5:11] = 7

        result = extend_track_from_db(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            db_path=tmp_path / "data.db",
        )

        assert result is not None
        assert result.candidate_label == 102
        assert result.weight == pytest.approx(0.8)

    def test_none_when_no_link(self, tmp_path):
        """A matched node with no LinkDB edge yields None (no spatial fallback)."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from tests.tracking_ultrack.test_reseed import _make_engine

        from itasc.tracking_ultrack.extend import extend_track_from_db

        engine = _make_engine(tmp_path / "data.db")
        y0, x0, y1, x1 = 8, 9, 14, 15
        with Session(engine) as session:
            _add_node(session, 50, 0, y0, x0, y1, x1)        # source node, no links
            _add_node(session, 101, 1, y0, x0, y1, x1)       # spatially aligned but unlinked
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

    def test_none_when_source_matches_no_node(self, tmp_path):
        """A source mask overlapping no node cannot be extended."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from tests.tracking_ultrack.test_reseed import _make_engine

        from itasc.tracking_ultrack.extend import extend_track_from_db

        engine = _make_engine(tmp_path / "data.db")
        with Session(engine) as session:
            _add_node(session, 101, 1, 5, 5, 11, 11)
            session.commit()

        tracked = np.zeros((2, 40, 40), dtype=np.uint32)
        tracked[0, 30:35, 30:35] = 7   # nowhere near any node at t=0

        result = extend_track_from_db(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            db_path=tmp_path / "data.db",
        )
        assert result is None

    def test_skips_candidate_with_mismatched_mask_shape(self, tmp_path):
        """A malformed linked candidate is skipped rather than crashing."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        from tests.tracking_ultrack.test_reseed import _make_engine

        from itasc.tracking_ultrack.extend import extend_track_from_db

        engine = _make_engine(tmp_path / "data.db")
        y0, x0, y1, x1 = 8, 9, 14, 15
        with Session(engine) as session:
            _add_node(session, 50, 0, y0, x0, y1, x1)
            session.add(
                NodeDB(
                    id=102,
                    t=1,
                    t_node_id=102,
                    t_hier_id=0,
                    z=0,
                    y=(y0 + y1) / 2.0,
                    x=(x0 + x1) / 2.0,
                    area=25,
                    pickle=SimpleNamespace(
                        bbox=np.array([y0, x0, y1, x1], dtype=np.int64),
                        mask=np.ones((5, 5), dtype=bool),  # shape != bbox
                    ),
                )
            )
            _add_link(session, 50, 102, weight=0.9)
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

    def test_none_when_db_missing(self, tmp_path):
        """A missing DB yields None rather than raising."""
        from itasc.tracking_ultrack.extend import extend_track_from_db

        tracked = np.zeros((2, 24, 24), dtype=np.uint32)
        tracked[0, 5:10, 5:10] = 7
        result = extend_track_from_db(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            db_path=tmp_path / "missing.db",
        )
        assert result is None


# ---------------------------------------------------------------------------
# list_extend_candidates
# ---------------------------------------------------------------------------

class TestListExtendCandidates:
    def _seed_two_links(self, tmp_path):
        """Two linked candidates at t=1 with distinct weights for a t=0 source."""
        from sqlalchemy.orm import Session
        from tests.tracking_ultrack.test_reseed import _make_engine

        engine = _make_engine(tmp_path / "data.db")
        with Session(engine) as session:
            _add_node(session, 50, 0, 5, 5, 11, 11)
            _add_node(session, 101, 1, 5, 5, 11, 11)    # weaker link
            _add_node(session, 102, 1, 6, 6, 12, 12)    # stronger link
            _add_link(session, 50, 101, weight=0.3)
            _add_link(session, 50, 102, weight=0.7)
            session.commit()

    def test_lists_ranked_candidates_with_masks(self, tmp_path):
        """The gallery API returns linked candidates, best-first, with masks."""
        pytest.importorskip("ultrack")
        from itasc.tracking_ultrack.extend import list_extend_candidates

        self._seed_two_links(tmp_path)
        tracked = np.zeros((2, 32, 32), dtype=np.uint32)
        tracked[0, 5:11, 5:11] = 7

        result = list_extend_candidates(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            db_path=tmp_path / "data.db",
        )

        assert result.target_frame == 1
        assert not result.is_empty()
        labels = [a.candidate_label for a in result.assignments]
        assert set(labels) == {101, 102}
        weights = [a.weight for a in result.assignments]
        assert weights == sorted(weights, reverse=True)
        assert labels[0] == 102   # highest weight first
        for a in result.assignments:
            assert a.mask_2d.shape == tracked.shape[1:]
            assert a.mask_2d.any()

    def test_winner_matches_extend_track_from_db(self, tmp_path):
        """The top gallery candidate is the one extend_track_from_db would apply."""
        pytest.importorskip("ultrack")
        from itasc.tracking_ultrack.extend import (
            extend_track_from_db,
            list_extend_candidates,
        )

        self._seed_two_links(tmp_path)
        tracked = np.zeros((2, 32, 32), dtype=np.uint32)
        tracked[0, 5:11, 5:11] = 7

        kw = dict(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            db_path=tmp_path / "data.db",
        )
        chosen = extend_track_from_db(**kw)
        listed = list_extend_candidates(**kw)
        assert chosen is not None
        assert listed.assignments[0].candidate_label == chosen.candidate_label

    def test_limit_caps_loaded_candidates(self, tmp_path):
        """Only the top ``limit`` linked candidates are returned."""
        pytest.importorskip("ultrack")
        from sqlalchemy.orm import Session
        from tests.tracking_ultrack.test_reseed import _make_engine

        from itasc.tracking_ultrack.extend import list_extend_candidates

        engine = _make_engine(tmp_path / "data.db")
        with Session(engine) as session:
            _add_node(session, 50, 0, 5, 5, 11, 11)
            for i in range(5):
                _add_node(session, 100 + i, 1, 5 + i, 5, 11 + i, 11)
                _add_link(session, 50, 100 + i, weight=0.1 * (i + 1))
            session.commit()

        tracked = np.zeros((2, 40, 40), dtype=np.uint32)
        tracked[0, 5:11, 5:11] = 7

        result = list_extend_candidates(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            db_path=tmp_path / "data.db",
            limit=2,
        )
        assert len(result.assignments) == 2
        # Highest weights are 0.5 (node 104) then 0.4 (node 103).
        assert [a.candidate_label for a in result.assignments] == [104, 103]

    def test_empty_when_target_frame_out_of_range(self, tmp_path):
        """Backward from frame 0 has no target frame: empty, but target_frame kept."""
        pytest.importorskip("ultrack")
        from itasc.tracking_ultrack.extend import list_extend_candidates

        self._seed_two_links(tmp_path)
        tracked = np.zeros((2, 32, 32), dtype=np.uint32)
        tracked[0, 5:11, 5:11] = 7

        result = list_extend_candidates(
            source_id=7,
            source_frame=0,
            direction="backward",
            tracked_labels=tracked,
            db_path=tmp_path / "data.db",
        )
        assert result.is_empty()
        assert result.target_frame == -1

    def test_empty_when_db_missing(self, tmp_path):
        """A missing DB yields an empty shortlist rather than raising."""
        from itasc.tracking_ultrack.extend import list_extend_candidates

        tracked = np.zeros((2, 32, 32), dtype=np.uint32)
        tracked[0, 5:11, 5:11] = 7
        result = list_extend_candidates(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            db_path=tmp_path / "does_not_exist.db",
        )
        assert result.is_empty()
        assert result.target_frame == 1
