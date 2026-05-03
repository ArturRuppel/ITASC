"""Tests for cellflow.tracking_ultrack.extend."""
from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from cellflow.database.hypotheses import HypothesisRecord, write_hypothesis_record
from cellflow.segmentation import NucleusHypothesisParams
from cellflow.tracking_ultrack.extend import ExtendResult, extend_track
from cellflow.tracking_ultrack.validation_nodes import _make_node_pickle


def _default_params(p: int = 0) -> NucleusHypothesisParams:
    return NucleusHypothesisParams(z_slice=p)


def _write_hyp_h5(path, records: list[tuple[int, int, np.ndarray]]) -> None:
    """Write (t, p, labels_2d) tuples into a minimal hypotheses.h5."""
    with h5py.File(path, "w") as f:
        f.attrs["version"] = 2
        f.attrs["stage"] = "nucleus_hypotheses"
        f.attrs["layout"] = "hypotheses/t{t:03d}/p{p:03d}/labels"
        for t, p, labels_2d in records:
            rec = HypothesisRecord(
                t=t, p=p,
                labels=labels_2d[np.newaxis],  # (1, Y, X)
                params=_default_params(p),
            )
            write_hypothesis_record(f, rec)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H, W = 60, 60


@pytest.fixture
def simple_hyp(tmp_path):
    """2-frame tracked array + one well-matched hypothesis at t=1 (p=0)."""
    tracked = np.zeros((2, H, W), dtype=np.uint32)
    tracked[0, 10:20, 10:20] = 1   # source cell at t=0, area=100

    lm = np.zeros((H, W), dtype=np.uint32)
    lm[11:21, 11:21] = 1           # t=1 candidate: shifted 1px, same area

    h5_path = tmp_path / "hypotheses.h5"
    _write_hyp_h5(h5_path, [(1, 0, lm)])
    return tracked, h5_path


@pytest.fixture
def fragment_vs_full_hyp(tmp_path):
    """t=1 has p=0 (fragment, ~25% area of source) and p=1 (near-full, ~96% area, slightly displaced)."""
    tracked = np.zeros((2, H, W), dtype=np.uint32)
    tracked[0, 10:20, 10:20] = 1   # source: 10×10 = 100 px, centroid=(14.5, 14.5)

    # p=0: small fragment entirely inside the source footprint — passes IoU gate but wrong score
    frag = np.zeros((H, W), dtype=np.uint32)
    frag[12:15, 12:15] = 1         # 3×3 = 9 px, centroid=(13, 13), dist≈2.1, area_ratio=0.09

    # p=1: nearly full cell, displaced 3 px — correct match
    full = np.zeros((H, W), dtype=np.uint32)
    full[11:21, 13:23] = 1         # 10×10 = 100 px, centroid=(15.5, 17.5), dist≈4.1, area_ratio=1.0

    h5_path = tmp_path / "hypotheses.h5"
    _write_hyp_h5(h5_path, [(1, 0, frag), (1, 1, full)])
    return tracked, h5_path


@pytest.fixture
def far_hyp(tmp_path):
    """t=1 hypothesis centroid is beyond d_max from source."""
    tracked = np.zeros((2, H, W), dtype=np.uint32)
    tracked[0, 5:10, 5:10] = 1   # centroid≈(7, 7)

    lm = np.zeros((H, W), dtype=np.uint32)
    lm[50:55, 50:55] = 1          # centroid≈(52, 52), dist≈63.6 > 40

    h5_path = tmp_path / "hypotheses.h5"
    _write_hyp_h5(h5_path, [(1, 0, lm)])
    return tracked, h5_path


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestExtendTrack:
    def test_extend_track_from_db_handles_deserialized_2d_node(self, tmp_path):
        """DB-backed extend should accept already-deserialized 2D NodeDB.pickle values."""
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        from tests.tracking_ultrack.test_reseed import _make_engine

        from cellflow.tracking_ultrack.extend import extend_track_from_db

        engine = _make_engine(tmp_path / "data.db")
        node_id = 101
        y0, x0, y1, x1 = 8, 9, 14, 15
        mask_2d = np.ones((y1 - y0, x1 - x0), dtype=bool)
        node_pickle = _make_node_pickle(
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

    def test_forward_single_match(self, simple_hyp):
        """Forward with one close hypothesis returns a valid ExtendResult."""
        tracked, h5_path = simple_hyp
        result = extend_track(
            source_id=1,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            hypotheses_path=h5_path,
        )
        assert result is not None
        assert isinstance(result, ExtendResult)
        assert result.target_frame == 1
        assert result.candidate_partition == 0
        assert result.mask_2d.dtype == bool
        assert result.mask_2d.shape == (H, W)
        assert result.mask_2d.any()
        assert 0.9 <= result.area_ratio <= 1.0
        assert result.centroid_distance <= 2.0
        assert result.existing_overlap == 0.0

    def test_fragment_vs_full_cell_full_wins(self, fragment_vs_full_hyp):
        """The near-full displaced cell beats the small fragment on area ratio."""
        tracked, h5_path = fragment_vs_full_hyp
        result = extend_track(
            source_id=1,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            hypotheses_path=h5_path,
        )
        assert result is not None
        # The full cell (p=1) has area_ratio≈1.0; fragment (p=0) has ≈0.09
        assert result.candidate_partition == 1
        assert result.area_ratio > 0.9

    def test_no_candidate_within_d_max_returns_none(self, far_hyp):
        """When the only hypothesis is beyond d_max, returns None."""
        tracked, h5_path = far_hyp
        result = extend_track(
            source_id=1,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            hypotheses_path=h5_path,
        )
        assert result is None

    def test_backward_at_t0_returns_none(self, simple_hyp):
        """Going backward from t=0 is out of range — returns None."""
        tracked, h5_path = simple_hyp
        result = extend_track(
            source_id=1,
            source_frame=0,
            direction="backward",
            tracked_labels=tracked,
            hypotheses_path=h5_path,
        )
        assert result is None

    def test_source_missing_at_source_frame_returns_none(self, simple_hyp):
        """Source cell ID not present in tracked_labels at source_frame → None."""
        tracked, h5_path = simple_hyp
        result = extend_track(
            source_id=99,   # non-existent
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            hypotheses_path=h5_path,
        )
        assert result is None

    def test_overlap_penalty_picks_clear_candidate(self, tmp_path):
        """A perfect-area candidate occluded by another cell loses to a smaller clear candidate.

        Score = area_ratio * (1 - existing_overlap).
        - Occluded full match: area_ratio=1.0, overlap=0.6 → 0.40
        - Clear smaller match: area_ratio=0.64, overlap=0.0 → 0.64 (wins)
        """
        H2, W2 = 60, 60
        tracked = np.zeros((2, H2, W2), dtype=np.uint32)
        tracked[0, 10:20, 10:20] = 1                  # source: 100 px, centroid=(14.5, 14.5)
        tracked[1, 11:18, 13:21] = 99                 # other cell occupying near-source region

        # p=0: full-size 10×10 candidate near source — but ~60% occluded by cell 99
        occluded = np.zeros((H2, W2), dtype=np.uint32)
        occluded[11:21, 11:21] = 1                    # area=100, dist≈1.4, overlap=60/100=0.60

        # p=1: smaller 8×8 candidate slightly further — but in clear space
        clear = np.zeros((H2, W2), dtype=np.uint32)
        clear[20:28, 8:16] = 1                        # area=64, centroid=(23.5, 11.5), dist≈9.5, overlap=0

        h5_path = tmp_path / "hypotheses.h5"
        _write_hyp_h5(h5_path, [(1, 0, occluded), (1, 1, clear)])

        result = extend_track(
            source_id=1,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            hypotheses_path=h5_path,
        )
        assert result is not None
        assert result.candidate_partition == 1
        assert result.existing_overlap == 0.0

    def test_paint_policy(self, tmp_path):
        """Paint policy: wipes existing source_id at target frame, preserves other cells, avoids collisions."""
        H2, W2 = 40, 40
        tracked = np.zeros((2, H2, W2), dtype=np.uint32)
        # Source cell at t=0
        tracked[0, 5:15, 5:15] = 7     # source cell, centroid=(9.5, 9.5), area=100
        # At t=1: source_id already exists somewhere (must be wiped), plus another cell
        tracked[1, 30:35, 30:35] = 7   # stale source_id pixels at target frame
        tracked[1, 8:12, 8:12] = 99    # another cell overlapping candidate bbox

        # Hypothesis at t=1: 10×10 block overlapping both clear and occupied pixels
        lm = np.zeros((H2, W2), dtype=np.uint32)
        lm[6:16, 6:16] = 1             # centroid=(10.5, 10.5), dist≈1.4, area=100

        h5_path = tmp_path / "hypotheses.h5"
        _write_hyp_h5(h5_path, [(1, 0, lm)])

        result = extend_track(
            source_id=7,
            source_frame=0,
            direction="forward",
            tracked_labels=tracked,
            hypotheses_path=h5_path,
        )
        assert result is not None

        # Apply paint policy
        t_prime = result.target_frame
        frame = tracked[t_prime].copy()

        # Step 1: wipe existing source_id at target frame
        frame[frame == 7] = 0

        # Step 2: paint candidate pixels where empty, respect collisions
        y0, x0, y1, x1 = result.bbox
        for y in range(y0, y1):
            for x in range(x0, x1):
                if result.mask_2d[y, x] and frame[y, x] == 0:
                    frame[y, x] = 7

        # Stale source_id must have been wiped
        assert not np.any(frame[30:35, 30:35] == 7)

        # Candidate pixels that were clear should now be source_id=7
        # e.g. corners of [6:16, 6:16] outside [8:12, 8:12]
        assert frame[6, 6] == 7

        # Collision area: cell 99 must be preserved (not overwritten)
        assert np.all(frame[8:12, 8:12] == 99)
