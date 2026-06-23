from __future__ import annotations

import numpy as np
import pytest
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.seed_prior import (
    _seed_node_prob,
    compute_drop_frac,
    compute_mask_circularity,
    write_seed_prior_node_probs,
)


def test_seed_node_prob_is_finite_at_zero_base():
    """A zero-quality node must score 0.0 for any exponent, never inf/NaN
    (0**0 == 1 and 0**negative == inf would corrupt the seed prior)."""
    import math

    for exponent in (8.0, 1.0, 0.0, -2.0, 0.5):
        p = _seed_node_prob(base=0.0, max_base=1.25, exponent=exponent)
        assert p == 0.0
        assert math.isfinite(p)


def test_seed_node_prob_normalizes_and_powers():
    # base/max_base = 0.5, ** 2 = 0.25
    assert _seed_node_prob(base=1.0, max_base=2.0, exponent=2.0) == pytest.approx(0.25)
    # max_base == 0 → use base directly (then powered)
    assert _seed_node_prob(base=0.5, max_base=0.0, exponent=1.0) == pytest.approx(0.5)


def test_seed_node_prob_clamps_negative_base():
    assert _seed_node_prob(base=-0.3, max_base=1.0, exponent=0.5) == 0.0


def test_compute_drop_frac_counts_outer_ring_pixels_below_inside_median():
    image = np.full((12, 12), 10.0, dtype=np.float32)
    image[2:8, 2:8] = np.minimum(image[2:8, 2:8], 5.0)
    image[3:7, 3:7] = 100.0
    mask = np.ones((4, 4), dtype=bool)

    assert compute_drop_frac(image, (3, 3, 7, 7), mask) == 1.0


def test_compute_mask_circularity_prefers_compact_masks():
    yy, xx = np.ogrid[:21, :21]
    disk = ((yy - 10) ** 2 + (xx - 10) ** 2) <= 6**2
    bar = np.zeros((21, 21), dtype=bool)
    bar[9:12, 2:19] = True

    disk_circularity = compute_mask_circularity(disk)
    bar_circularity = compute_mask_circularity(bar)

    assert 0.0 <= bar_circularity <= 1.0
    assert 0.0 <= disk_circularity <= 1.0
    assert disk_circularity > bar_circularity


def test_compute_mask_circularity_handles_empty_masks():
    assert compute_mask_circularity(np.zeros((5, 5), dtype=bool)) == 0.0


def test_write_seed_prior_node_probs_scores_candidates_and_anchors(tmp_path):
    pytest.importorskip("ultrack")
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, VarAnnotation
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        candidate = _make_node_row(1_000_001, 0, 3, 3, 7, 7)
        seed = _make_node_row(2_000_001, 1, 3, 3, 7, 7)
        seed.t_hier_id = 0
        seed.node_annot = VarAnnotation.REAL
        session.add_all([candidate, seed])
        session.commit()

    image = np.full((2, 12, 12), 10.0, dtype=np.float32)
    image[:, 2:8, 2:8] = np.minimum(image[:, 2:8, 2:8], 5.0)
    image[:, 3:7, 3:7] = 100.0
    image_path = tmp_path / "nucleus_zavg.tif"
    tifffile.imwrite(image_path, image)

    cfg = TrackingConfig(
        quality_weight=0.7,
        quality_exponent=2.0,
        circularity_weight=0.25,
    )

    report = write_seed_prior_node_probs(tmp_path, image_path, cfg)

    assert report.scored == 1
    assert report.seeds == 1
    with Session(engine) as session:
        candidate_row = session.query(NodeDB).filter_by(t=0, id=1_000_001).one()
        seed_row = session.query(NodeDB).filter_by(t=1, id=2_000_001).one()

    circularity = compute_mask_circularity(np.ones((4, 4), dtype=bool))
    base = cfg.quality_weight * 1.0 + cfg.circularity_weight * circularity
    max_base = cfg.quality_weight + cfg.circularity_weight
    expected = (base / max_base) ** cfg.quality_exponent
    assert candidate_row.node_prob == pytest.approx(expected)
    assert seed_row.node_prob == 1.0


def test_write_seed_prior_node_probs_respects_zero_quality_and_circularity_weights(tmp_path):
    pytest.importorskip("ultrack")
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB
    from tests.tracking_ultrack.test_reseed import _make_engine, _make_node_row

    engine = _make_engine(tmp_path / "data.db")
    with Session(engine) as session:
        candidate = _make_node_row(1_000_001, 0, 3, 3, 7, 7)
        session.add(candidate)
        session.commit()

    image = np.full((1, 12, 12), 10.0, dtype=np.float32)
    image[0, 2:8, 2:8] = 5.0
    image[0, 3:7, 3:7] = 100.0
    image_path = tmp_path / "nucleus_zavg.tif"
    tifffile.imwrite(image_path, image)

    cfg = TrackingConfig(
        quality_weight=0.0,
        circularity_weight=0.0,
    )

    report = write_seed_prior_node_probs(tmp_path, image_path, cfg)

    assert report.scored == 1
    with Session(engine) as session:
        candidate = session.query(NodeDB).filter_by(t=0, id=1_000_001).one()
    assert candidate.node_prob == pytest.approx(0.0)
