from __future__ import annotations

import numpy as np
import pytest
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.seed_prior import (
    compute_drop_frac,
    compute_mask_circularity,
    write_seed_prior_node_probs,
)


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


def test_write_seed_prior_node_probs_uses_drop_quality_and_best_seed_affinity(tmp_path):
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
        seed_weight=0.5,
        seed_sigma_space=25.0,
        seed_tau_time=2.0,
        seed_max_dt=5,
        seed_sigma_area=0.5,
    )

    report = write_seed_prior_node_probs(tmp_path, image_path, cfg)

    assert report.scored == 1
    with Session(engine) as session:
        candidate = session.query(NodeDB).filter_by(t=0, id=1_000_001).one()
        seed = session.query(NodeDB).filter_by(t=1, id=2_000_001).one()

    circularity = compute_mask_circularity(np.ones((4, 4), dtype=bool))
    expected_affinity = np.exp(-(1 / cfg.seed_tau_time))
    expected = (
        cfg.quality_weight * 1.0
        + cfg.circularity_weight * circularity
        + cfg.seed_weight * expected_affinity
    )
    assert candidate.node_prob == pytest.approx(expected)
    assert seed.node_prob == 1.0


def test_write_seed_prior_node_probs_respects_zero_quality_and_circularity_weights(tmp_path):
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
        seed_weight=0.0,
    )

    report = write_seed_prior_node_probs(tmp_path, image_path, cfg)

    assert report.scored == 1
    with Session(engine) as session:
        candidate = session.query(NodeDB).filter_by(t=0, id=1_000_001).one()
    assert candidate.node_prob == pytest.approx(0.0)
