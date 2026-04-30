from __future__ import annotations

import numpy as np
import pytest
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.seed_prior import compute_drop_frac, write_seed_prior_node_probs


def test_compute_drop_frac_counts_outer_ring_pixels_below_inside_median():
    image = np.full((12, 12), 10.0, dtype=np.float32)
    image[2:8, 2:8] = np.minimum(image[2:8, 2:8], 5.0)
    image[3:7, 3:7] = 100.0
    mask = np.ones((4, 4), dtype=bool)

    assert compute_drop_frac(image, (3, 3, 7, 7), mask) == 1.0


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
        quality_exponent=2.0,
        seed_weight=0.5,
        seed_sigma_space=25.0,
        seed_tau_time=2.0,
        seed_max_dt=5,
        seed_sigma_area=0.5,
    )

    report = write_seed_prior_node_probs(tmp_path, image_path, cfg)

    assert report.scored == 1
    with Session(engine) as session:
        candidate = session.get(NodeDB, 1_000_001)
        seed = session.get(NodeDB, 2_000_001)

    expected_affinity = np.exp(-(1 / cfg.seed_tau_time))
    assert candidate.node_prob == pytest.approx(1.0 + cfg.seed_weight * expected_affinity)
    assert seed.node_prob == 1.0
