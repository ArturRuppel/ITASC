from cellflow.tracking_ultrack.config import TrackingConfig


def test_tracking_config_exposes_seed_prior_defaults():
    cfg = TrackingConfig()

    assert cfg.power == 4.0
    assert cfg.quality_exponent == 8.0
    assert cfg.seed_weight == 0.5
    assert cfg.seed_sigma_space == 25.0
    assert cfg.seed_tau_time == 2.0
    assert cfg.seed_max_dt == 5
    assert cfg.seed_sigma_area == 0.5


def test_tracking_config_exposes_node_probability_weights():
    cfg = TrackingConfig()

    assert cfg.quality_weight == 1.0
    assert cfg.circularity_weight == 0.25
    assert cfg.quality_exponent == 8.0
