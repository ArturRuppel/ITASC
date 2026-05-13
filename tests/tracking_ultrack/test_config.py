import sys
import types

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


def test_build_ultrack_config_forwards_bias_to_tracking_config(monkeypatch, tmp_path):
    from cellflow.tracking_ultrack.ingest import _build_ultrack_config

    captured = {}

    class FakeSegmentationConfig:
        min_area = None
        max_area = None
        threshold = None
        min_frontier = None
        ws_hierarchy = None
        n_workers = None

    class FakeMainConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.segmentation_config = FakeSegmentationConfig()

    ultrack_config = types.ModuleType("ultrack.config")
    ultrack_config.MainConfig = FakeMainConfig
    segmentationconfig = types.ModuleType("ultrack.config.segmentationconfig")
    segmentationconfig.NAME_TO_WS_HIER = {
        "area": "area",
        "dynamics": "dynamics",
        "volume": "volume",
    }

    monkeypatch.setitem(sys.modules, "ultrack.config", ultrack_config)
    monkeypatch.setitem(
        sys.modules,
        "ultrack.config.segmentationconfig",
        segmentationconfig,
    )

    _build_ultrack_config(TrackingConfig(bias=-0.5), tmp_path)

    assert captured["tracking"]["bias"] == -0.5
