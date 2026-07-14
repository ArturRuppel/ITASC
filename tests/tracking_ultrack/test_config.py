import sys
import types

import numpy as np
import pytest

from itasc.tracking_ultrack.config import TrackingConfig


def test_tracking_config_exposes_seed_prior_defaults():
    cfg = TrackingConfig()

    assert cfg.power == 4
    assert isinstance(cfg.power, int)
    assert cfg.quality_exponent == 1.0


def test_tracking_config_solver_power_is_integer_typed():
    cfg = TrackingConfig(power=7)

    assert cfg.power == 7
    assert isinstance(cfg.power, int)


def test_tracking_config_exposes_node_probability_weights():
    cfg = TrackingConfig()

    assert cfg.quality_weight == 1.0
    assert cfg.circularity_weight == 0.25
    assert cfg.quality_exponent == 1.0


def test_build_ultrack_config_forwards_bias_to_tracking_config(monkeypatch, tmp_path):
    from itasc.tracking_ultrack.ingest import _build_ultrack_config

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


def test_signed_power_transform_preserves_negative_edge_penalties():
    from itasc.tracking_ultrack.ingest import _signed_power_transform

    values = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    transformed = _signed_power_transform(values, power=4, bias=0.0)

    np.testing.assert_allclose(transformed, [-16.0, -0.0625, 0.0, 0.0625, 16.0])


def test_build_ultrack_config_uses_signed_power_transform(tmp_path):
    pytest.importorskip("ultrack")

    from itasc.tracking_ultrack.ingest import _build_ultrack_config

    ultrack_cfg = _build_ultrack_config(TrackingConfig(power=4), tmp_path)

    transformed = ultrack_cfg.tracking_config.apply_link_function(
        np.array([-2.0, -0.5, 0.5, 2.0])
    )

    np.testing.assert_allclose(transformed, [-16.0, -0.0625, 0.0625, 16.0])


def test_tracking_config_exposes_atom_extraction_params():
    cfg = TrackingConfig()
    assert cfg.fg_window == 51
    assert cfg.fg_cutoff == 0.002
    assert cfg.contour_window == 20
    assert cfg.contour_floor == 0.05
    assert cfg.atom_min_area == 10


def test_tracking_config_atom_params_override():
    cfg = TrackingConfig(fg_cutoff=0.01, atom_min_area=250)
    assert cfg.fg_cutoff == 0.01
    assert cfg.atom_min_area == 250
