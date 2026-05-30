"""Ultrack configuration helpers for CellFlow tracking."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig


def _signed_power_transform(values, *, power: float, bias: float):
    values = np.asarray(values)
    return np.sign(values) * np.power(np.abs(values), power) + bias


def _install_signed_power_transform(ultrack_cfg) -> None:
    tracking_cfg = getattr(ultrack_cfg, "tracking_config", None)
    if tracking_cfg is None:
        return
    link_function = getattr(tracking_cfg, "link_function", None)
    if getattr(link_function, "value", link_function) != "power":
        return
    if not hasattr(tracking_cfg, "dict"):
        return

    base_cls = tracking_cfg.__class__

    class _CellFlowSignedPowerTrackingConfig(base_cls):
        @property
        def apply_link_function(self):
            return lambda values: _signed_power_transform(
                values,
                power=self.power,
                bias=self.bias,
            )

    ultrack_cfg.tracking_config = _CellFlowSignedPowerTrackingConfig.parse_obj(
        tracking_cfg.dict()
    )


def _build_ultrack_config(cfg: TrackingConfig, working_dir: Path):
    from ultrack.config import MainConfig
    from ultrack.config.segmentationconfig import NAME_TO_WS_HIER

    ultrack_cfg = MainConfig(
        data={"working_dir": str(working_dir)},
        linking={
            "max_distance": cfg.max_distance,
            "max_neighbors": cfg.max_neighbors,
            "distance_weight": cfg.distance_weight,
            "n_workers": cfg.link_n_workers,
        },
        tracking={
            "solver_name": _select_solver(),
            "appear_weight": cfg.appear_weight,
            "disappear_weight": cfg.disappear_weight,
            "division_weight": cfg.division_weight,
            "link_function": cfg.link_function,
            "power": cfg.power,
            "bias": cfg.bias,
            "solution_gap": cfg.solution_gap,
            "time_limit": cfg.time_limit,
            "window_size": cfg.window_size if cfg.window_size > 0 else None,
        },
    )
    _install_signed_power_transform(ultrack_cfg)
    sc = ultrack_cfg.segmentation_config
    sc.min_area = cfg.seg_min_area
    sc.max_area = cfg.seg_max_area
    sc.threshold = cfg.seg_foreground_threshold
    sc.min_frontier = cfg.seg_min_frontier
    sc.ws_hierarchy = NAME_TO_WS_HIER[cfg.seg_ws_hierarchy]
    sc.n_workers = cfg.seg_n_workers
    return ultrack_cfg


def _select_solver() -> str:
    try:
        import gurobipy  # noqa: F401
        return "GUROBI"
    except ImportError:
        return "CBC"
