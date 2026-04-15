"""Pydantic configuration models for cellflow-cellpose stages."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    """Raw data source configuration (s00 raw import)."""

    ndtiff_path: str
    root_dir: str
    positions: list[int]
    timepoints: Optional[list[int]] = None
    xy_downsample: int = 3


class CellposeConfig(BaseModel):
    """Parameters for Cellpose segmentation (nucleus_3d / cell_2d)."""

    model: str = "nuclei"
    diameter: float = 17.0
    anisotropy: float = 1.0
    min_size: int = 500
    use_gpu: bool = True
    gamma: Optional[float] = None


class FlowWatershedConfig(BaseModel):
    """Parameters for flow-guided watershed cell segmentation (flow_watershed).

    Unifies the Pydantic config from ``_config.py`` and the plain-class config
    from ``widgets/flow_watershed.py`` — all fields are in one place.
    """

    flow_scale: float = 1.0
    cellpose_prob_threshold: float = 0.0
    flow_smoothing_sigma: float = 0.0
    method: str = "distance"  # "distance" (fast) or "iterative"
    max_iterations: int = 50
    uniform_growth_rate: float = 0.2
    opening_radius: int = 1
    closing_radius: int = 1
    boundary_smoothness: float = 0.5
    fill_holes_threshold: float = 0.5


class CellposeContoursConfig(BaseModel):
    """Parameters for cellpose-native contour generation (contours)."""

    cellprob_threshold: float = 0.0
    do_3D: bool = True
    smooth_sigma: float = 0.5
    device: str = "cuda"
