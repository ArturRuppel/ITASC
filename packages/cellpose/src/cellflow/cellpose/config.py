"""Pydantic configuration models for cellflow-cellpose stages."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    """Raw data source configuration (s00 raw import)."""

    ndtiff_path: str
    root_dir: str
    positions: list[int]
    xy_downsample: int = 3


class CellposeConfig(BaseModel):
    """Parameters for Cellpose segmentation (nucleus_3d / cell_2d)."""

    model: str = "cpsam"
    diameter: float = 0.0
    anisotropy: float = 1.0
    min_size: int = 500
    use_gpu: bool = True
    gamma: Optional[float] = None


class CellSegmentationConfig(BaseModel):
    """Parameters for gravity-flow cell segmentation (cell_segmentation)."""

    flow_step_scale: float = 0.2
    capture_radius: float = 3.0
    flow_weight: float = 0.5
    gravity_falloff: float = 2.0
    cellpose_prob_threshold: float = 0.0
    flow_smoothing_sigma: float = 0.0
    max_iterations: int = 100

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, values):
        if isinstance(values, dict):
            for k in ("method", "flow_scale", "uniform_growth_rate",
                      "postprocess_steps", "opening_radius", "closing_radius",
                      "boundary_smoothness", "fill_holes_threshold",
                      "flow_mag_scale"):
                values.pop(k, None)
        return values


class ForegroundMaskConfig(BaseModel):
    """Parameters for the foreground mask stage (cell_foreground.tif)."""

    sigma: float = 2.0
    threshold: float = 0.1
    postprocess_steps: list = Field(default_factory=list)


class CellposeContoursConfig(BaseModel):
    """Parameters for cellpose-native contour generation (contours)."""

    cellprob_threshold: float = 0.0
    cellprob_min: float = 0.0
    cellprob_max: float = 0.0
    cellprob_step: float = 0.5
    do_3D: bool = True
    smooth_sigma: float = 0.5
    device: str = "cuda"
    save_masks: bool = False

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_cellprob(cls, values):
        """Seed min/max from cellprob_threshold when loading pre-sweep configs."""
        if isinstance(values, dict):
            if "cellprob_min" not in values and "cellprob_max" not in values:
                threshold = values.get("cellprob_threshold", 0.0)
                values["cellprob_min"] = threshold
                values["cellprob_max"] = threshold
        return values
