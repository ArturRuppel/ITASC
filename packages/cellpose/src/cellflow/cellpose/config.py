"""Pydantic configuration models for cellflow-cellpose stages."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


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

    The post-processing pipeline is expressed as an ordered list of step dicts
    (``postprocess_steps``).  Each step has a ``"type"`` key and optional
    parameters; see :func:`~cellflow.cellpose.processing.flow_watershed_postproc.run_postprocess_pipeline`
    for the full spec.

    Old configs that carry the flat fields ``opening_radius``, ``closing_radius``,
    ``boundary_smoothness``, and ``fill_holes_threshold`` are automatically
    migrated to the new format via the ``@model_validator``.
    """

    flow_scale: float = 1.0
    cellpose_prob_threshold: float = 0.0
    flow_smoothing_sigma: float = 0.0
    max_iterations: int = 50
    uniform_growth_rate: float = 0.2
    postprocess_steps: list = Field(
        default_factory=lambda: [
            {"type": "open",            "radius":     1  },
            {"type": "close",           "radius":     1  },
            {"type": "smooth_boundary", "smoothness": 0.5},
        ]
    )
    foreground_mask_sigma: float = 2.0
    foreground_mask_threshold: float = 0.1
    foreground_mask_postprocess_steps: list = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, values):
        """Convert flat postprocess params from old configs to postprocess_steps."""
        if isinstance(values, dict):
            values.pop("method", None)  # removed field — ignore if present in old configs
        if isinstance(values, dict) and "postprocess_steps" not in values:
            steps: list[dict] = []
            opening = values.pop("opening_radius",      1)
            closing = values.pop("closing_radius",      1)
            smooth  = values.pop("boundary_smoothness", 0.5)
            values.pop("fill_holes_threshold", None)  # removed — tissue_mask replaces this
            if opening > 0: steps.append({"type": "open",            "radius":     opening})
            if closing > 0: steps.append({"type": "close",           "radius":     closing})
            if smooth  > 0: steps.append({"type": "smooth_boundary", "smoothness": smooth})
            if steps:
                values["postprocess_steps"] = steps
        return values


class ForegroundMaskConfig(BaseModel):
    """Parameters for the foreground mask stage (cell_foreground.tif)."""

    sigma: float = 2.0
    threshold: float = 0.1
    postprocess_steps: list = Field(default_factory=list)


class CellposeContoursConfig(BaseModel):
    """Parameters for cellpose-native contour generation (contours)."""

    cellprob_threshold: float = 0.0
    do_3D: bool = True
    smooth_sigma: float = 0.5
    device: str = "cuda"
