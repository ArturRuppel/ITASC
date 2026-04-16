"""Pydantic configuration model for cellflow-ultrack tracking stage."""
from __future__ import annotations

from pydantic import BaseModel


class TrackingConfig(BaseModel):
    """Parameters for Ultrack tracking (tracking stage)."""

    # Segmentation hypothesis
    min_area: int = 100
    max_area: int = 1000000
    min_frontier: float = 0.0
    threshold: float = 0.5
    ws_hierarchy: str = "area"  # "area", "dynamics", or "volume"
    anisotropy_penalization: float = 0.0
    n_workers: int = 1  # parallel workers for segmentation; >1 uses a zarr temp store

    # Linking
    max_distance: float = 15.0
    max_neighbors: int = 5
    distance_weight: float = 0.0
    link_n_workers: int = 1

    # Solver / ILP
    appear_weight: float = -0.001
    disappear_weight: float = -0.001
    division_weight: float = -0.001
    link_function: str = "power"  # "power" or "identity"
    power: float = 4.0
    bias: float = 0.0
    solution_gap: float = 0.001
    time_limit: int = 36000
    window_size: int = 0  # 0 = solve all at once

    # Per-stage overwrite flags
    overwrite_segmentation: bool = True
    overwrite_linking: bool = True
    overwrite_solve: bool = True
