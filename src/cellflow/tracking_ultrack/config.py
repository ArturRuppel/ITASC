"""Pydantic configuration model for the Ultrack-based tracking stage."""
from __future__ import annotations

from multiprocessing import cpu_count

from pydantic import BaseModel


class TrackingConfig(BaseModel):
    # Node area filters (applied before NodeDB insert)
    min_area: int = 100
    max_area: int = 1_000_000

    # ID scheme — must match ultrack._generate_id
    max_segments_per_time: int = 1_000_000

    # Linking
    # Ultrack's multiprocessing_apply activates a Pool only when n_workers > 1,
    # and batch_index_range uses n_workers as a window stride (0 would div-zero).
    # Default to min(cpu_count(), 8) for automatic parallelism up to 8 threads.
    max_distance: float = 15.0
    max_neighbors: int = 5
    distance_weight: float = 0.0
    link_n_workers: int = min(cpu_count(), 8)
    linking_mode: str = "default"  # "default" or "iou"
    iou_weight: float = 1.0
    min_link_iou: float = 0.1

    # Solver / ILP
    appear_weight: float = -0.001
    disappear_weight: float = -0.001
    division_weight: float = -0.001
    link_function: str = "power"
    power: float = 4.0
    bias: float = 0.0
    solution_gap: float = 0.001
    time_limit: int = 36000
    window_size: int = 0  # 0 = solve all at once
