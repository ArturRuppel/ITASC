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

    # Atom extraction (stage ①) — see atoms.AtomParams
    fg_window: int = 51
    fg_cutoff: float = 0.002
    fg_strength: float = 1.0
    contour_window: int = 51
    contour_floor: float = 0.01
    contour_strength: float = 1.0
    atom_min_area: int = 100

    # Atom-union candidate enumeration (stage ②)
    atom_union_max_atoms: int = 3
    atom_union_max_area: int = 8000

    # Linking
    # Ultrack's multiprocessing_apply activates a Pool only when n_workers > 1,
    # and batch_index_range uses n_workers as a window stride (0 would div-zero).
    # Default to min(cpu_count(), 8) for automatic parallelism up to 8 threads.
    max_distance: float = 15.0
    max_neighbors: int = 5
    distance_weight: float = 0.05
    link_n_workers: int = min(cpu_count(), 8)
    linking_mode: str = "default"  # "default" or "shape"
    area_weight: float = 1.0
    iou_weight: float = 1.0
    min_link_iou: float = 0.1
    min_area_ratio: float = 0.3

    # Solver / ILP
    appear_weight: float = -0.001
    disappear_weight: float = -0.001
    division_weight: float = -0.001
    link_function: str = "power"
    power: int = 4
    bias: float = 0.0
    solution_gap: float = 0.001
    time_limit: int = 36000
    window_size: int = 0  # 0 = solve all at once

    # Segmentation (ultrack.segment / ultrack.core.segmentation.processing.segment)
    seg_min_area: int = 300
    seg_max_area: int = 100_000
    seg_foreground_threshold: float = 0.5
    seg_min_frontier: float = 0.0
    seg_ws_hierarchy: str = "area"    # "area", "dynamics", or "volume"
    seg_n_workers: int = 1

    # Resolve-from-validated node prior
    quality_weight: float = 1.0
    quality_exponent: float = 8.0
    circularity_weight: float = 0.25

    # Per-frame correction primitives
    anchor_radius_px: float = 15.0
    anchor_stamp_radius_px: float = 15.0
