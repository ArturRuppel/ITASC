"""Segmentation package — shared utilities and submodule re-exports."""
from __future__ import annotations

from cellflow.segmentation.contour_filtering import (
    ContourFilterParams,
    compute_filtered_contour_maps,
)

from cellflow.segmentation.cell_label_icm import (
    CellICMState,
    CellLabelICMParams,
    assemble_cost_field,
    balance_strength_to_weights,
    commit_labels,
    initialize_icm,
)

from cellflow.segmentation.cell_divergence_segmentation import (
    CellDivergenceParams,
    CellDivergenceResult,
    CellForegroundResult,
    clean_and_smooth_contours,
    compute_cell_foreground,
    segment_cells_divergence,
)

from cellflow.segmentation.nucleus_segmentation import (
    CancelledError,
    _fill_and_close_labels,
)

from cellflow.segmentation.lineage import (
    LineageModel,
    TrackLane,
    TrackSegment,
    build_lineage,
)

__all__ = [
    "CancelledError",
    "CellDivergenceParams",
    "CellDivergenceResult",
    "CellForegroundResult",
    "CellICMState",
    "CellLabelICMParams",
    "ContourFilterParams",
    "LineageModel",
    "TrackLane",
    "TrackSegment",
    "assemble_cost_field",
    "balance_strength_to_weights",
    "build_lineage",
    "clean_and_smooth_contours",
    "commit_labels",
    "compute_cell_foreground",
    "compute_filtered_contour_maps",
    "initialize_icm",
    "segment_cells_divergence",
    "_fill_and_close_labels",
]
