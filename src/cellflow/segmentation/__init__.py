"""Segmentation package — shared utilities and submodule re-exports."""
from __future__ import annotations

import numpy as np

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
    clean_and_smooth_contours,
    segment_cells_divergence,
)

from cellflow.segmentation.nucleus_segmentation import (
    CancelledError,
    ContourWatershedParams,
    compute_contour_watershed,
    _fill_and_close_labels,
)

from cellflow.segmentation.divergence_maps import (
    DivergenceMapsReport,
    build_divergence_maps,
)

from cellflow.segmentation.error_scan import (
    CellError,
    scan_errors,
)

from cellflow.segmentation.lineage import (
    LineageModel,
    TrackLane,
    TrackSegment,
    build_lineage,
)

from cellflow.segmentation.lineage_graph import (
    GraphEdge,
    GraphNode,
    LineageGraph,
    assign_columns,
    build_lineage_graph,
)

__all__ = [
    "CancelledError",
    "CellDivergenceParams",
    "CellDivergenceResult",
    "CellError",
    "CellICMState",
    "CellLabelICMParams",
    "ContourFilterParams",
    "ContourWatershedParams",
    "DivergenceMapsReport",
    "GraphEdge",
    "GraphNode",
    "LineageGraph",
    "LineageModel",
    "TrackLane",
    "TrackSegment",
    "apply_gamma",
    "assemble_cost_field",
    "assign_columns",
    "balance_strength_to_weights",
    "build_divergence_maps",
    "build_lineage",
    "build_lineage_graph",
    "clean_and_smooth_contours",
    "commit_labels",
    "compute_contour_watershed",
    "compute_filtered_contour_maps",
    "initialize_icm",
    "scan_errors",
    "segment_cells_divergence",
    "_fill_and_close_labels",
]


def apply_gamma(logits: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma-correct Cellpose probability logits: sigmoid → power → logit."""
    if gamma == 1.0:
        return logits
    probs = 1.0 / (1.0 + np.exp(-logits))
    probs = np.clip(np.power(probs, gamma), 1e-7, 1 - 1e-7)
    return np.log(probs / (1.0 - probs))
