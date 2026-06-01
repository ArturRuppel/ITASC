"""Segmentation package — shared utilities and submodule re-exports."""
from __future__ import annotations

import numpy as np

from cellflow.segmentation.flow_following import (
    FlowFollowingParams,
    compute_filtered_flow_vectors,
    compute_flow_following_frame,
    compute_flow_following_movie,
)

from cellflow.segmentation.contour_filtering import (
    ContourFilterParams,
    compute_filtered_contour_maps,
)

from cellflow.segmentation.cell_label_icm import (
    CellICMState,
    CellLabelICMParams,
    assemble_cost_field,
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

__all__ = [
    "CancelledError",
    "CellDivergenceParams",
    "CellDivergenceResult",
    "CellICMState",
    "CellLabelICMParams",
    "ContourFilterParams",
    "ContourWatershedParams",
    "DivergenceMapsReport",
    "FlowFollowingParams",
    "apply_gamma",
    "assemble_cost_field",
    "build_divergence_maps",
    "clean_and_smooth_contours",
    "commit_labels",
    "compute_contour_watershed",
    "compute_filtered_contour_maps",
    "compute_filtered_flow_vectors",
    "compute_flow_following_frame",
    "compute_flow_following_movie",
    "initialize_icm",
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
