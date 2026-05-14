"""Segmentation package — shared utilities and submodule re-exports."""
from __future__ import annotations

import numpy as np

from cellflow.segmentation.flow_following import (
    FlowFollowingParams,
    compute_filtered_flow_vectors,
    compute_flow_following_frame,
    build_consensus_boundary_flow_following,
)

from cellflow.segmentation.contour_filtering import (
    ContourFilterParams,
    compute_filtered_contour_maps,
)

from cellflow.segmentation.cell_label_icm import (
    CellICMState,
    CellLabelICMParams,
    commit_labels,
    initialize_icm,
    refine_icm,
)

from cellflow.segmentation.nucleus_segmentation import (
    ContourWatershedParams,
    NucleusAveragedMapsReport,
    build_consensus_boundary,
    build_nucleus_averaged_maps,
    compute_contour_watershed,
)

from cellflow.segmentation.cell_foreground import (
    compute_cellpose_foreground_masks,
)


def apply_gamma(logits: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma-correct Cellpose probability logits: sigmoid → power → logit."""
    if gamma == 1.0:
        return logits
    probs = 1.0 / (1.0 + np.exp(-logits))
    probs = np.clip(np.power(probs, gamma), 1e-7, 1 - 1e-7)
    return np.log(probs / (1.0 - probs))
