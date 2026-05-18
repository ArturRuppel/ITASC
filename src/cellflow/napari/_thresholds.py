"""Threshold and z-slice control parsing for the nucleus workflow widget."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget


def thresholds_from_values(
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    *,
    label: str,
) -> np.ndarray:
    if threshold_step <= 0:
        raise ValueError(f"{label} threshold step must be > 0.")
    if threshold_min > threshold_max:
        raise ValueError(f"{label} threshold min must be <= max.")
    if threshold_min < 0 or threshold_max > 1:
        raise ValueError(f"{label} thresholds must be between 0 and 1.")
    return np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)


def source_contour_thresholds(w: NucleusWorkflowWidget) -> np.ndarray:
    return thresholds_from_values(
        float(w.source_contour_threshold_min_spin.value()),
        float(w.source_contour_threshold_max_spin.value()),
        float(w.source_contour_threshold_step_spin.value()),
        label="Contour",
    )


def source_foreground_thresholds(w: NucleusWorkflowWidget) -> np.ndarray:
    return thresholds_from_values(
        float(w.source_foreground_threshold_min_spin.value()),
        float(w.source_foreground_threshold_max_spin.value()),
        float(w.source_foreground_threshold_step_spin.value()),
        label="Foreground",
    )
