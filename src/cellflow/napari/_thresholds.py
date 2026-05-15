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


def map_cellprob_thresholds(w: NucleusWorkflowWidget) -> np.ndarray:
    threshold_min = float(w.map_cellprob_min_spin.value())
    threshold_max = float(w.map_cellprob_max_spin.value())
    threshold_step = float(w.map_cellprob_step_spin.value())
    if threshold_step <= 0:
        raise ValueError("Cellprob threshold step must be > 0.")
    if threshold_min > threshold_max:
        raise ValueError("Cellprob threshold min must be <= max.")
    return np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)


def map_z_indices(w: NucleusWorkflowWidget) -> list[int] | slice | None:
    start = int(w.map_z_start_spin.value())
    stop = int(w.map_z_stop_spin.value())
    step = int(w.map_z_step_spin.value())
    if step <= 0:
        raise ValueError("Z step must be > 0.")
    if stop == -1:
        if start == 0 and step == 1:
            return None
        return slice(start, None, step)
    if start > stop:
        raise ValueError("Z start must be <= stop.")
    return list(range(start, stop + 1, step))
