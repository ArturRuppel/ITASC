"""Filtering helpers for nucleus contour-map stacks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


@dataclass(frozen=True, slots=True)
class ContourFilterParams:
    """Parameters for spatial and temporal contour-map filtering."""

    median_kernel_time: int = 1
    median_kernel_space: int = 1
    gaussian_sigma_time: float = 0.0
    gaussian_sigma_space: float = 0.0


def _normalize_contour_stack(contours: np.ndarray) -> tuple[np.ndarray, str]:
    arr = np.asarray(contours, dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis], "yx"
    if arr.ndim == 3:
        return arr, "tyx"
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0], "tcyx"
    raise ValueError(f"Unsupported contour maps shape {arr.shape}")


def _restore_contour_stack(contours_tyx: np.ndarray, layout: str) -> np.ndarray:
    if layout == "yx":
        return contours_tyx[0]
    if layout == "tcyx":
        return contours_tyx[:, np.newaxis]
    return contours_tyx


def compute_filtered_contour_maps(
    contours: np.ndarray,
    params: ContourFilterParams,
) -> np.ndarray:
    """Return contour maps after median and Gaussian filtering."""
    filtered, layout = _normalize_contour_stack(contours)
    if params.median_kernel_time > 1 or params.median_kernel_space > 1:
        filtered = median_filter(
            filtered,
            size=(
                int(params.median_kernel_time),
                int(params.median_kernel_space),
                int(params.median_kernel_space),
            ),
        )
    if params.gaussian_sigma_time > 0.0 or params.gaussian_sigma_space > 0.0:
        filtered = gaussian_filter(
            filtered,
            sigma=(
                float(params.gaussian_sigma_time),
                float(params.gaussian_sigma_space),
                float(params.gaussian_sigma_space),
            ),
        )
    return _restore_contour_stack(
        np.asarray(filtered, dtype=np.float32),
        layout,
    )
