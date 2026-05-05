"""3D temporal watershed for cell segmentation."""
from __future__ import annotations

import math

import numpy as np


def compute_3d_temporal_watershed(
    contours_tyx: np.ndarray,
    foreground_tyx: np.ndarray,
    seeds_tyx: np.ndarray,
    gaussian_sigma_space: float,
    gaussian_sigma_time: float,
    median_kernel_space: int,
    median_kernel_time: int,
    compactness_space: float,
    compactness_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """3D watershed treating time as a third spatial dimension.

    Anisotropic compactness is implemented by scaling the T axis by
    sqrt(compactness_time / compactness_space) before running skimage watershed,
    so the Euclidean distance in scaled space equals sqrt(cs·(dy²+dx²) + ct·dt²).

    Returns (smoothed_contours_tyx, labels_tyx), both at the original (T, Y, X) shape.
    smoothed_contours_tyx is float32 at original resolution (pre-scaling).
    labels_tyx is uint32.
    """
    from scipy.ndimage import gaussian_filter, median_filter, zoom
    from skimage.segmentation import watershed

    contours = np.asarray(contours_tyx, dtype=np.float32)
    foreground = np.asarray(foreground_tyx, dtype=bool)
    seeds = np.asarray(seeds_tyx, dtype=np.int32)

    if gaussian_sigma_time != 0.0 or gaussian_sigma_space != 0.0:
        contours = gaussian_filter(
            contours,
            sigma=(float(gaussian_sigma_time), float(gaussian_sigma_space), float(gaussian_sigma_space)),
        )

    if median_kernel_time != 1 or median_kernel_space != 1:
        contours = median_filter(
            contours,
            size=(int(median_kernel_time), int(median_kernel_space), int(median_kernel_space)),
        )

    smoothed = contours.copy()

    use_scaling = compactness_space > 0.0 and compactness_time > 0.0
    scale = math.sqrt(compactness_time / compactness_space) if use_scaling else 1.0
    effective_compactness = compactness_space if use_scaling else 0.0

    if use_scaling and scale != 1.0:
        scaled_contours = zoom(contours, (scale, 1.0, 1.0), order=1)
        scaled_foreground = zoom(foreground.astype(np.float32), (scale, 1.0, 1.0), order=1) >= 0.5
        scaled_seeds = zoom(seeds, (scale, 1.0, 1.0), order=0)
    else:
        scaled_contours = contours
        scaled_foreground = foreground
        scaled_seeds = seeds

    labels_scaled = watershed(
        scaled_contours,
        markers=scaled_seeds,
        mask=scaled_foreground,
        compactness=float(effective_compactness),
        watershed_line=False,
    )

    if use_scaling and scale != 1.0:
        target_t = contours_tyx.shape[0]
        labels = zoom(np.asarray(labels_scaled), (1.0 / scale, 1.0, 1.0), order=0)
        if labels.shape[0] > target_t:
            labels = labels[:target_t]
        elif labels.shape[0] < target_t:
            pad = np.zeros(
                (target_t - labels.shape[0],) + labels.shape[1:], dtype=labels.dtype
            )
            labels = np.concatenate([labels, pad], axis=0)
    else:
        labels = np.asarray(labels_scaled)

    return smoothed.astype(np.float32), labels.astype(np.uint32)
