"""Nucleus segmentation: contour-watershed and consensus boundary."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from cellflow.core.cancellation import CancelledError

__all__ = ["CancelledError"]


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise CancelledError("Operation cancelled.")


_LABEL_DTYPE = np.uint32


@dataclass(frozen=True, slots=True)
class ContourWatershedParams:
    """Parameters for contour-map watershed candidate generation."""

    seed_distance: int = 10
    foreground_threshold: float = 0.5
    ridge_threshold: float = 0.5
    min_size: int = 0
    min_circularity: float = 0.0
    noise_scale: float = 0.0
    noise_blur_sigma: float = 0.0
    run_index: int = 0

    def to_dict(self) -> dict[str, object]:
        return {"method": "contour_watershed", **asdict(self)}


def _remove_small_labels(labels: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return labels
    ids, counts = np.unique(labels, return_counts=True)
    small = ids[(ids > 0) & (counts < min_size)]
    if small.size == 0:
        return labels
    out = labels.copy()
    out[np.isin(labels, small)] = 0
    return out


def _remove_low_circularity_labels(labels: np.ndarray, min_circularity: float) -> np.ndarray:
    """Remove labels whose 4π·area/perimeter² is below min_circularity (0 = keep all)."""
    if min_circularity <= 0.0:
        return labels
    from skimage.measure import regionprops

    # Work on a 2D projection if labels is 3D with a single Z
    squeezed = labels.squeeze() if labels.ndim == 3 and labels.shape[0] == 1 else labels
    if squeezed.ndim != 2:
        return labels  # can't compute perimeter on >2D, skip

    import math
    remove = []
    for prop in regionprops(squeezed.astype(np.int32)):
        perimeter = prop.perimeter
        if perimeter < 1e-6:
            remove.append(prop.label)
            continue
        circularity = 4.0 * math.pi * prop.area / (perimeter ** 2)
        if circularity < min_circularity:
            remove.append(prop.label)

    if not remove:
        return labels
    out = labels.copy()
    out[np.isin(labels, remove)] = 0
    return out


def _fill_and_close_labels(labels: np.ndarray) -> np.ndarray:
    """Fill interior holes per label."""
    from scipy.ndimage import binary_fill_holes

    out = np.zeros_like(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        coords = np.nonzero(labels == label_id)
        # np.nonzero always returns a tuple of ndim arrays (truthy), so the only
        # real emptiness signal is a zero-length coordinate array.
        if coords[0].size == 0:
            continue
        slices = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coords)
        filled = binary_fill_holes(labels[slices] == label_id)
        out_view = out[slices]
        out_view[filled] = label_id
    return out


def compute_contour_watershed(
    boundary: np.ndarray,
    foreground_mask: np.ndarray,
    params: ContourWatershedParams,
) -> np.ndarray:
    """Run seeded watershed on a consensus boundary image and binary foreground mask.

    Seeds are placed at EDT maxima of fg_mask & (boundary < ridge_threshold),
    so contour ridges separating touching cells drive seed placement rather than
    foreground intensity peaks.

    boundary:   (Y, X) float32 — high at cell borders
    foreground_mask: (Y, X) binary — nonzero pixels are allowed segmentation area
    Returns:    (Y, X) uint32 label image
    """
    from scipy.ndimage import label as nd_label
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    boundary = np.asarray(boundary, dtype=np.float32)
    foreground_mask = np.asarray(foreground_mask)
    if foreground_mask.shape != boundary.shape:
        raise ValueError(
            f"Foreground mask shape {foreground_mask.shape} does not match boundary shape {boundary.shape}"
        )
    fg_mask = foreground_mask > 0

    boundary_pre = np.asarray(boundary, dtype=np.float32).copy()

    # Apply correlated noise perturbation. Seed a local RNG from run_index so
    # repeated runs (and multi-run ensembles) are reproducible — the previous
    # global np.random.normal left run_index dead and the output non-deterministic.
    if params.noise_scale > 0:
        rng = np.random.default_rng(params.run_index)
        noise = rng.normal(0, params.noise_scale, boundary_pre.shape)
        if params.noise_blur_sigma > 0:
            noise = gaussian_filter(noise, sigma=params.noise_blur_sigma)
        boundary_pre = np.clip(boundary_pre + noise, 0, 1)

    boundary_pre[boundary_pre < params.foreground_threshold] = 0

    from scipy.ndimage import distance_transform_edt

    # Carve strong contour ridges out of the mask so touching cells become
    # separate connected components before seeding.
    core = fg_mask & (boundary_pre < params.ridge_threshold)
    edt = distance_transform_edt(core)

    coords = peak_local_max(
        edt,
        min_distance=max(1, int(params.seed_distance)),
        threshold_abs=1.0,
        exclude_border=False,
    )
    marker_mask = np.zeros(boundary_pre.shape, dtype=bool)
    if coords.size:
        marker_mask[coords[:, 0], coords[:, 1]] = True
    markers, _ = nd_label(marker_mask)

    # Watershed floods fg_mask (not core) so basins fill back over carved ridges.
    labels = watershed(boundary_pre, markers=markers, mask=fg_mask, watershed_line=False)
    result = _fill_and_close_labels(np.asarray(labels, dtype=_LABEL_DTYPE))
    result = _remove_small_labels(result, params.min_size)
    return _remove_low_circularity_labels(result, params.min_circularity)
