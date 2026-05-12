"""Nucleus segmentation via watershed on Cellpose probability maps."""
from __future__ import annotations

from cellflow.segmentation.flow_following import (
    FlowFollowingParams,
    compute_filtered_flow_vectors,
    compute_flow_following_movie,
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

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

_LABEL_DTYPE = np.uint32


def apply_gamma(logits: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma-correct Cellpose probability logits: sigmoid → power → logit."""
    if gamma == 1.0:
        return logits
    probs = 1.0 / (1.0 + np.exp(-logits))
    probs = np.clip(np.power(probs, gamma), 1e-7, 1 - 1e-7)
    return np.log(probs / (1.0 - probs))


def _validate_foreground_gamma(gamma: float) -> float:
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    return gamma


def _validate_foreground_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return threshold


def _apply_post_average_gamma(score: np.ndarray, gamma: float) -> np.ndarray:
    score = np.clip(score, 0.0, 1.0).astype(np.float32, copy=False)
    if gamma == 1.0:
        return score
    return np.power(score, gamma).astype(np.float32)


def _normalize_foreground_score(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    lo = float(np.min(score))
    hi = float(np.max(score))
    if hi <= lo:
        return np.zeros_like(score, dtype=np.float32)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _flow_dp_magnitude_stack(data: np.ndarray) -> tuple[np.ndarray, bool]:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 3:
        return np.abs(data), False
    if data.ndim == 4:
        if data.shape[-1] in (2, 3):
            return np.sqrt(np.sum(data * data, axis=-1)).astype(np.float32), False
        if data.shape[1] in (2, 3):
            return np.sqrt(np.sum(data * data, axis=1)).astype(np.float32), False
        return np.abs(data), True
    if data.ndim == 5:
        if data.shape[2] in (2, 3):
            axis = 2
        elif data.shape[-1] in (2, 3):
            axis = -1
        else:
            raise ValueError(f"Unsupported flow_dp shape {data.shape}")
        return np.sqrt(np.sum(data * data, axis=axis)).astype(np.float32), True
    raise ValueError(f"Unsupported flow_dp shape {data.shape}")


def foreground_score_stack(data, source: str, gamma: float = 1.0) -> np.ndarray:
    """Return a foreground score image or time stack from probability or flow-DP data."""
    gamma = _validate_foreground_gamma(gamma)
    source_key = str(source).lower()
    arr = np.asarray(data, dtype=np.float32)

    if source_key == "probability":
        if arr.ndim == 3:
            score = 1.0 / (1.0 + np.exp(-arr.mean(axis=0)))
        elif arr.ndim == 4:
            score = 1.0 / (1.0 + np.exp(-arr.mean(axis=1)))
        else:
            raise ValueError(f"Unsupported probability shape {arr.shape}")
        return _apply_post_average_gamma(score, gamma)

    if source_key == "flow_dp":
        magnitude, has_time_axis = _flow_dp_magnitude_stack(arr)
        if has_time_axis:
            score = magnitude.mean(axis=1)
            normalized = np.empty_like(score, dtype=np.float32)
            for t in range(score.shape[0]):
                normalized[t] = _normalize_foreground_score(score[t])
        else:
            normalized = _normalize_foreground_score(magnitude.mean(axis=0))
        return _apply_post_average_gamma(normalized, gamma)

    raise ValueError(f"Unsupported foreground source {source!r}")


def foreground_mask_stack(
    data,
    source: str,
    threshold: float = 0.5,
    gamma: float = 1.0,
) -> np.ndarray:
    """Return a uint8 foreground mask with values 0/1."""
    threshold = _validate_foreground_threshold(threshold)
    score = foreground_score_stack(data, source, gamma=gamma)
    return (score >= threshold).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class ContourWatershedParams:
    """Parameters for contour-map watershed hypothesis generation."""

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
        if not coords or coords[0].size == 0:
            continue
        slices = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coords)
        filled = binary_fill_holes(labels[slices] == label_id)
        out_view = out[slices]
        out_view[filled] = label_id
    return out







def build_consensus_boundary(
    prob_3d: np.ndarray,
    dp_3d: np.ndarray,
    cellprob_thresholds: list[float],
    gamma: float = 1.0,
    flow_threshold: float = 0.0,
    reduction: str = "mean",
    *,
    mask_callback: Callable[[np.ndarray, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce mask boundaries and occupancy over (threshold × z-slice).

    prob_3d: (Z, Y, X) logits  dp_3d: (Z, 2, Y, X)
    reduction: "mean" averages across all (threshold × z-slice) combinations;
               "max" takes the per-pixel maximum instead.
    mask_callback: optional sink called as mask_callback(masks_zyx, thresh_idx) after each threshold.
    Returns: (boundary, foreground) both (Y, X) float32.
    """
    try:
        import torch
        from cellpose.dynamics import compute_masks
        from skimage.segmentation import find_boundaries
    except ImportError as exc:
        raise ImportError("cellpose, torch, and scikit-image required") from exc

    prob_3d = apply_gamma(np.asarray(prob_3d, dtype=np.float32), gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_z = prob_3d.shape[0]
    accum = np.zeros(prob_3d.shape[1:], dtype=np.float32)
    foreground_accum = np.zeros(prob_3d.shape[1:], dtype=np.float32)
    n_total = 0

    for i_thresh, thresh in enumerate(cellprob_thresholds):
        z_masks: list[np.ndarray] = []
        for z in range(n_z):
            result = compute_masks(
                dp_3d[z], prob_3d[z],
                cellprob_threshold=float(thresh),
                flow_threshold=float(flow_threshold),
                niter=200,
                do_3D=False,
                device=device,
            )
            masks = result[0] if isinstance(result, tuple) else result
            masks_arr = np.asarray(masks)
            boundary_slice = find_boundaries(masks_arr, mode="inner").astype(np.float32)
            fg_slice = (masks_arr > 0).astype(np.float32)
            if reduction == "max":
                np.maximum(accum, boundary_slice, out=accum)
                np.maximum(foreground_accum, fg_slice, out=foreground_accum)
            else:
                accum += boundary_slice
                foreground_accum += fg_slice
            n_total += 1
            if mask_callback is not None:
                z_masks.append(np.asarray(masks_arr, dtype=np.uint32))
        if mask_callback is not None:
            mask_callback(np.stack(z_masks), i_thresh)

    if reduction == "max":
        return accum, foreground_accum
    boundary = accum / n_total if n_total > 0 else accum
    foreground = foreground_accum / n_total if n_total > 0 else foreground_accum
    return boundary, foreground


def build_consensus_boundary_2d(
    prob_yx: np.ndarray,
    dp_cyx: np.ndarray,
    cellprob_thresholds: list[float],
    flow_threshold: float = 0.0,
    reduction: str = "mean",
    niter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Build consensus boundary from a Z-averaged probability map and 2D flow vectors.

    prob_yx:  (Y, X) Cellpose probability logits — already Z-projected and gamma-corrected.
    dp_cyx:   (2, Y, X) flow vectors (e.g. from filtered_dp).
    Returns:  (boundary, foreground) both (Y, X) float32.
    """
    try:
        import torch
        from cellpose.dynamics import compute_masks
        from skimage.segmentation import find_boundaries
    except ImportError as exc:
        raise ImportError("cellpose, torch, and scikit-image required") from exc

    prob_yx = np.asarray(prob_yx, dtype=np.float32)
    dp_cyx = np.asarray(dp_cyx, dtype=np.float32)
    if prob_yx.ndim != 2:
        raise ValueError(f"Expected (Y, X) prob, got {prob_yx.shape}")
    if dp_cyx.ndim != 3 or dp_cyx.shape[0] != 2:
        raise ValueError(f"Expected (2, Y, X) dp, got {dp_cyx.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accum = np.zeros(prob_yx.shape, dtype=np.float32)
    foreground_accum = np.zeros(prob_yx.shape, dtype=np.float32)

    for thresh in cellprob_thresholds:
        result = compute_masks(
            dp_cyx,
            prob_yx,
            cellprob_threshold=float(thresh),
            flow_threshold=float(flow_threshold),
            niter=int(niter),
            do_3D=False,
            device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        masks_arr = np.asarray(masks)
        boundary_slice = find_boundaries(masks_arr, mode="inner").astype(np.float32)
        fg_slice = (masks_arr > 0).astype(np.float32)
        if reduction == "max":
            np.maximum(accum, boundary_slice, out=accum)
            np.maximum(foreground_accum, fg_slice, out=foreground_accum)
        else:
            accum += boundary_slice
            foreground_accum += fg_slice

    n = len(cellprob_thresholds)
    if reduction != "max" and n > 0:
        accum /= n
        foreground_accum /= n

    return accum, foreground_accum



def compute_cellpose_foreground_masks(
    prob_tzyx: np.ndarray,
    filtered_dp_tcyx: np.ndarray,
    *,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.0,
    min_size: int = 15,
    niter: int = 200,
    progress_cb: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Generate binary cell foreground masks with Cellpose dynamics.

    prob_tzyx is Cellpose probability logits shaped (T, Z, Y, X), or a single
    volume shaped (Z, Y, X). filtered_dp_tcyx must be the filtered flow stack
    produced by the cell workflow, shaped (T, 2, Y, X).
    """
    prob = np.asarray(prob_tzyx, dtype=np.float32)
    if prob.ndim == 3:
        prob = prob[np.newaxis]
    if prob.ndim != 4:
        raise ValueError(
            f"Expected probability shape (T, Z, Y, X) or (Z, Y, X), got {prob.shape}"
        )

    filtered_dp = np.asarray(filtered_dp_tcyx, dtype=np.float32)
    if filtered_dp.ndim != 4 or filtered_dp.shape[1] != 2:
        raise ValueError(
            f"Expected filtered flow shape (T, 2, Y, X), got {filtered_dp.shape}"
        )
    if prob.shape[0] != filtered_dp.shape[0] or prob.shape[2:] != filtered_dp.shape[2:]:
        raise ValueError(
            "Cellpose probability and filtered flow shapes do not match: "
            f"probability {prob.shape}, filtered flow {filtered_dp.shape}"
        )

    try:
        import torch
        from cellpose.dynamics import compute_masks
    except ImportError as exc:
        raise ImportError(
            "cellpose and torch must be installed to generate Cellpose foreground masks"
        ) from exc

    prob_tyx = prob.mean(axis=1).astype(np.float32, copy=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = np.zeros(prob_tyx.shape, dtype=np.uint8)

    for t in range(prob_tyx.shape[0]):
        result = compute_masks(
            filtered_dp[t],
            prob_tyx[t],
            cellprob_threshold=float(cellprob_threshold),
            flow_threshold=float(flow_threshold),
            min_size=int(min_size),
            niter=int(niter),
            do_3D=False,
            device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        out[t] = (np.asarray(masks) > 0).astype(np.uint8)
        if progress_cb is not None:
            progress_cb(t + 1, prob_tyx.shape[0])

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

    # Apply correlated noise perturbation
    if params.noise_scale > 0:
        noise = np.random.normal(0, params.noise_scale, boundary_pre.shape)
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
