"""Nucleus segmentation: contour-watershed and consensus boundary."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile

def apply_gamma(logits: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma-correct Cellpose probability logits: sigmoid → power → logit."""
    if gamma == 1.0:
        return logits
    probs = 1.0 / (1.0 + np.exp(-logits))
    probs = np.clip(np.power(probs, gamma), 1e-7, 1 - 1e-7)
    return np.log(probs / (1.0 - probs))

_LABEL_DTYPE = np.uint32


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


@dataclass(frozen=True, slots=True)
class NucleusAveragedMapsReport:
    """Summary for simplified nucleus averaged-map generation."""

    frames: int
    z_indices: tuple[int, ...]
    cellprob_thresholds: tuple[float, ...]
    contours_path: Path
    foreground_scores_path: Path


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


def _as_tzyx(stack: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4:
        raise ValueError(f"{name} must be Z×Y×X or T×Z×Y×X.")
    return arr


def _as_tzcyx(stack: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 5 or arr.shape[2] != 2:
        raise ValueError(f"{name} must be Z×2×Y×X or T×Z×2×Y×X.")
    return arr


def _normalize_cellprob_thresholds(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    thresholds = tuple(float(v) for v in values)
    if not thresholds:
        raise ValueError("cellprob_thresholds must include at least one threshold.")
    return thresholds


def _normalize_z_indices(z_indices: list[int] | tuple[int, ...] | None, n_z: int) -> tuple[int, ...]:
    if z_indices is None:
        indices = tuple(range(n_z))
    else:
        indices = tuple(int(z) for z in z_indices)
    if not indices:
        raise ValueError("z_indices must include at least one z slice.")
    bad = [z for z in indices if z < 0 or z >= n_z]
    if bad:
        raise ValueError(f"z_indices out of range for {n_z} z slices: {bad}")
    return indices


def build_nucleus_averaged_maps(
    nucleus_prob_path: str | Path,
    nucleus_dp_path: str | Path,
    contours_path: str | Path,
    foreground_scores_path: str | Path,
    *,
    cellprob_thresholds: list[float] | tuple[float, ...],
    z_indices: list[int] | tuple[int, ...] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> NucleusAveragedMapsReport:
    """Build ``contours.tif`` and ``foreground_scores.tif`` from Cellpose outputs.

    This is the simplified Stage A nucleus pipeline: sweep Cellpose probability
    thresholds and selected z-slices, average labels into continuous contour and
    foreground-score maps, and write one ``T×Y×X`` stack for each signal.
    """
    prob_stack = _as_tzyx(tifffile.imread(str(nucleus_prob_path)), "nucleus_prob")
    dp_stack = _as_tzcyx(tifffile.imread(str(nucleus_dp_path)), "nucleus_dp")
    if prob_stack.shape[0] != dp_stack.shape[0]:
        raise ValueError("nucleus_prob and nucleus_dp must have the same frame count.")
    if prob_stack.shape[1] != dp_stack.shape[1]:
        raise ValueError("nucleus_prob and nucleus_dp must have the same z count.")
    if prob_stack.shape[2:] != dp_stack.shape[3:]:
        raise ValueError("nucleus_prob and nucleus_dp must have the same Y×X shape.")

    thresholds = _normalize_cellprob_thresholds(cellprob_thresholds)
    z_sel = _normalize_z_indices(z_indices, prob_stack.shape[1])

    contour_frames: list[np.ndarray] = []
    foreground_frames: list[np.ndarray] = []
    n_t = int(prob_stack.shape[0])
    for t in range(n_t):
        if progress_cb is not None:
            progress_cb(t + 1, n_t, f"Building averaged maps: frame {t + 1}/{n_t}")
        contours, foreground = build_consensus_boundary(
            prob_stack[t, z_sel],
            dp_stack[t, z_sel],
            list(thresholds),
            gamma=1.0,
            flow_threshold=0.0,
        )
        contour_frames.append(np.asarray(contours, dtype=np.float32))
        foreground_frames.append(np.asarray(foreground, dtype=np.float32))

    contours_path = Path(contours_path)
    foreground_scores_path = Path(foreground_scores_path)
    contours_path.parent.mkdir(parents=True, exist_ok=True)
    foreground_scores_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(contours_path), np.stack(contour_frames), compression="zlib")
    tifffile.imwrite(
        str(foreground_scores_path),
        np.stack(foreground_frames),
        compression="zlib",
    )
    return NucleusAveragedMapsReport(
        frames=n_t,
        z_indices=z_sel,
        cellprob_thresholds=thresholds,
        contours_path=contours_path,
        foreground_scores_path=foreground_scores_path,
    )


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
