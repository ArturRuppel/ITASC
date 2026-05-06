"""Nucleus segmentation via watershed on Cellpose probability maps."""
from __future__ import annotations

from cellflow.segmentation.flow_following import (
    FlowFollowingParams,
    compute_flow_following_movie,
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


@dataclass(frozen=True, slots=True)
class CellposeFlowHypothesisParams:
    """Parameters for native Cellpose flow-based mask generation (no sweep)."""

    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.0   # 0 = disabled; >0 removes masks with high flow error
    min_size: int = 15
    niter: int = 200

    def to_dict(self) -> dict[str, object]:
        return {"method": "cellpose_flow", **asdict(self)}


@dataclass(frozen=True, slots=True)
class NucleusHypothesisParams:
    """One parameter set for nucleus hypothesis generation."""

    basin: str = "prob"
    threshold_pct: float = 30.0
    compactness: float = 0.0
    smooth_sigma: float = 0.5
    seed_source: str = "auto"
    seed_distance: int = 5
    min_size: int = 0
    min_circularity: float = 0.0
    z_slice: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_01(arr: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if lo is None:
        lo = float(np.min(arr))
    if hi is None:
        hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, np.nextafter(np.float32(1.0), np.float32(0.0)))
    return scaled.astype(np.float32)


def _flow_magnitude(dp: np.ndarray) -> np.ndarray:
    """Compute L2 magnitude from a DP stack."""
    dp = np.asarray(dp, dtype=np.float32)
    if dp.ndim == 2:
        return np.abs(dp)
    if dp.ndim == 3:
        if dp.shape[0] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=0)).astype(np.float32)
        if dp.shape[-1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)
        return np.abs(dp).astype(np.float32)
    if dp.ndim >= 4:
        if dp.shape[1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=1)).astype(np.float32)
        if dp.shape[-1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)
    raise ValueError(f"Unsupported DP shape for magnitude: {dp.shape}")


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


def _centroid_markers_2d(labels: np.ndarray) -> np.ndarray:
    """Place one marker pixel at the centroid of each 2D label."""
    labels = np.asarray(labels)
    out = np.zeros_like(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        coords = np.argwhere(labels == label_id)
        centroid = coords.mean(axis=0)
        seed_yx = np.rint(centroid).astype(np.int64)
        if (
            seed_yx[0] < 0
            or seed_yx[0] >= labels.shape[0]
            or seed_yx[1] < 0
            or seed_yx[1] >= labels.shape[1]
            or labels[seed_yx[0], seed_yx[1]] != label_id
        ):
            distances = np.sum((coords - centroid) ** 2, axis=1)
            seed_yx = coords[int(np.argmin(distances))]
        out[int(seed_yx[0]), int(seed_yx[1])] = label_id
    return out


def centroid_markers_from_labels(labels: np.ndarray) -> np.ndarray:
    """Return one centroid seed pixel per non-zero label.

    For a 2D label image, each label is replaced by a single marker pixel at
    its rounded centroid. If the rounded centroid falls outside the label, the
    closest pixel belonging to that label is used instead. For a 3D stack, the
    operation is applied independently to each first-axis plane, matching
    time-first ``(T, Y, X)`` tracked nuclear labels.
    """
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return _centroid_markers_2d(labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected 2D labels or time-first 3D stack, got {labels.shape}")
    out = np.zeros_like(labels)
    for t in range(labels.shape[0]):
        out[t] = _centroid_markers_2d(labels[t])
    return out


def _peak_local_max_markers(basin: np.ndarray, min_distance: int) -> np.ndarray:
    from scipy.ndimage import label as nd_label
    from skimage.feature import peak_local_max

    coords = peak_local_max(basin, min_distance=max(1, min_distance), exclude_border=False)
    mask = np.zeros(basin.shape, dtype=bool)
    if coords.size:
        mask[coords[:, 0], coords[:, 1]] = True
    markers, _ = nd_label(mask)
    return markers.astype(np.int32)


def compute_hypothesis_labels(
    prob: np.ndarray,
    dp: np.ndarray | None,
    markers: np.ndarray | None,
    params: NucleusHypothesisParams,
    *,
    global_lo: float | None = None,
    global_hi: float | None = None,
) -> np.ndarray:
    """Compute a single nucleus hypothesis label image for one 2D slice.

    global_lo/global_hi: min/max of the basin computed over the full 3D volume,
    so threshold_pct is a fraction of the whole-frame dynamic range, not per-slice.
    """
    from skimage.segmentation import watershed

    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"Expected 2D probability slice, got shape {prob.shape}")

    if params.basin == "prob":
        basin = 1.0 / (1.0 + np.exp(-prob))  # logits → probabilities
    elif params.basin == "flow_mag":
        if dp is None:
            raise ValueError("flow_mag basin requested but no DP array provided")
        basin = _flow_magnitude(dp)
        if basin.ndim != 2:
            raise ValueError(f"Expected 2D flow magnitude slice, got shape {basin.shape}")
    else:
        raise ValueError(f"Unknown basin={params.basin!r}; expected 'prob' or 'flow_mag'")

    basin = _normalize_01(basin, lo=global_lo, hi=global_hi)
    if params.smooth_sigma > 0:
        basin = gaussian_filter(basin, sigma=float(params.smooth_sigma))

    if markers is None:
        markers = _peak_local_max_markers(basin, params.seed_distance)
    else:
        markers = np.asarray(markers, dtype=np.int32)
        if markers.shape != basin.shape:
            raise ValueError(
                f"Markers shape {markers.shape} does not match basin shape {basin.shape}"
            )

    from scipy.ndimage import binary_fill_holes

    threshold = float(params.threshold_pct) / 100.0
    mask = binary_fill_holes((basin >= threshold) | (markers > 0))

    labels = watershed(
        -basin,
        markers=markers,
        mask=mask,
        compactness=float(params.compactness),
        watershed_line=False,
    )
    result = _fill_and_close_labels(np.asarray(labels, dtype=_LABEL_DTYPE))
    result = _remove_small_labels(result, params.min_size)
    return _remove_low_circularity_labels(result, params.min_circularity)


def compute_cellpose_flow_hypothesis(
    prob_3d: np.ndarray,
    dp_3d: np.ndarray,
    params: CellposeFlowHypothesisParams,
) -> np.ndarray:
    """Run Cellpose native mask generation independently per z-slice.

    prob_3d: (Z, Y, X) logits from Cellpose (flows[2])
    dp_3d:   (Z, 2, Y, X) flow fields from Cellpose (flows[1])
    Returns: (Z, Y, X) uint32
    """
    try:
        import torch
        from cellpose.dynamics import compute_masks
    except ImportError as exc:
        raise ImportError(
            "cellpose and torch must be installed to use flow-based hypothesis generation"
        ) from exc

    prob_3d = np.asarray(prob_3d, dtype=np.float32)
    dp_3d = np.asarray(dp_3d, dtype=np.float32)
    if prob_3d.ndim != 3:
        raise ValueError(f"Expected (Z, Y, X) prob, got {prob_3d.shape}")
    if dp_3d.ndim != 4 or dp_3d.shape[1] != 2:
        raise ValueError(f"Expected (Z, 2, Y, X) dp, got {dp_3d.shape}")

    n_foreground = int(np.sum(prob_3d > params.cellprob_threshold))
    if n_foreground == 0:
        raise RuntimeError(
            f"No foreground pixels found: all prob values <= cellprob_threshold={params.cellprob_threshold}. "
            f"Prob range: [{float(prob_3d.min()):.2f}, {float(prob_3d.max()):.2f}]. "
            "Try lowering cellprob_threshold."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_z = prob_3d.shape[0]
    out = np.zeros_like(prob_3d, dtype=_LABEL_DTYPE)
    cp_min_size = params.min_size if params.min_size > 0 else -1
    for z in range(n_z):
        result = compute_masks(
            dp_3d[z],
            prob_3d[z],
            cellprob_threshold=params.cellprob_threshold,
            flow_threshold=params.flow_threshold,
            min_size=cp_min_size,
            niter=params.niter,
            do_3D=False,
            device=device,
        )
        # Cellpose ≥3.x returns just the mask array; older versions return (mask, p, tr).
        masks = result[0] if isinstance(result, tuple) else result
        out[z] = np.asarray(masks, dtype=_LABEL_DTYPE)
    return out


def build_consensus_boundary(
    prob_3d: np.ndarray,
    dp_3d: np.ndarray,
    cellprob_thresholds: list[float],
    gamma: float = 1.0,
    *,
    mask_callback: Callable[[np.ndarray, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average find_boundaries over (threshold × z-slice) to build a consensus boundary map.

    prob_3d: (Z, Y, X) logits  dp_3d: (Z, 2, Y, X)
    mask_callback: optional sink called as mask_callback(masks_zyx, thresh_idx) after each threshold.
    Returns: (boundary, foreground) both (Y, X) float32.
      boundary   — mean boundary density in [0, 1]
      foreground — sigmoid of z-averaged gamma-corrected prob logits
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
    n_total = 0

    for i_thresh, thresh in enumerate(cellprob_thresholds):
        z_masks: list[np.ndarray] = []
        for z in range(n_z):
            result = compute_masks(
                dp_3d[z], prob_3d[z],
                cellprob_threshold=float(thresh),
                flow_threshold=0.0,
                niter=200,
                do_3D=False,
                device=device,
            )
            masks = result[0] if isinstance(result, tuple) else result
            accum += find_boundaries(np.asarray(masks), mode="inner").astype(np.float32)
            n_total += 1
            if mask_callback is not None:
                z_masks.append(np.asarray(masks, dtype=np.uint32))
        if mask_callback is not None:
            mask_callback(np.stack(z_masks), i_thresh)

    boundary = accum / n_total if n_total > 0 else accum
    foreground = 1.0 / (1.0 + np.exp(-prob_3d.mean(axis=0).astype(np.float32)))
    return boundary, foreground


def compute_masks_for_threshold(
    dp_3d: np.ndarray, prob_3d: np.ndarray, threshold: float
) -> np.ndarray:
    """Run Cellpose mask generation for a specific threshold across all z-slices."""
    try:
        import torch
        from cellpose.dynamics import compute_masks
    except ImportError as exc:
        raise ImportError("cellpose and torch required") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_z = prob_3d.shape[0]
    out = np.zeros(prob_3d.shape, dtype=_LABEL_DTYPE)
    for z in range(n_z):
        result = compute_masks(
            dp_3d[z],
            prob_3d[z],
            cellprob_threshold=float(threshold),
            flow_threshold=0.0,
            niter=200,
            do_3D=False,
            device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        out[z] = np.asarray(masks, dtype=_LABEL_DTYPE)
    return out

def compute_cellpose_foreground_mask(
    prob: np.ndarray,
    dp: np.ndarray,
    threshold: float = 0.0,
    gamma: float = 1.0,
    niter: int = 200,
) -> np.ndarray:
    """Build a binary foreground mask by z-averaging then running Cellpose.

    For each time frame, the Z-axis of the probability and flow maps is
    averaged, gamma correction is applied to the prob logits, and Cellpose's
    ``compute_masks`` is called once on the resulting 2D frame.  All returned
    cell labels are merged into a single binary foreground.

    Parameters
    ----------
    prob: (T, Z, Y, X) float32
        Cellpose probability logits (``flows[2]``).
    dp: (T, Z, 2, Y, X) float32
        Cellpose flow fields (``flows[1]``).
    threshold: float
        Passed as ``cellprob_threshold`` to ``compute_masks``.
    gamma: float
        Gamma correction applied to the z-averaged prob logits before Cellpose.
    niter: int
        Passed as ``niter`` to ``compute_masks``.

    Returns
    -------
    (T, Y, X) uint8
        Binary foreground mask (0 = background, 1 = foreground).
    """
    try:
        import torch
        from cellpose.dynamics import compute_masks
    except ImportError as exc:
        raise ImportError(
            "cellpose and torch must be installed to use cellpose-based foreground masking"
        ) from exc

    prob = np.asarray(prob, dtype=np.float32)
    dp = np.asarray(dp, dtype=np.float32)

    if prob.ndim != 4:
        raise ValueError(f"Expected (T, Z, Y, X) prob, got {prob.shape}")
    if dp.ndim != 5 or dp.shape[2] != 2:
        raise ValueError(f"Expected (T, Z, 2, Y, X) dp, got {dp.shape}")
    if prob.shape[0] != dp.shape[0]:
        raise ValueError(
            f"Time dim mismatch: prob {prob.shape[0]} vs dp {dp.shape[0]}"
        )

    n_t = prob.shape[0]
    out_h, out_w = prob.shape[2], prob.shape[3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = np.zeros((n_t, out_h, out_w), dtype=np.uint8)

    for t in range(n_t):
        # Z-average
        prob_zavg = prob[t].mean(axis=0).astype(np.float32)       # (Y, X)
        dp_zavg = dp[t].mean(axis=0).astype(np.float32)            # (2, Y, X)

        # Gamma-correct prob logits
        prob_gamma = apply_gamma(prob_zavg, gamma)

        result = compute_masks(
            dp_zavg,
            prob_gamma,
            cellprob_threshold=float(threshold),
            flow_threshold=0.0,
            min_size=-1,
            niter=int(niter),
            do_3D=False,
            device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        masks = np.asarray(masks)
        out[t] = (masks > 0).astype(np.uint8)

    return out


@dataclass(frozen=True, slots=True)
class SeededWatershedParams:
    """Parameters for nucleus-seeded watershed cell hypothesis generation."""

    basin: str = "prob"
    foreground_threshold: float = 0.5
    compactness: float = 0.0

    def __post_init__(self) -> None:
        warnings.warn(
            "SeededWatershedParams is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    def to_dict(self) -> dict[str, object]:
        return {"method": "seeded_watershed", **asdict(self)}


def compute_seeded_watershed(
    prob_2d: np.ndarray,
    dp_2d: np.ndarray | None,
    seeds_2d: np.ndarray,
    params: SeededWatershedParams,
) -> np.ndarray:
    """Seeded watershed using nucleus labels as markers for one 2D z-slice.

    Foreground mask is always derived from sigmoid(prob_2d). Seeds whose
    centroid falls outside the mask are silently dropped by the watershed.
    """
    warnings.warn(
        "compute_seeded_watershed is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    from scipy.ndimage import binary_fill_holes
    from skimage.segmentation import watershed

    prob_2d = np.asarray(prob_2d, dtype=np.float32)
    seeds_2d = np.asarray(seeds_2d, dtype=np.int32)

    sigmoid_prob = 1.0 / (1.0 + np.exp(-prob_2d))
    fg_mask = binary_fill_holes(sigmoid_prob > params.foreground_threshold)

    if params.basin == "prob":
        basin = sigmoid_prob
    elif params.basin == "flow_mag":
        if dp_2d is None:
            raise ValueError("flow_mag basin requires a dp array")
        basin = _flow_magnitude(dp_2d)
        if basin.ndim != 2:
            raise ValueError(f"Expected 2D flow magnitude slice, got shape {basin.shape}")
    else:
        raise ValueError(f"Unknown basin={params.basin!r}; expected 'prob' or 'flow_mag'")

    labels = watershed(
        -basin,
        markers=seeds_2d,
        mask=fg_mask,
        compactness=float(params.compactness),
        watershed_line=False,
    )
    return np.asarray(labels, dtype=_LABEL_DTYPE)


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