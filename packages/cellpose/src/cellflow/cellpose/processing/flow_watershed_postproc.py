"""Post-processing for flow-watershed cell segmentation.

Individual operations (open, close, fill_holes, smooth_boundary,
mask_to_tissue_foreground) are exposed as standalone functions.
``run_postprocess_pipeline`` executes them in user-specified order,
allowing arbitrary combinations and repetitions.

Legacy API (``postprocess_flow_watershed``, ``morphological_smoothing``,
``boundary_smoothing``) is preserved for backward compatibility.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours
from scipy.interpolate import UnivariateSpline


# ── Default pipeline ──────────────────────────────────────────────────────────

DEFAULT_POSTPROCESS_STEPS: list[dict] = [
    {"type": "open",            "radius":     1   },
    {"type": "close",           "radius":     1   },
    {"type": "smooth_boundary", "smoothness": 0.5 },
]


# ── Contour helpers ───────────────────────────────────────────────────────────

def _resample_contour(contour: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Resample a contour to a fixed number of points."""
    if len(contour) < 2:
        return contour
    diffs = np.diff(contour, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    cumsum = np.concatenate([[0], np.cumsum(dists)])
    new_s = np.linspace(0, cumsum[-1], num_points)
    return np.column_stack([
        np.interp(new_s, cumsum, contour[:, 0]),
        np.interp(new_s, cumsum, contour[:, 1]),
    ])


def _smooth_contour_spline(contour: np.ndarray, smoothness: float = 0.5) -> np.ndarray:
    """Smooth a contour using B-spline interpolation."""
    if len(contour) < 4:
        return contour
    contour_closed = np.vstack([contour, contour[0:1]])
    t = np.arange(len(contour_closed))
    try:
        s_val = smoothness * len(contour) ** 2
        spl_row = UnivariateSpline(t, contour_closed[:, 0], s=s_val, k=min(3, len(contour_closed) - 1))
        spl_col = UnivariateSpline(t, contour_closed[:, 1], s=s_val, k=min(3, len(contour_closed) - 1))
        t_smooth = np.linspace(0, len(contour) - 1, len(contour) * 2)
        return np.column_stack([spl_row(t_smooth), spl_col(t_smooth)])
    except Exception:
        return contour


# ── Individual operations ─────────────────────────────────────────────────────

def open_labels(labels: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphological opening per cell — removes small noise / thin protrusions.

    Parameters
    ----------
    labels : np.ndarray
        Label image (H, W).
    radius : int
        Number of erosion + dilation iterations (0 = no-op).
    """
    if radius <= 0:
        return labels
    result = np.zeros_like(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        mask = ndimage.binary_opening(labels == label_id, iterations=radius)
        result[mask] = label_id
    return result


def close_labels(labels: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphological closing per cell — fills small gaps near borders.

    Parameters
    ----------
    labels : np.ndarray
        Label image (H, W).
    radius : int
        Number of dilation + erosion iterations (0 = no-op).
    """
    if radius <= 0:
        return labels
    result = np.zeros_like(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        mask = ndimage.binary_closing(labels == label_id, iterations=radius)
        result[mask] = label_id
    return result


def fill_label_holes(labels: np.ndarray, radius: int = 5) -> np.ndarray:
    """Fill enclosed background gaps between cells up to *radius* pixels wide.

    Background regions that touch the image border ("open" background) are left
    intact — only fully enclosed gaps are filled.  Each enclosed gap pixel is
    assigned to whichever neighbouring cell is closest (nearest-neighbour
    expansion), matching the behaviour of the interactive Fix Borders tool.

    Parameters
    ----------
    labels : np.ndarray
        Label image (H, W).
    radius : int
        Maximum distance (px) cells are allowed to expand into enclosed gaps.
        Use a large value (e.g. 999) to fill all enclosed gaps regardless of
        size.
    """
    from skimage.measure import label as cc_label
    from skimage.segmentation import expand_labels

    if radius <= 0:
        return labels

    bg = labels == 0
    if not np.any(bg):
        return labels

    # Find connected background components
    bg_labeled = cc_label(bg, connectivity=2)

    # Mark any component that touches the image border as "open"
    open_ids: set = set()
    for edge in (bg_labeled[0, :], bg_labeled[-1, :],
                 bg_labeled[:, 0], bg_labeled[:, -1]):
        open_ids.update(np.unique(edge))
    open_ids.discard(0)

    open_bg  = bg & np.isin(bg_labeled, list(open_ids))
    enclosed = bg & ~open_bg

    if not np.any(enclosed):
        return labels

    # Place a sentinel label on open-background pixels so expand_labels treats
    # them as occupied and won't overwrite them.
    SENTINEL = int(labels.max()) + 1
    work = labels.copy()
    work[open_bg] = SENTINEL

    expanded = expand_labels(work, distance=radius)

    # Remove sentinel and any expansion into open background
    expanded[open_bg]              = 0
    expanded[expanded == SENTINEL] = 0

    return expanded.astype(labels.dtype)


def smooth_label_boundaries(labels: np.ndarray, smoothness: float = 0.5) -> np.ndarray:
    """Smooth cell boundaries using spline-based contour fitting.

    Parameters
    ----------
    labels : np.ndarray
        Label image (H, W).
    smoothness : float
        User-facing smoothing factor 0–1.  Scaled by 0.01 before being passed
        to the spline fitter so that the effective range (0–0.01) stays subtle.
    """
    if smoothness <= 0:
        return labels
    return boundary_smoothing(labels, smoothness=smoothness * 0.01)


def compute_tissue_foreground_mask(
    tissue_image: np.ndarray,
    sigma: float = 2.0,
    threshold: float = 0.1,
) -> np.ndarray:
    """Compute a binary foreground mask from a tissue intensity image.

    Gaussian-smooths *tissue_image*, normalises it to [0, 1], and thresholds
    at *threshold*.  Returns a boolean array with ``True`` for foreground pixels.

    Parameters
    ----------
    tissue_image : np.ndarray
        Raw intensity image (H, W) — typically the z-projected membrane channel.
    sigma : float
        Gaussian smoothing radius in pixels (default 2.0; 0 = no smoothing).
    threshold : float
        Foreground cutoff in the [0, 1] range after normalisation to the image
        maximum (default 0.1).
    """
    img = tissue_image.astype(np.float32)
    if sigma > 0:
        img = ndimage.gaussian_filter(img, sigma=sigma)
    img_max = img.max()
    if img_max > 0:
        img = img / img_max
    return img > threshold


def mask_to_tissue_foreground(
    labels: np.ndarray,
    tissue_image: np.ndarray,
    sigma: float = 2.0,
    threshold: float = 0.1,
) -> np.ndarray:
    """Zero out labels outside the tissue foreground.

    Delegates mask computation to :func:`compute_tissue_foreground_mask`.

    Parameters
    ----------
    labels : np.ndarray
        Label image (H, W).
    tissue_image : np.ndarray
        Raw intensity image (H, W) — typically the z-projected membrane channel.
    sigma : float
        Gaussian smoothing radius in pixels before thresholding (default 2.0).
    threshold : float
        Foreground cutoff in [0, 1] after normalisation (default 0.1).
    """
    foreground = compute_tissue_foreground_mask(tissue_image, sigma=sigma, threshold=threshold)
    result = labels.copy()
    result[~foreground] = 0
    return result


# ── Binary mask operations ────────────────────────────────────────────────────

def open_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphological opening on a binary mask — removes small islands / protrusions.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or uint8 binary mask (H, W).
    radius : int
        Number of erosion + dilation iterations (0 = no-op).
    """
    if radius <= 0:
        return mask.astype(bool)
    return ndimage.binary_opening(mask, iterations=radius)


def close_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphological closing on a binary mask — fills small holes and gaps.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or uint8 binary mask (H, W).
    radius : int
        Number of dilation + erosion iterations (0 = no-op).
    """
    if radius <= 0:
        return mask.astype(bool)
    return ndimage.binary_closing(mask, iterations=radius)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill all enclosed background holes in a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or uint8 binary mask (H, W).
    """
    return ndimage.binary_fill_holes(mask)


def smooth_mask_boundary(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Smooth binary mask boundary via Gaussian blur + rethreshold at 0.5.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or uint8 binary mask (H, W).
    sigma : float
        Gaussian blur radius in pixels (0 = no-op).
    """
    if sigma <= 0:
        return mask.astype(bool)
    blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return blurred > 0.5


def run_mask_postprocess_pipeline(mask: np.ndarray, steps: list[dict]) -> np.ndarray:
    """Execute binary mask postprocessing steps in the given order.

    Each step is a dict with a ``"type"`` key and optional parameters:

    ==================  ================  ====================================
    type                extra keys        description
    ==================  ================  ====================================
    ``open``            ``radius`` (int)  binary opening (remove islands)
    ``close``           ``radius`` (int)  binary closing (fill gaps)
    ``fill_holes``      —                 fill all enclosed background holes
    ``smooth_boundary`` ``sigma`` (float) Gaussian blur + rethreshold at 0.5
    ==================  ================  ====================================

    Parameters
    ----------
    mask : np.ndarray
        Binary input mask (H, W) — bool or uint8.
    steps : list[dict]
        Ordered list of step dicts.

    Returns
    -------
    np.ndarray
        Processed mask (H, W), dtype uint8 (0/1).
    """
    result = mask.astype(bool)
    for step in steps:
        t = step.get("type")
        if t == "open":
            result = open_mask(result, step.get("radius", 1))
        elif t == "close":
            result = close_mask(result, step.get("radius", 1))
        elif t == "fill_holes":
            result = fill_mask_holes(result)
        elif t == "smooth_boundary":
            result = smooth_mask_boundary(result, step.get("sigma", 2.0))
        # unknown types silently skipped
    return result.astype(np.uint8)


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_postprocess_pipeline(
    labels: np.ndarray,
    steps: list[dict],
    tissue_image: np.ndarray | None = None,
    foreground_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Execute postprocessing steps in the given order.

    Each step is a dict with a ``"type"`` key and optional parameters:

    ==================  =================  ===================================
    type                extra keys         description
    ==================  =================  ===================================
    ``open``            ``radius`` (int)   morphological opening per cell
    ``close``           ``radius`` (int)   morphological closing per cell
    ``fill_holes``      ``radius`` (int)   expand cells into enclosed bg gaps
    ``smooth_boundary`` ``smoothness``     spline-based boundary smoothing
    ``tissue_mask``     ``sigma`` (float)  legacy — mask labels to tissue fg
    ==================  =================  ===================================

    ``tissue_mask`` steps are silently skipped when *tissue_image* is None.
    Unknown step types (including the old ``trim_low_prob``) are silently
    skipped.

    After all steps, *foreground_mask* (if provided) is applied as a final
    binary spatial mask — pixels where the mask is False/0 are zeroed out.
    This is the preferred way to apply tissue masking; ``tissue_mask`` steps
    are kept only for backward compatibility with old configs.

    Parameters
    ----------
    labels : np.ndarray
        Label image (H, W).
    steps : list[dict]
        Ordered list of step dicts.
    tissue_image : np.ndarray, optional
        Raw intensity frame (H, W) used by legacy ``tissue_mask`` steps.
    foreground_mask : np.ndarray, optional
        Binary mask (H, W) — True/1 for foreground, False/0 for background.
        Applied after all pipeline steps.

    Returns
    -------
    np.ndarray
        Processed labels (H, W), dtype int32.
    """
    result = labels.copy()
    for step in steps:
        t = step.get("type")
        if t == "open":
            result = open_labels(result, step.get("radius", 1))
        elif t == "close":
            result = close_labels(result, step.get("radius", 1))
        elif t == "fill_holes":
            result = fill_label_holes(result, step.get("radius", 5))
        elif t == "smooth_boundary":
            result = smooth_label_boundaries(result, step.get("smoothness", 0.5))
        elif t == "tissue_mask":
            # Legacy step — kept for backward compatibility with old configs.
            if tissue_image is not None:
                result = mask_to_tissue_foreground(
                    result, tissue_image,
                    sigma=step.get("sigma", 2.0),
                    threshold=step.get("threshold", 0.1),
                )
        # unknown step types are silently skipped
    if foreground_mask is not None:
        result[~foreground_mask.astype(bool)] = 0
    return result.astype(np.int32)


# ── Legacy public API (backward compatibility) ────────────────────────────────

def morphological_smoothing(
    labels: np.ndarray,
    opening_radius: int = 1,
    closing_radius: int = 1,
) -> np.ndarray:
    """Apply opening then closing to each cell label.

    .. deprecated::
        Use :func:`open_labels` / :func:`close_labels` directly.
    """
    result = np.zeros_like(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        mask = labels == label_id
        if opening_radius > 0:
            mask = ndimage.binary_opening(mask, iterations=opening_radius)
        if closing_radius > 0:
            mask = ndimage.binary_closing(mask, iterations=closing_radius)
        result[mask] = label_id
    return result


def boundary_smoothing(
    labels: np.ndarray,
    smoothness: float = 0.5,
) -> np.ndarray:
    """Smooth cell boundaries using contour smoothing."""
    H, W = labels.shape
    result = np.zeros_like(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        mask = (labels == label_id).astype(np.uint8)
        contours = find_contours(mask, level=0.5)
        if not contours:
            result[mask.astype(bool)] = label_id
            continue
        contour = contours[0]
        if len(contour) > 3:
            smoothed_contour = _smooth_contour_spline(contour, smoothness=smoothness)
            from skimage.draw import polygon
            try:
                rows = np.clip(smoothed_contour[:, 0].astype(int), 0, H - 1)
                cols = np.clip(smoothed_contour[:, 1].astype(int), 0, W - 1)
                rr, cc = polygon(rows, cols, shape=(H, W))
                result[rr, cc] = label_id
            except Exception:
                result[mask.astype(bool)] = label_id
        else:
            result[mask.astype(bool)] = label_id
    return result


def postprocess_flow_watershed(
    labels: np.ndarray,
    opening_radius: int = 1,
    closing_radius: int = 1,
    boundary_smoothness: float = 0.5,
) -> np.ndarray:
    """Complete post-processing pipeline — delegates to ``run_postprocess_pipeline``.

    .. deprecated::
        Build a ``steps`` list and call :func:`run_postprocess_pipeline` directly.
    """
    steps: list[dict] = []
    if opening_radius > 0:
        steps.append({"type": "open",            "radius":     opening_radius})
    if closing_radius > 0:
        steps.append({"type": "close",           "radius":     closing_radius})
    if boundary_smoothness > 0:
        steps.append({"type": "smooth_boundary", "smoothness": boundary_smoothness})
    return run_postprocess_pipeline(labels, steps)
