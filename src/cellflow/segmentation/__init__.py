"""Nucleus segmentation via watershed on Cellpose probability maps."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

_LABEL_DTYPE = np.uint32


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

    threshold = float(params.threshold_pct) / 100.0
    mask = (basin >= threshold) | (markers > 0)

    labels = watershed(
        -basin,
        markers=markers,
        mask=mask,
        compactness=float(params.compactness),
        watershed_line=False,
    )
    return _remove_small_labels(np.asarray(labels, dtype=_LABEL_DTYPE), params.min_size)
