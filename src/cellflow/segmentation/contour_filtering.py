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
    memory_tau: float = 0.0
    memory_floor: float = 0.01


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


def _ema_pass(contours: np.ndarray, tau: float, floor: float) -> np.ndarray:
    """Single-direction EMA with signal-adaptive alpha.

    contour_out[t] = α · contour[t] + (1 - α) · contour_out[t-1]

    where α = clamp(contour[t] / (contour[t] + τ), floor, 1.0)

    α ≈ 1 where contour is strong  → trust the data
    α ≈ 0 where contour is weak    → trust the memory

    ``floor`` ensures slow decay even with zero signal, preventing
    permanent ghosts from a single strong frame.
    """
    T = contours.shape[0]
    out = np.empty_like(contours)
    out[0] = contours[0]

    for t in range(1, T):
        alpha = contours[t] / (contours[t] + tau)
        alpha = np.clip(alpha, floor, 1.0)
        out[t] = alpha * contours[t] + (1.0 - alpha) * out[t - 1]

    return out


def contour_memory_filter(
    contours: np.ndarray,
    tau: float = 0.1,
    floor: float = 0.01,
) -> np.ndarray:
    """Bidirectional contour memory filter.

    Runs a signal-adaptive EMA forward (t=0→T) and backward (t=T→0),
    then averages.  Strong ridges reset the memory; weak/absent ridges
    inherit from their temporal neighbours.

    Parameters
    ----------
    contours : (T, H, W) float32
    tau : Signal threshold controlling the crossover between "trust data"
        and "trust memory".  Set to roughly the contour value you consider
        "weak".  Lower = more aggressive persistence.
    floor : Minimum alpha per frame.  Prevents permanent ghosting — even
        with zero signal the memory decays at rate ``(1 - floor)`` per
        frame.  At 0.01 a ghost halves in ~69 frames; at 0.05 ~14 frames.

    Returns
    -------
    filtered : (T, H, W) float32
    """
    contours = np.asarray(contours, dtype=np.float32)
    if contours.shape[0] < 2:
        return contours.copy()
    forward = _ema_pass(contours, tau, floor)
    backward = _ema_pass(contours[::-1], tau, floor)[::-1]
    return (forward + backward) / 2.0


def compute_filtered_contour_maps(
    contours: np.ndarray,
    params: ContourFilterParams,
) -> np.ndarray:
    """Return contour maps after median, Gaussian, and memory filtering."""
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
    filtered = np.asarray(filtered, dtype=np.float32)
    if params.memory_tau > 0.0 and filtered.shape[0] > 1:
        filtered = contour_memory_filter(
            filtered, tau=params.memory_tau, floor=params.memory_floor,
        )
    return _restore_contour_stack(filtered, layout)