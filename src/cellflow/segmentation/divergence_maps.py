"""Divergence-based foreground & contour maps from Cellpose prob/dp outputs.

Replaces the (cellprob x z) mask sweep with a direct computation:

    foreground = reduce_z(sigmoid(prob))
    contours   = reduce_z(clip(div(filter(dp)), 0, inf))

See ``notes/divergence_maps_spec.md`` for the rationale.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, median_filter

from cellflow.segmentation.nucleus_segmentation import (
    CancelledError,
    _check_cancel,
)


ZReduction = Literal["mean", "max"]


@dataclass(frozen=True, slots=True)
class DivergenceMapsReport:
    """Summary returned by :func:`build_divergence_maps`."""

    frames: int
    foreground_z_reduction: ZReduction
    contour_z_reduction: ZReduction
    smoothing_sigma: float
    median_radius: int
    contours_path: Path
    foreground_path: Path


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid on float32 logits."""
    x = np.clip(np.asarray(x, dtype=np.float32), -88.0, 88.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def _reduce_z(arr_tzyx: np.ndarray, reduction: ZReduction) -> np.ndarray:
    if reduction == "mean":
        return arr_tzyx.mean(axis=1, dtype=np.float32).astype(np.float32, copy=False)
    if reduction == "max":
        return arr_tzyx.max(axis=1).astype(np.float32, copy=False)
    raise ValueError(f"reduction must be 'mean' or 'max', got {reduction!r}")


def foreground_from_prob(
    prob_tzyx: np.ndarray, *, reduction: ZReduction
) -> np.ndarray:
    """``sigmoid(prob)`` reduced across z. Returns ``(T, Y, X)`` float32 in [0, 1]."""
    p = sigmoid(prob_tzyx)
    return _reduce_z(p, reduction)


def divergence_2d(flow_yx: np.ndarray) -> np.ndarray:
    """Divergence of a ``(2, Y, X)`` flow field with channels ``[dy, dx]``."""
    flow_yx = np.asarray(flow_yx, dtype=np.float32)
    if flow_yx.ndim != 3 or flow_yx.shape[0] != 2:
        raise ValueError(
            f"flow must be (2, Y, X) with channels [dy, dx]; got {flow_yx.shape}"
        )
    d_dy = np.gradient(flow_yx[0], axis=0)
    d_dx = np.gradient(flow_yx[1], axis=1)
    return (d_dy + d_dx).astype(np.float32, copy=False)


def _filter_flow(
    flow_2yx: np.ndarray, *, smoothing_sigma: float, median_radius: int,
) -> np.ndarray:
    """Apply median -> gaussian per channel. Order matters (spec)."""
    out = flow_2yx
    if median_radius > 0:
        size = 2 * int(median_radius) + 1
        out = np.stack(
            [median_filter(out[0], size=size), median_filter(out[1], size=size)],
            axis=0,
        )
    if smoothing_sigma > 0.0:
        sigma = float(smoothing_sigma)
        out = np.stack(
            [gaussian_filter(out[0], sigma=sigma), gaussian_filter(out[1], sigma=sigma)],
            axis=0,
        )
    return out


def contour_from_dp(
    dp_tzcyx: np.ndarray,
    *,
    smoothing_sigma: float,
    median_radius: int,
    reduction: ZReduction,
) -> np.ndarray:
    """Per (t, z): filter -> divergence -> clip(>=0); then reduce across z.

    ``dp_tzcyx``: shape ``(T, Z, 2, Y, X)`` with channels ``[dy, dx]``.
    Returns ``(T, Y, X)`` float32.
    """
    arr = np.asarray(dp_tzcyx, dtype=np.float32)
    if arr.ndim != 5 or arr.shape[2] != 2:
        raise ValueError(
            f"dp must be (T, Z, 2, Y, X) with channels [dy, dx]; got {arr.shape}"
        )
    n_t, n_z, _, n_y, n_x = arr.shape
    pos = np.empty((n_t, n_z, n_y, n_x), dtype=np.float32)
    for t in range(n_t):
        for z in range(n_z):
            filt = _filter_flow(
                arr[t, z], smoothing_sigma=smoothing_sigma, median_radius=median_radius,
            )
            div = divergence_2d(filt)
            np.clip(div, 0.0, None, out=div)
            pos[t, z] = div
    return _reduce_z(pos, reduction)


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


def build_divergence_maps(
    prob_path: str | Path,
    dp_path: str | Path,
    contours_out: str | Path,
    foreground_out: str | Path,
    *,
    foreground_z_reduction: ZReduction,
    contour_z_reduction: ZReduction,
    smoothing_sigma: float,
    median_radius: int,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> DivergenceMapsReport:
    """Compute and write ``contours`` and ``foreground`` from Cellpose prob/dp.

    Output stacks are ``T x Y x X`` float32. ``progress_cb`` is called per frame.
    """
    prob_stack = _as_tzyx(tifffile.imread(str(prob_path)), "prob")
    dp_stack = _as_tzcyx(tifffile.imread(str(dp_path)), "dp")
    if prob_stack.shape[0] != dp_stack.shape[0]:
        raise ValueError("prob and dp must have the same frame count.")
    if prob_stack.shape[1] != dp_stack.shape[1]:
        raise ValueError("prob and dp must have the same z count.")
    if prob_stack.shape[2:] != dp_stack.shape[3:]:
        raise ValueError("prob and dp must have the same Y×X shape.")

    n_t = int(prob_stack.shape[0])
    contour_frames: list[np.ndarray] = []
    foreground_frames: list[np.ndarray] = []
    for t in range(n_t):
        _check_cancel(cancel)
        if progress_cb is not None:
            progress_cb(t, n_t, f"Divergence maps: frame {t + 1}/{n_t}")
        fg = foreground_from_prob(
            prob_stack[t : t + 1], reduction=foreground_z_reduction,
        )
        contour = contour_from_dp(
            dp_stack[t : t + 1],
            smoothing_sigma=smoothing_sigma,
            median_radius=median_radius,
            reduction=contour_z_reduction,
        )
        foreground_frames.append(fg[0])
        contour_frames.append(contour[0])

    _check_cancel(cancel)
    contours_out = Path(contours_out)
    foreground_out = Path(foreground_out)
    contours_out.parent.mkdir(parents=True, exist_ok=True)
    foreground_out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(contours_out), np.stack(contour_frames).astype(np.float32),
        compression="zlib",
    )
    tifffile.imwrite(
        str(foreground_out), np.stack(foreground_frames).astype(np.float32),
        compression="zlib",
    )
    if progress_cb is not None:
        progress_cb(n_t, n_t, f"Divergence maps: wrote {n_t} frames")
    return DivergenceMapsReport(
        frames=n_t,
        foreground_z_reduction=foreground_z_reduction,
        contour_z_reduction=contour_z_reduction,
        smoothing_sigma=float(smoothing_sigma),
        median_radius=int(median_radius),
        contours_path=contours_out,
        foreground_path=foreground_out,
    )


__all__ = [
    "CancelledError",
    "DivergenceMapsReport",
    "build_divergence_maps",
    "contour_from_dp",
    "divergence_2d",
    "foreground_from_prob",
    "sigmoid",
]
