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

from cellflow.core.cancellation import CancelledError
from cellflow.core.tiff import imwrite_grayscale


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise CancelledError("Operation cancelled.")


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
    foreground_smoothing_sigma: float = 0.0
    foreground_median_radius: int = 0


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


def _filter_2d(
    arr_yx: np.ndarray, *, smoothing_sigma: float, median_radius: int,
) -> np.ndarray:
    """Apply median -> gaussian to a single ``(Y, X)`` map. Order matters (spec)."""
    out = np.asarray(arr_yx, dtype=np.float32)
    if median_radius > 0:
        out = median_filter(out, size=2 * int(median_radius) + 1)
    if smoothing_sigma > 0.0:
        out = gaussian_filter(out, sigma=float(smoothing_sigma))
    return out.astype(np.float32, copy=False)


def foreground_from_prob(
    prob_tzyx: np.ndarray,
    *,
    reduction: ZReduction,
    smoothing_sigma: float = 0.0,
    median_radius: int = 0,
) -> np.ndarray:
    """``sigmoid(prob)`` reduced across z, optionally smoothed.

    Returns ``(T, Y, X)`` float32 in [0, 1]. When ``smoothing_sigma`` or
    ``median_radius`` is set, each reduced frame is median- then gaussian-
    filtered (defaults of ``0`` leave the foreground untouched).
    """
    p = sigmoid(prob_tzyx)
    fg = _reduce_z(p, reduction)
    if smoothing_sigma > 0.0 or median_radius > 0:
        fg = np.stack(
            [
                _filter_2d(
                    fg[t], smoothing_sigma=smoothing_sigma, median_radius=median_radius,
                )
                for t in range(fg.shape[0])
            ]
        )
    return fg


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
        pos[t] = _positive_divergence_z_stack(
            arr[t],
            smoothing_sigma=smoothing_sigma,
            median_radius=median_radius,
        )
    return _reduce_z(pos, reduction)


def _positive_divergence_z_stack(
    dp_zcyx: np.ndarray,
    *,
    smoothing_sigma: float,
    median_radius: int,
    progress_cb: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    n_z, _, n_y, n_x = dp_zcyx.shape
    pos = np.empty((n_z, n_y, n_x), dtype=np.float32)
    for z in range(n_z):
        filt = _filter_flow(
            dp_zcyx[z], smoothing_sigma=smoothing_sigma, median_radius=median_radius,
        )
        div = divergence_2d(filt)
        np.clip(div, 0.0, None, out=div)
        pos[z] = div
        if progress_cb is not None:
            progress_cb(z + 1, n_z)
    return pos


class _LazyTiffStack:
    """Frame-at-a-time reader for a (T,Z,Y,X) / (T,Z,2,Y,X) TIFF stack.

    Reads only the TIFF header on construction (so callers can validate shapes
    without paying for a full decompress) and decodes one frame's pages per
    :meth:`frame` call. Compressed stacks store one page per 2D plane, so a
    per-frame ``key`` slice touches only that frame's pages — this is what lets
    :func:`build_divergence_maps` start reporting progress immediately instead
    of stalling while both whole stacks are read up front.
    """

    def __init__(self, path: str | Path, *, ndim: int, name: str) -> None:
        self._path = Path(path)
        self._name = name
        with tifffile.TiffFile(str(self._path)) as tf:
            series = tf.series[0]
            disk_shape = tuple(int(x) for x in series.shape)
            axes = str(series.axes)
        shape = self._canonical_shape(disk_shape, axes, ndim, name)
        self.shape = shape
        self._frame_shape = shape[1:]
        # Pages per frame = product of the non-(Y,X) per-frame axes (Z, or Z*2).
        self._pages_per_frame = int(np.prod(shape[1:-2], dtype=int))

    @staticmethod
    def _canonical_shape(
        disk_shape: tuple[int, ...], axes: str, ndim: int, name: str,
    ) -> tuple[int, ...]:
        """Recover the logical ``(T,Z,Y,X)`` / ``(T,Z,2,Y,X)`` shape.

        TIFF drops singleton leading axes on write, so a 2D+t prob ``(T,1,Y,X)``
        lands on disk as ``(T,Y,X)`` — indistinguishable from a single z-stack by
        shape alone. When the writer recorded axis labels (our ``write_outputs``),
        map the surviving letters back to canonical order, inserting size-1 for
        any dropped ``T``/``Z``/``C``. Otherwise fall back to the length heuristic
        for legacy/metadata-less files (which are never singleton-squeezed).
        """
        canonical = "TZYX" if ndim == 4 else "TZCYX"
        label = "Z×Y×X or T×Z×Y×X" if ndim == 4 else "Z×2×Y×X or T×Z×2×Y×X"
        if len(axes) == len(disk_shape) and set("YX") <= set(axes) <= set(canonical):
            present = dict(zip(axes, disk_shape))
            shape = tuple(present.get(a, 1) for a in canonical)
        else:
            shape = disk_shape
            # A frameless 3D/4D stack is a single timepoint: prepend T=1.
            if len(shape) == ndim - 1:
                shape = (1,) + shape
        if len(shape) != ndim:
            raise ValueError(f"{name} must be {label}.")
        if ndim == 5 and shape[2] != 2:
            raise ValueError(f"{name} must be Z×2×Y×X or T×Z×2×Y×X.")
        return shape

    def frame(self, t: int) -> np.ndarray:
        """Frame ``t`` as float32, shaped ``(Z,Y,X)`` or ``(Z,2,Y,X)``."""
        start = t * self._pages_per_frame
        arr = tifffile.imread(
            str(self._path), key=slice(start, start + self._pages_per_frame),
        )
        return np.asarray(arr, dtype=np.float32).reshape(self._frame_shape)


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
    foreground_smoothing_sigma: float = 0.0,
    foreground_median_radius: int = 0,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> DivergenceMapsReport:
    """Compute and write ``contours`` and ``foreground`` from Cellpose prob/dp.

    Output stacks are ``T x Y x X`` float32. ``progress_cb`` is called for each
    foreground frame, contour z-slice, and output write. ``smoothing_sigma`` /
    ``median_radius`` filter the flow before divergence (contours); the
    ``foreground_*`` knobs filter the reduced foreground map independently.
    """
    prob_stack = _LazyTiffStack(prob_path, ndim=4, name="prob")
    dp_stack = _LazyTiffStack(dp_path, ndim=5, name="dp")
    if prob_stack.shape[0] != dp_stack.shape[0]:
        raise ValueError("prob and dp must have the same frame count.")
    if prob_stack.shape[1] != dp_stack.shape[1]:
        raise ValueError("prob and dp must have the same z count.")
    if prob_stack.shape[2:] != dp_stack.shape[3:]:
        raise ValueError("prob and dp must have the same Y×X shape.")

    n_t = int(prob_stack.shape[0])
    n_z = int(prob_stack.shape[1])
    if n_t == 0:
        raise ValueError("prob/dp stack has no frames (T=0); nothing to compute.")
    total_steps = n_t + (n_t * n_z) + 2
    progress_done = 0
    contour_frames: list[np.ndarray] = []
    foreground_frames: list[np.ndarray] = []
    for t in range(n_t):
        _check_cancel(cancel)
        fg = foreground_from_prob(
            prob_stack.frame(t)[np.newaxis],
            reduction=foreground_z_reduction,
            smoothing_sigma=foreground_smoothing_sigma,
            median_radius=foreground_median_radius,
        )
        foreground_frames.append(fg[0])
        progress_done += 1
        if progress_cb is not None:
            progress_cb(
                progress_done,
                total_steps,
                f"Divergence maps: foreground frame {t + 1}/{n_t}",
            )

        def _progress_z(z_done: int, z_total: int) -> None:
            nonlocal progress_done

            _check_cancel(cancel)
            progress_done += 1
            if progress_cb is not None:
                progress_cb(
                    progress_done,
                    total_steps,
                    (
                        f"Divergence maps: contours frame {t + 1}/{n_t} "
                        f"z {z_done}/{z_total}"
                    ),
                )

        pos_div = _positive_divergence_z_stack(
            dp_stack.frame(t),
            smoothing_sigma=smoothing_sigma,
            median_radius=median_radius,
            progress_cb=_progress_z,
        )
        contour_frames.append(
            _reduce_z(pos_div[np.newaxis, ...], contour_z_reduction)[0]
        )

    _check_cancel(cancel)
    contours_out = Path(contours_out)
    foreground_out = Path(foreground_out)
    contours_out.parent.mkdir(parents=True, exist_ok=True)
    foreground_out.parent.mkdir(parents=True, exist_ok=True)
    imwrite_grayscale(
        contours_out, np.stack(contour_frames).astype(np.float32),
        compression="zlib", metadata={"axes": "TYX"},
    )
    progress_done += 1
    if progress_cb is not None:
        progress_cb(progress_done, total_steps, "Divergence maps: writing contours")
    imwrite_grayscale(
        foreground_out, np.stack(foreground_frames).astype(np.float32),
        compression="zlib", metadata={"axes": "TYX"},
    )
    progress_done += 1
    if progress_cb is not None:
        progress_cb(progress_done, total_steps, "Divergence maps: writing foreground")
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
