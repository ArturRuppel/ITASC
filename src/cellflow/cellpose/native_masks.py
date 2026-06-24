"""Capture Cellpose-SAM *native* masks, Qt-free, for the standalone distro.

The app's :mod:`cellflow.cellpose.cellpose_runner` keeps only the probability /
flow maps (``_, flows, _ = model.eval(...)``) because the integrated pipeline
derives its labels from divergence maps + Ultrack. The independently-shipped
``cellflow-cellpose`` tool instead wants the labelled masks Cellpose computes and
the runner throws away — that is index ``0`` of ``model.eval``.

This module re-runs the same eval loop but keeps the masks, producing a canonical
``(T, Z, Y, X)`` label stack per channel. It reuses the runner's model loading,
gamma correction and cancellation so the two paths stay consistent; it does not
touch the runner itself, so the app's behaviour is unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal
from collections.abc import Callable

import numpy as np

from cellflow.core.tiff import imwrite_grayscale
from cellflow.cellpose import cellpose_runner
from cellflow.cellpose.cellpose_runner import (
    CancelledError,
    CellParams,
    NucleusParams,
    _apply_gamma,
    _check_cancel,
    _diameter_kwarg,
    _NORMALIZE,
)

__all__ = [
    "run_nucleus_masks_frame",
    "run_cell_masks_frame",
    "run_nucleus_masks_stack",
    "run_cell_masks_stack",
    "write_masks",
    "offset_slice_labels",
]


def offset_slice_labels(slices: list[np.ndarray]) -> np.ndarray:
    """Stack per-z 2D label images into ``(Z, Y, X)`` with frame-unique labels.

    Each 2D Cellpose run labels its objects from ``1`` independently, so naively
    stacking would collide labels across z. We offset every slice's non-zero
    labels by the running maximum so a frame's labels are globally unique; the
    background ``0`` is preserved. For the common single-z case this is a no-op.
    """
    out: list[np.ndarray] = []
    running_max = 0
    for sl in slices:
        sl = np.asarray(sl).astype(np.int32, copy=True)
        nonzero = sl > 0
        if nonzero.any():
            sl[nonzero] += running_max
            running_max = int(sl.max())
        out.append(sl)
    return np.stack(out, axis=0)


def _eval_masks(img: np.ndarray, **eval_kwargs) -> np.ndarray:
    """Run ``model.eval`` and return its native masks (index 0) as ``int32``."""
    model = cellpose_runner.get_model()
    masks, _flows, _styles = model.eval(img, normalize=_NORMALIZE, **eval_kwargs)
    return np.asarray(masks, dtype=np.int32)


def run_nucleus_masks_frame(
    frame: np.ndarray,
    z: int | None,
    params: NucleusParams,
) -> np.ndarray:
    """Native nucleus masks for one frame.

    ``z is None`` runs full 3D over ``(Z, Y, X)`` and returns labels ``(Z, Y, X)``;
    an integer ``z`` runs 2D on ``frame[z]`` and returns labels ``(Y, X)``.
    """
    diameter = _diameter_kwarg(params.diameter)
    if z is None:
        volume = _apply_gamma(frame, params.gamma)
        return _eval_masks(
            volume,
            do_3D=True,
            z_axis=0,
            diameter=diameter,
            anisotropy=params.anisotropy,
            min_size=params.min_size,
        )
    slice_2d = _apply_gamma(frame[z], params.gamma)
    return _eval_masks(slice_2d, diameter=diameter, min_size=params.min_size)


def run_cell_masks_frame(
    frame: np.ndarray,
    z: int,
    params: CellParams,
) -> np.ndarray:
    """Native cell masks for a single 2D z-slice. Returns labels ``(Y, X)``."""
    diameter = _diameter_kwarg(params.diameter)
    slice_2d = _apply_gamma(frame[z], params.gamma)
    return _eval_masks(slice_2d, diameter=diameter, min_size=params.min_size)


def run_nucleus_masks_stack(
    stack: np.ndarray,
    params: NucleusParams,
    *,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Process a ``(T, Z, Y, X)`` stack into a ``(T, Z, Y, X)`` label stack.

    ``params.do_3d`` runs a coherent 3D segmentation per frame; otherwise each
    z-slice is segmented in 2D and the slices are stacked with frame-unique
    labels (see :func:`offset_slice_labels`).
    """
    if stack.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {stack.shape}")
    T, Z = stack.shape[:2]
    frames: list[np.ndarray] = []
    for t in range(T):
        _check_cancel(cancel_cb)
        if progress_cb is not None:
            progress_cb(t, T, f"Nucleus masks: frame {t + 1}/{T}...")
        if params.do_3d:
            masks = run_nucleus_masks_frame(stack[t], z=None, params=params)
        else:
            slices = []
            for z in range(Z):
                if progress_cb is not None:
                    progress_cb(
                        t, T,
                        f"Nucleus masks: frame {t + 1}/{T}, z {z + 1}/{Z}...",
                    )
                slices.append(run_nucleus_masks_frame(stack[t], z=z, params=params))
            masks = offset_slice_labels(slices)
        frames.append(masks.astype(np.int32))
        if progress_cb is not None:
            progress_cb(t + 1, T, f"Nucleus masks: frame {t + 1}/{T}...")
    return np.stack(frames, axis=0)


def run_cell_masks_stack(
    stack: np.ndarray,
    params: CellParams,
    *,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Process a ``(T, Z, Y, X)`` stack slice-by-slice into ``(T, Z, Y, X)`` labels."""
    if stack.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {stack.shape}")
    T, Z = stack.shape[:2]
    frames: list[np.ndarray] = []
    for t in range(T):
        _check_cancel(cancel_cb)
        if progress_cb is not None:
            progress_cb(t, T, f"Cell masks: frame {t + 1}/{T}...")
        slices = [run_cell_masks_frame(stack[t], z=z, params=params) for z in range(Z)]
        frames.append(offset_slice_labels(slices))
        if progress_cb is not None:
            progress_cb(t + 1, T, f"Cell masks: frame {t + 1}/{T}...")
    return np.stack(frames, axis=0)


def write_masks(
    masks_tzyx: np.ndarray,
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
) -> Path:
    """Write ``{channel}_masks.tif`` (``int32``, axes ``TZYX``); return its path."""
    if channel not in ("nucleus", "cell"):
        raise ValueError(f"channel must be 'nucleus' or 'cell', got {channel!r}")
    if masks_tzyx.ndim != 4:
        raise ValueError(f"masks must be (T, Z, Y, X), got {masks_tzyx.shape}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{channel}_masks.tif"
    imwrite_grayscale(
        path, masks_tzyx.astype(np.int32),
        compression="zlib", metadata={"axes": "TZYX"},
    )
    return path
