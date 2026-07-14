"""Capture Cellpose-SAM *native* masks, Qt-free, for the standalone distro.

The app's :mod:`itasc.cellpose.cellpose_runner` keeps only the probability /
flow maps (``_, flows, _ = model.eval(...)``) because the integrated pipeline
derives its labels from divergence maps + Ultrack. The independently-shipped
``itasc-cellpose`` tool instead wants the labelled masks Cellpose computes and
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

from itasc.core.tiff import imwrite_grayscale
from itasc.cellpose import cellpose_runner
from itasc.cellpose.cellpose_runner import (
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
    "run_nucleus_maps_frame",
    "iter_nucleus_maps_stack",
    "write_masks",
    "offset_slice_labels",
]


def _niter_kwarg(niter: int) -> int | None:
    """``0`` means auto — pass ``None`` so Cellpose derives niter from the diameter."""
    return None if int(niter) == 0 else int(niter)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Map Cellpose's cellprob logits to a ``[0, 1]`` probability map."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))


def _flow_to_rgb(dp: np.ndarray) -> np.ndarray:
    """HSV→RGB visualization of a ``(2, Y, X)`` flow field → ``(Y, X, 3)`` uint8.

    Hue encodes flow direction (the colour wheel Cellpose users recognise) and
    value encodes magnitude (per-frame normalised), so touching objects pulled
    apart by their flows are visible. Saturation is fixed at 1.
    """
    dp = np.asarray(dp, dtype=np.float32)
    dy, dx = dp[0], dp[1]
    hue = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)  # [0, 1)
    mag = np.hypot(dy, dx)
    mmax = float(mag.max()) if mag.size else 0.0
    val = mag / mmax if mmax > 0.0 else np.zeros_like(mag)
    h6 = hue * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = np.zeros_like(val)         # saturation 1 → p = v*(1-s) = 0
    q = val * (1.0 - f)
    tt = val * f
    conds = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    r = np.select(conds, [val, q, p, p, tt, val])
    g = np.select(conds, [tt, val, val, q, p, p])
    b = np.select(conds, [p, p, tt, val, val, q])
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _eval_maps(img: np.ndarray, **eval_kwargs):
    """Run ``model.eval`` once and keep masks **and** the maps it would discard.

    Returns ``(masks int32, prob (sigmoid of cellprob) float32, dp float32)`` —
    the labelled masks, the probability map and the raw flow field — from a single
    forward pass, so no channel is segmented twice to surface its intermediates.
    """
    model = cellpose_runner.get_model()
    masks, flows, _styles = model.eval(img, normalize=_NORMALIZE, **eval_kwargs)
    dp = np.asarray(flows[1], dtype=np.float32)
    cellprob = np.asarray(flows[2], dtype=np.float32)
    return np.asarray(masks, dtype=np.int32), _sigmoid(cellprob), dp


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
    niter = _niter_kwarg(params.niter)
    if z is None:
        volume = _apply_gamma(frame, params.gamma)
        return _eval_masks(
            volume,
            do_3D=True,
            z_axis=0,
            diameter=diameter,
            anisotropy=params.anisotropy,
            min_size=params.min_size,
            cellprob_threshold=params.cellprob_threshold,
            flow_threshold=params.flow_threshold,
            niter=niter,
        )
    slice_2d = _apply_gamma(frame[z], params.gamma)
    return _eval_masks(
        slice_2d,
        diameter=diameter,
        min_size=params.min_size,
        cellprob_threshold=params.cellprob_threshold,
        flow_threshold=params.flow_threshold,
        niter=niter,
    )


def run_nucleus_maps_frame(
    frame: np.ndarray,
    z: int,
    params: NucleusParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native masks **+ sigmoid prob map + RGB flow** for one 2D z-slice.

    A single ``model.eval`` pass yields ``(masks (Y, X) int32, prob (Y, X)
    float32, flow_rgb (Y, X, 3) uint8)`` — the labels plus the two intermediates
    (probability and flow) the standalone tool surfaces so Channel 1 can be tuned.
    The standalone anchor is always per-plane 2D, so ``z`` is an integer slice.
    """
    diameter = _diameter_kwarg(params.diameter)
    slice_2d = _apply_gamma(frame[z], params.gamma)
    masks, prob, dp = _eval_maps(
        slice_2d,
        diameter=diameter,
        min_size=params.min_size,
        cellprob_threshold=params.cellprob_threshold,
        flow_threshold=params.flow_threshold,
        niter=_niter_kwarg(params.niter),
    )
    return masks, prob, _flow_to_rgb(dp)


def iter_nucleus_maps_stack(
    stack: np.ndarray,
    params: NucleusParams,
    *,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
):
    """Yield per-frame ``(t, masks, prob, flow_rgb)`` so results stream into a viewer.

    For a ``(T, Z, Y, X)`` input each yield carries one time-frame: ``masks`` and
    ``prob`` are ``(Z, Y, X)`` and ``flow_rgb`` is ``(Z, Y, X, 3)`` uint8, with
    per-frame-unique mask labels (:func:`offset_slice_labels`). Always per-plane 2D
    (the standalone anchor never runs do_3d). A caller can drop each frame into a
    napari layer as it arrives instead of waiting for the whole stack.
    """
    stack = np.asarray(stack)
    if stack.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {stack.shape}")
    T, Z = stack.shape[:2]
    for t in range(T):
        _check_cancel(cancel_cb)
        mask_slices: list[np.ndarray] = []
        prob_slices: list[np.ndarray] = []
        flow_slices: list[np.ndarray] = []
        for z in range(Z):
            if progress_cb is not None:
                progress_cb(t, T, f"Channel 1 maps: frame {t + 1}/{T}, z {z + 1}/{Z}...")
            m, p, f = run_nucleus_maps_frame(stack[t], z=z, params=params)
            mask_slices.append(m)
            prob_slices.append(p)
            flow_slices.append(f)
        masks = offset_slice_labels(mask_slices)
        prob = np.stack(prob_slices, axis=0).astype(np.float32)
        flow = np.stack(flow_slices, axis=0)
        if progress_cb is not None:
            progress_cb(t + 1, T, f"Channel 1 maps: frame {t + 1}/{T}...")
        yield t, masks, prob, flow


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
