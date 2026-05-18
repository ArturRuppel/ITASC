"""Local Cellpose-SAM runner — Qt-free, used by the napari Cellpose widget.

Vendored and adapted from /home/aruppel/Projects/HPC/cellpose_full/cellpose_full.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import tifffile

_NORMALIZE = {"tile_norm_blocksize": 128}


@dataclass(frozen=True)
class NucleusParams:
    do_3d: bool
    anisotropy: float
    diameter: float  # 0 means "let cpsam decide" (None passed to model)
    min_size: int
    gamma: float


@dataclass(frozen=True)
class CellParams:
    diameter: float
    min_size: int
    gamma: float


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Min/max-normalized gamma correction matching cellpose_full.py."""
    if gamma == 1.0:
        return np.asarray(img)
    img = np.asarray(img, dtype=np.float32)
    img_min = float(np.min(img))
    img_max = float(np.max(img))
    if img_max <= img_min:
        return img
    scaled = (img - img_min) / (img_max - img_min)
    return (scaled ** gamma) * (img_max - img_min) + img_min


def _diameter_kwarg(diameter: float) -> float | None:
    return None if diameter == 0 else float(diameter)


_MODEL = None


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def device_label() -> str:
    return "cuda:0" if _cuda_available() else "cpu"


def is_model_loaded() -> bool:
    return _MODEL is not None


def get_model():
    """Lazy-load the cpsam model once per process; cached at module level."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from cellpose.models import CellposeModel

    use_gpu = _cuda_available()
    _MODEL = CellposeModel(
        gpu=use_gpu,
        pretrained_model="cpsam",
        use_bfloat16=use_gpu,
    )
    return _MODEL


def run_nucleus_frame(
    frame: np.ndarray,
    z: int | None,
    params: NucleusParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-frame nucleus inference.

    If ``z`` is None, runs full 3D over (Z, Y, X) and returns
    prob with shape (Z, Y, X) and dp with shape (3, Z, Y, X).
    If ``z`` is an integer, runs 2D on frame[z] and returns
    prob with shape (Y, X) and dp with shape (2, Y, X).
    """
    model = get_model()
    diameter = _diameter_kwarg(params.diameter)
    if z is None:
        volume = _apply_gamma(frame, params.gamma)
        _, flows, _ = model.eval(
            volume,
            do_3D=True,
            z_axis=0,
            diameter=diameter,
            anisotropy=params.anisotropy,
            min_size=params.min_size,
            normalize=_NORMALIZE,
        )
    else:
        slice_2d = _apply_gamma(frame[z], params.gamma)
        _, flows, _ = model.eval(
            slice_2d,
            diameter=diameter,
            min_size=params.min_size,
            normalize=_NORMALIZE,
        )
    dp = np.asarray(flows[1], dtype=np.float32)
    prob = np.asarray(flows[2], dtype=np.float32)
    return prob, dp


def run_cell_frame(
    frame: np.ndarray,
    z: int,
    params: CellParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Single 2D-slice cell inference. Returns (prob (Y,X), dp (2,Y,X))."""
    model = get_model()
    diameter = _diameter_kwarg(params.diameter)
    slice_2d = _apply_gamma(frame[z], params.gamma)
    _, flows, _ = model.eval(
        slice_2d,
        diameter=diameter,
        min_size=params.min_size,
        normalize=_NORMALIZE,
    )
    dp = np.asarray(flows[1], dtype=np.float32)
    prob = np.asarray(flows[2], dtype=np.float32)
    return prob, dp
