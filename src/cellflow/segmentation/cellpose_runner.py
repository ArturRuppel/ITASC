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
