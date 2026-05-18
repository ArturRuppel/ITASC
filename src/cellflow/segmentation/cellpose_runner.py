"""Local Cellpose-SAM runner, Qt-free, used by the napari Cellpose widget."""
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


class CancelledError(RuntimeError):
    """Raised by run_*_stack when cancel_cb returns True between frames."""


def _check_cancel(cancel_cb: Callable[[], bool] | None) -> None:
    if cancel_cb is not None and cancel_cb():
        raise CancelledError("cellpose run cancelled")


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


def run_nucleus_stack(
    stack: np.ndarray,
    params: NucleusParams,
    *,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a (T, Z, Y, X) stack frame-by-frame.

    Returns (prob_3dt, dp_3dt). For do_3d=True dp has shape (T, 3, Z, Y, X);
    for do_3d=False dp has shape (T, Z, 2, Y, X).
    """
    if stack.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {stack.shape}")
    T = stack.shape[0]
    prob_frames: list[np.ndarray] = []
    dp_frames: list[np.ndarray] = []
    for t in range(T):
        _check_cancel(cancel_cb)
        if progress_cb is not None:
            progress_cb(t, T, f"Nucleus: frame {t + 1}/{T}...")
        if params.do_3d:
            prob, dp = run_nucleus_frame(stack[t], z=None, params=params)
        else:
            Z = stack.shape[1]
            slice_probs: list[np.ndarray] = []
            slice_dps: list[np.ndarray] = []
            for z in range(Z):
                if progress_cb is not None:
                    progress_cb(
                        t,
                        T,
                        f"Nucleus: frame {t + 1}/{T}, z {z + 1}/{Z}...",
                    )
                p, d = run_nucleus_frame(stack[t], z=z, params=params)
                slice_probs.append(p)
                slice_dps.append(d)
            prob = np.stack(slice_probs, axis=0)
            dp = np.stack(slice_dps, axis=0)
        prob_frames.append(prob)
        dp_frames.append(dp)
        if progress_cb is not None:
            progress_cb(t + 1, T, f"Nucleus: frame {t + 1}/{T}...")
    return np.stack(prob_frames, axis=0), np.stack(dp_frames, axis=0)


def run_cell_stack(
    stack: np.ndarray,
    params: CellParams,
    *,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a (T, Z, Y, X) stack slice-by-slice in 2D.

    Returns (prob_3dt (T, Z, Y, X), dp_3dt (T, Z, 2, Y, X)).
    """
    if stack.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {stack.shape}")
    T, Z = stack.shape[:2]
    prob_frames: list[np.ndarray] = []
    dp_frames: list[np.ndarray] = []
    for t in range(T):
        _check_cancel(cancel_cb)
        if progress_cb is not None:
            progress_cb(t, T, f"Cell: frame {t + 1}/{T}...")
        slice_probs: list[np.ndarray] = []
        slice_dps: list[np.ndarray] = []
        for z in range(Z):
            p, d = run_cell_frame(stack[t], z=z, params=params)
            slice_probs.append(p)
            slice_dps.append(d)
        prob_frames.append(np.stack(slice_probs, axis=0))
        dp_frames.append(np.stack(slice_dps, axis=0))
        if progress_cb is not None:
            progress_cb(t + 1, T, f"Cell: frame {t + 1}/{T}...")
    return np.stack(prob_frames, axis=0), np.stack(dp_frames, axis=0)


def write_outputs(
    prob_3dt: np.ndarray,
    dp_3dt: np.ndarray,
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
) -> None:
    """Write the three canonical TIFFs under output_dir.

    Writes ``{channel}_prob_3dt.tif``, ``{channel}_dp_3dt.tif``, and the
    z-averaged probability ``{channel}_prob_zavg.tif``.
    """
    if channel not in ("nucleus", "cell"):
        raise ValueError(f"channel must be 'nucleus' or 'cell', got {channel!r}")
    if prob_3dt.ndim != 4:
        raise ValueError(f"prob_3dt must be (T, Z, Y, X), got {prob_3dt.shape}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prob_path = output_dir / f"{channel}_prob_3dt.tif"
    dp_path = output_dir / f"{channel}_dp_3dt.tif"
    zavg_path = output_dir / f"{channel}_prob_zavg.tif"
    tifffile.imwrite(str(prob_path), prob_3dt.astype(np.float32), compression="zlib")
    tifffile.imwrite(str(dp_path), dp_3dt.astype(np.float32), compression="zlib")
    zavg = prob_3dt.mean(axis=1, dtype=np.float32).astype(np.float32)
    tifffile.imwrite(str(zavg_path), zavg, compression="zlib")
