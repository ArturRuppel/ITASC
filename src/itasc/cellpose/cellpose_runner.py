"""Local Cellpose-SAM runner, Qt-free, used by the napari Cellpose widget."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from collections.abc import Callable

import numpy as np

from itasc.core.tiff import imwrite_grayscale

_NORMALIZE = {"tile_norm_blocksize": 128}

InputLayout = Literal["2D", "2D+t", "3D", "3D+t"]

# layout -> (has_time, has_z); canonical processing shape is always (T, Z, Y, X).
_LAYOUT_AXES: dict[str, tuple[bool, bool]] = {
    "2D": (False, False),
    "2D+t": (True, False),
    "3D": (False, True),
    "3D+t": (True, True),
}


def layout_has_time(layout: str) -> bool:
    return _LAYOUT_AXES[layout][0]


def layout_has_z(layout: str) -> bool:
    return _LAYOUT_AXES[layout][1]


def infer_layout_from_ndim(ndim: int) -> str | None:
    """Best-effort layout from array ndim.

    ``2 -> "2D"`` and ``4 -> "3D+t"`` are unambiguous; ``3`` is ambiguous
    (``2D+t`` vs ``3D``) so returns ``None`` and the caller keeps the user's
    explicit choice.
    """
    if ndim == 2:
        return "2D"
    if ndim == 4:
        return "3D+t"
    return None


def to_tzyx(arr: np.ndarray, layout: str) -> np.ndarray:
    """Normalize an input array to canonical ``(T, Z, Y, X)`` for its layout.

    Singleton ``T`` and/or ``Z`` axes are inserted so 2D/2D+t/3D/3D+t inputs all
    become 4-D; the runner then iterates uniformly over frames and z-slices.
    """
    if layout not in _LAYOUT_AXES:
        raise ValueError(f"unknown input layout {layout!r}")
    has_time, has_z = _LAYOUT_AXES[layout]
    arr = np.asarray(arr)
    expected_ndim = 2 + int(has_time) + int(has_z)
    if arr.ndim != expected_ndim:
        raise ValueError(
            f"{layout} input must be {expected_ndim}-D, got shape {arr.shape}"
        )
    if not has_time:
        arr = arr[np.newaxis]  # add T at axis 0
    if not has_z:
        arr = arr[:, np.newaxis]  # add Z at axis 1 (after T)
    return arr


@dataclass(frozen=True)
class NucleusParams:
    do_3d: bool
    anisotropy: float
    diameter: float  # 0 means "let cpsam decide" (None passed to model)
    min_size: int
    gamma: float
    # Cellpose cellprob (logit) threshold; 0.0 is Cellpose's default. Defaulted so
    # the app (which never sets it) and existing callers are byte-for-byte unchanged.
    cellprob_threshold: float = 0.0
    # Cellpose flow-error QC: masks whose flow error exceeds this are removed; 0.4
    # is Cellpose's default (0 disables QC, higher keeps more masks).
    flow_threshold: float = 0.4
    # Euler-integration steps for the flow dynamics; 0 -> auto (None passed to
    # Cellpose, which derives it from the diameter). Both defaulted, so the app and
    # existing callers stay byte-for-byte unchanged.
    niter: int = 0


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


def run_cell_frame_joint(
    cell_frame: np.ndarray,
    nucleus_frame: np.ndarray,
    z: int,
    params: CellParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-channel cell inference on one 2D z-slice, guided by the nucleus channel.

    A single Cellpose-SAM pass runs over a two-channel image (cell + nucleus) so
    the nucleus channel informs the cell-body flow/probability fields. Returns
    ``(prob (Y, X), dp (2, Y, X))`` — the same shapes as :func:`run_cell_frame`,
    so the two-channel result is a drop-in replacement for the single-channel one.

    Channel order matches the classic ``[cytoplasm, nucleus]`` convention and the
    single-channel cell path (which places the cell channel at index 0): the cell
    channel is the primary "to-segment" channel (index 0), the nucleus is the
    auxiliary guide (index 1). ``channel_axis`` is passed explicitly (cpsam
    zero-pads to 3 channels internally); the same per-plane gamma is applied to
    both channels and cpsam normalises each channel independently.
    """
    model = get_model()
    diameter = _diameter_kwarg(params.diameter)
    cell_2d = _apply_gamma(cell_frame[z], params.gamma)
    nucleus_2d = _apply_gamma(nucleus_frame[z], params.gamma)
    img = np.stack([cell_2d, nucleus_2d], axis=-1)  # (Y, X, 2), cell first
    _, flows, _ = model.eval(
        img,
        channel_axis=2,
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


def run_cell_stack_joint(
    cell_stack: np.ndarray,
    nucleus_stack: np.ndarray,
    params: CellParams,
    *,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-channel cell inference over paired ``(T, Z, Y, X)`` cell + nucleus stacks.

    Like :func:`run_cell_stack` but each 2D slice is segmented from a two-channel
    image (cell + nucleus) so the nucleus channel guides the cell-body flow.
    Returns ``(prob_3dt (T, Z, Y, X), dp_3dt (T, Z, 2, Y, X))`` — identical shapes
    to :func:`run_cell_stack`, so downstream flow-following is unaffected.
    """
    cell_stack = np.asarray(cell_stack)
    nucleus_stack = np.asarray(nucleus_stack)
    if cell_stack.ndim != 4 or nucleus_stack.ndim != 4:
        raise ValueError(
            f"expected (T, Z, Y, X), got {cell_stack.shape} and {nucleus_stack.shape}"
        )
    if cell_stack.shape != nucleus_stack.shape:
        raise ValueError(
            "cell_stack and nucleus_stack must have the same shape, got "
            f"{cell_stack.shape} and {nucleus_stack.shape}"
        )
    T, Z = cell_stack.shape[:2]
    prob_frames: list[np.ndarray] = []
    dp_frames: list[np.ndarray] = []
    for t in range(T):
        _check_cancel(cancel_cb)
        if progress_cb is not None:
            progress_cb(t, T, f"Cell: frame {t + 1}/{T}...")
        slice_probs: list[np.ndarray] = []
        slice_dps: list[np.ndarray] = []
        for z in range(Z):
            p, d = run_cell_frame_joint(cell_stack[t], nucleus_stack[t], z=z, params=params)
            slice_probs.append(p)
            slice_dps.append(d)
        prob_frames.append(np.stack(slice_probs, axis=0))
        dp_frames.append(np.stack(slice_dps, axis=0))
        if progress_cb is not None:
            progress_cb(t + 1, T, f"Cell: frame {t + 1}/{T}...")
    return np.stack(prob_frames, axis=0), np.stack(dp_frames, axis=0)


def _dp_axes(dp_3dt: np.ndarray) -> str | None:
    """Axis labels for the flow stack so singleton T/Z survive the TIFF round-trip.

    2D-per-slice flow is ``(T, Z, 2, Y, X)``; true-3D flow is ``(T, 3, Z, Y, X)``.
    """
    if dp_3dt.ndim == 5 and dp_3dt.shape[2] == 2:
        return "TZCYX"
    if dp_3dt.ndim == 5 and dp_3dt.shape[1] == 3:
        return "TCZYX"
    return None


def write_outputs(
    prob_3dt: np.ndarray,
    dp_3dt: np.ndarray,
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
) -> None:
    """Write the two canonical TIFFs under output_dir.

    Writes ``{channel}_prob.tif`` and ``{channel}_dp.tif``. Axis labels
    are recorded as metadata so singleton ``T``/``Z`` axes (2D / 2D+t / single
    3D-stack inputs) survive the TIFF round-trip and are not misread downstream.
    """
    if channel not in ("nucleus", "cell"):
        raise ValueError(f"channel must be 'nucleus' or 'cell', got {channel!r}")
    if prob_3dt.ndim != 4:
        raise ValueError(f"prob_3dt must be (T, Z, Y, X), got {prob_3dt.shape}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prob_path = output_dir / f"{channel}_prob.tif"
    dp_path = output_dir / f"{channel}_dp.tif"
    imwrite_grayscale(
        prob_path, prob_3dt.astype(np.float32),
        compression="zlib", metadata={"axes": "TZYX"},
    )
    dp_kwargs: dict = {"compression": "zlib"}
    dp_axes = _dp_axes(dp_3dt)
    if dp_axes is not None:
        dp_kwargs["metadata"] = {"axes": dp_axes}
    imwrite_grayscale(dp_path, dp_3dt.astype(np.float32), **dp_kwargs)
