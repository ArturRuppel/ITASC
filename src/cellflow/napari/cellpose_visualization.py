from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile

__all__ = [
    "load_sigmoid_prob",
    "load_flow_vectors",
    "add_cellpose_viz_layers",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float32)))).astype(np.float32)


def _prob_filename(channel: str) -> str:
    return f"{channel}_prob_3dt.tif"


def _dp_filename(channel: str) -> str:
    return f"{channel}_dp_3dt.tif"


def _mode_label(mode: str) -> str:
    return "z-avg" if mode == "zavg" else "3D+t"


def _layer_names(channel: str, mode: str) -> tuple[str, str]:
    label = _mode_label(mode)
    return (
        f"Cellpose viz: {channel} prob ({label})",
        f"Cellpose viz: {channel} flow ({label})",
    )


def load_sigmoid_prob(
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
    mode: Literal["zavg", "3dt"],
) -> np.ndarray:
    path = Path(output_dir) / _prob_filename(channel)
    raw = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    prob = _sigmoid(raw)  # (T, Z, Y, X)
    if mode == "zavg":
        return prob.mean(axis=1)  # (T, Y, X)
    return prob  # (T, Z, Y, X)


def load_flow_vectors(
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
    mode: Literal["zavg", "3dt"],
    *,
    stride: int,
    scale: float,
) -> np.ndarray:
    path = Path(output_dir) / _dp_filename(channel)
    dp = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    # dp shape: (T, Z, 2, Y, X) — channels are (dy, dx)
    T, Z, _, Y, X = dp.shape

    if mode == "zavg":
        # mean over Z → (T, 2, Y, X)
        dp_2d = dp.mean(axis=1)
        ys = np.arange(0, Y, stride)
        xs = np.arange(0, X, stride)
        ts = np.arange(T)
        tg, yg, xg = np.meshgrid(ts, ys, xs, indexing="ij")  # (T, nY, nX)
        tg = tg.ravel().astype(np.float32)
        yg_r = yg.ravel().astype(np.float32)
        xg_r = xg.ravel().astype(np.float32)
        dy = dp_2d[:, 0, :, :][:, ys[:, None], xs[None, :]].ravel() * scale
        dx = dp_2d[:, 1, :, :][:, ys[:, None], xs[None, :]].ravel() * scale
        N = len(tg)
        vectors = np.zeros((N, 2, 3), dtype=np.float32)
        vectors[:, 0, 0] = tg
        vectors[:, 0, 1] = yg_r
        vectors[:, 0, 2] = xg_r
        vectors[:, 1, 1] = dy
        vectors[:, 1, 2] = dx
        return vectors

    # 3D+t mode
    ys = np.arange(0, Y, stride)
    xs = np.arange(0, X, stride)
    ts = np.arange(T)
    zs = np.arange(Z)
    tg, zg, yg, xg = np.meshgrid(ts, zs, ys, xs, indexing="ij")  # (T, Z, nY, nX)
    tg = tg.ravel().astype(np.float32)
    zg_r = zg.ravel().astype(np.float32)
    yg_r = yg.ravel().astype(np.float32)
    xg_r = xg.ravel().astype(np.float32)
    dy = dp[:, :, 0, :, :][:, :, ys[:, None], xs[None, :]].ravel() * scale
    dx = dp[:, :, 1, :, :][:, :, ys[:, None], xs[None, :]].ravel() * scale
    N = len(tg)
    vectors = np.zeros((N, 2, 4), dtype=np.float32)
    vectors[:, 0, 0] = tg
    vectors[:, 0, 1] = zg_r
    vectors[:, 0, 2] = yg_r
    vectors[:, 0, 3] = xg_r
    # vectors[:, 1, 1] = 0  (dz = 0, already zeros)
    vectors[:, 1, 2] = dy
    vectors[:, 1, 3] = dx
    return vectors


def add_cellpose_viz_layers(
    viewer: Any,
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
    mode: Literal["zavg", "3dt"],
    *,
    stride: int,
    scale: float,
) -> list[Any]:
    output_dir = Path(output_dir)
    prob_path = output_dir / _prob_filename(channel)
    dp_path = output_dir / _dp_filename(channel)

    prob_name, flow_name = _layer_names(channel, mode)

    # Remove pre-existing layers for this (channel, mode)
    for name in (prob_name, flow_name):
        to_remove = [la for la in list(viewer.layers) if la.name == name]
        for la in to_remove:
            viewer.layers.remove(la)

    if not prob_path.is_file() or not dp_path.is_file():
        return []

    prob_data = load_sigmoid_prob(output_dir, channel, mode)
    flow_data = load_flow_vectors(output_dir, channel, mode, stride=stride, scale=scale)

    prob_layer = viewer.add_image(
        prob_data,
        name=prob_name,
        colormap="magma",
        contrast_limits=(0.0, 1.0),
        blending="translucent",
    )
    flow_layer = viewer.add_vectors(
        flow_data,
        name=flow_name,
        edge_width=1,
        length=1,
        vector_style="arrow",
        edge_color="cyan",
    )
    return [prob_layer, flow_layer]
