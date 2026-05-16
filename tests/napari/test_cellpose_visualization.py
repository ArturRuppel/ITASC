from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _load_viz_mod():
    """Import cellpose_visualization without the rest of the package."""
    src_root = Path(__file__).resolve().parents[2] / "src" / "cellflow"
    package_root = src_root / "napari"

    if "cellflow.napari" not in sys.modules:
        pkg = types.ModuleType("cellflow.napari")
        pkg.__path__ = [str(package_root)]
        sys.modules["cellflow.napari"] = pkg

    sys.modules.pop("cellflow.napari.cellpose_visualization", None)
    return importlib.import_module("cellflow.napari.cellpose_visualization")


@pytest.fixture
def mod():
    return _load_viz_mod()


def _write_prob(tmp_path: Path, channel: str, data: np.ndarray) -> Path:
    p = tmp_path / f"{channel}_prob_3dt.tif"
    tifffile.imwrite(str(p), data)
    return p


def _write_dp(tmp_path: Path, channel: str, data: np.ndarray) -> Path:
    p = tmp_path / f"{channel}_dp_3dt.tif"
    tifffile.imwrite(str(p), data)
    return p


# ----- load_sigmoid_prob -----

def test_load_sigmoid_prob_zavg_shape(mod, tmp_path):
    T, Z, Y, X = 3, 4, 8, 8
    data = np.random.randn(T, Z, Y, X).astype(np.float32)
    _write_prob(tmp_path, "nucleus", data)
    result = mod.load_sigmoid_prob(tmp_path, "nucleus", "zavg")
    assert result.shape == (T, Y, X)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_load_sigmoid_prob_3dt_shape(mod, tmp_path):
    T, Z, Y, X = 2, 5, 6, 6
    data = np.random.randn(T, Z, Y, X).astype(np.float32)
    _write_prob(tmp_path, "cell", data)
    result = mod.load_sigmoid_prob(tmp_path, "cell", "3dt")
    assert result.shape == (T, Z, Y, X)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_load_sigmoid_prob_zavg_sigmoid_then_mean(mod, tmp_path):
    """Regression: sigmoid-then-mean must differ from mean-then-sigmoid."""
    T, Z, Y, X = 2, 4, 8, 8
    rng = np.random.default_rng(42)
    data = (rng.random((T, Z, Y, X)) * 10 - 5).astype(np.float32)
    _write_prob(tmp_path, "nucleus", data)

    result = mod.load_sigmoid_prob(tmp_path, "nucleus", "zavg")

    # mean-then-sigmoid would give a different result
    wrong = 1.0 / (1.0 + np.exp(-data.mean(axis=1)))
    assert not np.allclose(result, wrong), (
        "sigmoid-then-mean should differ from mean-then-sigmoid"
    )


# ----- load_flow_vectors -----

def test_load_flow_vectors_zavg_shape(mod, tmp_path):
    T, Z, Y, X = 2, 4, 16, 16
    dp = np.random.randn(T, Z, 2, Y, X).astype(np.float32)
    _write_dp(tmp_path, "nucleus", dp)
    stride = 4
    result = mod.load_flow_vectors(tmp_path, "nucleus", "zavg", stride=stride, scale=1.0)
    N_expected = T * (Y // stride) * (X // stride)
    assert result.shape == (N_expected, 2, 3)


def test_load_flow_vectors_zavg_respects_stride(mod, tmp_path):
    T, Z, Y, X = 1, 2, 12, 12
    dp = np.ones((T, Z, 2, Y, X), dtype=np.float32)
    _write_dp(tmp_path, "nucleus", dp)
    stride = 3
    result = mod.load_flow_vectors(tmp_path, "nucleus", "zavg", stride=stride, scale=1.0)
    N_expected = T * (Y // stride) * (X // stride)
    assert result.shape == (N_expected, 2, 3)


def test_load_flow_vectors_3dt_shape_and_dz_zero(mod, tmp_path):
    T, Z, Y, X = 2, 3, 16, 16
    dp = np.random.randn(T, Z, 2, Y, X).astype(np.float32)
    _write_dp(tmp_path, "nucleus", dp)
    stride = 4
    result = mod.load_flow_vectors(tmp_path, "nucleus", "3dt", stride=stride, scale=1.0)
    N_expected = T * Z * (Y // stride) * (X // stride)
    assert result.shape == (N_expected, 2, 4)
    # dz component: result[:, 1, 1] is the Z-delta of the displacement vector
    dz_components = result[:, 1, 1]  # Z-delta is at index 1 of the delta vector
    assert np.all(dz_components == 0.0)


def test_load_flow_vectors_scale_applied(mod, tmp_path):
    T, Z, Y, X = 1, 2, 8, 8
    dy_val, dx_val = 2.0, 3.0
    dp = np.zeros((T, Z, 2, Y, X), dtype=np.float32)
    dp[:, :, 0, :, :] = dy_val  # dy channel
    dp[:, :, 1, :, :] = dx_val  # dx channel
    _write_dp(tmp_path, "nucleus", dp)
    stride = 2
    scale = 0.5
    result = mod.load_flow_vectors(tmp_path, "nucleus", "zavg", stride=stride, scale=scale)
    # After Z-mean, dy=2.0, dx=3.0. After scale: dy*0.5=1.0, dx*0.5=1.5
    dy_col = result[:, 1, 1]  # dy is at index 1 of the delta vector for zavg (t, dy, dx)
    dx_col = result[:, 1, 2]  # dx is at index 2
    assert np.allclose(dy_col, dy_val * scale)
    assert np.allclose(dx_col, dx_val * scale)


# ----- add_cellpose_viz_layers -----

import napari
from qtpy.QtWidgets import QApplication


def _make_viewer():
    app = QApplication.instance() or QApplication([])
    viewer = napari.Viewer(show=False)
    return app, viewer


def test_add_cellpose_viz_layers_adds_prob_and_flow(mod, tmp_path):
    T, Z, Y, X = 2, 3, 16, 16
    prob = np.random.randn(T, Z, Y, X).astype(np.float32)
    dp = np.random.randn(T, Z, 2, Y, X).astype(np.float32)
    _write_prob(tmp_path, "nucleus", prob)
    _write_dp(tmp_path, "nucleus", dp)

    app, viewer = _make_viewer()
    try:
        layers = mod.add_cellpose_viz_layers(
            viewer, tmp_path, "nucleus", "zavg", stride=4, scale=1.0
        )
        assert len(layers) == 2
        names = [la.name for la in viewer.layers]
        assert "Cellpose viz: nucleus prob (z-avg)" in names
        assert "Cellpose viz: nucleus flow (z-avg)" in names
    finally:
        viewer.close()
        app.processEvents()


def test_add_cellpose_viz_layers_replaces_existing_layers(mod, tmp_path):
    T, Z, Y, X = 2, 3, 16, 16
    prob = np.random.randn(T, Z, Y, X).astype(np.float32)
    dp = np.random.randn(T, Z, 2, Y, X).astype(np.float32)
    _write_prob(tmp_path, "nucleus", prob)
    _write_dp(tmp_path, "nucleus", dp)

    app, viewer = _make_viewer()
    try:
        mod.add_cellpose_viz_layers(
            viewer, tmp_path, "nucleus", "zavg", stride=4, scale=1.0
        )
        assert len(viewer.layers) == 2

        mod.add_cellpose_viz_layers(
            viewer, tmp_path, "nucleus", "zavg", stride=8, scale=2.0
        )
        # Should still be exactly 2 layers (old ones removed, new ones added)
        assert len(viewer.layers) == 2
    finally:
        viewer.close()
        app.processEvents()


def test_add_cellpose_viz_layers_missing_files_returns_empty(mod, tmp_path):
    # Neither prob nor dp files present
    app, viewer = _make_viewer()
    try:
        layers = mod.add_cellpose_viz_layers(
            viewer, tmp_path, "nucleus", "zavg", stride=4, scale=1.0
        )
        assert layers == []
        assert len(viewer.layers) == 0
    finally:
        viewer.close()
        app.processEvents()


def test_add_cellpose_viz_layers_3dt_mode(mod, tmp_path):
    T, Z, Y, X = 2, 3, 16, 16
    prob = np.random.randn(T, Z, Y, X).astype(np.float32)
    dp = np.random.randn(T, Z, 2, Y, X).astype(np.float32)
    _write_prob(tmp_path, "nucleus", prob)
    _write_dp(tmp_path, "nucleus", dp)

    app, viewer = _make_viewer()
    try:
        layers = mod.add_cellpose_viz_layers(
            viewer, tmp_path, "nucleus", "3dt", stride=4, scale=1.0
        )
        assert len(layers) == 2
        names = [la.name for la in viewer.layers]
        assert "Cellpose viz: nucleus prob (3D+t)" in names
        assert "Cellpose viz: nucleus flow (3D+t)" in names
    finally:
        viewer.close()
        app.processEvents()
