"""Tests for cellflow.cellpose.native_masks.

cellpose is mocked at import time (no torch/GPU). The fake model returns simple
labelled masks as ``eval`` index 0 so the mask-capturing path is exercised.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile


@pytest.fixture(autouse=True)
def _mock_cellpose(monkeypatch):
    fake_cellpose = types.ModuleType("cellpose")
    fake_models = types.ModuleType("cellpose.models")

    class _FakeModel:
        def __init__(self, *_, **__):
            pass

        def eval(self, img, **_kwargs):
            arr = np.asarray(img, dtype=np.float32)
            # one labelled object filling the frame; flows/styles unused here.
            masks = np.ones(arr.shape, dtype=np.int32)
            if arr.ndim == 2:
                flows = (None, np.zeros((2, *arr.shape), np.float32), np.zeros(arr.shape, np.float32))
            else:
                flows = (None, np.zeros((3, *arr.shape), np.float32), np.zeros(arr.shape, np.float32))
            return masks, flows, None

    fake_models.CellposeModel = _FakeModel
    fake_cellpose.models = fake_models
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)
    for mod in ("cellflow.cellpose.cellpose_runner", "cellflow.cellpose.native_masks"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    yield


def _mod():
    import importlib

    runner = importlib.import_module("cellflow.cellpose.cellpose_runner")
    importlib.reload(runner)
    nm = importlib.import_module("cellflow.cellpose.native_masks")
    return importlib.reload(nm), runner


def _install_recording_model(monkeypatch, runner, label_value=1):
    """Replace the runner's cached model with one returning constant labels."""
    calls = []

    class _Recorder:
        def eval(self, img, **kwargs):
            arr = np.asarray(img, dtype=np.float32)
            calls.append({"shape": arr.shape, **kwargs})
            masks = np.full(arr.shape, label_value, dtype=np.int32)
            return masks, (None, None, None), None

    monkeypatch.setattr(runner, "_MODEL", _Recorder())
    return calls


def test_offset_slice_labels_makes_frame_unique():
    nm, _ = _mod()
    a = np.array([[0, 1], [1, 0]], dtype=np.int32)
    b = np.array([[2, 0], [0, 1]], dtype=np.int32)
    out = nm.offset_slice_labels([a, b])
    assert out.shape == (2, 2, 2)
    # background stays 0; slice b's labels are offset past slice a's max (1).
    assert set(np.unique(out)) == {0, 1, 2, 3}
    assert out[1].max() == 3


def test_offset_slice_labels_single_slice_is_noop():
    nm, _ = _mod()
    a = np.array([[0, 1], [2, 0]], dtype=np.int32)
    out = nm.offset_slice_labels([a])
    np.testing.assert_array_equal(out[0], a)


def test_run_nucleus_masks_frame_3d(monkeypatch):
    nm, runner = _mod()
    calls = _install_recording_model(monkeypatch, runner)
    frame = np.zeros((4, 8, 8), dtype=np.float32)
    params = runner.NucleusParams(do_3d=True, anisotropy=1.5, diameter=25.0, min_size=15, gamma=1.0)
    masks = nm.run_nucleus_masks_frame(frame, z=None, params=params)
    assert masks.shape == (4, 8, 8)
    assert masks.dtype == np.int32
    assert calls[0]["do_3D"] is True
    assert calls[0]["anisotropy"] == 1.5


def test_run_nucleus_masks_frame_2d_slice(monkeypatch):
    nm, runner = _mod()
    calls = _install_recording_model(monkeypatch, runner)
    frame = np.zeros((4, 8, 8), dtype=np.float32)
    params = runner.NucleusParams(do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0)
    masks = nm.run_nucleus_masks_frame(frame, z=2, params=params)
    assert masks.shape == (8, 8)
    assert calls[0]["shape"] == (8, 8)
    assert calls[0]["diameter"] is None


def test_run_nucleus_masks_stack_3d(monkeypatch):
    nm, runner = _mod()
    _install_recording_model(monkeypatch, runner)
    stack = np.zeros((3, 4, 6, 6), dtype=np.float32)
    params = runner.NucleusParams(do_3d=True, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0)
    progress = []
    masks = nm.run_nucleus_masks_stack(
        stack, params, progress_cb=lambda d, t, m: progress.append((d, t, m))
    )
    assert masks.shape == (3, 4, 6, 6)
    assert masks.dtype == np.int32
    assert progress[0][1] == 3


def test_run_nucleus_masks_stack_2d_offsets_per_z(monkeypatch):
    nm, runner = _mod()
    # each z-slice returns label 1; offset must make them unique within a frame.
    _install_recording_model(monkeypatch, runner, label_value=1)
    stack = np.zeros((2, 3, 5, 5), dtype=np.float32)
    params = runner.NucleusParams(do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0)
    masks = nm.run_nucleus_masks_stack(stack, params)
    assert masks.shape == (2, 3, 5, 5)
    # 3 z-slices, each a full-frame object -> labels 1,2,3 within every frame.
    assert sorted(np.unique(masks[0]).tolist()) == [1, 2, 3]


def test_run_cell_masks_stack(monkeypatch):
    nm, runner = _mod()
    calls = _install_recording_model(monkeypatch, runner)
    stack = np.zeros((2, 3, 6, 6), dtype=np.float32)
    params = runner.CellParams(diameter=30.0, min_size=10, gamma=1.0)
    masks = nm.run_cell_masks_stack(stack, params)
    assert masks.shape == (2, 3, 6, 6)
    assert len(calls) == 2 * 3
    assert calls[0]["diameter"] == 30.0


def test_run_cell_masks_stack_respects_cancel(monkeypatch):
    nm, runner = _mod()
    _install_recording_model(monkeypatch, runner)
    stack = np.zeros((5, 2, 4, 4), dtype=np.float32)
    params = runner.CellParams(diameter=0.0, min_size=0, gamma=1.0)
    seen = []

    with pytest.raises(runner.CancelledError):
        nm.run_cell_masks_stack(
            stack, params,
            progress_cb=lambda d, t, m: seen.append(d),
            cancel_cb=lambda: len(seen) >= 1,
        )


def test_write_masks_roundtrip(tmp_path: Path):
    nm, _ = _mod()
    masks = (np.random.rand(2, 3, 4, 5) * 5).astype(np.int32)
    path = nm.write_masks(masks, tmp_path, "nucleus")
    assert path == tmp_path / "nucleus_masks.tif"
    written = tifffile.imread(str(path))
    np.testing.assert_array_equal(written.reshape(masks.shape), masks)


def test_write_masks_creates_dir_and_validates(tmp_path: Path):
    nm, _ = _mod()
    masks = np.zeros((1, 1, 3, 3), dtype=np.int32)
    nm.write_masks(masks, tmp_path / "out", "cell")
    assert (tmp_path / "out" / "cell_masks.tif").exists()
    with pytest.raises(ValueError):
        nm.write_masks(masks, tmp_path, "bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        nm.write_masks(np.zeros((3, 3), np.int32), tmp_path, "cell")
