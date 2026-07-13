"""Tests for cellflow.cellpose.cellpose_runner.

The cellpose package is mocked at import time so these tests run without
torch/GPU.
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
    """Install a fake cellpose package so the runner imports cleanly."""
    fake_cellpose = types.ModuleType("cellpose")
    fake_models = types.ModuleType("cellpose.models")

    class _FakeModel:
        def __init__(self, *_, **__):
            pass

        def eval(self, img, **_kwargs):
            img = np.asarray(img, dtype=np.float32)
            if img.ndim == 2:
                dp = np.zeros((2, *img.shape), dtype=np.float32)
                prob = np.zeros(img.shape, dtype=np.float32)
            else:
                dp = np.zeros((3, *img.shape), dtype=np.float32)
                prob = np.zeros(img.shape, dtype=np.float32)
            return None, (None, dp, prob), None

    fake_models.CellposeModel = _FakeModel
    fake_cellpose.models = fake_models
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)

    # Ensure the runner module is freshly imported under the mock.
    monkeypatch.delitem(sys.modules, "cellflow.cellpose.cellpose_runner", raising=False)
    yield


def _runner():
    import importlib

    parent = sys.modules.get("cellflow.cellpose")
    if parent is not None and hasattr(parent, "cellpose_runner"):
        delattr(parent, "cellpose_runner")
    sys.modules.pop("cellflow.cellpose.cellpose_runner", None)
    runner = importlib.import_module("cellflow.cellpose.cellpose_runner")
    return importlib.reload(runner)


def test_dataclasses_have_expected_fields():
    r = _runner()
    n = r.NucleusParams(do_3d=True, anisotropy=1.5, diameter=25.0, min_size=15, gamma=1.0)
    c = r.CellParams(diameter=0.0, min_size=0, gamma=1.0)
    assert n.do_3d is True
    assert n.anisotropy == 1.5
    assert n.diameter == 25.0
    assert n.min_size == 15
    assert n.gamma == 1.0
    assert c.diameter == 0.0
    assert c.min_size == 0
    assert c.gamma == 1.0


def test_dataclasses_are_frozen():
    r = _runner()
    n = r.NucleusParams(do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0)
    with pytest.raises(Exception):
        n.do_3d = True


def test_apply_gamma_identity_when_one():
    r = _runner()
    img = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = r._apply_gamma(img, 1.0)
    np.testing.assert_array_equal(out, img)


def test_apply_gamma_handles_constant_image():
    r = _runner()
    img = np.full((4, 4), 7.0, dtype=np.float32)
    out = r._apply_gamma(img, 0.5)
    np.testing.assert_array_equal(out, img)


def test_apply_gamma_warps_dynamic_range():
    r = _runner()
    img = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    out = r._apply_gamma(img, 2.0)
    expected = img ** 2.0
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_is_model_loaded_false_initially():
    r = _runner()
    assert r.is_model_loaded() is False


def test_get_model_caches_across_calls(monkeypatch):
    r = _runner()
    calls = {"n": 0}
    import cellpose.models as models

    real = models.CellposeModel

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(models, "CellposeModel", _counting)
    a = r.get_model()
    b = r.get_model()
    assert a is b
    assert calls["n"] == 1
    assert r.is_model_loaded() is True


def test_get_model_uses_cpu_when_cuda_unavailable(monkeypatch):
    r = _runner()
    received_kwargs = {}
    import cellpose.models as models

    class _Probe:
        def __init__(self, **kwargs):
            received_kwargs.update(kwargs)

    monkeypatch.setattr(models, "CellposeModel", _Probe)
    monkeypatch.setattr(r, "_cuda_available", lambda: False)
    r.get_model()
    assert received_kwargs["gpu"] is False
    assert received_kwargs["pretrained_model"] == "cpsam"
    assert received_kwargs["use_bfloat16"] is False


def test_get_model_uses_gpu_when_cuda_available(monkeypatch):
    r = _runner()
    received_kwargs = {}
    import cellpose.models as models

    class _Probe:
        def __init__(self, **kwargs):
            received_kwargs.update(kwargs)

    monkeypatch.setattr(models, "CellposeModel", _Probe)
    monkeypatch.setattr(r, "_cuda_available", lambda: True)
    r.get_model()
    assert received_kwargs["gpu"] is True
    assert received_kwargs["use_bfloat16"] is True


def test_device_label_reflects_cuda(monkeypatch):
    r = _runner()
    monkeypatch.setattr(r, "_cuda_available", lambda: True)
    assert r.device_label() == "cuda:0"
    monkeypatch.setattr(r, "_cuda_available", lambda: False)
    assert r.device_label() == "cpu"


def _install_recording_model(monkeypatch, r):
    """Replace the runner's model with one that records eval kwargs."""
    calls = []

    class _Recorder:
        def eval(self, img, **kwargs):
            calls.append({"shape": np.asarray(img).shape, **kwargs})
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 2:
                dp = np.ones((2, *arr.shape), dtype=np.float32)
                prob = np.full(arr.shape, 0.5, dtype=np.float32)
            else:
                dp = np.ones((3, *arr.shape), dtype=np.float32)
                prob = np.full(arr.shape, 0.5, dtype=np.float32)
            return None, (None, dp, prob), None

    monkeypatch.setattr(r, "_MODEL", _Recorder())
    return calls


def test_run_nucleus_frame_3d_passes_do_3d_and_anisotropy(monkeypatch):
    r = _runner()
    calls = _install_recording_model(monkeypatch, r)
    frame = np.zeros((4, 8, 8), dtype=np.float32)
    params = r.NucleusParams(do_3d=True, anisotropy=1.5, diameter=25.0, min_size=15, gamma=1.0)
    prob, dp = r.run_nucleus_frame(frame, z=None, params=params)
    assert prob.shape == (4, 8, 8)
    assert dp.shape == (3, 4, 8, 8)
    assert len(calls) == 1
    assert calls[0]["do_3D"] is True
    assert calls[0]["z_axis"] == 0
    assert calls[0]["anisotropy"] == 1.5
    assert calls[0]["diameter"] == 25.0
    assert calls[0]["min_size"] == 15


def test_run_nucleus_frame_2d_slices_z(monkeypatch):
    r = _runner()
    calls = _install_recording_model(monkeypatch, r)
    frame = np.zeros((4, 8, 8), dtype=np.float32)
    params = r.NucleusParams(do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0)
    prob, dp = r.run_nucleus_frame(frame, z=2, params=params)
    assert prob.shape == (8, 8)
    assert dp.shape == (2, 8, 8)
    assert len(calls) == 1
    assert calls[0]["shape"] == (8, 8)
    assert calls[0].get("do_3D", False) is False
    assert calls[0]["diameter"] is None


def test_run_cell_frame_runs_2d_slice(monkeypatch):
    r = _runner()
    calls = _install_recording_model(monkeypatch, r)
    frame = np.zeros((4, 8, 8), dtype=np.float32)
    params = r.CellParams(diameter=30.0, min_size=10, gamma=1.0)
    prob, dp = r.run_cell_frame(frame, z=1, params=params)
    assert prob.shape == (8, 8)
    assert dp.shape == (2, 8, 8)
    assert len(calls) == 1
    assert calls[0]["shape"] == (8, 8)
    assert calls[0]["diameter"] == 30.0
    assert calls[0]["min_size"] == 10


def test_run_nucleus_frame_applies_gamma(monkeypatch):
    r = _runner()
    received = {}

    class _Sniffer:
        def eval(self, img, **kwargs):
            received["img"] = np.asarray(img).copy()
            arr = np.asarray(img, dtype=np.float32)
            return None, (None, np.zeros((3, *arr.shape), dtype=np.float32), np.zeros(arr.shape, dtype=np.float32)), None

    monkeypatch.setattr(r, "_MODEL", _Sniffer())
    frame = np.linspace(0.0, 1.0, num=2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    params = r.NucleusParams(do_3d=True, anisotropy=1.0, diameter=0.0, min_size=0, gamma=2.0)
    r.run_nucleus_frame(frame, z=None, params=params)
    expected = r._apply_gamma(frame, 2.0)
    np.testing.assert_allclose(received["img"], expected, atol=1e-6)


def test_run_nucleus_stack_3d_iterates_frames_and_reports_progress(monkeypatch):
    r = _runner()
    _install_recording_model(monkeypatch, r)
    stack = np.zeros((3, 4, 6, 6), dtype=np.float32)
    params = r.NucleusParams(do_3d=True, anisotropy=1.5, diameter=0.0, min_size=0, gamma=1.0)
    progress = []
    prob, dp = r.run_nucleus_stack(
        stack, params,
        progress_cb=lambda d, t, msg: progress.append((d, t, msg)),
    )
    assert prob.shape == (3, 4, 6, 6)
    assert dp.shape == (3, 3, 4, 6, 6)
    assert progress[0] == (0, 3, "Nucleus: frame 1/3...")
    assert progress[-1] == (3, 3, "Nucleus: frame 3/3...")


def test_run_nucleus_stack_2d_returns_per_slice_outputs(monkeypatch):
    r = _runner()
    _install_recording_model(monkeypatch, r)
    stack = np.zeros((2, 3, 6, 6), dtype=np.float32)
    params = r.NucleusParams(do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0)
    prob, dp = r.run_nucleus_stack(stack, params)
    assert prob.shape == (2, 3, 6, 6)
    assert dp.shape == (2, 3, 2, 6, 6)


def test_run_nucleus_stack_2d_reports_current_z_slice(monkeypatch):
    r = _runner()
    _install_recording_model(monkeypatch, r)
    stack = np.zeros((2, 3, 6, 6), dtype=np.float32)
    params = r.NucleusParams(do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0)
    progress = []

    r.run_nucleus_stack(
        stack, params,
        progress_cb=lambda d, t, msg: progress.append((d, t, msg)),
    )

    assert "z 1/3" in progress[1][2]
    assert "z 2/3" in progress[2][2]
    assert "z 3/3" in progress[3][2]
    assert all("z " not in msg for _, _, msg in (progress[0], progress[-1]))


def test_run_cell_stack_iterates_t_then_z(monkeypatch):
    r = _runner()
    calls = _install_recording_model(monkeypatch, r)
    stack = np.zeros((2, 3, 6, 6), dtype=np.float32)
    params = r.CellParams(diameter=0.0, min_size=0, gamma=1.0)
    progress = []
    prob, dp = r.run_cell_stack(
        stack, params,
        progress_cb=lambda d, t, msg: progress.append((d, t, msg)),
    )
    assert prob.shape == (2, 3, 6, 6)
    assert dp.shape == (2, 3, 2, 6, 6)
    assert len(calls) == 2 * 3
    assert progress[0] == (0, 2, "Cell: frame 1/2...")


def test_run_nucleus_stack_respects_cancel_between_frames(monkeypatch):
    r = _runner()
    _install_recording_model(monkeypatch, r)
    stack = np.zeros((5, 2, 4, 4), dtype=np.float32)
    params = r.NucleusParams(do_3d=True, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0)
    seen_frames = []

    def _cancel():
        return len(seen_frames) >= 2

    def _record(done, total, msg):
        seen_frames.append(done)

    with pytest.raises(r.CancelledError):
        r.run_nucleus_stack(stack, params, progress_cb=_record, cancel_cb=_cancel)


def test_run_cell_stack_respects_cancel_between_frames(monkeypatch):
    r = _runner()
    _install_recording_model(monkeypatch, r)
    stack = np.zeros((5, 2, 4, 4), dtype=np.float32)
    params = r.CellParams(diameter=0.0, min_size=0, gamma=1.0)
    seen = []

    def _cancel():
        return len(seen) >= 1

    def _record(done, total, msg):
        seen.append(done)

    with pytest.raises(r.CancelledError):
        r.run_cell_stack(stack, params, progress_cb=_record, cancel_cb=_cancel)


def test_write_outputs_nucleus(tmp_path: Path):
    r = _runner()
    prob = np.random.rand(2, 3, 4, 5).astype(np.float32)
    dp = np.random.rand(2, 3, 3, 4, 5).astype(np.float32)
    r.write_outputs(prob, dp, tmp_path, "nucleus")
    prob_3dt = tmp_path / "nucleus_prob.tif"
    dp_3dt = tmp_path / "nucleus_dp.tif"
    assert prob_3dt.exists() and dp_3dt.exists()
    assert not (tmp_path / "nucleus_prob_zavg.tif").exists()
    written_prob = tifffile.imread(str(prob_3dt))
    np.testing.assert_allclose(written_prob, prob)


def test_write_outputs_cell(tmp_path: Path):
    r = _runner()
    prob = np.random.rand(2, 3, 4, 5).astype(np.float32)
    dp = np.random.rand(2, 3, 2, 4, 5).astype(np.float32)
    r.write_outputs(prob, dp, tmp_path, "cell")
    assert (tmp_path / "cell_prob.tif").exists()
    assert (tmp_path / "cell_dp.tif").exists()
    assert not (tmp_path / "cell_prob_zavg.tif").exists()


def test_write_outputs_creates_missing_dir(tmp_path: Path):
    r = _runner()
    target = tmp_path / "1_cellpose"
    prob = np.zeros((1, 2, 3, 3), dtype=np.float32)
    dp = np.zeros((1, 2, 3, 2, 3), dtype=np.float32)
    r.write_outputs(prob, dp, target, "cell")
    assert (target / "cell_prob.tif").exists()


def test_write_outputs_rejects_bad_channel(tmp_path: Path):
    r = _runner()
    prob = np.zeros((1, 1, 2, 2), dtype=np.float32)
    dp = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        r.write_outputs(prob, dp, tmp_path, "blah")  # type: ignore[arg-type]


# ── input-layout canonicalization ──────────────────────────────────────────────

def test_to_tzyx_normalizes_all_layouts():
    r = _runner()
    cases = {
        "2D": (5, 6),
        "2D+t": (3, 5, 6),
        "3D": (4, 5, 6),
        "3D+t": (2, 4, 5, 6),
    }
    for layout, shape in cases.items():
        out = r.to_tzyx(np.zeros(shape, dtype=np.float32), layout)
        assert out.ndim == 4, layout
        assert out.shape[-2:] == (5, 6), layout


def test_to_tzyx_places_singleton_axes_correctly():
    r = _runner()
    # 2D+t: time preserved, singleton Z inserted at axis 1.
    out = r.to_tzyx(np.zeros((3, 5, 6), dtype=np.float32), "2D+t")
    assert out.shape == (3, 1, 5, 6)
    # 3D: singleton T inserted at axis 0, z preserved.
    out = r.to_tzyx(np.zeros((4, 5, 6), dtype=np.float32), "3D")
    assert out.shape == (1, 4, 5, 6)
    # 2D: both singletons.
    out = r.to_tzyx(np.zeros((5, 6), dtype=np.float32), "2D")
    assert out.shape == (1, 1, 5, 6)


def test_to_tzyx_rejects_wrong_ndim_for_layout():
    r = _runner()
    with pytest.raises(ValueError):
        r.to_tzyx(np.zeros((5, 6), dtype=np.float32), "3D+t")
    with pytest.raises(ValueError):
        r.to_tzyx(np.zeros((2, 4, 5, 6), dtype=np.float32), "2D")
    with pytest.raises(ValueError):
        r.to_tzyx(np.zeros((5, 6), dtype=np.float32), "bogus")


def test_infer_layout_from_ndim():
    r = _runner()
    assert r.infer_layout_from_ndim(2) == "2D"
    assert r.infer_layout_from_ndim(4) == "3D+t"
    assert r.infer_layout_from_ndim(3) is None  # ambiguous: 2D+t vs 3D


def test_layout_has_z_and_time():
    r = _runner()
    assert r.layout_has_z("3D") and r.layout_has_z("3D+t")
    assert not r.layout_has_z("2D") and not r.layout_has_z("2D+t")
    assert r.layout_has_time("2D+t") and r.layout_has_time("3D+t")
    assert not r.layout_has_time("2D") and not r.layout_has_time("3D")
