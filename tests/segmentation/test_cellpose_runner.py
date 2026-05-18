"""Tests for cellflow.segmentation.cellpose_runner.

The cellpose package is mocked at import time so these tests run without
torch/GPU.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


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
    monkeypatch.delitem(sys.modules, "cellflow.segmentation.cellpose_runner", raising=False)
    yield


def _runner():
    import importlib

    import cellflow.segmentation.cellpose_runner as runner
    importlib.reload(runner)
    return runner


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
