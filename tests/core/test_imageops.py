"""Generic image-op primitives shared across CellFlow pieces."""
from __future__ import annotations

import numpy as np

from cellflow.core import imageops
from cellflow.core.imageops import residual


def test_residual_strength_zero_is_clipped_raw_map_without_gaussian(monkeypatch):
    """strength=0 (the default config) must return the non-negative raw map and
    must not pay for the local-mean gaussian."""
    rng = np.random.default_rng(0)
    frame = rng.normal(size=(32, 48)).astype(np.float32)

    def _boom(*args, **kwargs):  # pragma: no cover - fails the test if reached
        raise AssertionError("threshold_local should be skipped when strength=0")

    monkeypatch.setattr(imageops, "threshold_local", _boom)

    out = residual(frame, window=15, strength=0.0)

    np.testing.assert_array_equal(out, np.clip(frame, 0.0, None))
    assert out.dtype == np.float32


def test_residual_strength_zero_matches_full_path_result():
    """The guarded fast path is numerically identical to the general formula."""
    rng = np.random.default_rng(1)
    frame = rng.normal(size=(24, 24)).astype(np.float32)

    fast = residual(frame, window=11, strength=0.0)
    reference = np.clip(frame - 0.0, 0.0, None).astype(np.float32)

    np.testing.assert_array_equal(fast, reference)


def test_residual_nonzero_strength_still_subtracts_background():
    frame = np.zeros((16, 16), dtype=np.float32)
    frame[8, 8] = 10.0
    out = residual(frame, window=7, strength=1.0)
    # A lone bright pixel survives; flat background stays at zero.
    assert out[8, 8] > 0.0
    assert out[0, 0] == 0.0
