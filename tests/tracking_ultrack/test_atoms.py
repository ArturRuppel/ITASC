# tests/tracking_ultrack/test_atoms.py
from __future__ import annotations

import numpy as np

from cellflow.tracking_ultrack.atoms import residual


def test_residual_is_zero_on_flat_input():
    frame = np.full((40, 40), 0.5, dtype=np.float32)
    out = residual(frame, window=11)
    assert out.shape == frame.shape
    assert out.dtype == np.float32
    assert np.allclose(out, 0.0, atol=1e-5)


def test_residual_is_nonnegative_and_peaks_on_local_bump():
    frame = np.zeros((40, 40), dtype=np.float32)
    frame[18:22, 18:22] = 1.0  # a bright patch on flat background
    out = residual(frame, window=11)
    assert out.min() >= 0.0
    assert out[20, 20] > 0.0


def test_residual_forces_odd_window():
    frame = np.random.default_rng(0).random((30, 30)).astype(np.float32)
    # even window must not raise and must equal the next odd window result
    assert np.allclose(residual(frame, window=10), residual(frame, window=11))
