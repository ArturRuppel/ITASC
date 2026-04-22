from __future__ import annotations

import numpy as np

from cellflow.cellpose.stages.raw_import import _estimate_z_shift, _shift_volume


def _profile_fn(z: np.ndarray) -> np.ndarray:
    return (
        2.0 * np.exp(-0.5 * ((z - 5.0) / 1.2) ** 2)
        + 0.8 * np.exp(-0.5 * ((z - 11.0) / 0.9) ** 2)
        + 0.1 * z
    )


def test_estimate_z_shift_recovers_affine_offset():
    z = np.arange(16, dtype=np.float64)
    reference = _profile_fn(z)
    true_shift = 1.75
    target = 1.6 * _profile_fn(z - true_shift) + 0.35

    shift, scale, offset, mse = _estimate_z_shift(
        reference,
        target,
        max_shift_slices=4.0,
    )

    assert abs(shift - true_shift) < 0.15
    assert scale > 0.0
    assert mse < 1e-1
    assert np.isfinite(offset)


def test_shift_volume_moves_slices_linearly():
    volume = np.stack(
        [
            np.full((2, 2), 10, dtype=np.uint16),
            np.full((2, 2), 20, dtype=np.uint16),
            np.full((2, 2), 30, dtype=np.uint16),
            np.full((2, 2), 40, dtype=np.uint16),
        ],
        axis=0,
    )

    shifted = _shift_volume(volume, 1.0)

    assert shifted.shape == volume.shape
    assert shifted.dtype == np.uint16
    assert np.all(shifted[0] == 0)
    assert np.all(shifted[1] == 10)
    assert np.all(shifted[2] == 20)
