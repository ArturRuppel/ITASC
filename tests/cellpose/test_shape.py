"""Tests for itasc.cellpose.shape — layout-free canonicalisation."""
from __future__ import annotations

import numpy as np

from itasc.cellpose.shape import describe_axes, to_canonical_tzyx


def test_2d_gains_singleton_t_and_z():
    out = to_canonical_tzyx(np.zeros((6, 8), dtype=np.int32))
    assert out.shape == (1, 1, 6, 8)


def test_3d_read_as_time():
    # one leading axis → time (the common 2D+t case); Z singleton.
    out = to_canonical_tzyx(np.zeros((10, 6, 8), dtype=np.int32))
    assert out.shape == (10, 1, 6, 8)


def test_4d_shorter_axis_is_z_already_ordered():
    # (T=40, Z=5, Y, X): leading axes already (long, short) → unchanged.
    out = to_canonical_tzyx(np.zeros((40, 5, 6, 8), dtype=np.int32))
    assert out.shape == (40, 5, 6, 8)


def test_4d_shorter_axis_is_z_swapped():
    # (5, 40, Y, X): axis0 is shorter → it is Z; reorder to (T=40, Z=5, Y, X).
    arr = np.arange(5 * 40 * 6 * 8).reshape(5, 40, 6, 8)
    out = to_canonical_tzyx(arr)
    assert out.shape == (40, 5, 6, 8)
    # content preserved under the swap.
    assert np.array_equal(out, np.swapaxes(arr, 0, 1))


def test_describe_axes_reports_inferred_t_z():
    assert describe_axes((6, 512, 512)) == "6×512×512 → T=6, Z=1"
    assert describe_axes((40, 5, 512, 512)) == "40×5×512×512 → T=40, Z=5"
    assert describe_axes((5, 40, 512, 512)) == "5×40×512×512 → T=40, Z=5"


def test_invalid_ndim_raises():
    import pytest

    with pytest.raises(ValueError):
        to_canonical_tzyx(np.zeros((2, 3, 4, 5, 6)))
