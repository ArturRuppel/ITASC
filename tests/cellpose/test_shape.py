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


def test_4d_z_axis_override_bypasses_heuristic():
    # A genuine (T=3, Z=5) acquisition has fewer timepoints than z-slices, so the
    # shorter-axis-is-Z heuristic would wrongly transpose it to (T=5, Z=3).
    # Declaring z_axis=1 keeps the intended layout.
    arr = np.arange(3 * 5 * 6 * 8).reshape(3, 5, 6, 8)
    out = to_canonical_tzyx(arr, z_axis=1)
    assert out.shape == (3, 5, 6, 8)
    assert np.array_equal(out, arr)


def test_4d_z_axis_zero_swaps():
    arr = np.arange(3 * 5 * 6 * 8).reshape(3, 5, 6, 8)
    out = to_canonical_tzyx(arr, z_axis=0)
    assert out.shape == (5, 3, 6, 8)
    assert np.array_equal(out, np.swapaxes(arr, 0, 1))


def test_4d_invalid_z_axis_raises():
    import pytest

    with pytest.raises(ValueError):
        to_canonical_tzyx(np.zeros((3, 5, 6, 8)), z_axis=2)


def test_ambiguous_4d_leading_axes_warns(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="itasc.cellpose.shape"):
        to_canonical_tzyx(np.zeros((3, 5, 6, 8)))
    assert any("ambiguous leading axes" in rec.message for rec in caplog.records)


def test_unambiguous_4d_does_not_warn(caplog):
    import logging

    # A singleton Z axis is unambiguous — no guess is made, so no warning.
    with caplog.at_level(logging.WARNING, logger="itasc.cellpose.shape"):
        to_canonical_tzyx(np.zeros((40, 1, 6, 8)))
    assert not caplog.records


def test_describe_axes_reports_inferred_t_z():
    assert describe_axes((6, 512, 512)) == "6×512×512 → T=6, Z=1"
    assert describe_axes((40, 5, 512, 512)) == "40×5×512×512 → T=40, Z=5"
    assert describe_axes((5, 40, 512, 512)) == "5×40×512×512 → T=40, Z=5"


def test_invalid_ndim_raises():
    import pytest

    with pytest.raises(ValueError):
        to_canonical_tzyx(np.zeros((2, 3, 4, 5, 6)))
