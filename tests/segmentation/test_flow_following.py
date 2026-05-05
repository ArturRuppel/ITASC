"""Tests for the flow-following cell segmentation backend."""
from __future__ import annotations

import numpy as np
import pytest

from cellflow.segmentation.flow_following import (
    FlowFollowingParams,
    _fill_foreground,
)


def test_flow_following_params_defaults_match_spec():
    p = FlowFollowingParams()
    assert p.median_kernel_time == 3
    assert p.median_kernel_space == 5
    assert p.gaussian_sigma_time == 0.0
    assert p.gaussian_sigma_space == 0.0
    assert p.flow_weight == 0.5
    assert p.flow_step_scale == 0.2
    assert p.max_iterations == 100
    assert p.capture_radius == 3.0


def test_fill_foreground_voronoi_assigns_unlabelled_foreground():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1, 1] = 7
    labels[4, 4] = 9
    fg = np.ones((6, 6), dtype=bool)

    out = _fill_foreground(labels, fg)

    # Closer to seed 7 (top-left) ⇒ label 7
    assert out[0, 0] == 7
    assert out[2, 2] == 7
    # Closer to seed 9 (bottom-right) ⇒ label 9
    assert out[5, 5] == 9
    assert out[3, 3] == 9
    # Original seeds preserved
    assert out[1, 1] == 7
    assert out[4, 4] == 9


def test_fill_foreground_skips_when_no_unlabelled_foreground():
    labels = np.array([[1, 1], [2, 2]], dtype=np.int32)
    fg = np.ones((2, 2), dtype=bool)

    out = _fill_foreground(labels, fg)
    np.testing.assert_array_equal(out, labels)


def test_fill_foreground_leaves_background_zero():
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0, 0] = 5
    fg = np.zeros((4, 4), dtype=bool)
    fg[0, 0] = True
    fg[0, 1] = True

    out = _fill_foreground(labels, fg)
    assert out[0, 0] == 5
    assert out[0, 1] == 5
    # Outside foreground stays zero
    assert out[2, 2] == 0
    assert out[3, 3] == 0
