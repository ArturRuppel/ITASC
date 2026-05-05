"""Tests for the flow-following cell segmentation backend."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

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


def test_flow_integrate_captures_foreground_pixel_into_seed_label():
    from cellflow.segmentation.flow_following import _flow_integrate

    H = W = 16
    nuclear_labels = np.zeros((H, W), dtype=np.int32)
    nuclear_labels[8, 8] = 5  # single seed at the centre
    prob_mask = np.ones((H, W), dtype=bool)

    # Pure-gravity setup: zero flow, gravity points each pixel toward the seed.
    flow = np.zeros((H, W, 2), dtype=np.float32)
    yi, xi = np.indices((H, W))
    dy = (8 - yi).astype(np.float32)
    dx = (8 - xi).astype(np.float32)
    norm = np.hypot(dy, dx)
    safe = np.where(norm > 0, norm, 1.0)
    grav_y = (dy / safe).astype(np.float32)
    grav_x = (dx / safe).astype(np.float32)
    grav_y[8, 8] = 0.0
    grav_x[8, 8] = 0.0

    dist = distance_transform_edt(nuclear_labels == 0).astype(np.float32)
    _, (ny, nx) = distance_transform_edt(nuclear_labels == 0, return_indices=True)

    result = _flow_integrate(
        nuclear_labels,
        flow, grav_y, grav_x,
        dist,
        ny.astype(np.int32), nx.astype(np.int32),
        prob_mask,
        n_steps=50,
        flow_step_scale=1.0,
        flow_weight=0.0,            # pure gravity
        capture_radius=1.5,
    )

    # Every foreground pixel should arrive at the seed and inherit label 5.
    assert (result == 5).all()


def test_flow_integrate_skips_pixels_outside_prob_mask():
    from cellflow.segmentation.flow_following import _flow_integrate

    H = W = 8
    nuclear_labels = np.zeros((H, W), dtype=np.int32)
    nuclear_labels[4, 4] = 3
    prob_mask = np.zeros((H, W), dtype=bool)
    prob_mask[4, 4] = True  # only the seed itself is foreground

    flow = np.zeros((H, W, 2), dtype=np.float32)
    grav_y = np.zeros((H, W), dtype=np.float32)
    grav_x = np.zeros((H, W), dtype=np.float32)
    dist = distance_transform_edt(nuclear_labels == 0).astype(np.float32)
    _, (ny, nx) = distance_transform_edt(nuclear_labels == 0, return_indices=True)

    result = _flow_integrate(
        nuclear_labels,
        flow, grav_y, grav_x,
        dist,
        ny.astype(np.int32), nx.astype(np.int32),
        prob_mask,
        n_steps=10,
        flow_step_scale=0.2,
        flow_weight=0.5,
        capture_radius=1.5,
    )

    assert result[4, 4] == 3
    # No assignments outside foreground.
    background = np.ones_like(result, dtype=bool)
    background[4, 4] = False
    assert (result[background] == 0).all()


def test_flow_integrate_preserves_existing_labels():
    from cellflow.segmentation.flow_following import _flow_integrate

    H = W = 6
    nuclear_labels = np.zeros((H, W), dtype=np.int32)
    nuclear_labels[2, 2] = 11
    nuclear_labels[3, 3] = 22
    prob_mask = np.ones((H, W), dtype=bool)

    flow = np.zeros((H, W, 2), dtype=np.float32)
    grav_y = np.zeros((H, W), dtype=np.float32)
    grav_x = np.zeros((H, W), dtype=np.float32)
    dist = distance_transform_edt(nuclear_labels == 0).astype(np.float32)
    _, (ny, nx) = distance_transform_edt(nuclear_labels == 0, return_indices=True)

    result = _flow_integrate(
        nuclear_labels,
        flow, grav_y, grav_x,
        dist,
        ny.astype(np.int32), nx.astype(np.int32),
        prob_mask,
        n_steps=5,
        flow_step_scale=0.2,
        flow_weight=0.5,
        capture_radius=1.5,
    )

    assert result[2, 2] == 11
    assert result[3, 3] == 22
