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


def _make_inward_flow(H: int, W: int, cy: float, cx: float) -> np.ndarray:
    """Flow vectors pointing each pixel toward (cy, cx), unit magnitude."""
    yi, xi = np.indices((H, W))
    dy = (cy - yi).astype(np.float32)
    dx = (cx - xi).astype(np.float32)
    norm = np.hypot(dy, dx)
    safe = np.where(norm > 0, norm, 1.0)
    return np.stack([dy / safe, dx / safe], axis=0).astype(np.float32)


def test_compute_flow_following_movie_assigns_foreground_pixels_to_nucleus():
    from cellflow.segmentation.flow_following import compute_flow_following_movie

    T, H, W = 2, 24, 24
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[0, 12, 12] = 7
    labels[1, 12, 12] = 7

    flow_t0 = _make_inward_flow(H, W, 12.0, 12.0)
    flow_t1 = _make_inward_flow(H, W, 12.0, 12.0)
    dp = np.stack([flow_t0, flow_t1], axis=0).astype(np.float32)

    params = FlowFollowingParams(
        median_kernel_time=1,
        median_kernel_space=1,
        gaussian_sigma_time=0.0,
        gaussian_sigma_space=0.0,
        flow_weight=0.5,
        flow_step_scale=0.5,
        max_iterations=100,
        capture_radius=3.0,
    )

    filtered_dp, cell_labels = compute_flow_following_movie(
        foreground, dp, labels, params
    )

    assert filtered_dp.shape == dp.shape
    assert filtered_dp.dtype == np.float32
    assert cell_labels.shape == (T, H, W)
    assert cell_labels.dtype == np.int32
    # Every foreground pixel collapses onto the single nucleus, so all are label 7.
    assert (cell_labels == 7).all()


def test_compute_flow_following_movie_voronoi_fills_zero_flow_foreground():
    from cellflow.segmentation.flow_following import compute_flow_following_movie

    T, H, W = 1, 12, 12
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[0, 2, 2] = 4
    labels[0, 9, 9] = 8

    # Zero flow → integrator never converges; Voronoi must fill.
    dp = np.zeros((T, 2, H, W), dtype=np.float32)

    params = FlowFollowingParams(
        median_kernel_time=1, median_kernel_space=1,
        gaussian_sigma_time=0.0, gaussian_sigma_space=0.0,
        flow_weight=1.0,        # ignore gravity → integrator will not move
        flow_step_scale=0.2,
        max_iterations=10,
        capture_radius=0.5,
    )

    _, cell_labels = compute_flow_following_movie(foreground, dp, labels, params)

    # Every foreground pixel must end up labelled, partitioned by EDT distance.
    assert (cell_labels[0] > 0).all()
    assert cell_labels[0, 0, 0] == 4   # closer to seed 4
    assert cell_labels[0, 11, 11] == 8 # closer to seed 8


def test_compute_flow_following_movie_returns_zeros_for_empty_foreground_frame():
    from cellflow.segmentation.flow_following import compute_flow_following_movie

    T, H, W = 2, 8, 8
    foreground = np.ones((T, H, W), dtype=bool)
    foreground[1] = False                          # second frame is empty
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[0, 4, 4] = 1
    dp = np.zeros((T, 2, H, W), dtype=np.float32)

    _, cell_labels = compute_flow_following_movie(
        foreground, dp, labels, FlowFollowingParams(
            median_kernel_time=1, median_kernel_space=1,
        ),
    )
    assert (cell_labels[1] == 0).all()


def test_compute_flow_following_movie_returns_zeros_for_no_nuclei_frame():
    from cellflow.segmentation.flow_following import compute_flow_following_movie

    T, H, W = 1, 8, 8
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)   # no nuclei in t=0
    dp = np.zeros((T, 2, H, W), dtype=np.float32)

    _, cell_labels = compute_flow_following_movie(
        foreground, dp, labels, FlowFollowingParams(
            median_kernel_time=1, median_kernel_space=1,
        ),
    )
    assert (cell_labels[0] == 0).all()


def test_compute_flow_following_movie_applies_median_and_gaussian_filters(monkeypatch):
    from cellflow.segmentation import flow_following as ff

    T, H, W = 2, 6, 6
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[0, 3, 3] = 1
    labels[1, 3, 3] = 1
    dp = np.ones((T, 2, H, W), dtype=np.float32)

    median_calls: list[tuple] = []
    gauss_calls: list[tuple] = []

    real_median = ff.median_filter
    real_gauss = ff.gaussian_filter

    def spy_median(arr, size):
        median_calls.append(tuple(size))
        return real_median(arr, size=size)

    def spy_gauss(arr, sigma):
        gauss_calls.append(tuple(sigma))
        return real_gauss(arr, sigma=sigma)

    monkeypatch.setattr(ff, "median_filter", spy_median)
    monkeypatch.setattr(ff, "gaussian_filter", spy_gauss)

    params = FlowFollowingParams(
        median_kernel_time=3,
        median_kernel_space=3,
        gaussian_sigma_time=1.0,
        gaussian_sigma_space=1.0,
    )
    ff.compute_flow_following_movie(foreground, dp, labels, params)

    # Channel axis is left at size 1 so the filter operates only on (T, Y, X).
    assert median_calls == [(1, 3, 3, 3)]
    assert gauss_calls == [(0, 1.0, 1.0, 1.0)]


def test_compute_flow_following_movie_skips_filter_when_kernels_off(monkeypatch):
    from cellflow.segmentation import flow_following as ff

    T, H, W = 1, 6, 6
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[0, 3, 3] = 1
    dp = np.zeros((T, 2, H, W), dtype=np.float32)

    median_calls: list[tuple] = []
    gauss_calls: list[tuple] = []
    monkeypatch.setattr(ff, "median_filter",
                        lambda arr, size: (median_calls.append(size), arr)[1])
    monkeypatch.setattr(ff, "gaussian_filter",
                        lambda arr, sigma: (gauss_calls.append(sigma), arr)[1])

    ff.compute_flow_following_movie(
        foreground, dp, labels,
        FlowFollowingParams(
            median_kernel_time=1, median_kernel_space=1,
            gaussian_sigma_time=0.0, gaussian_sigma_space=0.0,
        ),
    )

    assert median_calls == []
    assert gauss_calls == []


def test_compute_flow_following_movie_progress_callback_invoked_per_frame():
    from cellflow.segmentation.flow_following import compute_flow_following_movie

    T, H, W = 3, 6, 6
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[:, 3, 3] = 1
    dp = np.zeros((T, 2, H, W), dtype=np.float32)

    calls: list[tuple[int, int]] = []
    compute_flow_following_movie(
        foreground, dp, labels,
        FlowFollowingParams(median_kernel_time=1, median_kernel_space=1),
        progress_cb=lambda done, total: calls.append((done, total)),
    )
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_flow_following_symbols_reexported_from_segmentation_package():
    from cellflow.segmentation import (
        FlowFollowingParams as PkgParams,
        compute_flow_following_movie as pkg_fn,
    )
    assert PkgParams().capture_radius == 3.0
    assert callable(pkg_fn)
