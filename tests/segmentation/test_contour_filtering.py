from __future__ import annotations

import numpy as np


def test_compute_filtered_contour_maps_applies_median_and_gaussian(monkeypatch):
    from cellflow.segmentation import contour_filtering as cf
    from cellflow.segmentation import ContourFilterParams

    contours = np.ones((2, 5, 5), dtype=np.float32)
    median_calls: list[tuple[int, ...]] = []
    gauss_calls: list[tuple[float, ...]] = []

    def spy_median(arr, size):
        median_calls.append(tuple(size))
        return arr

    def spy_gauss(arr, sigma):
        gauss_calls.append(tuple(sigma))
        return arr

    monkeypatch.setattr(cf, "median_filter", spy_median)
    monkeypatch.setattr(cf, "gaussian_filter", spy_gauss)

    filtered = cf.compute_filtered_contour_maps(
        contours,
        ContourFilterParams(
            median_kernel_time=3,
            median_kernel_space=5,
            gaussian_sigma_time=1.0,
            gaussian_sigma_space=2.0,
        ),
    )

    assert filtered.dtype == np.float32
    assert median_calls == [(3, 5, 5)]
    assert gauss_calls == [(1.0, 2.0, 2.0)]


def test_compute_filtered_contour_maps_skips_noop_filters(monkeypatch):
    from cellflow.segmentation import contour_filtering as cf
    from cellflow.segmentation import ContourFilterParams

    contours = np.arange(9, dtype=np.float32).reshape(3, 3)
    median_calls: list[tuple[int, ...]] = []
    gauss_calls: list[tuple[float, ...]] = []

    monkeypatch.setattr(
        cf,
        "median_filter",
        lambda arr, size: (median_calls.append(tuple(size)), arr)[1],
    )
    monkeypatch.setattr(
        cf,
        "gaussian_filter",
        lambda arr, sigma: (gauss_calls.append(tuple(sigma)), arr)[1],
    )

    filtered = cf.compute_filtered_contour_maps(contours, ContourFilterParams())

    np.testing.assert_array_equal(filtered, contours)
    assert filtered.dtype == np.float32
    assert median_calls == []
    assert gauss_calls == []
