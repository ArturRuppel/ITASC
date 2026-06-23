"""Tests for cellflow.segmentation.nucleus_segmentation."""
from __future__ import annotations

import numpy as np

from cellflow.segmentation.nucleus_segmentation import (
    ContourWatershedParams,
    compute_contour_watershed,
)


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    # Two blobs separated by a faint ridge so noise can perturb the seeding.
    boundary = np.zeros((32, 32), dtype=np.float32)
    boundary[:, 15:17] = 0.6  # ridge between left/right halves
    fg = np.zeros((32, 32), dtype=np.uint8)
    fg[4:28, 4:28] = 1
    return boundary, fg


def test_contour_watershed_is_reproducible_for_same_run_index():
    """noise_scale>0 must be deterministic given run_index (seeded local RNG)."""
    boundary, fg = _inputs()
    params = ContourWatershedParams(noise_scale=0.2, noise_blur_sigma=1.0, run_index=7)
    a = compute_contour_watershed(boundary, fg, params)
    b = compute_contour_watershed(boundary, fg, params)
    np.testing.assert_array_equal(a, b)


def test_contour_watershed_seeds_rng_with_run_index(monkeypatch):
    """run_index must actually feed the noise RNG (not be dead): the local RNG
    is constructed from params.run_index. Guards against a hardcoded seed that
    would still satisfy the reproducibility test."""
    import cellflow.segmentation.nucleus_segmentation as ns

    seeds: list[int] = []
    real = np.random.default_rng

    def spy(seed=None):
        seeds.append(seed)
        return real(seed)

    monkeypatch.setattr(ns.np.random, "default_rng", spy)
    boundary, fg = _inputs()
    compute_contour_watershed(
        boundary, fg, ContourWatershedParams(noise_scale=0.2, run_index=42)
    )
    assert 42 in seeds


def test_contour_watershed_no_noise_is_deterministic():
    boundary, fg = _inputs()
    params = ContourWatershedParams(noise_scale=0.0)
    a = compute_contour_watershed(boundary, fg, params)
    b = compute_contour_watershed(boundary, fg, params)
    np.testing.assert_array_equal(a, b)
