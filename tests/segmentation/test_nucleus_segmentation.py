"""Tests for cellflow.segmentation.nucleus_segmentation."""
from __future__ import annotations

import numpy as np

from cellflow.segmentation.nucleus_segmentation import (
    ContourWatershedParams,
    _fill_and_close_labels,
    compute_contour_watershed,
)


def test_fill_and_close_labels_fills_per_label_holes():
    """Exercises the per-label loop (incl. the emptiness guard) and fills an
    interior hole without touching background or other labels."""
    labels = np.zeros((10, 10), dtype=np.uint32)
    labels[1:5, 1:5] = 1
    labels[2, 2] = 0  # hole inside label 1
    labels[6:9, 6:9] = 2
    out = _fill_and_close_labels(labels)
    assert out[2, 2] == 1  # hole filled
    assert out[0, 0] == 0  # background untouched
    assert np.array_equal(out == 2, labels == 2)  # other label unchanged


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
