"""Tests for itasc.segmentation.nucleus_segmentation."""
from __future__ import annotations

import numpy as np

from itasc.segmentation.nucleus_segmentation import _fill_and_close_labels


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
