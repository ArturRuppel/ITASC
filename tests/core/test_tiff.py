from __future__ import annotations

import warnings

import numpy as np

from cellflow.core.tiff import imwrite_grayscale


def test_imwrite_grayscale_avoids_rgb_shape_warning(tmp_path) -> None:
    stack = np.zeros((2, 4, 4), dtype=np.uint32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        imwrite_grayscale(tmp_path / "labels.tif", stack, compression=None)

    messages = [str(item.message) for item in caught]
    assert not any("stored as RGB" in message for message in messages)
