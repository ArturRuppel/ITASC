"""Nucleus segmentation helpers: cancellation and per-label hole filling."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from itasc.core.cancellation import CancelledError

__all__ = ["CancelledError"]


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise CancelledError("Operation cancelled.")


def _fill_and_close_labels(labels: np.ndarray) -> np.ndarray:
    """Fill interior holes per label."""
    from scipy.ndimage import binary_fill_holes

    out = np.zeros_like(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        coords = np.nonzero(labels == label_id)
        # np.nonzero always returns a tuple of ndim arrays (truthy), so the only
        # real emptiness signal is a zero-length coordinate array.
        if coords[0].size == 0:
            continue
        slices = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coords)
        filled = binary_fill_holes(labels[slices] == label_id)
        out_view = out[slices]
        out_view[filled] = label_id
    return out
