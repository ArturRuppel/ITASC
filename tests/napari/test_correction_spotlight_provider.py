"""Unit tests for the spotlight-mask-provider hook on CorrectionWidget.

The hook lets the correction widget widen the spotlight cutout to a whole
track's union (used by the "comet" track-path overlay). We exercise the pure
resolver logic without building the full Qt widget by binding the unbound method
to a minimal stand-in.
"""

from __future__ import annotations

import types

import numpy as np

from cellflow.napari.correction_widget import CorrectionWidget


def _default_mask():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    return mask


def test_no_provider_returns_default_mask():
    obj = types.SimpleNamespace(_spotlight_mask_provider=None)
    default = _default_mask()
    assert CorrectionWidget._resolve_spotlight_mask(obj, 0, 5, default) is default


def test_provider_override_is_used():
    union = np.ones((4, 4), dtype=bool)
    obj = types.SimpleNamespace(
        _spotlight_mask_provider=lambda t, lab, m: union
    )
    out = CorrectionWidget._resolve_spotlight_mask(obj, 0, 5, _default_mask())
    np.testing.assert_array_equal(out, union)


def test_none_override_falls_back_to_default():
    obj = types.SimpleNamespace(
        _spotlight_mask_provider=lambda t, lab, m: None
    )
    default = _default_mask()
    assert CorrectionWidget._resolve_spotlight_mask(obj, 0, 5, default) is default


def test_mismatched_shape_override_falls_back_to_default():
    obj = types.SimpleNamespace(
        _spotlight_mask_provider=lambda t, lab, m: np.ones((2, 2), dtype=bool)
    )
    default = _default_mask()
    assert CorrectionWidget._resolve_spotlight_mask(obj, 0, 5, default) is default


def test_empty_override_falls_back_to_default():
    obj = types.SimpleNamespace(
        _spotlight_mask_provider=lambda t, lab, m: np.zeros((4, 4), dtype=bool)
    )
    default = _default_mask()
    assert CorrectionWidget._resolve_spotlight_mask(obj, 0, 5, default) is default


def test_provider_exception_falls_back_to_default():
    def _boom(t, lab, m):
        raise RuntimeError("provider failed")

    obj = types.SimpleNamespace(_spotlight_mask_provider=_boom)
    default = _default_mask()
    assert CorrectionWidget._resolve_spotlight_mask(obj, 0, 5, default) is default
