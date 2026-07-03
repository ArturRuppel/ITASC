"""Unit tests for the spotlight-mask-provider hook on CorrectionWidget.

The hook lets the correction widget widen the spotlight cutout to a whole
track's union (used by the "comet" track-path overlay). We exercise the pure
resolver logic without building the full Qt widget by binding the unbound method
to a minimal stand-in.
"""

from __future__ import annotations

import types

import numpy as np

from cellflow.napari.correction.correction_widget import CorrectionWidget


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


def _highlight_stub(highlight_style, *, seg2d, captured):
    """Minimal stand-in exercising the mask-selection branch of
    ``_update_highlight`` without building the Qt widget."""
    return types.SimpleNamespace(
        _highlight_style=highlight_style,
        _selected_label=0,
        _selected_t=-1,
        _layer=object(),
        _goto_cell_id=types.SimpleNamespace(
            blockSignals=lambda v: False, setValue=lambda v: None
        ),
        viewer=types.SimpleNamespace(
            layers=types.SimpleNamespace(
                selection=types.SimpleNamespace(active=None)
            )
        ),
        _frame_view=lambda layer, t: seg2d,
        # widen to a different mask so we can tell whether it was consulted
        _resolve_spotlight_mask=lambda t, lab, m: np.ones_like(m),
        _update_spotlight=lambda mask: captured.append(mask),
        _notify_selection_changed=lambda t, lab, prev: None,
    )


def test_border_style_outlines_only_the_cell_ignoring_provider():
    seg2d = np.zeros((4, 4), dtype=int)
    seg2d[0, 0] = 5
    captured: list[np.ndarray] = []
    obj = _highlight_stub("border", seg2d=seg2d, captured=captured)
    CorrectionWidget._update_highlight(obj, 0, 5, notify=False)
    np.testing.assert_array_equal(captured[-1], seg2d == 5)


def test_spotlight_style_widens_to_provider_mask():
    seg2d = np.zeros((4, 4), dtype=int)
    seg2d[0, 0] = 5
    captured: list[np.ndarray] = []
    obj = _highlight_stub("spotlight", seg2d=seg2d, captured=captured)
    CorrectionWidget._update_highlight(obj, 0, 5, notify=False)
    np.testing.assert_array_equal(captured[-1], np.ones((4, 4), dtype=int))


def test_set_highlight_style_switches_and_rerenders():
    rerendered: list[tuple[int, int]] = []
    obj = types.SimpleNamespace(
        _highlight_style="spotlight",
        _selected_label=5,
        _selected_t=0,
        _layer=object(),
        _update_highlight=lambda t, lab, notify: rerendered.append((t, lab)),
    )
    CorrectionWidget.set_highlight_style(obj, "border")
    assert obj._highlight_style == "border"
    assert rerendered == [(0, 5)]


def test_set_highlight_style_noop_when_unchanged():
    rerendered: list = []
    obj = types.SimpleNamespace(
        _highlight_style="spotlight",
        _selected_label=5,
        _selected_t=0,
        _layer=object(),
        _update_highlight=lambda *a, **k: rerendered.append(a),
    )
    CorrectionWidget.set_highlight_style(obj, "spotlight")
    assert rerendered == []
