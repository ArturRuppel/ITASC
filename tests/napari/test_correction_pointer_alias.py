"""Alt+left standing in for the right mouse button.

Right-drag is the gesture a touchpad handles worst, so the correction canvas
accepts Alt+left in its place. The aliasing is one pure function applied before
any branch reads the button, which is exactly what these tests pin: that every
right-button gesture inherits the alias, and that nothing else moves.
"""

import types

from itasc.napari.correction.correction_widget import (
    CorrectionWidget,
    alias_pointer_button,
)


def test_alt_left_becomes_the_right_button():
    assert alias_pointer_button(1, {"Alt"}) == (2, set())


def test_alt_is_consumed_so_the_remaining_modifiers_still_match_exactly():
    # The branches compare with `mods == {"Shift"}`, so a leftover "Alt" would
    # silently disable every gesture the alias is supposed to reach.
    assert alias_pointer_button(1, {"Alt", "Shift"}) == (2, {"Shift"})
    assert alias_pointer_button(1, {"Alt", "Control"}) == (2, {"Control"})


def test_left_button_without_alt_is_untouched():
    assert alias_pointer_button(1, set()) == (1, set())
    assert alias_pointer_button(1, {"Shift"}) == (1, {"Shift"})
    assert alias_pointer_button(1, {"Control", "Shift"}) == (1, {"Control", "Shift"})


def test_a_real_right_button_is_untouched():
    assert alias_pointer_button(2, {"Shift"}) == (2, {"Shift"})


def test_middle_button_keeps_alt():
    # Middle-click has no alias, so Alt is not its to consume — passing it
    # through leaves the (currently unmatched) combination unmatched rather
    # than turning it into a bare middle-click that spawns or erases a cell.
    assert alias_pointer_button(3, {"Alt"}) == (3, {"Alt"})


def test_the_caller_s_modifier_set_is_not_mutated():
    mods = {"Alt", "Shift"}
    alias_pointer_button(1, mods)
    assert mods == {"Alt", "Shift"}


# -- napari's own Alt gesture ------------------------------------------------
#
# The layer resizes its brush while Alt is held, which would fire on every
# aliased drag. Correction parks that callback while it owns the layer.


class BrushSizeOnMouseMove:
    """Stands in for napari's callback — matched by class name, so this is it."""


def _layer(*callbacks):
    return types.SimpleNamespace(mouse_move_callbacks=list(callbacks))


def test_the_brush_size_callback_is_parked_while_correction_owns_the_layer():
    other = object()
    brush = BrushSizeOnMouseMove()
    layer = _layer(other, brush)
    obj = types.SimpleNamespace(_layer=layer, _suspended_move_cbs=[])

    CorrectionWidget._suspend_brush_size_on_alt(obj, layer)
    assert layer.mouse_move_callbacks == [other]

    CorrectionWidget._restore_brush_size_on_alt(obj)
    assert layer.mouse_move_callbacks == [other, brush]
    assert obj._suspended_move_cbs == []


def test_unrelated_move_callbacks_are_left_alone():
    other = object()
    layer = _layer(other)
    obj = types.SimpleNamespace(_layer=layer, _suspended_move_cbs=[])

    CorrectionWidget._suspend_brush_size_on_alt(obj, layer)
    CorrectionWidget._restore_brush_size_on_alt(obj)
    assert layer.mouse_move_callbacks == [other]


def test_restoring_twice_does_not_duplicate_the_callback():
    # _remove_callbacks runs on every deactivate, and a layer that was never
    # suspended (or was already restored) must not grow a second copy.
    brush = BrushSizeOnMouseMove()
    layer = _layer(brush)
    obj = types.SimpleNamespace(_layer=layer, _suspended_move_cbs=[])

    CorrectionWidget._suspend_brush_size_on_alt(obj, layer)
    CorrectionWidget._restore_brush_size_on_alt(obj)
    CorrectionWidget._restore_brush_size_on_alt(obj)
    assert layer.mouse_move_callbacks == [brush]


def test_a_layer_without_move_callbacks_is_survivable():
    obj = types.SimpleNamespace(_layer=None, _suspended_move_cbs=[])
    CorrectionWidget._suspend_brush_size_on_alt(obj, object())
    CorrectionWidget._restore_brush_size_on_alt(obj)
    assert obj._suspended_move_cbs == []
