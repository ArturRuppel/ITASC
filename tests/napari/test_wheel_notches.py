"""Wheel events accumulated into detents.

Qt measures wheel travel in eighths of a degree; a mouse detent is 120 of them
in one event, while a high-resolution touchpad spreads the same travel over
dozens of small ones. Reading only the sign of ``angleDelta`` therefore fires a
full zoom step per touchpad event — which is what made Ctrl+wheel unusable on a
clickpad. These tests pin the accumulator that makes both devices agree.
"""

from itasc.napari._widget_helpers import WHEEL_NOTCH, WheelNotches


def test_one_mouse_detent_is_one_step():
    assert WheelNotches().steps(WHEEL_NOTCH) == 1


def test_direction_is_carried_by_the_sign():
    assert WheelNotches().steps(-WHEEL_NOTCH) == -1


def test_a_touchpad_swipe_costs_the_same_as_a_detent():
    # 30 small events summing to exactly one detent must yield exactly one
    # step, not thirty. This is the bug.
    n = WheelNotches()
    total = sum(n.steps(4) for _ in range(30))
    assert total == 1


def test_small_events_are_not_rounded_away():
    # Each event on its own is below the threshold; the remainder has to be
    # kept or slow travel would never zoom at all.
    n = WheelNotches()
    assert n.steps(50) == 0
    assert n.steps(50) == 0
    assert n.steps(50) == 1


def test_a_fast_flick_can_carry_several_notches():
    assert WheelNotches().steps(3 * WHEEL_NOTCH) == 3


def test_the_remainder_survives_across_notches():
    n = WheelNotches()
    assert n.steps(180) == 1          # 1.5 detents -> one step, 60 banked
    assert n.steps(60) == 1           # the banked 60 completes the second


def test_reversing_direction_drops_banked_travel():
    # Without this, a shove one way followed by a nudge back would over-shoot
    # on the way home: the banked travel would count toward the new direction.
    n = WheelNotches()
    assert n.steps(110) == 0
    assert n.steps(-20) == 0
    assert n.steps(-100) == -1


def test_zero_deltas_are_inert():
    n = WheelNotches()
    assert n.steps(0) == 0
    assert n.steps(WHEEL_NOTCH) == 1


def test_instances_do_not_share_residue():
    a, b = WheelNotches(), WheelNotches()
    a.steps(100)
    assert b.steps(100) == 0
