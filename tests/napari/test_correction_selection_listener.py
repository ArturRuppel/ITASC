"""The additive selection-listener registry on CorrectionWidget.

The workflow widget owns the single ``set_selection_callback`` slot, so the
track-path comet registers via ``add_selection_listener`` instead. These tests
exercise the pure notify/registration logic by binding the unbound methods to a
minimal stand-in (no Qt widget needed).
"""

from __future__ import annotations

import types

from cellflow.napari.correction_widget import CorrectionWidget


def _stub(callback=None, listeners=None):
    return types.SimpleNamespace(
        _selection_callback=callback,
        _selection_listeners=list(listeners or []),
    )


def test_listener_fires_alongside_single_slot_callback():
    seen = []
    obj = _stub(
        callback=lambda t, lab: seen.append(("cb", t, lab)),
        listeners=[lambda t, lab: seen.append(("listener", t, lab))],
    )
    CorrectionWidget._notify_selection_changed(obj, 3, 7, previous_label=0)
    assert seen == [("cb", 3, 7), ("listener", 3, 7)]


def test_listener_fires_even_without_a_single_slot_callback():
    # This is the comet's case: the workflow widget may set the slot to None.
    seen = []
    obj = _stub(callback=None, listeners=[lambda t, lab: seen.append((t, lab))])
    CorrectionWidget._notify_selection_changed(obj, 1, 9, previous_label=2)
    assert seen == [(1, 9)]


def test_no_notification_when_label_unchanged():
    seen = []
    obj = _stub(listeners=[lambda t, lab: seen.append((t, lab))])
    CorrectionWidget._notify_selection_changed(obj, 1, 5, previous_label=5)
    assert seen == []


def test_one_listener_raising_does_not_block_the_others():
    seen = []

    def boom(t, lab):
        raise RuntimeError("listener failed")

    obj = _stub(listeners=[boom, lambda t, lab: seen.append((t, lab))])
    CorrectionWidget._notify_selection_changed(obj, 0, 4, previous_label=0)
    assert seen == [(0, 4)]


def test_add_selection_listener_dedupes():
    obj = _stub()
    fn = lambda t, lab: None  # noqa: E731
    CorrectionWidget.add_selection_listener(obj, fn)
    CorrectionWidget.add_selection_listener(obj, fn)
    assert obj._selection_listeners == [fn]
