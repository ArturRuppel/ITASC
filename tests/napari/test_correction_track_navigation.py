"""Shift-arrow track stepping and the correction-mode gate-claim guard.

Both behaviours are pure control logic, so they are exercised by binding the
unbound methods to minimal stand-ins (no Qt widget / live viewer needed), the
same pattern as ``test_correction_selection_listener``.
"""

from __future__ import annotations

import types

import numpy as np

from cellflow.napari.nucleus_correction_widget import NucleusCorrectionWidget
from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget


# ── Shift-arrow track stepping (_step_track) ────────────────────────────────


def _track_stub(data, *, selected_label, current_t):
    """A stand-in exposing just what ``_step_track`` reaches for."""
    layer = types.SimpleNamespace(data=data)
    cell_ids_by_t = {
        t: set(int(v) for v in np.unique(data[t])) - {0}
        for t in range(data.shape[0])
    }
    nav_calls: list = []
    status: list = []
    return types.SimpleNamespace(
        correction_widget=types.SimpleNamespace(_selected_label=selected_label),
        _correction_tracked_layer=lambda: layer,
        _current_t=lambda: current_t,
        _current_cell_ids=lambda t: cell_ids_by_t.get(t, set()),
        _navigate_to_cell=lambda t, cell_id, *, from_lineage: nav_calls.append(
            (t, cell_id, from_lineage)
        ),
        _correction_status=lambda msg: status.append(msg),
        _nav_calls=nav_calls,
        _status=status,
    )


def _three_frame_stack():
    # Global track list across the stack is [1, 2, 3, 5]; no single frame holds
    # all of them, so stepping must walk the global list, not a frame scan.
    data = np.zeros((3, 4, 4), dtype=int)
    data[0, 0, 0] = 1
    data[0, 1, 1] = 2
    data[1, 0, 0] = 2
    data[1, 1, 1] = 3
    data[2, 0, 0] = 3
    data[2, 1, 1] = 5
    return data


def test_step_track_forward_moves_to_next_id_in_global_list():
    obj = _track_stub(_three_frame_stack(), selected_label=2, current_t=0)
    NucleusCorrectionWidget._step_track(obj, 1)
    # Next after 2 in [1, 2, 3, 5] is 3; absent from frame 0, so jump to the
    # first frame that contains it (frame 1) and recenter (from_lineage=False).
    assert obj._nav_calls == [(1, 3, False)]


def test_step_track_backward_moves_to_previous_id():
    obj = _track_stub(_three_frame_stack(), selected_label=3, current_t=1)
    NucleusCorrectionWidget._step_track(obj, -1)
    # Previous before 3 is 2, present in the current frame → stay on frame 1.
    assert obj._nav_calls == [(1, 2, False)]


def test_step_track_wraps_around_the_list():
    obj = _track_stub(_three_frame_stack(), selected_label=5, current_t=2)
    NucleusCorrectionWidget._step_track(obj, 1)
    # 5 is last → wraps to 1 (first frame with 1 is frame 0).
    assert obj._nav_calls == [(0, 1, False)]


def test_step_track_with_no_selection_starts_at_an_end():
    fwd = _track_stub(_three_frame_stack(), selected_label=0, current_t=0)
    NucleusCorrectionWidget._step_track(fwd, 1)
    assert fwd._nav_calls == [(0, 1, False)]  # first id

    back = _track_stub(_three_frame_stack(), selected_label=0, current_t=0)
    NucleusCorrectionWidget._step_track(back, -1)
    assert back._nav_calls == [(2, 5, False)]  # last id, on its frame


def test_step_track_on_empty_stack_does_not_navigate():
    obj = _track_stub(np.zeros((2, 4, 4), dtype=int), selected_label=0, current_t=0)
    NucleusCorrectionWidget._step_track(obj, 1)
    assert obj._nav_calls == []
    assert obj._status  # reported "no cells"


# ── Gate-claim guard on activation bail-out ─────────────────────────────────


def _gate_stub(events):
    return types.SimpleNamespace(
        claim_viewer=lambda tok: events.append(("claim", tok)),
        release_viewer=lambda tok: events.append(("release", tok)),
    )


def test_guarded_toggle_releases_when_activation_bails():
    # Activation with no tracked data reverts the button to unchecked from
    # inside the handler; the gate must follow the *resulting* state and not be
    # left owning the viewer (which froze the workflow behind the banner).
    events: list = []
    obj = types.SimpleNamespace(
        _on_correction_active_button_toggled=lambda checked: events.append(
            ("inner", checked)
        ),
        correction_active_btn=types.SimpleNamespace(isChecked=lambda: False),
        gate=_gate_stub(events),
    )
    NucleusWorkflowWidget._on_guarded_correction_active_button_toggled(obj, True)
    assert events == [("inner", True), ("release", "correction:nucleus")]


def test_guarded_toggle_claims_when_activation_sticks():
    events: list = []
    obj = types.SimpleNamespace(
        _on_correction_active_button_toggled=lambda checked: events.append(
            ("inner", checked)
        ),
        correction_active_btn=types.SimpleNamespace(isChecked=lambda: True),
        gate=_gate_stub(events),
    )
    NucleusWorkflowWidget._on_guarded_correction_active_button_toggled(obj, True)
    assert events == [("inner", True), ("claim", "correction:nucleus")]
