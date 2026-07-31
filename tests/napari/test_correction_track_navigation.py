"""Shift-arrow track stepping and the correction-mode gate-claim guard.

Both behaviours are pure control logic, so they are exercised by binding the
unbound methods to minimal stand-ins (no Qt widget / live viewer needed), the
same pattern as ``test_correction_selection_listener``.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np

from itasc.napari.correction.nucleus_correction_widget import NucleusCorrectionWidget
from itasc.napari.nucleus_workflow_widget import NucleusWorkflowWidget
from itasc.napari.correction._correction_navigation import ensure_cell_visible
from itasc.napari.correction._correction_keymap import HeldKeyRepeater, KEY_REPEAT_DELAY_MS
from itasc.napari.correction._correction_playback import (
    nav_repeat_interval_ms,
    playback_fps,
    playback_loops,
)


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


# ── Arrow-key frame stepping with no track selected (_step_film_frame) ──────


def _film_stub(*, has_strip, current_t=1, n_frames=5, loop_mode="once"):
    """Stand-in for ``_step_film_frame`` and its ``_step_viewer_frame`` fallback.

    ``has_strip`` is the lineage canvas's answer to "is a thumbnail band laid
    out?" — False whenever no track is selected, the case that used to leave the
    arrow keys (and the gamepad stick, which is just Left/Right) dead.
    """
    strip_calls: list = []
    dims = types.SimpleNamespace(
        current_step=(int(current_t), 0, 0), nsteps=(n_frames, 8, 8),
    )
    viewer = types.SimpleNamespace(
        dims=dims,
        window=types.SimpleNamespace(
            _qt_viewer=types.SimpleNamespace(
                dims=types.SimpleNamespace(
                    slider_widgets=[types.SimpleNamespace(loop_mode=loop_mode)]
                )
            )
        ),
    )
    obj = types.SimpleNamespace(
        viewer=viewer,
        _workspace_splitter=object(),  # focus mode is docked
        _lineage_canvas=types.SimpleNamespace(
            has_film_strip=lambda: has_strip,
            step_film_frame=lambda **kw: strip_calls.append(kw),
        ),
        _strip_calls=strip_calls,
    )
    # The fallback is the real method, run against this stub.
    obj._step_viewer_frame = lambda dx: NucleusCorrectionWidget._step_viewer_frame(
        obj, dx
    )
    return obj


def test_step_film_frame_walks_the_band_when_a_track_is_selected():
    obj = _film_stub(has_strip=True)
    NucleusCorrectionWidget._step_film_frame(obj, dx=1)
    # Delegated to the band; the frame slider is not touched directly.
    assert obj._strip_calls == [{"dx": 1, "dy": 0, "wrap": False}]
    assert obj.viewer.dims.current_step[0] == 1


def test_step_film_frame_without_a_selection_steps_the_frame_slider():
    obj = _film_stub(has_strip=False, current_t=1)
    NucleusCorrectionWidget._step_film_frame(obj, dx=1)
    assert obj._strip_calls == []
    assert obj.viewer.dims.current_step == (2, 0, 0)

    back = _film_stub(has_strip=False, current_t=1)
    NucleusCorrectionWidget._step_film_frame(back, dx=-1)
    assert back.viewer.dims.current_step == (0, 0, 0)


def test_frame_slider_fallback_stops_at_the_ends_unless_looping():
    # Loop mode "once" clamps: no step past either end.
    last = _film_stub(has_strip=False, current_t=4, n_frames=5, loop_mode="once")
    NucleusCorrectionWidget._step_film_frame(last, dx=1)
    assert last.viewer.dims.current_step[0] == 4

    first = _film_stub(has_strip=False, current_t=0, loop_mode="once")
    NucleusCorrectionWidget._step_film_frame(first, dx=-1)
    assert first.viewer.dims.current_step[0] == 0

    # With the viewer looping, both ends wrap — the same rule the band step uses.
    wrap_end = _film_stub(has_strip=False, current_t=4, n_frames=5, loop_mode="loop")
    NucleusCorrectionWidget._step_film_frame(wrap_end, dx=1)
    assert wrap_end.viewer.dims.current_step[0] == 0

    wrap_start = _film_stub(has_strip=False, current_t=0, n_frames=5, loop_mode="loop")
    NucleusCorrectionWidget._step_film_frame(wrap_start, dx=-1)
    assert wrap_start.viewer.dims.current_step[0] == 4


def test_up_down_stay_inert_without_a_film_strip():
    # A "row" is a property of the wrapped band; with no band there is nothing
    # for Up/Down to mean, so they must not silently become a frame step.
    obj = _film_stub(has_strip=False, current_t=1)
    NucleusCorrectionWidget._step_film_frame(obj, dy=1)
    assert obj.viewer.dims.current_step[0] == 1
    assert obj._strip_calls == []


def test_step_film_frame_does_nothing_outside_focus_mode():
    obj = _film_stub(has_strip=False, current_t=1)
    obj._workspace_splitter = None
    NucleusCorrectionWidget._step_film_frame(obj, dx=1)
    assert obj.viewer.dims.current_step[0] == 1


# ── Navigating onto a frame the track skips (_navigate_to_cell) ─────────────


def _navigate_stub(*, present_at):
    """Stand-in exposing what ``_navigate_to_cell`` reaches for.

    ``present_at`` maps frame -> set of present cell ids, so the helper can tell
    an occupied frame from an empty placeholder frame.
    """
    step = [0, 0, 0]
    dims = types.SimpleNamespace(current_step=tuple(step))
    viewer = types.SimpleNamespace(dims=dims)
    select_calls: list = []
    visible_calls: list = []
    obj = types.SimpleNamespace(
        viewer=viewer,
        correction_widget=types.SimpleNamespace(
            select_label=lambda t, cid: select_calls.append((t, cid))
        ),
        _current_cell_ids=lambda t: present_at.get(t, set()),
        _ensure_cell_visible=lambda t, cid: visible_calls.append((t, cid)),
        _navigating_from_lineage=False,
        _select_calls=select_calls,
        _visible_calls=visible_calls,
    )
    return obj


def test_navigate_to_present_frame_selects_and_checks_visibility():
    obj = _navigate_stub(present_at={2: {7}})
    NucleusCorrectionWidget._navigate_to_cell(obj, 2, 7, from_lineage=True)
    assert obj.viewer.dims.current_step[0] == 2
    assert obj._select_calls == [(2, 7)]
    assert obj._visible_calls == [(2, 7)]
    # Framing goes through _ensure_cell_visible and nowhere else: the stub viewer
    # has no ``camera``, so a pan/zoom written straight from _navigate_to_cell
    # would raise here.
    assert not hasattr(obj.viewer, "camera")


def test_navigate_to_empty_placeholder_frame_keeps_selection():
    # Track 7 is absent on frame 2 (an empty film-strip placeholder): jump to the
    # frame but do not re-select (which would clear the track) or pan.
    obj = _navigate_stub(present_at={0: {7}})
    NucleusCorrectionWidget._navigate_to_cell(obj, 2, 7, from_lineage=True)
    assert obj.viewer.dims.current_step[0] == 2
    assert obj._select_calls == []
    assert obj._visible_calls == []


# ── Off-screen-only panning (ensure_cell_visible) ───────────────────────────


def _framing_env(data, *, canvas, center=(0.0, 50.0, 50.0), zoom=4.0):
    """A ``(viewer, layer)`` pair for :func:`ensure_cell_visible`.

    ``data_to_world`` is identity (unit scale), so world coords == data coords.
    With the defaults the camera sits at (50, 50) and a 200x400 canvas at zoom 4
    shows y within +/-25 and x within +/-50 of that, shrunk by the 0.8 margin to
    +/-20 (y) and +/-40 (x).
    """
    layer = types.SimpleNamespace(
        data=data,
        data_to_world=lambda coord: np.asarray(coord, dtype=float),
    )
    camera = types.SimpleNamespace(center=center, zoom=zoom)
    canvas_obj = None if canvas is None else types.SimpleNamespace(size=canvas)
    viewer = types.SimpleNamespace(
        camera=camera,
        window=types.SimpleNamespace(
            _qt_viewer=types.SimpleNamespace(canvas=canvas_obj)
        ),
    )
    return viewer, layer


def _cell_at(y, x, *, cell_id=7, frame=0, shape=(3, 200, 200)):
    data = np.zeros(shape, dtype=int)
    data[frame, y, x] = cell_id
    return data


def test_on_screen_cell_does_not_move_the_camera():
    # (55, 60) is 5 off-centre in y and 10 in x — inside the 20/40 margins.
    viewer, layer = _framing_env(_cell_at(55, 60), canvas=(200, 400))
    ensure_cell_visible(viewer, layer, 0, 7)
    assert viewer.camera.center == (0.0, 50.0, 50.0)
    assert viewer.camera.zoom == 4.0


def test_off_screen_cell_pans_without_zooming():
    # (150, 60) is 100 off-centre in y, well outside the 20 margin.
    viewer, layer = _framing_env(_cell_at(150, 60), canvas=(200, 400))
    ensure_cell_visible(viewer, layer, 0, 7)
    assert viewer.camera.center == (0.0, 150.0, 60.0)
    assert viewer.camera.zoom == 4.0  # untouched — this is the whole point


def test_cell_just_inside_the_margin_stays_put():
    # y offset 20 is exactly the margin (0.8 * 200/4 / 2); x offset 40 likewise.
    viewer, layer = _framing_env(_cell_at(70, 90), canvas=(200, 400))
    ensure_cell_visible(viewer, layer, 0, 7)
    assert viewer.camera.center == (0.0, 50.0, 50.0)


def test_cell_in_the_outer_margin_is_treated_as_off_screen():
    # y offset 24 is on canvas (raw half-height 25) but inside the 0.8 margin,
    # so it is panned to rather than left hugging the edge.
    viewer, layer = _framing_env(_cell_at(74, 50), canvas=(200, 400))
    ensure_cell_visible(viewer, layer, 0, 7)
    assert viewer.camera.center == (0.0, 74.0, 50.0)


def test_visibility_uses_the_current_frame_not_the_whole_track():
    # Track 7 is on screen at frame 0 and far away at frame 2. Selecting it on
    # frame 0 must not pan just because the track wanders off screen later.
    data = _cell_at(55, 60)
    data[2, 190, 190] = 7
    viewer, layer = _framing_env(data, canvas=(200, 400))
    ensure_cell_visible(viewer, layer, 0, 7)
    assert viewer.camera.center == (0.0, 50.0, 50.0)
    ensure_cell_visible(viewer, layer, 2, 7)
    assert viewer.camera.center == (0.0, 190.0, 190.0)


def test_without_canvas_size_the_camera_is_left_alone():
    # Unknown viewport → assume visible rather than pan blindly.
    viewer, layer = _framing_env(_cell_at(150, 60), canvas=None)
    ensure_cell_visible(viewer, layer, 0, 7)
    assert viewer.camera.center == (0.0, 50.0, 50.0)
    assert viewer.camera.zoom == 4.0


def test_absent_cell_is_a_noop():
    viewer, layer = _framing_env(_cell_at(150, 60), canvas=(200, 400))
    ensure_cell_visible(viewer, layer, 0, 999)
    assert viewer.camera.center == (0.0, 50.0, 50.0)
    assert viewer.camera.zoom == 4.0


def test_pan_targets_the_mask_centroid():
    data = np.zeros((3, 200, 200), dtype=int)
    data[0, 150:161, 60:71] = 7  # centroid (155, 65)
    viewer, layer = _framing_env(data, canvas=(200, 400))
    ensure_cell_visible(viewer, layer, 0, 7)
    assert viewer.camera.center == (0.0, 155.0, 65.0)


# ── Space-bar movie play/stop (_toggle_movie_playback) ──────────────────────


def _playback_stub(*, is_playing):
    calls: list = []
    qt_dims = types.SimpleNamespace(
        is_playing=is_playing,
        play=lambda **kw: calls.append(("play", kw)),
        stop=lambda: calls.append(("stop", {})),
    )
    viewer = types.SimpleNamespace(
        window=types.SimpleNamespace(_qt_viewer=types.SimpleNamespace(dims=qt_dims))
    )
    return types.SimpleNamespace(viewer=viewer, _calls=calls)


def test_toggle_movie_starts_playback_on_axis_0_when_stopped():
    obj = _playback_stub(is_playing=False)
    NucleusCorrectionWidget._toggle_movie_playback(obj)
    assert obj._calls == [("play", {"axis": 0})]


def test_toggle_movie_stops_playback_when_playing():
    obj = _playback_stub(is_playing=True)
    NucleusCorrectionWidget._toggle_movie_playback(obj)
    assert obj._calls == [("stop", {})]


def test_stop_movie_is_a_noop_when_not_playing():
    obj = _playback_stub(is_playing=False)
    NucleusCorrectionWidget._stop_movie_playback(obj)
    assert obj._calls == []


# ── Held-key auto-repeat (_begin/_end/_tick + generator handler) ────────────


def _repeater(*, interval_ms=100):
    """A :class:`HeldKeyRepeater` with a mocked timer (no event loop needed).

    ``interval_provider`` is fixed (the viewer-fps lookup is tested separately),
    so the steady-repeat interval is deterministic.
    """
    return HeldKeyRepeater(MagicMock(), interval_provider=lambda: interval_ms)


def test_begin_key_repeat_fires_once_and_arms_the_timer():
    r = _repeater()
    action = MagicMock()
    r.begin("Up", action)
    action.assert_called_once_with()
    r._timer.start.assert_called_once()
    assert r._key == "Up"
    assert r._action is action


def test_begin_key_repeat_ignores_reentrant_same_key():
    # An OS auto-repeat re-press of a held key must not double-fire; the steady
    # timer stays the single source of repeats.
    r = _repeater()
    action = MagicMock()
    r.begin("Up", action)
    r.begin("Up", action)
    action.assert_called_once_with()
    assert r._timer.start.call_count == 1


def test_press_arms_the_initial_delay_not_a_steady_repeat():
    r = _repeater()
    action = MagicMock()
    r.begin("Up", action)
    # A press fires once and arms a single-shot for the initial hold delay — so a
    # quick tap released before it elapses fires exactly once (no double-trigger).
    action.assert_called_once_with()
    r._timer.setSingleShot.assert_called_once_with(True)
    r._timer.start.assert_called_once_with(KEY_REPEAT_DELAY_MS)


def test_first_repeat_tick_settles_into_the_steady_fps_interval():
    r = _repeater(interval_ms=100)
    r.begin("Up", MagicMock())
    r._timer.reset_mock()
    r._timer.isSingleShot.side_effect = [True, False]
    r._on_tick()  # initial delay elapsed → switch to steady repeat
    r._timer.setSingleShot.assert_called_once_with(False)
    r._timer.start.assert_called_once_with(100)  # interval_provider() → 100 ms
    r._on_tick()  # already steady → no second switch
    r._timer.setSingleShot.assert_called_once_with(False)


def test_repeat_tick_fires_the_armed_action():
    r = _repeater()
    action = MagicMock()
    r.begin("Up", action)
    r._on_tick()
    r._on_tick()
    assert action.call_count == 3  # 1 on press + 2 ticks


def test_end_key_repeat_only_stops_the_matching_key():
    r = _repeater()
    r.begin("Up", MagicMock())
    r.end("Down")  # mismatch → no stop
    r._timer.stop.assert_not_called()
    assert r._key == "Up"
    r.end("Up")
    r._timer.stop.assert_called_once()
    assert r._key is None
    assert r._action is None


def test_repeating_key_handler_press_then_release_cycle():
    r = _repeater()
    slot = MagicMock()
    handler = r.key_handler("Up", slot)
    gen = handler("LAYER")  # napari binds the layer as the first arg
    next(gen)               # press half: fire once + arm
    slot.assert_called_once_with()
    r._timer.start.assert_called_once()
    try:
        next(gen)           # release half: disarm
    except StopIteration:
        pass
    r._timer.stop.assert_called_once()
    assert r._key is None


# ── Matching the viewer's playback fps + loop mode ──────────────────────────


def _settings_viewer(*, slider_fps=None, slider_mode=None):
    """A viewer whose frame slider carries the given fps / loop mode.

    Pass ``None`` for either to drop the slider attribute, exercising the
    fall-through to napari's global preference.
    """
    slider = None
    if slider_fps is not None or slider_mode is not None:
        slider = types.SimpleNamespace()
        if slider_fps is not None:
            slider.fps = slider_fps
        if slider_mode is not None:
            slider.loop_mode = slider_mode
    sliders = [slider] if slider is not None else []
    return types.SimpleNamespace(
        window=types.SimpleNamespace(
            _qt_viewer=types.SimpleNamespace(
                dims=types.SimpleNamespace(slider_widgets=sliders)
            )
        )
    )


def test_playback_fps_reads_the_frame_slider():
    assert playback_fps(_settings_viewer(slider_fps=25.0)) == 25.0
    # Reverse playback (negative fps) still yields a positive repeat rate.
    assert playback_fps(_settings_viewer(slider_fps=-30.0)) == 30.0


def test_nav_repeat_interval_matches_fps():
    assert nav_repeat_interval_ms(_settings_viewer(slider_fps=25.0)) == 40
    assert nav_repeat_interval_ms(_settings_viewer(slider_fps=10.0)) == 100
    # A zero / bogus slider fps falls through to the default (10 fps → 100 ms).
    assert nav_repeat_interval_ms(_settings_viewer(slider_fps=0.0)) == 100


def test_playback_loops_follows_the_loop_mode():
    assert playback_loops(_settings_viewer(slider_mode="once")) is False
    assert playback_loops(_settings_viewer(slider_mode="loop")) is True
    assert playback_loops(_settings_viewer(slider_mode="back_and_forth")) is True
    # A LoopMode-like enum (``.value``) is honoured too.
    enum_like = types.SimpleNamespace(value="once")
    assert playback_loops(_settings_viewer(slider_mode=enum_like)) is False


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
