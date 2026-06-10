"""Read the napari viewer's frame-playback settings (fps + loop mode).

These are pure queries over the viewer's frame (axis-0) slider — the widget that
the right-click play-button popup writes to — falling back to napari's global
preferences when the slider is absent. They carry no widget state, so they live
here as free functions rather than on the correction widget; the held-key
auto-repeat (:mod:`cellflow.napari._correction_keymap`) paces itself from
:func:`nav_repeat_interval_ms`, and the film-strip stepping reads
:func:`playback_loops` to decide whether to wrap at the ends.
"""

from __future__ import annotations

DEFAULT_NAV_FPS = 10.0  # napari's default when no slider/preference exists


def frame_slider_widget(viewer):
    """The frame (axis-0) playback slider widget, or ``None`` if absent.

    It carries the viewer's live fps / loop mode (the right-click play-button
    popup writes here, defaulting from the global preference).
    """
    try:
        sliders = viewer.window._qt_viewer.dims.slider_widgets
        return sliders[0] if sliders else None
    except Exception:
        return None


def playback_fps(viewer) -> float:
    """The viewer's playback fps — the slider's, else the global preference."""
    slider = frame_slider_widget(viewer)
    if slider is not None:
        try:
            fps = abs(float(slider.fps))
            if fps > 0:
                return fps
        except Exception:
            pass
    try:
        from napari.settings import get_settings

        return float(get_settings().application.playback_fps)
    except Exception:
        return DEFAULT_NAV_FPS


def playback_loops(viewer) -> bool:
    """True when the viewer's playback loop mode wraps at the ends.

    ``once`` stops at an end; ``loop`` / ``back_and_forth`` both wrap around.
    """
    slider = frame_slider_widget(viewer)
    mode = getattr(slider, "loop_mode", None) if slider is not None else None
    if mode is None:
        try:
            from napari.settings import get_settings

            mode = get_settings().application.playback_mode
        except Exception:
            return True
    return str(getattr(mode, "value", mode)).lower() != "once"


def nav_repeat_interval_ms(viewer) -> int:
    """Held-key repeat interval (ms) matching the viewer's playback fps."""
    fps = max(playback_fps(viewer), 1.0)
    return max(int(round(1000.0 / fps)), 1)
