"""Auto-repeat for held navigation keys in correction focus mode.

napari only auto-repeats *bare* arrows (not Shift+arrow), so the correction
widget drives a steady repeat itself: a press fires the action once and arms a
one-shot for an initial hold delay; if the key is still held when that elapses,
the timer settles into a steady repeat paced to the viewer's playback fps. The
matching release disarms it.

The state machine and its timer are isolated here so the widget only wires the
key bindings. The QTimer is injected (the widget passes ``QTimer(self)``) so the
logic can be unit-tested with a mock timer and no event loop.
"""

from __future__ import annotations

from collections.abc import Callable

KEY_REPEAT_DELAY_MS = 300  # hold threshold before repeating kicks in


class HeldKeyRepeater:
    """Drive a steady repeat for whichever navigation key is currently held.

    ``interval_provider`` returns the steady repeat interval (ms) at the moment
    repeating begins, so it can track the viewer's live playback fps.
    """

    def __init__(self, timer, interval_provider: Callable[[], int]) -> None:
        self._timer = timer
        self._timer.timeout.connect(self._on_tick)
        self._interval_provider = interval_provider
        self._key: str | None = None
        self._action: Callable[[], None] | None = None

    def begin(self, key: str, action: Callable[[], None]) -> None:
        # An OS auto-repeat re-press of the same held key would re-enter here;
        # ignore it so the steady timer stays the single source of repeats.
        if self._key == key:
            return
        self._key = key
        self._action = action
        action()
        # Arm a one-shot for the initial hold delay; a tap released before it
        # elapses never repeats. The first tick then settles into the steady
        # fps-paced repeat (see _on_tick).
        self._timer.setSingleShot(True)
        self._timer.start(KEY_REPEAT_DELAY_MS)

    def end(self, key: str) -> None:
        if self._key == key:
            self.stop()

    def stop(self) -> None:
        self._timer.stop()
        self._key = None
        self._action = None

    def _on_tick(self) -> None:
        if self._action is None:
            return
        self._action()
        if self._timer.isSingleShot():
            # The initial delay just elapsed; from here repeat steadily at the
            # viewer's playback fps until the key is released.
            self._timer.setSingleShot(False)
            self._timer.start(self._interval_provider())

    def key_handler(self, key: str, slot: Callable[[], None]):
        """A napari generator keybinding: press starts repeat, release stops it.

        napari runs the pre-``yield`` half on key press and the post-``yield``
        half on release (it stashes the paused generator per key). Between them
        the repeat timer drives ``slot`` at a steady rate.
        """
        def handler(_layer, _key=key, _slot=slot):
            self.begin(_key, _slot)
            yield
            self.end(_key)

        return handler
