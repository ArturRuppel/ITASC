"""Domain events for the nucleus correction widget.

The widget emits one of these when something *happens* — a hand mask edit, a
track rebuild, a selection change — and the display collaborators subscribe to
repaint themselves. They already pull the current layer / selection / frame /
enabled-state through their provider callbacks (see how the controllers are
constructed), so an event only has to say *when* to refresh, not *what* changed.

This inverts the older shape, where every operation hand-called each controller's
refresh, scattering display coupling across the edit logic. With events, an
operation announces what it did and stays ignorant of who listens; the
subscriber map lives in one place (``NucleusCorrectionWidget._wire_events``).
"""

from __future__ import annotations

from qtpy.QtCore import QObject, Signal


class CorrectionEvents(QObject):
    """Signals announcing correction-domain changes (see module docstring)."""

    # A hand mask edit landed at one frame: ``(t, changed_ids)``.
    labels_edited = Signal(object, object)
    # The selected track's geometry changed across frames (extend / retrack):
    # repaint the live track views (comet, lineage canvas, candidate gallery).
    tracks_rebuilt = Signal()
    # Per-frame validation / anchor *flags* changed, but not the labels
    # themselves (validate / invalidate). Flag-only, so the lineage canvas
    # recolours (status refresh) rather than rebuilding the whole stack.
    validation_changed = Signal()
