"""The per-position status rail: four dots, one per pipeline stage.

Each :class:`StatusDot` renders one stage's :mod:`cellflow.napari._stage_status`
state as a small coloured circle with a plain-language tooltip; :class:`StatusRail`
lays the four out left→right (Cellpose → Nucleus → Cell → Contacts) and repaints
them from a status dict on refresh. Rendering only — the clickable "load this
stage" behaviour is added in a later step.
"""
from __future__ import annotations

from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget

from cellflow.napari._stage_status import (
    DONE,
    MISSING,
    STAGE_CELL,
    STAGE_CELLPOSE,
    STAGE_CONTACTS,
    STAGE_NUCLEUS,
    STAGES,
    STALE,
    UNKNOWN,
    WORKING,
)

#: Human-facing stage names for tooltips.
_STAGE_LABELS: dict[str, str] = {
    STAGE_CELLPOSE: "Cellpose",
    STAGE_NUCLEUS: "Nucleus tracking",
    STAGE_CELL: "Cell workflow",
    STAGE_CONTACTS: "Contact analysis",
}

#: Human-facing state names for tooltips.
_STATE_LABELS: dict[str, str] = {
    MISSING: "not started",
    WORKING: "working (uncommitted)",
    DONE: "done",
    STALE: "stale (re-run since commit)",
    UNKNOWN: "unknown",
}

#: state → (fill, border) colours. ``MISSING`` is a hollow ring; the rest fill.
#: Chosen to read on napari's dark theme.
_STATE_COLOURS: dict[str, tuple[str, str]] = {
    MISSING: ("transparent", "#7a7a7a"),
    WORKING: ("#e0a020", "#a8791a"),  # amber — working, not committed
    DONE: ("#3aa84a", "#2c7d38"),     # green — committed / present
    STALE: ("#d9534f", "#a83836"),    # red — committed but stale
    UNKNOWN: ("#5a5a5a", "#4a4a4a"),  # muted grey — no canonical root
}

_DOT_PX = 12


class StatusDot(QLabel):
    """One stage's status, drawn as a small coloured circle."""

    def __init__(self, stage: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.stage = stage
        self.state = UNKNOWN
        self.setFixedSize(_DOT_PX, _DOT_PX)
        self.set_state(UNKNOWN)

    def set_state(self, state: str) -> None:
        self.state = state
        fill, border = _STATE_COLOURS.get(state, _STATE_COLOURS[UNKNOWN])
        radius = _DOT_PX // 2
        self.setStyleSheet(
            f"background-color: {fill}; border: 1px solid {border}; "
            f"border-radius: {radius}px;"
        )
        stage_label = _STAGE_LABELS.get(self.stage, self.stage)
        state_label = _STATE_LABELS.get(state, state)
        self.setToolTip(f"{stage_label}: {state_label}")


class StatusRail(QWidget):
    """Four :class:`StatusDot`s, one per pipeline stage, in canonical order."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.dots: list[StatusDot] = [StatusDot(stage) for stage in STAGES]
        for dot in self.dots:
            layout.addWidget(dot)
        layout.addStretch(1)

    def set_status(self, status: dict[str, str]) -> None:
        """Repaint every dot; stages absent from *status* render as ``unknown``."""
        for dot in self.dots:
            dot.set_state(status.get(dot.stage, UNKNOWN))
