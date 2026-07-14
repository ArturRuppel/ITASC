"""The per-position status rail: one dot per pipeline stage.

Each :class:`StatusDot` renders one stage's :mod:`itasc.napari._stage_status`
state as a small coloured circle with a plain-language tooltip; :class:`StatusRail`
lays them out left→right and repaints them from a status dict on refresh. The
stage set is configurable: the full app passes the four-stage default
(Cellpose → Nucleus → Cell → Contacts); the standalone aggregate app passes the
three-stage :data:`~itasc.napari._stage_status.CONTACT_STAGES`
(cell labels → nucleus labels → contact analysis). Rendering only.
"""
from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget

from itasc.napari._stage_status import (
    DONE,
    MISSING,
    STAGE_CELL,
    STAGE_CELL_LABELS,
    STAGE_CELLPOSE,
    STAGE_CONTACTS,
    STAGE_NUCLEUS,
    STAGE_NUCLEUS_LABELS,
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
    STAGE_CELL_LABELS: "Cell labels",
    STAGE_NUCLEUS_LABELS: "Nucleus labels",
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
    """One stage's status, drawn as a small coloured circle.

    Clicking the dot emits :attr:`clicked` with the stage name — the catalog wires
    this to load that stage's output into the viewer.
    """

    clicked = Signal(str)  # stage name

    def __init__(self, stage: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.stage = stage
        self.state = UNKNOWN
        self.setFixedSize(_DOT_PX, _DOT_PX)
        self.setCursor(Qt.PointingHandCursor)
        self.set_state(UNKNOWN)

    def mousePressEvent(self, event) -> None:
        # Consume the event (do not fall through to row selection) and load.
        self.clicked.emit(self.stage)
        event.accept()

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
    """One :class:`StatusDot` per stage in *stages*, in the given order.

    Defaults to the four-stage pipeline :data:`STAGES`; pass an explicit stage
    list (e.g. :data:`CONTACT_STAGES`) for a different app's rail.
    :attr:`dotClicked` re-emits any dot's click with its stage name.
    """

    dotClicked = Signal(str)  # stage name

    def __init__(
        self,
        stages: Sequence[str] = STAGES,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.dots: list[StatusDot] = [StatusDot(stage) for stage in stages]
        for dot in self.dots:
            dot.clicked.connect(self.dotClicked)
            layout.addWidget(dot)
        layout.addStretch(1)

    def set_status(self, status: dict[str, str]) -> None:
        """Repaint every dot; stages absent from *status* render as ``unknown``."""
        for dot in self.dots:
            dot.set_state(status.get(dot.stage, UNKNOWN))
