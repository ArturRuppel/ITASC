from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cellflow.napari._stage_status import (
    CONTACT_STAGES,
    DONE,
    MISSING,
    STAGE_CELL,
    STAGE_CELL_LABELS,
    STAGE_CELLPOSE,
    STAGE_CONTACTS,
    STAGE_NUCLEUS,
    STAGE_NUCLEUS_LABELS,
    STALE,
    UNKNOWN,
    WORKING,
)
from cellflow.napari._status_rail import StatusRail


def _app():
    from napari.qt import get_qapp

    return get_qapp()


def test_rail_has_one_dot_per_stage_in_order():
    _app()
    rail = StatusRail()
    assert [dot.stage for dot in rail.dots] == [
        STAGE_CELLPOSE,
        STAGE_NUCLEUS,
        STAGE_CELL,
        STAGE_CONTACTS,
    ]


def test_set_status_updates_each_dot_state():
    _app()
    rail = StatusRail()
    rail.set_status(
        {
            STAGE_CELLPOSE: DONE,
            STAGE_NUCLEUS: WORKING,
            STAGE_CELL: STALE,
            STAGE_CONTACTS: MISSING,
        }
    )
    states = {dot.stage: dot.state for dot in rail.dots}
    assert states == {
        STAGE_CELLPOSE: DONE,
        STAGE_NUCLEUS: WORKING,
        STAGE_CELL: STALE,
        STAGE_CONTACTS: MISSING,
    }


def test_dot_tooltip_names_stage_and_state():
    _app()
    rail = StatusRail()
    rail.set_status({STAGE_NUCLEUS: WORKING})
    nucleus = next(dot for dot in rail.dots if dot.stage == STAGE_NUCLEUS)
    tip = nucleus.toolTip().lower()
    assert "nucleus" in tip
    assert "uncommitted" in tip or "working" in tip


def test_missing_keys_default_to_unknown():
    _app()
    rail = StatusRail()
    rail.set_status({})  # no stages supplied
    assert all(dot.state == UNKNOWN for dot in rail.dots)


def test_rail_with_contact_stages_has_three_dots_in_order():
    _app()
    rail = StatusRail(CONTACT_STAGES)
    assert [dot.stage for dot in rail.dots] == [
        STAGE_CELL_LABELS,
        STAGE_NUCLEUS_LABELS,
        STAGE_CONTACTS,
    ]


def test_contact_rail_labels_the_input_stages_in_tooltips():
    _app()
    rail = StatusRail(CONTACT_STAGES)
    rail.set_status({STAGE_CELL_LABELS: DONE, STAGE_NUCLEUS_LABELS: MISSING})
    labels = {dot.stage: dot.toolTip().lower() for dot in rail.dots}
    assert "cell labels" in labels[STAGE_CELL_LABELS]
    assert "nucleus labels" in labels[STAGE_NUCLEUS_LABELS]


def test_clicking_a_dot_emits_its_stage():
    from qtpy.QtCore import QEvent, QPointF, Qt
    from qtpy.QtGui import QMouseEvent

    _app()
    rail = StatusRail()
    seen: list[str] = []
    rail.dotClicked.connect(seen.append)
    nucleus = next(dot for dot in rail.dots if dot.stage == STAGE_NUCLEUS)
    event = QMouseEvent(
        QEvent.MouseButtonPress,
        QPointF(1, 1),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    nucleus.mousePressEvent(event)
    assert seen == [STAGE_NUCLEUS]
