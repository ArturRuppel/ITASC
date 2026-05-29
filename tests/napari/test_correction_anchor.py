from __future__ import annotations

from cellflow.napari._correction_anchor import (
    anchor_correction,
    without_anchor_correction,
)
from cellflow.tracking_ultrack.corrections import Correction


def test_without_anchor_correction_removes_only_matching_anchor() -> None:
    corrections = [
        Correction(cell_id=3, t=1, kind="anchor", y=4.5, x=2.0),
        Correction(cell_id=3, t=1, kind="validated", y=4.5, x=2.0),
        Correction(cell_id=3, t=2, kind="anchor", y=5.0, x=2.5),
        Correction(cell_id=7, t=1, kind="anchor", y=1.0, x=1.0),
    ]

    result = without_anchor_correction(corrections, cell_id=3, frame=1)

    assert result.removed is True
    assert result.remaining == [
        Correction(cell_id=3, t=1, kind="validated", y=4.5, x=2.0),
        Correction(cell_id=3, t=2, kind="anchor", y=5.0, x=2.5),
        Correction(cell_id=7, t=1, kind="anchor", y=1.0, x=1.0),
    ]


def test_without_anchor_correction_reports_when_no_anchor_was_removed() -> None:
    corrections = [
        Correction(cell_id=3, t=1, kind="validated", y=4.5, x=2.0),
        Correction(cell_id=3, t=2, kind="anchor", y=5.0, x=2.5),
    ]

    result = without_anchor_correction(corrections, cell_id=3, frame=1)

    assert result.removed is False
    assert result.remaining == corrections


def test_anchor_correction_builds_anchor_for_selected_target() -> None:
    correction = anchor_correction(cell_id=3, frame=1, y=4.5, x=2.0)

    assert correction == Correction(cell_id=3, t=1, kind="anchor", y=4.5, x=2.0)
