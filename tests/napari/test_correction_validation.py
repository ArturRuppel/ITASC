from __future__ import annotations

import numpy as np

from cellflow.napari._correction_validation import (
    SelectedCorrectionTarget,
    correction_for_label_frame,
    selected_correction_target,
)


def test_selected_correction_target_returns_centroid_for_selected_label() -> None:
    stack = np.zeros((2, 6, 7), dtype=np.uint32)
    stack[1, 2:5, 1:4] = 9

    target = selected_correction_target(stack, cell_id=9, frame=1)

    assert target == SelectedCorrectionTarget(
        cell_id=9,
        frame=1,
        y=3.0,
        x=2.0,
    )


def test_selected_correction_target_returns_none_when_label_absent_or_frame_ambiguous() -> None:
    assert selected_correction_target(
        np.zeros((1, 3, 3), dtype=np.uint32),
        cell_id=4,
        frame=0,
    ) is None
    assert selected_correction_target(
        np.zeros((1, 2, 3, 3), dtype=np.uint32),
        cell_id=4,
        frame=0,
    ) is None


def test_correction_for_label_frame_builds_validated_correction_at_centroid() -> None:
    stack = np.zeros((3, 8, 8), dtype=np.uint32)
    stack[2, 4:6, 1:4] = 5

    correction = correction_for_label_frame(stack, cell_id=5, frame=2)

    assert correction is not None
    assert correction.cell_id == 5
    assert correction.t == 2
    assert correction.kind == "validated"
    assert correction.y == 4.5
    assert correction.x == 2.0


def test_correction_for_label_frame_returns_none_when_label_is_absent() -> None:
    correction = correction_for_label_frame(
        np.zeros((1, 4, 4), dtype=np.uint32),
        cell_id=5,
        frame=0,
    )

    assert correction is None
