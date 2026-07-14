from __future__ import annotations

import numpy as np

from itasc.napari.correction._correction_protection import (
    protected_cell_ids_at_frame,
    protected_cell_mask,
)
from itasc.tracking_ultrack.corrections import Correction


def test_protected_cell_ids_at_frame_combines_validated_tracks_and_anchors() -> None:
    validated_tracks = {
        3: {0, 2},
        5: {1},
    }
    corrections = [
        Correction(cell_id=7, t=2, kind="anchor", y=1.0, x=1.0),
        Correction(cell_id=8, t=2, kind="validated", y=2.0, x=2.0),
        Correction(cell_id=9, t=1, kind="anchor", y=3.0, x=3.0),
    ]

    protected = protected_cell_ids_at_frame(
        validated_tracks,
        corrections,
        frame=2,
    )

    assert protected == {3, 7}


def test_protected_cell_ids_at_frame_can_exclude_source_cell() -> None:
    protected = protected_cell_ids_at_frame(
        {3: {2}, 5: {2}},
        [Correction(cell_id=7, t=2, kind="anchor", y=1.0, x=1.0)],
        frame=2,
        exclude_cell_id=3,
    )

    assert protected == {5, 7}


def test_protected_cell_mask_marks_only_protected_ids() -> None:
    frame = np.array(
        [
            [0, 3, 3],
            [5, 7, 0],
        ],
        dtype=np.uint32,
    )

    mask = protected_cell_mask(frame, {3, 7})

    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [False, True, True],
                [False, True, False],
            ],
            dtype=bool,
        ),
    )


def test_protected_cell_mask_returns_empty_mask_without_protected_ids() -> None:
    frame = np.ones((2, 3), dtype=np.uint32)

    mask = protected_cell_mask(frame, set())

    assert mask.dtype == bool
    assert mask.shape == frame.shape
    assert not mask.any()
