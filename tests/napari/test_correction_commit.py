from __future__ import annotations

import numpy as np

from itasc.napari.correction._correction_commit import (
    prepare_committed_labels,
    remove_unvalidated_from_data,
)


def test_prepare_committed_labels_reassigns_ids_before_removing_unvalidated() -> None:
    stack = np.zeros((2, 4, 4), dtype=np.uint32)
    stack[0, 0:2, 0:2] = 7
    stack[0, 2:4, 2:4] = 42
    stack[1, 0:2, 0:2] = 42
    stack[1, 2:4, 0:2] = 99

    result = prepare_committed_labels(stack, {42: {0, 1}})

    assert result.n_cells == 3
    assert result.old_to_new == {7: 1, 42: 2, 99: 3}
    assert result.validated_tracks == {2: {0, 1}}
    assert result.changed_frames == 2
    assert result.changed_pixels == 8
    assert not np.any(result.stack == 1)
    assert not np.any(result.stack == 3)
    assert np.all(result.stack[0, 2:4, 2:4] == 2)
    assert np.all(result.stack[1, 0:2, 0:2] == 2)
    assert np.any(stack == 7)
    assert np.any(stack == 99)


def test_prepare_committed_labels_drops_validation_for_labels_not_in_stack() -> None:
    stack = np.zeros((1, 3, 3), dtype=np.uint32)
    stack[0, 1:, 1:] = 5

    result = prepare_committed_labels(stack, {99: {0}})

    assert result.old_to_new == {5: 1}
    assert result.validated_tracks == {}
    assert result.changed_frames == 1
    assert result.changed_pixels == 4
    assert not np.any(result.stack)


def test_remove_unvalidated_from_data_mutates_supplied_data() -> None:
    labels = np.array(
        [
            [0, 4, 4],
            [8, 8, 0],
        ],
        dtype=np.uint32,
    )

    result = remove_unvalidated_from_data(labels, {4: {0}})

    assert result.changed_frames == 1
    assert result.changed_pixels == 2
    np.testing.assert_array_equal(
        labels,
        np.array([[0, 4, 4], [0, 0, 0]], dtype=np.uint32),
    )
