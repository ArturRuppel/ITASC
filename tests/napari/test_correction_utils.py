from __future__ import annotations

import numpy as np

from cellflow.napari._correction_utils import (
    frame_view_2d,
    reassign_ids_stack,
    remove_unvalidated_labels,
)


def test_reassign_ids_stack_returns_empty_stack_without_mapping() -> None:
    stack = np.zeros((0, 0, 0), dtype=np.uint32)

    remapped, n_cells, old_to_new = reassign_ids_stack(stack)

    assert remapped is stack
    assert n_cells == 0
    assert old_to_new == {}


def test_reassign_ids_stack_returns_all_zero_stack_without_mapping() -> None:
    stack = np.zeros((2, 3, 4), dtype=np.uint32)

    remapped, n_cells, old_to_new = reassign_ids_stack(stack)

    assert remapped is stack
    assert n_cells == 0
    assert old_to_new == {}


def test_reassign_ids_stack_compacts_nonconsecutive_ids_across_frames() -> None:
    stack = np.array(
        [
            [[0, 7, 7], [42, 0, 100]],
            [[100, 42, 0], [0, 7, 0]],
        ],
        dtype=np.uint32,
    )

    remapped, n_cells, old_to_new = reassign_ids_stack(stack)

    expected = np.array(
        [
            [[0, 1, 1], [2, 0, 3]],
            [[3, 2, 0], [0, 1, 0]],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(remapped, expected)
    assert n_cells == 3
    assert old_to_new == {7: 1, 42: 2, 100: 3}


def test_remove_unvalidated_labels_preserves_only_frame_validated_ids() -> None:
    stack = np.zeros((2, 5, 5), dtype=np.uint32)
    stack[0, 1:3, 1:3] = 7
    stack[0, 3:5, 3:5] = 9
    stack[1, 1:3, 1:3] = 7
    stack[1, 3:5, 3:5] = 11

    changed_frames, changed_pixels = remove_unvalidated_labels(stack, {7: {0}})

    assert changed_frames == 2
    assert changed_pixels == 12
    assert np.all(stack[0, 1:3, 1:3] == 7)
    assert not np.any(stack[0, 3:5, 3:5] == 9)
    assert not np.any(stack[1] == 7)
    assert not np.any(stack[1] == 11)


def test_remove_unvalidated_labels_treats_2d_arrays_as_single_frame() -> None:
    labels = np.array(
        [
            [0, 5, 5],
            [8, 8, 0],
        ],
        dtype=np.uint32,
    )

    changed_frames, changed_pixels = remove_unvalidated_labels(labels, {5: {0}})

    expected = np.array(
        [
            [0, 5, 5],
            [0, 0, 0],
        ],
        dtype=np.uint32,
    )
    assert changed_frames == 1
    assert changed_pixels == 2
    np.testing.assert_array_equal(labels, expected)


def test_remove_unvalidated_labels_reports_no_changes_when_all_labels_validated() -> None:
    labels = np.array([[[0, 3], [4, 0]]], dtype=np.uint32)

    changed_frames, changed_pixels = remove_unvalidated_labels(
        labels,
        {3: {0}, 4: {0}},
    )

    assert changed_frames == 0
    assert changed_pixels == 0
    np.testing.assert_array_equal(labels, np.array([[[0, 3], [4, 0]]], dtype=np.uint32))


def test_frame_view_2d_returns_timepoint_view() -> None:
    stack = np.arange(2 * 3 * 4, dtype=np.uint32).reshape(2, 3, 4)

    view = frame_view_2d(stack, 1)

    assert np.shares_memory(view, stack)
    np.testing.assert_array_equal(view, stack[1])


def test_frame_view_2d_squeezes_singleton_spatial_prefix_axes() -> None:
    stack = np.arange(2 * 1 * 1 * 3 * 4, dtype=np.uint32).reshape(2, 1, 1, 3, 4)

    view = frame_view_2d(stack, 1)

    np.testing.assert_array_equal(view, stack[1, 0, 0])


def test_frame_view_2d_rejects_missing_or_ambiguous_timepoint() -> None:
    assert frame_view_2d(np.zeros((3, 4), dtype=np.uint32), 0) is None
    assert frame_view_2d(np.zeros((1, 3, 4), dtype=np.uint32), -1) is None
    assert frame_view_2d(np.zeros((1, 3, 4), dtype=np.uint32), 1) is None
    assert frame_view_2d(np.zeros((1, 2, 3, 4), dtype=np.uint32), 0) is None
