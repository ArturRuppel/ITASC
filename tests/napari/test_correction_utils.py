from __future__ import annotations

import numpy as np

from cellflow.napari._correction_utils import (
    frame_view_2d,
    reassign_ids_ordered,
    reassign_ids_stack,
    reorder_stack_by_quality,
    retrack_stack_direction,
    remove_unvalidated_labels,
    track_order_by_frame_and_size,
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


def test_retrack_stack_direction_retracks_forward_and_skips_validated_frames() -> None:
    stack = np.zeros((4, 2, 2), dtype=np.uint32)
    stack[0, 0, 0] = 1
    stack[1, 0, 1] = 2
    stack[2, 1, 0] = 3
    stack[3, 1, 1] = 4
    calls = []

    def retrack(previous, current, locked, *, max_dist_px, reserved_ids, **weights):
        calls.append(
            (previous.copy(), current.copy(), locked, max_dist_px, reserved_ids, weights)
        )
        return previous + 10

    result = retrack_stack_direction(
        stack,
        start_frame=0,
        direction="forward",
        fully_validated_frames={2},
        validated_cells_at_frame=lambda t: {99} if t == 1 else set(),
        retrack_frame=retrack,
        max_dist_px=7.5,
        reserved_ids={5, 6},
        area_weight=2.0,
        iou_weight=3.0,
        distance_weight=0.1,
    )

    assert result.n_retracked == 2
    assert result.n_skipped == 1
    assert result.first_target_frame == 1
    assert len(calls) == 2
    assert calls[0][2:] == (
        {99},
        7.5,
        {5, 6},
        {"area_weight": 2.0, "iou_weight": 3.0, "distance_weight": 0.1},
    )
    np.testing.assert_array_equal(result.stack[1], stack[0] + 10)
    np.testing.assert_array_equal(result.stack[2], stack[2])
    np.testing.assert_array_equal(result.stack[3], result.stack[2] + 10)
    np.testing.assert_array_equal(stack[1], np.array([[0, 2], [0, 0]], dtype=np.uint32))


def test_retrack_stack_direction_retracks_backward() -> None:
    stack = np.zeros((3, 2, 2), dtype=np.uint32)
    stack[0, 0, 0] = 1
    stack[1, 0, 1] = 2
    stack[2, 1, 1] = 3

    def retrack(previous, current, locked, *, max_dist_px, reserved_ids, **weights):
        return previous + 20

    result = retrack_stack_direction(
        stack,
        start_frame=2,
        direction="backward",
        fully_validated_frames=set(),
        validated_cells_at_frame=lambda _t: set(),
        retrack_frame=retrack,
        max_dist_px=4.0,
        reserved_ids=set(),
    )

    assert result.n_retracked == 2
    assert result.n_skipped == 0
    assert result.first_target_frame == 1
    np.testing.assert_array_equal(result.stack[1], stack[2] + 20)
    np.testing.assert_array_equal(result.stack[0], result.stack[1] + 20)


def test_reassign_ids_ordered_follows_explicit_order_then_ascending() -> None:
    stack = np.array(
        [
            [[0, 5, 5], [9, 0, 3]],
            [[3, 9, 0], [0, 5, 0]],
        ],
        dtype=np.uint32,
    )

    # Best -> 1: order says 9 then 3; remaining (5) follows ascending.
    remapped, n_cells, old_to_new = reassign_ids_ordered(stack, [9, 3])

    assert n_cells == 3
    assert old_to_new == {9: 1, 3: 2, 5: 3}
    assert remapped[0, 1, 0] == 1  # was 9
    assert remapped[0, 1, 2] == 2  # was 3
    assert remapped[0, 0, 1] == 3  # was 5


def test_reassign_ids_ordered_ignores_absent_ids_in_order() -> None:
    stack = np.array([[1, 2], [0, 1]], dtype=np.uint32)

    remapped, n_cells, old_to_new = reassign_ids_ordered(stack, [99, 2, 1])

    # 99 is not present; 2 then 1 by order.
    assert old_to_new == {2: 1, 1: 2}
    assert n_cells == 2


def test_reassign_ids_ordered_empty_order_matches_compaction() -> None:
    stack = np.array([[0, 7], [42, 7]], dtype=np.uint32)

    remapped, n_cells, old_to_new = reassign_ids_ordered(stack, [])

    assert old_to_new == {7: 1, 42: 2}
    np.testing.assert_array_equal(remapped, np.array([[0, 1], [2, 1]]))


def test_track_order_sorts_by_start_frame_then_length() -> None:
    # Track 1: frames 0-2 (len 3, start 0); 2: frame 0 only (len 1, start 0);
    # 3: frames 1-2 (len 2, start 1).
    stack = np.zeros((3, 1, 3), dtype=np.uint32)
    stack[0, 0, 0] = 1
    stack[0, 0, 1] = 2
    stack[1, 0, 0] = 1
    stack[2, 0, 0] = 1
    stack[1, 0, 2] = 3
    stack[2, 0, 2] = 3

    # Start frame wins: 1 and 2 (start 0) precede 3 (start 1); within start 0
    # the longer track 1 comes before 2.
    assert track_order_by_frame_and_size(stack) == [1, 2, 3]


def test_track_order_keeps_priority_group_first() -> None:
    stack = np.zeros((2, 1, 3), dtype=np.uint32)
    stack[0, 0, 0] = 1  # unvalidated, starts at frame 0
    stack[1, 0, 1] = 2  # validated, starts at frame 1
    stack[1, 0, 2] = 3  # validated, starts at frame 1

    # Validated tracks lead despite their later start frame; within the group
    # they still follow start frame / length order.
    assert track_order_by_frame_and_size(stack, priority_ids=[3, 2]) == [2, 3, 1]


def test_track_order_ignores_absent_priority_ids() -> None:
    stack = np.array([[1, 2], [0, 1]], dtype=np.uint32)

    assert track_order_by_frame_and_size(stack, priority_ids=[99]) == [1, 2]


def test_track_order_handles_2d_single_frame() -> None:
    labels = np.array([[0, 5, 5], [8, 8, 8]], dtype=np.uint32)

    # Equal start frame and length -> numeric id breaks the tie.
    assert track_order_by_frame_and_size(labels) == [5, 8]


def test_reorder_stack_by_quality_relabels_best_track_to_one() -> None:
    stack = np.array([[0, 1, 2], [3, 1, 2]], dtype=np.uint32)
    scores = {1: 0.1, 2: 5.0, 3: 2.0}  # best -> 2, then 3, then 1

    relabeled, old_to_new = reorder_stack_by_quality(stack, scores)

    assert old_to_new == {2: 1, 3: 2, 1: 3}
    assert relabeled[0, 2] == 1  # was track 2 (best)


def test_reorder_stack_by_quality_no_scores_is_noop() -> None:
    stack = np.array([[1, 2]], dtype=np.uint32)

    relabeled, old_to_new = reorder_stack_by_quality(stack, {})

    assert old_to_new == {}
    assert relabeled is stack


def test_reorder_stack_by_quality_remaps_validated_tracks(monkeypatch) -> None:
    import cellflow.tracking_ultrack.validation_state as validation

    calls = {}

    def fake_remap(pos_dir, old_to_new):
        calls["pos_dir"] = pos_dir
        calls["old_to_new"] = old_to_new

    monkeypatch.setattr(validation, "remap_validated_tracks", fake_remap)

    stack = np.array([[1, 2]], dtype=np.uint32)
    scores = {1: 1.0, 2: 9.0}  # best -> 2

    _relabeled, old_to_new = reorder_stack_by_quality(stack, scores, pos_dir="/tmp/pos")

    assert old_to_new == {2: 1, 1: 2}
    assert calls["old_to_new"] == old_to_new
    assert str(calls["pos_dir"]) == "/tmp/pos"
