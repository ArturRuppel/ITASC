import numpy as np

from cellflow.tracking_ultrack.metrics import (
    binary_labelmap_iou,
    tracked_label_summary,
)


def test_tracked_label_summary_counts_tracks_and_average_length():
    labels = np.array(
        [
            [[0, 1], [2, 2]],
            [[0, 1], [0, 3]],
            [[4, 1], [0, 3]],
        ],
        dtype=np.uint32,
    )

    summary = tracked_label_summary(labels)

    assert summary.n_tracks == 4
    assert summary.average_length == 1.75
    assert summary.track_lengths == {1: 3, 2: 1, 3: 2, 4: 1}


def test_binary_labelmap_iou_compares_foreground_pixels():
    lhs = np.array(
        [
            [[0, 1], [2, 0]],
            [[0, 0], [3, 3]],
        ],
        dtype=np.uint32,
    )
    rhs = np.array(
        [
            [[0, 9], [0, 0]],
            [[8, 0], [7, 7]],
        ],
        dtype=np.uint32,
    )

    assert binary_labelmap_iou(lhs, rhs) == 3 / 5


def test_binary_labelmap_iou_is_one_for_two_empty_labelmaps():
    empty = np.zeros((2, 3, 3), dtype=np.uint32)

    assert binary_labelmap_iou(empty, empty) == 1.0
