import numpy as np

from cellflow.tracking.consensus_movie import (
    apply_vote_thresholds,
    build_consensus_movie,
    build_consensus_movie_with_thresholds,
    collapse_z_by_label_presence,
    load_compactness_groups,
    resolve_vote_thresholds,
    smooth_consensus_labels,
    vote_label_footprints,
    vote_consensus_labels,
)


def test_vote_consensus_labels_keeps_pixels_with_enough_vote_support():
    labels = np.array(
        [
            [[1, 2], [0, 3]],
            [[1, 2], [4, 0]],
            [[1, 5], [4, 3]],
            [[2, 5], [4, 3]],
        ],
        dtype=np.uint32,
    )

    consensus, support = vote_consensus_labels(labels, vote_threshold=0.5)

    np.testing.assert_array_equal(consensus, np.array([[1, 2], [4, 3]], dtype=np.uint32))
    np.testing.assert_allclose(support, np.array([[0.75, 0.5], [0.75, 0.75]], dtype=np.float32))


def test_vote_consensus_labels_sends_uncertain_pixels_to_background():
    labels = np.array(
        [
            [[1, 2]],
            [[1, 3]],
            [[4, 3]],
            [[4, 2]],
        ],
        dtype=np.uint32,
    )

    consensus, support = vote_consensus_labels(labels, vote_threshold=0.75)

    np.testing.assert_array_equal(consensus, np.array([[0, 0]], dtype=np.uint32))
    np.testing.assert_allclose(support, np.array([[0.5, 0.5]], dtype=np.float32))


def test_smooth_consensus_labels_uses_neighbor_support_to_stabilize_labels():
    labels = np.array(
        [
            [[1, 0]],
            [[1, 2]],
            [[1, 0]],
        ],
        dtype=np.uint32,
    )
    support = np.array(
        [
            [[1.0, 0.0]],
            [[0.4, 0.6]],
            [[1.0, 0.0]],
        ],
        dtype=np.float32,
    )

    smoothed, smoothed_support = smooth_consensus_labels(labels, support, vote_threshold=0.5)

    np.testing.assert_array_equal(smoothed[1], np.array([[1, 0]], dtype=np.uint32))
    assert smoothed_support[1, 0, 0] > 0.5


def test_apply_vote_thresholds_accepts_per_frame_thresholds():
    labels = np.array(
        [
            [[1, 2]],
            [[1, 2]],
        ],
        dtype=np.uint32,
    )
    support = np.array(
        [
            [[0.4, 0.7]],
            [[0.6, 0.7]],
        ],
        dtype=np.float32,
    )

    thresholded = apply_vote_thresholds(labels, support, np.array([0.5, 0.65], dtype=np.float32))

    np.testing.assert_array_equal(
        thresholded,
        np.array(
            [
                [[0, 2]],
                [[0, 2]],
            ],
            dtype=np.uint32,
        ),
    )


def test_resolve_vote_thresholds_can_use_clipped_frame_percentiles():
    labels = np.array(
        [
            [[1, 2, 0, 0]],
            [[1, 2, 3, 0]],
        ],
        dtype=np.uint32,
    )
    support = np.array(
        [
            [[0.2, 0.8, 0.9, 0.95]],
            [[0.4, 0.6, 0.9, 0.95]],
        ],
        dtype=np.float32,
    )

    thresholds = resolve_vote_thresholds(
        support,
        labels,
        mode="percentile",
        percentile=50.0,
        min_threshold=0.35,
        max_threshold=0.75,
    )

    np.testing.assert_allclose(thresholds, np.array([0.5, 0.6], dtype=np.float32))


def test_collapse_z_by_label_presence_ignores_background_misses():
    labels = np.array(
        [
            [[0, 0, 2], [0, 4, 0]],
            [[1, 0, 2], [0, 0, 0]],
            [[1, 3, 0], [0, 4, 5]],
            [[0, 3, 0], [0, 6, 5]],
        ],
        dtype=np.uint32,
    )

    collapsed = collapse_z_by_label_presence(labels)

    np.testing.assert_array_equal(
        collapsed,
        np.array(
            [
                [1, 3, 2],
                [0, 4, 5],
            ],
            dtype=np.uint32,
        ),
    )


def test_vote_label_footprints_ignores_background_as_a_competing_category():
    labels = np.array(
        [
            [[0, 0, 3]],
            [[0, 1, 3]],
            [[1, 1, 0]],
            [[2, 0, 0]],
            [[0, 0, 0]],
        ],
        dtype=np.uint32,
    )

    voted, support = vote_label_footprints(labels)

    np.testing.assert_array_equal(voted, np.array([[1, 1, 3]], dtype=np.uint32))
    np.testing.assert_allclose(support, np.array([[0.2, 0.4, 0.4]], dtype=np.float32))


def test_vote_label_footprints_can_weight_high_coverage_parameters():
    labels = np.array(
        [
            [[1]],
            [[2]],
            [[2]],
            [[0]],
        ],
        dtype=np.uint32,
    )

    voted, support = vote_label_footprints(
        labels,
        weights=np.array([4.0, 1.0, 1.0, 4.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(voted, np.array([[1]], dtype=np.uint32))
    np.testing.assert_allclose(support, np.array([[0.4]], dtype=np.float32))


def test_load_compactness_groups_sorts_parameters_by_compactness_and_threshold(tmp_path):
    h5_path = tmp_path / "hypotheses.h5"
    with _minimal_hypotheses_h5(h5_path) as root:
        _write_hypothesis(root, 0, 2, np.zeros((1, 2, 2), dtype=np.uint32), compactness=0.1, threshold=0.3)
        _write_hypothesis(root, 0, 0, np.zeros((1, 2, 2), dtype=np.uint32), compactness=0.0, threshold=0.2)
        _write_hypothesis(root, 0, 1, np.zeros((1, 2, 2), dtype=np.uint32), compactness=0.1, threshold=0.1)

    groups = load_compactness_groups(h5_path)

    assert [group.compactness for group in groups] == [0.0, 0.1]
    assert [member.p for member in groups[1].members] == [1, 2]


def test_build_consensus_movie_votes_across_parameters_and_z_slices(tmp_path):
    h5_path = tmp_path / "hypotheses.h5"
    with _minimal_hypotheses_h5(h5_path) as root:
        _write_hypothesis(
            root,
            0,
            0,
            np.array(
                [
                    [[1, 1], [0, 2]],
                    [[1, 0], [0, 2]],
                ],
                dtype=np.uint32,
            ),
            compactness=0.1,
            threshold=0.1,
        )
        _write_hypothesis(
            root,
            0,
            1,
            np.array(
                [
                    [[1, 1], [3, 2]],
                    [[1, 4], [3, 2]],
                ],
                dtype=np.uint32,
            ),
            compactness=0.1,
            threshold=0.2,
        )

    groups = load_compactness_groups(h5_path)
    labels, support = build_consensus_movie(
        h5_path,
        groups[0],
        vote_threshold=0.5,
        smooth_temporally=False,
    )

    np.testing.assert_array_equal(labels[0], np.array([[1, 1], [0, 2]], dtype=np.uint32))
    np.testing.assert_allclose(support[0], np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32))


def test_build_consensus_movie_can_return_dynamic_thresholds(tmp_path):
    h5_path = tmp_path / "hypotheses.h5"
    with _minimal_hypotheses_h5(h5_path) as root:
        _write_hypothesis(
            root,
            0,
            0,
            np.array([[[1, 1, 2, 3, 6]]], dtype=np.uint32),
            compactness=0.1,
            threshold=0.1,
        )
        _write_hypothesis(
            root,
            0,
            1,
            np.array([[[1, 4, 5, 4, 7]]], dtype=np.uint32),
            compactness=0.1,
            threshold=0.2,
        )
        _write_hypothesis(
            root,
            1,
            0,
            np.array([[[1, 1, 2, 3, 6]]], dtype=np.uint32),
            compactness=0.1,
            threshold=0.1,
        )
        _write_hypothesis(
            root,
            1,
            1,
            np.array([[[1, 1, 5, 0, 6]]], dtype=np.uint32),
            compactness=0.1,
            threshold=0.2,
        )

    groups = load_compactness_groups(h5_path)
    movie = build_consensus_movie_with_thresholds(
        h5_path,
        groups[0],
        threshold_mode="percentile",
        threshold_percentile=50.0,
        min_vote_threshold=0.35,
        max_vote_threshold=0.75,
        smooth_temporally=False,
    )

    np.testing.assert_allclose(movie.thresholds, np.array([0.5, 0.75], dtype=np.float32))
    np.testing.assert_array_equal(movie.labels[0], np.array([[1, 1, 2, 3, 6]], dtype=np.uint32))
    np.testing.assert_array_equal(movie.labels[1], np.array([[1, 1, 0, 0, 6]], dtype=np.uint32))


def _minimal_hypotheses_h5(path):
    import h5py

    h5 = h5py.File(path, "w")
    h5.create_group("hypotheses")
    return h5


def _write_hypothesis(root, t, p, labels, *, compactness, threshold):
    group = root["hypotheses"].require_group(f"t{t:03d}").create_group(f"p{p:03d}")
    group.create_dataset("labels", data=labels)
    group.attrs["compactness"] = compactness
    group.attrs["foreground_threshold"] = threshold
    group.attrs["basin"] = "flow_mag"
