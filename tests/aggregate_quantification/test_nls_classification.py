import numpy as np
import pytest
import tifffile

from cellflow.aggregate_quantification.contacts.nls_classification import (
    NLSClassificationError,
    auto_threshold,
    classify_by_threshold,
    classify_position_nls_to_csv,
    measure_track_nls_intensity,
    read_nls_classification_csv,
    split_tracks_otsu,
    split_tracks_two_clusters,
    write_nls_classification_csv,
)


# ------------------------------------------------------------------- measurement


def test_measure_track_nls_intensity_is_p90_of_per_frame_medians():
    labels = np.asarray(
        [
            [[1, 1, 0], [2, 0, 0]],
            [[1, 0, 0], [2, 2, 0]],
        ],
        dtype=np.uint16,
    )
    nls = np.asarray(
        [
            [[10.0, 30.0, 0.0], [100.0, 0.0, 0.0]],
            [[50.0, 0.0, 0.0], [200.0, 300.0, 0.0]],
        ],
        dtype=float,
    )

    measurements = measure_track_nls_intensity(nls, labels)

    assert set(measurements) == {1, 2}
    # Track 1 per-frame medians: median(10, 30)=20, median(50)=50 → 90th pct = 47.0.
    assert measurements[1].intensity == 47.0
    assert measurements[1].pixel_count == 3
    assert measurements[1].frame_count == 2
    # Track 2 per-frame medians: median(100)=100, median(200, 300)=250 → 90th pct = 235.0.
    assert measurements[2].intensity == 235.0


def test_measure_track_nls_intensity_ignores_partial_frame_contamination():
    # A negative track (intrinsically dim) whose mask catches a few bright pixels
    # in one frame — e.g. brushing past a positive neighbour. The per-frame median
    # ignores that minority, so the track stays dim and won't trip the threshold.
    labels = np.asarray(
        [
            [[1, 1, 1, 0]],
            [[1, 1, 1, 0]],
        ],
        dtype=np.uint16,
    )
    nls = np.asarray(
        [
            [[10.0, 10.0, 10.0, 0.0]],
            [[10.0, 10.0, 1000.0, 0.0]],  # one contaminated pixel
        ],
        dtype=float,
    )

    measurements = measure_track_nls_intensity(nls, labels)

    # Per-frame medians: median(10,10,10)=10, median(10,10,1000)=10 → p90 = 10.0.
    # (A per-frame *mean* would have spiked the second frame to ~340.)
    assert measurements[1].intensity == 10.0


def test_measure_track_nls_intensity_rejects_mismatched_shapes():
    with pytest.raises(NLSClassificationError, match="shapes do not match"):
        measure_track_nls_intensity(np.zeros((1, 2, 2)), np.zeros((1, 3, 3)))


# --------------------------------------------------------------------- splitting


def test_split_tracks_otsu_splits_synthetic_bimodal_track_medians():
    threshold, assignments = split_tracks_otsu({1: 10.0, 2: 12.0, 3: 100.0, 4: 110.0})

    assert 12.0 < threshold < 100.0
    assert assignments == {1: "low", 2: "low", 3: "high", 4: "high"}


def test_split_tracks_otsu_rejects_invalid_splits():
    with pytest.raises(NLSClassificationError, match="fewer than two"):
        split_tracks_otsu({1: 10.0})
    with pytest.raises(NLSClassificationError, match="identical"):
        split_tracks_otsu({1: 10.0, 2: 10.0})


def test_split_tracks_two_clusters_keeps_faint_positive_cluster_high():
    medians = dict(
        enumerate(
            [106.0, 106.0, 106.0, 106.0, 107.0, 107.0, 107.0, 108.0, 124.0, 132.0, 168.0, 306.0],
            start=1,
        )
    )

    threshold, assignments = split_tracks_two_clusters(medians)

    assert 108.0 < threshold < 124.0
    assert [t for t, s in assignments.items() if s == "high"] == [9, 10, 11, 12]


# ----------------------------------------------------------- threshold model


def test_auto_threshold_lands_between_clusters():
    threshold = auto_threshold({1: 10.0, 2: 12.0, 3: 100.0, 4: 110.0})
    assert 12.0 < threshold < 100.0


def test_classify_by_threshold_is_strictly_above_threshold():
    measurements = measure_track_nls_intensity(
        np.asarray([[[10.0, 50.0, 100.0]]]),
        np.asarray([[[1, 2, 3]]], dtype=np.uint16),
    )
    assignments = classify_by_threshold(measurements, 50.0)
    assert assignments == {1: "negative", 2: "negative", 3: "positive"}


# ------------------------------------------------------------------ sidecar CSV


def test_write_and_read_nls_classification_csv_round_trips_two_columns(tmp_path):
    csv_path = tmp_path / "nls_classification.csv"
    # assignments map track id -> positive/negative status; the row label is the
    # caller's positive/negative string.
    write_nls_classification_csv(
        csv_path,
        {2: "positive", 1: "negative"},
        positive_label="GFP+",
        negative_label="GFP-",
    )

    # Exactly two columns, sorted by id.
    lines = csv_path.read_text().splitlines()
    assert lines == ["id,label", "1,GFP-", "2,GFP+"]
    assert read_nls_classification_csv(csv_path) == {1: "GFP-", 2: "GFP+"}


def test_classify_position_nls_to_csv_uses_auto_threshold(tmp_path):
    csv_path = tmp_path / "nls_classification.csv"
    nls_path = tmp_path / "NLS_zavg.tif"
    labels_path = tmp_path / "tracked_labels.tif"
    tifffile.imwrite(labels_path, np.asarray([[[1, 2]]], dtype=np.uint16))
    tifffile.imwrite(nls_path, np.asarray([[[10.0, 100.0]]], dtype=np.float32))

    summary = classify_position_nls_to_csv(csv_path, nls_path, labels_path)

    assert summary.csv_path == csv_path
    assert summary.positive_track_count == 1
    assert summary.negative_track_count == 1
    # No .h5 is touched — only the sidecar CSV is written.
    assert read_nls_classification_csv(csv_path) == {1: "negative", 2: "positive"}


def test_classify_position_nls_to_csv_honours_explicit_threshold(tmp_path):
    csv_path = tmp_path / "nls_classification.csv"
    nls_path = tmp_path / "NLS_zavg.tif"
    labels_path = tmp_path / "tracked_labels.tif"
    tifffile.imwrite(labels_path, np.asarray([[[1, 2]]], dtype=np.uint16))
    tifffile.imwrite(nls_path, np.asarray([[[10.0, 100.0]]], dtype=np.float32))

    # A threshold above both track intensities forces everything negative.
    summary = classify_position_nls_to_csv(csv_path, nls_path, labels_path, threshold=200.0)

    assert summary.positive_track_count == 0
    assert read_nls_classification_csv(csv_path) == {1: "negative", 2: "negative"}


def test_classify_position_nls_to_csv_creates_missing_parent_dir(tmp_path):
    csv_path = tmp_path / "aggregate_quantification" / "nls_classification.csv"
    nls_path = tmp_path / "NLS_zavg.tif"
    labels_path = tmp_path / "tracked_labels.tif"
    tifffile.imwrite(labels_path, np.asarray([[[1, 2]]], dtype=np.uint16))
    tifffile.imwrite(nls_path, np.asarray([[[10.0, 100.0]]], dtype=np.float32))

    classify_position_nls_to_csv(csv_path, nls_path, labels_path)

    assert csv_path.is_file()
