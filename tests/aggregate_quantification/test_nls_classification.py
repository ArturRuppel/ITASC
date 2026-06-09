import h5py
import numpy as np
import pytest
import tifffile

from cellflow.aggregate_quantification.contacts.nls_classification import (
    NLSClassificationError,
    auto_threshold,
    classify_by_threshold,
    measure_track_nls_intensity,
    patch_position_contact_analysis_nls_classes,
    split_tracks_otsu,
    split_tracks_two_clusters,
    write_nls_classification,
)


def _write_minimal_position_h5(path, cell_ids):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        cells = h5.create_group("cells/table")
        cells.create_dataset("frame", data=np.zeros(len(cell_ids), dtype=np.int64))
        cells.create_dataset("cell_id", data=np.asarray(cell_ids, dtype=np.int64))
        cells.create_dataset(
            "class_label",
            data=np.asarray(["old"] * len(cell_ids), dtype=object),
            dtype=string_dtype,
        )
        h5.create_group("cells/measurements")


# ------------------------------------------------------------------- measurement


def test_measure_track_nls_intensity_aggregates_all_pixels_across_frames():
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
    assert measurements[1].median_intensity == 30.0
    assert measurements[1].pixel_count == 3
    assert measurements[1].frame_count == 2
    assert measurements[2].median_intensity == 200.0


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


# ------------------------------------------------------------------ write to H5


def test_write_nls_classification_round_trips_custom_labels(tmp_path):
    h5_path = tmp_path / "contact_analysis.h5"
    _write_minimal_position_h5(h5_path, [1, 2, 99])
    measurements = measure_track_nls_intensity(
        np.asarray([[[10.0, 11.0, 100.0, 110.0]]]),
        np.asarray([[[1, 1, 2, 2]]], dtype=np.uint16),
    )
    assignments = classify_by_threshold(measurements, 50.0)

    write_nls_classification(
        h5_path,
        cell_ids=np.asarray([1, 2, 99]),
        measurements=measurements,
        assignments=assignments,
        threshold=50.0,
        positive_label="GFP+",
        negative_label="GFP-",
        nls_path="nls.tif",
        labels_path="labels.tif",
    )

    with h5py.File(h5_path, "r") as h5:
        cells = h5["cells/table"]
        assert cells["class_label"].asstr()[:].tolist() == ["GFP-", "GFP+", ""]
        assert cells["nls_status"].asstr()[:].tolist() == ["negative", "positive", ""]
        np.testing.assert_allclose(cells["nls_track_median_intensity"][:2], [10.5, 105.0])
        assert np.isnan(cells["nls_track_median_intensity"][2])
        meta = h5["cells/measurements/nls_classification"].attrs
        assert meta["positive_label"] == "GFP+"
        assert meta["negative_label"] == "GFP-"
        assert meta["positive_track_count"] == 1
        assert meta["negative_track_count"] == 1
        assert meta["threshold"] == 50.0


def test_patch_position_contact_analysis_uses_auto_threshold(tmp_path):
    h5_path = tmp_path / "contact_analysis.h5"
    _write_minimal_position_h5(h5_path, [1, 2])
    nls_path = tmp_path / "NLS_zavg.tif"
    labels_path = tmp_path / "tracked_labels.tif"
    tifffile.imwrite(labels_path, np.asarray([[[1, 2]]], dtype=np.uint16))
    tifffile.imwrite(nls_path, np.asarray([[[10.0, 100.0]]], dtype=np.float32))

    summary = patch_position_contact_analysis_nls_classes(h5_path, nls_path, labels_path)

    assert summary.positive_track_count == 1
    assert summary.negative_track_count == 1
    with h5py.File(h5_path, "r") as h5:
        assert h5["cells/table/nls_status"].asstr()[:].tolist() == ["negative", "positive"]


def test_patch_position_contact_analysis_honours_explicit_threshold(tmp_path):
    h5_path = tmp_path / "contact_analysis.h5"
    _write_minimal_position_h5(h5_path, [1, 2])
    nls_path = tmp_path / "NLS_zavg.tif"
    labels_path = tmp_path / "tracked_labels.tif"
    tifffile.imwrite(labels_path, np.asarray([[[1, 2]]], dtype=np.uint16))
    tifffile.imwrite(nls_path, np.asarray([[[10.0, 100.0]]], dtype=np.float32))

    # A threshold above both medians forces everything negative.
    summary = patch_position_contact_analysis_nls_classes(
        h5_path, nls_path, labels_path, threshold=200.0
    )

    assert summary.positive_track_count == 0
    with h5py.File(h5_path, "r") as h5:
        assert h5["cells/table/nls_status"].asstr()[:].tolist() == ["negative", "negative"]


def test_write_nls_classification_validates_before_mutating(tmp_path):
    h5_path = tmp_path / "contact_analysis.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_group("provenance")  # no cells/table

    with pytest.raises(NLSClassificationError, match="missing cells/table"):
        write_nls_classification(
            h5_path,
            cell_ids=np.asarray([1]),
            measurements={},
            assignments={1: "positive"},
            threshold=1.0,
            nls_path="nls.tif",
            labels_path="labels.tif",
        )
