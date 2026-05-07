import h5py
import numpy as np
import pytest
import tifffile

from cellflow.analysis.nls_classification import (
    NLSClassificationError,
    measure_track_nls_intensity,
    patch_position_artifact_nls_classes,
    split_tracks_otsu,
)


def _write_minimal_position_h5(path, cell_ids):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        cells = h5.create_group("cells/table")
        cells.create_dataset("frame", data=np.zeros(len(cell_ids), dtype=np.int64))
        cells.create_dataset("cell_id", data=np.asarray(cell_ids, dtype=np.int64))
        cells.create_dataset("class_label", data=np.asarray(["old"] * len(cell_ids), dtype=object), dtype=string_dtype)
        h5.create_group("cells/measurements")


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
    assert measurements[2].pixel_count == 3
    assert measurements[2].frame_count == 2


def test_split_tracks_otsu_splits_synthetic_bimodal_track_medians():
    threshold, assignments = split_tracks_otsu({1: 10.0, 2: 12.0, 3: 100.0, 4: 110.0})

    assert 12.0 < threshold < 100.0
    assert assignments == {1: "low", 2: "low", 3: "high", 4: "high"}


def test_patch_position_artifact_writes_classes_audit_columns_and_metadata(tmp_path):
    h5_path = tmp_path / "position_analysis.h5"
    _write_minimal_position_h5(h5_path, [1, 2, 99])
    nls_path = tmp_path / "NLS_zavg.tif"
    labels_path = tmp_path / "tracked_labels.tif"
    labels = np.asarray([[[1, 1, 0, 2, 2], [0, 0, 0, 0, 0]]], dtype=np.uint16)
    nls = np.asarray([[[10.0, 12.0, 0.0, 100.0, 110.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    tifffile.imwrite(nls_path, nls)
    tifffile.imwrite(labels_path, labels)

    summary = patch_position_artifact_nls_classes(h5_path, nls_path, labels_path)

    assert summary.h5_path == h5_path
    assert summary.track_count == 2
    assert summary.high_track_count == 1
    assert summary.low_track_count == 1
    with h5py.File(h5_path, "r") as h5:
        cells = h5["cells/table"]
        assert cells["class_label"].asstr()[:].tolist() == ["vimentin_ko", "ctrl", ""]
        assert cells["nls_status"].asstr()[:].tolist() == ["low", "high", ""]
        np.testing.assert_allclose(cells["nls_track_median_intensity"][:2], [11.0, 105.0])
        assert np.isnan(cells["nls_track_median_intensity"][2])
        assert cells["nls_track_pixel_count"][:].tolist() == [2, 2, 0]
        assert cells["nls_track_frame_count"][:].tolist() == [1, 1, 0]
        meta = h5["cells/measurements/nls_classification"].attrs
        assert meta["method"] == "otsu_track_median"
        assert meta["high_label"] == "ctrl"
        assert meta["low_label"] == "vimentin_ko"
        assert meta["classified_track_count"] == 2
        assert meta["nls_zavg_path"] == str(nls_path)
        assert meta["nucleus_tracked_labels_path"] == str(labels_path)


def test_patch_position_artifact_replaces_existing_nls_columns_on_rerun(tmp_path):
    h5_path = tmp_path / "position_analysis.h5"
    _write_minimal_position_h5(h5_path, [1, 2])
    nls_path = tmp_path / "NLS_zavg.tif"
    labels_path = tmp_path / "tracked_labels.tif"
    labels = np.asarray([[[1, 2]]], dtype=np.uint16)
    tifffile.imwrite(labels_path, labels)
    tifffile.imwrite(nls_path, np.asarray([[[10.0, 100.0]]], dtype=np.float32))
    patch_position_artifact_nls_classes(h5_path, nls_path, labels_path)
    tifffile.imwrite(nls_path, np.asarray([[[120.0, 10.0]]], dtype=np.float32))

    patch_position_artifact_nls_classes(h5_path, nls_path, labels_path)

    with h5py.File(h5_path, "r") as h5:
        cells = h5["cells/table"]
        assert cells["class_label"].asstr()[:].tolist() == ["ctrl", "vimentin_ko"]
        np.testing.assert_allclose(cells["nls_track_median_intensity"][:], [120.0, 10.0])


def test_patch_position_artifact_validates_before_mutating_h5(tmp_path):
    h5_path = tmp_path / "position_analysis.h5"
    _write_minimal_position_h5(h5_path, [1, 2])
    nls_path = tmp_path / "NLS_zavg.tif"
    labels_path = tmp_path / "tracked_labels.tif"
    tifffile.imwrite(nls_path, np.zeros((1, 2, 2), dtype=np.float32))
    tifffile.imwrite(labels_path, np.zeros((1, 3, 3), dtype=np.uint16))

    with pytest.raises(NLSClassificationError, match="shapes do not match"):
        patch_position_artifact_nls_classes(h5_path, nls_path, labels_path)

    with h5py.File(h5_path, "r") as h5:
        assert set(h5["cells/table"].keys()) == {"frame", "cell_id", "class_label"}
        assert h5["cells/table/class_label"].asstr()[:].tolist() == ["old", "old"]


def test_split_tracks_otsu_rejects_invalid_splits():
    with pytest.raises(NLSClassificationError, match="fewer than two"):
        split_tracks_otsu({1: 10.0})
    with pytest.raises(NLSClassificationError, match="identical"):
        split_tracks_otsu({1: 10.0, 2: 10.0})
