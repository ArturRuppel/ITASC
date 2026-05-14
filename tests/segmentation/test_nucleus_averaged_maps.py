from __future__ import annotations

import numpy as np
import tifffile


def test_build_nucleus_averaged_maps_writes_contours_and_foreground_scores(
    tmp_path, monkeypatch
):
    import cellflow.segmentation.nucleus_segmentation as ns

    prob = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    dp = np.zeros((2, 3, 2, 4, 5), dtype=np.float32)
    prob_path = tmp_path / "nucleus_prob_3dt.tif"
    dp_path = tmp_path / "nucleus_dp_3dt.tif"
    contours_path = tmp_path / "contours.tif"
    scores_path = tmp_path / "foreground_scores.tif"
    tifffile.imwrite(prob_path, prob)
    tifffile.imwrite(dp_path, dp)
    calls = []

    def fake_consensus(prob_3d, dp_3d, thresholds, *, gamma, flow_threshold):
        calls.append((prob_3d.copy(), dp_3d.copy(), tuple(thresholds), gamma, flow_threshold))
        value = float(prob_3d[0, 0, 0])
        return (
            np.full(prob_3d.shape[1:], value, dtype=np.float32),
            np.full(prob_3d.shape[1:], value + 0.5, dtype=np.float32),
        )

    monkeypatch.setattr(ns, "build_consensus_boundary", fake_consensus)

    report = ns.build_nucleus_averaged_maps(
        prob_path,
        dp_path,
        contours_path,
        scores_path,
        cellprob_thresholds=[-2.0, -1.0, 0.0],
        z_indices=[1, 2],
    )

    assert report.frames == 2
    assert report.z_indices == (1, 2)
    assert report.cellprob_thresholds == (-2.0, -1.0, 0.0)
    assert len(calls) == 2
    assert calls[0][0].shape == (2, 4, 5)
    np.testing.assert_array_equal(calls[0][0], prob[0, 1:3])
    np.testing.assert_array_equal(calls[1][1], dp[1, 1:3])
    assert calls[0][2] == (-2.0, -1.0, 0.0)
    assert calls[0][3] == 1.0
    assert calls[0][4] == 0.0

    contours = tifffile.imread(contours_path)
    scores = tifffile.imread(scores_path)
    assert contours.shape == (2, 4, 5)
    assert scores.shape == (2, 4, 5)
    assert contours.dtype == np.float32
    assert scores.dtype == np.float32
    np.testing.assert_allclose(contours[0], np.full((4, 5), prob[0, 1, 0, 0]))
    np.testing.assert_allclose(scores[1], np.full((4, 5), prob[1, 1, 0, 0] + 0.5))


def test_build_nucleus_averaged_maps_rejects_empty_z_selection(tmp_path):
    from cellflow.segmentation.nucleus_segmentation import build_nucleus_averaged_maps

    tifffile.imwrite(tmp_path / "prob.tif", np.zeros((1, 2, 3, 4), dtype=np.float32))
    tifffile.imwrite(tmp_path / "dp.tif", np.zeros((1, 2, 2, 3, 4), dtype=np.float32))

    try:
        build_nucleus_averaged_maps(
            tmp_path / "prob.tif",
            tmp_path / "dp.tif",
            tmp_path / "contours.tif",
            tmp_path / "foreground_scores.tif",
            cellprob_thresholds=[0.0],
            z_indices=[],
        )
    except ValueError as exc:
        assert "z_indices" in str(exc)
    else:
        raise AssertionError("expected empty z selection to fail")
