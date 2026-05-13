from __future__ import annotations

import numpy as np
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig


def test_foreground_scores_from_logits_sigmoid_and_z_average():
    from cellflow.tracking_ultrack.progressive_merge import foreground_scores_from_logits

    logits = np.array(
        [
            [
                [[0.0, 2.0], [-2.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ],
        dtype=np.float32,
    )

    scores = foreground_scores_from_logits(logits)
    expected = (1.0 / (1.0 + np.exp(-logits))).mean(axis=1)

    assert scores.shape == (1, 2, 2)
    assert scores.dtype == np.float32
    np.testing.assert_allclose(scores, expected, rtol=1e-6)
    assert not np.array_equal(scores, scores.astype(bool))


def test_contour_maps_from_masks_extracts_inner_boundaries():
    from cellflow.tracking_ultrack.progressive_merge import contour_maps_from_masks

    masks = np.zeros((1, 8, 8), dtype=np.uint16)
    masks[0, 2:6, 2:6] = 3

    contours = contour_maps_from_masks(masks)

    assert contours.shape == (1, 8, 8)
    assert contours.dtype == np.float32
    assert contours[0, 2:6, 2:6].sum() > 0
    assert contours[0, 3:5, 3:5].sum() == 0


def test_contour_maps_from_masks_max_projects_z_stack():
    from cellflow.tracking_ultrack.progressive_merge import contour_maps_from_masks

    masks = np.zeros((1, 2, 8, 8), dtype=np.uint16)
    masks[0, 1, 2:6, 2:6] = 3

    contours = contour_maps_from_masks(masks)

    assert contours.shape == (1, 8, 8)
    assert contours.sum() > 0


def test_write_progressive_inputs_creates_float32_tiffs(tmp_path):
    from cellflow.tracking_ultrack.progressive_merge import write_progressive_inputs

    prob = np.zeros((2, 2, 4, 5), dtype=np.float32)
    masks = np.zeros((2, 4, 5), dtype=np.uint16)
    masks[:, 1:3, 1:4] = 1
    prob_path = tmp_path / "prob.tif"
    masks_path = tmp_path / "masks.tif"
    tifffile.imwrite(prob_path, prob)
    tifffile.imwrite(masks_path, masks)

    foreground_path, contour_path = write_progressive_inputs(
        prob_path, masks_path, tmp_path / "inputs"
    )

    foreground = tifffile.imread(foreground_path)
    contours = tifffile.imread(contour_path)
    assert foreground.shape == (2, 4, 5)
    assert contours.shape == (2, 4, 5)
    assert foreground.dtype == np.float32
    assert contours.dtype == np.float32


def test_build_progressive_ultrack_database_forwards_arguments(monkeypatch, tmp_path):
    from cellflow.tracking_ultrack import progressive_merge

    calls = {}
    cfg = TrackingConfig(seg_foreground_threshold=0.3)

    def fake_build_ultrack_database(**kwargs):
        calls.update(kwargs)
        return type("Report", (), {"scored_nodes": 4})()

    monkeypatch.setattr(
        progressive_merge,
        "build_ultrack_database",
        fake_build_ultrack_database,
    )

    report = progressive_merge.build_progressive_ultrack_database(
        foreground_scores_path=tmp_path / "foreground_scores.tif",
        contour_maps_path=tmp_path / "contour_maps.tif",
        nucleus_prob_zavg_path=tmp_path / "nucleus.tif",
        working_dir=tmp_path / "work",
        cfg=cfg,
        validated_tracks={7: {0}},
        tracked_labels=np.zeros((1, 8, 8), dtype=np.uint32),
        use_validated=True,
    )

    assert report.scored_nodes == 4
    assert calls["contour_maps_path"] == tmp_path / "contour_maps.tif"
    assert calls["foreground_masks_path"] == tmp_path / "foreground_scores.tif"
    assert calls["nucleus_prob_zavg_path"] == tmp_path / "nucleus.tif"
    assert calls["working_dir"] == tmp_path / "work"
    assert calls["cfg"] is cfg
    assert calls["cfg"].seg_foreground_threshold == 0.3
    assert calls["validated_tracks"] == {7: {0}}
    assert calls["use_validated"] is True
