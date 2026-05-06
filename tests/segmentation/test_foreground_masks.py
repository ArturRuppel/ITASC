import numpy as np
import pytest

from cellflow import segmentation


def test_probability_scores_average_z_logits_then_apply_sigmoid_and_gamma():
    logits = np.array(
        [
            [
                [[0.0, 2.0], [-2.0, 4.0]],
                [[2.0, 0.0], [2.0, -4.0]],
            ],
            [
                [[-4.0, 0.0], [0.0, 4.0]],
                [[4.0, 0.0], [0.0, -4.0]],
            ],
        ],
        dtype=np.float32,
    )

    score = segmentation.foreground_score_stack(logits, "probability", gamma=2.0)

    z_avg = logits.mean(axis=1)
    expected = (1.0 / (1.0 + np.exp(-z_avg))) ** 2.0
    np.testing.assert_allclose(score, expected.astype(np.float32), rtol=1e-6)
    assert score.shape == (2, 2, 2)


def test_probability_single_volume_returns_yx_uint8_mask():
    logits = np.array(
        [
            [[-2.0, 0.0], [2.0, 4.0]],
            [[-2.0, 0.0], [2.0, -4.0]],
        ],
        dtype=np.float32,
    )

    mask = segmentation.foreground_mask_stack(logits, "probability", threshold=0.5)

    expected_score = 1.0 / (1.0 + np.exp(-logits.mean(axis=0)))
    expected = (expected_score >= 0.5).astype(np.uint8)
    np.testing.assert_array_equal(mask, expected)
    assert mask.shape == (2, 2)
    assert mask.dtype == np.uint8


def test_flow_dp_vector_scores_use_magnitude_z_average_and_per_timepoint_normalization():
    dp = np.zeros((2, 2, 2, 2, 2), dtype=np.float32)
    dp[0, :, 0] = np.array([[[3.0, 0.0], [0.0, 6.0]], [[1.0, 0.0], [0.0, 2.0]]])
    dp[0, :, 1] = np.array([[[4.0, 0.0], [0.0, 8.0]], [[0.0, 0.0], [0.0, 0.0]]])
    dp[1, :, 0] = 5.0
    dp[1, :, 1] = 0.0

    score = segmentation.foreground_score_stack(dp, "flow_dp", gamma=0.5)

    expected_t0 = np.sqrt(np.array([[0.5, 0.0], [0.0, 1.0]], dtype=np.float32))
    expected_t1 = np.zeros((2, 2), dtype=np.float32)
    expected = np.stack([expected_t0, expected_t1], axis=0)
    np.testing.assert_allclose(score, expected, rtol=1e-6)
    assert score.shape == (2, 2, 2)


def test_flow_dp_accepts_single_volume_channels_last_and_thresholds_to_uint8():
    dp = np.zeros((2, 3, 4, 2), dtype=np.float32)
    dp[0, :, :, 0] = [[0.0, 3.0, 6.0, 0.0], [0.0, 3.0, 6.0, 0.0], [0.0, 3.0, 6.0, 0.0]]
    dp[0, :, :, 1] = [[0.0, 4.0, 8.0, 0.0], [0.0, 4.0, 8.0, 0.0], [0.0, 4.0, 8.0, 0.0]]
    dp[1, :, :, 0] = [[0.0, 1.0, 2.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 1.0, 2.0, 0.0]]

    mask = segmentation.foreground_mask_stack(dp, "flow_dp", threshold=0.5)

    expected = np.array(
        [[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(mask, expected)
    assert mask.shape == (3, 4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"source": "unknown"}, "source"),
        ({"source": "probability", "gamma": 0.0}, "gamma"),
        ({"source": "probability", "threshold": 1.1}, "threshold"),
    ],
)
def test_foreground_mask_validation(kwargs, match):
    data = np.zeros((1, 2, 2), dtype=np.float32)
    source = kwargs.pop("source")

    with pytest.raises(ValueError, match=match):
        segmentation.foreground_mask_stack(data, source, **kwargs)
