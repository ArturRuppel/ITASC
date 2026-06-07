"""Unit tests for cellflow.cellpose.divergence_maps."""
from __future__ import annotations

import numpy as np
import pytest


def test_sigmoid_clamps_extreme_logits():
    from cellflow.cellpose.divergence_maps import sigmoid

    x = np.array([-1e6, 0.0, 1e6], dtype=np.float32)
    out = sigmoid(x)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-30)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0, abs=1e-30)


def test_foreground_from_prob_mean_matches_sigmoid_mean():
    from cellflow.cellpose.divergence_maps import foreground_from_prob

    rng = np.random.default_rng(0)
    prob = rng.normal(0, 3, size=(2, 4, 5, 6)).astype(np.float32)
    expected = (1.0 / (1.0 + np.exp(-np.clip(prob, -88, 88)))).mean(axis=1)
    out = foreground_from_prob(prob, reduction="mean")
    assert out.shape == (2, 5, 6)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)


def test_foreground_from_prob_max_matches_sigmoid_max():
    from cellflow.cellpose.divergence_maps import foreground_from_prob

    prob = np.array(
        [[[[-2.0, 2.0]], [[0.0, -3.0]]]],  # T=1, Z=2, Y=1, X=2
        dtype=np.float32,
    )
    out = foreground_from_prob(prob, reduction="max")
    expected = (1.0 / (1.0 + np.exp(-prob))).max(axis=1)
    assert out.shape == (1, 1, 2)
    np.testing.assert_allclose(out, expected)


def test_foreground_from_prob_rejects_unknown_reduction():
    from cellflow.cellpose.divergence_maps import foreground_from_prob

    prob = np.zeros((1, 1, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="reduction"):
        foreground_from_prob(prob, reduction="median")


def test_divergence_2d_linear_field_returns_constant():
    from cellflow.cellpose.divergence_maps import divergence_2d

    # dy(y, x) = 2y -> d_dy/d_y = 2; dx(y, x) = 3x -> d_dx/d_x = 3; sum = 5.
    y, x = np.mgrid[0:8, 0:8].astype(np.float32)
    flow = np.stack([2.0 * y, 3.0 * x], axis=0)
    div = divergence_2d(flow)
    assert div.shape == (8, 8)
    assert div.dtype == np.float32
    # Interior values should equal 5 exactly under central differences.
    np.testing.assert_allclose(div[1:-1, 1:-1], 5.0, atol=1e-5)


def test_divergence_2d_rejects_wrong_shape():
    from cellflow.cellpose.divergence_maps import divergence_2d

    with pytest.raises(ValueError, match=r"\(2, Y, X\)"):
        divergence_2d(np.zeros((3, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\(2, Y, X\)"):
        divergence_2d(np.zeros((4, 5), dtype=np.float32))


def test_contour_from_dp_skips_filters_when_off(monkeypatch):
    import cellflow.cellpose.divergence_maps as dm

    calls = {"median": 0, "gaussian": 0}

    def _no_median(*a, **kw):
        calls["median"] += 1
        return a[0]

    def _no_gauss(*a, **kw):
        calls["gaussian"] += 1
        return a[0]

    monkeypatch.setattr(dm, "median_filter", _no_median)
    monkeypatch.setattr(dm, "gaussian_filter", _no_gauss)

    # T=1, Z=1, channels=2, Y=4, X=4 -> flat field -> zero divergence.
    dp = np.zeros((1, 1, 2, 4, 4), dtype=np.float32)
    out = dm.contour_from_dp(
        dp, smoothing_sigma=0.0, median_radius=0, reduction="mean",
    )
    assert calls == {"median": 0, "gaussian": 0}
    assert out.shape == (1, 4, 4)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, 0.0)


def test_contour_from_dp_clips_negative_divergence():
    from cellflow.cellpose.divergence_maps import contour_from_dp

    # Construct a convergent field (div < 0 interior) -- all output should be 0.
    y, x = np.mgrid[0:6, 0:6].astype(np.float32)
    dy = -y  # d/dy = -1
    dx = -x  # d/dx = -1
    flow = np.stack([dy, dx], axis=0)
    dp = flow[np.newaxis, np.newaxis]  # (T=1, Z=1, 2, Y, X)
    out = contour_from_dp(
        dp, smoothing_sigma=0.0, median_radius=0, reduction="mean",
    )
    # Interior of negative-divergence field clips to 0.
    np.testing.assert_allclose(out[0, 1:-1, 1:-1], 0.0)


def test_contour_from_dp_invokes_filters_in_order(monkeypatch):
    import cellflow.cellpose.divergence_maps as dm

    order: list[str] = []

    def _median(a, size):
        order.append(f"median:{size}")
        return a

    def _gauss(a, sigma):
        order.append(f"gauss:{sigma}")
        return a

    monkeypatch.setattr(dm, "median_filter", _median)
    monkeypatch.setattr(dm, "gaussian_filter", _gauss)

    dp = np.zeros((1, 1, 2, 4, 4), dtype=np.float32)
    dm.contour_from_dp(dp, smoothing_sigma=1.5, median_radius=2, reduction="max")

    # Each z-slice runs median then gaussian, once per channel (dy, dx).
    assert order == [
        "median:5", "median:5",  # 2 * radius + 1 = 5, applied to dy, dx
        "gauss:1.5", "gauss:1.5",
    ]


def test_contour_from_dp_reduces_max_vs_mean():
    from cellflow.cellpose.divergence_maps import contour_from_dp

    # T=1, Z=2; z=0 has zero div, z=1 has +1 div.
    y, x = np.mgrid[0:6, 0:6].astype(np.float32)
    z0 = np.zeros((2, 6, 6), dtype=np.float32)
    # dy = y/2 -> 0.5; dx = x/2 -> 0.5; sum = 1.0
    z1 = np.stack([y * 0.5, x * 0.5], axis=0).astype(np.float32)
    dp = np.stack([z0, z1], axis=0)[np.newaxis]  # (1, 2, 2, 6, 6)

    mean_out = contour_from_dp(dp, smoothing_sigma=0.0, median_radius=0, reduction="mean")
    max_out = contour_from_dp(dp, smoothing_sigma=0.0, median_radius=0, reduction="max")
    # Interior: mean ~ (0 + 1)/2 = 0.5; max ~ 1.0.
    np.testing.assert_allclose(mean_out[0, 1:-1, 1:-1], 0.5, atol=1e-5)
    np.testing.assert_allclose(max_out[0, 1:-1, 1:-1], 1.0, atol=1e-5)


def test_build_divergence_maps_writes_and_reports(tmp_path):
    import tifffile
    from cellflow.cellpose.divergence_maps import (
        DivergenceMapsReport, build_divergence_maps,
    )

    rng = np.random.default_rng(0)
    prob = rng.normal(0, 2, size=(2, 3, 5, 6)).astype(np.float32)
    dp = rng.normal(0, 1, size=(2, 3, 2, 5, 6)).astype(np.float32)

    prob_path = tmp_path / "prob.tif"
    dp_path = tmp_path / "dp.tif"
    contours_out = tmp_path / "out_contours.tif"
    fg_out = tmp_path / "out_foreground.tif"
    tifffile.imwrite(str(prob_path), prob)
    tifffile.imwrite(str(dp_path), dp)

    calls: list[tuple[int, int, str]] = []
    report = build_divergence_maps(
        prob_path,
        dp_path,
        contours_out,
        fg_out,
        foreground_z_reduction="mean",
        contour_z_reduction="mean",
        smoothing_sigma=0.0,
        median_radius=0,
        progress_cb=lambda d, n, m: calls.append((d, n, m)),
    )

    assert isinstance(report, DivergenceMapsReport)
    assert report.frames == 2
    assert report.contours_path == contours_out
    assert report.foreground_path == fg_out

    fg = tifffile.imread(str(fg_out))
    assert fg.shape == (2, 5, 6)
    assert fg.dtype == np.float32
    assert 0.0 <= fg.min() and fg.max() <= 1.0
    contours = tifffile.imread(str(contours_out))
    assert contours.shape == (2, 5, 6)
    assert contours.dtype == np.float32
    assert contours.min() >= 0.0
    assert len(calls) >= 1
    assert calls[-1][0] == calls[-1][1]  # final report has done==total


def test_build_divergence_maps_reports_per_z_contour_progress(tmp_path):
    import tifffile
    from cellflow.cellpose.divergence_maps import build_divergence_maps

    prob = np.zeros((1, 3, 4, 4), dtype=np.float32)
    dp = np.zeros((1, 3, 2, 4, 4), dtype=np.float32)
    prob_path = tmp_path / "prob.tif"
    dp_path = tmp_path / "dp.tif"
    tifffile.imwrite(str(prob_path), prob)
    tifffile.imwrite(str(dp_path), dp)

    calls: list[tuple[int, int, str]] = []
    build_divergence_maps(
        prob_path,
        dp_path,
        tmp_path / "out_contours.tif",
        tmp_path / "out_foreground.tif",
        foreground_z_reduction="mean",
        contour_z_reduction="mean",
        smoothing_sigma=0.0,
        median_radius=0,
        progress_cb=lambda d, n, m: calls.append((d, n, m)),
    )

    compute_calls = [call for call in calls if call[1] == 6]
    assert compute_calls == [
        (1, 6, "Divergence maps: foreground frame 1/1"),
        (2, 6, "Divergence maps: contours frame 1/1 z 1/3"),
        (3, 6, "Divergence maps: contours frame 1/1 z 2/3"),
        (4, 6, "Divergence maps: contours frame 1/1 z 3/3"),
        (5, 6, "Divergence maps: writing contours"),
        (6, 6, "Divergence maps: writing foreground"),
    ]


def test_build_divergence_maps_respects_cancel(tmp_path):
    import tifffile
    from cellflow.cellpose.divergence_maps import build_divergence_maps
    from cellflow.core.cancellation import CancelledError

    prob = np.zeros((3, 1, 2, 2), dtype=np.float32)
    dp = np.zeros((3, 1, 2, 2, 2), dtype=np.float32)
    prob_path = tmp_path / "prob.tif"
    dp_path = tmp_path / "dp.tif"
    tifffile.imwrite(str(prob_path), prob)
    tifffile.imwrite(str(dp_path), dp)

    with pytest.raises(CancelledError):
        build_divergence_maps(
            prob_path,
            dp_path,
            tmp_path / "c.tif",
            tmp_path / "f.tif",
            foreground_z_reduction="mean",
            contour_z_reduction="mean",
            smoothing_sigma=0.0,
            median_radius=0,
            cancel=lambda: True,
        )


def test_build_divergence_maps_reads_frames_lazily(tmp_path, monkeypatch):
    """Progress must start before the whole stack is read (no 0% stall).

    Spies on the per-frame reader and asserts that when the first foreground
    progress callback fires, only frame 0 has been touched — i.e. inputs stream
    in instead of being decompressed whole up front.
    """
    import tifffile
    import cellflow.cellpose.divergence_maps as dm

    prob = np.zeros((4, 2, 5, 6), dtype=np.float32)
    dp = np.zeros((4, 2, 2, 5, 6), dtype=np.float32)
    prob_path = tmp_path / "prob.tif"
    dp_path = tmp_path / "dp.tif"
    tifffile.imwrite(str(prob_path), prob, compression="zlib")
    tifffile.imwrite(str(dp_path), dp, compression="zlib")

    reads: list[int] = []
    orig_frame = dm._LazyTiffStack.frame

    def _spy_frame(self, t):
        reads.append(int(t))
        return orig_frame(self, t)

    monkeypatch.setattr(dm._LazyTiffStack, "frame", _spy_frame)

    reads_at_first_progress: list[int] = []

    def _progress(done, total, msg):
        if not reads_at_first_progress:
            reads_at_first_progress.append(len(reads))

    dm.build_divergence_maps(
        prob_path, dp_path,
        tmp_path / "c.tif", tmp_path / "f.tif",
        foreground_z_reduction="mean", contour_z_reduction="mean",
        smoothing_sigma=0.0, median_radius=0,
        progress_cb=_progress,
    )

    # First callback is the frame-0 foreground step: exactly one frame read so
    # far, proving we did not eagerly load all 4 frames before reporting.
    assert reads_at_first_progress == [1]


def test_build_divergence_maps_validates_shapes(tmp_path):
    import tifffile
    from cellflow.cellpose.divergence_maps import build_divergence_maps

    prob = np.zeros((2, 3, 5, 6), dtype=np.float32)
    dp = np.zeros((3, 3, 2, 5, 6), dtype=np.float32)  # T mismatch
    prob_path = tmp_path / "prob.tif"
    dp_path = tmp_path / "dp.tif"
    tifffile.imwrite(str(prob_path), prob)
    tifffile.imwrite(str(dp_path), dp)

    with pytest.raises(ValueError, match="same frame count"):
        build_divergence_maps(
            prob_path, dp_path,
            tmp_path / "c.tif", tmp_path / "f.tif",
            foreground_z_reduction="mean", contour_z_reduction="mean",
            smoothing_sigma=0.0, median_radius=0,
        )


# ── singleton-axis round-trip (2D / 2D+t / 3D / 3D+t) ──────────────────────────

@pytest.mark.parametrize(
    "prob_shape, dp_shape, expected_frames",
    [
        ((1, 1, 8, 8), (1, 1, 2, 8, 8), 1),  # 2D single image
        ((3, 1, 8, 8), (3, 1, 2, 8, 8), 3),  # 2D+t  (T squeezes on disk — must stay T)
        ((1, 4, 8, 8), (1, 4, 2, 8, 8), 1),  # single 3D z-stack
        ((2, 4, 8, 8), (2, 4, 2, 8, 8), 2),  # 3D+t
    ],
)
def test_build_divergence_maps_preserves_frames_across_layouts(
    tmp_path, prob_shape, dp_shape, expected_frames
):
    """write_outputs records axis labels so singleton T/Z survive the TIFF
    round-trip; the builder must recover the right frame count (a 2D+t stack
    must not be misread as a single z-stack)."""
    import numpy as np
    import tifffile
    from cellflow.cellpose.cellpose_runner import write_outputs
    from cellflow.cellpose.divergence_maps import build_divergence_maps

    write_outputs(
        np.zeros(prob_shape, np.float32), np.zeros(dp_shape, np.float32),
        tmp_path, "nucleus",
    )
    report = build_divergence_maps(
        tmp_path / "nucleus_prob_3dt.tif",
        tmp_path / "nucleus_dp_3dt.tif",
        tmp_path / "nucleus_contours.tif",
        tmp_path / "nucleus_foreground.tif",
        foreground_z_reduction="mean",
        contour_z_reduction="mean",
        smoothing_sigma=0.0,
        median_radius=0,
    )
    assert report.frames == expected_frames
    fg = tifffile.imread(str(tmp_path / "nucleus_foreground.tif"))
    assert fg.shape == (expected_frames, 8, 8)
