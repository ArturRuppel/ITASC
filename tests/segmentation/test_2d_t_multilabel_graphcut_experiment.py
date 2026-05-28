import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).parent.parent.parent
    / "scripts"
    / "experiment_cell_2d_t_multilabel_graphcut.py"
)

pytestmark = pytest.mark.skipif(
    not _SCRIPT.exists(),
    reason="archived exploratory graphcut script is outside the maintained test surface",
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "experiment_cell_2d_t_multilabel_graphcut", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_combine_unaries_adds_weighted_geodesic_and_flow_costs_without_mutating_inputs():
    mod = _load_module()
    geodesic = {
        (0, 1): np.array([[1.0, 2.0]], dtype=np.float32),
        (0, 2): np.array([[3.0, mod._INF]], dtype=np.float32),
    }
    flow = {
        (0, 1): np.array([[10.0, 20.0]], dtype=np.float32),
        (0, 2): np.array([[30.0, 40.0]], dtype=np.float32),
    }

    combined = mod._combine_unaries(
        geodesic, flow, lambda_geodesic=0.0, lambda_flow=0.25
    )

    np.testing.assert_allclose(combined[(0, 1)], [[2.5, 5.0]])
    assert combined[(0, 2)][0, 0] == 7.5
    assert combined[(0, 2)][0, 1] == mod._INF
    np.testing.assert_allclose(geodesic[(0, 1)], [[1.0, 2.0]])
    np.testing.assert_allclose(flow[(0, 1)], [[10.0, 20.0]])


def test_tiny_flip_threshold_stops_after_small_nonzero_round():
    mod = _load_module()

    assert not mod._should_stop_after_round(total_flips=100, min_flips=100)
    assert mod._should_stop_after_round(total_flips=99, min_flips=100)


def test_worse_round_energy_triggers_revert_with_tolerance():
    mod = _load_module()

    assert mod._round_energy_is_worse(current_energy=10.001, best_energy=10.0)
    assert not mod._round_energy_is_worse(current_energy=10.0, best_energy=10.0)
    assert not mod._round_energy_is_worse(current_energy=9.999, best_energy=10.0)


def test_connectivity_filter_rejects_flip_that_splits_source_label():
    mod = _load_module()
    labels = np.zeros((1, 5, 5), dtype=np.uint32)
    labels[0, 2, 1:4] = 1
    labels[0, 1, 2] = 2
    flip_mask = np.zeros_like(labels, dtype=bool)
    flip_mask[0, 2, 2] = True

    filtered = mod._filter_connectivity_preserving_flips(
        current_labels=labels,
        flip_mask=flip_mask,
        alpha=2,
    )

    assert not filtered[0, 2, 2]


def test_connectivity_filter_allows_leaf_flip_from_source_label():
    mod = _load_module()
    labels = np.zeros((1, 5, 5), dtype=np.uint32)
    labels[0, 2, 1:4] = 1
    labels[0, 1, 2] = 2
    flip_mask = np.zeros_like(labels, dtype=bool)
    flip_mask[0, 2, 1] = True

    filtered = mod._filter_connectivity_preserving_flips(
        current_labels=labels,
        flip_mask=flip_mask,
        alpha=2,
    )

    assert filtered[0, 2, 1]


def test_foreground_inverse_boundary_signal_inverts_and_clips_scores():
    mod = _load_module()
    contours = np.array([[[0.2, 0.8]]], dtype=np.float32)
    foreground_scores = np.array([[[-0.25, 1.25]]], dtype=np.float32)

    signal = mod._prepare_boundary_signal(
        contours,
        foreground_scores,
        mode="foreground_inverse",
    )

    np.testing.assert_allclose(signal, [[[1.0, 0.0]]])
    assert signal.dtype == np.float32


def test_build_nucleus_pixels_caps_samples_and_preserves_shape():
    mod = _load_module()
    T, Y, X = 2, 8, 8
    nuc_tracks = np.zeros((T, Y, X), dtype=np.uint32)
    # Label 1: large nucleus (>max_px pixels at frame 0)
    nuc_tracks[0, :, :] = 1   # 64 pixels
    # Label 2: small nucleus at frame 1
    nuc_tracks[1, 2, 3] = 2
    label_ids = np.array([1, 2], dtype=np.uint32)

    nuc_px, nuc_cnt = mod._build_nucleus_pixels(nuc_tracks, label_ids, max_px=16)

    assert nuc_px.shape == (2, T, 16, 2)
    assert nuc_cnt[0, 0] == 16  # capped at max_px
    assert nuc_cnt[1, 1] == 1   # single pixel
    assert nuc_cnt[0, 1] == 0   # label 1 dead at frame 1
    assert nuc_cnt[1, 0] == 0   # label 2 dead at frame 0


def test_normalize_flow_unary_applies_per_frame_median():
    mod = _load_module()
    T, Y, X, K = 1, 2, 2, 2
    # All foreground, both labels alive
    fg = np.ones((T, Y, X), dtype=bool)
    nuc_cnt = np.array([[1], [1]], dtype=np.int32)  # (K=2, T=1) — both alive at frame 0
    # Raw distances: label 0 gets [2, 4, 6, 8], label 1 gets [1, 3, 5, 7]
    raw = np.array([[[[2., 1.], [4., 3.]], [[6., 5.], [8., 7.]]]], dtype=np.float32)

    result = mod._normalize_flow_unary(raw, fg, nuc_cnt, cap_px=100.0)

    # Median of all values [1,2,3,4,5,6,7,8] = 4.5
    med = np.median([1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_allclose(result[0, 0, 0, 0], 2.0 / med, rtol=1e-5)
    np.testing.assert_allclose(result[0, 1, 1, 1], 7.0 / med, rtol=1e-5)


def test_icm_solver_smoke():
    """ICM path runs end-to-end on a tiny crop without error."""
    import subprocess, sys
    result = subprocess.run(
        [
            sys.executable, str(_SCRIPT),
            "--crop", "0", "3", "0", "128", "0", "128",
            "--solver", "icm",
            "--n-iters", "1",
            "--lambda-s", "1.0", "--lambda-t", "0.5",
            "--no-unary-cache",
            "--timestamp", "test-icm-smoke",
            "--overwrite",
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ICM done" in result.stdout
