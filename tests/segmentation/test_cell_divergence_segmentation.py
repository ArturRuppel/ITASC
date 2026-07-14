"""Tests for the unary-only divergence cell-segmentation helper."""
from __future__ import annotations

import numpy as np
import pytest

from itasc.segmentation import (
    CellDivergenceParams,
    CellDivergenceResult,
    segment_cells_divergence,
)
from itasc.segmentation.cell_divergence_segmentation import (
    clean_and_smooth_contours,
)
from itasc.segmentation.cell_label_icm import (
    assemble_cost_field,
    balance_strength_to_weights,
)


def _make_inputs(T=3, Y=24, X=24, seed=0):
    rng = np.random.default_rng(seed)
    fg = np.clip(rng.normal(0.6, 0.1, (T, Y, X)), 0, 1).astype(np.float32)
    contours = np.abs(rng.normal(0, 1, (T, Y, X))).astype(np.float32)
    nuc = np.zeros((T, Y, X), np.uint32)
    nuc[:, 5:8, 5:8] = 1
    nuc[:, 16:19, 16:19] = 2
    return contours, fg, nuc


def test_full_stack_returns_all_intermediates_with_matching_shapes():
    contours, fg, nuc = _make_inputs()
    result = segment_cells_divergence(contours, fg, nuc, CellDivergenceParams())

    assert isinstance(result, CellDivergenceResult)
    for arr in (
        result.foreground_raw, result.foreground_clean,
        result.contours_raw, result.contours_clean,
        result.foreground_mask, result.cost_field, result.labels,
    ):
        assert arr.shape == contours.shape


def test_labels_are_nucleus_track_ids_and_seeds_keep_their_id():
    contours, fg, nuc = _make_inputs()
    result = segment_cells_divergence(contours, fg, nuc, CellDivergenceParams())

    assert set(np.unique(result.labels).tolist()) <= {0, 1, 2}
    # Each seed pixel keeps its own track id.
    assert np.all(result.labels[nuc == 1] == 1)
    assert np.all(result.labels[nuc == 2] == 2)


def test_contours_clean_is_normalized_to_unit_range():
    contours, fg, nuc = _make_inputs()
    result = segment_cells_divergence(contours, fg, nuc, CellDivergenceParams())
    assert result.contours_clean.min() >= 0.0
    assert result.contours_clean.max() <= 1.0 + 1e-6


def test_fg_strength_zero_is_a_no_op_on_foreground():
    contours, fg, nuc = _make_inputs()
    result = segment_cells_divergence(
        contours, fg, nuc, CellDivergenceParams(fg_strength=0.0)
    )
    # residual(strength=0) returns the clipped raw map → identical to raw input.
    np.testing.assert_allclose(result.foreground_clean, np.clip(fg, 0, 1))


def test_contour_threshold_zeros_sub_threshold_values():
    contours, fg, nuc = _make_inputs()
    high = segment_cells_divergence(
        contours, fg, nuc, CellDivergenceParams(contour_threshold=0.5)
    )
    assert np.all((high.contours_clean == 0) | (high.contours_clean >= 0.5))


def test_single_frame_mode_returns_2d_and_skips_temporal():
    contours, fg, nuc = _make_inputs()
    params = CellDivergenceParams(memory_tau=0.1)
    result = segment_cells_divergence(contours, fg, nuc, params, frame=1)

    assert result.labels.ndim == 2
    assert result.cost_field.ndim == 2
    assert result.labels.shape == contours.shape[1:]


def test_cost_field_matches_solver_assembly():
    contours, fg, nuc = _make_inputs()
    params = CellDivergenceParams()
    result = segment_cells_divergence(contours, fg, nuc, params, frame=0)

    alpha, gamma = balance_strength_to_weights(
        params.balance, params.feature_strength
    )
    expected = assemble_cost_field(
        result.contours_clean, result.foreground_mask,
        alpha, result.foreground_clean, gamma,
    )
    finite = np.isfinite(expected)
    np.testing.assert_allclose(result.cost_field[finite], expected[finite])
    assert np.array_equal(np.isfinite(result.cost_field), finite)


def test_foreground_mask_includes_nucleus_pixels():
    contours, fg, nuc = _make_inputs()
    # An impossibly high threshold removes all sigmoid foreground, leaving only
    # the nucleus seeds in the mask.
    result = segment_cells_divergence(
        contours, fg, nuc, CellDivergenceParams(fg_threshold=2.0)
    )
    assert np.all(result.foreground_mask[nuc > 0])


def test_clean_and_smooth_contours_matches_full_run_contours_clean():
    contours, fg, nuc = _make_inputs()
    params = CellDivergenceParams(memory_tau=0.1)

    smoothed = clean_and_smooth_contours(contours, params)
    full = segment_cells_divergence(contours, fg, nuc, params)

    assert smoothed.shape == contours.shape
    np.testing.assert_array_equal(smoothed, full.contours_clean)


def test_single_frame_override_reproduces_full_run_frame():
    # With temporal smoothing on, the per-frame preview can only match the full
    # run by reusing the whole-movie smoothed contours via the override.
    contours, fg, nuc = _make_inputs()
    params = CellDivergenceParams(memory_tau=0.1)

    full = segment_cells_divergence(contours, fg, nuc, params)
    smoothed = clean_and_smooth_contours(contours, params)

    t = 1
    single = segment_cells_divergence(
        contours, fg, nuc, params, frame=t,
        contours_clean_override=smoothed[t],
    )

    np.testing.assert_array_equal(single.contours_clean, full.contours_clean[t])
    np.testing.assert_array_equal(single.labels, full.labels[t])
    finite = np.isfinite(full.cost_field[t])
    np.testing.assert_allclose(single.cost_field[finite], full.cost_field[t][finite])


def test_single_frame_without_override_diverges_under_smoothing():
    # Sanity check that the override is doing real work: the naive per-frame
    # path (no whole-movie smoothing) need not equal the full run for the frame.
    contours, fg, nuc = _make_inputs()
    params = CellDivergenceParams(memory_tau=0.1)

    full = segment_cells_divergence(contours, fg, nuc, params)
    naive = segment_cells_divergence(contours, fg, nuc, params, frame=1)

    assert not np.array_equal(naive.contours_clean, full.contours_clean[1])


def test_override_shape_mismatch_raises():
    contours, fg, nuc = _make_inputs()
    bad = np.zeros((3, 3), np.float32)
    with pytest.raises(ValueError):
        segment_cells_divergence(
            contours, fg, nuc, CellDivergenceParams(), frame=0,
            contours_clean_override=bad,
        )


def test_override_ignored_for_full_stack_run():
    # Override is single-frame only; a full-stack run must ignore it.
    contours, fg, nuc = _make_inputs()
    params = CellDivergenceParams()
    baseline = segment_cells_divergence(contours, fg, nuc, params)
    with_override = segment_cells_divergence(
        contours, fg, nuc, params,
        contours_clean_override=np.zeros(contours.shape[1:], np.float32),
    )
    np.testing.assert_array_equal(
        baseline.contours_clean, with_override.contours_clean
    )


def test_progress_callback_receives_messages():
    contours, fg, nuc = _make_inputs()
    msgs: list[str] = []
    segment_cells_divergence(
        contours, fg, nuc, CellDivergenceParams(), progress_cb=msgs.append
    )
    assert msgs  # at least one status string emitted
