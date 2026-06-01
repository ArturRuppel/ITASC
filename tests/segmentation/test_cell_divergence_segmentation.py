"""Tests for the unary-only divergence cell-segmentation helper."""
from __future__ import annotations

import numpy as np
import pytest

from cellflow.segmentation import (
    CellDivergenceParams,
    CellDivergenceResult,
    segment_cells_divergence,
)
from cellflow.segmentation.cell_label_icm import assemble_cost_field


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

    expected = assemble_cost_field(
        result.contours_clean, result.foreground_mask,
        params.alpha, result.foreground_clean, params.gamma,
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


def test_progress_callback_receives_messages():
    contours, fg, nuc = _make_inputs()
    msgs: list[str] = []
    segment_cells_divergence(
        contours, fg, nuc, CellDivergenceParams(), progress_cb=msgs.append
    )
    assert msgs  # at least one status string emitted
