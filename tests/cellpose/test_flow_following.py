"""Tests for itasc.cellpose.flow_following (Qt-free, numpy/scipy only).

Covers the integrator capture behaviour, foreground masking, label preservation,
the bounded orphan drop, and the movie orchestration (cell ids inherit nucleus
ids). The numba kernel is lazily compiled; these tests exercise it through the
public dispatch (so they pass with or without numba installed).
"""
from __future__ import annotations

import numpy as np

from itasc.cellpose import flow_following as ff


def _inward_flow(H: int, W: int, cy: float, cx: float) -> np.ndarray:
    """Unit flow vectors (2, H, W) pointing each pixel toward (cy, cx)."""
    yi, xi = np.indices((H, W))
    dy = (cy - yi).astype(np.float32)
    dx = (cx - xi).astype(np.float32)
    norm = np.hypot(dy, dx)
    safe = np.where(norm > 0, norm, 1.0)
    return np.stack([dy / safe, dx / safe], axis=0).astype(np.float32)


def test_params_defaults():
    p = ff.FlowFollowingParams()
    assert p.fg_threshold == 0.5
    assert p.flow_weight == 0.5
    assert p.flow_step_scale == 0.2
    assert p.max_iterations == 100
    assert p.shell_width == 5.0
    assert p.max_assign_radius == 30.0


def test_single_nucleus_captures_all_foreground():
    """Flow pointing at the one nucleus → every fg pixel inherits its id."""
    H = W = 24
    labels = np.zeros((H, W), dtype=np.int32)
    labels[12, 12] = 7
    foreground = np.ones((H, W), dtype=bool)
    dp = _inward_flow(H, W, 12.0, 12.0)

    out = ff.flow_follow_frame(
        foreground, dp, labels,
        ff.FlowFollowingParams(flow_weight=1.0, flow_step_scale=1.0, max_iterations=80),
    )
    assert set(np.unique(out)) == {7}


def test_two_nuclei_split_by_flow():
    """Pixels flow to their own nucleus; both labels survive, partitioning fg."""
    H, W = 16, 32
    labels = np.zeros((H, W), dtype=np.int32)
    labels[8, 8] = 3
    labels[8, 24] = 9
    foreground = np.ones((H, W), dtype=bool)
    # Left half flows to (8,8); right half flows to (8,24).
    dp = np.zeros((2, H, W), dtype=np.float32)
    dp[:, :, :16] = _inward_flow(H, W, 8.0, 8.0)[:, :, :16]
    dp[:, :, 16:] = _inward_flow(H, W, 8.0, 24.0)[:, :, 16:]

    out = ff.flow_follow_frame(
        foreground, dp, labels,
        ff.FlowFollowingParams(flow_weight=1.0, flow_step_scale=1.0, max_iterations=80),
    )
    assert set(np.unique(out)) == {0, 3, 9} or set(np.unique(out)) == {3, 9}
    # Each nucleus owns its own column region.
    assert out[8, 8] == 3 and out[8, 24] == 9
    assert (out[:, :16] != 9).all()
    assert (out[:, 16:] != 3).all()


def test_background_pixels_never_assigned():
    """Only foreground is labelled; non-foreground stays 0 (except nuclei)."""
    H = W = 12
    labels = np.zeros((H, W), dtype=np.int32)
    labels[6, 6] = 4
    foreground = np.zeros((H, W), dtype=bool)
    foreground[6, 6] = True  # only the nucleus pixel is foreground
    dp = np.zeros((2, H, W), dtype=np.float32)

    out = ff.flow_follow_frame(foreground, dp, labels, ff.FlowFollowingParams())
    assert out[6, 6] == 4
    bg = np.ones((H, W), dtype=bool)
    bg[6, 6] = False
    assert (out[bg] == 0).all()


def test_orphans_beyond_radius_are_dropped():
    """Foreground far from any nucleus, with no flow, stays background."""
    H = W = 60
    labels = np.zeros((H, W), dtype=np.int32)
    labels[5, 5] = 2
    foreground = np.zeros((H, W), dtype=bool)
    foreground[5, 5] = True
    foreground[55, 55] = True  # isolated, ~70px from the nucleus
    dp = np.zeros((2, H, W), dtype=np.float32)  # no flow → no displacement

    out = ff.flow_follow_frame(
        foreground, dp, labels,
        ff.FlowFollowingParams(flow_weight=0.0, max_assign_radius=10.0),
    )
    assert out[5, 5] == 2
    assert out[55, 55] == 0  # orphan dropped, not force-assigned


def test_empty_inputs_return_background():
    H = W = 8
    labels = np.zeros((H, W), dtype=np.int32)
    foreground = np.ones((H, W), dtype=bool)
    dp = np.zeros((2, H, W), dtype=np.float32)
    # No nuclei → all background.
    out = ff.flow_follow_frame(foreground, dp, labels, ff.FlowFollowingParams())
    assert (out == 0).all()


def test_movie_inherits_nucleus_ids_each_frame():
    T, H, W = 3, 20, 20
    foreground = np.ones((T, H, W), dtype=bool)
    labels = np.zeros((T, H, W), dtype=np.int32)
    labels[:, 10, 10] = 5  # same nucleus id across time
    dp = np.stack([_inward_flow(H, W, 10.0, 10.0) for _ in range(T)], axis=0)

    out = ff.flow_follow_movie(
        foreground, dp, labels,
        ff.FlowFollowingParams(flow_weight=1.0, flow_step_scale=1.0, max_iterations=60),
    )
    assert out.shape == (T, H, W)
    for t in range(T):
        assert set(np.unique(out[t])) == {5}


def test_movie_validates_shapes():
    import pytest

    foreground = np.ones((2, 6, 6), dtype=bool)
    labels = np.ones((2, 6, 6), dtype=np.int32)
    bad_dp = np.zeros((2, 3, 6, 6), dtype=np.float32)  # 3 != 2 flow components
    with pytest.raises(ValueError):
        ff.flow_follow_movie(foreground, bad_dp, labels, ff.FlowFollowingParams())
