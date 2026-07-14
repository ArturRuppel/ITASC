"""Tests for itasc.cellpose.joint orchestration.

The model-backed steps (nucleus masks, cell flows) and the laptrack tracker are
monkeypatched at the seam, so the test runs without cellpose/laptrack and asserts
the composition: cell labels inherit the *tracked* nucleus ids, paired 1:1.
"""
from __future__ import annotations

import numpy as np

from itasc.cellpose import joint
from itasc.cellpose.cellpose_runner import CellParams, NucleusParams
from itasc.cellpose.flow_following import FlowFollowingParams


def test_cell_foreground_from_prob_uses_sigmoid_threshold():
    prob = np.array([[-10.0, 0.0, 10.0]], dtype=np.float32)
    fg = joint.cell_foreground_from_prob(prob, 0.5)
    # sigmoid(-10)≈0, sigmoid(0)=0.5 (not > 0.5), sigmoid(10)≈1.
    assert fg.tolist() == [[False, False, True]]


def _inward_flow(H, W, cy, cx):
    yi, xi = np.indices((H, W))
    dy = (cy - yi).astype(np.float32)
    dx = (cx - xi).astype(np.float32)
    norm = np.hypot(dy, dx)
    safe = np.where(norm > 0, norm, 1.0)
    return np.stack([dy / safe, dx / safe], axis=0).astype(np.float32)


def test_joint_segment_track_pairs_cell_ids_to_tracked_nucleus_ids(monkeypatch):
    T, Z, H, W = 2, 1, 20, 20
    nucleus_stack = np.zeros((T, Z, H, W), dtype=np.float32)
    cell_stack = np.zeros((T, Z, H, W), dtype=np.float32)

    # Nucleus masks: a single object centred at (10,10) each frame, raw label 1.
    nuc_masks = np.zeros((T, Z, H, W), dtype=np.int32)
    nuc_masks[:, 0, 10, 10] = 1
    monkeypatch.setattr(
        joint.native_masks, "run_nucleus_masks_stack",
        lambda stack, params, **kw: nuc_masks.copy(),
    )

    # Tracker assigns the moving nucleus a stable track id of 4 (1-based output).
    nuc_tracked = np.zeros((T, Z, H, W), dtype=np.int32)
    nuc_tracked[:, 0, 10, 10] = 4
    monkeypatch.setattr(
        joint.track_laptrack, "track_masks",
        lambda masks, **kw: nuc_tracked.copy(),
    )

    # Cell flows: high prob everywhere (all foreground), flow toward (10,10).
    # The joint path segments the cell body from BOTH channels together, so it
    # calls run_cell_stack_joint(cell_stack, nucleus_stack, ...); record the
    # stacks it receives to assert both channels reach the two-channel pass.
    prob = np.full((T, Z, H, W), 10.0, dtype=np.float32)
    dp = np.stack(
        [np.stack([_inward_flow(H, W, 10.0, 10.0)], axis=0) for _ in range(T)],
        axis=0,
    )  # (T, Z, 2, Y, X)
    seen = {}

    def _fake_joint_flows(cell, nucleus, params, **kw):
        seen["cell"] = cell
        seen["nucleus"] = nucleus
        return prob.copy(), dp.copy()

    monkeypatch.setattr(
        joint.cellpose_runner, "run_cell_stack_joint", _fake_joint_flows,
    )

    nuc_out, cell_out = joint.joint_segment_track(
        nucleus_stack, cell_stack,
        NucleusParams(do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0),
        CellParams(diameter=0.0, min_size=0, gamma=1.0),
        FlowFollowingParams(flow_weight=1.0, flow_step_scale=1.0, max_iterations=60),
    )

    assert nuc_out.shape == (T, Z, H, W)
    assert cell_out.shape == (T, Z, H, W)
    # Cell labels carry the tracked nucleus id, paired 1:1.
    assert set(np.unique(nuc_out)) == {0, 4}
    assert set(np.unique(cell_out)) == {4}  # all foreground assigned to nucleus 4
    # The two-channel cell pass received BOTH the cell and nucleus stacks.
    assert seen["cell"] is cell_stack
    assert seen["nucleus"] is nucleus_stack


def test_joint_validates_4d_inputs(monkeypatch):
    import pytest

    with pytest.raises(ValueError):
        joint.joint_segment_track(
            np.zeros((2, 6, 6)), np.zeros((2, 1, 6, 6)),
            NucleusParams(do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0),
            CellParams(diameter=0.0, min_size=0, gamma=1.0),
            FlowFollowingParams(),
        )
