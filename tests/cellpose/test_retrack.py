"""Tests for itasc.cellpose.retrack — the DB-free standalone retracker.

Pure numpy/scipy/skimage, so everything is exercised directly with no viewer.
"""
from __future__ import annotations

import numpy as np
import pytest

from itasc.cellpose import retrack as rt


def _block(frame, label, y0, x0, size=4):
    frame[y0:y0 + size, x0:x0 + size] = label


def test_retrack_frame_matches_moving_cell_keeps_ref_id():
    # One cell that moved a little; target id differs from the reference id.
    ref = np.zeros((20, 20), dtype=np.int32)
    tgt = np.zeros((20, 20), dtype=np.int32)
    _block(ref, 7, 2, 2)
    _block(tgt, 99, 3, 3)  # same cell, shifted by (1, 1), wrong id
    out = rt.retrack_frame(ref, tgt, max_dist_px=50.0)
    # The single target cell adopts the reference id.
    assert set(np.unique(out)) == {0, 7}
    assert (out[3:7, 3:7] == 7).all()


def test_retrack_frame_unmatched_target_gets_fresh_id():
    # Target cell is too far from the only reference cell → fresh id, not 0.
    ref = np.zeros((40, 40), dtype=np.int32)
    tgt = np.zeros((40, 40), dtype=np.int32)
    _block(ref, 5, 0, 0)
    _block(tgt, 5, 34, 34)  # 48 px away, beyond max_dist
    out = rt.retrack_frame(ref, tgt, max_dist_px=10.0)
    ids = set(int(v) for v in np.unique(out) if v != 0)
    assert ids and 5 not in ids  # relabelled to a fresh id above existing ids
    assert max(ids) > 5


def test_retrack_frame_two_cells_assigned_to_nearest():
    ref = np.zeros((30, 30), dtype=np.int32)
    tgt = np.zeros((30, 30), dtype=np.int32)
    _block(ref, 1, 2, 2)
    _block(ref, 2, 2, 20)
    # Targets swapped ids relative to position; retrack should fix by geometry.
    _block(tgt, 2, 3, 3)    # near ref id 1
    _block(tgt, 1, 3, 21)   # near ref id 2
    out = rt.retrack_frame(ref, tgt, max_dist_px=50.0)
    assert int(out[3, 3]) == 1
    assert int(out[3, 21]) == 2


def test_retrack_frame_no_collision_when_one_unmatched():
    # Two targets, one reference. The matched target keeps the ref id; the
    # unmatched one must NOT reuse that id.
    ref = np.zeros((40, 40), dtype=np.int32)
    tgt = np.zeros((40, 40), dtype=np.int32)
    _block(ref, 3, 2, 2)
    _block(tgt, 10, 2, 2)    # matches ref 3
    _block(tgt, 11, 30, 30)  # far → fresh id
    out = rt.retrack_frame(ref, tgt, max_dist_px=10.0)
    ids = sorted(int(v) for v in np.unique(out) if v != 0)
    assert len(ids) == 2
    assert 3 in ids
    assert ids.count(3) == 1 and len(set(ids)) == 2


def test_retrack_frame_empty_target_returns_zeros():
    ref = np.zeros((10, 10), dtype=np.int32)
    _block(ref, 1, 1, 1)
    out = rt.retrack_frame(ref, np.zeros((10, 10), dtype=np.int32))
    assert not out.any()


def _moving_stack():
    """(T=3, Y=30, X=30): one cell drifting +2 x per frame, with garbled ids."""
    stack = np.zeros((3, 30, 30), dtype=np.int32)
    _block(stack[0], 1, 10, 2)
    _block(stack[1], 50, 10, 4)
    _block(stack[2], 77, 10, 6)
    return stack


def test_retrack_stack_forward_propagates_id():
    stack = _moving_stack()
    out = rt.retrack_stack(stack, start_frame=0, direction="forward", max_dist_px=50.0)
    # Anchor frame unchanged; later frames adopt the anchor's id 1.
    assert int(out[0, 10, 2]) == 1
    assert int(out[1, 10, 4]) == 1
    assert int(out[2, 10, 6]) == 1


def test_retrack_stack_backward_propagates_id():
    stack = _moving_stack()
    out = rt.retrack_stack(stack, start_frame=2, direction="backward", max_dist_px=50.0)
    assert int(out[2, 10, 6]) == 77  # anchor kept
    assert int(out[1, 10, 4]) == 77
    assert int(out[0, 10, 2]) == 77


def test_retrack_stack_does_not_mutate_input():
    stack = _moving_stack()
    snapshot = stack.copy()
    rt.retrack_stack(stack, start_frame=0, direction="forward")
    assert np.array_equal(stack, snapshot)


def test_retrack_stack_rejects_non_timefirst():
    with pytest.raises(ValueError):
        rt.retrack_stack(np.zeros((5, 5), dtype=np.int32), start_frame=0, direction="forward")
    with pytest.raises(ValueError):
        rt.retrack_stack(np.zeros((1, 5, 5), dtype=np.int32), start_frame=0, direction="forward")


def test_retrack_stack_rejects_bad_direction():
    with pytest.raises(ValueError):
        rt.retrack_stack(_moving_stack(), start_frame=0, direction="sideways")
