"""Tests for itasc.cellpose.track_laptrack.

The centroid/relabel helpers are tested directly (numpy + scikit-image only).
The laptrack orchestration is tested by monkeypatching the isolated
``_run_laptrack`` seam, plus one real integration test gated on laptrack being
installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from itasc.cellpose import track_laptrack as tl


def test_stitch_z_merges_overlapping_planes_into_one_object():
    # (T=1, Z=3): a 4x4 block at the same (y,x) in all 3 z-planes, each plane
    # independently labelled (10, 11, 12). They overlap fully → one object.
    masks = np.zeros((1, 3, 6, 6), dtype=np.int32)
    masks[0, 0, 1:5, 1:5] = 10
    masks[0, 1, 1:5, 1:5] = 11
    masks[0, 2, 1:5, 1:5] = 12
    out = tl.stitch_z(masks, iou_threshold=0.25)
    block = out[0, :, 1:5, 1:5]
    assert (block == block[0, 0, 0]).all() and block[0, 0, 0] != 0


def test_stitch_z_keeps_non_overlapping_objects_separate():
    masks = np.zeros((1, 2, 8, 8), dtype=np.int32)
    masks[0, 0, 0:3, 0:3] = 5
    masks[0, 1, 5:8, 5:8] = 7
    out = tl.stitch_z(masks, iou_threshold=0.25)
    assert len({int(v) for v in np.unique(out) if v != 0}) == 2


def test_stitch_z_single_slice_is_noop():
    masks = np.zeros((2, 1, 4, 4), dtype=np.int32)
    masks[0, 0, 1, 1] = 3
    assert np.array_equal(tl.stitch_z(masks), masks)


def test_track_axiswise_single_slice_tracks_time(monkeypatch):
    masks = np.zeros((2, 1, 6, 6), dtype=np.int32)
    masks[0, 0, 2:4, 0:2] = 1
    masks[1, 0, 2:4, 2:4] = 1

    def _fake_run(df, *, max_distance, max_frame_gap):
        df = df.copy()
        df["track_id"] = 0
        return df

    monkeypatch.setattr(tl, "_run_laptrack", _fake_run)
    tracked = tl.track_axiswise(masks, max_distance=10.0, max_frame_gap=0)
    assert tracked.shape == (2, 1, 6, 6)
    assert set(np.unique(tracked)) == {0, 1}


def test_track_axiswise_stitches_z_before_tracking(monkeypatch):
    # (T=2, Z=2): an object spans both z-planes per frame, planes labelled
    # separately. Stitch must merge them so the tracker sees ONE object/frame.
    masks = np.zeros((2, 2, 8, 8), dtype=np.int32)
    masks[0, 0, 1:4, 1:4] = 1
    masks[0, 1, 1:4, 1:4] = 2
    masks[1, 0, 1:4, 1:4] = 3
    masks[1, 1, 1:4, 1:4] = 4

    seen = {}

    def _fake_run(df, *, max_distance, max_frame_gap):
        for f in df["frame"].unique():
            seen[int(f)] = int(df[df["frame"] == f]["label"].nunique())
        df = df.copy()
        df["track_id"] = 0
        return df

    monkeypatch.setattr(tl, "_run_laptrack", _fake_run)
    tracked = tl.track_axiswise(masks, max_distance=10.0)
    assert seen == {0: 1, 1: 1}  # one stitched object per frame, not two per-z
    assert set(np.unique(tracked)) == {0, 1}


def _two_frame_stack():
    """(T=2, Z=1, Y=6, X=6) with one object that moves +2 in x between frames."""
    masks = np.zeros((2, 1, 6, 6), dtype=np.int32)
    masks[0, 0, 2:4, 0:2] = 1
    masks[1, 0, 2:4, 2:4] = 1
    return masks


def test_build_track_dataframe_columns_and_rows():
    masks = _two_frame_stack()
    df = tl.build_track_dataframe(masks)
    assert list(df.columns) == ["frame", "label", "z", "y", "x"]
    assert len(df) == 2
    assert set(df["frame"]) == {0, 1}
    # object moved +2 in x between the two frames.
    x0 = df[df.frame == 0]["x"].iloc[0]
    x1 = df[df.frame == 1]["x"].iloc[0]
    assert x1 - x0 == pytest.approx(2.0)
    # single-slice input -> z constant 0.
    assert set(df["z"]) == {0.0}


def test_build_track_dataframe_skips_empty_frames():
    masks = np.zeros((3, 1, 4, 4), dtype=np.int32)
    masks[1, 0, 1:3, 1:3] = 1
    df = tl.build_track_dataframe(masks)
    assert list(df["frame"]) == [1]


def test_build_track_dataframe_rejects_bad_ndim():
    with pytest.raises(ValueError):
        tl.build_track_dataframe(np.zeros((4, 4), dtype=np.int32))


def test_relabel_by_tracks_assigns_consistent_ids():
    masks = _two_frame_stack()
    # both frames' object id 1 belong to track 0 -> output label 1 everywhere.
    track_of = {(0, 1): 0, (1, 1): 0}
    out = tl.relabel_by_tracks(masks, track_of)
    assert out.dtype == np.int32
    assert set(np.unique(out)) == {0, 1}
    assert (out[0, 0, 2:4, 0:2] == 1).all()
    assert (out[1, 0, 2:4, 2:4] == 1).all()


def test_relabel_by_tracks_separate_tracks():
    masks = _two_frame_stack()
    track_of = {(0, 1): 0, (1, 1): 1}  # different tracks -> labels 1 and 2
    out = tl.relabel_by_tracks(masks, track_of)
    assert out[0].max() == 1
    assert out[1].max() == 2


def test_relabel_by_tracks_drops_unmapped_labels():
    masks = _two_frame_stack()
    out = tl.relabel_by_tracks(masks, {(0, 1): 0})  # frame 1 unmapped
    assert out[0].max() == 1
    assert out[1].max() == 0


def test_track_masks_orchestration_with_stubbed_laptrack(monkeypatch):
    """track_masks should build df -> run linker -> relabel, all wired up."""
    masks = _two_frame_stack()

    def _fake_run(df, *, max_distance, max_frame_gap):
        # link both detections into a single track id 0.
        df = df.copy()
        df["track_id"] = 0
        return df

    monkeypatch.setattr(tl, "_run_laptrack", _fake_run)
    out = tl.track_masks(masks, max_distance=10.0)
    assert out.shape == masks.shape
    assert set(np.unique(out)) == {0, 1}  # one consistent track


def test_track_masks_empty_returns_zeros(monkeypatch):
    masks = np.zeros((2, 1, 4, 4), dtype=np.int32)
    # _run_laptrack must not be called for empty input.
    monkeypatch.setattr(
        tl, "_run_laptrack",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
    )
    out = tl.track_masks(masks)
    assert out.shape == masks.shape
    assert not out.any()


def test_track_masks_real_laptrack_links_moving_object():
    """Integration: real laptrack links a single moving object across frames."""
    pytest.importorskip("laptrack")
    masks = _two_frame_stack()
    out = tl.track_masks(masks, max_distance=10.0)
    # one object, present in both frames, should carry a single id.
    assert set(np.unique(out)) == {0, 1}
    assert out[0].max() == 1
    assert out[1].max() == 1
