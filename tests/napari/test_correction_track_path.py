from __future__ import annotations

import numpy as np

from cellflow.napari._correction_track_path import (
    build_track_path_overlay,
)


def _viridis_endpoints():
    from matplotlib import colormaps

    cmap = colormaps["viridis"]
    return np.asarray(cmap(0.0), dtype=float), np.asarray(cmap(1.0), dtype=float)


def test_three_frame_track_returns_one_entry_per_occupied_frame():
    stack = np.zeros((3, 4, 4), dtype=np.uint32)
    stack[0, 0, 0] = 5
    stack[1, 1, 1] = 5
    stack[2, 2, 2] = 5

    overlay = build_track_path_overlay(stack, 5)

    assert overlay.frames == (0, 1, 2)
    assert overlay.colors.shape == (3, 4)
    assert overlay.centroids.shape == (3, 2)
    np.testing.assert_allclose(
        overlay.centroids, [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    )
    assert overlay.frame_number_labels() == ["0", "1", "2"]
    # Union covers every frame's mask.
    assert overlay.union_mask.sum() == 3
    assert overlay.union_mask[0, 0]
    assert overlay.union_mask[2, 2]


def test_colors_run_viridis_start_dark_to_finish_yellow():
    stack = np.zeros((3, 4, 4), dtype=np.uint32)
    stack[0, 0, 0] = 5
    stack[1, 1, 1] = 5
    stack[2, 2, 2] = 5

    overlay = build_track_path_overlay(stack, 5)

    dark, yellow = _viridis_endpoints()
    np.testing.assert_allclose(overlay.colors[0], dark)
    np.testing.assert_allclose(overlay.colors[-1], yellow)
    # Distinct endpoints -> the comet visibly fades start to finish.
    assert not np.allclose(overlay.colors[0], overlay.colors[-1])


def test_barely_moving_track_stacks_newest_frame_on_top():
    # Both frames paint the same pixel; the later frame must win.
    stack = np.zeros((2, 3, 3), dtype=np.uint32)
    stack[0, 1, 1] = 7
    stack[1, 1, 1] = 7

    overlay = build_track_path_overlay(stack, 7)

    np.testing.assert_allclose(overlay.overlay[1, 1], overlay.colors[-1])


def test_only_occupied_frames_are_kept():
    stack = np.zeros((4, 3, 3), dtype=np.uint32)
    stack[1, 0, 0] = 9  # gap at t=0, present at t=1 and t=3
    stack[3, 2, 2] = 9

    overlay = build_track_path_overlay(stack, 9)

    assert overlay.frames == (1, 3)
    assert overlay.frame_number_labels() == ["1", "3"]


def test_absent_track_returns_empty_overlay():
    stack = np.zeros((2, 3, 3), dtype=np.uint32)
    stack[0, 0, 0] = 1

    overlay = build_track_path_overlay(stack, 999)

    assert overlay.is_empty()
    assert overlay.frames == ()
    assert overlay.overlay.shape == (3, 3, 4)
    assert not overlay.union_mask.any()
    assert overlay.centroids.shape == (0, 2)


def test_overlay_is_transparent_outside_track_masks():
    stack = np.zeros((1, 3, 3), dtype=np.uint32)
    stack[0, 0, 0] = 3

    overlay = build_track_path_overlay(stack, 3)

    # Painted pixel non-zero, untouched pixels stay fully transparent (all-zero).
    assert overlay.overlay[0, 0].any()
    assert not overlay.overlay[2, 2].any()


def test_filled_blob_paints_only_its_outline():
    # A solid 3x3 mask sitting inside a larger plane: the border is painted,
    # the interior pixel stays transparent, but the union stays filled.
    stack = np.zeros((1, 5, 5), dtype=np.uint32)
    stack[0, 1:4, 1:4] = 2

    overlay = build_track_path_overlay(stack, 2)

    assert overlay.overlay[1, 1].any()      # corner of the blob (border)
    assert overlay.overlay[1, 2].any()      # edge of the blob (border)
    assert not overlay.overlay[2, 2].any()  # interior pixel stays transparent
    assert overlay.union_mask[2, 2]         # union is still filled
    assert overlay.union_mask.sum() == 9


def test_2d_plane_treated_as_single_frame():
    plane = np.zeros((3, 3), dtype=np.uint32)
    plane[1, 1] = 4

    overlay = build_track_path_overlay(plane, 4)

    assert overlay.frames == (0,)
    np.testing.assert_allclose(overlay.centroids, [[1.0, 1.0]])
