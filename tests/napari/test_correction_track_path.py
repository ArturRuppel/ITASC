from __future__ import annotations

import numpy as np
import pytest

from cellflow.napari._correction_track_path import (
    build_track_film_strip,
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


def test_overlay_is_transparent_away_from_the_path():
    stack = np.zeros((1, 11, 11), dtype=np.uint32)
    stack[0, 0, 0] = 3

    overlay = build_track_path_overlay(stack, 3, thickness=2)

    # The trajectory point is painted; pixels well away from it stay transparent.
    assert overlay.overlay[0, 0].any()
    assert not overlay.overlay[10, 10].any()


def test_path_only_draws_the_track_not_its_mask():
    # A solid 3x3 mask: the overlay draws the trajectory (a thick dot at the
    # centroid), not the nucleus mask — but union_mask still tracks the footprint.
    stack = np.zeros((1, 9, 9), dtype=np.uint32)
    stack[0, 1:4, 1:4] = 2  # centroid at (2, 2)

    overlay = build_track_path_overlay(stack, 2, thickness=2)

    assert overlay.overlay[2, 2].any()   # the centroid (the track) is painted
    assert not overlay.overlay[8, 8].any()
    assert overlay.union_mask.sum() == 9  # union still covers the whole mask


def test_path_bridges_centroids_across_frames():
    # Cell present in frames 0 and 2 (gap at 1), far apart on the same row. The
    # trajectory line connects them, painting pixels that lie on no mask.
    stack = np.zeros((3, 5, 9), dtype=np.uint32)
    stack[0, 2, 0] = 6   # centroid (2, 0)
    stack[2, 2, 8] = 6   # centroid (2, 8)

    overlay = build_track_path_overlay(stack, 6, thickness=2)

    assert overlay.frames == (0, 2)
    assert overlay.overlay[2, 4].any()    # midpoint of the bridge, not a mask pixel
    assert overlay.union_mask.sum() == 2  # only the two single-pixel masks


def test_2d_plane_treated_as_single_frame():
    plane = np.zeros((3, 3), dtype=np.uint32)
    plane[1, 1] = 4

    overlay = build_track_path_overlay(plane, 4)

    assert overlay.frames == (0,)
    np.testing.assert_allclose(overlay.centroids, [[1.0, 1.0]])


# --- film strip -------------------------------------------------------------


def _film_stacks():
    # A 5x5 nucleus (bright center) that drifts across frames on a dim
    # background, with room to stay centered in every tile. 5x5 keeps a real
    # interior even after the outline is thickened.
    tracked = np.zeros((3, 20, 20), dtype=np.uint32)
    intensity = np.full((3, 20, 20), 1000, dtype=np.uint16)
    for t, (y, x) in enumerate([(6, 6), (10, 10), (13, 13)]):
        tracked[t, y - 2 : y + 3, x - 2 : x + 3] = 8
        intensity[t, y - 2 : y + 3, x - 2 : x + 3] = 4000
        intensity[t, y, x] = 6000  # brighter core so normalization is non-degenerate
    return tracked, intensity


def _const_cmap(value):
    def _map(v):
        return np.broadcast_to(np.asarray(value, dtype=float), v.shape + (3,))

    return _map


def _red_cmap(v):
    out = np.zeros(v.shape + (3,), dtype=float)
    out[..., 0] = v  # intensity -> red channel only
    return out


def test_film_strip_tiles_are_uniform_square_rgb():
    tracked, intensity = _film_stacks()

    strip = build_track_film_strip(tracked, intensity, 8, margin=2)

    assert strip.frames == (0, 1, 2)
    shapes = {tile.rgb.shape for tile in strip.tiles}
    assert len(shapes) == 1
    h, w, c = strip.tiles[0].rgb.shape
    assert h == w and c == 3  # square tiles
    assert strip.tiles[0].rgb.dtype == np.uint8


def test_film_strip_nucleus_is_centered_in_each_tile():
    tracked, intensity = _film_stacks()

    strip = build_track_film_strip(
        tracked, intensity, 8, margin=2, colormap=_red_cmap, spotlight_dilation=0
    )

    # The bright nucleus core lands at the tile center for every frame, even as
    # the nucleus moves in source coordinates.
    for tile in strip.tiles:
        size = tile.rgb.shape[0]
        center = tile.rgb[size // 2, size // 2]
        assert center[0] > 200  # red core (normalized ~1) at the center


def test_film_strip_spotlight_dims_outside_the_nucleus():
    tracked, intensity = _film_stacks()

    strip = build_track_film_strip(
        tracked, intensity, 8, margin=4, colormap=_const_cmap([0.8, 0.8, 0.8])
    )

    tile = strip.tiles[0].rgb
    size = tile.shape[0]
    corner = int(tile[0, 0, 0])              # outside the spotlight -> dimmed
    center = int(tile[size // 2, size // 2, 0])  # nucleus -> full brightness
    assert corner < center
    assert corner == round(0.8 * 0.35 * 255)  # dimmed by spotlight_dim


def test_film_strip_outline_is_drawn_in_the_frames_viridis_color():
    from matplotlib import colormaps

    tracked, intensity = _film_stacks()
    strip = build_track_film_strip(tracked, intensity, 8, margin=2)

    last_color = np.asarray(colormaps["viridis"](1.0))[:3]
    expected = (last_color * 255.0).round().astype(np.uint8)
    tile = strip.tiles[-1].rgb
    assert np.any(np.all(tile == expected, axis=-1))  # outline pixels present


def test_film_strip_thicker_outline_paints_more_pixels():
    tracked, intensity = _film_stacks()
    color = np.asarray(__import__("matplotlib").colormaps["viridis"](1.0))[:3]
    expected = (color * 255.0).round().astype(np.uint8)

    thin = build_track_film_strip(
        tracked, intensity, 8, margin=2, outline_width=1
    ).tiles[-1].rgb
    thick = build_track_film_strip(
        tracked, intensity, 8, margin=2, outline_width=3
    ).tiles[-1].rgb

    n_thin = np.all(thin == expected, axis=-1).sum()
    n_thick = np.all(thick == expected, axis=-1).sum()
    assert n_thick > n_thin


def test_film_strip_outline_uses_label_color_when_given():
    # An explicit outline_color (the label layer's colour) overrides viridis.
    tracked, intensity = _film_stacks()
    strip = build_track_film_strip(
        tracked, intensity, 8, margin=2, outline_color=(1.0, 0.0, 0.0)
    )

    expected = np.array([255, 0, 0], dtype=np.uint8)
    for tile in strip.tiles:
        assert np.any(np.all(tile.rgb == expected, axis=-1))


def test_film_strip_border_coincides_with_spotlight_edge():
    # The coloured border must sit *between* the bright spotlight and the dim
    # background: no bright (undimmed) pixel may touch a dimmed pixel, otherwise
    # a bright ring would read as a second (white) border outside the contour.
    tracked, intensity = _film_stacks()
    strip = build_track_film_strip(
        tracked,
        intensity,
        8,
        margin=4,
        colormap=_const_cmap([0.8, 0.8, 0.8]),
        outline_color=(1.0, 0.0, 0.0),
    )

    tile = strip.tiles[0].rgb
    bright = round(0.8 * 255)
    dim = round(0.8 * 0.35 * 255)
    bright_mask = np.all(tile == bright, axis=-1)
    dim_mask = np.all(tile == dim, axis=-1)

    touches_dim = np.zeros_like(bright_mask)
    touches_dim[1:, :] |= dim_mask[:-1, :]
    touches_dim[:-1, :] |= dim_mask[1:, :]
    touches_dim[:, 1:] |= dim_mask[:, :-1]
    touches_dim[:, :-1] |= dim_mask[:, 1:]

    assert bright_mask.any() and dim_mask.any()
    assert not np.any(bright_mask & touches_dim)


def test_film_strip_flags_validated_and_anchored_frames():
    tracked, intensity = _film_stacks()
    strip = build_track_film_strip(
        tracked, intensity, 8, margin=2,
        validated_frames={0}, anchored_frames={2},
    )

    by_frame = {tile.frame: tile for tile in strip.tiles}
    assert by_frame[0].validated and not by_frame[0].anchored
    assert by_frame[2].anchored and not by_frame[2].validated
    assert not by_frame[1].validated and not by_frame[1].anchored


def test_film_strip_frames_restricts_the_scan():
    # Handing the known occupied frames keeps only those tiles (and lets the
    # canvas skip re-scanning every empty frame per track).
    tracked, intensity = _film_stacks()

    strip = build_track_film_strip(tracked, intensity, 8, margin=2, frames=[0, 2])

    assert strip.frames == (0, 2)


def test_film_strip_frames_ignores_out_of_range_indices():
    tracked, intensity = _film_stacks()

    strip = build_track_film_strip(tracked, intensity, 8, margin=2, frames=[0, 99, -1])

    assert strip.frames == (0,)


def test_film_strip_absent_track_is_empty():
    tracked, intensity = _film_stacks()

    strip = build_track_film_strip(tracked, intensity, 999)

    assert strip.is_empty()
    assert strip.frames == ()


def test_film_strip_rejects_shape_mismatch():
    tracked = np.zeros((2, 5, 5), dtype=np.uint32)
    intensity = np.zeros((2, 6, 6), dtype=np.uint16)

    with pytest.raises(ValueError):
        build_track_film_strip(tracked, intensity, 1)
