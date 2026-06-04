from __future__ import annotations

import numpy as np
import pytest

from cellflow.napari._correction_track_path import (
    build_all_tracks_data,
    build_track_film_strip,
)


# --- all-tracks data --------------------------------------------------------


def test_build_all_tracks_data_groups_rows_by_track():
    stack = np.zeros((3, 6, 6), dtype=np.uint32)
    stack[0, 0, 0] = 5
    stack[1, 1, 1] = 5
    stack[2, 2, 2] = 5
    stack[0, 4, 4] = 9  # a second, single-frame track

    data, props, row_index = build_all_tracks_data(stack)

    # One row per (track, occupied frame): 3 for track 5, 1 for track 9.
    assert data.shape == (4, 4)
    assert sorted(row_index) == [5, 9]
    # Rows are [track_id, t, y, x] and time-ascending within a track.
    track5 = data[row_index[5]]
    np.testing.assert_array_equal(track5[:, 0], [5, 5, 5])
    np.testing.assert_array_equal(track5[:, 1], [0, 1, 2])
    np.testing.assert_allclose(track5[:, 2:], [[0, 0], [1, 1], [2, 2]])
    # Every property array aligns with data row-for-row.
    for key in ("track_id", "time"):
        assert len(props[key]) == len(data)


def test_time_property_is_normalised_per_track():
    stack = np.zeros((3, 4, 4), dtype=np.uint32)
    stack[0, 0, 0] = 5
    stack[1, 1, 1] = 5
    stack[2, 2, 2] = 5

    _, props, row_index = build_all_tracks_data(stack)

    time5 = props["time"][row_index[5]]
    np.testing.assert_allclose(time5, [0.0, 0.5, 1.0])


def test_single_frame_track_gets_zero_time():
    stack = np.zeros((1, 4, 4), dtype=np.uint32)
    stack[0, 1, 1] = 7

    _, props, row_index = build_all_tracks_data(stack)

    np.testing.assert_array_equal(props["time"][row_index[7]], [0.0])


def test_build_all_tracks_data_empty_stack():
    data, props, row_index = build_all_tracks_data(np.zeros((2, 3, 3), dtype=np.uint32))

    assert data.shape == (0, 4)
    assert row_index == {}
    assert all(len(props[key]) == 0 for key in ("track_id", "time"))


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
