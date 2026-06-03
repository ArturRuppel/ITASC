"""Tests for the film-strip view's numpy->QImage conversion.

Only ``rgb_to_qimage`` is exercised here (it is the one piece of non-trivial
view logic that does not need a laid-out widget). A QApplication is required for
QtGui, so the whole module skips cleanly if Qt cannot start headless.
"""

from __future__ import annotations

import numpy as np
import pytest

qtpy = pytest.importorskip("qtpy")
from qtpy.QtGui import QImage  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_rgb_to_qimage_roundtrips_shape_format_and_a_pixel(_qapp):
    from cellflow.napari._correction_film_strip import rgb_to_qimage

    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[1, 2] = (10, 20, 30)

    image = rgb_to_qimage(rgb)

    assert image.width() == 4
    assert image.height() == 3
    assert image.format() == QImage.Format_RGB888
    px = image.pixelColor(2, 1)
    assert (px.red(), px.green(), px.blue()) == (10, 20, 30)


def test_rgb_to_qimage_owns_its_buffer(_qapp):
    # The source array is deleted; a copied QImage must still be valid.
    from cellflow.napari._correction_film_strip import rgb_to_qimage

    rgb = np.full((2, 2, 3), 7, dtype=np.uint8)
    image = rgb_to_qimage(rgb)
    del rgb
    assert image.pixelColor(0, 0).red() == 7


def test_rgb_to_qimage_rejects_non_rgb(_qapp):
    from cellflow.napari._correction_film_strip import rgb_to_qimage

    with pytest.raises(ValueError):
        rgb_to_qimage(np.zeros((4, 4), dtype=np.uint8))


def test_panel_tile_size_param_drives_rendered_pixmap_height(_qapp):
    from cellflow.napari._correction_film_strip import TrackFilmStripPanel
    from cellflow.napari._correction_track_path import FilmStripTile, TrackFilmStrip

    strip = TrackFilmStrip(
        tiles=(FilmStripTile(frame=0, rgb=np.zeros((5, 5, 3), dtype=np.uint8)),)
    )
    panel = TrackFilmStripPanel(tile_px=64)
    panel.set_strip(strip)

    assert panel._frame_items[0].pixmap().height() == 64

    panel.set_tile_size(128)
    assert panel._tile_px == 128
    assert panel._frame_items[0].pixmap().height() == 128


def test_panel_tile_size_is_clamped(_qapp):
    from cellflow.napari._correction_film_strip import TrackFilmStripPanel

    panel = TrackFilmStripPanel()
    panel.set_tile_size(99999)
    assert panel._tile_px == 512  # _TILE_PX_MAX
    panel.set_tile_size(1)
    assert panel._tile_px == 20  # _TILE_PX_MIN


def test_tiles_flow_left_to_right_within_a_row(_qapp):
    # Time runs across a row: the second frame sits to the right of the first.
    from cellflow.napari._correction_film_strip import TrackFilmStripPanel
    from cellflow.napari._correction_track_path import FilmStripTile, TrackFilmStrip

    strip = TrackFilmStrip(
        tiles=(
            FilmStripTile(frame=0, rgb=np.zeros((5, 5, 3), np.uint8)),
            FilmStripTile(frame=1, rgb=np.zeros((5, 5, 3), np.uint8)),
        )
    )
    panel = TrackFilmStripPanel(tile_px=32)
    panel.set_strip(strip)

    r0 = panel._tile_rects[0]
    r1 = panel._tile_rects[1]
    assert r1.left() > r0.right() - 1   # frame 1 is right of frame 0
    assert r1.top() == r0.top()         # same row


def test_panel_highlights_current_frame_with_a_border(_qapp):
    from cellflow.napari._correction_film_strip import TrackFilmStripPanel
    from cellflow.napari._correction_track_path import FilmStripTile, TrackFilmStrip

    strip = TrackFilmStrip(
        tiles=(
            FilmStripTile(frame=2, rgb=np.zeros((5, 5, 3), dtype=np.uint8)),
            FilmStripTile(frame=5, rgb=np.zeros((5, 5, 3), dtype=np.uint8)),
        )
    )
    panel = TrackFilmStripPanel()
    panel.set_strip(strip)
    panel.set_current_frame(5)

    # The border item sits on the current frame's tile, and moves on change.
    assert panel._border_item is not None
    assert panel._border_item.rect() == panel._tile_rects[5]

    panel.set_current_frame(2)
    assert panel._border_item.rect() == panel._tile_rects[2]


def test_changing_frames_rebuilds_tiles(_qapp):
    from cellflow.napari._correction_film_strip import TrackFilmStripPanel
    from cellflow.napari._correction_track_path import FilmStripTile, TrackFilmStrip

    panel = TrackFilmStripPanel(tile_px=32)
    panel.set_strip(
        TrackFilmStrip(tiles=(FilmStripTile(frame=0, rgb=np.zeros((5, 5, 3), np.uint8)),))
    )
    first = panel._frame_items[0]

    panel.set_strip(
        TrackFilmStrip(
            tiles=(
                FilmStripTile(frame=3, rgb=np.zeros((5, 5, 3), np.uint8)),
                FilmStripTile(frame=4, rgb=np.zeros((5, 5, 3), np.uint8)),
            )
        )
    )
    assert set(panel._frame_items) == {3, 4}
    assert first is not panel._frame_items.get(3)


def test_panel_tooltip_notes_validated_and_anchored(_qapp):
    from cellflow.napari._correction_film_strip import TrackFilmStripPanel
    from cellflow.napari._correction_track_path import FilmStripTile, TrackFilmStrip

    strip = TrackFilmStrip(
        tiles=(
            FilmStripTile(
                frame=0, rgb=np.zeros((5, 5, 3), dtype=np.uint8),
                validated=True, anchored=True,
            ),
        )
    )
    panel = TrackFilmStripPanel()
    panel.set_strip(strip)

    tip = panel._frame_items[0].toolTip()
    assert "validated" in tip and "anchored" in tip
