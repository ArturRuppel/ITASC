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

    def _first_pixmap_height():
        # tiles are laid out in order in the wrapping flow layout
        cell = panel._row.itemAt(0).widget()
        thumb = cell.layout().itemAt(0).widget()
        return thumb.pixmap().height()

    assert _first_pixmap_height() == 64

    panel.set_tile_size(128)
    assert panel._tile_px == 128
    assert _first_pixmap_height() == 128


def test_panel_tile_size_is_clamped(_qapp):
    from cellflow.napari._correction_film_strip import TrackFilmStripPanel

    panel = TrackFilmStripPanel()
    panel.set_tile_size(99999)
    assert panel._tile_px == 512  # _TILE_PX_MAX
    panel.set_tile_size(1)
    assert panel._tile_px == 20  # _TILE_PX_MIN


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

    # The current frame's cell gets a visible white border; others stay clear.
    assert "#ffffff" in panel._tile_cells[5][0].styleSheet()
    assert "transparent" in panel._tile_cells[2][0].styleSheet()

    panel.set_current_frame(2)
    assert "#ffffff" in panel._tile_cells[2][0].styleSheet()
    assert "transparent" in panel._tile_cells[5][0].styleSheet()


def test_resetting_same_frames_reuses_tile_widgets(_qapp):
    # Swapping a single track rebuilds the strip on every keypress. As long as
    # the frame set is unchanged, the tiles must be repainted in place rather
    # than destroyed and recreated -- that per-keypress widget churn is what
    # flashed transient popups in the docked strip. Same frames => same widgets.
    from cellflow.napari._correction_film_strip import TrackFilmStripPanel
    from cellflow.napari._correction_track_path import FilmStripTile, TrackFilmStrip

    def _strip(pixel):
        return TrackFilmStrip(
            tiles=(
                FilmStripTile(frame=0, rgb=np.full((5, 5, 3), pixel, dtype=np.uint8)),
                FilmStripTile(frame=1, rgb=np.full((5, 5, 3), pixel, dtype=np.uint8)),
            )
        )

    panel = TrackFilmStripPanel(tile_px=32)
    panel.set_strip(_strip(10))
    cells_before = [panel._row.itemAt(i).widget() for i in range(panel._row.count())]
    thumbs_before = list(panel._thumbs)

    # A swap changes the pixels but keeps the same frames.
    panel.set_strip(_strip(200))

    cells_after = [panel._row.itemAt(i).widget() for i in range(panel._row.count())]
    assert cells_after == cells_before  # no widgets torn down / recreated
    assert panel._thumbs == thumbs_before
    # ...and the repaint actually took effect.
    assert panel._thumbs[0].pixmap().height() == 32


def test_changing_frames_rebuilds_tiles(_qapp):
    from cellflow.napari._correction_film_strip import TrackFilmStripPanel
    from cellflow.napari._correction_track_path import FilmStripTile, TrackFilmStrip

    panel = TrackFilmStripPanel(tile_px=32)
    panel.set_strip(
        TrackFilmStrip(tiles=(FilmStripTile(frame=0, rgb=np.zeros((5, 5, 3), np.uint8)),))
    )
    first = panel._thumbs[0]

    # Selecting a track with a different frame set must rebuild the row.
    panel.set_strip(
        TrackFilmStrip(
            tiles=(
                FilmStripTile(frame=3, rgb=np.zeros((5, 5, 3), np.uint8)),
                FilmStripTile(frame=4, rgb=np.zeros((5, 5, 3), np.uint8)),
            )
        )
    )
    assert len(panel._thumbs) == 2
    assert first not in panel._thumbs


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

    cell = panel._tile_cells[0][0]
    thumb = cell.layout().itemAt(0).widget()
    tip = thumb.toolTip()
    assert "validated" in tip and "anchored" in tip
