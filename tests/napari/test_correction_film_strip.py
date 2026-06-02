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
