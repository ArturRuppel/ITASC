"""Rendering + click coverage for the candidate gallery panel (offscreen Qt)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

from cellflow.napari._correction_candidates import (  # noqa: E402
    CandidateSpec,
    build_candidate_strip,
)
from cellflow.napari._correction_candidate_panel import (  # noqa: E402
    CandidateGalleryPanel,
)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _strip(keys):
    shape = (40, 40)
    rng = np.random.default_rng(0)
    intensity = rng.random(shape).astype(np.float32)
    specs = []
    for i, key in enumerate(keys):
        mask = np.zeros(shape, dtype=bool)
        mask[5 + i : 11 + i, 5 + i : 11 + i] = True
        specs.append(CandidateSpec(key=key, mask=mask))
    return build_candidate_strip(intensity, specs)


def test_set_column_populates_tiles(_app):
    panel = CandidateGalleryPanel()
    panel.set_column(panel.SWAP, _strip([101, 102, 103]))
    tiles = panel.column(panel.SWAP).tiles()
    assert [t.key for t in tiles] == [101, 102, 103]
    # The other columns stay empty.
    assert panel.column(panel.EXTEND_BACKWARD).tiles() == []


def test_clicking_a_tile_emits_which_and_key(_app):
    panel = CandidateGalleryPanel()
    panel.set_column(panel.EXTEND_FORWARD, _strip([7, 9]))
    received: list[tuple[str, int]] = []
    panel.candidate_activated.connect(lambda which, key: received.append((which, key)))

    panel.column(panel.EXTEND_FORWARD).tiles()[1].clicked.emit(9)

    assert received == [(panel.EXTEND_FORWARD, 9)]


def test_empty_strip_shows_no_tiles(_app):
    panel = CandidateGalleryPanel()
    panel.set_column(panel.SWAP, _strip([1, 2]))
    panel.set_column(panel.SWAP, _strip([]))  # replace with empty
    assert panel.column(panel.SWAP).tiles() == []


def test_clear_empties_every_column(_app):
    panel = CandidateGalleryPanel()
    for which in (panel.EXTEND_BACKWARD, panel.SWAP, panel.EXTEND_FORWARD):
        panel.set_column(which, _strip([1, 2]))
    panel.clear()
    for which in (panel.EXTEND_BACKWARD, panel.SWAP, panel.EXTEND_FORWARD):
        assert panel.column(which).tiles() == []


def test_set_tile_size_resizes_thumbnails_and_clamps(_app):
    panel = CandidateGalleryPanel(tile_px=64)
    panel.set_column(panel.SWAP, _strip([1, 2]))

    panel.set_tile_size(96)
    assert panel._tile_px == 96
    tile = panel.column(panel.SWAP).tiles()[0]
    image = tile.layout().itemAt(0).widget()  # the thumbnail QLabel
    assert image.pixmap().height() == 96

    panel.set_tile_size(99999)
    assert panel._tile_px == 256  # _TILE_PX_MAX
    panel.set_tile_size(1)
    assert panel._tile_px == 20   # _TILE_PX_MIN


def test_blocks_stack_vertically_and_tiles_wrap(_app):
    from cellflow.napari._flow_layout import FlowLayout

    panel = CandidateGalleryPanel()
    cols = [
        panel.column(w)
        for w in (panel.EXTEND_BACKWARD, panel.SWAP, panel.EXTEND_FORWARD)
    ]
    # The three blocks are stacked top-to-bottom (ascending y), not side by side.
    ys = [c.mapTo(panel, c.rect().topLeft()).y() for c in cols]
    assert ys == sorted(ys) and len(set(ys)) == 3
    # Each block flows its thumbnails with the wrapping FlowLayout.
    assert isinstance(cols[0]._body_lay, FlowLayout)
