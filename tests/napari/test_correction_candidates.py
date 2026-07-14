"""Tests for the pure candidate-thumbnail builder."""
from __future__ import annotations

import numpy as np
import pytest

from itasc.napari.correction._correction_candidates import (
    CandidateSpec,
    CandidateStrip,
    build_candidate_strip,
)


def _frame(shape=(40, 40)) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random(shape).astype(np.float32)


def _square_mask(shape, y0, x0, side) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[y0 : y0 + side, x0 : x0 + side] = True
    return m


class TestBuildCandidateStrip:
    def test_empty_specs_gives_empty_strip(self):
        strip = build_candidate_strip(_frame(), [])
        assert isinstance(strip, CandidateStrip)
        assert strip.is_empty()
        assert strip.keys == ()

    def test_one_tile_per_candidate_in_order(self):
        shape = (40, 40)
        specs = [
            CandidateSpec(key=11, mask=_square_mask(shape, 5, 5, 6)),
            CandidateSpec(key=22, mask=_square_mask(shape, 20, 20, 4)),
        ]
        strip = build_candidate_strip(_frame(shape), specs)
        assert strip.keys == (11, 22)
        assert len(strip.tiles) == 2

    def test_tiles_are_uniform_square_rgb_uint8(self):
        shape = (40, 40)
        specs = [
            CandidateSpec(key=1, mask=_square_mask(shape, 5, 5, 8)),   # bigger
            CandidateSpec(key=2, mask=_square_mask(shape, 25, 25, 3)),
        ]
        strip = build_candidate_strip(_frame(shape), specs, margin=6)
        sizes = {(t.height, t.width) for t in strip.tiles}
        assert len(sizes) == 1  # one shared window size
        (h, w) = sizes.pop()
        assert h == w == 8 + 2 * 6  # max extent + 2*margin
        for t in strip.tiles:
            assert t.rgb.dtype == np.uint8
            assert t.rgb.shape == (h, w, 3)

    def test_area_and_default_caption(self):
        shape = (40, 40)
        strip = build_candidate_strip(
            _frame(shape), [CandidateSpec(key=7, mask=_square_mask(shape, 5, 5, 4))]
        )
        tile = strip.tiles[0]
        assert tile.area == 16
        assert tile.caption == "16 px"

    def test_explicit_caption_overrides_default(self):
        shape = (40, 40)
        strip = build_candidate_strip(
            _frame(shape),
            [CandidateSpec(key=7, mask=_square_mask(shape, 5, 5, 4), caption="iou 0.9")],
        )
        assert strip.tiles[0].caption == "iou 0.9"

    def test_empty_mask_is_skipped(self):
        shape = (40, 40)
        specs = [
            CandidateSpec(key=1, mask=np.zeros(shape, dtype=bool)),
            CandidateSpec(key=2, mask=_square_mask(shape, 5, 5, 4)),
        ]
        strip = build_candidate_strip(_frame(shape), specs)
        assert strip.keys == (2,)

    def test_outline_color_paints_into_tile(self):
        shape = (40, 40)
        strip = build_candidate_strip(
            _frame(shape),
            [CandidateSpec(key=1, mask=_square_mask(shape, 10, 10, 8))],
            colormap=None,
            outline_color=(1.0, 0.0, 0.0),
            spotlight_dim=0.0,  # outside the mask goes black, so red stands out
        )
        rgb = strip.tiles[0].rgb
        reds = (rgb[..., 0] > 200) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)
        assert reds.any()

    def test_mask_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_candidate_strip(
                _frame((40, 40)),
                [CandidateSpec(key=1, mask=np.ones((10, 10), dtype=bool))],
            )

    def test_non_2d_intensity_raises(self):
        with pytest.raises(ValueError):
            build_candidate_strip(
                np.zeros((3, 40, 40), dtype=np.float32),
                [CandidateSpec(key=1, mask=np.ones((40, 40), dtype=bool))],
            )
