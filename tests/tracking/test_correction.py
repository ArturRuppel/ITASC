
from cellflow.correction.labels import (
    clean_stranded_pixels,
    draw_cell_path,
    expand_label_to_foreground,
    fill_label_holes,
    fix_label_semiholes,
    split_across,
)
from cellflow.segmentation import apply_gamma
import numpy as np
import pytest


def test_correction_package_does_not_reexport_apply_gamma():
    import cellflow.correction as correction

    assert not hasattr(correction, "apply_gamma")


def test_gamma_identity():
    logits = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(apply_gamma(logits, 1.0), logits)

def test_gamma_values():
    logits = np.array([0.0]) # prob 0.5
    # gamma 2.0 -> prob 0.25 -> logit log(0.25/0.75) = log(1/3) = -1.0986
    corrected = apply_gamma(logits, 2.0)
    assert np.isclose(corrected, np.log(1/3))
    
    # gamma 0.5 -> prob 0.707 -> logit log(0.707/0.293) = 0.857
    corrected = apply_gamma(logits, 0.5)
    assert np.isclose(corrected, np.log(np.sqrt(0.5) / (1 - np.sqrt(0.5))))


def test_draw_cell_path_rejects_disconnected_extension():
    seg = np.zeros((12, 12), dtype=np.uint32)
    seg[1:3, 1:3] = 1
    before = seg.copy()

    positions = [
        (7, 7),
        (7, 10),
        (10, 10),
        (10, 7),
    ]

    ok = draw_cell_path(seg, positions, curlabel=1)

    assert ok is False
    assert np.array_equal(seg, before)


def test_draw_cell_path_extension_fills_enclosed_holes():
    seg = np.zeros((12, 12), dtype=np.uint32)
    seg[4:8, 4:8] = 1
    seg[5:7, 5:7] = 0

    positions = [
        (4, 7),
        (4, 9),
        (7, 9),
        (7, 7),
    ]

    ok = draw_cell_path(seg, positions, curlabel=1)

    assert ok is True
    assert seg[5, 5] == 1
    assert seg[0, 0] == 0


def test_split_across_uses_caller_supplied_fresh_label():
    stack = np.zeros((2, 12, 12), dtype=np.uint32)
    stack[0, 2:10, 2:10] = 5
    stack[1, 1:3, 1:3] = 6

    ok = split_across(stack[0], None, (3, 3), (8, 8), new_label=7)

    assert ok is True
    assert 7 in stack[0]
    assert 6 not in stack[0]


def test_draw_cell_path_uses_caller_supplied_fresh_label_for_new_cell():
    stack = np.zeros((2, 12, 12), dtype=np.uint32)
    stack[0, 1:3, 1:3] = 5
    stack[1, 8:10, 8:10] = 6

    ok = draw_cell_path(stack[0], [(5, 1), (5, 4), (8, 4), (8, 1)], new_label=7)

    assert ok is True
    assert 7 in stack[0]
    assert 6 not in stack[0]


def test_draw_cell_path_extension_preserves_protected_pixels():
    seg = np.zeros((12, 12), dtype=np.uint32)
    seg[3:6, 3:6] = 1
    seg[4:8, 6:9] = 9
    seg[7:10, 6:9] = 8
    protected_mask = seg == 9

    ok = draw_cell_path(
        seg,
        [(3, 5), (3, 10), (10, 10), (10, 5)],
        curlabel=1,
        protected_mask=protected_mask,
    )

    assert ok is True
    assert np.all(seg[protected_mask] == 9)
    assert np.any(seg[7:10, 6:9] == 1)


def test_clean_stranded_pixels_cleans_fragments_without_filling_background_holes():
    seg = np.ones((8, 8), dtype=np.uint32)
    seg[2:6, 2:6] = 2
    seg[3, 3] = 0
    seg[0, 7] = 2

    changed = clean_stranded_pixels(seg, min_size=4)

    assert changed == 1
    assert seg[3, 3] == 0
    assert seg[0, 7] != 2


def test_fill_label_holes_expands_cells_into_enclosed_gaps_by_radius():
    seg = np.zeros((12, 12), dtype=np.uint32)
    seg[1:11, 1:4] = 1
    seg[1:11, 8:11] = 2
    seg[1:3, 4:8] = 1
    seg[9:11, 4:8] = 2
    seg[3:9, 4:8] = 0

    result = fill_label_holes(seg, radius=1)

    assert np.any(result[3:9, 4:8] == 1)
    assert np.any(result[3:9, 4:8] == 2)
    assert np.any(result[3:9, 4:8] == 0)
    np.testing.assert_array_equal(result[0, :], np.zeros(12, dtype=np.uint32))


def test_fill_label_holes_zero_radius_is_noop():
    seg = np.ones((6, 6), dtype=np.uint32)
    seg[2:4, 2:4] = 0

    result = fill_label_holes(seg, radius=0)

    np.testing.assert_array_equal(result, seg)


def test_fill_label_holes_leaves_open_background_unchanged():
    seg = np.ones((8, 8), dtype=np.uint32)
    seg[2:6, 2:6] = 0
    seg[0:3, 3] = 0

    result = fill_label_holes(seg, radius=999)

    np.testing.assert_array_equal(result, seg)


def test_fix_label_semiholes_repairs_narrow_border_connected_gap():
    seg = np.ones((8, 8), dtype=np.uint32)
    seg[1:7, 1:7] = 2
    seg[2:6, 2:6] = 0
    seg[0, 3] = 0
    seg[1, 3] = 0

    result = fix_label_semiholes(seg, radius=999, max_opening=1)

    assert not np.any(result[2:6, 2:6] == 0)
    assert result[0, 3] != 0


def test_fix_label_semiholes_leaves_wide_border_opening_unchanged():
    seg = np.ones((8, 8), dtype=np.uint32)
    seg[1:7, 1:7] = 2
    seg[2:6, 2:6] = 0
    seg[0, 2:5] = 0
    seg[1, 3] = 0

    result = fix_label_semiholes(seg, radius=999, max_opening=1)

    np.testing.assert_array_equal(result, seg)


def test_fix_label_semiholes_zero_radius_or_opening_is_noop():
    seg = np.ones((8, 8), dtype=np.uint32)
    seg[1:7, 1:7] = 2
    seg[2:6, 2:6] = 0
    seg[0, 3] = 0
    seg[1, 3] = 0

    np.testing.assert_array_equal(
        fix_label_semiholes(seg, radius=0, max_opening=1),
        seg,
    )
    np.testing.assert_array_equal(
        fix_label_semiholes(seg, radius=999, max_opening=0),
        seg,
    )


def test_expand_label_to_foreground_adds_background_foreground_pixels():
    seg = np.zeros((7, 7), dtype=np.uint32)
    seg[3, 3] = 4
    foreground = np.zeros_like(seg, dtype=np.uint8)
    foreground[2:5, 2:5] = 1

    added = expand_label_to_foreground(seg, foreground, 4, max_distance=1)

    assert added == 4
    assert seg[3, 3] == 4
    assert seg[2, 3] == 4
    assert seg[3, 2] == 4
    assert seg[3, 4] == 4
    assert seg[4, 3] == 4
    assert seg[2, 2] == 0


def test_expand_label_to_foreground_respects_max_distance():
    seg = np.zeros((9, 9), dtype=np.uint32)
    seg[4, 4] = 2
    foreground = np.zeros_like(seg, dtype=np.uint8)
    foreground[1:8, 1:8] = 1

    added = expand_label_to_foreground(seg, foreground, 2, max_distance=2)

    assert added == 12
    assert seg[4, 6] == 2
    assert seg[4, 7] == 0


def test_expand_label_to_foreground_does_not_overwrite_neighboring_labels():
    seg = np.zeros((5, 7), dtype=np.uint32)
    seg[2, 2] = 1
    seg[2, 4] = 9
    foreground = np.ones_like(seg, dtype=np.uint8)

    expand_label_to_foreground(seg, foreground, 1, max_distance=3)

    assert seg[2, 4] == 9


def test_expand_label_to_foreground_does_not_cross_foreground_gaps():
    seg = np.zeros((5, 9), dtype=np.uint32)
    seg[2, 1] = 3
    foreground = np.zeros_like(seg, dtype=np.uint8)
    foreground[1:4, 0:3] = 1
    foreground[1:4, 5:8] = 1

    expand_label_to_foreground(seg, foreground, 3, max_distance=0)

    assert np.all(seg[1:4, 0:3][foreground[1:4, 0:3] > 0] == 3)
    assert not np.any(seg[1:4, 5:8] == 3)


def test_expand_label_to_foreground_zero_distance_is_unlimited():
    seg = np.zeros((5, 5), dtype=np.uint32)
    seg[2, 2] = 8
    foreground = np.ones_like(seg, dtype=np.uint8)

    added = expand_label_to_foreground(seg, foreground, 8, max_distance=0)

    assert added == 24
    assert np.all(seg == 8)


def test_expand_label_to_foreground_raises_on_shape_mismatch():
    seg = np.zeros((5, 5), dtype=np.uint32)
    foreground = np.zeros((5, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        expand_label_to_foreground(seg, foreground, 1, max_distance=1)


def test_expand_label_to_foreground_returns_zero_without_mutation_when_absent_or_disconnected():
    seg = np.zeros((6, 6), dtype=np.uint32)
    seg[1, 1] = 6
    foreground = np.zeros_like(seg, dtype=np.uint8)
    foreground[4:6, 4:6] = 1
    before = seg.copy()

    assert expand_label_to_foreground(seg, foreground, 7, max_distance=0) == 0
    np.testing.assert_array_equal(seg, before)
    assert expand_label_to_foreground(seg, foreground, 6, max_distance=0) == 0
    np.testing.assert_array_equal(seg, before)
