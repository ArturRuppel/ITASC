
from cellflow.correction.labels import (
    apply_gamma,
    clean_stranded_pixels,
    draw_cell_path,
    fill_label_holes,
    split_across,
)
import numpy as np

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


def test_clean_stranded_pixels_reports_filled_holes_and_border_islands():
    seg = np.ones((8, 8), dtype=np.uint32)
    seg[2:6, 2:6] = 2
    seg[3, 3] = 0
    seg[0, 7] = 2

    changed = clean_stranded_pixels(seg, min_size=4)

    assert changed >= 2
    assert seg[3, 3] != 0
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
