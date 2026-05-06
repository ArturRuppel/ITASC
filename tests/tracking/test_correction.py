
from cellflow.correction.labels import apply_gamma, draw_cell_path, split_across
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
