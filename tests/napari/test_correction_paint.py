"""Pure label-painting helper shared by the extend paths."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cellflow.napari._correction_paint import paint_assignments


def _assignment(cell_id, mask):
    return SimpleNamespace(cell_id=cell_id, mask_2d=mask)


def _mask(shape, *cells):
    m = np.zeros(shape, dtype=bool)
    for y, x in cells:
        m[y, x] = True
    return m


def test_greedy_overwrites_outside_protected():
    frame = np.array([[0, 5], [5, 9]], dtype=int)
    protected = _mask((2, 2), (1, 1))  # cell 9 is protected
    assignment = _assignment(7, _mask((2, 2), (0, 0), (0, 1), (1, 1)))
    changed = paint_assignments(frame, (assignment,), protected, greedy=True)
    # 7 painted where allowed; the protected (1,1) keeps 9; old 5 at (0,1) lost.
    assert frame.tolist() == [[7, 7], [5, 9]]
    assert changed == {5, 7}  # 9 unchanged (protected), 0->7 discarded as a key only


def test_non_greedy_only_fills_background():
    frame = np.array([[0, 5], [0, 0]], dtype=int)
    protected = _mask((2, 2))  # nothing protected
    assignment = _assignment(7, _mask((2, 2), (0, 0), (0, 1), (1, 0)))
    changed = paint_assignments(frame, (assignment,), protected, greedy=False)
    # Only background (0) pixels fill; the existing 5 at (0,1) survives.
    assert frame.tolist() == [[7, 5], [7, 0]]
    assert changed == {7}


def test_existing_id_is_cleared_before_repaint():
    # The assignment's own id is wiped everywhere first, so a shrinking mask
    # doesn't leave stale pixels of that id behind.
    frame = np.array([[7, 7], [7, 0]], dtype=int)
    protected = _mask((2, 2))
    assignment = _assignment(7, _mask((2, 2), (0, 0)))
    changed = paint_assignments(frame, (assignment,), protected, greedy=True)
    assert frame.tolist() == [[7, 0], [0, 0]]
    assert changed == {7}


def test_no_change_returns_empty_set():
    frame = np.array([[7, 0], [0, 0]], dtype=int)
    protected = _mask((2, 2))
    assignment = _assignment(7, _mask((2, 2), (0, 0)))
    changed = paint_assignments(frame, (assignment,), protected, greedy=True)
    assert frame.tolist() == [[7, 0], [0, 0]]
    assert changed == set()


def test_multiple_assignments_painted_together():
    frame = np.zeros((2, 3), dtype=int)
    protected = _mask((2, 3))
    a1 = _assignment(3, _mask((2, 3), (0, 0)))
    a2 = _assignment(4, _mask((2, 3), (1, 2)))
    changed = paint_assignments(frame, (a1, a2), protected, greedy=True)
    assert frame.tolist() == [[3, 0, 0], [0, 0, 4]]
    assert changed == {3, 4}
