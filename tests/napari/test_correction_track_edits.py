"""Selection-agnostic track edits + Ctrl+middle-click spawn-into-selected.

Track-editing mouse actions must fire however the cell became selected (image
click, lineage canvas, gallery, goto), not only after a left-click that happens
to record a click position. These bind the unbound ``CorrectionWidget`` handlers
to minimal stand-ins (no Qt widget / live viewer needed), the same pattern as
``test_correction_track_navigation``, and drive them with a selection that has
**no** recorded click position (``_selected_pos is None``) to prove they no
longer depend on it.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np

from cellflow.napari.correction.correction_widget import CorrectionWidget


def _edit_stub(*, selected_label):
    """A stand-in exposing just what the click handlers reach for.

    Crucially ``_selected_pos`` is ``None`` and ``_selected_t`` is ``-1`` — i.e.
    the cell was selected by something other than a left-click on the image.
    """
    stub = types.SimpleNamespace(
        _selected_label=selected_label,
        _selected_pos=None,
        _selected_t=-1,
        _swap_first_pos=None,
        _swap_first_t=-1,
        _record_history=MagicMock(),
        _update_highlight=MagicMock(),
        _set_status=MagicMock(),
        _intensity_frame=lambda t: None,
        _protected_mask=lambda t, seg2d: None,
        _cell_radius_spin=types.SimpleNamespace(value=lambda: 2),
    )
    # The swap handlers derive the selected cell's position from its label; bind
    # the real helper so the delegation works against the stub.
    stub._selected_label_pos = lambda seg2d: CorrectionWidget._selected_label_pos(
        stub, seg2d
    )
    return stub


def _layer():
    return types.SimpleNamespace(name="Labels", refresh=MagicMock())


def _last_status(stub) -> str:
    return stub._set_status.call_args[0][0]


# ── _selected_label_pos ─────────────────────────────────────────────────────


def test_selected_label_pos_lands_inside_the_mask():
    seg = np.zeros((8, 8), dtype=int)
    seg[2:5, 3:6] = 5
    stub = _edit_stub(selected_label=5)
    pos = CorrectionWidget._selected_label_pos(stub, seg)
    assert pos is not None
    r, c = int(round(pos[0])), int(round(pos[1]))
    assert seg[r, c] == 5


def test_selected_label_pos_none_when_absent():
    seg = np.zeros((8, 8), dtype=int)
    stub = _edit_stub(selected_label=5)
    assert CorrectionWidget._selected_label_pos(stub, seg) is None


# ── Ctrl+middle-click: spawn into the selected track ────────────────────────


def test_spawn_into_selected_merges_when_present_in_frame():
    seg = np.zeros((20, 20), dtype=int)
    seg[1:4, 1:4] = 5  # selected cell already in this frame
    stub = _edit_stub(selected_label=5)
    layer = _layer()

    CorrectionWidget._spawn_into_selected(stub, seg, (12.0, 12.0), 0, layer)

    # The spawned blob carries the selected ID (a disconnected region of 5).
    assert seg[12, 12] == 5
    assert np.count_nonzero(seg == 5) > 9  # original 9 px + the new disk
    stub._record_history.assert_called_once()
    assert "Merged" in _last_status(stub)


def test_spawn_into_selected_links_when_absent_from_frame():
    seg = np.zeros((20, 20), dtype=int)  # selected cell NOT in this frame
    stub = _edit_stub(selected_label=5)
    layer = _layer()

    CorrectionWidget._spawn_into_selected(stub, seg, (12.0, 12.0), 3, layer)

    assert seg[12, 12] == 5
    stub._record_history.assert_called_once()
    assert "Linked" in _last_status(stub)


def test_spawn_into_selected_needs_a_selection():
    seg = np.zeros((20, 20), dtype=int)
    stub = _edit_stub(selected_label=0)
    CorrectionWidget._spawn_into_selected(stub, seg, (5.0, 5.0), 0, _layer())
    assert np.count_nonzero(seg) == 0
    assert "select a cell" in _last_status(stub).lower()


def test_spawn_into_selected_rejects_click_on_a_cell():
    seg = np.zeros((20, 20), dtype=int)
    seg[1:4, 1:4] = 5
    seg[10:13, 10:13] = 7
    stub = _edit_stub(selected_label=5)
    CorrectionWidget._spawn_into_selected(stub, seg, (11.0, 11.0), 0, _layer())
    # Clicked on cell 7 — nothing painted, 7 untouched.
    assert seg[11, 11] == 7
    stub._record_history.assert_not_called()


# ── Ctrl+right-click: swap / attach / two-click, regardless of selection ────


def test_ctrl_right_click_swaps_with_selection_made_off_image():
    seg = np.zeros((10, 10), dtype=int)
    seg[1:4, 1:4] = 5
    seg[6:9, 6:9] = 7
    stub = _edit_stub(selected_label=5)  # _selected_pos is None
    layer = _layer()

    CorrectionWidget._ctrl_right_click_swap(stub, seg, (7.0, 7.0), 0, layer)

    # 5 and 7 exchanged across the frame.
    assert seg[2, 2] == 7
    assert seg[7, 7] == 5
    # Track stays selected after the swap (it now lives where 7 was), matching
    # the attach-to-track path.
    stub._update_highlight.assert_called_once_with(0, 5)
    assert "Swapped" in _last_status(stub)


def test_ctrl_right_click_attaches_when_selected_cell_absent_here():
    seg = np.zeros((10, 10), dtype=int)
    seg[6:9, 6:9] = 7  # selected cell 5 is NOT in this frame
    stub = _edit_stub(selected_label=5)
    layer = _layer()

    CorrectionWidget._ctrl_right_click_swap(stub, seg, (7.0, 7.0), 0, layer)

    # With the selected cell on another frame there is nothing here to swap
    # with, so the clicked cell is attached to that track (relabelled to 5).
    assert seg[7, 7] == 5
    assert not np.any(seg == 7)
    stub._record_history.assert_called_once()
    assert "Attached to track 5" in _last_status(stub)


def test_ctrl_right_click_arms_two_click_swap_without_selection():
    seg = np.zeros((10, 10), dtype=int)
    seg[6:9, 6:9] = 7
    stub = _edit_stub(selected_label=0)
    CorrectionWidget._ctrl_right_click_swap(stub, seg, (7.0, 7.0), 2, _layer())
    # No selection: this click is remembered as the first swap cell.
    assert stub._swap_first_pos == (7.0, 7.0)
    assert stub._swap_first_t == 2


def test_ctrl_right_click_finishes_two_click_swap():
    seg = np.zeros((10, 10), dtype=int)
    seg[1:4, 1:4] = 5
    seg[6:9, 6:9] = 7
    stub = _edit_stub(selected_label=0)
    stub._swap_first_pos = (2.0, 2.0)  # first cell already armed at this frame
    stub._swap_first_t = 0
    layer = _layer()

    CorrectionWidget._ctrl_right_click_swap(stub, seg, (7.0, 7.0), 0, layer)

    assert seg[2, 2] == 7
    assert seg[7, 7] == 5
    assert stub._swap_first_pos is None  # armed swap consumed
    assert "Swapped" in _last_status(stub)
